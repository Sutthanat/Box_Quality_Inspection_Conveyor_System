# Box Quality Inspection Conveyor System


*** Report --> docs/Report.pdf ***

Sutthanat Sinthupinyo 6630318321

## Brief overview of the project
  This project is a **machine vision quality inspection system** for boxes on a conveyor, developed with a **Raspberry Pi 4 B** and a **Cytron Motion 2350 Pro** microcontroller using a customized **YOLO** model (ONNX) to detect 5 types of defect on boxes, and uses **OpenCV** to recognize sticker colours on the boxes. The conveyor will stop if the box is defective. There's also a **Flask** web interface dashboard to monitor and log detection results, together with controlling and monitoring the conveyor’s status on the same local Wi-Fi network. No Windows PC, C-drive path, password, or external inference service is required at runtime.

---

## Table of Contents

- [Project Rationale & Scope](#project-rationale--scope)
- [What It Does](#what-it-does)
- [System Architecture & Hardware](#system-architecture--hardware)
- [Physical Operation & Safety](#physical-operation--safety)
- [Installation on the Raspberry Pi](#installation-on-the-raspberry-pi)
- [Configuration & Data Export](#configuration--data-export)
- [Developer Checks](#developer-checks)
- [Future Work](#future-work)
- [Background & References](#background--references)

---

## Project Rationale & Scope

Visual quality inspection is a routine step in packaging lines. In many small and medium factories, inspection is still performed manually, which is slow, fatiguing, inconsistent, and rarely logged for quality analysis. Commercial automated optical inspection (AOI) systems are often expensive and closed, while rule-based computer vision alone struggles to generalize across diverse defect types (structural damage, missing seals, and surface stains).

This project implements an accessible, embedded mechatronics inspection loop combining deep learning and classical computer vision:
- **Multi-defect detection:** Detects 5 defect classes simultaneously on a single box—**open lid**, **tape defect**, **sticker defect**, **crushed corner**, and **stain**—using a custom YOLO11n model exported to ONNX.
- **Classical CV where rules suffice:** Uses OpenCV for fixed-region cropping (`crop.py`), drawing annotated bounding boxes, and recognizing post-it/sticker colors (`postit_color.py`).
- **Complete mechatronics loop:** Sensing $\rightarrow$ Inspection $\rightarrow$ Decision $\rightarrow$ Sorting/Stopping $\rightarrow$ Logging.
- **Why Raspberry Pi 4 B?** Unlike microcontrollers (e.g., Arduino), the Raspberry Pi provides native USB camera support, sufficient CPU performance for live video and lightweight ONNX inference, strong Python/OpenCV ecosystem support, built upon a strong Raspberry Pi community and compact form factor compared to a mini PC or higher-cost Jetson boards.

**Scope & Limitations:** The system uses a single top-down camera, so side-wall damage is not inspected. Boxes need to be presented in a fixed orientation, and the model was trained on a dataset of 20 boxes of one design under controlled lab lighting.

---

## What it does

- Starts each session at box 1 and resets that session's dashboard totals.
- Runs the belt forward while waiting for sensor 1. Sensor 1 does not stop the
  belt; sensor 2 is armed in the controller firmware and performs its existing
  controlled sensor-land at the box's leading edge.
- Measures sensor 1 to sensor 2 using live controller telemetry, reverses for
  half the measured time, then waits for the conveyor to settle.
- Uses the exact `crop_fixed` function from `crop.py` for YOLO and sticker
  color analysis. The crop is saved with each YOLO box annotated.
- Classifies a box as defective if one or more classes exceed the configured
  confidence threshold. Repeated predictions of one defect type are counted
  once per box.
- Sends good boxes forward through sensor 2 without stopping. It then re-arms
  sensor 2 for the next box.
- Stops a defective box at sensor 2 and waits for the operator to remove it
  before resuming forward motion.
- Logs the box number, unique defect types, sticker color, confidence,
  timestamp, image filename, and detection boxes. CSV exports include session
  performance totals and per-type defective-box counts. ZIP exports include
  both CSVs and annotated photos.
- Provides a live camera view from the existing Pi camera service, avoiding a
  second V4L2 camera owner.

## Hardware

## Defect classes

## Physical operation and safety

The web server binds to all Pi network interfaces so any device on the local
Wi-Fi can open it. **There is no login:** any connected client can start, stop,
or emergency-stop the conveyor. Run only on a trusted, isolated network.

The Start button requires an operator confirmation that the belt path is
clear, a physical emergency stop is accessible, and other conveyor-control
applications are idle. The app additionally requires fresh controller
telemetry, zero velocity/power, clear sensors, and an idle existing capture
workflow before starting. It does not reset a stale controller automatically.
Keep the physical emergency stop available; the web emergency-stop button is
not a substitute for it.

Do not run the sorter at the same time as the existing conveyor capture app,
the controller dashboard, or another program that commands the belt. The
separate existing apps do not share a cross-process motion lock.

The Start/Stop controls have not been tested on the physical conveyor by the
software checks. Commission with the belt clear, an operator at the physical
emergency stop, reduced speed, and one box at a time. Verify sensor order,
direction, timing, removal, and YOLO decisions before unattended operation.
Sensor timing is sampled at the configured 50 ms telemetry interval; the
reverse duration is therefore approximate. A YOLO miss is treated as Good, so
use a threshold appropriate to the risk of sorting defects as good.

## Install on the Pi

Copy this project folder to `~/paml_project/Mechatronics_Project`, including:

- `sorter_app.py`, `templates/index.html`, and the provided
  `crop.py`/`postit_color.py` utilities
- `best.onnx`, exported for 640 x 640 images with the five classes in the
  configured order
- `config.yaml`, `requirements.txt`, tests, README, and service unit

The Pi camera service must provide 1280 x 720 snapshots on port 9000. The
existing conveyor API must be available on port 8000, and the original capture
interface status API on port 8100. `config.yaml` uses only project-relative
paths for model and session output.

The service uses the Pi's existing Python packages for OpenCV, NumPy, Flask,
PyYAML, and Requests. If a package is missing, install the OS OpenCV bindings
and then install Python dependencies in a virtual environment configured to
see those bindings, or use the distributions supported by your Pi OS. Do not
replace a working system OpenCV installation until ONNX compatibility has
been checked.

After copying the project and confirming that its ONNX model loads, install
and start the service:

```sh
sudo cp paml-box-sorter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now paml-box-sorter.service
systemctl status paml-box-sorter.service
```

Find the Pi's Wi-Fi address with `hostname -I`, then open
`http://<pi-address>:8101/` from a device on the same network. Confirm the
network permits connections to port 8101.

## Configure and export

Edit `config.yaml` to change the listening port, camera/conveyor service URLs,
YOLO threshold, motion limits, default speed, and output directory. Conveyor
speed can also be adjusted from the web page within the configured minimum
and maximum; changes apply at the next motion phase.

The ONNX class order must match `yolo.class_names` and the training dataset.
The supplied crop and sticker-color scripts are the runtime sources of their
respective algorithms; adjust sticker hue and threshold settings in
`postit_color.py` if lighting or labels change.

Each Start creates a new, persistent session directory under `sorting_data/`.
Session box numbering and dashboard counts reset on each Start, while earlier
session folders remain on the Pi. Download the active or most recent session's
CSV/performance and image ZIP from the page before starting another session.

## Developer checks

Python 3.10 or later is required. Run the hardware-free tests from this directory:

```sh
python3 -m unittest -v test_sorter_app
```

The tests cover crop geometry, ONNX output mapping, repeated-defect handling,
CSV/performance output, image export, speed validation, and the Start safety
gate. They mock controller and camera access; they never move the conveyor.
