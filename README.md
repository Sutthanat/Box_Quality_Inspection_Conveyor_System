# Box Quality Inspection Conveyor System

Sutthanat Sinthupinyo 6630318321

**PDF Report --> ![Report](docs/Report.pdf)** 

## Brief overview of the project
  This project is a **machine vision quality inspection system** for boxes on a conveyor, developed with a **Raspberry Pi 4 B** and a **Cytron Motion 2350 Pro** microcontroller using a customized **YOLO** model (ONNX) to detect 5 types of defect on boxes, and uses **OpenCV** to recognize sticker colours on the boxes. The conveyor will stop if the box is defective. There's also a **Flask** web interface dashboard to monitor and log detection results, together with controlling and monitoring the conveyor’s status on the same local Wi-Fi network. No Windows PC, C-drive path, password, or external inference service is required at runtime.

---

## Table of Contents

- [Project Rationale & Scope](#project-rationale--scope)
- [Hardware & Wiring Diagram](#hardware--wiring-diagram)
- [Defect classes](#defect-classes)
- [What It Does](#what-it-does)
- [Part List](#part-list)
- [Software Stack](#software-stack)
- [Physical Operation & Safety](#physical-operation--safety)
- [Installation on the Pi](#installation-on-the-pi)
- [Configure and export](#configure-and-export)
- [Developer Checks](#developer-checks)
- [Future Works](#future-works)
- [Background & Related Previous Works](#background--related-previous-works)

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

## Hardware & Wiring Diagram

The physical plant consists of a top-mounted industrial vision camera, a Raspberry Pi 4 B, a Cytron MOTION 2350 Pro microcontroller, a 12 V DC gear motor with quadrature encoder driving the conveyor belt, and two optical sensors.

**Overall System**
![Overall System](media/Hardware_Setup.jpg)

**Wiring Diagram**
![Wiring Diagram](media/Wiring_Diagram.jpg)

## Defect classes

## What it does

- **Session Management:** Each new session starts with box No.1, and resets the previous session's dashboard totals while keeping previous session folders intact.

- **Positioning:** Runs the belt forward while waiting for a box to pass Sensor 1. When Sensor 1 detects a box, it does not stop the belt; Sensor 2 is armed in the controller firmware and performs a controlled sensor-land at the box's leading edge, stopping the conveyor when Sensor 2 detects a box.

- **Centering Under Camera:** Measures the travel time from Sensor 1 to Sensor 2 using live controller telemetry, reverses for half the measured time to center the box under the lens, and waits for the conveyor to settle.

- **Capture the box image:** Uses the exact `crop_fixed` function from `crop.py` for YOLO ONNX inference and sticker color analysis (`postit_color.py`). The cropped image is saved with each detected defect box annotated.

- **Defect Decision:** Classifies a box as defective if one or more classes exceed the configured confidence threshold. Repeated predictions of the same defect type are counted once per box.

- **Sorting Action:**
  - **Good boxes** are sent forward through Sensor 2 without stopping, then Sensor 2 is re-armed for the next box.
  - **Defective boxes** stop at Sensor 2 and wait for the operator to remove them before resuming forward motion.

- **Logging & Exports:** Logs box number, unique defect types, sticker color, confidence, timestamp, image filename, and detection boxes. CSV exports include session performance totals and per-type defective-box counts; ZIP exports include both CSVs and annotated photos.

- **Live Monitoring:** Streams a live camera view from the existing Pi camera service (avoiding V4L2 device conflicts) to any browser on the local Wi-Fi network.

---

## Part List

| # | Component | Model | Key Specifications | Approx. Price |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **Optical Sensors (x2)** | Keyence PZ2 Series | Photoelectric sensor with built-in amplifier, IP67 waterproof, detects glossy objects | Quote only |
| 2 | **Industrial Vision Camera** | ZWAK-5600 (AR0234) | 1080p, onsemi AR0234 1/2.6" global shutter sensor ($3\ \mu\text{m}$ pixels, $1920 \times 1200$), 2.8–12 mm f/1.4 CS-mount IR-corrected manual lens | 2,738 THB |
| 3 | **DC Gear Motor** | XYT JGB37-3530-CE | Brushed DC gear motor, 12 V, 1,600 RPM, $35\ \text{kg}\cdot\text{cm}$ torque, 0.6 A, 2.5/4.8 W | 205 THB |
| 4 | **Gear Motor with Encoder** | YFROBOT GB37-3530 | 37 mm brushed DC gearmotor, 12 V, 448 CPR quadrature encoder (3.5–5 V supply), 6 mm D-shaft | 1,458 THB |
| 5 | **Single-Board Computer** | Raspberry Pi 4 Model B | Broadcom BCM2711 quad-core Cortex-A72 @ 1.8 GHz, LPDDR4, 5 V / 3 A supply | 1,458–2,754 THB |
| 6 | **Microcontroller** | Cytron MOTION 2350 Pro | RP2350 dual-core Cortex-M33 @ 150 MHz, 520 KB RAM, 2 MB flash, 4x DC motor drivers (10 A max/ch, 3.6–16 V), CircuitPython | 1,123–1,300 THB |

---

## Software Stack
- **Python 3.10+**
- **Ultralytics YOLO11n** (exported to **ONNX** for $640 \times 640$ CPU inference)
- **OpenCV & NumPy** (image cropping, HSV sticker color detection, bounding box annotation)
- **Flask** (local web dashboard and API)
- **PyYAML & Requests** (configuration and inter-service communication)

**Codes are in code/ folder.**

---

## Physical operation & safety

- **Network Access:** The web server binds to all Pi network interfaces so any device on the local Wi-Fi can open it. **There is no login:** any connected client can start, stop, or emergency-stop the conveyor. Run only on a trusted, isolated network.

- **Start Safety Gate:** The Start button requires operator confirmation that the belt path is clear, a physical emergency stop is accessible, and other conveyor-control applications are idle. The app also checks for fresh controller telemetry, zero velocity/power, clear sensors, and an idle capture workflow before starting. It does not automatically reset a stale controller.

- **Emergency Stop:** Always keep a physical emergency stop accessible. The web emergency-stop button is **not** a substitute for a hardware E-stop.

- **Process Exclusivity:** Do not run the sorter at the same time as any other program that commands the belt or uses the camera. Separate apps do not share a cross-process motion lock.

- **Commissioning Notes:** Commission with the belt clear, an operator at the physical emergency stop, reduced speed, and one box at a time. Verify sensor order, direction, timing, removal, and YOLO decisions before unattended operation. Sensor timing is sampled at a 50 ms telemetry interval (reverse duration is approximate). A YOLO miss is treated as **Good**, so tune your confidence threshold accordingly.

---

## Installation on the Pi

Copy this project folder to `~/paml_project/Mechatronics_Project`, including:

- `sorter_app.py`, `templates/index.html`, `crop.py`, and `postit_color.py`
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

- **`config.yaml`:** Edit to adjust the listening port, camera/conveyor service URLs, YOLO confidence threshold, motion limits, default speed, and output directory. All model and session paths are project-relative.

- **Speed Control:** Conveyor speed can be adjusted directly from the web dashboard within the configured min/max limits; changes apply on the next motion phase.

- **Model & Vision Tuning:** The ONNX class order must match `yolo.class_names` in `config.yaml` and the training dataset. The supplied crop and sticker-color scripts are the runtime sources of their respective algorithms; Adjust hue/threshold values in `postit_color.py` if lighting or sticker colors change.

- **Session Logs & Downloads:** Each Start creates a new persistent folder under `sorting_data/`. Download the active or most recent session's CSV/performance summary and annotated image ZIP from the dashboard before starting a new session.

---

## Developer checks

Python 3.10 or later is required. Run the hardware-free tests from this directory:

```sh
python3 -m unittest -v test_sorter_app
```

The tests cover crop geometry, ONNX output mapping, repeated-defect handling,
CSV/performance output, image export, speed validation, and the Start safety
gate. They mock controller and camera access; they never move the conveyor.

---

## Future Works

- **Robotic Sorting:** Integrate a **Dobot** robotic arm to automatically pick defective boxes off the conveyor and sort them into reject bins.

- **Database Integration:** Migrate from flat-file session CSVs to a structured database (e.g., SQLite/SQLAlchemy or MongoDB) for long-term quality analytics.

- **Model Accuracy:** Expand the dataset with more defect samples and lighting variations to retrain a more robust YOLO model, including ability to detect rotated boxes.

- **Dashboard & Reliability:** Enhance UI visualizations and overall mechanical/software fault tolerance.

- **VLM Integration:** Add anomaly detection and natural-language defect explanations using Vision Language Models (VLMs).

---

## Background & Related Previous Works

**Visual inspection in manufacturing:**

Automated visual inspection replaces or supports a human inspector with a camera and software. A typical system has controlled lighting, a trigger that places the object at the same spot, image processing, a decision, and an actuator that removes rejected items. In practice, the quality of the images (lighting, focus, a fixed position) often matters more than the algorithm.


**Classical machine vision:**

Early inspection systems relied on hand-designed rules: thresholds, edge detection, shape measurement and colour checks, often written with libraries such as OpenCV. These methods are fast and easy to explain, and they work well when a defect can be described by a measurement. Their weakness is sensitivity to changes in lighting and appearance, and the effort needed to write a rule for each new defect. Monti et al. [2] present DIE-VIS, a traditional computer-vision system for cardboard box manufacturing. They compare it with YOLOv8 and argue that, in this setting, a customised classical solution still has important advantages. In this project, classical methods are kept where a rule is enough, such as cropping the box and recognising the post-it colour.


**Deep learning for defect detection:**

Convolutional neural networks (CNN) learn visual features directly from labelled images, and this removes much of the hand-design effort. For defect detection there are four main tasks: classification (is the image good or bad), object detection (what defects are present, and where), segmentation (which pixels are defective) and anomaly detection (does this differ from normal). Object detection suits this project, because a box can carry several defects, and each one is drawn as a rectangle.


**The YOLO family:**

YOLO ("You Only Look Once") [1] treats detection as a single pass through a network that predicts rectangles and class probabilities for the whole image. This makes it fast, and the Ultralytics implementation [3] offers small models, easy training from pre-trained weights, and export to formats such as ONNX and NCNN for embedded devices. It is widely used in industrial inspection. One licensing point should be mentioned: Ultralytics YOLO is released under the AGPL-3.0 licence, and closed commercial products need a separate commercial licence.


**Related works on paper box defects:**

Several recent papers apply YOLO-type detectors to packaging:

- Lu et al. [4] improve YOLOv5s with a deformable-convolution module for appearance defects on carton packaging film, and report a recognition rate of 99.3 percent, an accuracy gain of 2.7 percent over the base model, and a lower scrap rate. They also note that work specific to carton packaging is still limited.
- DScanNet [5] is a YOLO-based model for small defects in logistics packaging. Its authors point out that few packaging defect datasets exist, and they build their own.
- Monti et al. [2], mentioned above, offer a dataset for cardboard box inspection and compare classical and deep methods.
These works concentrate on improving the network. They usually do not cover the complete system around it, and they use large or specially collected datasets.


**Embedded and low-cost deployment:**

Running the model on the inspection line, instead of on a server, reduces cost and delay. Okano et al. [6] deployed YOLOv8 Nano on a Raspberry Pi 500 for visual conformity checks and report about 470 ms per image with a mAP@0.5 of 0.938, which shows that CPU-only inference can be fast enough for real use. Another study [7] builds a conveyor-based inspection system for PCB components using a Raspberry Pi 5, YOLOv11 and NCNN, with a finite-state-machine controller that stops the conveyor at the inspection position and triggers a pass or fail sorter. It reports 5 to 10 frames per second and about 3 seconds per board. The structure of that system is close to this project.


*Note on Licensing: Ultralytics YOLO is released under the AGPL-3.0 license; closed commercial deployments require a separate commercial license.*

1. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," *CVPR*, 2016. [IEEE Xplore](https://ieeexplore.ieee.org/document/7780460)
2. F. Monti, M. Marinacci, F. Leotta, and M. Mecella, "DIE-VIS: an Automated Visual Inspection System for Cardboard Box Manufacturing," *VISION Workshop, ECCV*, 2024.
3. Ultralytics, "Ultralytics YOLO Documentation (v8.4.215, YOLO11n)," [docs.ultralytics.com](https://docs.ultralytics.com).
4. Z. Lu, X. Yang, J. Jiang, W. Li, W. Zhang, and M. Huang, "A Deep Learning-based Method for Production Carton Packaging Defect Detection," *Elektronika ir Elektrotechnika*, 2026. [DOI: 10.5755/j02.eie.45566](https://eejournal.ktu.lt/index.php/elt/article/view/45566)
5. Y. Luo, Y. Du, Z. Wang, J. Mo, W. Yu, and S. Dou, "DScanNet: Packaging Defect Detection Algorithm Based on Selective State Space Models," *Algorithms*, vol. 18, no. 6, p. 370, 2025. [DOI: 10.3390/a18060370](https://doi.org/10.3390/a18060370)
6. M. T. Okano et al., "Edge AI for Industrial Visual Inspection: YOLOv8-Based Visual Conformity Detection Using Raspberry Pi," *Algorithms*, vol. 18, no. 8, p. 510, 2025. [MDPI](https://www.mdpi.com/1999-4893/18/8/510)
7. B. Jaswanth, D. Chandiran, J. Mohammed Sami, K. Gobivel, and S. Gayathri Monicka, "Edge-Based Deep Learning System for Automated PCB Component Inspection," *EPJ Web Conf.*, vol. 363, 01006, 2026. [DOI: 10.1051/epjconf/202636301006](https://doi.org/10.1051/epjconf/202636301006)
