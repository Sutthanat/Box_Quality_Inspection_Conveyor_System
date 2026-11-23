#!/usr/bin/env python3
"""LAN-hosted box sorter with Pi-side YOLO, conveyor control, and session logs."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import signal
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import yaml
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)

import crop
from crop import crop_fixed
from postit_color import detect_postit_color


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PAML_SORTER_CONFIG", ROOT / "config.yaml"))
CSV_FIELDS = (
    "row_type",
    "box_number",
    "captured_at",
    "status",
    "defect_types",
    "sticker_color",
    "confidence",
    "image_file",
    "detections_json",
    "metric",
    "value",
)

logging.basicConfig(
    level=os.environ.get("PAML_SORTER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("paml_box_sorter")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load project-relative settings and reject incomplete sorter config."""
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except OSError as exc:
        raise RuntimeError(f"Unable to read config file {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"Config file {path} must contain a YAML mapping.")

    required = ("app", "camera", "conveyor", "yolo", "sorting")
    missing = [key for key in required if not isinstance(config.get(key), dict)]
    if missing:
        raise RuntimeError(f"Config is missing mappings: {', '.join(missing)}")
    return config


def _read_model_input(cropped: np.ndarray, model_size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize the training crop without distortion and add YOLO letterbox padding."""
    height, width = cropped.shape[:2]
    scale = model_size / max(height, width)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    pad_x = (model_size - resized_width) // 2
    pad_y = (model_size - resized_height) // 2
    resized = cv2.resize(cropped, (resized_width, resized_height))
    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_y,
        model_size - resized_height - pad_y,
        pad_x,
        model_size - resized_width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    blob = cv2.dnn.blobFromImage(
        letterboxed,
        scalefactor=1 / 255.0,
        size=(model_size, model_size),
        swapRB=True,
    )
    return blob, scale, pad_x, pad_y


def decode_yolo_output(
    output: np.ndarray,
    class_names: list[str],
    crop_shape: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Decode exported YOLOv8 center/size output into boxes on the training crop."""
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Expected a single batched YOLO output, received {output.shape}.")
    if output.shape[1] != 4 + len(class_names):
        raise ValueError(
            f"Model output has {output.shape[1]} channels, but the configured "
            f"classes require {4 + len(class_names)}."
        )

    candidates = output[0].T
    class_ids = np.argmax(candidates[:, 4:], axis=1)
    scores = candidates[np.arange(len(candidates)), 4 + class_ids]
    crop_height, crop_width = crop_shape
    detections: list[dict[str, Any]] = []

    for class_id, class_name in enumerate(class_names):
        candidate_indices = np.flatnonzero(
            (class_ids == class_id) & (scores >= confidence_threshold)
        )
        boxes: list[list[int]] = []
        confidences: list[float] = []
        source_boxes: list[tuple[int, int, int, int]] = []
        for index in candidate_indices:
            center_x, center_y, box_width, box_height = candidates[index, :4]
            x0 = round((center_x - box_width / 2 - pad_x) / scale)
            y0 = round((center_y - box_height / 2 - pad_y) / scale)
            x1 = round((center_x + box_width / 2 - pad_x) / scale)
            y1 = round((center_y + box_height / 2 - pad_y) / scale)
            x0, x1 = max(0, min(crop_width, x0)), max(0, min(crop_width, x1))
            y0, y1 = max(0, min(crop_height, y0)), max(0, min(crop_height, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            source_boxes.append((x0, y0, x1, y1))
            boxes.append([x0, y0, x1 - x0, y1 - y0])
            confidences.append(float(scores[index]))

        if not boxes:
            continue
        retained = cv2.dnn.NMSBoxes(
            boxes, confidences, confidence_threshold, iou_threshold
        )
        for retained_index in np.asarray(retained).reshape(-1):
            x0, y0, x1, y1 = source_boxes[int(retained_index)]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidences[int(retained_index)], 4),
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                }
            )
    return sorted(detections, key=lambda detection: detection["confidence"], reverse=True)


class BoxInspector:
    """Run one on-demand camera snapshot through the crop, YOLO, and sticker tools."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.camera_url = str(config["camera"]["url"]).rstrip("/")
        self.camera_timeout = float(config["camera"]["timeout_seconds"])
        self.crop_window = (crop.X0, crop.Y0, crop.X1, crop.Y1)
        self.frame_width = int(config["camera"]["frame_width"])
        self.frame_height = int(config["camera"]["frame_height"])
        self.model_size = int(config["yolo"]["input_size"])
        self.class_names = list(config["yolo"]["class_names"])
        self.confidence = float(config["yolo"]["confidence_threshold"])
        self.iou = float(config["yolo"]["iou_threshold"])
        if not self.class_names or self.model_size < 32:
            raise ValueError("YOLO needs configured class names and a valid input size.")
        if not 0 < self.confidence <= 1 or not 0 < self.iou <= 1:
            raise ValueError("YOLO confidence and IoU thresholds must be in (0, 1].")
        x0, y0, x1, y1 = self.crop_window
        if not (
            0 <= x0 < x1 <= self.frame_width
            and 0 <= y0 < y1 <= self.frame_height
        ):
            raise ValueError("The fixed crop does not fit the configured camera frame.")
        model_path = Path(config["yolo"]["model"])
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")
        self.model_path = model_path.resolve()
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net_lock = threading.Lock()

        self.jpeg_quality = int(config["sorting"]["jpeg_quality"])

    def inspect(self) -> dict[str, Any]:
        """Capture a frame and return an annotated crop plus decisions and timings."""
        response = requests.get(
            f"{self.camera_url}/snapshot.jpg", timeout=self.camera_timeout
        )
        response.raise_for_status()
        frame = cv2.imdecode(
            np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise RuntimeError("The camera service returned an unreadable JPEG.")
        if frame.shape[:2] != (self.frame_height, self.frame_width):
            raise ValueError(
                f"Camera image must be {self.frame_width}x{self.frame_height} "
                "to match the training crop; "
                f"received {frame.shape[1]}x{frame.shape[0]}."
            )

        cropped = crop_fixed(frame)
        if cropped.size == 0:
            raise ValueError(f"The configured crop {self.crop_window} is empty.")
        blob, scale, pad_x, pad_y = _read_model_input(cropped, self.model_size)
        with self.net_lock:
            self.net.setInput(blob)
            inference_started = time.perf_counter()
            output = self.net.forward()
        inference_ms = (time.perf_counter() - inference_started) * 1000
        detections = decode_yolo_output(
            output,
            self.class_names,
            cropped.shape[:2],
            scale,
            pad_x,
            pad_y,
            self.confidence,
            self.iou,
        )

        postit_color, postit_info = detect_postit_color(cropped)
        annotated = cropped.copy()
        for detection in detections:
            x, y = detection["x"], detection["y"]
            x2, y2 = x + detection["width"], y + detection["height"]
            color = self._class_color(detection["class_id"])
            cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
            label = f"{detection['class_name']} {detection['confidence']:.0%}"
            cv2.putText(
                annotated,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("OpenCV failed to encode the annotated crop.")
        defect_types = list(dict.fromkeys(item["class_name"] for item in detections))
        return {
            "image": encoded.tobytes(),
            "detections": detections,
            "defect_types": defect_types,
            "status": "Defective" if defect_types else "Good",
            "sticker_color": postit_color,
            "sticker_info": {
                key: value
                for key, value in postit_info.items()
                if key != "bbox"
            },
            "inference_ms": round(inference_ms, 1),
        }

    @staticmethod
    def _class_color(class_id: int) -> tuple[int, int, int]:
        """Return a stable BGR annotation color for each class index."""
        palette = (
            (255, 80, 30),
            (30, 190, 255),
            (255, 80, 220),
            (0, 80, 255),
            (70, 220, 70),
            (255, 220, 0),
        )
        return palette[class_id % len(palette)]


class SorterController:
    """Serialize conveyor states and persist each session's images and CSV rows."""

    def __init__(self, config: dict[str, Any], inspector: BoxInspector) -> None:
        self.config = config
        self.inspector = inspector
        conveyor = config["conveyor"]
        self.conveyor_url = str(conveyor["url"]).rstrip("/")
        self.request_timeout = float(conveyor["request_timeout_seconds"])
        self.motion_timeout = float(conveyor["motion_timeout_seconds"])
        self.stale_seconds = float(conveyor["telemetry_stale_seconds"])
        self.poll_seconds = float(conveyor["poll_seconds"])
        self.stop_timeout = float(conveyor["stop_timeout_seconds"])
        self.max_acceleration = float(conveyor["max_acceleration"])
        self.min_speed = int(conveyor["min_speed"])
        self.max_speed = int(conveyor["max_speed"])
        self._speed = int(conveyor["default_speed"])
        self._data_dir = Path(config["sorting"]["data_directory"])
        if not self._data_dir.is_absolute():
            self._data_dir = ROOT / self._data_dir
        self._data_dir = self._data_dir.resolve()
        self._settle_seconds = float(config["sorting"]["settle_seconds"])
        if (
            self.poll_seconds <= 0
            or self.request_timeout <= 0
            or self.stale_seconds <= 0
            or self.motion_timeout <= 0
            or self.stop_timeout <= 0
            or self.max_acceleration <= 0
            or self._settle_seconds < 0
        ):
            raise ValueError("Conveyor timings and acceleration must be positive.")
        if (
            self.min_speed <= 0
            or self.max_speed < self.min_speed
            or not self.min_speed <= self._speed <= self.max_speed
        ):
            raise ValueError("Conveyor speed limits or default speed are invalid.")

        self._lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._state = "stopped"
        self._message = "Stopped. Confirm the lane is clear before starting."
        self._session_id: str | None = None
        self._session_dir: Path | None = None
        self._session_started: str | None = None
        self._records: list[dict[str, Any]] = []
        self._last_box: dict[str, Any] | None = None
        self._error: str | None = None
        self._last_tick: int | None = None
        self._last_tick_change: float | None = None
        self._telemetry: dict[str, Any] | None = None
        self._telemetry_error: str | None = None
        self._latest_frame: bytes | None = None

    def set_speed(self, speed: int) -> dict[str, Any]:
        """Save a bounded speed that takes effect at the next motion phase."""
        if not self.min_speed <= speed <= self.max_speed:
            raise ValueError(
                f"Speed must be between {self.min_speed} and {self.max_speed} counts/s."
            )
        with self._lock:
            self._speed = speed
            return self.snapshot()

    def start(self, confirmed_clear: bool) -> dict[str, Any]:
        """Begin a fresh session only after explicit confirmation and safe telemetry."""
        if not confirmed_clear:
            raise ValueError("Confirm that the conveyor lane is clear before starting.")
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("A sorting session is already running.")
            if self._state not in ("stopped", "error"):
                raise RuntimeError("Stop the current sorting session before restarting.")
            self._verify_stationary_controller()
            self._session_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "_"
                + uuid.uuid4().hex[:8]
            )
            self._session_started = datetime.now(timezone.utc).isoformat()
            self._session_dir = self._data_dir / self._session_id
            self._session_dir.mkdir(parents=True, exist_ok=False)
            self._records = []
            self._last_box = None
            self._error = None
            self._stop_event.clear()
            self._write_csv_header()
            self._set_state_locked(
                "starting",
                "Starting forward motion. Waiting for sensor 1, then sensor 2.",
            )
            self._worker = threading.Thread(
                target=self._run_session,
                name=f"sorter-{self._session_id}",
                daemon=True,
            )
            self._worker.start()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        """Stop in a controlled way; use the physical emergency stop if needed."""
        with self._lock:
            worker = self._worker
            self._stop_event.set()
            self._set_state_locked("stopping", "Controlled conveyor stop requested.")

        try:
            self._post_controller("/api/abort")
            self._wait_until_stopped(timeout=self.stop_timeout)
        except Exception as exc:
            logger.exception("Controlled stop failed; sending emergency stop")
            try:
                self._post_controller("/api/estop")
            except Exception:
                logger.exception("Emergency stop request also failed")
            self._finish("error", f"Controlled stop failed: {exc}")
            raise RuntimeError(
                f"Controlled stop failed ({exc}); emergency stop was requested."
            ) from exc

        if worker is not None and worker.is_alive():
            worker.join(timeout=self.stop_timeout)
        if worker is not None and worker.is_alive():
            raise RuntimeError("The sorter worker did not stop after the conveyor halted.")
        with self._lock:
            self._worker = None
        self._finish("stopped", "Sorting session stopped.")
        return self.snapshot()

    def emergency_stop(self) -> dict[str, Any]:
        """Send the controller emergency-stop command without waiting for motion."""
        with self._lock:
            worker = self._worker
            self._stop_event.set()
            self._set_state_locked("stopping", "Emergency stop requested.")
        try:
            self._post_controller("/api/estop")
        except Exception as exc:
            self._finish("error", f"Emergency-stop request failed: {exc}")
            raise RuntimeError(f"Emergency-stop request failed: {exc}") from exc
        if worker is not None and worker.is_alive():
            worker.join(timeout=self.stop_timeout)
        if worker is not None and worker.is_alive():
            raise RuntimeError("Emergency stop was sent, but the sorter worker did not exit.")
        with self._lock:
            self._worker = None
        self._finish("stopped", "Emergency stop command sent.")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return session metrics and live state as JSON-serializable values."""
        with self._lock:
            good = sum(record["status"] == "Good" for record in self._records)
            defective = len(self._records) - good
            defect_counts = {name: 0 for name in self.inspector.class_names}
            for record in self._records:
                for defect in record["defect_types"]:
                    defect_counts[defect] = defect_counts.get(defect, 0) + 1
            return {
                "state": self._state,
                "message": self._message,
                "running": self._worker is not None and self._worker.is_alive(),
                "session_id": self._session_id,
                "session_started": self._session_started,
                "box_count": len(self._records),
                "good_count": good,
                "defective_count": defective,
                "defect_counts": defect_counts,
                "speed": self._speed,
                "speed_min": self.min_speed,
                "speed_max": self.max_speed,
                "error": self._error,
                "telemetry": self._telemetry,
                "controller_error": self._telemetry_error,
                "last_box": self._last_box,
                "records": list(reversed(self._records[-100:])),
            }

    def current_image(self, session_id: str, filename: str) -> Path:
        """Resolve only images belonging to the active or most recent session."""
        with self._lock:
            if session_id != self._session_id or self._session_dir is None:
                raise FileNotFoundError("Session image is unavailable.")
            image = (self._session_dir / filename).resolve()
            if image.parent != self._session_dir.resolve() or image.suffix != ".jpg":
                raise FileNotFoundError("Session image is unavailable.")
            return image

    def export_csv(self) -> io.BytesIO:
        """Build a portable CSV from the current session records."""
        with self._lock:
            if self._session_dir is None:
                raise FileNotFoundError("No sorting session has been started.")
            data = io.StringIO(newline="")
            writer = csv.DictWriter(data, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for record in self._records:
                writer.writerow(self._csv_row(record))
            writer.writerows(self._performance_rows_locked())
            self._write_session_csv_locked()
            return io.BytesIO(data.getvalue().encode("utf-8-sig"))

    def export_zip(self) -> io.BytesIO:
        """Bundle the current CSV and all annotated box images for download."""
        with self._lock:
            if self._session_dir is None:
                raise FileNotFoundError("No sorting session has been started.")
            self._write_session_csv_locked()
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
                zipped.write(self._session_dir / "boxes.csv", "boxes.csv")
                performance = io.StringIO(newline="")
                performance_writer = csv.DictWriter(
                    performance, fieldnames=("metric", "value")
                )
                performance_writer.writeheader()
                performance_writer.writerows(
                    {"metric": row["metric"], "value": row["value"]}
                    for row in self._performance_rows_locked()
                )
                zipped.writestr("performance.csv", performance.getvalue())
                for record in self._records:
                    image = self._session_dir / record["image_file"]
                    if image.is_file():
                        zipped.write(image, record["image_file"])
            archive.seek(0)
            return archive

    def _verify_stationary_controller(self) -> None:
        """Require live zero-motion telemetry before an operator can start sorting."""
        deadline = time.monotonic() + self.stale_seconds + 1
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                telemetry = self._read_telemetry()
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_error = exc
                time.sleep(self.poll_seconds)
                continue
            if self._telemetry_is_fresh(telemetry) and self._is_stationary(telemetry):
                if int(telemetry["s1"]) or int(telemetry["s2"]):
                    raise RuntimeError(
                        "A conveyor sensor is already detecting an object; clear it before starting."
                    )
                self._check_other_capture_session()
                return
            time.sleep(self.poll_seconds)
        if last_error:
            raise RuntimeError(f"Conveyor controller is unavailable: {last_error}") from last_error
        raise RuntimeError(
            "Fresh, stationary conveyor telemetry was not observed; the belt was not started."
        )

    def _check_other_capture_session(self) -> None:
        """Avoid knowingly sharing the belt with the existing capture workflow."""
        capture_url = self.config["conveyor"].get("capture_app_status_url")
        if not capture_url:
            return
        try:
            response = requests.get(str(capture_url), timeout=self.request_timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Unable to confirm the existing capture workflow is idle: {exc}"
            ) from exc
        state = response.json()
        if state.get("phase") not in ("idle", "stopped", "error"):
            raise RuntimeError(
                "The existing conveyor capture workflow is active; stop it before sorting."
            )

    def _read_telemetry(self) -> dict[str, Any]:
        """Read a controller snapshot and attach local freshness information."""
        response = requests.get(
            f"{self.conveyor_url}/api/status", timeout=self.request_timeout
        )
        response.raise_for_status()
        payload = response.json()
        telemetry = payload.get("telemetry")
        if not isinstance(telemetry, dict):
            raise ValueError("Controller status did not include telemetry.")
        tick = int(telemetry["tick"])
        for key in ("vel", "pwr", "s1", "s2", "vm", "sl"):
            if key not in telemetry:
                raise ValueError(f"Controller telemetry is missing {key}.")
        now = time.monotonic()
        with self._lock:
            if self._last_tick is None:
                self._last_tick = tick
            elif tick != self._last_tick:
                self._last_tick = tick
                self._last_tick_change = now
            fresh = (
                self._last_tick_change is not None
                and now - self._last_tick_change <= self.stale_seconds
            )
            telemetry["_fresh"] = fresh
            self._telemetry = telemetry
            self._telemetry_error = None
        return telemetry

    def refresh_status(self) -> dict[str, Any]:
        """Refresh dashboard telemetry and preserve controller failures explicitly."""
        try:
            self._read_telemetry()
        except (requests.RequestException, ValueError, KeyError) as exc:
            with self._lock:
                self._telemetry = None
                self._telemetry_error = str(exc)
        return self.snapshot()

    def _telemetry_is_fresh(self, telemetry: dict[str, Any]) -> bool:
        return bool(telemetry.get("_fresh"))

    @staticmethod
    def _is_stationary(telemetry: dict[str, Any]) -> bool:
        return (
            abs(float(telemetry["vel"])) < 50
            and abs(float(telemetry["pwr"])) < 0.05
            and int(telemetry["vm"]) == 0
        )

    def _run_session(self) -> None:
        """Drive the full wait, detect, inspect, sort, and removal state sequence."""
        try:
            initial = self._read_telemetry()
            baseline_land = int(initial["sl"])
            if self._stop_event.is_set():
                raise InterruptedError("Sorting was stopped.")
            self._continuous_forward(sensor=2)
            self._set_state(
                "waiting",
                "Moving forward and waiting for sensor 1 to detect a box.",
            )

            while not self._stop_event.is_set():
                s1_time, s2_time = self._wait_for_sensor2(baseline_land)
                sensor_interval = s2_time - s1_time
                if not 0 < sensor_interval < self.motion_timeout:
                    raise RuntimeError(
                        f"Invalid sensor interval measured: {sensor_interval:.3f} seconds."
                    )
                self._wait_until_stopped(timeout=self.stop_timeout)
                if self._stop_event.is_set():
                    break
                self._set_state(
                    "reverse",
                    f"Sensor 2 reached; reversing for half the "
                    f"{sensor_interval:.2f}-second sensor interval.",
                )
                self._continuous_reverse()
                self._stop_event.wait(sensor_interval / 2)
                if self._stop_event.is_set():
                    break
                self._post_controller("/api/abort")
                self._wait_until_stopped(timeout=self.stop_timeout)
                if self._stop_event.wait(self._settle_seconds):
                    break

                self._set_state("yolo", "Capturing the cropped box for YOLO inspection.")
                inspection = self.inspector.inspect()
                record = self._save_inspection(inspection)
                if self._stop_event.is_set():
                    break

                if record["status"] == "Good":
                    baseline_land = self._good_box_cycle()
                else:
                    baseline_land = self._defective_box_cycle(
                        int(self._read_telemetry()["sl"])
                    )

        except InterruptedError:
            self._finish("stopped", "Sorting session stopped.")
            return
        except Exception as exc:
            logger.exception("Sorting cycle failed")
            try:
                self._post_controller("/api/estop")
                stop_message = "Emergency stop command sent after sorting error."
            except Exception as stop_exc:
                logger.exception("Emergency-stop request failed")
                stop_message = f"Emergency-stop request also failed: {stop_exc}"
            self._finish("error", f"{exc}. {stop_message}")
            return

        self._finish("stopped", "Sorting session stopped.")

    def _wait_for_sensor2(self, baseline_land: int) -> tuple[float, float]:
        """Measure S1-to-S2 time while an S2 edge performs controlled sensor land."""
        sensor1_at: float | None = None
        deadline: float | None = None
        while not self._stop_event.is_set():
            telemetry = self._read_telemetry()
            if not self._telemetry_is_fresh(telemetry):
                raise RuntimeError(
                    "Controller telemetry stopped updating; the conveyor was stopped."
                )
            now = time.monotonic()
            if sensor1_at is None and int(telemetry["s1"]):
                sensor1_at = now
                deadline = now + self.motion_timeout
                self._set_state(
                    "detecting",
                    "Sensor 1 detected the box; continuing forward to sensor 2.",
                )
            if int(telemetry["sl"]) > baseline_land:
                if sensor1_at is None:
                    raise RuntimeError("Sensor 2 landed before sensor 1 detected the box.")
                return sensor1_at, now
            if deadline is not None and now >= deadline:
                raise TimeoutError("Timed out waiting for sensor 2 after sensor 1.")
            if not self._stop_event.wait(self.poll_seconds):
                continue
        if self._stop_event.is_set():
            raise InterruptedError("Sorting was stopped.")
        raise TimeoutError("Timed out waiting for the next box.")

    def _good_box_cycle(self) -> int:
        """Pass good boxes through sensor 2 without stopping, then resume waiting."""
        telemetry = self._read_telemetry()
        was_at_sensor2 = bool(int(telemetry["s2"]))
        self._set_state(
            "good_continue",
            "Good box: moving forward through sensor 2 without stopping.",
        )
        self._continuous_forward(sensor=None)
        deadline = time.monotonic() + self.motion_timeout
        seen_sensor2 = was_at_sensor2
        last_s2 = int(telemetry["s2"])
        sensor2_cleared = not was_at_sensor2
        while time.monotonic() < deadline and not self._stop_event.is_set():
            current = self._read_telemetry()
            if not self._telemetry_is_fresh(current):
                raise RuntimeError("Controller telemetry stopped updating.")
            s2 = int(current["s2"])
            if s2 and not last_s2:
                seen_sensor2 = True
            if seen_sensor2 and not s2:
                sensor2_cleared = True
                break
            last_s2 = s2
            self._stop_event.wait(self.poll_seconds)
        if self._stop_event.is_set():
            return int(telemetry["sl"])
        if not sensor2_cleared:
            raise TimeoutError("Timed out waiting for the good box to clear sensor 2.")
        baseline_land = int(self._read_telemetry()["sl"])
        self._continuous_forward(sensor=2)
        self._set_state(
            "waiting",
            "Moving forward and waiting for sensor 1 to detect the next box.",
        )
        return baseline_land

    def _defective_box_cycle(self, baseline_land: int) -> int:
        """Advance defective boxes to sensor 2, stop, and wait for removal."""
        telemetry = self._read_telemetry()
        if int(telemetry["s2"]):
            # The centered box may still cover S2; do not re-arm an active edge.
            self._post_controller("/api/abort")
            self._wait_until_stopped(timeout=self.stop_timeout)
            landed = True
        else:
            self._set_state(
                "defective_eject",
                "Defective box: moving forward until sensor 2 detects it.",
            )
            self._continuous_forward(sensor=2)
            self._wait_for_land_count(baseline_land)
            self._wait_until_stopped(timeout=self.stop_timeout)
            landed = True

        if landed:
            self._set_state(
                "waiting_removal",
                "Defective box is stopped at sensor 2. Remove it to continue.",
            )
            while not self._stop_event.is_set():
                current = self._read_telemetry()
                if not self._telemetry_is_fresh(current):
                    raise RuntimeError("Controller telemetry stopped updating.")
                if not int(current["s2"]):
                    baseline_land = int(current["sl"])
                    self._continuous_forward(sensor=2)
                    self._set_state(
                        "waiting",
                        "Box removed. Moving forward and waiting for the next box.",
                    )
                    return baseline_land
                self._stop_event.wait(self.poll_seconds)
        return baseline_land

    def _wait_for_land_count(self, baseline_land: int) -> None:
        """Wait for the armed S2 edge, failing closed on stale telemetry or timeout."""
        deadline = time.monotonic() + self.motion_timeout
        while time.monotonic() < deadline and not self._stop_event.is_set():
            telemetry = self._read_telemetry()
            if not self._telemetry_is_fresh(telemetry):
                raise RuntimeError("Controller telemetry stopped updating.")
            if int(telemetry["sl"]) > baseline_land:
                return
            self._stop_event.wait(self.poll_seconds)
        if not self._stop_event.is_set():
            raise TimeoutError("Timed out waiting for sensor 2 to land the conveyor.")

    def _continuous_forward(self, sensor: int | None) -> None:
        self._post_controller(
            "/api/continuous",
            {
                "dir": 1,
                "max_vel": self._current_speed(),
                "max_accel": self.max_acceleration,
                "sensor": sensor,
            },
        )

    def _continuous_reverse(self) -> None:
        self._post_controller(
            "/api/continuous",
            {
                "dir": -1,
                "max_vel": self._current_speed(),
                "max_accel": self.max_acceleration,
            },
        )

    def _current_speed(self) -> int:
        with self._lock:
            return self._speed

    def _wait_until_stopped(self, timeout: float) -> None:
        """Wait for the controller to finish a controlled stop before a new move."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            telemetry = self._read_telemetry()
            if not self._telemetry_is_fresh(telemetry):
                raise RuntimeError("Controller telemetry stopped updating.")
            if self._is_stationary(telemetry):
                return
            time.sleep(self.poll_seconds)
        raise TimeoutError("Timed out waiting for the conveyor to stop.")

    def _post_controller(
        self, endpoint: str, payload: dict[str, Any] | None = None
    ) -> None:
        """Send one command through the Pi's existing, single-owner controller API."""
        if endpoint in ("/api/continuous", "/api/move") and self._stop_event.is_set():
            raise InterruptedError("Sorting was stopped.")
        with self._command_lock:
            if endpoint in ("/api/continuous", "/api/move") and self._stop_event.is_set():
                raise InterruptedError("Sorting was stopped.")
            response = requests.post(
                f"{self.conveyor_url}{endpoint}",
                json=payload or {},
                timeout=self.request_timeout,
            )
        response.raise_for_status()

    def _save_inspection(self, inspection: dict[str, Any]) -> dict[str, Any]:
        """Write an annotated crop, append a CSV row, and update this session's dashboard."""
        with self._lock:
            if self._session_dir is None:
                raise RuntimeError("The active sorting session has no output folder.")
            box_number = len(self._records) + 1
            filename = f"box_{box_number:04d}.jpg"
            image_path = self._session_dir / filename
            temporary_path = self._session_dir / f".{filename}.tmp"
            temporary_path.write_bytes(inspection["image"])
            temporary_path.replace(image_path)

            record = {
                "box_number": box_number,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "status": inspection["status"],
                "defect_types": inspection["defect_types"],
                "sticker_color": inspection["sticker_color"],
                "confidence": max(
                    (item["confidence"] for item in inspection["detections"]),
                    default=0.0,
                ),
                "image_file": filename,
                "detections": inspection["detections"],
                "sticker_info": inspection["sticker_info"],
                "inference_ms": inspection["inference_ms"],
            }
            self._records.append(record)
            self._last_box = record
            self._append_csv_row_locked(record)
            self._latest_frame = inspection["image"]
            return record

    def _csv_row(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "row_type": "box",
            "box_number": record["box_number"],
            "captured_at": record["captured_at"],
            "status": record["status"],
            "defect_types": ";".join(record["defect_types"]),
            "sticker_color": record["sticker_color"],
            "confidence": record["confidence"],
            "image_file": record["image_file"],
            "detections_json": json.dumps(record["detections"], separators=(",", ":")),
            "metric": "",
            "value": "",
        }

    def _performance_rows_locked(self) -> list[dict[str, Any]]:
        """Create CSV rows that summarize total boxes and per-class defects."""
        good = sum(record["status"] == "Good" for record in self._records)
        defective = len(self._records) - good
        rows = [
            {"metric": "total_boxes", "value": len(self._records)},
            {"metric": "good_boxes", "value": good},
            {"metric": "defective_boxes", "value": defective},
        ]
        counts = {name: 0 for name in self.inspector.class_names}
        for record in self._records:
            for defect in record["defect_types"]:
                counts[defect] = counts.get(defect, 0) + 1
        rows.extend(
            {"metric": f"boxes_with_{name}", "value": count}
            for name, count in counts.items()
        )
        return [
            {
                "row_type": "performance",
                "box_number": "",
                "captured_at": "",
                "status": "",
                "defect_types": "",
                "sticker_color": "",
                "confidence": "",
                "image_file": "",
                "detections_json": "",
                **row,
            }
            for row in rows
        ]

    def _write_csv_header(self) -> None:
        with self._lock:
            if self._session_dir is None:
                return
            with (self._session_dir / "boxes.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                csv.DictWriter(file, fieldnames=CSV_FIELDS).writeheader()

    def _append_csv_row_locked(self, record: dict[str, Any]) -> None:
        if self._session_dir is None:
            raise RuntimeError("The active sorting session has no output folder.")
        with (self._session_dir / "boxes.csv").open(
            "a", newline="", encoding="utf-8"
        ) as file:
            csv.DictWriter(file, fieldnames=CSV_FIELDS).writerow(self._csv_row(record))

    def _write_session_csv_locked(self) -> None:
        if self._session_dir is None:
            return
        with (self._session_dir / "boxes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._csv_row(record) for record in self._records)

    def _set_state(self, state: str, message: str) -> None:
        with self._lock:
            self._set_state_locked(state, message)

    def _set_state_locked(self, state: str, message: str) -> None:
        self._state = state
        self._message = message
        logger.info("session=%s state=%s message=%s", self._session_id, state, message)

    def _finish(self, state: str, message: str) -> None:
        with self._lock:
            self._error = message if state == "error" else None
            self._set_state_locked(state, message)
            if self._worker is threading.current_thread():
                self._worker = None

    def close(self) -> None:
        """Stop the motion worker and motor before shutting down the web service."""
        with self._lock:
            active = self._worker is not None and self._worker.is_alive()
        if active:
            self.emergency_stop()


def create_app(config_path: Path = CONFIG_PATH) -> tuple[Flask, SorterController]:
    """Build the LAN dashboard and its JSON, image, and export endpoints."""
    config = load_config(config_path)
    inspector = BoxInspector(config)
    controller = SorterController(config, inspector)
    web = Flask(__name__, template_folder="templates")

    @web.get("/")
    def index():
        return render_template(
            "index.html",
            speed_min=controller.min_speed,
            speed_max=controller.max_speed,
            default_speed=controller._current_speed(),
        )

    @web.get("/api/state")
    def state():
        return jsonify(controller.refresh_status())

    @web.post("/api/start")
    def start():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="A JSON safety confirmation is required."), 400
        try:
            return jsonify(controller.start(body.get("confirmed_clear") is True)), 202
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409

    @web.post("/api/stop")
    def stop():
        try:
            return jsonify(controller.stop())
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503

    @web.post("/api/emergency-stop")
    def emergency_stop():
        try:
            return jsonify(controller.emergency_stop())
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503

    @web.post("/api/speed")
    def speed():
        body = request.get_json(silent=True) or {}
        try:
            speed_value = body["speed"]
            if isinstance(speed_value, bool):
                raise ValueError("Speed must be an integer.")
            parsed_speed = int(speed_value)
            if float(speed_value) != parsed_speed:
                raise ValueError("Speed must be an integer.")
            return jsonify(controller.set_speed(parsed_speed))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return jsonify(error=str(exc)), 400

    @web.get("/api/export.csv")
    def export_csv():
        try:
            export = controller.export_csv()
        except FileNotFoundError as exc:
            abort(404, description=str(exc))
        return send_file(
            export,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"{controller.snapshot()['session_id']}_boxes.csv",
        )

    @web.get("/api/export/images.zip")
    def export_zip():
        try:
            export = controller.export_zip()
        except FileNotFoundError as exc:
            abort(404, description=str(exc))
        return send_file(
            export,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{controller.snapshot()['session_id']}_images.zip",
        )

    @web.get("/sessions/<session_id>/<filename>")
    def session_image(session_id: str, filename: str):
        if not filename.startswith("box_") or not filename.endswith(".jpg"):
            abort(404)
        try:
            image_path = controller.current_image(session_id, filename)
        except FileNotFoundError:
            abort(404)
        return send_from_directory(image_path.parent, image_path.name)

    @web.get("/api/preview.jpg")
    def preview():
        image = controller._latest_frame
        if image is None:
            return jsonify(error="No sorter capture is available yet."), 404
        return web.response_class(image, mimetype="image/jpeg")

    web.extensions["sorter_controller"] = controller
    return web, controller


def main() -> None:
    """Run the LAN server; service termination always stops active conveyor motion."""
    web, controller = create_app()
    config = controller.config

    def shutdown(_signum: int, _frame: Any) -> None:
        logger.warning("Shutdown requested; stopping conveyor before exit.")
        controller.close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    host = str(config["app"]["host"])
    port = int(config["app"]["port"])
    logger.info(
        "Box sorter listening on http://%s:%d/; use the existing Pi camera service.",
        host,
        port,
    )
    web.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
