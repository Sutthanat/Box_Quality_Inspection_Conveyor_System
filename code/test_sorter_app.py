"""Hardware-free checks for crop/model geometry, dashboard data, and safe controls."""

import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import sorter_app


def test_config(directory: Path) -> dict:
    """Provide isolated settings so tests never target the Pi camera or motors."""
    return {
        "app": {"host": "127.0.0.1", "port": 0},
        "camera": {
            "url": "http://127.0.0.1:19000",
            "timeout_seconds": 1,
            "frame_width": 1280,
            "frame_height": 720,
        },
        "conveyor": {
            "url": "http://127.0.0.1:18000",
            "capture_app_status_url": None,
            "request_timeout_seconds": 1,
            "poll_seconds": 0.05,
            "telemetry_stale_seconds": 2,
            "motion_timeout_seconds": 120,
            "stop_timeout_seconds": 20,
            "max_acceleration": 10000,
            "min_speed": 500,
            "max_speed": 3000,
            "default_speed": 1500,
        },
        "yolo": {
            "model": "not-loaded-in-this-test.onnx",
            "input_size": 640,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "class_names": [
                "lid_open",
                "tape_defect",
                "sticker_defect",
                "crushed_corner",
                "stain",
            ],
        },
        "sorting": {
            "data_directory": str(directory / "sessions"),
            "settle_seconds": 0,
            "jpeg_quality": 90,
        },
    }


class FakeInspector:
    """Stand in for YOLO while exercising dashboard accounting."""

    class_names = ["lid_open", "tape_defect", "sticker_defect", "crushed_corner", "stain"]


class SorterTests(unittest.TestCase):
    def test_training_crop_uses_exact_1280_by_720_window(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        cropped = sorter_app.crop_fixed(frame)

        self.assertEqual(cropped.shape, (620, 320, 3))

    def test_letterbox_matches_model_input_and_training_crop_aspect(self):
        cropped = np.zeros((620, 320, 3), dtype=np.uint8)

        blob, scale, pad_x, pad_y = sorter_app._read_model_input(cropped, 640)

        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertAlmostEqual(scale, 640 / 620)
        self.assertEqual((pad_x, pad_y), (155, 0))

    def test_decoder_maps_box_to_crop_and_deduplicates_predictions(self):
        output = np.zeros((1, 9, 8400), dtype=np.float32)
        output[0, :4, 0] = [180, 90, 40, 30]
        output[0, 4 + 1, 0] = 0.9
        output[0, :4, 1] = [181, 90, 40, 30]
        output[0, 4 + 1, 1] = 0.7
        output[0, :4, 2] = [181, 90, 40, 30]
        output[0, 4 + 4, 2] = 0.8

        detections = sorter_app.decode_yolo_output(
            output,
            FakeInspector.class_names,
            crop_shape=(620, 320),
            scale=1,
            pad_x=155,
            pad_y=0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        self.assertEqual(len(detections), 2)
        self.assertEqual(
            {item["class_name"] for item in detections}, {"tape_defect", "stain"}
        )
        self.assertEqual(detections[0]["x"], 5)

    def test_performance_counts_defect_type_once_per_box_and_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = test_config(Path(temporary))
            controller = sorter_app.SorterController(config, FakeInspector())
            session = Path(temporary) / "session"
            session.mkdir()
            controller._session_dir = session
            controller._session_id = "session"
            controller._write_csv_header()
            ok, image = cv2.imencode(
                ".jpg", np.zeros((20, 20, 3), dtype=np.uint8)
            )
            self.assertTrue(ok)

            first = controller._save_inspection(
                {
                    "image": image.tobytes(),
                    "status": "Defective",
                    "defect_types": ["tape_defect"],
                    "sticker_color": "blue",
                    "detections": [
                        {"class_name": "tape_defect", "confidence": 0.8},
                        {"class_name": "tape_defect", "confidence": 0.6},
                    ],
                    "sticker_info": {},
                    "inference_ms": 80.0,
                }
            )
            controller._save_inspection(
                {
                    "image": image.tobytes(),
                    "status": "Good",
                    "defect_types": [],
                    "sticker_color": "yellow",
                    "detections": [],
                    "sticker_info": {},
                    "inference_ms": 78.0,
                }
            )

            snapshot = controller.snapshot()
            self.assertEqual(first["box_number"], 1)
            self.assertEqual(snapshot["box_count"], 2)
            self.assertEqual(snapshot["good_count"], 1)
            self.assertEqual(snapshot["defective_count"], 1)
            self.assertEqual(snapshot["defect_counts"]["tape_defect"], 1)

            exported = list(csv.DictReader(io.StringIO(controller.export_csv().getvalue().decode("utf-8-sig"))))
            self.assertEqual(
                [row["row_type"] for row in exported],
                ["box", "box"] + ["performance"] * 8,
            )
            self.assertEqual(
                next(row["value"] for row in exported if row["metric"] == "total_boxes"),
                "2",
            )
            with zipfile.ZipFile(controller.export_zip()) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        "boxes.csv",
                        "performance.csv",
                        "box_0001.jpg",
                        "box_0002.jpg",
                    },
                )

    def test_start_requires_clearance_confirmation_without_motor_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = sorter_app.SorterController(
                test_config(Path(temporary)), FakeInspector()
            )
            with patch.object(controller, "_verify_stationary_controller") as verify:
                with self.assertRaisesRegex(ValueError, "Confirm"):
                    controller.start(confirmed_clear=False)

            verify.assert_not_called()
            self.assertEqual(controller.snapshot()["box_count"], 0)

    def test_sensor_one_detection_does_not_stop_forward_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = sorter_app.SorterController(
                test_config(Path(temporary)), FakeInspector()
            )
            telemetry = [
                {"s1": 0, "s2": 0, "sl": 4, "_fresh": True},
                {"s1": 1, "s2": 0, "sl": 4, "_fresh": True},
                {"s1": 1, "s2": 1, "sl": 5, "_fresh": True},
            ]
            controller._read_telemetry = Mock(side_effect=telemetry)
            controller._set_state = Mock()
            controller._stop_event.clear()

            sensor1_at, sensor2_at = controller._wait_for_sensor2(4)

            self.assertLess(sensor1_at, sensor2_at)
            self.assertEqual(controller._read_telemetry.call_count, 3)
            controller._set_state.assert_called_once()

    def test_good_box_continues_through_sensor_two_without_stopping(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = sorter_app.SorterController(
                test_config(Path(temporary)), FakeInspector()
            )
            controller._read_telemetry = Mock(
                side_effect=[
                    {"s2": 0, "sl": 10, "_fresh": True},
                    {"s2": 1, "sl": 10, "_fresh": True},
                    {"s2": 0, "sl": 10, "_fresh": True},
                    {"s2": 0, "sl": 11, "_fresh": True},
                ]
            )
            forward_sensors = []
            controller._continuous_forward = Mock(
                side_effect=lambda sensor: forward_sensors.append(sensor)
            )
            controller._set_state = Mock()
            controller._stop_event.clear()

            baseline = controller._good_box_cycle()

            self.assertEqual(forward_sensors, [None, 2])
            self.assertEqual(baseline, 11)

    def test_defective_box_stops_then_waits_for_sensor_two_to_clear(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = sorter_app.SorterController(
                test_config(Path(temporary)), FakeInspector()
            )
            controller._read_telemetry = Mock(
                side_effect=[
                    {"s2": 0, "sl": 11, "_fresh": True},
                    {"s2": 1, "sl": 12, "_fresh": True},
                    {"s2": 0, "sl": 12, "_fresh": True},
                ]
            )
            forward_sensors = []
            controller._continuous_forward = Mock(
                side_effect=lambda sensor: forward_sensors.append(sensor)
            )
            controller._wait_for_land_count = Mock()
            controller._wait_until_stopped = Mock()
            controller._set_state = Mock()
            controller._stop_event.clear()

            baseline = controller._defective_box_cycle(11)

            self.assertEqual(forward_sensors, [2, 2])
            self.assertEqual(baseline, 12)
            controller._wait_for_land_count.assert_called_once_with(11)
            controller._wait_until_stopped.assert_called_once_with(timeout=20)

    def test_stop_event_prevents_a_late_motion_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = sorter_app.SorterController(
                test_config(Path(temporary)), FakeInspector()
            )
            controller._stop_event.set()

            with patch.object(sorter_app.requests, "post") as post:
                with self.assertRaises(InterruptedError):
                    controller._continuous_forward(sensor=2)

            post.assert_not_called()

    def test_speed_is_bounded_and_changes_only_the_configured_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = sorter_app.SorterController(
                test_config(Path(temporary)), FakeInspector()
            )

            result = controller.set_speed(2200)

            self.assertEqual(result["speed"], 2200)
            with self.assertRaisesRegex(ValueError, "between 500 and 3000"):
                controller.set_speed(4000)

    def test_web_start_and_speed_validation_never_command_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = test_config(Path(temporary))
            config_path = Path(temporary) / "config.yaml"
            config_path.write_text(sorter_app.yaml.safe_dump(config), encoding="utf-8")
            with patch.object(sorter_app, "BoxInspector", return_value=FakeInspector()):
                web, _controller = sorter_app.create_app(config_path)
            client = web.test_client()

            page = client.get("/")
            no_confirmation = client.post(
                "/api/start", json={"confirmed_clear": False}
            )
            invalid_speed = client.post("/api/speed", json={"speed": 4000})

            self.assertEqual(page.status_code, 200)
            self.assertIn(b"Start new session", page.data)
            self.assertEqual(no_confirmation.status_code, 400)
            self.assertIn(b"Confirm", no_confirmation.data)
            self.assertEqual(invalid_speed.status_code, 400)

    def test_csv_rows_are_standard_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = sorter_app.SorterController(
                test_config(Path(temporary)), FakeInspector()
            )
            controller._session_dir = Path(temporary)
            controller._session_id = "csv"
            controller._records = [
                {
                    "box_number": 1,
                    "captured_at": "2026-01-01T00:00:00+00:00",
                    "status": "Good",
                    "defect_types": [],
                    "sticker_color": "none",
                    "confidence": 0.0,
                    "image_file": "box_0001.jpg",
                    "detections": [],
                }
            ]

            data = controller.export_csv().getvalue().decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(data)))

            self.assertEqual(rows[0]["row_type"], "box")
            self.assertEqual(rows[0]["status"], "Good")
            self.assertTrue(
                any(
                    row["row_type"] == "performance"
                    and row["metric"] == "good_boxes"
                    and row["value"] == "1"
                    for row in rows
                )
            )


if __name__ == "__main__":
    unittest.main()
