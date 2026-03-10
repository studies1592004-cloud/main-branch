"""
tests/test_multi_object_tracking.py
=====================================
Unit tests for code/standardized/multi_object_tracking.py

Run:
    pytest tests/test_multi_object_tracking.py -v

Dependencies:
    pip install numpy opencv-python pytest
    (No GPU or supervision installation needed — mocked.)
"""

import numpy as np
import pytest
import sys
import os
import types
import csv
import json
import tempfile

# ── Mock ultralytics + supervision ───────────────────────────────────────────
for mod in ("ultralytics", "roboflow"):
    sys.modules[mod] = types.ModuleType(mod)
sys.modules["ultralytics"].YOLO = object

sup = types.ModuleType("supervision")
class MockDetections:
    def __init__(self, xyxy, confidence, class_id, tracker_id=None):
        self.xyxy       = xyxy
        self.confidence = confidence
        self.class_id   = class_id
        self.tracker_id = tracker_id
    def __len__(self): return len(self.xyxy)
    def __getitem__(self, mask):
        return MockDetections(
            self.xyxy[mask], self.confidence[mask], self.class_id[mask],
            self.tracker_id[mask] if self.tracker_id is not None else None,
        )
sup.Detections    = MockDetections
sup.ByteTracker   = object
sys.modules["supervision"] = sup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "standardized"))
from multi_object_tracking import (
    DetectionResult,
    TrackHistory,
    MOTEvaluator,
    TrackExporter,
    Visualizer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def two_boxes():
    return np.array([
        [10., 20., 100., 120.],
        [200., 50., 350., 200.],
    ], dtype=np.float32)

@pytest.fixture
def basic_result(two_boxes):
    return DetectionResult(
        frame_id=0,
        boxes_xyxy=two_boxes,
        confidences=np.array([0.9, 0.75], dtype=np.float32),
        class_ids=np.array([0, 1], dtype=np.int32),
        track_ids=np.array([1, 2], dtype=np.int32),
        class_names={0: "person", 1: "car"},
    )

@pytest.fixture
def populated_history():
    """TrackHistory with 10 frames of two tracks."""
    th = TrackHistory(max_trail_length=10, fps=30.0)
    for f in range(10):
        boxes = np.array([
            [float(f * 5), 100., float(f * 5 + 50), 200.],
            [300., float(f * 3), 400., float(f * 3 + 60)],
        ], dtype=np.float32)
        result = DetectionResult(
            f, boxes,
            np.array([0.9, 0.8]),
            np.array([0, 0]),
            np.array([1, 2]),
        )
        th.update(result)
    return th

@pytest.fixture
def gt_csv(tmp_path):
    """Write a minimal MOT-format GT CSV and return its path."""
    path = tmp_path / "gt.txt"
    rows = [
        [1, 1, 10, 20, 50, 80, 1, -1, -1, -1],
        [1, 2, 200, 50, 60, 100, 1, -1, -1, -1],
        [2, 1, 15, 22, 50, 80, 1, -1, -1, -1],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return str(path)


# ── DetectionResult ───────────────────────────────────────────────────────────

class TestDetectionResult:
    def test_repr(self, basic_result):
        assert "DetectionResult" in repr(basic_result)

    def test_len(self, basic_result):
        assert len(basic_result) == 2

    def test_from_config(self):
        dr = DetectionResult.from_config({"frame_id": 5})
        assert dr.frame_id == 5
        assert len(dr) == 0

    def test_to_dict_structure(self, basic_result):
        d = basic_result.to_dict()
        assert d["frame_id"] == 0
        assert len(d["detections"]) == 2

    def test_to_dict_track_ids(self, basic_result):
        d = basic_result.to_dict()
        tids = [det["track_id"] for det in d["detections"]]
        assert set(tids) == {1, 2}

    def test_to_dict_class_names(self, basic_result):
        d = basic_result.to_dict()
        names = {det["class_name"] for det in d["detections"]}
        assert "person" in names
        assert "car" in names

    def test_filter_by_class_keeps_correct(self, basic_result):
        filtered = basic_result.filter_by_class([0])
        assert len(filtered) == 1
        assert filtered.class_ids[0] == 0

    def test_filter_by_class_empty_result(self, basic_result):
        filtered = basic_result.filter_by_class([99])
        assert len(filtered) == 0

    def test_filter_by_class_multiple(self, basic_result):
        filtered = basic_result.filter_by_class([0, 1])
        assert len(filtered) == 2

    def test_filter_preserves_frame_id(self, basic_result):
        filtered = basic_result.filter_by_class([0])
        assert filtered.frame_id == basic_result.frame_id


# ── TrackHistory ──────────────────────────────────────────────────────────────

class TestTrackHistory:
    def test_repr(self):
        th = TrackHistory()
        assert "TrackHistory" in repr(th)

    def test_from_config(self):
        th = TrackHistory.from_config({"max_trail_length": 20, "fps": 25.0})
        assert th.max_trail_length == 20
        assert abs(th.fps - 25.0) < 1e-6

    def test_get_trail_length_capped(self, populated_history):
        trail = populated_history.get_trail(1)
        assert len(trail) <= populated_history.max_trail_length

    def test_get_trail_returns_tuples(self, populated_history):
        trail = populated_history.get_trail(1)
        for pt in trail:
            assert len(pt) == 2

    def test_get_trail_unknown_id_empty(self, populated_history):
        assert populated_history.get_trail(999) == []

    def test_track_lifetime_correct(self, populated_history):
        assert populated_history.track_lifetime(1) == 10
        assert populated_history.track_lifetime(2) == 10

    def test_track_lifetime_unknown_zero(self, populated_history):
        assert populated_history.track_lifetime(999) == 0

    def test_estimate_speed_positive(self, populated_history):
        spd = populated_history.estimate_speed(1)
        assert spd > 0.0

    def test_estimate_speed_stationary(self):
        th = TrackHistory(fps=30.0)
        for f in range(10):
            result = DetectionResult(
                f,
                np.array([[100., 100., 200., 200.]], dtype=np.float32),
                np.array([0.9]),
                np.array([0]),
                np.array([5]),
            )
            th.update(result)
        spd = th.estimate_speed(5)
        assert spd < 0.5   # stationary → near-zero speed

    def test_estimate_speed_unknown_track(self, populated_history):
        assert populated_history.estimate_speed(999) == 0.0

    def test_summary_total_tracks(self, populated_history):
        s = populated_history.summary()
        assert s["total_tracks"] == 2

    def test_summary_mean_lifetime(self, populated_history):
        s = populated_history.summary()
        assert abs(s["mean_lifetime_frames"] - 10.0) < 1e-6

    def test_summary_keys(self, populated_history):
        s = populated_history.summary()
        for key in ("total_tracks", "mean_lifetime_frames",
                    "max_lifetime_frames", "active_track_ids"):
            assert key in s

    def test_update_untracked_ignored(self):
        """Track IDs of -1 should not be added to history."""
        th = TrackHistory()
        result = DetectionResult(
            0,
            np.array([[0., 0., 50., 50.]], dtype=np.float32),
            np.array([0.9]),
            np.array([0]),
            np.array([-1]),   # untracked
        )
        th.update(result)
        assert th.summary()["total_tracks"] == 0


# ── MOTEvaluator ─────────────────────────────────────────────────────────────

class TestMOTEvaluator:
    def test_repr(self):
        ev = MOTEvaluator()
        assert "MOTEvaluator" in repr(ev)

    def test_from_config(self):
        ev = MOTEvaluator.from_config({"iou_threshold": 0.75})
        assert abs(ev.iou_threshold - 0.75) < 1e-6

    def test_box_iou_identical(self):
        iou = MOTEvaluator._box_iou(
            np.array([0., 0., 10., 10.]),
            np.array([0., 0., 10., 10.])
        )
        assert abs(iou - 1.0) < 1e-5

    def test_box_iou_no_overlap(self):
        iou = MOTEvaluator._box_iou(
            np.array([0., 0., 5., 5.]),
            np.array([10., 10., 15., 15.])
        )
        assert iou == 0.0

    def test_box_iou_partial(self):
        iou = MOTEvaluator._box_iou(
            np.array([0., 0., 10., 10.]),
            np.array([5., 0., 15., 10.])
        )
        expected = 50.0 / 150.0
        assert abs(iou - expected) < 1e-4

    def test_box_iou_symmetry(self):
        a = np.array([0., 0., 10., 20.])
        b = np.array([5., 5., 15., 25.])
        assert abs(MOTEvaluator._box_iou(a, b) - MOTEvaluator._box_iou(b, a)) < 1e-6

    def test_load_gt_frame_count(self, gt_csv):
        ev = MOTEvaluator()
        gt = ev.load_gt(gt_csv)
        assert len(gt) == 2   # 2 unique frames

    def test_load_gt_object_count(self, gt_csv):
        ev = MOTEvaluator()
        gt = ev.load_gt(gt_csv)
        total = sum(len(v) for v in gt.values())
        assert total == 3

    def test_load_gt_missing_file_raises(self):
        ev = MOTEvaluator()
        with pytest.raises(FileNotFoundError):
            ev.load_gt("/nonexistent/gt.txt")

    def test_evaluate_perfect_mota(self, gt_csv):
        ev = MOTEvaluator()
        gt = ev.load_gt(gt_csv)
        # Build predictions that exactly match GT
        results = []
        for frame_id, gt_objs in gt.items():
            boxes = np.array([g["box_xyxy"] for g in gt_objs], dtype=np.float32)
            tids  = np.array([g["id"]       for g in gt_objs], dtype=np.int32)
            results.append(DetectionResult(
                frame_id - 1, boxes,
                np.ones(len(boxes)), np.zeros(len(boxes), dtype=np.int32), tids
            ))
        metrics = ev.evaluate(results, gt)
        assert abs(metrics["MOTA"] - 1.0) < 0.01
        assert abs(metrics["IDF1"] - 1.0) < 0.01

    def test_evaluate_no_predictions_mota_negative_or_zero(self, gt_csv):
        ev = MOTEvaluator()
        gt = ev.load_gt(gt_csv)
        results = [
            DetectionResult(f, np.empty((0, 4), np.float32),
                            np.empty(0), np.empty(0, dtype=np.int32),
                            np.empty(0, dtype=np.int32))
            for f in range(max(gt.keys()))
        ]
        metrics = ev.evaluate(results, gt)
        assert metrics["MOTA"] <= 0.0

    def test_evaluate_metric_keys(self, gt_csv):
        ev = MOTEvaluator()
        gt = ev.load_gt(gt_csv)
        metrics = ev.evaluate([], gt)
        for key in ("MOTA", "IDF1", "precision", "recall",
                    "num_tp", "num_fp", "num_fn", "num_id_switches"):
            assert key in metrics


# ── TrackExporter ─────────────────────────────────────────────────────────────

class TestTrackExporter:
    def test_repr(self, tmp_path):
        exp = TrackExporter(output_dir=str(tmp_path))
        assert "TrackExporter" in repr(exp)

    def test_from_config(self, tmp_path):
        exp = TrackExporter.from_config({"log_dir": str(tmp_path)})
        assert exp.output_dir == str(tmp_path)

    def _make_results(self, n_frames=5):
        return [
            DetectionResult(
                f,
                np.array([[float(f * 10), 50., float(f * 10 + 80), 150.]], dtype=np.float32),
                np.array([0.9]),
                np.array([0]),
                np.array([1]),
                {0: "person"},
            )
            for f in range(n_frames)
        ]

    def test_to_mot_csv_creates_file(self, tmp_path):
        exp  = TrackExporter(output_dir=str(tmp_path))
        path = exp.to_mot_csv(self._make_results(), filename="tracks_test.csv")
        assert os.path.isfile(path)

    def test_to_mot_csv_row_count(self, tmp_path):
        exp     = TrackExporter(output_dir=str(tmp_path))
        results = self._make_results(5)
        path    = exp.to_mot_csv(results, filename="tracks_count.csv")
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 5   # one row per frame (one track each)

    def test_to_mot_csv_1_indexed_frames(self, tmp_path):
        exp  = TrackExporter(output_dir=str(tmp_path))
        path = exp.to_mot_csv(self._make_results(1), filename="frame_idx.csv")
        with open(path) as f:
            row = next(csv.reader(f))
        assert int(row[0]) == 1   # MOT standard: 1-indexed

    def test_to_mot_csv_skips_untracked(self, tmp_path):
        exp = TrackExporter(output_dir=str(tmp_path))
        result = DetectionResult(
            0,
            np.array([[10., 10., 50., 50.]], dtype=np.float32),
            np.array([0.9]),
            np.array([0]),
            np.array([-1]),   # untracked
        )
        path = exp.to_mot_csv([result], filename="untracked.csv")
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 0

    def test_to_json_creates_file(self, tmp_path):
        exp  = TrackExporter(output_dir=str(tmp_path))
        path = exp.to_json(self._make_results(), filename="tracks_test.json")
        assert os.path.isfile(path)

    def test_to_json_frame_count(self, tmp_path):
        exp  = TrackExporter(output_dir=str(tmp_path))
        path = exp.to_json(self._make_results(4), filename="json_count.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 4

    def test_to_json_structure(self, tmp_path):
        exp  = TrackExporter(output_dir=str(tmp_path))
        path = exp.to_json(self._make_results(1), filename="json_struct.json")
        with open(path) as f:
            data = json.load(f)
        frame = data[0]
        assert "frame_id" in frame
        assert "detections" in frame

    def test_to_csv_summary_creates_file(self, tmp_path, populated_history):
        exp  = TrackExporter(output_dir=str(tmp_path))
        path = exp.to_csv_summary(populated_history, filename="summary_test.csv")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


# ── Visualizer ────────────────────────────────────────────────────────────────

class TestVisualizer:
    def test_repr(self, tmp_path):
        v = Visualizer(output_dir=str(tmp_path))
        assert "Visualizer" in repr(v)

    def test_from_config(self, tmp_path):
        v = Visualizer.from_config({"log_dir": str(tmp_path)})
        assert v.output_dir == str(tmp_path)

    def test_annotate_frame_output_shape(self, basic_result):
        v     = Visualizer(output_dir="/tmp")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out   = v.annotate_frame(frame, basic_result)
        assert out.shape == frame.shape

    def test_annotate_frame_does_not_modify_input(self, basic_result):
        v      = Visualizer(output_dir="/tmp")
        frame  = np.zeros((480, 640, 3), dtype=np.uint8)
        orig   = frame.copy()
        v.annotate_frame(frame, basic_result)
        assert np.array_equal(frame, orig)

    def test_annotate_frame_with_trail(self, basic_result, populated_history, tmp_path):
        v   = Visualizer(output_dir=str(tmp_path))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = v.annotate_frame(img, basic_result, populated_history)
        assert out.shape == img.shape

    def test_save_frame_creates_file(self, tmp_path, basic_result):
        v    = Visualizer(output_dir=str(tmp_path))
        img  = np.zeros((128, 128, 3), dtype=np.uint8)
        path = v.save_frame(img, basic_result, None, "frame_test.jpg")
        assert os.path.isfile(path)

    def test_plot_track_count_creates_file(self, tmp_path, basic_result):
        v    = Visualizer(output_dir=str(tmp_path))
        path = v.plot_track_count_over_time([basic_result] * 20,
                                             filename="count_test.png")
        assert os.path.isfile(path)

    def test_plot_track_lifetimes_creates_file(self, tmp_path, populated_history):
        v    = Visualizer(output_dir=str(tmp_path))
        path = v.plot_track_lifetimes(populated_history, filename="lt_test.png")
        assert os.path.isfile(path)

    def test_plot_metrics_bar_creates_file(self, tmp_path):
        v    = Visualizer(output_dir=str(tmp_path))
        path = v.plot_metrics_bar(
            {"MOTA": 0.82, "IDF1": 0.79, "precision": 0.91, "recall": 0.87},
            filename="metrics_test.png",
        )
        assert os.path.isfile(path)

    def test_color_cycles_by_track_id(self):
        v = Visualizer(output_dir="/tmp")
        c1 = v._color(0)
        c2 = v._color(len(v.PALETTE))  # should wrap around
        assert c1 == c2
