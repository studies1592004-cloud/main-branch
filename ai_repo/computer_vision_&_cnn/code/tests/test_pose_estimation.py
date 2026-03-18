"""
tests/test_pose_estimation.py
===============================
Unit tests for code/standardized/pose_estimation.py

Run:
    pytest tests/test_pose_estimation.py -v

Dependencies:
    pip install numpy opencv-python pytest
    (No Roboflow API key, GPU, or MediaPipe needed.)
"""

import numpy as np
import pytest
import sys
import os
import types

# ── Mock ultralytics + roboflow + mediapipe ───────────────────────────────────
for mod in ("ultralytics", "roboflow", "mediapipe",
            "mediapipe.solutions", "mediapipe.solutions.pose"):
    sys.modules[mod] = types.ModuleType(mod)
sys.modules["ultralytics"].YOLO = object

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "standardized"))
from pose_estimation import (
    KeypointNormalizer,
    ActionClassifier,
    Evaluator,
    Visualizer,
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_person_kps(positions: dict, n_kps: int = 17) -> np.ndarray:
    """Build a (n_kps, 3) array from a name→(x,y) dict. Unset kps have vis=0."""
    kps = np.zeros((n_kps, 3), dtype=np.float32)
    for name, (x, y) in positions.items():
        idx = COCO_KEYPOINT_NAMES.index(name)
        kps[idx] = [x, y, 1.0]
    return kps


# ── KeypointNormalizer ────────────────────────────────────────────────────────

class TestKeypointNormalizer:
    def test_repr(self):
        kn = KeypointNormalizer()
        assert "KeypointNormalizer" in repr(kn)

    def test_from_config(self):
        kn = KeypointNormalizer.from_config({"num_keypoints": 17, "image_size": 512})
        assert kn.image_size == 512

    # to_pixels
    def test_to_pixels_shape_2col(self):
        kn  = KeypointNormalizer()
        src = np.random.rand(17, 2).astype(np.float32)
        out = kn.to_pixels(src, (480, 640))
        assert out.shape == (17, 2)

    def test_to_pixels_shape_3col(self):
        kn  = KeypointNormalizer()
        src = np.random.rand(17, 3).astype(np.float32)
        out = kn.to_pixels(src, (480, 640))
        assert out.shape == (17, 3)

    def test_to_pixels_scales_correctly(self):
        kn  = KeypointNormalizer()
        src = np.array([[0.5, 0.5, 1.0]], dtype=np.float32)
        out = kn.to_pixels(src, (100, 200))
        assert abs(out[0, 0] - 100.0) < 0.01   # x: 0.5 × 200 = 100
        assert abs(out[0, 1] - 50.0)  < 0.01   # y: 0.5 × 100 = 50

    # to_normalised
    def test_to_normalised_round_trip(self):
        kn   = KeypointNormalizer()
        orig = np.random.rand(17, 2).astype(np.float32)
        px   = kn.to_pixels(orig, (480, 640))
        back = kn.to_normalised(px, (480, 640))
        assert np.allclose(back, orig, atol=1e-4)

    # compute_oks
    def test_oks_perfect(self):
        kn   = KeypointNormalizer()
        kps  = np.random.rand(17, 2).astype(np.float32) * 200 + 100
        vis  = np.ones(17, dtype=np.float32)
        bbox = np.array([50., 50., 400., 400.])
        oks  = kn.compute_oks(kps, kps, bbox, vis)
        assert abs(oks - 1.0) < 0.01

    def test_oks_far_off(self):
        kn   = KeypointNormalizer()
        kps  = np.random.rand(17, 2).astype(np.float32) * 100 + 50
        vis  = np.ones(17, dtype=np.float32)
        bbox = np.array([0., 0., 100., 100.])
        oks  = kn.compute_oks(kps + 1000.0, kps, bbox, vis)
        assert oks < 0.05

    def test_oks_all_invisible_returns_zero(self):
        kn   = KeypointNormalizer()
        kps  = np.random.rand(17, 2).astype(np.float32) * 100
        vis  = np.zeros(17, dtype=np.float32)
        bbox = np.array([0., 0., 200., 200.])
        oks  = kn.compute_oks(kps, kps, bbox, vis)
        assert oks == 0.0

    def test_oks_in_range(self):
        kn   = KeypointNormalizer()
        pred = np.random.rand(17, 2).astype(np.float32) * 200
        gt   = np.random.rand(17, 2).astype(np.float32) * 200
        vis  = np.ones(17, dtype=np.float32)
        oks  = kn.compute_oks(pred, gt, None, vis)
        assert 0.0 <= oks <= 1.0

    # procrustes_align
    def test_procrustes_output_shape(self):
        kn  = KeypointNormalizer()
        src = np.random.rand(17, 2).astype(np.float32) * 100
        tgt = np.random.rand(17, 2).astype(np.float32) * 100
        out = kn.procrustes_align(src, tgt)
        assert out.shape == (17, 2)

    def test_procrustes_centroid_matches_target(self):
        kn  = KeypointNormalizer()
        src = np.random.rand(17, 2).astype(np.float32) * 100
        tgt = src * 2.5 + np.array([50., 30.])
        out = kn.procrustes_align(src, tgt)
        assert np.allclose(out.mean(axis=0), tgt.mean(axis=0), atol=1.0)


# ── ActionClassifier ──────────────────────────────────────────────────────────

class TestActionClassifier:
    def test_repr(self):
        ac = ActionClassifier()
        assert "ActionClassifier" in repr(ac)

    def test_from_config(self):
        ac = ActionClassifier.from_config({"conf_threshold": 0.3})
        assert abs(ac.conf_threshold - 0.3) < 1e-6

    def test_arms_raised(self):
        """Wrists above shoulders → arms_raised."""
        ac = ActionClassifier()
        kps = make_person_kps({
            "left_shoulder":  (200, 300), "right_shoulder": (300, 300),
            "left_wrist":     (190, 100), "right_wrist":    (310, 100),
            "left_elbow":     (195, 200), "right_elbow":    (305, 200),
            "left_hip":       (210, 400), "right_hip":      (290, 400),
            "left_knee":      (210, 500), "left_ankle":     (210, 600),
        })
        assert ac.classify(kps) == "arms_raised"

    def test_standing(self):
        """Straight legs → standing."""
        ac = ActionClassifier()
        kps = make_person_kps({
            "left_shoulder":  (200, 100), "right_shoulder": (300, 100),
            "left_elbow":     (180, 200), "right_elbow":    (320, 200),
            "left_wrist":     (170, 300), "right_wrist":    (330, 300),
            "left_hip":       (210, 300), "right_hip":      (290, 300),
            "left_knee":      (210, 400), "right_knee":     (290, 400),
            "left_ankle":     (210, 500), "right_ankle":    (290, 500),
        })
        assert ac.classify(kps) in ("standing", "unknown")

    def test_classify_returns_string(self):
        ac  = ActionClassifier()
        kps = np.zeros((17, 3), dtype=np.float32)
        result = ac.classify(kps)
        assert isinstance(result, str)

    def test_classify_batch_length(self):
        ac = ActionClassifier()
        persons = [
            {"keypoints": np.zeros((17, 3), dtype=np.float32)},
            {"keypoints": np.zeros((17, 3), dtype=np.float32)},
            {"keypoints": np.zeros((17, 3), dtype=np.float32)},
        ]
        results = ac.classify_batch(persons)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_classify_empty_batch(self):
        ac = ActionClassifier()
        assert ac.classify_batch([]) == []

    def test_classify_known_actions_subset(self):
        """All returned labels should be from the defined action set."""
        ac      = ActionClassifier()
        valid   = set(ac._rules.keys())
        kps_set = [np.random.rand(17, 3).astype(np.float32) for _ in range(10)]
        for kps in kps_set:
            assert ac.classify(kps) in valid


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_repr(self):
        ev = Evaluator()
        assert "Evaluator" in repr(ev)

    def test_from_config(self):
        ev = Evaluator.from_config({"num_keypoints": 17, "pck_threshold": 0.1})
        assert abs(ev.pck_threshold - 0.1) < 1e-6

    # compute_pck
    def test_pck_perfect(self):
        ev   = Evaluator()
        kps  = np.random.rand(17, 2).astype(np.float32) * 200 + 50
        bbox = np.array([0., 0., 400., 400.])
        vis  = np.ones(17, dtype=np.float32)
        res  = ev.compute_pck(kps, kps, bbox, vis)
        assert abs(res["pck"] - 1.0) < 1e-6

    def test_pck_zero(self):
        ev   = Evaluator()
        pred = np.random.rand(17, 2).astype(np.float32) * 50
        gt   = pred + 1000.0
        bbox = np.array([0., 0., 100., 100.])
        vis  = np.ones(17, dtype=np.float32)
        res  = ev.compute_pck(pred, gt, bbox, vis)
        assert abs(res["pck"] - 0.0) < 1e-6

    def test_pck_in_range(self):
        ev   = Evaluator()
        pred = np.random.rand(17, 2).astype(np.float32) * 200
        gt   = np.random.rand(17, 2).astype(np.float32) * 200
        bbox = np.array([0., 0., 300., 300.])
        vis  = np.ones(17, dtype=np.float32)
        res  = ev.compute_pck(pred, gt, bbox, vis)
        assert 0.0 <= res["pck"] <= 1.0

    def test_pck_ignores_invisible_keypoints(self):
        ev   = Evaluator()
        pred = np.zeros((17, 2), dtype=np.float32)
        gt   = np.ones((17, 2), dtype=np.float32) * 1000.0   # all wrong
        bbox = np.array([0., 0., 100., 100.])
        vis  = np.zeros(17, dtype=np.float32)   # all invisible
        res  = ev.compute_pck(pred, gt, bbox, vis)
        # With all invisible, PCK should be 0 (no valid points)
        assert res["num_valid"] == 0

    def test_pck_per_keypoint_length(self):
        ev   = Evaluator(num_keypoints=17)
        pred = np.random.rand(17, 2).astype(np.float32) * 100
        gt   = pred.copy()
        bbox = np.array([0., 0., 200., 200.])
        vis  = np.ones(17, dtype=np.float32)
        res  = ev.compute_pck(pred, gt, bbox, vis)
        assert len(res["per_keypoint_correct"]) == 17

    # compute_dataset_metrics
    def test_dataset_metrics_perfect(self):
        ev   = Evaluator()
        kps  = np.random.rand(17, 2).astype(np.float32) * 200
        vis  = np.ones(17, dtype=np.float32)
        kps3 = np.concatenate([kps, vis[:, None]], axis=1)
        person = {"keypoints": kps3, "box": [0., 0., 300., 300.]}
        metrics = ev.compute_dataset_metrics([[person]], [[person]])
        assert abs(metrics["pck"]  - 1.0) < 0.01
        assert abs(metrics["mOKS"] - 1.0) < 0.01

    def test_dataset_metrics_keys(self):
        ev   = Evaluator()
        kps  = np.random.rand(17, 2).astype(np.float32) * 200
        vis  = np.ones(17, dtype=np.float32)
        kps3 = np.concatenate([kps, vis[:, None]], axis=1)
        person = {"keypoints": kps3, "box": [0., 0., 300., 300.]}
        metrics = ev.compute_dataset_metrics([[person]], [[person]])
        for key in ("pck", "mOKS", "per_keypoint_pck"):
            assert key in metrics


# ── Visualizer ────────────────────────────────────────────────────────────────

class TestVisualizer:
    def test_repr(self, tmp_path):
        v = Visualizer(output_dir=str(tmp_path))
        assert "Visualizer" in repr(v)

    def test_from_config(self, tmp_path):
        v = Visualizer.from_config({"log_dir": str(tmp_path)})
        assert v.output_dir == str(tmp_path)

    def test_draw_poses_output_shape(self, tmp_path):
        import cv2
        v = Visualizer(output_dir=str(tmp_path), kp_conf_threshold=0.3)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        kps = make_person_kps({"left_shoulder": (100, 200), "right_shoulder": (200, 200),
                                "left_hip": (110, 350), "right_hip": (190, 350)})
        persons = [{"box": [50., 100., 300., 450.], "confidence": 0.9,
                    "keypoints": kps, "keypoint_names": COCO_KEYPOINT_NAMES}]
        out = v.draw_poses(img, persons, actions=["standing"])
        assert out.shape == img.shape

    def test_save_result_creates_file(self, tmp_path):
        v = Visualizer(output_dir=str(tmp_path))
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        persons = [{"box": [10., 10., 100., 100.], "confidence": 0.8,
                    "keypoints": kps, "keypoint_names": COCO_KEYPOINT_NAMES}]
        path = v.save_result(img, persons, "pose_test.jpg", actions=["unknown"])
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_plot_per_keypoint_pck_creates_file(self, tmp_path):
        v    = Visualizer(output_dir=str(tmp_path))
        path = v.plot_per_keypoint_pck([0.8] * 17, COCO_KEYPOINT_NAMES,
                                        filename="kp_pck_test.png")
        assert os.path.isfile(path)

    def test_plot_action_distribution_creates_file(self, tmp_path):
        v    = Visualizer(output_dir=str(tmp_path))
        path = v.plot_action_distribution(
            ["standing", "arms_raised", "standing", "unknown"],
            filename="action_dist_test.png"
        )
        assert os.path.isfile(path)

    def test_plot_keypoint_heatmap_creates_file(self, tmp_path):
        v     = Visualizer(output_dir=str(tmp_path), kp_conf_threshold=0.0)
        kps   = np.random.rand(17, 3).astype(np.float32)
        kps[:, 0] *= 640; kps[:, 1] *= 480; kps[:, 2] = 1.0
        persons = [{"keypoints": kps}]
        path = v.plot_keypoint_heatmap(persons, (480, 640),
                                        filename="heatmap_test.png")
        assert os.path.isfile(path)
