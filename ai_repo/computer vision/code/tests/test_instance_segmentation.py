"""
tests/test_instance_segmentation.py
=====================================
Unit tests for code/standardized/instance_segmentation.py

Run:
    pytest tests/test_instance_segmentation.py -v

Dependencies:
    pip install numpy opencv-python pytest
    (No Roboflow API key or GPU needed.)
"""

import numpy as np
import pytest
import sys
import os
import types

# ── Mock ultralytics + roboflow ───────────────────────────────────────────────
for mod in ("ultralytics", "roboflow"):
    sys.modules[mod] = types.ModuleType(mod)
sys.modules["ultralytics"].YOLO = object

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "standardized"))
from instance_segmentation import (
    MaskProcessor,
    Evaluator,
    Visualizer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def square_mask():
    """64×64 binary mask with a 40×40 square in the centre."""
    m = np.zeros((64, 64), dtype=np.uint8)
    m[12:52, 12:52] = 1
    return m

@pytest.fixture
def small_mask():
    """64×64 binary mask with a 10×10 square in a corner."""
    m = np.zeros((64, 64), dtype=np.uint8)
    m[2:12, 2:12] = 1
    return m

@pytest.fixture
def bgr_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)


# ── MaskProcessor ─────────────────────────────────────────────────────────────

class TestMaskProcessor:
    def test_repr(self):
        mp = MaskProcessor()
        assert "MaskProcessor" in repr(mp)

    def test_from_config(self):
        mp = MaskProcessor.from_config({"mask_threshold": 0.6})
        assert abs(mp.mask_threshold - 0.6) < 1e-6

    # binarise
    def test_binarise_above_threshold(self):
        mp  = MaskProcessor(mask_threshold=0.5)
        soft = np.array([[0.3, 0.6], [0.8, 0.1]], dtype=np.float32)
        out  = mp.binarise(soft)
        expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        assert np.array_equal(out, expected)

    def test_binarise_output_dtype(self):
        mp   = MaskProcessor()
        soft = np.random.rand(32, 32).astype(np.float32)
        out  = mp.binarise(soft)
        assert out.dtype == np.uint8

    def test_binarise_threshold_boundary(self):
        """Value exactly at threshold should be included (>=)."""
        mp  = MaskProcessor(mask_threshold=0.5)
        soft = np.array([[0.5]])
        out  = mp.binarise(soft)
        assert out[0, 0] == 1

    # polygon_to_mask
    def test_polygon_to_mask_shape(self):
        mp   = MaskProcessor()
        poly = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        mask = mp.polygon_to_mask(poly, height=64, width=64)
        assert mask.shape == (64, 64)

    def test_polygon_to_mask_filled(self):
        mp   = MaskProcessor()
        poly = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        mask = mp.polygon_to_mask(poly, height=64, width=64)
        assert mask.sum() > 0

    def test_polygon_to_mask_stays_in_bounds(self):
        mp   = MaskProcessor()
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mask = mp.polygon_to_mask(poly, height=32, width=32)
        assert mask.shape == (32, 32)
        assert mask.max() <= 1

    # mask_to_polygon
    def test_mask_to_polygon_returns_ndarray(self, square_mask):
        mp   = MaskProcessor()
        poly = mp.mask_to_polygon(square_mask)
        assert poly is not None
        assert isinstance(poly, np.ndarray)

    def test_mask_to_polygon_two_columns(self, square_mask):
        mp   = MaskProcessor()
        poly = mp.mask_to_polygon(square_mask)
        assert poly.shape[1] == 2

    def test_mask_to_polygon_normalised(self, square_mask):
        mp   = MaskProcessor()
        poly = mp.mask_to_polygon(square_mask)
        assert poly.max() <= 1.0 + 1e-5
        assert poly.min() >= 0.0 - 1e-5

    def test_empty_mask_returns_none(self):
        mp    = MaskProcessor()
        empty = np.zeros((64, 64), dtype=np.uint8)
        assert mp.mask_to_polygon(empty) is None

    # mask_iou
    def test_iou_identical_masks(self, square_mask):
        mp  = MaskProcessor()
        iou = mp.mask_iou(square_mask, square_mask)
        assert abs(iou - 1.0) < 1e-5

    def test_iou_no_overlap(self, square_mask, small_mask):
        mp = MaskProcessor()
        a  = np.zeros((64, 64), dtype=np.uint8); a[0:10, 0:10] = 1
        b  = np.zeros((64, 64), dtype=np.uint8); b[54:64, 54:64] = 1
        iou = mp.mask_iou(a, b)
        assert abs(iou) < 1e-5

    def test_iou_partial_overlap(self):
        mp = MaskProcessor()
        a  = np.zeros((10, 10), dtype=np.uint8); a[:, :5] = 1   # left half
        b  = np.zeros((10, 10), dtype=np.uint8); b[:, 5:] = 1   # right half
        iou = mp.mask_iou(a, b)
        assert abs(iou) < 0.01   # no overlap

    def test_iou_range(self, square_mask, small_mask):
        mp  = MaskProcessor()
        iou = mp.mask_iou(square_mask, small_mask)
        assert 0.0 <= iou <= 1.0

    def test_iou_handles_different_shapes(self):
        mp = MaskProcessor()
        a  = np.ones((32, 32), dtype=np.uint8)
        b  = np.ones((64, 64), dtype=np.uint8)
        iou = mp.mask_iou(a, b)
        assert 0.0 <= iou <= 1.0

    # compute_mask_ap
    def test_ap_perfect(self, square_mask, small_mask):
        mp  = MaskProcessor()
        ap  = mp.compute_mask_ap([square_mask, small_mask],
                                  [0.9, 0.8],
                                  [square_mask, small_mask],
                                  iou_threshold=0.5)
        assert abs(ap - 1.0) < 0.01

    def test_ap_no_predictions(self, square_mask):
        mp = MaskProcessor()
        ap = mp.compute_mask_ap([], [], [square_mask], iou_threshold=0.5)
        assert ap == 0.0

    def test_ap_no_gt(self):
        mp = MaskProcessor()
        ap = mp.compute_mask_ap([], [], [], iou_threshold=0.5)
        assert ap == 1.0   # vacuously perfect

    def test_ap_higher_threshold_lower_or_equal(self, square_mask):
        """Stricter IoU threshold should give ≤ AP."""
        mp  = MaskProcessor()
        pred = [square_mask]
        pred_scores = [0.9]
        gt   = [square_mask]
        ap50  = mp.compute_mask_ap(pred, pred_scores, gt, iou_threshold=0.50)
        ap75  = mp.compute_mask_ap(pred, pred_scores, gt, iou_threshold=0.75)
        assert ap75 <= ap50 + 1e-6

    # instance_stats
    def test_stats_keys(self, square_mask):
        mp    = MaskProcessor()
        stats = mp.instance_stats(square_mask)
        for key in ("area", "bbox_xyxy", "centroid", "equiv_diameter"):
            assert key in stats

    def test_stats_area_correct(self, square_mask):
        mp    = MaskProcessor()
        stats = mp.instance_stats(square_mask)
        assert stats["area"] == int(square_mask.sum())

    def test_stats_empty_mask(self):
        mp    = MaskProcessor()
        empty = np.zeros((64, 64), dtype=np.uint8)
        stats = mp.instance_stats(empty)
        assert stats["area"] == 0

    def test_stats_equiv_diameter_positive(self, square_mask):
        mp    = MaskProcessor()
        stats = mp.instance_stats(square_mask)
        assert stats["equiv_diameter"] > 0


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_repr(self):
        ev = Evaluator(class_names=["tumor"])
        assert "Evaluator" in repr(ev)

    def test_from_config(self):
        ev = Evaluator.from_config({"class_names": ["a", "b", "c"]})
        assert len(ev.class_names) == 3

    def test_evaluate_perfect_predictions(self, square_mask, small_mask):
        ev = Evaluator(class_names=["tumor"])
        pred_insts = [[
            {"mask": square_mask, "confidence": 0.9, "class_id": 0},
            {"mask": small_mask,  "confidence": 0.8, "class_id": 0},
        ]]
        gt_insts = [[
            {"mask": square_mask, "class_id": 0},
            {"mask": small_mask,  "class_id": 0},
        ]]
        result = ev.evaluate(pred_insts, gt_insts)
        assert result["mask_AP50"] > 0.9
        assert "per_class_AP50" in result

    def test_evaluate_returns_required_keys(self, square_mask):
        ev = Evaluator(class_names=["tumor"])
        pred = [[{"mask": square_mask, "confidence": 0.9, "class_id": 0}]]
        gt   = [[{"mask": square_mask, "class_id": 0}]]
        result = ev.evaluate(pred, gt)
        for key in ("mask_AP50", "mask_AP50_95", "per_class_AP50"):
            assert key in result


# ── Visualizer ────────────────────────────────────────────────────────────────

class TestVisualizer:
    def test_repr(self, tmp_path):
        v = Visualizer(output_dir=str(tmp_path))
        assert "Visualizer" in repr(v)

    def test_from_config(self, tmp_path):
        v = Visualizer.from_config({"log_dir": str(tmp_path)})
        assert v.output_dir == str(tmp_path)

    def test_draw_instances_output_shape(self, bgr_image, square_mask):
        v = Visualizer(output_dir="/tmp")
        instances = [
            {"box": [12., 12., 52., 52.], "confidence": 0.9,
             "class_id": 0, "class_name": "tumor", "mask": square_mask, "mask_stats": {}},
        ]
        out = v.draw_instances(bgr_image, instances, {0: "tumor"})
        assert out.shape == bgr_image.shape

    def test_draw_instances_no_mask(self, bgr_image):
        """Should not crash when mask is None."""
        v = Visualizer(output_dir="/tmp")
        instances = [
            {"box": [10., 10., 50., 50.], "confidence": 0.8,
             "class_id": 0, "class_name": "tumor", "mask": None, "mask_stats": {}},
        ]
        out = v.draw_instances(bgr_image, instances)
        assert out.shape == bgr_image.shape

    def test_save_result_creates_file(self, tmp_path, bgr_image, square_mask):
        v = Visualizer(output_dir=str(tmp_path))
        instances = [
            {"box": [12., 12., 52., 52.], "confidence": 0.9,
             "class_id": 0, "class_name": "tumor",
             "mask": square_mask, "mask_stats": {"area": 1600}},
        ]
        path = v.save_result(bgr_image, instances, {0: "tumor"}, "inst_test.jpg")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_plot_ap_bar_creates_file(self, tmp_path):
        v    = Visualizer(output_dir=str(tmp_path))
        path = v.plot_ap_bar({"tumor": 0.82}, map50=0.82, filename="ap_bar_test.png")
        assert os.path.isfile(path)

    def test_plot_size_distribution_creates_file(self, tmp_path, square_mask):
        v = Visualizer(output_dir=str(tmp_path))
        instances_by_file = {
            "img1.jpg": [{"mask_stats": {"area": 1600}}],
            "img2.jpg": [{"mask_stats": {"area": 400}}, {"mask_stats": {"area": 800}}],
        }
        path = v.plot_instance_size_distribution(instances_by_file,
                                                  filename="size_dist_test.png")
        assert os.path.isfile(path)
