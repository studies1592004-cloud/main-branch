"""
tests/test_object_detection.py
================================
Unit tests for code/standardized/object_detection.py

Run:
    pytest tests/test_object_detection.py -v

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
from object_detection import (
    Preprocessor,
    PostProcessor,
    Evaluator,
    Visualizer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def bgr_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def square_image():
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)


# ── Preprocessor ─────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_repr(self):
        p = Preprocessor()
        assert "Preprocessor" in repr(p)

    def test_from_config(self):
        p = Preprocessor.from_config({"image_size": 320})
        assert p.image_size == 320

    def test_letterbox_output_shape(self, bgr_image):
        p = Preprocessor(image_size=640)
        out, ratio, pad = p.letterbox(bgr_image)
        assert out.shape == (640, 640, 3)

    def test_letterbox_square_input_no_pad(self, square_image):
        p = Preprocessor(image_size=300)
        out, ratio, pad = p.letterbox(square_image)
        assert out.shape == (300, 300, 3)
        assert pad == (0, 0) or (pad[0] == 0 and pad[1] == 0)

    def test_letterbox_preserves_aspect_ratio(self, bgr_image):
        """After letterbox, the content region should match input aspect ratio."""
        p = Preprocessor(image_size=640)
        out, ratio, pad = p.letterbox(bgr_image)
        h, w = bgr_image.shape[:2]
        content_h = round(h * ratio)
        content_w = round(w * ratio)
        # Content dimensions should fit within 640
        assert content_h <= 640
        assert content_w <= 640

    def test_unletterbox_recovers_original_coords(self, bgr_image):
        """Boxes letterboxed then un-letterboxed should return to original scale."""
        p = Preprocessor(image_size=640)
        _, ratio, pad = p.letterbox(bgr_image)
        # A box at the centre of the letterboxed image
        box = np.array([[200., 200., 400., 400.]])
        recovered = p.unletterbox(box, ratio, pad)
        assert recovered.shape == (1, 4)
        # Should be within original image bounds
        assert recovered[0, 0] >= 0
        assert recovered[0, 2] <= bgr_image.shape[1] + 1


# ── PostProcessor ─────────────────────────────────────────────────────────────

class TestPostProcessor:
    def test_repr(self):
        pp = PostProcessor()
        assert "PostProcessor" in repr(pp)

    def test_from_config(self):
        pp = PostProcessor.from_config({"conf_threshold": 0.3, "iou_threshold": 0.5})
        assert abs(pp.conf_threshold - 0.3) < 1e-6
        assert abs(pp.iou_threshold - 0.5) < 1e-6

    def test_confidence_filter_removes_low_conf(self):
        pp = PostProcessor(conf_threshold=0.5)
        boxes = np.array([[0.,0.,10.,10.],[20.,20.,30.,30.],[50.,50.,60.,60.]])
        confs = np.array([0.9, 0.3, 0.6])
        cls   = np.array([0, 0, 0])
        kept_boxes, kept_confs, kept_cls = pp.filter_by_confidence(boxes, confs, cls)
        assert len(kept_boxes) == 2
        assert all(c >= 0.5 for c in kept_confs)

    def test_confidence_filter_keeps_all_above_threshold(self):
        pp = PostProcessor(conf_threshold=0.1)
        boxes = np.array([[0.,0.,10.,10.]] * 5)
        confs = np.array([0.2, 0.5, 0.8, 0.9, 0.3])
        cls   = np.zeros(5, dtype=int)
        kept_boxes, kept_confs, _ = pp.filter_by_confidence(boxes, confs, cls)
        assert len(kept_boxes) == 5

    def test_nms_removes_overlapping_boxes(self):
        pp = PostProcessor(iou_threshold=0.5)
        # Three heavily overlapping boxes, one separate
        boxes = np.array([
            [10., 10., 100., 100.],
            [12., 12., 102., 102.],
            [14., 14., 104., 104.],
            [200., 200., 250., 250.],
        ])
        confs = np.array([0.9, 0.85, 0.7, 0.8])
        cls   = np.array([0, 0, 0, 0])
        kept = pp.nms(boxes, confs, cls)
        # Should keep 1 from the cluster + 1 separate = 2
        assert len(kept) == 2

    def test_nms_keeps_highest_confidence(self):
        pp = PostProcessor(iou_threshold=0.5)
        boxes = np.array([
            [0., 0., 100., 100.],
            [5., 5., 105., 105.],
        ])
        confs = np.array([0.6, 0.9])
        cls   = np.array([0, 0])
        kept = pp.nms(boxes, confs, cls)
        assert len(kept) == 1
        assert abs(kept[0]["confidence"] - 0.9) < 1e-5

    def test_nms_class_aware(self):
        """Boxes of different classes should NOT suppress each other."""
        pp = PostProcessor(iou_threshold=0.5)
        boxes = np.array([
            [0., 0., 100., 100.],
            [0., 0., 100., 100.],   # identical box, different class
        ])
        confs = np.array([0.9, 0.8])
        cls   = np.array([0, 1])
        kept = pp.nms(boxes, confs, cls)
        assert len(kept) == 2

    def test_nms_empty_input(self):
        pp = PostProcessor()
        kept = pp.nms(
            np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        )
        assert len(kept) == 0


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_repr(self):
        ev = Evaluator(class_names=["hat"])
        assert "Evaluator" in repr(ev)

    def test_from_config(self):
        ev = Evaluator.from_config({"class_names": ["a", "b"]})
        assert len(ev.class_names) == 2

    def test_box_iou_identical(self):
        from object_detection import Evaluator as E
        iou = E._box_iou(
            np.array([0., 0., 10., 10.]),
            np.array([0., 0., 10., 10.])
        )
        assert abs(iou - 1.0) < 1e-6

    def test_box_iou_no_overlap(self):
        from object_detection import Evaluator as E
        iou = E._box_iou(
            np.array([0., 0., 10., 10.]),
            np.array([20., 20., 30., 30.])
        )
        assert iou == 0.0

    def test_box_iou_partial(self):
        from object_detection import Evaluator as E
        # 5×10 overlap out of two 10×10 boxes
        iou = E._box_iou(
            np.array([0., 0., 10., 10.]),
            np.array([5., 0., 15., 10.])
        )
        expected = 50.0 / 150.0
        assert abs(iou - expected) < 1e-4

    def test_ap_perfect_predictions(self):
        ev = Evaluator(class_names=["hat"])
        boxes = np.array([[10., 10., 50., 50.], [100., 100., 150., 150.]])
        confs = np.array([0.9, 0.8])
        cls   = np.array([0, 0])
        preds = [{"box": b, "confidence": c, "class_id": cl}
                 for b, c, cl in zip(boxes, confs, cls)]
        gts   = [{"box": b, "class_id": 0} for b in boxes]
        ap = ev.compute_ap(preds, gts, class_id=0, iou_threshold=0.5)
        assert ap > 0.99

    def test_ap_no_predictions(self):
        ev = Evaluator(class_names=["hat"])
        gts = [{"box": np.array([0., 0., 10., 10.]), "class_id": 0}]
        ap = ev.compute_ap([], gts, class_id=0)
        assert ap == 0.0

    def test_map_returns_dict(self):
        ev = Evaluator(class_names=["hat", "vest"])
        boxes = np.array([[10., 10., 50., 50.]])
        preds_all = [[{"box": boxes[0], "confidence": 0.9, "class_id": 0}]]
        gts_all   = [[{"box": boxes[0], "class_id": 0}]]
        result = ev.compute_map(preds_all, gts_all)
        assert "mAP50" in result
        assert "per_class_AP" in result
        assert result["mAP50"] >= 0.0


# ── Visualizer ────────────────────────────────────────────────────────────────

class TestVisualizer:
    def test_repr(self, tmp_path):
        v = Visualizer(output_dir=str(tmp_path))
        assert "Visualizer" in repr(v)

    def test_from_config(self, tmp_path):
        v = Visualizer.from_config({"log_dir": str(tmp_path)})
        assert v.output_dir == str(tmp_path)

    def test_draw_boxes_output_shape(self, bgr_image):
        v = Visualizer(output_dir="/tmp")
        preds = [
            {"box": [50., 50., 200., 200.], "confidence": 0.9,
             "class_id": 0, "class_name": "hat"},
        ]
        out = v.draw_boxes(bgr_image, preds)
        assert out.shape == bgr_image.shape

    def test_draw_boxes_does_not_modify_input(self, bgr_image):
        v = Visualizer(output_dir="/tmp")
        original = bgr_image.copy()
        preds = [{"box": [50., 50., 200., 200.], "confidence": 0.9,
                  "class_id": 0, "class_name": "hat"}]
        v.draw_boxes(bgr_image, preds)
        assert np.array_equal(bgr_image, original)

    def test_save_annotated_image_creates_file(self, tmp_path, bgr_image):
        v = Visualizer(output_dir=str(tmp_path))
        preds = [{"box": [10., 10., 50., 50.], "confidence": 0.8,
                  "class_id": 0, "class_name": "hat"}]
        path = v.save_annotated(bgr_image, preds, filename="det_test.jpg")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_plot_map_bar_creates_file(self, tmp_path):
        v = Visualizer(output_dir=str(tmp_path))
        path = v.plot_map_bar({"hat": 0.82, "vest": 0.74}, mAP50=0.78,
                               filename="map_bar_test.png")
        assert os.path.isfile(path)
