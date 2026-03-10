"""
tests/test_semantic_segmentation.py
=====================================
Unit tests for code/standardized/semantic_segmentation.py

Run:
    pytest tests/test_semantic_segmentation.py -v

Dependencies:
    pip install torch torchvision numpy pytest
    (No Roboflow API key or GPU needed.)
"""

import numpy as np
import pytest
import sys
import os
import types

# ── Mock roboflow + transformers ──────────────────────────────────────────────
for mod in ("roboflow", "transformers"):
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "standardized"))

import torch
import torch.nn as nn
from semantic_segmentation import (
    SegmentationLoss,
    MetricsCalculator,
    Visualizer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def num_classes():
    return 4

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def hw():
    """Height and width for test tensors."""
    return 32, 32


# ── SegmentationLoss ──────────────────────────────────────────────────────────

class TestSegmentationLoss:
    def test_repr(self):
        loss = SegmentationLoss(num_classes=4)
        assert "SegmentationLoss" in repr(loss)

    def test_from_config(self):
        loss = SegmentationLoss.from_config({"num_classes": 5, "dice_weight": 0.6})
        assert loss.num_classes == 5

    def test_forward_returns_scalar(self, num_classes, batch_size, hw):
        H, W = hw
        loss_fn = SegmentationLoss(num_classes=num_classes)
        logits = torch.randn(batch_size, num_classes, H, W)
        target = torch.randint(0, num_classes, (batch_size, H, W))
        loss_val = loss_fn(logits, target)
        assert loss_val.ndim == 0   # scalar
        assert not torch.isnan(loss_val)

    def test_loss_decreases_toward_perfect(self, num_classes, hw):
        """Loss should be lower when predictions are nearly perfect."""
        H, W = hw
        loss_fn = SegmentationLoss(num_classes=num_classes)
        target  = torch.zeros(1, H, W, dtype=torch.long)

        # High-confidence correct predictions
        logits_good = torch.zeros(1, num_classes, H, W)
        logits_good[:, 0, :, :] = 10.0   # class 0 dominates

        # Wrong predictions
        logits_bad = torch.zeros(1, num_classes, H, W)
        logits_bad[:, 1, :, :] = 10.0    # class 1 dominates (wrong)

        loss_good = loss_fn(logits_good, target).item()
        loss_bad  = loss_fn(logits_bad,  target).item()
        assert loss_good < loss_bad

    def test_ce_component_present(self, num_classes, hw):
        """Cross-entropy component alone should be non-negative."""
        H, W = hw
        loss_fn = SegmentationLoss(num_classes=num_classes, dice_weight=0.0)
        logits  = torch.randn(1, num_classes, H, W)
        target  = torch.randint(0, num_classes, (1, H, W))
        loss_val = loss_fn(logits, target)
        assert loss_val.item() >= 0.0

    def test_ignore_index_excluded(self, num_classes, hw):
        """Pixels with ignore_index label should not affect loss."""
        H, W = hw
        loss_fn = SegmentationLoss(num_classes=num_classes, ignore_index=255)
        logits  = torch.randn(1, num_classes, H, W)
        target  = torch.full((1, H, W), 255, dtype=torch.long)
        # All pixels ignored — loss should be zero or very small
        loss_val = loss_fn(logits, target)
        assert loss_val.item() < 1e-3 or torch.isnan(loss_val)  # implementation may return 0 or nan for all-ignored


# ── MetricsCalculator ─────────────────────────────────────────────────────────

class TestMetricsCalculator:
    def test_repr(self):
        mc = MetricsCalculator(num_classes=4)
        assert "MetricsCalculator" in repr(mc)

    def test_from_config(self):
        mc = MetricsCalculator.from_config({"num_classes": 6})
        assert mc.num_classes == 6

    def test_perfect_predictions_pixel_accuracy(self, num_classes, hw):
        H, W = hw
        mc   = MetricsCalculator(num_classes=num_classes)
        pred = torch.randint(0, num_classes, (1, H, W))
        mc.update(pred, pred)
        metrics = mc.compute()
        assert abs(metrics["pixel_accuracy"] - 1.0) < 1e-5

    def test_perfect_predictions_miou(self, num_classes, hw):
        H, W = hw
        mc   = MetricsCalculator(num_classes=num_classes)
        pred = torch.randint(0, num_classes, (1, H, W))
        mc.update(pred, pred)
        metrics = mc.compute()
        assert abs(metrics["mIoU"] - 1.0) < 1e-5

    def test_wrong_predictions_lower_accuracy(self, num_classes, hw):
        H, W  = hw
        mc    = MetricsCalculator(num_classes=num_classes)
        pred  = torch.zeros(1, H, W, dtype=torch.long)       # all class 0
        gt    = torch.ones(1, H, W, dtype=torch.long)         # all class 1
        mc.update(pred, gt)
        metrics = mc.compute()
        assert metrics["pixel_accuracy"] < 0.01

    def test_reset_clears_state(self, num_classes, hw):
        H, W = hw
        mc   = MetricsCalculator(num_classes=num_classes)
        pred = torch.randint(0, num_classes, (1, H, W))
        mc.update(pred, pred)
        mc.reset()
        # After reset, confusion matrix should be all zeros
        assert mc.confusion_matrix.sum() == 0

    def test_metrics_dict_has_required_keys(self, num_classes, hw):
        H, W = hw
        mc   = MetricsCalculator(num_classes=num_classes)
        pred = torch.randint(0, num_classes, (1, H, W))
        mc.update(pred, pred)
        metrics = mc.compute()
        for key in ("pixel_accuracy", "mIoU", "mean_dice"):
            assert key in metrics, f"Missing key: {key}"

    def test_per_class_iou_length(self, num_classes, hw):
        H, W = hw
        mc   = MetricsCalculator(num_classes=num_classes)
        pred = torch.randint(0, num_classes, (1, H, W))
        mc.update(pred, pred)
        metrics = mc.compute()
        assert len(metrics["per_class_iou"]) == num_classes

    def test_ignore_index_not_counted(self, hw):
        H, W = hw
        mc   = MetricsCalculator(num_classes=3, ignore_index=255)
        pred = torch.zeros(1, H, W, dtype=torch.long)
        gt   = torch.full((1, H, W), 255, dtype=torch.long)   # all ignored
        mc.update(pred, gt)
        metrics = mc.compute()
        # With all pixels ignored, IoU denominator is 0 → should handle gracefully
        assert not np.isnan(metrics.get("mIoU", 0.0))


# ── Visualizer ────────────────────────────────────────────────────────────────

class TestVisualizer:
    def test_repr(self, tmp_path):
        v = Visualizer(output_dir=str(tmp_path))
        assert "Visualizer" in repr(v)

    def test_from_config(self, tmp_path):
        v = Visualizer.from_config({"log_dir": str(tmp_path)})
        assert v.output_dir == str(tmp_path)

    def test_colour_mask_output_shape(self, num_classes):
        v    = Visualizer(output_dir="/tmp")
        mask = np.random.randint(0, num_classes, (64, 64), dtype=np.int32)
        colored = v.colour_mask(mask, num_classes=num_classes)
        assert colored.shape == (64, 64, 3)
        assert colored.dtype == np.uint8

    def test_overlay_output_shape(self, num_classes):
        v     = Visualizer(output_dir="/tmp")
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask  = np.random.randint(0, num_classes, (64, 64), dtype=np.int32)
        out   = v.overlay_mask(image, mask, num_classes=num_classes, alpha=0.5)
        assert out.shape == (64, 64, 3)

    def test_save_prediction_grid_creates_file(self, tmp_path, num_classes):
        v      = Visualizer(output_dir=str(tmp_path))
        image  = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        gt     = np.random.randint(0, num_classes, (64, 64), dtype=np.int32)
        pred   = np.random.randint(0, num_classes, (64, 64), dtype=np.int32)
        path   = v.save_prediction_grid(image, gt, pred,
                                        num_classes=num_classes,
                                        filename="seg_grid_test.png")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_plot_iou_bar_creates_file(self, tmp_path, num_classes):
        v        = Visualizer(output_dir=str(tmp_path))
        iou_vals = {f"class_{i}": float(i) / num_classes for i in range(num_classes)}
        path     = v.plot_iou_bar(iou_vals, miou=0.5, filename="iou_bar_test.png")
        assert os.path.isfile(path)

    def test_plot_training_curves_creates_file(self, tmp_path):
        v = Visualizer(output_dir=str(tmp_path))
        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.4],
            "val_loss":   [1.1, 0.9, 0.7, 0.5],
            "train_miou": [0.2, 0.4, 0.6, 0.7],
            "val_miou":   [0.18, 0.38, 0.55, 0.65],
        }
        path = v.plot_training_curves(history, filename="seg_curves_test.png")
        assert os.path.isfile(path)
