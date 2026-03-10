from __future__ import annotations

"""
semantic_segmentation.py
========================
Industry-Standard Semantic Segmentation Pipeline — DeepLabV3+ / SegFormer

Installation
------------
    pip install torch torchvision roboflow opencv-python matplotlib \
                numpy scikit-learn seaborn tqdm transformers

Usage
-----
    # Download dataset and train (DeepLabV3+ default)
    python semantic_segmentation.py

    # Use SegFormer backbone
    python semantic_segmentation.py --backbone segformer

    # Evaluate a saved checkpoint
    python semantic_segmentation.py --mode evaluate --checkpoint ./checkpoints/best.pt

    # Run inference on an image
    python semantic_segmentation.py --mode infer --source path/to/image.jpg \
                                    --checkpoint ./checkpoints/best.pt

Author: CV Course — Section 4
Python: 3.9+  |  PyTorch 2.x
"""

# ============================================================
# GLOBAL CONFIGURATION — edit these before running
# ============================================================
ROBOFLOW_API_KEY: str   = "YOUR_API_KEY_HERE"   # free key at roboflow.com
ROBOFLOW_WORKSPACE: str = "alex-hyams-cosqx"
ROBOFLOW_PROJECT: str   = "sidewalk-semantic"
ROBOFLOW_VERSION: int   = 1

BACKBONE: str           = "deeplabv3plus"  # "deeplabv3plus" or "segformer"
IMAGE_SIZE: int         = 512
BATCH_SIZE: int         = 8
NUM_EPOCHS: int         = 50
LEARNING_RATE: float    = 6e-5
EARLY_STOP_PATIENCE: int = 8
DEVICE: str             = "cuda"           # overridden at runtime
SEED: int               = 42
NUM_WORKERS: int        = 4
CHECKPOINT_DIR: str     = "./checkpoints"
LOG_DIR: str            = "./logs/segmentation"

# ImageNet normalisation
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

# Ignore index for pixels with no valid label (e.g. boundary/void regions)
IGNORE_INDEX: int = 255

# ============================================================
# IMPORTS
# ============================================================
import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models.segmentation import (
        deeplabv3_resnet101,
        DeepLabV3_ResNet101_Weights,
    )
except ImportError as exc:
    sys.exit(f"PyTorch/torchvision import failed: {exc}\nRun: pip install torch torchvision")

try:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
except ImportError as exc:
    sys.exit(f"scikit-learn import failed: {exc}\nRun: pip install scikit-learn")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):   # type: ignore[misc]
        return iterable

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Segmentation")


# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed: int = SEED) -> None:
    """Fix random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", dev)
    return dev


def ensure_dir(path: str) -> None:
    """Create directory tree if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS: DatasetDownloader
# ============================================================
class DatasetDownloader:
    """Download a Roboflow semantic segmentation dataset.

    Expected layout after download (semantic segmentation format):
        <root>/train/images/   ← RGB images
        <root>/train/masks/    ← single-channel PNG masks (class index per pixel)
        <root>/valid/images/
        <root>/valid/masks/
        <root>/test/images/
        <root>/test/masks/

    Args:
        api_key:   Roboflow API key.
        workspace: Roboflow workspace slug.
        project:   Project slug.
        version:   Dataset version number.
        dest_dir:  Local destination directory.

    Raises:
        RuntimeError: API key is placeholder or download fails.
        ImportError:  ``roboflow`` package not installed.
    """

    def __init__(
        self,
        api_key: str = ROBOFLOW_API_KEY,
        workspace: str = ROBOFLOW_WORKSPACE,
        project: str = ROBOFLOW_PROJECT,
        version: int = ROBOFLOW_VERSION,
        dest_dir: str = "./data/segmentation",
    ) -> None:
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self.dest_dir = dest_dir
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"DatasetDownloader(project='{self.project}', "
            f"version={self.version})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "DatasetDownloader":
        """Construct from a flat config dictionary."""
        return cls(
            api_key=config.get("api_key", ROBOFLOW_API_KEY),
            workspace=config.get("workspace", ROBOFLOW_WORKSPACE),
            project=config.get("project", ROBOFLOW_PROJECT),
            version=config.get("version", ROBOFLOW_VERSION),
            dest_dir=config.get("dest_dir", "./data/segmentation"),
        )

    def download(self) -> str:
        """Download dataset and return local root path.

        Returns:
            Path to the dataset root folder.

        Raises:
            RuntimeError: API key is placeholder or download fails.
            ImportError:  ``roboflow`` not installed.
        """
        if self.api_key == "YOUR_API_KEY_HERE":
            raise RuntimeError(
                "Please set a valid Roboflow API key.\n"
                "  1. Sign up free at https://roboflow.com\n"
                "  2. Copy your key from Account → Roboflow API\n"
                "  3. Set ROBOFLOW_API_KEY at the top of this file."
            )

        try:
            from roboflow import Roboflow
        except ImportError as exc:
            raise ImportError(
                f"roboflow package not found: {exc}\nRun: pip install roboflow"
            ) from exc

        self._logger.info(
            "Downloading %s/%s v%d …", self.workspace, self.project, self.version
        )
        try:
            rf = Roboflow(api_key=self.api_key)
            dataset = (
                rf.workspace(self.workspace)
                .project(self.project)
                .version(self.version)
                .download("semantic-seg", location=self.dest_dir)
            )
            root = dataset.location
        except Exception as exc:
            raise RuntimeError(
                f"Roboflow download failed: {exc}\n"
                "Check API key, workspace/project slugs, and internet connection."
            ) from exc

        self._logger.info("Dataset ready at: %s", root)
        return root


# ============================================================
# CLASS: SegmentationDataset
# ============================================================
class SegmentationDataset(Dataset):
    """Load image–mask pairs for semantic segmentation training.

    Scans ``<root>/<split>/images/`` for images and matches each to a
    corresponding mask in ``<root>/<split>/masks/`` by stem name.

    Mask format:
        Single-channel PNG where each pixel value is the class index.
        Pixels with value ``IGNORE_INDEX`` (255) are excluded from loss.

    Augmentation strategy (training only):
        - Random horizontal flip (p=0.5)
        - Random crop to ``image_size``
        - Random brightness / contrast jitter
        - Both image and mask are transformed identically so spatial
          alignment is preserved.

    Args:
        root:        Dataset root directory.
        split:       One of ``"train"``, ``"valid"``, ``"test"``.
        image_size:  Spatial size for training crops.
        augment:     Apply random augmentations (True for train split only).
        num_classes: Expected number of semantic classes.

    Raises:
        FileNotFoundError: If the split directory does not exist.
        RuntimeError:      If no image–mask pairs are found.
    """

    SPLIT_ALIASES: dict[str, list[str]] = {
        "valid": ["valid", "val", "validation"],
        "test":  ["test"],
        "train": ["train"],
    }
    IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp"}
    MASK_EXTENSIONS:  set[str] = {".png", ".jpg"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = IMAGE_SIZE,
        augment: bool = True,
        num_classes: int = 2,
    ) -> None:
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.num_classes = num_classes
        self._logger = logging.getLogger(self.__class__.__name__)

        split_dir = self._resolve_split_dir(root, split)
        self.image_dir = os.path.join(split_dir, "images")
        self.mask_dir  = os.path.join(split_dir, "masks")

        for d in (self.image_dir, self.mask_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Expected directory not found: '{d}'\n"
                    "The dataset layout should have images/ and masks/ sub-folders."
                )

        self.samples: list[tuple[str, str]] = self._build_sample_list()

        if not self.samples:
            raise RuntimeError(
                f"No image–mask pairs found in '{split_dir}'.\n"
                "Check that images/ and masks/ directories contain matching filenames."
            )

        # Normalisation transform (applied to image only)
        self._normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        self._logger.info(
            "Split='%s'  samples=%d  augment=%s",
            split, len(self.samples), self.augment,
        )

    def __repr__(self) -> str:
        return (
            f"SegmentationDataset(split='{self.split}', "
            f"n={len(self)}, augment={self.augment})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image_tensor, mask_tensor) for sample at ``idx``.

        Returns:
            image: Float32 tensor ``(3, H, W)``, normalised.
            mask:  Long tensor ``(H, W)``, pixel-wise class indices.
        """
        img_path, mask_path = self.samples[idx]

        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Could not read image: '{img_path}'")
        if mask is None:
            raise RuntimeError(f"Could not read mask: '{mask_path}'")

        img, mask = self._resize(img, mask)

        if self.augment:
            img, mask = self._augment(img, mask)

        # Image → tensor → normalise
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_t = self._normalize(img_t)

        # Mask → long tensor (class indices)
        mask_t = torch.from_numpy(mask.astype(np.int64))

        return img_t, mask_t

    # ── Private helpers ───────────────────────────────────────────────────

    def _resize(
        self, img: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize image and mask to ``(image_size, image_size)``."""
        img  = cv2.resize(img,  (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def _augment(
        self, img: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply identical spatial augmentations to image and mask.

        Augmentations:
            - Random horizontal flip (p=0.5) — applied identically to both.
            - Random brightness / contrast jitter (image only — mask unchanged).

        Args:
            img:  RGB uint8 image.
            mask: Grayscale uint8 mask.

        Returns:
            Augmented (img, mask) pair.
        """
        # Horizontal flip
        if random.random() < 0.5:
            img  = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # Brightness jitter (image only)
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Contrast jitter
        if random.random() < 0.3:
            mean = img.mean()
            factor = random.uniform(0.8, 1.2)
            img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        return img, mask

    @classmethod
    def _resolve_split_dir(cls, root: str, split: str) -> str:
        """Resolve the split directory with alias fallback.

        Args:
            root:  Dataset root.
            split: Requested split.

        Returns:
            Resolved absolute path.

        Raises:
            FileNotFoundError: No matching directory found.
        """
        candidates = cls.SPLIT_ALIASES.get(split, [split])
        for name in candidates:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(
            f"Could not find split '{split}' in '{root}'.\n"
            f"Tried: {candidates}.  Available: {os.listdir(root)}"
        )

    def _build_sample_list(self) -> list[tuple[str, str]]:
        """Pair each image in images/ with its mask in masks/.

        Returns:
            List of (image_path, mask_path) tuples.
        """
        pairs = []
        for fname in sorted(os.listdir(self.image_dir)):
            stem = Path(fname).stem
            ext  = Path(fname).suffix.lower()
            if ext not in self.IMAGE_EXTENSIONS:
                continue

            img_path = os.path.join(self.image_dir, fname)
            mask_path = None
            for mext in self.MASK_EXTENSIONS:
                candidate = os.path.join(self.mask_dir, stem + mext)
                if os.path.isfile(candidate):
                    mask_path = candidate
                    break

            if mask_path:
                pairs.append((img_path, mask_path))
            else:
                self._logger.debug("No mask found for '%s' — skipping.", fname)

        return pairs


# ============================================================
# CLASS: ModelBuilder
# ============================================================
class ModelBuilder:
    """Build a semantic segmentation model with a configurable backbone.

    Supported backbones:
        ``"deeplabv3plus"``:
            DeepLabV3 with ResNet-101 backbone (pretrained on COCO).
            Final classifier replaced to match ``num_classes``.
            ASPP module provides multi-scale context aggregation.
            Produces full-resolution predictions via bilinear upsampling.

        ``"segformer"``:
            SegFormer-B2 from HuggingFace Transformers.
            Hierarchical Transformer encoder + lightweight MLP decoder.
            Requires ``transformers`` package.

    Args:
        num_classes:  Number of semantic classes.
        backbone:     ``"deeplabv3plus"`` or ``"segformer"``.
        pretrained:   Use ImageNet/COCO pretrained weights.

    Raises:
        ValueError: If backbone name is not recognised.
        ImportError: If SegFormer is requested but ``transformers`` not installed.
    """

    VALID_BACKBONES: tuple[str, ...] = ("deeplabv3plus", "segformer")

    def __init__(
        self,
        num_classes: int,
        backbone: str = BACKBONE,
        pretrained: bool = True,
    ) -> None:
        if backbone not in self.VALID_BACKBONES:
            raise ValueError(
                f"backbone='{backbone}' not recognised. "
                f"Choose from {self.VALID_BACKBONES}."
            )
        self.num_classes = num_classes
        self.backbone = backbone
        self.pretrained = pretrained
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"ModelBuilder(num_classes={self.num_classes}, "
            f"backbone='{self.backbone}', pretrained={self.pretrained})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "ModelBuilder":
        """Construct from a flat config dictionary."""
        return cls(
            num_classes=config["num_classes"],
            backbone=config.get("backbone", BACKBONE),
            pretrained=config.get("pretrained", True),
        )

    def build(self) -> nn.Module:
        """Build and return the segmentation model.

        Returns:
            ``nn.Module`` ready for training on the target number of classes.

        Raises:
            ImportError: If SegFormer is requested without ``transformers``.
        """
        if self.backbone == "deeplabv3plus":
            return self._build_deeplabv3()
        else:
            return self._build_segformer()

    def _build_deeplabv3(self) -> nn.Module:
        """Build DeepLabV3+ (ResNet-101 backbone).

        Architecture notes:
            - Atrous Spatial Pyramid Pooling (ASPP): captures multi-scale
              context via parallel dilated convolutions with rates 6, 12, 18.
            - Output stride 16 by default (can be set to 8 for higher res).
            - We replace the final ``classifier[4]`` (1×1 conv that outputs
              num_classes channels) with a new one for our task.

        Returns:
            Modified DeepLabV3 model.
        """
        weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1 if self.pretrained else None
        model = deeplabv3_resnet101(weights=weights)

        # Replace the output classifier
        in_ch = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_ch, self.num_classes, kernel_size=1)

        # Replace aux classifier too (used during training for deep supervision)
        if model.aux_classifier is not None:
            in_ch_aux = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(in_ch_aux, self.num_classes, kernel_size=1)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._logger.info(
            "DeepLabV3+ ResNet-101  pretrained=%s  trainable=%s params",
            self.pretrained, f"{trainable:,}"
        )
        return model

    def _build_segformer(self) -> nn.Module:
        """Build SegFormer-B2 via HuggingFace Transformers.

        Architecture notes:
            - Mix Transformer (MiT-B2) encoder with 4 hierarchical stages.
            - Lightweight All-MLP decoder: simply upsamples + concatenates
              multi-scale features and applies a linear head.
            - No positional encoding — uses depth-wise convolutions for
              positional bias (works well at variable resolutions).

        Returns:
            SegFormer model wrapped in an ``nn.Module`` adapter.

        Raises:
            ImportError: If ``transformers`` is not installed.
        """
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError as exc:
            raise ImportError(
                f"transformers package not found: {exc}\nRun: pip install transformers"
            ) from exc

        config_sf = SegformerConfig.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )
        if self.pretrained:
            model_sf = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512",
                config=config_sf,
                ignore_mismatched_sizes=True,
            )
        else:
            model_sf = SegformerForSemanticSegmentation(config_sf)

        # Wrap to return a dict with 'out' key (matches DeepLab interface)
        class SegFormerWrapper(nn.Module):
            """Adapter: makes SegFormer output compatible with DeepLab interface."""

            def __init__(self, model: nn.Module, target_size: int) -> None:
                super().__init__()
                self.model = model
                self.target_size = target_size

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                out = self.model(pixel_values=x)
                logits = out.logits  # (B, C, H/4, W/4)
                # Upsample to input resolution
                logits = F.interpolate(
                    logits,
                    size=(self.target_size, self.target_size),
                    mode="bilinear",
                    align_corners=False,
                )
                return {"out": logits}

        wrapper = SegFormerWrapper(model_sf, IMAGE_SIZE)
        trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
        self._logger.info(
            "SegFormer-B2  pretrained=%s  trainable=%s params",
            self.pretrained, f"{trainable:,}"
        )
        return wrapper

    def get_optimizer(
        self, model: nn.Module, lr: float = LEARNING_RATE
    ) -> optim.Optimizer:
        """Build an AdamW optimiser with layer-wise learning rates.

        Backbone layers use ``lr × 0.1`` to preserve pretrained features.
        The segmentation head uses the full ``lr``.

        Args:
            model: The model returned by ``build()``.
            lr:    Base learning rate for the decoder / classifier head.

        Returns:
            Configured ``torch.optim.AdamW`` instance.
        """
        # Identify backbone vs head parameters by name
        backbone_keywords = ("backbone", "encoder", "layer")
        backbone_params, head_params = [], []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name.lower() for kw in backbone_keywords):
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * 0.1, "name": "backbone"},
            {"params": head_params,     "lr": lr,       "name": "head"},
        ]
        self._logger.info(
            "AdamW: backbone LR=%.2e (%d params)  head LR=%.2e (%d params)",
            lr * 0.1, len(backbone_params), lr, len(head_params),
        )
        return optim.AdamW(param_groups, weight_decay=1e-4)


# ============================================================
# CLASS: SegmentationLoss
# ============================================================
class SegmentationLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss for semantic segmentation.

    Cross-Entropy loss:
        Standard per-pixel classification loss.
        ``ignore_index`` pixels are excluded.
        Optional ``class_weights`` to handle class imbalance.

    Dice loss:
        Overlap-based loss: 1 - 2|A∩B| / (|A|+|B|).
        Minimising Dice directly optimises the IoU metric.
        Soft Dice is differentiable: uses softmax probabilities
        instead of binary predictions.

    Combined:
        L = α × CE + (1-α) × Dice

    Args:
        num_classes:    Number of semantic classes.
        alpha:          Weight for CE component (0.5 = equal weighting).
        class_weights:  Optional tensor of per-class weights for CE.
        ignore_index:   Class index to ignore in loss computation.
        smooth:         Smoothing term in Dice denominator (avoids div/0).
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=0.05,
        )
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"SegmentationLoss(alpha={self.alpha}, "
            f"num_classes={self.num_classes})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "SegmentationLoss":
        """Construct from config dict."""
        return cls(
            num_classes=config["num_classes"],
            alpha=config.get("alpha", 0.5),
            ignore_index=config.get("ignore_index", IGNORE_INDEX),
        )

    def dice_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute multi-class soft Dice loss.

        Steps:
            1. Convert logits to softmax probabilities: P_{B,C,H,W}.
            2. One-hot encode targets (ignoring the ``ignore_index`` pixels).
            3. For each class c: Dice_c = 2 Σ(P_c · T_c) / (Σ P_c² + Σ T_c²).
            4. Mean over valid classes.

        Args:
            logits:  Raw model output, shape ``(B, C, H, W)``.
            targets: Class index mask, shape ``(B, H, W)``.

        Returns:
            Scalar Dice loss.
        """
        probs = F.softmax(logits, dim=1)   # (B, C, H, W)

        # Create ignore mask
        valid = (targets != self.ignore_index).float()  # (B, H, W)

        # Clamp targets so one_hot doesn't fail on ignore_index=255
        targets_clamped = targets.clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(targets_clamped, self.num_classes).permute(0, 3, 1, 2).float()
        # Zero out ignored pixels
        one_hot = one_hot * valid.unsqueeze(1)
        probs   = probs   * valid.unsqueeze(1)

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))  # (C,)
        cardinality  = (probs + one_hot).sum(dim=(0, 2, 3))  # (C,)
        dice_per_cls = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return dice_per_cls.mean()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.

        Args:
            logits:  ``(B, C, H, W)`` raw model output.
            targets: ``(B, H, W)`` ground-truth class indices.

        Returns:
            Tuple of (total_loss, breakdown_dict).
            breakdown_dict has keys ``ce``, ``dice``, ``total``.
        """
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        total     = self.alpha * ce_loss + (1.0 - self.alpha) * dice_loss
        return total, {"ce": ce_loss, "dice": dice_loss, "total": total}


# ============================================================
# CLASS: MetricsCalculator
# ============================================================
class MetricsCalculator:
    """Compute pixel-wise segmentation metrics from prediction arrays.

    Metrics:
        Pixel Accuracy:
            Fraction of correctly classified pixels.
            PA = Σ_c n_{cc} / Σ_c Σ_{c'} n_{cc'}

        Mean IoU (mIoU / Jaccard Index):
            Per-class IoU = TP / (TP + FP + FN).
            mIoU = mean of per-class IoUs (ignoring void classes).

        Mean Dice (F1):
            Per-class Dice = 2TP / (2TP + FP + FN).
            Mean Dice is numerically related to mIoU.

        Frequency-Weighted IoU (FWIoU):
            Weighted average of per-class IoU, weights = class frequency.
            Gives more weight to common classes.

    Args:
        num_classes:  Number of semantic classes.
        ignore_index: Pixels with this label are excluded from all metrics.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._logger = logging.getLogger(self.__class__.__name__)
        self._reset()

    def __repr__(self) -> str:
        return (
            f"MetricsCalculator(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "MetricsCalculator":
        """Construct from config dict."""
        return cls(
            num_classes=config["num_classes"],
            ignore_index=config.get("ignore_index", IGNORE_INDEX),
        )

    def _reset(self) -> None:
        """Reset the accumulated confusion matrix."""
        self._conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Accumulate predictions for a batch.

        Args:
            pred:   Integer array of predicted class ids, any shape.
            target: Integer array of ground-truth class ids, same shape.
        """
        pred   = pred.ravel()
        target = target.ravel()

        # Exclude ignore_index pixels
        valid  = target != self.ignore_index
        pred   = pred[valid]
        target = target[valid]

        # Clamp to valid range
        pred   = np.clip(pred,   0, self.num_classes - 1)
        target = np.clip(target, 0, self.num_classes - 1)

        np.add.at(
            self._conf_mat,
            (target, pred),
            1,
        )

    def compute(self) -> dict[str, float]:
        """Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dict with ``pixel_acc``, ``mean_iou``, ``mean_dice``,
            ``fw_iou``, and ``per_class_iou``.
        """
        cm = self._conf_mat.astype(np.float64)

        # Pixel accuracy
        pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-8)

        # Per-class IoU:  TP / (row_sum + col_sum - TP)
        tp   = np.diag(cm)
        fp   = cm.sum(axis=0) - tp   # predicted as class c but aren't
        fn   = cm.sum(axis=1) - tp   # actually class c but predicted as other
        iou  = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        # Only average over classes that appear in ground truth
        present = cm.sum(axis=1) > 0
        mean_iou  = float(iou[present].mean()) if present.any() else 0.0
        mean_dice = float(dice[present].mean()) if present.any() else 0.0

        # Frequency-weighted IoU
        freq  = cm.sum(axis=1) / (cm.sum() + 1e-8)
        fw_iou = float((freq[present] * iou[present]).sum())

        self._logger.debug(
            "Metrics: PA=%.4f  mIoU=%.4f  mDice=%.4f  FWIoU=%.4f",
            pixel_acc, mean_iou, mean_dice, fw_iou,
        )
        return {
            "pixel_acc":     float(pixel_acc),
            "mean_iou":      mean_iou,
            "mean_dice":     mean_dice,
            "fw_iou":        fw_iou,
            "per_class_iou": iou.tolist(),
        }

    def reset(self) -> None:
        """Reset accumulated state (call between epochs/splits)."""
        self._reset()


# ============================================================
# CLASS: Trainer
# ============================================================
class Trainer:
    """Training loop with poly LR schedule, deep supervision, early stopping.

    Poly LR schedule:
        lr(epoch) = lr_0 × (1 - epoch/max_epochs)^power
        Standard for segmentation — maintains high LR early, decays smoothly.

    Deep supervision:
        DeepLabV3 returns both ``"out"`` (main) and ``"aux"`` logits.
        Total loss = main_loss + 0.4 × aux_loss.
        Auxiliary head is applied to intermediate ResNet-101 features.

    Args:
        model:          Segmentation model.
        criterion:      ``SegmentationLoss`` instance.
        optimizer:      AdamW instance.
        device:         Training device.
        num_classes:    Number of semantic classes.
        checkpoint_dir: Where to save best checkpoints.
        patience:       Early stopping patience.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: SegmentationLoss,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_classes: int,
        checkpoint_dir: str = CHECKPOINT_DIR,
        patience: int = EARLY_STOP_PATIENCE,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        ensure_dir(checkpoint_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_miou": [], "val_miou": [],
        }
        self._best_miou: float = 0.0
        self._best_weights: Optional[dict] = None
        self._no_improve: int = 0

    def __repr__(self) -> str:
        return (
            f"Trainer(device={self.device}, patience={self.patience}, "
            f"backbone_type=DeepLabV3+)"
        )

    def _poly_lr(self, epoch: int, max_epochs: int, power: float = 0.9) -> float:
        """Poly LR decay schedule.

        Args:
            epoch:      Current epoch (0-indexed).
            max_epochs: Total epochs.
            power:      Exponent controlling the decay speed.

        Returns:
            Multiplicative LR factor for the scheduler.
        """
        return (1.0 - epoch / max_epochs) ** power

    def _run_epoch(
        self,
        loader: DataLoader,
        metrics_calc: MetricsCalculator,
        train: bool,
        epoch: int = 0,
        max_epochs: int = 1,
    ) -> tuple[float, float]:
        """Run one epoch over the dataloader.

        Args:
            loader:        DataLoader instance.
            metrics_calc:  MetricsCalculator (reset before calling).
            train:         True → gradient updates, False → eval only.
            epoch:         Current epoch number (for poly LR).
            max_epochs:    Total epochs.

        Returns:
            Tuple of (mean_loss, mean_iou).
        """
        self.model.train(train)
        metrics_calc.reset()
        total_loss, n_batches = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for images, masks in tqdm(
                loader,
                desc=f"{'Train' if train else 'Val  '} E{epoch:03d}",
                leave=False,
                ncols=90,
            ):
                images = images.to(self.device, non_blocking=True)
                masks  = masks.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad()

                output = self.model(images)

                # Main output
                logits = output["out"]
                # Upsample to mask size if needed (SegFormer may differ)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )

                loss, _ = self.criterion(logits, masks)

                # Deep supervision auxiliary loss (DeepLabV3 only)
                if "aux" in output:
                    aux_logits = output["aux"]
                    if aux_logits.shape[-2:] != masks.shape[-2:]:
                        aux_logits = F.interpolate(
                            aux_logits, size=masks.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                    aux_loss, _ = self.criterion(aux_logits, masks)
                    loss = loss + 0.4 * aux_loss

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

                # Accumulate metrics
                preds = logits.argmax(dim=1).cpu().numpy()
                tgts  = masks.cpu().numpy()
                metrics_calc.update(preds, tgts)

        mean_loss = total_loss / max(n_batches, 1)
        results   = metrics_calc.compute()
        return mean_loss, results["mean_iou"]

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = NUM_EPOCHS,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Args:
            train_loader: DataLoader for training split.
            val_loader:   DataLoader for validation split.
            num_epochs:   Maximum number of training epochs.

        Returns:
            Training history dict.
        """
        train_metrics = MetricsCalculator(self.num_classes)
        val_metrics   = MetricsCalculator(self.num_classes)
        base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]

        self._logger.info(
            "Training for up to %d epochs  (patience=%d)", num_epochs, self.patience
        )
        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            # Poly LR update
            factor = self._poly_lr(epoch - 1, num_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr * factor

            train_loss, train_miou = self._run_epoch(
                train_loader, train_metrics, train=True,
                epoch=epoch, max_epochs=num_epochs,
            )
            val_loss, val_miou = self._run_epoch(
                val_loader, val_metrics, train=False,
                epoch=epoch, max_epochs=num_epochs,
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_miou"].append(train_miou)
            self.history["val_miou"].append(val_miou)

            current_lr = self.optimizer.param_groups[-1]["lr"]
            self._logger.info(
                "Epoch %3d/%d  loss=%.4f/%.4f  mIoU=%.4f/%.4f  lr=%.2e",
                epoch, num_epochs,
                train_loss, val_loss,
                train_miou, val_miou,
                current_lr,
            )

            if val_miou > self._best_miou:
                self._best_miou = val_miou
                self._best_weights = copy.deepcopy(self.model.state_dict())
                self._no_improve = 0
                self.save_checkpoint(epoch, val_miou, "best.pt")
                self._logger.info("  ✓ New best mIoU=%.4f — saved.", val_miou)
            else:
                self._no_improve += 1

            if self._no_improve >= self.patience:
                self._logger.info(
                    "Early stopping after %d epochs without improvement.", self.patience
                )
                break

        elapsed = time.time() - t0
        self._logger.info(
            "Training complete in %.1fs  best_mIoU=%.4f", elapsed, self._best_miou
        )

        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)
        return self.history

    def save_checkpoint(
        self, epoch: int, val_miou: float, filename: str = "checkpoint.pt"
    ) -> str:
        """Save model state dict and metadata.

        Args:
            epoch:     Current epoch.
            val_miou:  Validation mIoU at this epoch.
            filename:  Output filename inside ``checkpoint_dir``.

        Returns:
            Full path to saved file.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "val_miou": val_miou,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        return path

    @staticmethod
    def load_checkpoint(
        model: nn.Module, path: str, device: torch.device
    ) -> tuple[nn.Module, int, float]:
        """Load a checkpoint from disk.

        Args:
            model:  Model instance matching the saved architecture.
            path:   Path to ``.pt`` checkpoint.
            device: Target device.

        Returns:
            Tuple of (model, epoch, val_miou).

        Raises:
            FileNotFoundError: Checkpoint file not found.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Checkpoint not found: '{path}'\n"
                "Train first or check the --checkpoint argument."
            )
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded checkpoint '%s'  epoch=%d  val_miou=%.4f",
            path, ckpt["epoch"], ckpt["val_miou"],
        )
        return model, ckpt["epoch"], ckpt["val_miou"]


# ============================================================
# CLASS: Visualizer
# ============================================================
class Visualizer:
    """Overlay colour-coded segmentation masks on images and plot metrics.

    Args:
        output_dir:  Directory for saved figures.
        class_names: Ordered list of class name strings.
        alpha:       Blending factor for mask overlay (0=invisible, 1=opaque).
    """

    # 20 distinct BGR colours for class mask overlay
    CLASS_COLORS: list[tuple[int, int, int]] = [
        (128, 64,  128), (244, 35,  232), (70,  70,  70),  (102, 102, 156),
        (190, 153, 153), (153, 153, 153), (250, 170, 30),  (220, 220, 0),
        (107, 142, 35),  (152, 251, 152), (70,  130, 180), (220, 20,  60),
        (255, 0,   0),   (0,   0,   142), (0,   0,   70),  (0,   60,  100),
        (0,   80,  100), (0,   0,   230), (119, 11,  32),  (111, 74,  0),
    ]

    def __init__(
        self,
        output_dir: str = LOG_DIR,
        class_names: Optional[list[str]] = None,
        alpha: float = 0.5,
    ) -> None:
        self.output_dir = output_dir
        self.class_names = class_names or []
        self.alpha = alpha
        ensure_dir(output_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"Visualizer(output_dir='{self.output_dir}', "
            f"n_classes={len(self.class_names)})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "Visualizer":
        """Construct from config dict."""
        return cls(
            output_dir=config.get("log_dir", LOG_DIR),
            class_names=config.get("class_names"),
        )

    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert an integer class mask to a BGR colour image.

        Args:
            mask: 2-D uint8/int array of class indices, shape ``(H, W)``.

        Returns:
            BGR colour image, shape ``(H, W, 3)``.
        """
        colour_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls_id, color in enumerate(self.CLASS_COLORS):
            colour_mask[mask == cls_id] = color
        return colour_mask

    def overlay_mask(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Blend colour mask over original image.

        Args:
            image: BGR uint8 image.
            mask:  Integer class mask, shape (H, W).

        Returns:
            Blended BGR image.
        """
        colour_mask = self.colorize_mask(mask)
        if colour_mask.shape[:2] != image.shape[:2]:
            colour_mask = cv2.resize(
                colour_mask, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return cv2.addWeighted(image, 1.0 - self.alpha, colour_mask, self.alpha, 0)

    def save_prediction_grid(
        self,
        images: list[np.ndarray],
        gt_masks: list[np.ndarray],
        pred_masks: list[np.ndarray],
        filename: str = "predictions.png",
        n_samples: int = 4,
    ) -> str:
        """Save a grid of image | GT mask | predicted mask.

        Args:
            images:     List of BGR images.
            gt_masks:   List of integer GT masks.
            pred_masks: List of integer predicted masks.
            filename:   Output filename.
            n_samples:  Number of samples to display.

        Returns:
            Full path to saved figure.
        """
        n = min(n_samples, len(images))
        fig, axes = plt.subplots(n, 3, figsize=(12, n * 3))
        if n == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Image", "Ground Truth", "Prediction"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=11, fontweight="bold")

        for i in range(n):
            rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            gt_overlay   = cv2.cvtColor(self.overlay_mask(images[i], gt_masks[i]),   cv2.COLOR_BGR2RGB)
            pred_overlay = cv2.cvtColor(self.overlay_mask(images[i], pred_masks[i]), cv2.COLOR_BGR2RGB)

            axes[i, 0].imshow(rgb);          axes[i, 0].axis("off")
            axes[i, 1].imshow(gt_overlay);   axes[i, 1].axis("off")
            axes[i, 2].imshow(pred_overlay); axes[i, 2].axis("off")

        # Legend
        patches = [
            mpatches.Patch(
                color=np.array(self.CLASS_COLORS[j % len(self.CLASS_COLORS)]) / 255.0,
                label=self.class_names[j] if j < len(self.class_names) else f"class_{j}",
            )
            for j in range(min(len(self.class_names), 10))
        ]
        if patches:
            fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8, frameon=False)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Prediction grid saved → %s", path)
        return path

    def plot_training_curves(
        self, history: dict[str, list[float]], filename: str = "training_curves.png"
    ) -> str:
        """Plot loss and mIoU curves side by side.

        Args:
            history:  Training history dict.
            filename: Output filename.

        Returns:
            Full path to saved figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], "b-o", ms=3, label="Train")
        ax1.plot(epochs, history["val_loss"],   "r-o", ms=3, label="Val")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["train_miou"], "b-o", ms=3, label="Train mIoU")
        ax2.plot(epochs, history["val_miou"],   "r-o", ms=3, label="Val mIoU")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU")
        ax2.set_title("mIoU Curves"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Training curves saved → %s", path)
        return path

    def plot_iou_bar(
        self,
        per_class_iou: list[float],
        mean_iou: float,
        filename: str = "per_class_iou.png",
    ) -> str:
        """Horizontal bar chart of per-class IoU.

        Args:
            per_class_iou: List of IoU values, one per class.
            mean_iou:      Overall mean IoU.
            filename:      Output filename.

        Returns:
            Full path to saved figure.
        """
        names = (
            self.class_names
            if len(self.class_names) == len(per_class_iou)
            else [f"class_{i}" for i in range(len(per_class_iou))]
        )
        colors = [
            np.array(self.CLASS_COLORS[i % len(self.CLASS_COLORS)]) / 255.0
            for i in range(len(names))
        ]
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.5)))
        bars = ax.barh(names, per_class_iou, color=colors)
        ax.axvline(mean_iou, color="red", linestyle="--", label=f"mIoU={mean_iou:.3f}")
        ax.set_xlabel("IoU"); ax.set_title("Per-class IoU")
        ax.set_xlim(0, 1); ax.legend()
        for bar, v in zip(bars, per_class_iou):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Per-class IoU chart saved → %s", path)
        return path


# ============================================================
# CLASS: SegmentationPipeline  (orchestrator)
# ============================================================
class SegmentationPipeline:
    """End-to-end orchestrator: download → train → evaluate → infer.

    Args:
        mode:           ``"train"``, ``"evaluate"``, or ``"infer"``.
        backbone:       ``"deeplabv3plus"`` or ``"segformer"``.
        data_root:      Existing dataset root (skips Roboflow download).
        checkpoint:     Checkpoint path for evaluate / infer modes.
        source:         Image path for inference.
        output_dir:     Directory for all outputs.
        config:         Optional flat config dict.

    Example::

        pipeline = SegmentationPipeline(mode="train", backbone="deeplabv3plus")
        pipeline.run()
    """

    def __init__(
        self,
        mode: str = "train",
        backbone: str = BACKBONE,
        data_root: Optional[str] = None,
        checkpoint: Optional[str] = None,
        source: Optional[str] = None,
        output_dir: str = LOG_DIR,
        config: Optional[dict] = None,
    ) -> None:
        self.mode = mode
        self.backbone = backbone
        self.data_root = data_root
        self.checkpoint = checkpoint
        self.source = source
        self.output_dir = output_dir
        self.config = config or {}
        self.device = get_device()
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"SegmentationPipeline(mode='{self.mode}', "
            f"backbone='{self.backbone}')"
        )

    @classmethod
    def from_config(cls, config: dict) -> "SegmentationPipeline":
        """Construct from a flat config dict."""
        return cls(
            mode=config.get("mode", "train"),
            backbone=config.get("backbone", BACKBONE),
            data_root=config.get("data_root"),
            checkpoint=config.get("checkpoint"),
            source=config.get("source"),
            output_dir=config.get("output_dir", LOG_DIR),
            config=config,
        )

    def _get_data_root(self) -> str:
        """Download or verify the dataset root."""
        if self.data_root and os.path.isdir(self.data_root):
            self._logger.info("Using existing dataset: %s", self.data_root)
            return self.data_root
        downloader = DatasetDownloader.from_config(self.config)
        return downloader.download()

    def _count_classes(self, root: str) -> int:
        """Scan train masks to determine number of classes.

        Args:
            root: Dataset root.

        Returns:
            Number of unique class indices found in training masks.
        """
        mask_dir = None
        for cand in ("train/masks", "train/labels"):
            d = os.path.join(root, cand)
            if os.path.isdir(d):
                mask_dir = d
                break
        if not mask_dir:
            self._logger.warning("Could not find masks dir; defaulting to 2 classes.")
            return 2

        class_ids = set()
        for fname in os.listdir(mask_dir):
            if not fname.endswith(".png"):
                continue
            m = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                unique = np.unique(m)
                class_ids.update(int(v) for v in unique if v != IGNORE_INDEX)

        n = max(class_ids) + 1 if class_ids else 2
        self._logger.info("Auto-detected %d classes from masks.", n)
        return n

    def run(self) -> dict:
        """Execute the pipeline.

        Returns:
            Result dict (contents depend on mode).
        """
        set_seed(SEED)
        self._logger.info("=" * 55)
        self._logger.info(
            "Segmentation Pipeline  mode='%s'  backbone='%s'",
            self.mode, self.backbone,
        )
        self._logger.info("=" * 55)

        if self.mode == "train":
            return self._run_train()
        elif self.mode == "evaluate":
            return self._run_evaluate()
        elif self.mode == "infer":
            return self._run_infer()
        else:
            raise ValueError(
                f"Unknown mode='{self.mode}'. Choose: train / evaluate / infer."
            )

    def _run_train(self) -> dict:
        """Download data, train, evaluate, visualise."""
        root = self._get_data_root()
        num_classes = self._count_classes(root)

        train_ds = SegmentationDataset(root, "train", IMAGE_SIZE, augment=True,  num_classes=num_classes)
        val_ds   = SegmentationDataset(root, "valid", IMAGE_SIZE, augment=False, num_classes=num_classes)
        try:
            test_ds = SegmentationDataset(root, "test", IMAGE_SIZE, augment=False, num_classes=num_classes)
        except FileNotFoundError:
            self._logger.warning("No test split found — using valid for final evaluation.")
            test_ds = val_ds

        class_names = getattr(train_ds, "class_names", [f"class_{i}" for i in range(num_classes)])

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

        builder  = ModelBuilder(num_classes=num_classes, backbone=self.backbone)
        model    = builder.build()
        optimizer = builder.get_optimizer(model, LEARNING_RATE)
        criterion = SegmentationLoss(num_classes=num_classes)

        trainer = Trainer(
            model, criterion, optimizer, self.device,
            num_classes=num_classes, checkpoint_dir=CHECKPOINT_DIR,
        )
        history = trainer.train(train_loader, val_loader, NUM_EPOCHS)

        # Final evaluation
        metrics_calc = MetricsCalculator(num_classes)
        test_loss, test_miou = trainer._run_epoch(
            test_loader, metrics_calc, train=False
        )
        test_metrics = metrics_calc.compute()
        self._logger.info(
            "Test  loss=%.4f  mIoU=%.4f  pixAcc=%.4f",
            test_loss, test_metrics["mean_iou"], test_metrics["pixel_acc"],
        )

        # Visualise
        viz = Visualizer(output_dir=self.output_dir, class_names=class_names)
        paths = {
            "curves":   viz.plot_training_curves(history),
            "iou_bar":  viz.plot_iou_bar(
                test_metrics["per_class_iou"], test_metrics["mean_iou"]
            ),
        }

        # Save metrics
        ensure_dir(self.output_dir)
        mpath = os.path.join(self.output_dir, "metrics.json")
        with open(mpath, "w") as f:
            json.dump(test_metrics, f, indent=2)
        self._logger.info("Metrics saved → %s", mpath)

        return {"metrics": test_metrics, "history": history, "output_paths": paths}

    def _run_evaluate(self) -> dict:
        """Load checkpoint, run evaluation on val split."""
        if not self.checkpoint:
            raise ValueError("--checkpoint is required for evaluate mode.")

        root = self._get_data_root()
        num_classes = self._count_classes(root)

        val_ds = SegmentationDataset(root, "valid", IMAGE_SIZE, augment=False, num_classes=num_classes)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        builder = ModelBuilder(num_classes=num_classes, backbone=self.backbone, pretrained=False)
        model   = builder.build()
        model, epoch, _ = Trainer.load_checkpoint(model, self.checkpoint, self.device)
        model = model.to(self.device)
        model.eval()

        criterion = SegmentationLoss(num_classes=num_classes)
        metrics_calc = MetricsCalculator(num_classes)
        dummy_trainer = Trainer.__new__(Trainer)
        dummy_trainer.model = model
        dummy_trainer.criterion = criterion
        dummy_trainer.device = self.device
        dummy_trainer.num_classes = num_classes
        dummy_trainer._logger = logging.getLogger("EvalTrainer")

        _, _ = dummy_trainer._run_epoch(val_loader, metrics_calc, train=False)
        metrics = metrics_calc.compute()
        self._logger.info("Evaluation mIoU=%.4f", metrics["mean_iou"])

        viz = Visualizer(output_dir=self.output_dir)
        paths = {"iou_bar": viz.plot_iou_bar(metrics["per_class_iou"], metrics["mean_iou"])}
        return {"metrics": metrics, "output_paths": paths}

    def _run_infer(self) -> dict:
        """Run inference on a single image."""
        if not self.source:
            raise ValueError("--source is required for infer mode.")
        if not self.checkpoint:
            raise ValueError("--checkpoint is required for infer mode.")
        if not os.path.isfile(self.source):
            raise FileNotFoundError(f"Source image not found: '{self.source}'")

        # Determine num_classes from checkpoint
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        state = ckpt["model_state_dict"]
        cls_keys = [k for k in state if "classifier.4.weight" in k or "decode_head.classifier.weight" in k]
        num_classes = state[cls_keys[0]].shape[0] if cls_keys else 2

        builder = ModelBuilder(num_classes=num_classes, backbone=self.backbone, pretrained=False)
        model   = builder.build()
        model, _, _ = Trainer.load_checkpoint(model, self.checkpoint, self.device)
        model = model.to(self.device)
        model.eval()

        img = cv2.imread(self.source)
        if img is None:
            raise RuntimeError(f"Could not read image: '{self.source}'")

        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_t       = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        from torchvision import transforms as T
        normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        img_t = normalize(img_t).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(img_t)
            logits = output["out"]
            pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Resize prediction back to original image size
        pred_orig = cv2.resize(
            pred.astype(np.uint8), (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        viz  = Visualizer(output_dir=self.output_dir)
        stem = Path(self.source).stem
        out_path = os.path.join(self.output_dir, f"{stem}_segmentation.png")
        overlay  = viz.overlay_mask(img, pred_orig)
        cv2.imwrite(out_path, overlay)
        self._logger.info("Segmentation saved → %s", out_path)

        return {
            "pred_mask":  pred_orig,
            "output_image": out_path,
            "unique_classes": np.unique(pred_orig).tolist(),
        }


# ============================================================
# ENTRY POINT
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic Segmentation — Section 4")
    parser.add_argument("--mode", choices=["train", "evaluate", "infer"],
                        default="train", help="Pipeline mode.")
    parser.add_argument("--backbone", choices=["deeplabv3plus", "segformer"],
                        default=BACKBONE, help="Model backbone.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Existing dataset root (skips download).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint (evaluate / infer modes).")
    parser.add_argument("--source", type=str, default=None,
                        help="Image path for inference mode.")
    parser.add_argument("--output-dir", type=str, default=LOG_DIR,
                        help="Directory for output files.")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    pipeline = SegmentationPipeline(
        mode=args.mode,
        backbone=args.backbone,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        source=args.source,
        output_dir=args.output_dir,
    )
    result = pipeline.run()
    if "metrics" in result:
        print(json.dumps(
            {k: v for k, v in result["metrics"].items() if isinstance(v, float)},
            indent=2,
        ))


if __name__ == "__main__":
    main()
