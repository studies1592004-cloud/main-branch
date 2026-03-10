from __future__ import annotations

"""
image_classification.py
========================
Industry-Standard Image Classification Pipeline with Transfer Learning

Installation
------------
    pip install torch torchvision roboflow matplotlib scikit-learn seaborn tqdm

Usage
-----
    # Download dataset and train in fine_tuning mode (default)
    python image_classification.py

    # Choose a transfer learning mode
    python image_classification.py --mode feature_extraction
    python image_classification.py --mode fine_tuning
    python image_classification.py --mode linear_probe

    # Inference on a single image
    python image_classification.py --infer path/to/image.jpg --checkpoint ./checkpoints/best.pt

Author: CV Course — Section 2
Python: 3.9+  |  PyTorch: 2.x
"""

# ============================================================
# GLOBAL CONFIGURATION — edit these before running
# ============================================================
ROBOFLOW_API_KEY: str  = "YOUR_API_KEY_HERE"   # free key at roboflow.com
ROBOFLOW_WORKSPACE: str = "roboflow-universe-projects"
ROBOFLOW_PROJECT: str  = "rock-paper-scissors"
ROBOFLOW_VERSION: int  = 14

DEVICE: str           = "cuda"   # overridden at runtime via torch
SEED: int             = 42
IMAGE_SIZE: int       = 224      # ResNet50 canonical input size
BATCH_SIZE: int       = 32
NUM_WORKERS: int      = 2
LEARNING_RATE: float  = 1e-4
NUM_EPOCHS: int       = 30
EARLY_STOP_PATIENCE: int = 7
CHECKPOINT_DIR: str   = "./checkpoints"
LOG_DIR: str          = "./logs"

# ImageNet normalisation constants
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float]  = [0.229, 0.224, 0.225]

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
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from torchvision.datasets import ImageFolder
except ImportError as exc:
    sys.exit(
        f"PyTorch import failed: {exc}\n"
        "Run:  pip install torch torchvision"
    )

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
except ImportError as exc:
    sys.exit(
        f"scikit-learn import failed: {exc}\n"
        "Run:  pip install scikit-learn"
    )

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback — tqdm is optional
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Classification")


# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed: int = SEED) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", dev)
    return dev


def ensure_dir(path: str) -> None:
    """Create directory tree if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS: DatasetDownloader
# ============================================================
class DatasetDownloader:
    """Download a Roboflow classification dataset to local disk.

    The Roboflow Python SDK handles authentication and unpacking.
    The downloaded folder follows the ImageFolder layout expected by
    torchvision:
        <root>/train/<class_name>/<image>.jpg
        <root>/valid/<class_name>/<image>.jpg
        <root>/test/<class_name>/<image>.jpg

    Args:
        api_key:   Roboflow API key (free at roboflow.com).
        workspace: Roboflow workspace slug.
        project:   Project slug.
        version:   Dataset version number.
        dest_dir:  Local destination directory.

    Raises:
        ImportError:  If the ``roboflow`` package is not installed.
        RuntimeError: If the API key is the placeholder value or the
                      download fails for any reason.
    """

    def __init__(
        self,
        api_key: str = ROBOFLOW_API_KEY,
        workspace: str = ROBOFLOW_WORKSPACE,
        project: str = ROBOFLOW_PROJECT,
        version: int = ROBOFLOW_VERSION,
        dest_dir: str = "./data",
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
            dest_dir=config.get("dest_dir", "./data"),
        )

    def download(self) -> str:
        """Download dataset and return the local root directory path.

        Returns:
            Path to the dataset root folder (contains train/valid/test).

        Raises:
            ImportError:  ``roboflow`` is not installed.
            RuntimeError: API key is placeholder or download fails.
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
                f"roboflow package not found: {exc}\n"
                "Run:  pip install roboflow"
            ) from exc

        self._logger.info(
            "Downloading %s/%s  version=%d",
            self.workspace,
            self.project,
            self.version,
        )
        try:
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace).project(self.project)
            dataset = project.version(self.version).download(
                "folder", location=self.dest_dir
            )
            root = dataset.location
        except Exception as exc:
            raise RuntimeError(
                f"Roboflow download failed: {exc}\n"
                "Check your API key, workspace/project slugs, and internet connection."
            ) from exc

        self._logger.info("Dataset downloaded to: %s", root)
        return root


# ============================================================
# CLASS: TransformBuilder
# ============================================================
class TransformBuilder:
    """Build train and validation torchvision transform pipelines.

    Train augmentations:
        RandomResizedCrop → RandomHorizontalFlip → ColorJitter →
        RandomGrayscale → ToTensor → Normalize

    Val / test transforms (deterministic):
        Resize → CenterCrop → ToTensor → Normalize

    Args:
        image_size: Spatial size fed to the model (224 for ResNet50).
        mean:       Per-channel normalisation mean (ImageNet defaults).
        std:        Per-channel normalisation std (ImageNet defaults).
    """

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        return f"TransformBuilder(image_size={self.image_size})"

    @classmethod
    def from_config(cls, config: dict) -> "TransformBuilder":
        """Construct from config dict."""
        return cls(
            image_size=config.get("image_size", IMAGE_SIZE),
            mean=config.get("mean", IMAGENET_MEAN),
            std=config.get("std", IMAGENET_STD),
        )

    def train_transform(self) -> transforms.Compose:
        """Return augmented training transform pipeline.

        Returns:
            ``torchvision.transforms.Compose`` pipeline.
        """
        return transforms.Compose([
            transforms.RandomResizedCrop(
                self.image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def val_transform(self) -> transforms.Compose:
        """Return deterministic validation / inference transform pipeline.

        Returns:
            ``torchvision.transforms.Compose`` pipeline.
        """
        return transforms.Compose([
            transforms.Resize(int(self.image_size * 256 / 224)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])


# ============================================================
# CLASS: ClassificationDataset
# ============================================================
class ClassificationDataset(Dataset):
    """Thin wrapper around ``torchvision.datasets.ImageFolder``.

    Adds class-count validation and graceful missing-split handling.

    Args:
        root:      Dataset root folder (contains train/valid/test sub-dirs).
        split:     One of ``"train"``, ``"valid"``, ``"test"``.
        transform: torchvision transform pipeline.

    Raises:
        FileNotFoundError: If the requested split directory does not exist.
        RuntimeError:      If fewer than 2 classes are found.
    """

    SPLIT_ALIASES: dict[str, list[str]] = {
        "valid": ["valid", "val", "validation"],
        "test":  ["test"],
        "train": ["train"],
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = root
        self.split = split
        split_dir = self._resolve_split_dir(root, split)
        self._dataset = ImageFolder(split_dir, transform=transform)
        self.classes: list[str] = self._dataset.classes
        self.class_to_idx: dict[str, int] = self._dataset.class_to_idx

        if len(self.classes) < 2:
            raise RuntimeError(
                f"Found only {len(self.classes)} class(es) in '{split_dir}'. "
                "Expected at least 2."
            )

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info(
            "Split='%s'  samples=%d  classes=%d  %s",
            split,
            len(self._dataset),
            len(self.classes),
            self.classes,
        )

    def __repr__(self) -> str:
        return (
            f"ClassificationDataset(split='{self.split}', "
            f"n={len(self)}, classes={self.classes})"
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._dataset[idx]

    @classmethod
    def _resolve_split_dir(cls, root: str, split: str) -> str:
        """Find the split directory, trying common aliases.

        Args:
            root:  Dataset root.
            split: Requested split name.

        Returns:
            Resolved path string.

        Raises:
            FileNotFoundError: If no alias matches an existing directory.
        """
        candidates = cls.SPLIT_ALIASES.get(split, [split])
        for name in candidates:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(
            f"Could not find split '{split}' in '{root}'.\n"
            f"Tried: {candidates}\n"
            f"Available directories: {os.listdir(root)}"
        )


# ============================================================
# CLASS: ModelBuilder
# ============================================================
class ModelBuilder:
    """Build a ResNet-50 model configured for transfer learning.

    Three modes:
        ``feature_extraction``:
            All backbone layers frozen.  Only the final FC head is trained.
            Best for: tiny datasets (<1 K samples), similar domain.

        ``linear_probe``:
            Same as feature_extraction but the head is a single linear layer
            with no non-linearity.  Used for evaluating representation quality.

        ``fine_tuning``:
            Differential learning rates — backbone uses LR × 0.1, head uses
            full LR.  Backbone is unfrozen after a warmup phase.
            Best for: medium/large datasets, any domain.

    Args:
        num_classes:  Number of output classes.
        mode:         Transfer learning mode (see above).
        pretrained:   Use ImageNet-pretrained weights.
        dropout_rate: Dropout before the final FC layer (0 = disabled).
    """

    VALID_MODES: tuple[str, ...] = (
        "feature_extraction", "fine_tuning", "linear_probe"
    )

    def __init__(
        self,
        num_classes: int,
        mode: str = "fine_tuning",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode='{mode}' is not valid. Choose from {self.VALID_MODES}."
            )
        self.num_classes = num_classes
        self.mode = mode
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"ModelBuilder(num_classes={self.num_classes}, "
            f"mode='{self.mode}', pretrained={self.pretrained})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "ModelBuilder":
        """Construct from config dict."""
        return cls(
            num_classes=config["num_classes"],
            mode=config.get("mode", "fine_tuning"),
            pretrained=config.get("pretrained", True),
            dropout_rate=config.get("dropout_rate", 0.3),
        )

    def build(self) -> nn.Module:
        """Construct and configure the ResNet-50 model.

        Returns:
            Configured ``nn.Module``, ready for training.
        """
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        model = models.resnet50(weights=weights)
        self._logger.info(
            "Loaded ResNet-50  pretrained=%s", self.pretrained
        )

        # Replace the final classification head
        in_features = model.fc.in_features   # 2048 for ResNet-50
        if self.mode == "linear_probe":
            model.fc = nn.Linear(in_features, self.num_classes)
        else:
            model.fc = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, self.num_classes),
            )

        # Freeze / unfreeze layers based on mode
        if self.mode in ("feature_extraction", "linear_probe"):
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
            frozen = sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            )
            trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            self._logger.info(
                "Mode='%s'  frozen=%s  trainable=%s params",
                self.mode,
                f"{frozen:,}",
                f"{trainable:,}",
            )
        else:
            trainable = sum(p.numel() for p in model.parameters())
            self._logger.info(
                "Mode='fine_tuning'  all %s params trainable "
                "(backbone LR × 0.1)",
                f"{trainable:,}",
            )

        return model

    def get_optimizer(
        self, model: nn.Module, lr: float = LEARNING_RATE
    ) -> optim.Optimizer:
        """Build an AdamW optimiser with differential learning rates.

        For ``fine_tuning``: backbone gets ``lr × 0.1``, head gets ``lr``.
        For other modes: only head parameters are included.

        Args:
            model: The model returned by ``build()``.
            lr:    Base learning rate for the classification head.

        Returns:
            Configured ``torch.optim.AdamW`` instance.
        """
        if self.mode == "fine_tuning":
            head_params = list(model.fc.parameters())
            head_ids = {id(p) for p in head_params}
            backbone_params = [
                p for p in model.parameters()
                if id(p) not in head_ids and p.requires_grad
            ]
            param_groups = [
                {"params": backbone_params, "lr": lr * 0.1, "name": "backbone"},
                {"params": head_params,     "lr": lr,       "name": "head"},
            ]
            self._logger.info(
                "AdamW: backbone LR=%.2e  head LR=%.2e", lr * 0.1, lr
            )
        else:
            param_groups = [
                {"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": lr,
                 "name": "head"}
            ]
            self._logger.info("AdamW: head LR=%.2e", lr)

        return optim.AdamW(param_groups, weight_decay=1e-4)


# ============================================================
# CLASS: Trainer
# ============================================================
class Trainer:
    """Training loop with validation, LR scheduling, and early stopping.

    Args:
        model:       The PyTorch model to train.
        criterion:   Loss function (e.g. ``nn.CrossEntropyLoss``).
        optimizer:   Optimiser instance.
        device:      ``torch.device`` to run on.
        checkpoint_dir: Directory to save best checkpoints.
        patience:    Early stopping patience (epochs without val improvement).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = CHECKPOINT_DIR,
        patience: int = EARLY_STOP_PATIENCE,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        ensure_dir(checkpoint_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

        # History tracking
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }
        self._best_val_acc: float = 0.0
        self._best_weights: Optional[dict] = None
        self._no_improve: int = 0

    def __repr__(self) -> str:
        return (
            f"Trainer(device={self.device}, patience={self.patience}, "
            f"checkpoint_dir='{self.checkpoint_dir}')"
        )

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> tuple[float, float]:
        """Run one full pass over a DataLoader.

        Args:
            loader: DataLoader instance.
            train:  If True, runs in training mode with gradient updates.

        Returns:
            Tuple of (mean_loss, accuracy) for this epoch.
        """
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for images, labels in tqdm(
                loader,
                desc="Train" if train else "Val  ",
                leave=False,
                ncols=80,
            ):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = NUM_EPOCHS,
    ) -> dict[str, list[float]]:
        """Full training loop with cosine-annealing LR and early stopping.

        Args:
            train_loader: DataLoader for training split.
            val_loader:   DataLoader for validation split.
            num_epochs:   Maximum number of epochs.

        Returns:
            Training history dictionary with loss and accuracy curves.
        """
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-7
        )

        self._logger.info(
            "Training for up to %d epochs  (patience=%d)",
            num_epochs,
            self.patience,
        )
        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   train=False)
            scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            current_lr = self.optimizer.param_groups[-1]["lr"]
            self._logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                "train_acc=%.4f  val_acc=%.4f  lr=%.2e",
                epoch, num_epochs,
                train_loss, val_loss, train_acc, val_acc, current_lr,
            )

            # Checkpoint if improved
            if val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                self._best_weights = copy.deepcopy(self.model.state_dict())
                self._no_improve = 0
                self.save_checkpoint(epoch, val_acc, filename="best.pt")
                self._logger.info("  ✓ New best val_acc=%.4f — saved.", val_acc)
            else:
                self._no_improve += 1

            # Early stopping
            if self._no_improve >= self.patience:
                self._logger.info(
                    "Early stopping after %d epochs without improvement.", self.patience
                )
                break

        elapsed = time.time() - t0
        self._logger.info(
            "Training complete in %.1fs  best_val_acc=%.4f", elapsed, self._best_val_acc
        )

        # Restore best weights
        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)

        return self.history

    def save_checkpoint(
        self, epoch: int, val_acc: float, filename: str = "checkpoint.pt"
    ) -> str:
        """Save model state dict and metadata to disk.

        Args:
            epoch:    Current epoch number.
            val_acc:  Validation accuracy at this epoch.
            filename: Output filename inside ``checkpoint_dir``.

        Returns:
            Full path to the saved checkpoint.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "val_acc": val_acc,
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
            model:  Model instance (must match the saved architecture).
            path:   Path to ``.pt`` checkpoint file.
            device: Device to map tensors to.

        Returns:
            Tuple of (model_with_loaded_weights, epoch, val_acc).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Checkpoint not found: '{path}'\n"
                "Train the model first or check the --checkpoint argument."
            )
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded checkpoint from '%s'  epoch=%d  val_acc=%.4f",
            path, ckpt["epoch"], ckpt["val_acc"],
        )
        return model, ckpt["epoch"], ckpt["val_acc"]


# ============================================================
# CLASS: Evaluator
# ============================================================
class Evaluator:
    """Compute and display classification metrics on a held-out set.

    Metrics:
        - Top-1 accuracy
        - Top-5 accuracy (where applicable)
        - Per-class precision, recall, F1
        - Macro / weighted averages
        - Confusion matrix

    Args:
        model:       Trained model.
        device:      Inference device.
        class_names: Ordered list of class name strings.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: list[str],
    ) -> None:
        self.model = model
        self.device = device
        self.class_names = class_names
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"Evaluator(classes={self.class_names}, device={self.device})"
        )

    @torch.no_grad()
    def _collect_predictions(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference over a DataLoader and collect outputs.

        Args:
            loader: DataLoader with labelled samples.

        Returns:
            Tuple of (all_labels, all_preds, all_probs) as NumPy arrays.
        """
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        for images, labels in tqdm(loader, desc="Eval", leave=False, ncols=80):
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Compute all metrics and log a classification report.

        Args:
            loader: DataLoader for the evaluation split.

        Returns:
            Dictionary of scalar metric values.
        """
        labels, preds, probs = self._collect_predictions(loader)
        n_classes = len(self.class_names)

        # Top-1
        top1 = float((preds == labels).mean())

        # Top-5 (only meaningful if n_classes >= 5)
        if n_classes >= 5:
            top5_correct = 0
            for i, lbl in enumerate(labels):
                top5_idx = np.argsort(probs[i])[-5:]
                top5_correct += int(lbl in top5_idx)
            top5 = top5_correct / len(labels)
        else:
            top5 = top1

        precision = precision_score(labels, preds, average="weighted", zero_division=0)
        recall    = recall_score(labels, preds, average="weighted", zero_division=0)
        f1        = f1_score(labels, preds, average="weighted", zero_division=0)

        metrics = {
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "precision_weighted": precision,
            "recall_weighted":    recall,
            "f1_weighted":        f1,
        }

        self._logger.info(
            "Evaluation results:\n"
            "  Top-1 Acc : %.4f\n"
            "  Top-5 Acc : %.4f\n"
            "  Precision : %.4f\n"
            "  Recall    : %.4f\n"
            "  F1        : %.4f",
            top1, top5, precision, recall, f1,
        )

        report = classification_report(
            labels, preds,
            target_names=self.class_names,
            zero_division=0,
        )
        self._logger.info("Per-class report:\n%s", report)

        # Cache for visualisation
        self._last_labels = labels
        self._last_preds  = preds
        self._last_probs  = probs
        return metrics

    def confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix from the last ``evaluate()`` call.

        Raises:
            RuntimeError: If ``evaluate()`` has not been called yet.
        """
        if not hasattr(self, "_last_labels"):
            raise RuntimeError("Call evaluate() before confusion_matrix().")
        return confusion_matrix(self._last_labels, self._last_preds)


# ============================================================
# CLASS: Visualizer
# ============================================================
class Visualizer:
    """Save training curves, confusion matrix, and misclassified samples.

    Args:
        output_dir:  Directory for saved figures.
        class_names: Ordered class label strings.
    """

    def __init__(
        self, output_dir: str = LOG_DIR, class_names: Optional[list[str]] = None
    ) -> None:
        self.output_dir = output_dir
        self.class_names = class_names or []
        ensure_dir(output_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"Visualizer(output_dir='{self.output_dir}')"

    def plot_training_curves(
        self, history: dict[str, list[float]], filename: str = "training_curves.png"
    ) -> str:
        """Plot loss and accuracy curves side by side.

        Args:
            history:  Dict with keys train_loss, val_loss, train_acc, val_acc.
            filename: Output filename.

        Returns:
            Full path to the saved figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], "b-o", ms=3, label="Train")
        ax1.plot(epochs, history["val_loss"],   "r-o", ms=3, label="Val")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["train_acc"], "b-o", ms=3, label="Train")
        ax2.plot(epochs, history["val_acc"],   "r-o", ms=3, label="Val")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy Curves"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Training curves saved → %s", path)
        return path

    def plot_confusion_matrix(
        self, cm: np.ndarray, filename: str = "confusion_matrix.png"
    ) -> str:
        """Save a heatmap of the normalised confusion matrix.

        Args:
            cm:       Raw (count) confusion matrix from ``Evaluator``.
            filename: Output filename.

        Returns:
            Full path to the saved figure.
        """
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        n = len(cm)
        fig_size = max(6, n * 0.9)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names or range(n),
            yticklabels=self.class_names or range(n),
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title("Confusion Matrix (normalised)", fontsize=13)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Confusion matrix saved → %s", path)
        return path

    def plot_misclassified(
        self,
        loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        n_samples: int = 16,
        filename: str = "misclassified.png",
    ) -> str:
        """Visualise images the model got wrong.

        Shows the true label, predicted label, and confidence.

        Args:
            loader:    DataLoader (val or test split, shuffle=False).
            model:     Trained model.
            device:    Inference device.
            n_samples: Number of wrong samples to display.
            filename:  Output filename.

        Returns:
            Full path to the saved figure.
        """
        model.eval()
        wrong_imgs, wrong_true, wrong_pred, wrong_conf = [], [], [], []
        inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
        inv_std  = [1.0 / s for s in IMAGENET_STD]
        inv_norm = transforms.Normalize(mean=inv_mean, std=inv_std)

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                probs  = torch.softmax(logits, dim=1).cpu()
                preds  = probs.argmax(dim=1)
                confs  = probs.max(dim=1).values

                for i in range(len(labels)):
                    if preds[i] != labels[i]:
                        wrong_imgs.append(inv_norm(images[i].cpu()))
                        wrong_true.append(labels[i].item())
                        wrong_pred.append(preds[i].item())
                        wrong_conf.append(confs[i].item())
                        if len(wrong_imgs) >= n_samples:
                            break
                if len(wrong_imgs) >= n_samples:
                    break

        if not wrong_imgs:
            self._logger.info("No misclassified samples found.")
            return ""

        cols = 4
        rows = (len(wrong_imgs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).ravel()

        for ax, img_t, true_idx, pred_idx, conf in zip(
            axes, wrong_imgs, wrong_true, wrong_pred, wrong_conf
        ):
            img_np = img_t.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
            true_name = self.class_names[true_idx] if self.class_names else str(true_idx)
            pred_name = self.class_names[pred_idx] if self.class_names else str(pred_idx)
            ax.set_title(
                f"True: {true_name}\nPred: {pred_name} ({conf:.2f})",
                fontsize=7,
                color="red",
            )
            ax.axis("off")

        for ax in axes[len(wrong_imgs):]:
            ax.set_visible(False)

        plt.suptitle("Misclassified Samples", fontsize=12)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Misclassified grid saved → %s", path)
        return path


# ============================================================
# CLASS: ClassificationPipeline  (orchestrator)
# ============================================================
class ClassificationPipeline:
    """End-to-end orchestrator: download → build → train → evaluate → visualise.

    Args:
        mode:           Transfer learning mode.
        data_root:      If provided, skips Roboflow download and uses this folder.
        checkpoint_dir: Where to save model checkpoints.
        log_dir:        Where to save visualisation outputs.
        config:         Optional flat config dict to override defaults.

    Example::

        pipeline = ClassificationPipeline(mode="fine_tuning")
        pipeline.run()
    """

    def __init__(
        self,
        mode: str = "fine_tuning",
        data_root: Optional[str] = None,
        checkpoint_dir: str = CHECKPOINT_DIR,
        log_dir: str = LOG_DIR,
        config: Optional[dict] = None,
    ) -> None:
        self.mode = mode
        self.data_root = data_root
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.config = config or {}
        self.device = get_device()
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"ClassificationPipeline(mode='{self.mode}', "
            f"data_root='{self.data_root}')"
        )

    @classmethod
    def from_config(cls, config: dict) -> "ClassificationPipeline":
        """Construct from a flat config dict."""
        return cls(
            mode=config.get("mode", "fine_tuning"),
            data_root=config.get("data_root"),
            checkpoint_dir=config.get("checkpoint_dir", CHECKPOINT_DIR),
            log_dir=config.get("log_dir", LOG_DIR),
            config=config,
        )

    def _download_data(self) -> str:
        """Download dataset via Roboflow if ``data_root`` is not set."""
        if self.data_root and os.path.isdir(self.data_root):
            self._logger.info("Using existing dataset: %s", self.data_root)
            return self.data_root
        downloader = DatasetDownloader.from_config(self.config)
        return downloader.download()

    def run(self) -> dict:
        """Execute the full pipeline.

        Steps:
            1. Download dataset (Roboflow).
            2. Build DataLoaders.
            3. Build model + optimiser.
            4. Train with early stopping.
            5. Evaluate on test set.
            6. Save visualisations.

        Returns:
            Dictionary with ``metrics``, ``history``, and ``output_paths``.
        """
        set_seed(SEED)
        self._logger.info("=" * 55)
        self._logger.info("Classification Pipeline  mode='%s'", self.mode)
        self._logger.info("=" * 55)

        # 1. Data
        root = self._download_data()
        tb = TransformBuilder.from_config(self.config)

        try:
            train_ds = ClassificationDataset(root, "train", tb.train_transform())
            val_ds   = ClassificationDataset(root, "valid", tb.val_transform())
        except FileNotFoundError as exc:
            self._logger.error("Dataset split not found: %s", exc)
            raise

        # Try loading test split — fall back to val if absent
        try:
            test_ds = ClassificationDataset(root, "test", tb.val_transform())
        except FileNotFoundError:
            self._logger.warning("No 'test' split found — using 'valid' for evaluation.")
            test_ds = val_ds

        class_names = train_ds.classes
        n_classes = len(class_names)
        self._logger.info(
            "Dataset: %d classes, train=%d, val=%d, test=%d",
            n_classes, len(train_ds), len(val_ds), len(test_ds),
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

        # 2. Model
        builder = ModelBuilder(
            num_classes=n_classes,
            mode=self.mode,
            pretrained=True,
        )
        model = builder.build()
        optimizer = builder.get_optimizer(model, lr=LEARNING_RATE)

        # 3. Train
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        trainer = Trainer(
            model, criterion, optimizer, self.device,
            checkpoint_dir=self.checkpoint_dir,
        )
        history = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)

        # 4. Evaluate
        evaluator = Evaluator(trainer.model, self.device, class_names)
        metrics = evaluator.evaluate(test_loader)
        cm = evaluator.confusion_matrix()

        # 5. Visualise
        viz = Visualizer(output_dir=self.log_dir, class_names=class_names)
        paths = {
            "curves": viz.plot_training_curves(history),
            "confusion_matrix": viz.plot_confusion_matrix(cm),
            "misclassified": viz.plot_misclassified(
                test_loader, trainer.model, self.device
            ),
        }

        # Save metrics JSON
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        ensure_dir(self.log_dir)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self._logger.info("Metrics saved → %s", metrics_path)

        self._logger.info("=" * 55)
        self._logger.info("Pipeline complete.")
        self._logger.info("=" * 55)

        return {"metrics": metrics, "history": history, "output_paths": paths}

    def infer(self, image_path: str, checkpoint: str) -> dict:
        """Run inference on a single image using a saved checkpoint.

        Args:
            image_path: Path to the image file.
            checkpoint: Path to a ``.pt`` checkpoint saved by ``Trainer``.

        Returns:
            Dictionary with ``class_name``, ``confidence``, and ``all_probs``.

        Raises:
            FileNotFoundError: If image or checkpoint does not exist.
        """
        from PIL import Image

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: '{image_path}'")

        # Load model — we need num_classes; try to infer from checkpoint
        ckpt = torch.load(checkpoint, map_location=self.device)
        # Determine num_classes from final layer shape
        state = ckpt["model_state_dict"]
        # Works for both Sequential(Dropout, Linear) and plain Linear heads
        fc_keys = [k for k in state if "fc" in k and "weight" in k]
        if not fc_keys:
            raise RuntimeError("Could not determine num_classes from checkpoint.")
        n_classes = state[fc_keys[-1]].shape[0]

        builder = ModelBuilder(num_classes=n_classes, mode=self.mode, pretrained=False)
        model = builder.build()
        model, _, _ = Trainer.load_checkpoint(model, checkpoint, self.device)
        model.eval()

        tb = TransformBuilder.from_config(self.config)
        transform = tb.val_transform()

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx   = int(probs.argmax())
        confidence = float(probs[pred_idx])

        result = {
            "class_idx":  pred_idx,
            "class_name": f"class_{pred_idx}",  # override with real names if available
            "confidence": confidence,
            "all_probs":  probs.tolist(),
        }
        self._logger.info(
            "Inference → class=%d  confidence=%.4f", pred_idx, confidence
        )
        return result


# ============================================================
# ENTRY POINT
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Classification — Section 2")
    parser.add_argument(
        "--mode",
        choices=["feature_extraction", "fine_tuning", "linear_probe"],
        default="fine_tuning",
        help="Transfer learning mode.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to existing dataset root (skips Roboflow download).",
    )
    parser.add_argument(
        "--infer",
        type=str,
        default=None,
        metavar="IMAGE_PATH",
        help="Run inference on a single image (requires --checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "best.pt"),
        help="Path to checkpoint for inference.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    pipeline = ClassificationPipeline(
        mode=args.mode,
        data_root=args.data_root,
    )
    if args.infer:
        result = pipeline.infer(args.infer, args.checkpoint)
        print(f"\nPredicted class: {result['class_name']}  "
              f"Confidence: {result['confidence']:.4f}")
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
