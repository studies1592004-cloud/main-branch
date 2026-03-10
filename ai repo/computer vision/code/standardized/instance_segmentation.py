from __future__ import annotations

"""
instance_segmentation.py
========================
Industry-Standard Instance Segmentation Pipeline — YOLOv8-seg / Mask R-CNN

Installation
------------
    pip install ultralytics roboflow opencv-python matplotlib numpy \
                scikit-learn seaborn tqdm torch torchvision

Usage
-----
    # Download dataset and train (YOLOv8-seg default)
    python instance_segmentation.py --mode train

    # Use Mask R-CNN backbone instead
    python instance_segmentation.py --mode train --backbone maskrcnn

    # Evaluate a trained model
    python instance_segmentation.py --mode evaluate --weights ./runs/segment/train/weights/best.pt

    # Run inference on image / video / directory
    python instance_segmentation.py --mode infer --source path/to/image.jpg
    python instance_segmentation.py --mode infer --source path/to/video.mp4

Author: CV Course — Section 5
Python: 3.9+  |  Ultralytics YOLOv8-seg / PyTorch Mask R-CNN
"""

# ============================================================
# GLOBAL CONFIGURATION — edit these before running
# ============================================================
ROBOFLOW_API_KEY: str   = "YOUR_API_KEY_HERE"   # free key at roboflow.com
ROBOFLOW_WORKSPACE: str = "roboflow-universe-projects"
ROBOFLOW_PROJECT: str   = "brain-tumor-detection-dqxdf"
ROBOFLOW_VERSION: int   = 1

BACKBONE: str               = "yolov8seg"  # "yolov8seg" or "maskrcnn"
YOLO_MODEL_SIZE: str        = "yolov8n-seg"
IMAGE_SIZE: int             = 640
BATCH_SIZE: int             = 8
NUM_EPOCHS: int             = 50
CONFIDENCE_THRESHOLD: float = 0.25
NMS_THRESHOLD: float        = 0.45
MASK_THRESHOLD: float       = 0.5    # binarise soft masks
DEVICE: str                 = "0"    # "0" = first GPU, "cpu" = CPU
SEED: int                   = 42
NUM_WORKERS: int            = 4
CHECKPOINT_DIR: str         = "./runs/segment"
LOG_DIR: str                = "./logs/instance_seg"

# ============================================================
# IMPORTS
# ============================================================
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    sys.exit(f"Ultralytics import failed: {exc}\nRun: pip install ultralytics")

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
logger = logging.getLogger("InstanceSeg")


def ensure_dir(path: str) -> None:
    """Create directory tree if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS: DatasetDownloader
# ============================================================
class DatasetDownloader:
    """Download a Roboflow instance segmentation dataset.

    For YOLOv8-seg, the layout is:
        <root>/data.yaml
        <root>/train/images/  ← RGB images
        <root>/train/labels/  ← YOLO polygon .txt files
        <root>/valid/images/
        <root>/valid/labels/

    For Mask R-CNN (COCO format):
        <root>/train/        ← images + _annotations.coco.json
        <root>/valid/

    Args:
        api_key:    Roboflow API key.
        workspace:  Roboflow workspace slug.
        project:    Project slug.
        version:    Dataset version number.
        dest_dir:   Local destination directory.
        fmt:        Roboflow export format (``"yolov8"`` or ``"coco-segmentation"``).

    Raises:
        RuntimeError: API key is placeholder or download fails.
        ImportError:  ``roboflow`` package not installed.
    """

    FORMAT_MAP: dict[str, str] = {
        "yolov8seg": "yolov8",
        "maskrcnn":  "coco-segmentation",
    }

    def __init__(
        self,
        api_key: str = ROBOFLOW_API_KEY,
        workspace: str = ROBOFLOW_WORKSPACE,
        project: str = ROBOFLOW_PROJECT,
        version: int = ROBOFLOW_VERSION,
        dest_dir: str = "./data/instance_seg",
        fmt: str = "yolov8",
    ) -> None:
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self.dest_dir = dest_dir
        self.fmt = fmt
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"DatasetDownloader(project='{self.project}', "
            f"version={self.version}, fmt='{self.fmt}')"
        )

    @classmethod
    def from_config(cls, config: dict) -> "DatasetDownloader":
        """Construct from a flat config dictionary."""
        backbone = config.get("backbone", "yolov8seg")
        fmt = cls.FORMAT_MAP.get(backbone, "yolov8")
        return cls(
            api_key=config.get("api_key", ROBOFLOW_API_KEY),
            workspace=config.get("workspace", ROBOFLOW_WORKSPACE),
            project=config.get("project", ROBOFLOW_PROJECT),
            version=config.get("version", ROBOFLOW_VERSION),
            dest_dir=config.get("dest_dir", "./data/instance_seg"),
            fmt=fmt,
        )

    def download(self) -> str:
        """Download and return local root path (or data.yaml path for YOLO).

        Returns:
            For YOLOv8 format: path to ``data.yaml``.
            For COCO format:   path to the dataset root directory.

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
                f"roboflow not found: {exc}\nRun: pip install roboflow"
            ) from exc

        self._logger.info(
            "Downloading %s/%s v%d (format='%s') …",
            self.workspace, self.project, self.version, self.fmt,
        )
        try:
            rf = Roboflow(api_key=self.api_key)
            dataset = (
                rf.workspace(self.workspace)
                .project(self.project)
                .version(self.version)
                .download(self.fmt, location=self.dest_dir)
            )
            root = dataset.location
        except Exception as exc:
            raise RuntimeError(
                f"Roboflow download failed: {exc}\n"
                "Check API key, workspace/project slugs, and internet connection."
            ) from exc

        if self.fmt == "yolov8":
            yaml_path = os.path.join(root, "data.yaml")
            if not os.path.isfile(yaml_path):
                for f in Path(root).rglob("data.yaml"):
                    yaml_path = str(f)
                    break
            self._logger.info("Dataset ready.  data.yaml → %s", yaml_path)
            return yaml_path
        else:
            self._logger.info("Dataset ready at: %s", root)
            return root


# ============================================================
# CLASS: MaskProcessor
# ============================================================
class MaskProcessor:
    """Convert between mask representations and compute mask-level metrics.

    Ultralytics YOLOv8-seg returns masks as float arrays in [0, 1].
    This class handles:
        - Binarisation (soft → binary mask)
        - Polygon ↔ binary mask conversion
        - Per-instance IoU between predicted and ground-truth masks
        - Mask area and bounding-box extraction

    Args:
        mask_threshold: Threshold for binarising soft masks.
        image_size:     Canonical (H, W) to resize masks to (if needed).
    """

    def __init__(
        self,
        mask_threshold: float = MASK_THRESHOLD,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.mask_threshold = mask_threshold
        self.image_size = image_size
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"MaskProcessor(mask_threshold={self.mask_threshold}, "
            f"image_size={self.image_size})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "MaskProcessor":
        """Construct from config dict."""
        return cls(
            mask_threshold=config.get("mask_threshold", MASK_THRESHOLD),
            image_size=config.get("image_size", IMAGE_SIZE),
        )

    def binarise(self, soft_mask: np.ndarray) -> np.ndarray:
        """Binarise a soft mask from the model output.

        Args:
            soft_mask: Float array in [0, 1], shape ``(H, W)``.

        Returns:
            uint8 binary mask, same shape (0 or 1).
        """
        return (soft_mask >= self.mask_threshold).astype(np.uint8)

    def polygon_to_mask(
        self,
        polygon: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Rasterise a polygon to a binary mask.

        Converts normalised (x, y) polygon coordinates (YOLO format, values
        in [0, 1]) to pixel coordinates, then fills with ``cv2.fillPoly``.

        Args:
            polygon: ``(N, 2)`` float array of normalised (x, y) points.
            height:  Output mask height in pixels.
            width:   Output mask width in pixels.

        Returns:
            Binary uint8 mask, shape ``(height, width)``.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        pts  = (polygon * np.array([width, height])).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=1)
        return mask

    def mask_to_polygon(
        self, binary_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract the largest contour from a binary mask as a polygon.

        Args:
            binary_mask: ``(H, W)`` uint8 binary mask.

        Returns:
            ``(N, 2)`` float array of normalised (x, y) coordinates, or
            ``None`` if no contour is found.
        """
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        h, w = binary_mask.shape
        pts  = largest.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= w
        pts[:, 1] /= h
        return pts

    def mask_iou(
        self, mask_a: np.ndarray, mask_b: np.ndarray
    ) -> float:
        """Compute IoU between two binary masks.

        IoU = |A ∩ B| / |A ∪ B|

        Args:
            mask_a: Binary mask ``(H, W)``.
            mask_b: Binary mask ``(H, W)``, same spatial size.

        Returns:
            IoU float in [0, 1].
        """
        if mask_a.shape != mask_b.shape:
            mask_b = cv2.resize(
                mask_b.astype(np.uint8),
                (mask_a.shape[1], mask_a.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        intersection = np.logical_and(mask_a, mask_b).sum()
        union        = np.logical_or(mask_a, mask_b).sum()
        return float(intersection / (union + 1e-8))

    def compute_mask_ap(
        self,
        pred_masks: list[np.ndarray],
        pred_scores: list[float],
        gt_masks: list[np.ndarray],
        iou_threshold: float = 0.5,
    ) -> float:
        """Compute Average Precision for instance segmentation at one IoU threshold.

        Follows the COCO evaluation protocol:
            1. Sort predictions by descending confidence.
            2. For each prediction, find the highest-IoU unmatched GT mask.
            3. TP if IoU >= threshold; else FP.
            4. Compute precision-recall curve; return area (all-points interpolation).

        Args:
            pred_masks:    List of binary predicted masks.
            pred_scores:   List of confidence scores (same order as masks).
            gt_masks:      List of binary ground-truth masks.
            iou_threshold: IoU threshold for TP assignment.

        Returns:
            AP float in [0, 1].
        """
        if not gt_masks:
            return 1.0 if not pred_masks else 0.0
        if not pred_masks:
            return 0.0

        order = np.argsort(pred_scores)[::-1]
        pred_masks  = [pred_masks[i]  for i in order]
        pred_scores = [pred_scores[i] for i in order]

        n_gt = len(gt_masks)
        matched_gt = np.zeros(n_gt, dtype=bool)
        tp = np.zeros(len(pred_masks))
        fp = np.zeros(len(pred_masks))

        for i, pm in enumerate(pred_masks):
            best_iou, best_j = 0.0, -1
            for j, gm in enumerate(gt_masks):
                if matched_gt[j]:
                    continue
                iou = self.mask_iou(pm, gm)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_threshold and best_j >= 0:
                tp[i] = 1
                matched_gt[best_j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall    = tp_cum / (n_gt + 1e-8)
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)

        mrec = np.concatenate([[0.], recall,    [1.]])
        mpre = np.concatenate([[0.], precision, [0.]])
        for k in range(len(mpre) - 2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def instance_stats(self, binary_mask: np.ndarray) -> dict:
        """Compute per-instance statistics from a binary mask.

        Returns bounding box (xyxy), area in pixels, centroid, and
        equivalent diameter (diameter of circle with same area as mask).

        Args:
            binary_mask: ``(H, W)`` uint8 binary mask for a single instance.

        Returns:
            Dict with keys ``area``, ``bbox_xyxy``, ``centroid``,
            ``equiv_diameter``.
        """
        area = int(binary_mask.sum())
        if area == 0:
            return {"area": 0, "bbox_xyxy": [0, 0, 0, 0],
                    "centroid": [0.0, 0.0], "equiv_diameter": 0.0}

        ys, xs = np.where(binary_mask > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cx = float(xs.mean())
        cy = float(ys.mean())
        equiv_diam = float(np.sqrt(4 * area / np.pi))

        return {
            "area":           area,
            "bbox_xyxy":      [x1, y1, x2, y2],
            "centroid":       [cx, cy],
            "equiv_diameter": equiv_diam,
        }


# ============================================================
# CLASS: Segmentor
# ============================================================
class Segmentor:
    """Wrap Ultralytics YOLOv8-seg for train / eval / infer.

    YOLOv8-seg extends the detection head with a lightweight mask branch:
        - 32 prototype masks (low-res 160×160 feature maps).
        - Per-detection linear combination coefficients (32 values).
        - Final mask = sigmoid(coefficients · prototypes), upsampled to input.

    This gives high-quality instance masks in real time without the two-stage
    overhead of Mask R-CNN, at the cost of some accuracy on complex scenes.

    Args:
        weights:  Path to ``.pt`` weights or model-size string (e.g. ``"yolov8n-seg"``).
        device:   ``"0"`` for first GPU, ``"cpu"`` for CPU.
        conf:     Confidence threshold for inference.
        iou:      NMS IoU threshold.

    Raises:
        FileNotFoundError: If a custom weights path is given but does not exist.
    """

    def __init__(
        self,
        weights: str = YOLO_MODEL_SIZE,
        device: str = DEVICE,
        conf: float = CONFIDENCE_THRESHOLD,
        iou: float = NMS_THRESHOLD,
    ) -> None:
        self.weights = weights
        self.device = device
        self.conf = conf
        self.iou = iou
        self._model: Optional[YOLO] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"Segmentor(weights='{self.weights}', device='{self.device}', "
            f"conf={self.conf}, iou={self.iou})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "Segmentor":
        """Construct from config dict."""
        return cls(
            weights=config.get("weights", YOLO_MODEL_SIZE),
            device=config.get("device", DEVICE),
            conf=config.get("confidence_threshold", CONFIDENCE_THRESHOLD),
            iou=config.get("nms_threshold", NMS_THRESHOLD),
        )

    def _load_model(self) -> YOLO:
        """Lazily load the YOLO model.

        Returns:
            Loaded ``YOLO`` instance.

        Raises:
            FileNotFoundError: Custom weights path does not exist.
        """
        if self._model is not None:
            return self._model

        if self.weights.endswith(".pt") and not os.path.isfile(self.weights):
            raise FileNotFoundError(
                f"Weights not found: '{self.weights}'\n"
                "Train first with --mode train or check --weights."
            )

        self._logger.info("Loading model: %s", self.weights)
        self._model = YOLO(self.weights)
        return self._model

    def train(
        self,
        data_yaml: str,
        epochs: int = NUM_EPOCHS,
        imgsz: int = IMAGE_SIZE,
        batch: int = BATCH_SIZE,
        project: str = CHECKPOINT_DIR,
        name: str = "train",
    ) -> str:
        """Fine-tune YOLOv8-seg on a custom dataset.

        Args:
            data_yaml: Path to Ultralytics ``data.yaml``.
            epochs:    Number of training epochs.
            imgsz:     Training image size.
            batch:     Batch size.
            project:   Root output directory.
            name:      Run subdirectory name.

        Returns:
            Path to the best weights file.

        Raises:
            FileNotFoundError: If ``data_yaml`` does not exist.
        """
        if not os.path.isfile(data_yaml):
            raise FileNotFoundError(
                f"data.yaml not found: '{data_yaml}'\n"
                "Download the dataset first."
            )

        model = self._load_model()
        self._logger.info(
            "Training %s  data=%s  epochs=%d  imgsz=%d",
            self.weights, data_yaml, epochs, imgsz,
        )

        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=project,
            name=name,
            seed=SEED,
            workers=NUM_WORKERS,
            exist_ok=True,
            verbose=True,
        )

        best_weights = os.path.join(project, name, "weights", "best.pt")
        self._logger.info("Training complete.  Best weights → %s", best_weights)
        return best_weights

    def infer_image(
        self,
        image: Union[str, np.ndarray],
        mask_processor: Optional[MaskProcessor] = None,
    ) -> list[dict]:
        """Run instance segmentation inference on a single image.

        Args:
            image:          File path or BGR ``np.ndarray``.
            mask_processor: Optional ``MaskProcessor`` for binary mask extraction.

        Returns:
            List of instance dicts, each with keys:
              ``box`` (xyxy), ``confidence``, ``class_id``, ``class_name``,
              ``mask`` (binary ``np.ndarray H×W``), ``mask_stats``.

        Raises:
            FileNotFoundError: Image path does not exist.
        """
        if isinstance(image, str) and not os.path.isfile(image):
            raise FileNotFoundError(f"Image not found: '{image}'")

        mp = mask_processor or MaskProcessor()
        model = self._load_model()

        results = model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        instances = []
        for r in results:
            if r.boxes is None:
                continue

            masks_np = (
                r.masks.data.cpu().numpy()
                if r.masks is not None
                else [None] * len(r.boxes)
            )

            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0].cpu())
                conf   = float(box.conf[0].cpu())
                xyxy   = box.xyxy[0].cpu().numpy().tolist()

                soft_mask   = masks_np[i] if masks_np[i] is not None else None
                binary_mask = mp.binarise(soft_mask) if soft_mask is not None else None
                stats       = mp.instance_stats(binary_mask) if binary_mask is not None else {}

                instances.append({
                    "box":        xyxy,
                    "confidence": conf,
                    "class_id":   cls_id,
                    "class_name": model.names[cls_id],
                    "mask":       binary_mask,
                    "mask_stats": stats,
                })

        self._logger.info(
            "infer_image: %d instances (conf≥%.2f)", len(instances), self.conf
        )
        return instances

    def infer_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> list[list[dict]]:
        """Run frame-by-frame instance segmentation on a video.

        Args:
            video_path:  Path to input video file.
            output_path: If provided, write annotated video here.
            max_frames:  Stop after this many frames (``None`` = all).

        Returns:
            List of per-frame instance lists.

        Raises:
            FileNotFoundError: Video file does not exist.
            RuntimeError:      Video cannot be opened.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: '{video_path}'")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: '{video_path}'")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit  = min(total, max_frames) if max_frames else total

        writer = None
        if output_path:
            ensure_dir(os.path.dirname(output_path) or ".")
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        mp  = MaskProcessor()
        viz = Visualizer()
        all_frames = []

        for _ in tqdm(range(limit), desc="Video inference", ncols=80):
            ret, frame = cap.read()
            if not ret:
                break

            instances = self.infer_image(frame, mp)
            all_frames.append(instances)

            if writer is not None:
                names = self._model.names if self._model else {}
                annotated = viz.draw_instances(frame, instances, names)
                writer.write(annotated)

        cap.release()
        if writer:
            writer.release()
            self._logger.info("Annotated video saved → %s", output_path)

        return all_frames

    def evaluate(
        self,
        data_yaml: str,
        weights: Optional[str] = None,
        split: str = "val",
    ) -> dict:
        """Run Ultralytics validation and return metric dict.

        Args:
            data_yaml: Path to ``data.yaml``.
            weights:   Weights to evaluate (defaults to ``self.weights``).
            split:     Dataset split (``"val"`` or ``"test"``).

        Returns:
            Dict with mask mAP50, mAP50-95, box mAP, precision, recall.

        Raises:
            FileNotFoundError: Weights or data_yaml not found.
        """
        eval_weights = weights or self.weights
        if eval_weights.endswith(".pt") and not os.path.isfile(eval_weights):
            raise FileNotFoundError(
                f"Weights not found: '{eval_weights}'\n"
                "Train first or check --weights."
            )
        if not os.path.isfile(data_yaml):
            raise FileNotFoundError(f"data.yaml not found: '{data_yaml}'")

        model = YOLO(eval_weights)
        results = model.val(
            data=data_yaml,
            split=split,
            imgsz=IMAGE_SIZE,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        try:
            metrics = {
                "mask_mAP50":    float(results.seg.map50),
                "mask_mAP50_95": float(results.seg.map),
                "box_mAP50":     float(results.box.map50),
                "precision":     float(results.seg.mp),
                "recall":        float(results.seg.mr),
            }
        except AttributeError as exc:
            self._logger.warning("Could not extract metrics: %s", exc)
            metrics = {"mask_mAP50": 0., "mask_mAP50_95": 0., "box_mAP50": 0.,
                       "precision": 0., "recall": 0.}

        self._logger.info(
            "Results: mask_mAP50=%.4f  mask_mAP50-95=%.4f  "
            "box_mAP50=%.4f  P=%.4f  R=%.4f",
            metrics["mask_mAP50"], metrics["mask_mAP50_95"],
            metrics["box_mAP50"],  metrics["precision"], metrics["recall"],
        )
        return metrics


# ============================================================
# CLASS: MaskRCNNSegmentor
# ============================================================
class MaskRCNNSegmentor:
    """Mask R-CNN instance segmentation — PyTorch / torchvision.

    Architecture overview:
        Stage 1 — Region Proposal Network (RPN):
            Slide an anchor grid over FPN feature maps.
            Predict objectness score + box delta for each anchor.
            NMS → top-K region proposals.

        Stage 2 — RoI Head:
            For each proposal: RoI-Align to fixed 7×7 feature map.
            Box head: FC → box regression + classification.
            Mask head: 4× conv → deconv → 28×28 binary mask per class.

    We use Mask R-CNN with a ResNet-50-FPN backbone pretrained on COCO,
    replacing the box and mask heads for the target number of classes.

    Args:
        num_classes: Number of foreground classes + 1 (background).
        pretrained:  Use COCO-pretrained backbone weights.
        device:      Training / inference device string.
        conf:        Score threshold for inference.

    Raises:
        ImportError: If ``torch`` or ``torchvision`` are not installed.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        device: str = "cpu",
        conf: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.device_str = device
        self.conf = conf
        self._model = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"MaskRCNNSegmentor(num_classes={self.num_classes}, "
            f"pretrained={self.pretrained}, device='{self.device_str}')"
        )

    @classmethod
    def from_config(cls, config: dict) -> "MaskRCNNSegmentor":
        """Construct from config dict."""
        return cls(
            num_classes=config.get("num_classes", 2),
            pretrained=config.get("pretrained", True),
            device=config.get("device", "cpu"),
            conf=config.get("confidence_threshold", CONFIDENCE_THRESHOLD),
        )

    def build(self):
        """Build and return the Mask R-CNN model.

        Returns:
            Configured ``maskrcnn_resnet50_fpn`` PyTorch model.

        Raises:
            ImportError: If torch/torchvision not installed.
        """
        try:
            import torch
            from torchvision.models.detection import (
                maskrcnn_resnet50_fpn,
                MaskRCNN_ResNet50_FPN_Weights,
            )
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        except ImportError as exc:
            raise ImportError(
                f"torch/torchvision not found: {exc}\n"
                "Run: pip install torch torchvision"
            ) from exc

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if self.pretrained else None
        model   = maskrcnn_resnet50_fpn(weights=weights)

        # Replace box predictor head
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features_box, self.num_classes
        )

        # Replace mask predictor head (28×28 binary masks per class)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_dim=256, num_classes=self.num_classes
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.device_str != "cpu" else "cpu"
        )
        model = model.to(device)
        self._device = device
        self._model  = model

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._logger.info(
            "Mask R-CNN (ResNet-50-FPN)  pretrained=%s  "
            "num_classes=%d  trainable=%s params",
            self.pretrained, self.num_classes, f"{trainable:,}",
        )
        return model

    def infer_image(self, image: np.ndarray) -> list[dict]:
        """Run Mask R-CNN inference on a BGR image.

        Args:
            image: BGR uint8 ``np.ndarray``.

        Returns:
            List of instance dicts with keys ``box``, ``confidence``,
            ``class_id``, ``mask``, ``mask_stats``.

        Raises:
            RuntimeError: If ``build()`` has not been called.
        """
        if self._model is None:
            raise RuntimeError("Call build() before infer_image().")

        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(f"torch not found: {exc}") from exc

        self._model.eval()
        rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_t = transforms.ToTensor()(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            predictions = self._model(img_t)

        pred    = predictions[0]
        boxes   = pred["boxes"].cpu().numpy()
        scores  = pred["scores"].cpu().numpy()
        labels  = pred["labels"].cpu().numpy()
        masks   = pred["masks"].cpu().numpy()[:, 0]  # (N, H, W) soft masks

        mp = MaskProcessor(mask_threshold=self.conf)
        instances = []
        for i, score in enumerate(scores):
            if score < self.conf:
                continue
            binary = mp.binarise(masks[i])
            instances.append({
                "box":        boxes[i].tolist(),
                "confidence": float(score),
                "class_id":   int(labels[i]),
                "class_name": str(labels[i]),
                "mask":       binary,
                "mask_stats": mp.instance_stats(binary),
            })

        self._logger.info("%d instances detected.", len(instances))
        return instances


# ============================================================
# CLASS: Evaluator
# ============================================================
class Evaluator:
    """Compute instance segmentation metrics: mask AP@50 and AP@50:95.

    Uses the ``MaskProcessor.compute_mask_ap`` method per IoU threshold,
    averaged over classes for multi-class datasets.

    Args:
        class_names:    Ordered list of class name strings.
        iou_thresholds: IoU thresholds for COCO-style evaluation.
    """

    def __init__(
        self,
        class_names: list[str],
        iou_thresholds: Optional[list[float]] = None,
    ) -> None:
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds or [
            round(t, 2) for t in np.arange(0.5, 1.0, 0.05)
        ]
        self.mask_proc = MaskProcessor()
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"Evaluator(n_classes={len(self.class_names)}, "
            f"iou_thresholds={self.iou_thresholds})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "Evaluator":
        """Construct from config dict."""
        return cls(
            class_names=config.get("class_names", []),
            iou_thresholds=config.get("iou_thresholds"),
        )

    def evaluate(
        self,
        all_pred_instances: list[list[dict]],
        all_gt_instances: list[list[dict]],
    ) -> dict:
        """Compute mask AP@50 and AP@50:95 across all images.

        Args:
            all_pred_instances: List of per-image prediction lists.
            all_gt_instances:   List of per-image GT lists.

        Returns:
            Dict with ``mask_AP50``, ``mask_AP50_95``, and ``per_class_AP50``.
        """
        n_cls = max(len(self.class_names), 1)
        ap50_per_class: dict[int, list[float]] = {c: [] for c in range(n_cls)}

        for pred_insts, gt_insts in zip(all_pred_instances, all_gt_instances):
            for cls_id in range(n_cls):
                pm = [p["mask"]       for p in pred_insts if p["class_id"] == cls_id and p.get("mask") is not None]
                ps = [p["confidence"] for p in pred_insts if p["class_id"] == cls_id and p.get("mask") is not None]
                gm = [g["mask"]       for g in gt_insts   if g["class_id"] == cls_id and g.get("mask") is not None]
                ap = self.mask_proc.compute_mask_ap(pm, ps, gm, iou_threshold=0.5)
                ap50_per_class[cls_id].append(ap)

        mean_ap50_cls = {
            c: float(np.mean(aps)) if aps else 0.0
            for c, aps in ap50_per_class.items()
        }
        map50 = float(np.mean(list(mean_ap50_cls.values())))

        ap_by_iou = []
        for iou_t in self.iou_thresholds:
            aps = []
            for pred_insts, gt_insts in zip(all_pred_instances, all_gt_instances):
                for cls_id in range(n_cls):
                    pm = [p["mask"]       for p in pred_insts if p["class_id"] == cls_id and p.get("mask") is not None]
                    ps = [p["confidence"] for p in pred_insts if p["class_id"] == cls_id and p.get("mask") is not None]
                    gm = [g["mask"]       for g in gt_insts   if g["class_id"] == cls_id and g.get("mask") is not None]
                    aps.append(self.mask_proc.compute_mask_ap(pm, ps, gm, iou_t))
            ap_by_iou.append(float(np.mean(aps)) if aps else 0.0)

        map5095 = float(np.mean(ap_by_iou))

        per_class_named = {
            self.class_names[c] if c < len(self.class_names) else f"class_{c}": v
            for c, v in mean_ap50_cls.items()
        }

        self._logger.info("Mask AP@50=%.4f  Mask AP@50:95=%.4f", map50, map5095)
        return {
            "mask_AP50":      map50,
            "mask_AP50_95":   map5095,
            "per_class_AP50": per_class_named,
        }


# ============================================================
# CLASS: Visualizer
# ============================================================
class Visualizer:
    """Draw instance masks, contours, and labels; plot metric charts.

    Args:
        output_dir: Directory for saved outputs.
        alpha:      Mask overlay opacity (0 = transparent, 1 = opaque).
    """

    PALETTE: list[tuple[int, int, int]] = [
        (255, 56,  56),  (255, 157, 151), (255, 112, 31),  (255, 178, 29),
        (207, 210, 49),  (72,  249, 10),  (146, 204, 23),  (61,  219, 134),
        (26,  147, 52),  (0,   212, 187), (44,  153, 168), (0,   194, 255),
        (52,  69,  147), (100, 115, 255), (0,   24,  236), (132, 56,  255),
        (82,  0,   133), (203, 56,  255), (255, 149, 200), (255, 55,  198),
    ]

    def __init__(
        self,
        output_dir: str = LOG_DIR,
        alpha: float = 0.45,
    ) -> None:
        self.output_dir = output_dir
        self.alpha = alpha
        ensure_dir(output_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"Visualizer(output_dir='{self.output_dir}', alpha={self.alpha})"

    @classmethod
    def from_config(cls, config: dict) -> "Visualizer":
        """Construct from config dict."""
        return cls(output_dir=config.get("log_dir", LOG_DIR))

    def draw_instances(
        self,
        image: np.ndarray,
        instances: list[dict],
        class_names: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw per-instance coloured masks, contours, and label boxes.

        Each instance gets a unique colour from ``PALETTE`` cycling by instance
        index (not class id), so overlapping instances of the same class are
        still visually distinct.

        Args:
            image:       BGR uint8 image.
            instances:   Output of ``Segmentor.infer_image``.
            class_names: Optional ``{id: name}`` dict.

        Returns:
            Annotated BGR image (copy of input).
        """
        vis = image.copy()

        for idx, inst in enumerate(instances):
            color = self.PALETTE[idx % len(self.PALETTE)]
            x1, y1, x2, y2 = [int(v) for v in inst["box"]]
            conf   = inst["confidence"]
            cls_id = inst["class_id"]
            name   = (
                class_names[cls_id]
                if class_names and cls_id in class_names
                else inst.get("class_name", str(cls_id))
            )

            mask = inst.get("mask")
            if mask is not None:
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                coloured = np.zeros_like(vis)
                coloured[mask > 0] = color
                vis = cv2.addWeighted(vis, 1.0, coloured, self.alpha, 0)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, color, 2)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
            label = f"{name} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y1, th + 4)
            cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw, ly + bl - 2), color, -1)
            cv2.putText(vis, label, (x1, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    def save_result(
        self,
        image: np.ndarray,
        instances: list[dict],
        class_names: Optional[dict],
        filename: str,
    ) -> str:
        """Draw instances and save annotated image.

        Args:
            image:       Original BGR image.
            instances:   Instance list from ``Segmentor.infer_image``.
            class_names: Optional class id → name mapping.
            filename:    Output filename.

        Returns:
            Full path to saved file.
        """
        annotated = self.draw_instances(image, instances, class_names)
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, annotated)
        self._logger.info("Saved → %s  (%d instances)", path, len(instances))
        return path

    def plot_instance_size_distribution(
        self,
        instances_by_file: dict[str, list[dict]],
        filename: str = "size_distribution.png",
    ) -> str:
        """Histogram of instance mask areas across a batch.

        Args:
            instances_by_file: Output of a batch inference run.
            filename:          Output filename.

        Returns:
            Full path to saved figure.
        """
        areas = []
        for insts in instances_by_file.values():
            for inst in insts:
                stats = inst.get("mask_stats", {})
                if stats.get("area", 0) > 0:
                    areas.append(stats["area"])

        if not areas:
            self._logger.warning("No instance areas to plot.")
            return ""

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(areas, bins=30, color="steelblue", edgecolor="white")
        ax.set_xlabel("Mask Area (pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Instance Size Distribution")
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Size distribution saved → %s", path)
        return path

    def plot_ap_bar(
        self,
        per_class_ap: dict[str, float],
        map50: float,
        filename: str = "mask_ap_per_class.png",
    ) -> str:
        """Horizontal bar chart of per-class mask AP@50.

        Args:
            per_class_ap: Dict of class name → AP@50 float.
            map50:        Overall mask mAP@50.
            filename:     Output filename.

        Returns:
            Full path to saved figure.
        """
        names  = list(per_class_ap.keys())
        values = list(per_class_ap.values())

        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.5)))
        bars = ax.barh(names, values, color="steelblue")
        ax.axvline(map50, color="red", linestyle="--", label=f"mAP50={map50:.3f}")
        ax.set_xlabel("Mask AP@50")
        ax.set_title("Per-class Mask AP@50")
        ax.set_xlim(0, 1)
        ax.legend()
        for bar, v in zip(bars, values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Mask AP bar chart saved → %s", path)
        return path


# ============================================================
# CLASS: InstanceSegPipeline  (orchestrator)
# ============================================================
class InstanceSegPipeline:
    """End-to-end orchestrator: download → train → evaluate → infer.

    Args:
        mode:       ``"train"``, ``"evaluate"``, or ``"infer"``.
        backbone:   ``"yolov8seg"`` or ``"maskrcnn"``.
        weights:    Weights path or YOLO model-size string.
        data_root:  If provided, skip Roboflow download.
        source:     Image / video path for inference.
        output_dir: Directory for all outputs.
        config:     Optional flat config dict.

    Example::

        pipeline = InstanceSegPipeline(mode="train", backbone="yolov8seg")
        pipeline.run()
    """

    def __init__(
        self,
        mode: str = "train",
        backbone: str = BACKBONE,
        weights: str = YOLO_MODEL_SIZE,
        data_root: Optional[str] = None,
        source: Optional[str] = None,
        output_dir: str = LOG_DIR,
        config: Optional[dict] = None,
    ) -> None:
        self.mode = mode
        self.backbone = backbone
        self.weights = weights
        self.data_root = data_root
        self.source = source
        self.output_dir = output_dir
        self.config = config or {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"InstanceSegPipeline(mode='{self.mode}', "
            f"backbone='{self.backbone}')"
        )

    @classmethod
    def from_config(cls, config: dict) -> "InstanceSegPipeline":
        """Construct from config dict."""
        return cls(
            mode=config.get("mode", "train"),
            backbone=config.get("backbone", BACKBONE),
            weights=config.get("weights", YOLO_MODEL_SIZE),
            data_root=config.get("data_root"),
            source=config.get("source"),
            output_dir=config.get("output_dir", LOG_DIR),
            config=config,
        )

    def _get_yaml(self) -> str:
        """Download or locate data.yaml."""
        if self.data_root:
            yaml_path = os.path.join(self.data_root, "data.yaml")
            if os.path.isfile(yaml_path):
                return yaml_path
            for f in Path(self.data_root).rglob("data.yaml"):
                return str(f)
            raise FileNotFoundError(
                f"data.yaml not found under '{self.data_root}'."
            )
        cfg = {**self.config, "backbone": self.backbone}
        downloader = DatasetDownloader.from_config(cfg)
        return downloader.download()

    def run(self) -> dict:
        """Execute pipeline according to ``self.mode``.

        Returns:
            Result dict (contents depend on mode).

        Raises:
            ValueError: Unknown mode.
        """
        self._logger.info("=" * 55)
        self._logger.info(
            "Instance Seg Pipeline  mode='%s'  backbone='%s'",
            self.mode, self.backbone,
        )
        self._logger.info("=" * 55)

        if self.backbone == "maskrcnn":
            return self._run_maskrcnn()

        segmentor = Segmentor(weights=self.weights, conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD)
        viz = Visualizer(output_dir=self.output_dir)

        if self.mode == "train":
            yaml_path    = self._get_yaml()
            best_weights = segmentor.train(data_yaml=yaml_path)
            metrics      = segmentor.evaluate(yaml_path, weights=best_weights)
            ensure_dir(self.output_dir)
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            return {"metrics": metrics, "best_weights": best_weights}

        elif self.mode == "evaluate":
            yaml_path = self._get_yaml()
            metrics   = segmentor.evaluate(yaml_path, weights=self.weights)
            return {"metrics": metrics}

        elif self.mode == "infer":
            if not self.source:
                raise ValueError("--source is required for infer mode.")

            try:
                model = segmentor._load_model()
                class_names = model.names
            except Exception:
                class_names = {}

            if os.path.isdir(self.source):
                results = {}
                for fname in sorted(os.listdir(self.source)):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    fpath = os.path.join(self.source, fname)
                    img   = cv2.imread(fpath)
                    if img is None:
                        continue
                    results[fname] = segmentor.infer_image(fpath)
                viz.plot_instance_size_distribution(results)
                return {"batch_instances": results}

            elif os.path.isfile(self.source):
                ext = os.path.splitext(self.source)[1].lower()
                if ext in {".mp4", ".avi", ".mov", ".mkv"}:
                    out_video   = os.path.join(self.output_dir, "output_annotated.mp4")
                    frame_insts = segmentor.infer_video(self.source, out_video)
                    return {"frame_instances": frame_insts, "output_video": out_video}
                else:
                    img = cv2.imread(self.source)
                    if img is None:
                        raise RuntimeError(f"Could not read image: '{self.source}'")
                    insts    = segmentor.infer_image(self.source)
                    out_path = viz.save_result(
                        img, insts, class_names,
                        filename=f"result_{Path(self.source).stem}.jpg",
                    )
                    return {"instances": insts, "output_image": out_path}
            else:
                raise FileNotFoundError(f"Source not found: '{self.source}'")

        else:
            raise ValueError(
                f"Unknown mode='{self.mode}'. Choose: train / evaluate / infer."
            )

    def _run_maskrcnn(self) -> dict:
        """Build Mask R-CNN and run inference (if mode='infer')."""
        segmentor = MaskRCNNSegmentor(
            num_classes=self.config.get("num_classes", 2),
            pretrained=True,
        )
        model = segmentor.build()
        self._logger.info("Mask R-CNN model built successfully.")

        if self.mode == "infer" and self.source:
            img = cv2.imread(self.source)
            if img is None:
                raise FileNotFoundError(f"Image not found: '{self.source}'")
            instances = segmentor.infer_image(img)
            viz = Visualizer(output_dir=self.output_dir)
            out = viz.save_result(
                img, instances, None,
                filename=f"maskrcnn_{Path(self.source).stem}.jpg",
            )
            return {"instances": instances, "output_image": out}

        self._logger.info(
            "Mask R-CNN training: pass COCO-format data to a standard "
            "torchvision training loop (see torchvision detection tutorial)."
        )
        return {"model": "MaskRCNN built", "num_classes": segmentor.num_classes}


# ============================================================
# ENTRY POINT
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instance Segmentation — Section 5")
    parser.add_argument("--mode", choices=["train", "evaluate", "infer"],
                        default="train")
    parser.add_argument("--backbone", choices=["yolov8seg", "maskrcnn"],
                        default=BACKBONE)
    parser.add_argument("--weights", type=str, default=YOLO_MODEL_SIZE)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=LOG_DIR)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    pipeline = InstanceSegPipeline(
        mode=args.mode,
        backbone=args.backbone,
        weights=args.weights,
        data_root=args.data_root,
        source=args.source,
        output_dir=args.output_dir,
    )
    result = pipeline.run()
    printable = {k: v for k, v in result.items() if isinstance(v, (str, dict, float, int))}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
