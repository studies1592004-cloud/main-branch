from __future__ import annotations

"""
object_detection.py
===================
Industry-Standard Object Detection Pipeline — YOLOv8 (Ultralytics)

Installation
------------
    pip install ultralytics roboflow opencv-python matplotlib numpy \
                scikit-learn seaborn tqdm

Usage
-----
    # Download dataset and train
    python object_detection.py --mode train

    # Evaluate a trained model
    python object_detection.py --mode evaluate --weights runs/detect/train/weights/best.pt

    # Inference on image / video / directory
    python object_detection.py --mode infer --source path/to/image.jpg
    python object_detection.py --mode infer --source path/to/video.mp4
    python object_detection.py --mode infer --source path/to/folder/

Author: CV Course — Section 3
Python: 3.9+  |  Ultralytics YOLOv8
"""

# ============================================================
# GLOBAL CONFIGURATION — edit these before running
# ============================================================
ROBOFLOW_API_KEY: str   = "YOUR_API_KEY_HERE"   # free key at roboflow.com
ROBOFLOW_WORKSPACE: str = "joseph-nelson"
ROBOFLOW_PROJECT: str   = "hard-hat-sample"
ROBOFLOW_VERSION: int   = 2

YOLO_MODEL_SIZE: str    = "yolov8n"   # n / s / m / l / x  (nano→xlarge)
IMAGE_SIZE: int         = 640
BATCH_SIZE: int         = 16
NUM_EPOCHS: int         = 50
CONFIDENCE_THRESHOLD: float = 0.25
NMS_THRESHOLD: float        = 0.45
DEVICE: str             = "0"         # "0" for GPU, "cpu" for CPU
SEED: int               = 42
NUM_WORKERS: int        = 4
CHECKPOINT_DIR: str     = "./runs/detect"
LOG_DIR: str            = "./logs/detection"

# Letterbox padding colour (grey — standard for YOLO)
LETTERBOX_PAD_COLOR: tuple[int, int, int] = (114, 114, 114)

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
import seaborn as sns

try:
    from ultralytics import YOLO
    from ultralytics.utils.metrics import DetMetrics
except ImportError as exc:
    sys.exit(
        f"Ultralytics import failed: {exc}\n"
        "Run:  pip install ultralytics"
    )

try:
    from sklearn.metrics import average_precision_score
except ImportError as exc:
    sys.exit(
        f"scikit-learn import failed: {exc}\n"
        "Run:  pip install scikit-learn"
    )

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
logger = logging.getLogger("Detection")


def ensure_dir(path: str) -> None:
    """Create directory tree if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS: DatasetDownloader
# ============================================================
class DatasetDownloader:
    """Download a Roboflow detection dataset in YOLOv8 format.

    The downloaded folder layout is what Ultralytics expects:
        <root>/data.yaml          ← class names + split paths
        <root>/train/images/      ← training images
        <root>/train/labels/      ← YOLO-format .txt labels
        <root>/valid/images/
        <root>/valid/labels/
        <root>/test/images/
        <root>/test/labels/

    Args:
        api_key:   Roboflow API key (free at roboflow.com).
        workspace: Roboflow workspace slug.
        project:   Project slug.
        version:   Dataset version number.
        dest_dir:  Local destination directory.

    Raises:
        ImportError:  If the ``roboflow`` package is not installed.
        RuntimeError: If the API key is the placeholder value or download fails.
    """

    def __init__(
        self,
        api_key: str = ROBOFLOW_API_KEY,
        workspace: str = ROBOFLOW_WORKSPACE,
        project: str = ROBOFLOW_PROJECT,
        version: int = ROBOFLOW_VERSION,
        dest_dir: str = "./data/detection",
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
            dest_dir=config.get("dest_dir", "./data/detection"),
        )

    def download(self) -> str:
        """Download dataset and return path to ``data.yaml``.

        Returns:
            Path to the ``data.yaml`` file consumed by Ultralytics.

        Raises:
            RuntimeError: API key is placeholder or download fails.
            ImportError:  ``roboflow`` is not installed.
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
            "Downloading %s/%s v%d (YOLOv8 format) …",
            self.workspace, self.project, self.version,
        )
        try:
            rf = Roboflow(api_key=self.api_key)
            dataset = (
                rf.workspace(self.workspace)
                .project(self.project)
                .version(self.version)
                .download("yolov8", location=self.dest_dir)
            )
            root = dataset.location
        except Exception as exc:
            raise RuntimeError(
                f"Roboflow download failed: {exc}\n"
                "Check your API key, workspace/project slugs, and internet connection."
            ) from exc

        yaml_path = os.path.join(root, "data.yaml")
        if not os.path.isfile(yaml_path):
            # Search one level deep
            for f in Path(root).rglob("data.yaml"):
                yaml_path = str(f)
                break

        if not os.path.isfile(yaml_path):
            raise RuntimeError(
                f"data.yaml not found under '{root}'.\n"
                "The Roboflow download may have used a different structure."
            )

        self._logger.info("Dataset ready.  data.yaml → %s", yaml_path)
        return yaml_path


# ============================================================
# CLASS: Preprocessor
# ============================================================
class Preprocessor:
    """Prepare a single image for YOLOv8 inference.

    Letterbox padding:
        Resize the image so that its longest edge equals ``target_size``
        while preserving aspect ratio.  Then pad the shorter edge with a
        constant colour on both sides to reach ``target_size × target_size``.

        This avides distortion and is the standard YOLO pre-processing step.
        The padding amounts (top/bottom, left/right) are stored so that
        bounding boxes can be mapped back to the original image coordinate
        system after inference.

    Args:
        target_size: Output spatial size (both height and width).
        pad_color:   RGB tuple for letterbox padding.
        normalize:   If True, divide pixel values by 255 and return float32.
    """

    def __init__(
        self,
        target_size: int = IMAGE_SIZE,
        pad_color: tuple[int, int, int] = LETTERBOX_PAD_COLOR,
        normalize: bool = True,
    ) -> None:
        self.target_size = target_size
        self.pad_color = pad_color
        self.normalize = normalize
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"Preprocessor(target_size={self.target_size}, "
            f"normalize={self.normalize})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "Preprocessor":
        """Construct from config dict."""
        return cls(
            target_size=config.get("image_size", IMAGE_SIZE),
            pad_color=config.get("pad_color", LETTERBOX_PAD_COLOR),
            normalize=config.get("normalize", True),
        )

    def letterbox(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Apply letterbox resize + pad to a single BGR image.

        Steps:
            1. Compute scale = target_size / max(H, W).
            2. Resize image with bilinear interpolation.
            3. Compute padding: dH = target_size - new_H, dW = target_size - new_W.
            4. Split padding evenly top/bottom, left/right.
            5. Apply constant-colour border.

        Args:
            img: BGR uint8 image of any size.

        Returns:
            Tuple of:
              - padded_img: Letterboxed image, shape (target, target, 3).
              - scale:      Scale factor applied (new_size / orig_size).
              - padding:    (pad_top, pad_left) pixel offsets for box remapping.
        """
        orig_h, orig_w = img.shape[:2]
        scale = self.target_size / max(orig_h, orig_w)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dh = self.target_size - new_h
        dw = self.target_size - new_w
        top    = dh // 2
        bottom = dh - top
        left   = dw // 2
        right  = dw - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=self.pad_color,
        )
        return padded, scale, (top, left)

    def preprocess(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Full preprocessing: letterbox → optional normalise → CHW.

        Args:
            img: BGR uint8 image.

        Returns:
            Tuple of:
              - tensor: float32 array shape ``(3, H, W)`` (normalised if requested).
              - scale:  Scale factor for coordinate remapping.
              - padding: (top, left) pad offsets for box remapping.
        """
        padded, scale, padding = self.letterbox(img)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32)
        if self.normalize:
            chw /= 255.0
        self._logger.debug(
            "Preprocessed: orig=%s → padded=%s  scale=%.4f  pad=%s",
            img.shape, padded.shape, scale, padding,
        )
        return chw, scale, padding

    def unletterbox_boxes(
        self,
        boxes_xyxy: np.ndarray,
        scale: float,
        padding: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> np.ndarray:
        """Map bounding boxes from letterboxed space back to original image.

        Args:
            boxes_xyxy: Array of shape ``(N, 4)`` in (x1,y1,x2,y2) format,
                        pixel coords in the letterboxed image.
            scale:      Scale factor returned by ``letterbox()``.
            padding:    (pad_top, pad_left) returned by ``letterbox()``.
            orig_shape: (H, W) of the original image.

        Returns:
            ``(N, 4)`` boxes clipped to original image bounds.
        """
        pad_top, pad_left = padding
        boxes = boxes_xyxy.copy().astype(np.float32)
        # Remove padding offset
        boxes[:, 0] -= pad_left
        boxes[:, 1] -= pad_top
        boxes[:, 2] -= pad_left
        boxes[:, 3] -= pad_top
        # Undo scaling
        boxes /= scale
        # Clip to image bounds
        orig_h, orig_w = orig_shape
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h)
        return boxes


# ============================================================
# CLASS: PostProcessor
# ============================================================
class PostProcessor:
    """Filter raw model detections by confidence and apply NMS.

    Ultralytics already runs NMS internally, so this class is useful when
    working with raw logits from a custom inference loop, or when you want
    to re-apply NMS with different thresholds without re-running the model.

    Args:
        conf_threshold: Minimum confidence score to keep a detection.
        nms_threshold:  IoU threshold for Non-Maximum Suppression.
    """

    def __init__(
        self,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"PostProcessor(conf={self.conf_threshold}, "
            f"nms_iou={self.nms_threshold})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "PostProcessor":
        """Construct from config dict."""
        return cls(
            conf_threshold=config.get("confidence_threshold", CONFIDENCE_THRESHOLD),
            nms_threshold=config.get("nms_threshold", NMS_THRESHOLD),
        )

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and an array of boxes.

        Args:
            box:   Shape ``(4,)`` in (x1,y1,x2,y2).
            boxes: Shape ``(N, 4)`` in (x1,y1,x2,y2).

        Returns:
            IoU values, shape ``(N,)``.
        """
        xi1 = np.maximum(box[0], boxes[:, 0])
        yi1 = np.maximum(box[1], boxes[:, 1])
        xi2 = np.minimum(box[2], boxes[:, 2])
        yi2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
        a1    = (box[2] - box[0]) * (box[3] - box[1])
        a2    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = a1 + a2 - inter + 1e-8
        return inter / union

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        """Class-aware Non-Maximum Suppression from scratch.

        Boxes belonging to different classes never suppress each other.
        Within each class, sort by confidence and greedily remove boxes
        with IoU > ``nms_threshold`` with a higher-confidence box.

        Args:
            boxes:     ``(N, 4)`` float32, xyxy format.
            scores:    ``(N,)``   float32 confidence scores.
            class_ids: ``(N,)``   int32   class index per detection.

        Returns:
            ``keep_indices`` — indices of retained boxes into the input arrays.
        """
        keep = []
        for cls_id in np.unique(class_ids):
            mask  = class_ids == cls_id
            idxs  = np.where(mask)[0]
            order = idxs[np.argsort(scores[idxs])[::-1]]

            while len(order) > 0:
                best = order[0]
                keep.append(best)
                if len(order) == 1:
                    break
                ious = self._iou(boxes[best], boxes[order[1:]])
                order = order[1:][ious <= self.nms_threshold]

        return np.array(keep, dtype=np.int32)

    def filter(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Confidence filter then NMS.

        Args:
            boxes:     ``(N, 4)`` xyxy float32.
            scores:    ``(N,)`` float32.
            class_ids: ``(N,)`` int32.

        Returns:
            Tuple of filtered (boxes, scores, class_ids).
        """
        # 1. Confidence filter
        mask = scores >= self.conf_threshold
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        if len(boxes) == 0:
            self._logger.debug("No detections above conf_threshold=%.2f", self.conf_threshold)
            return boxes, scores, class_ids

        # 2. NMS
        keep = self.nms(boxes, scores, class_ids)
        self._logger.debug(
            "PostProcess: %d → %d boxes after conf+NMS", mask.sum(), len(keep)
        )
        return boxes[keep], scores[keep], class_ids[keep]


# ============================================================
# CLASS: Detector
# ============================================================
class Detector:
    """Wrap Ultralytics YOLOv8 for train / evaluate / infer modes.

    Args:
        weights:    Path to ``.pt`` weights file, or model size string
                    like ``"yolov8n"`` (auto-downloads pretrained COCO weights).
        device:     ``"0"`` for first GPU, ``"cpu"`` for CPU.
        conf:       Confidence threshold for inference.
        iou:        NMS IoU threshold for inference.

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
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model: Optional[YOLO] = None

    def __repr__(self) -> str:
        return (
            f"Detector(weights='{self.weights}', device='{self.device}', "
            f"conf={self.conf}, iou={self.iou})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "Detector":
        """Construct from config dict."""
        return cls(
            weights=config.get("weights", YOLO_MODEL_SIZE),
            device=config.get("device", DEVICE),
            conf=config.get("confidence_threshold", CONFIDENCE_THRESHOLD),
            iou=config.get("nms_threshold", NMS_THRESHOLD),
        )

    def _load_model(self) -> YOLO:
        """Load or lazily initialise the YOLO model.

        Returns:
            Loaded ``YOLO`` instance.

        Raises:
            FileNotFoundError: If a custom weights path does not exist.
        """
        if self._model is not None:
            return self._model

        # If it looks like a file path (not a model size string), validate it
        if self.weights.endswith(".pt") and not os.path.isfile(self.weights):
            raise FileNotFoundError(
                f"Weights file not found: '{self.weights}'\n"
                "Train first with --mode train, or check the --weights argument."
            )

        self._logger.info("Loading YOLO model: %s", self.weights)
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
        """Fine-tune YOLOv8 on a custom dataset.

        Args:
            data_yaml: Path to the Ultralytics-format ``data.yaml`` file.
            epochs:    Number of training epochs.
            imgsz:     Training image size.
            batch:     Batch size per GPU.
            project:   Root directory for run outputs.
            name:      Subdirectory name for this run.

        Returns:
            Path to the best weights file (``<project>/<name>/weights/best.pt``).

        Raises:
            FileNotFoundError: If ``data_yaml`` does not exist.
        """
        if not os.path.isfile(data_yaml):
            raise FileNotFoundError(
                f"data.yaml not found: '{data_yaml}'\n"
                "Download the dataset first with DatasetDownloader.download()."
            )

        model = self._load_model()
        self._logger.info(
            "Training %s  data=%s  epochs=%d  imgsz=%d  batch=%d",
            self.weights, data_yaml, epochs, imgsz, batch,
        )

        results = model.train(
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
        if os.path.isfile(best_weights):
            self._logger.info("Training complete. Best weights → %s", best_weights)
        else:
            self._logger.warning(
                "Training finished but best.pt not found at expected path: %s",
                best_weights,
            )
        return best_weights

    def infer_image(
        self, image: Union[str, np.ndarray]
    ) -> list[dict]:
        """Run inference on a single image.

        Args:
            image: File path string or BGR ``np.ndarray``.

        Returns:
            List of detection dicts, each with keys:
              ``box`` (xyxy list), ``confidence``, ``class_id``, ``class_name``.
        """
        model = self._load_model()

        if isinstance(image, str) and not os.path.isfile(image):
            raise FileNotFoundError(f"Image not found: '{image}'")

        results = model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                detections.append({
                    "box":        box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": float(box.conf[0].cpu()),
                    "class_id":   int(box.cls[0].cpu()),
                    "class_name": model.names[int(box.cls[0].cpu())],
                })

        self._logger.info(
            "infer_image: %d detections (conf≥%.2f)", len(detections), self.conf
        )
        return detections

    def infer_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> list[list[dict]]:
        """Run frame-by-frame inference on a video file.

        Reads each frame with OpenCV, runs ``infer_image``, optionally
        writes an annotated output video.

        Args:
            video_path:  Path to input video.
            output_path: If provided, write annotated frames to this path.
            max_frames:  Stop after this many frames (``None`` = all).

        Returns:
            List of per-frame detection lists (same format as ``infer_image``).

        Raises:
            FileNotFoundError: If ``video_path`` does not exist.
            RuntimeError:      If the video cannot be opened.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: '{video_path}'")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: '{video_path}'\n"
                "Check the file is a valid video format supported by OpenCV."
            )

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit  = min(total, max_frames) if max_frames else total
        self._logger.info(
            "Video: %s  %dx%d  %.1f fps  %d frames", video_path, width, height, fps, limit
        )

        writer = None
        if output_path:
            ensure_dir(os.path.dirname(output_path) or ".")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_frame_detections = []
        visualizer = Visualizer()

        for frame_idx in tqdm(range(limit), desc="Video inference", ncols=80):
            ret, frame = cap.read()
            if not ret:
                self._logger.warning("Could not read frame %d — stopping.", frame_idx)
                break

            dets = self.infer_image(frame)
            all_frame_detections.append(dets)

            if writer is not None:
                # Build class-name map from model
                class_names = self._model.names if self._model else {}
                annotated = visualizer.draw_detections(frame, dets, class_names)
                writer.write(annotated)

        cap.release()
        if writer:
            writer.release()
            self._logger.info("Annotated video saved → %s", output_path)

        return all_frame_detections

    def infer_batch(
        self,
        image_dir: str,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> dict[str, list[dict]]:
        """Run inference on all images in a directory.

        Args:
            image_dir:  Path to folder containing images.
            extensions: Accepted file extensions (case-insensitive).

        Returns:
            Dict mapping filename → list of detection dicts.

        Raises:
            FileNotFoundError: If the directory does not exist.
            RuntimeError:      If no matching images are found.
        """
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: '{image_dir}'")

        image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if os.path.splitext(f)[1].lower() in extensions
        ]

        if not image_paths:
            raise RuntimeError(
                f"No images with extensions {extensions} found in '{image_dir}'."
            )

        self._logger.info(
            "Batch inference: %d images in '%s'", len(image_paths), image_dir
        )

        results = {}
        for path in tqdm(image_paths, desc="Batch", ncols=80):
            results[os.path.basename(path)] = self.infer_image(path)

        total_dets = sum(len(v) for v in results.values())
        self._logger.info(
            "Batch complete: %d images, %d total detections.", len(results), total_dets
        )
        return results

    def evaluate(
        self,
        data_yaml: str,
        weights: Optional[str] = None,
        split: str = "test",
    ) -> dict:
        """Run Ultralytics validation and return metric dict.

        Args:
            data_yaml: Path to ``data.yaml``.
            weights:   Path to weights to evaluate (defaults to ``self.weights``).
            split:     Dataset split to evaluate on (``"val"`` or ``"test"``).

        Returns:
            Dict with mAP50, mAP50-95, precision, recall, and per-class AP.

        Raises:
            FileNotFoundError: If weights or data_yaml not found.
        """
        eval_weights = weights or self.weights
        if eval_weights.endswith(".pt") and not os.path.isfile(eval_weights):
            raise FileNotFoundError(
                f"Weights file not found: '{eval_weights}'\n"
                "Train first or supply a valid --weights path."
            )
        if not os.path.isfile(data_yaml):
            raise FileNotFoundError(f"data.yaml not found: '{data_yaml}'")

        model = YOLO(eval_weights)
        self._logger.info(
            "Evaluating %s on split='%s'", eval_weights, split
        )

        val_results = model.val(
            data=data_yaml,
            split=split,
            imgsz=IMAGE_SIZE,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        # Extract metrics safely
        try:
            map50    = float(val_results.box.map50)
            map5095  = float(val_results.box.map)
            prec     = float(val_results.box.mp)
            rec      = float(val_results.box.mr)
            per_class_ap = val_results.box.ap_class_index.tolist()
        except AttributeError as exc:
            self._logger.warning(
                "Could not extract some metrics: %s. "
                "Ultralytics API may have changed.", exc
            )
            map50 = map5095 = prec = rec = 0.0
            per_class_ap = []

        metrics = {
            "mAP50":         map50,
            "mAP50-95":      map5095,
            "precision":     prec,
            "recall":        rec,
            "per_class_ap":  per_class_ap,
        }

        self._logger.info(
            "Results: mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
            map50, map5095, prec, rec,
        )
        return metrics


# ============================================================
# CLASS: Evaluator
# ============================================================
class Evaluator:
    """Compute mAP@50, mAP@50:95, and per-class AP from prediction files.

    This class provides a transparent, from-scratch AP computation
    to complement the Ultralytics built-in evaluator — useful for
    custom datasets or when you need auditable metric computation.

    Args:
        class_names: Ordered list of class name strings.
        iou_thresholds: IoU thresholds for COCO-style mAP.
    """

    def __init__(
        self,
        class_names: list[str],
        iou_thresholds: Optional[list[float]] = None,
    ) -> None:
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds or [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]
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

    @staticmethod
    def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Vectorised IoU between one box and multiple boxes.

        Args:
            box:   Shape ``(4,)`` xyxy.
            boxes: Shape ``(N, 4)`` xyxy.

        Returns:
            IoU array, shape ``(N,)``.
        """
        xi1 = np.maximum(box[0], boxes[:, 0])
        yi1 = np.maximum(box[1], boxes[:, 1])
        xi2 = np.minimum(box[2], boxes[:, 2])
        yi2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
        a1    = (box[2] - box[0]) * (box[3] - box[1])
        a2    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (a1 + a2 - inter + 1e-8)

    def compute_ap_at_iou(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        iou_threshold: float,
    ) -> dict[int, float]:
        """Compute per-class AP at a single IoU threshold.

        Algorithm:
            For each class:
            1. Sort predictions by confidence (descending).
            2. For each prediction, match to the highest-IoU GT box
               that has not yet been matched.
            3. A prediction is TP if IoU >= threshold; else FP.
            4. Compute precision-recall curve.
            5. AP = area under the PR curve (all-points interpolation).

        Args:
            pred_boxes:   ``(N, 4)`` predicted boxes, xyxy.
            pred_scores:  ``(N,)``   confidence scores.
            pred_classes: ``(N,)``   predicted class ids.
            gt_boxes:     ``(M, 4)`` ground-truth boxes, xyxy.
            gt_classes:   ``(M,)``   ground-truth class ids.
            iou_threshold: IoU threshold for TP/FP assignment.

        Returns:
            Dict mapping class_id → AP float.
        """
        class_aps: dict[int, float] = {}

        for cls_id in range(len(self.class_names)):
            pred_mask = pred_classes == cls_id
            gt_mask   = gt_classes   == cls_id

            p_boxes   = pred_boxes[pred_mask]
            p_scores  = pred_scores[pred_mask]
            g_boxes   = gt_boxes[gt_mask]

            n_gt = len(g_boxes)
            if n_gt == 0:
                class_aps[cls_id] = 0.0 if len(p_boxes) > 0 else float("nan")
                continue
            if len(p_boxes) == 0:
                class_aps[cls_id] = 0.0
                continue

            # Sort by descending confidence
            order = np.argsort(p_scores)[::-1]
            p_boxes  = p_boxes[order]

            matched_gt = np.zeros(n_gt, dtype=bool)
            tp = np.zeros(len(p_boxes))
            fp = np.zeros(len(p_boxes))

            for i, pb in enumerate(p_boxes):
                if n_gt == 0:
                    fp[i] = 1
                    continue
                ious = self._box_iou(pb, g_boxes)
                best_iou_idx = ious.argmax()
                best_iou     = ious[best_iou_idx]

                if best_iou >= iou_threshold and not matched_gt[best_iou_idx]:
                    tp[i] = 1
                    matched_gt[best_iou_idx] = True
                else:
                    fp[i] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall    = tp_cum / (n_gt + 1e-8)
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)

            # All-points interpolation
            mrec = np.concatenate([[0.0], recall,    [1.0]])
            mpre = np.concatenate([[0.0], precision, [0.0]])
            for j in range(len(mpre) - 2, -1, -1):
                mpre[j] = max(mpre[j], mpre[j + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap  = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
            class_aps[cls_id] = ap

        return class_aps

    def compute_map(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> dict:
        """Compute mAP@50 and mAP@50:95 (COCO style).

        Args:
            pred_boxes:   ``(N, 4)`` xyxy.
            pred_scores:  ``(N,)`` confidence.
            pred_classes: ``(N,)`` class ids.
            gt_boxes:     ``(M, 4)`` xyxy.
            gt_classes:   ``(M,)`` class ids.

        Returns:
            Dict with ``mAP50``, ``mAP50-95``, and per-class AP at 0.50.
        """
        # AP at IoU = 0.50
        ap50_per_class = self.compute_ap_at_iou(
            pred_boxes, pred_scores, pred_classes,
            gt_boxes, gt_classes, iou_threshold=0.50,
        )
        valid_ap50 = [v for v in ap50_per_class.values() if not np.isnan(v)]
        map50 = float(np.mean(valid_ap50)) if valid_ap50 else 0.0

        # AP averaged over IoU thresholds 0.50 : 0.05 : 0.95
        all_maps = []
        for iou_t in self.iou_thresholds:
            ap_cls = self.compute_ap_at_iou(
                pred_boxes, pred_scores, pred_classes,
                gt_boxes, gt_classes, iou_threshold=iou_t,
            )
            valid = [v for v in ap_cls.values() if not np.isnan(v)]
            all_maps.append(float(np.mean(valid)) if valid else 0.0)

        map5095 = float(np.mean(all_maps))

        self._logger.info(
            "mAP@50=%.4f  mAP@50:95=%.4f", map50, map5095
        )
        return {
            "mAP50":        map50,
            "mAP50-95":     map5095,
            "per_class_ap50": {
                self.class_names[k]: v
                for k, v in ap50_per_class.items()
                if not np.isnan(v)
            },
        }


# ============================================================
# CLASS: Visualizer
# ============================================================
class Visualizer:
    """Draw bounding boxes, save result images, and plot metric charts.

    Args:
        output_dir:  Directory for saved figures.
        line_width:  Bounding box line thickness.
        font_scale:  OpenCV font scale for labels.
    """

    # 20 visually distinct colours (BGR)
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
        line_width: int = 2,
        font_scale: float = 0.55,
    ) -> None:
        self.output_dir = output_dir
        self.line_width = line_width
        self.font_scale = font_scale
        ensure_dir(output_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"Visualizer(output_dir='{self.output_dir}')"

    @classmethod
    def from_config(cls, config: dict) -> "Visualizer":
        """Construct from config dict."""
        return cls(output_dir=config.get("log_dir", LOG_DIR))

    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[dict],
        class_names: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw bounding boxes with class name and confidence on image.

        Args:
            image:       BGR uint8 image.
            detections:  List of detection dicts (from ``Detector.infer_image``).
            class_names: Optional ``{id: name}`` dict.  If ``None``, uses
                         the ``class_name`` field from each detection dict.

        Returns:
            Annotated BGR image.
        """
        vis = image.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cls_id   = det["class_id"]
            conf     = det["confidence"]
            name     = (
                class_names[cls_id]
                if class_names and cls_id in class_names
                else det.get("class_name", str(cls_id))
            )
            color = self.PALETTE[cls_id % len(self.PALETTE)]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, self.line_width)

            label = f"{name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            lbl_y = max(y1, th + 4)
            cv2.rectangle(vis, (x1, lbl_y - th - 4), (x1 + tw, lbl_y + baseline - 2), color, -1)
            cv2.putText(
                vis, label, (x1, lbl_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
        return vis

    def save_result(
        self,
        image: np.ndarray,
        detections: list[dict],
        class_names: Optional[dict],
        filename: str,
    ) -> str:
        """Draw detections and save annotated image to disk.

        Args:
            image:       Original BGR image.
            detections:  Detection list from ``Detector.infer_image``.
            class_names: Optional class id → name mapping.
            filename:    Output filename (saved inside ``output_dir``).

        Returns:
            Full path to the saved file.
        """
        annotated = self.draw_detections(image, detections, class_names)
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, annotated)
        self._logger.info(
            "Saved annotated image → %s  (%d detections)", path, len(detections)
        )
        return path

    def plot_class_distribution(
        self,
        detections_by_file: dict[str, list[dict]],
        class_names: dict[int, str],
        filename: str = "class_distribution.png",
    ) -> str:
        """Bar chart of detection counts per class across a batch.

        Args:
            detections_by_file: Output of ``Detector.infer_batch``.
            class_names:        ``{id: name}`` mapping.
            filename:           Output filename.

        Returns:
            Full path to the saved figure.
        """
        counts: dict[str, int] = {}
        for dets in detections_by_file.values():
            for d in dets:
                name = class_names.get(d["class_id"], str(d["class_id"]))
                counts[name] = counts.get(name, 0) + 1

        if not counts:
            self._logger.warning("No detections to plot.")
            return ""

        names = list(counts.keys())
        values = [counts[n] for n in names]
        colors = [
            f"#{self.PALETTE[i % len(self.PALETTE)][2]:02x}"
            f"{self.PALETTE[i % len(self.PALETTE)][1]:02x}"
            f"{self.PALETTE[i % len(self.PALETTE)][0]:02x}"
            for i in range(len(names))
        ]

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
        ax.bar(names, values, color=colors)
        ax.set_xlabel("Class"); ax.set_ylabel("Detection count")
        ax.set_title("Detection Count per Class")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Class distribution saved → %s", path)
        return path

    def plot_map_bar(
        self,
        per_class_ap: dict[str, float],
        map50: float,
        filename: str = "map_per_class.png",
    ) -> str:
        """Horizontal bar chart of per-class AP alongside overall mAP.

        Args:
            per_class_ap: Dict mapping class name → AP@0.50.
            map50:        Overall mAP@0.50.
            filename:     Output filename.

        Returns:
            Full path to the saved figure.
        """
        names  = list(per_class_ap.keys())
        values = list(per_class_ap.values())

        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.45)))
        bars = ax.barh(names, values, color="steelblue")
        ax.axvline(map50, color="red", linestyle="--", label=f"mAP50={map50:.3f}")
        ax.set_xlabel("AP@0.50"); ax.set_title("Per-class AP@0.50")
        ax.set_xlim(0, 1); ax.legend()
        for bar, v in zip(bars, values):
            ax.text(
                v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8,
            )
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("mAP bar chart saved → %s", path)
        return path


# ============================================================
# CLASS: DetectionPipeline  (orchestrator)
# ============================================================
class DetectionPipeline:
    """End-to-end orchestrator: download → train → evaluate → infer.

    Args:
        mode:           ``"train"``, ``"evaluate"``, or ``"infer"``.
        weights:        Weights path or YOLO model-size string.
        data_root:      If provided, skip Roboflow download.
        source:         Path for inference (image / video / directory).
        output_dir:     Root output directory.
        config:         Optional flat config dict.

    Example::

        pipeline = DetectionPipeline(mode="train")
        pipeline.run()
    """

    def __init__(
        self,
        mode: str = "train",
        weights: str = YOLO_MODEL_SIZE,
        data_root: Optional[str] = None,
        source: Optional[str] = None,
        output_dir: str = LOG_DIR,
        config: Optional[dict] = None,
    ) -> None:
        self.mode = mode
        self.weights = weights
        self.data_root = data_root
        self.source = source
        self.output_dir = output_dir
        self.config = config or {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"DetectionPipeline(mode='{self.mode}', weights='{self.weights}')"
        )

    @classmethod
    def from_config(cls, config: dict) -> "DetectionPipeline":
        """Construct from config dict."""
        return cls(
            mode=config.get("mode", "train"),
            weights=config.get("weights", YOLO_MODEL_SIZE),
            data_root=config.get("data_root"),
            source=config.get("source"),
            output_dir=config.get("output_dir", LOG_DIR),
            config=config,
        )

    def _get_yaml(self) -> str:
        """Download dataset or locate existing data.yaml."""
        if self.data_root:
            # User supplied an existing dataset root
            yaml_path = os.path.join(self.data_root, "data.yaml")
            if not os.path.isfile(yaml_path):
                for f in Path(self.data_root).rglob("data.yaml"):
                    yaml_path = str(f)
                    break
            if not os.path.isfile(yaml_path):
                raise FileNotFoundError(
                    f"data.yaml not found under '{self.data_root}'."
                )
            return yaml_path

        downloader = DatasetDownloader.from_config(self.config)
        return downloader.download()

    def run(self) -> dict:
        """Execute the pipeline according to ``self.mode``.

        Returns:
            Result dictionary (contents depend on mode).
        """
        self._logger.info("=" * 55)
        self._logger.info("Detection Pipeline  mode='%s'", self.mode)
        self._logger.info("=" * 55)

        detector = Detector(
            weights=self.weights,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_THRESHOLD,
        )
        viz = Visualizer(output_dir=self.output_dir)

        if self.mode == "train":
            return self._run_train(detector, viz)
        elif self.mode == "evaluate":
            return self._run_evaluate(detector, viz)
        elif self.mode == "infer":
            return self._run_infer(detector, viz)
        else:
            raise ValueError(
                f"Unknown mode='{self.mode}'. Choose: train / evaluate / infer."
            )

    def _run_train(self, detector: Detector, viz: Visualizer) -> dict:
        """Download data, train, then evaluate on test split."""
        yaml_path = self._get_yaml()
        best_weights = detector.train(data_yaml=yaml_path)

        # Evaluate best weights
        self._logger.info("Evaluating best weights on test split …")
        metrics = detector.evaluate(yaml_path, weights=best_weights, split="val")

        # Save metrics JSON
        ensure_dir(self.output_dir)
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self._logger.info("Metrics → %s", metrics_path)

        # Plot per-class AP bar chart
        if metrics.get("per_class_ap"):
            viz.plot_map_bar(metrics["per_class_ap"], metrics["mAP50"])

        return {"metrics": metrics, "best_weights": best_weights}

    def _run_evaluate(self, detector: Detector, viz: Visualizer) -> dict:
        """Run evaluation only (requires --weights)."""
        yaml_path = self._get_yaml()
        metrics = detector.evaluate(yaml_path, weights=self.weights, split="val")

        if metrics.get("per_class_ap"):
            viz.plot_map_bar(metrics["per_class_ap"], metrics["mAP50"])

        return {"metrics": metrics}

    def _run_infer(self, detector: Detector, viz: Visualizer) -> dict:
        """Infer on image, video, or directory (requires --source)."""
        if not self.source:
            raise ValueError(
                "Inference mode requires --source "
                "(path to image, video, or directory)."
            )

        src = self.source
        class_names = None  # will be populated from model if available

        try:
            model = detector._load_model()
            class_names = model.names
        except Exception:
            pass

        if os.path.isdir(src):
            results = detector.infer_batch(src)
            if class_names:
                viz.plot_class_distribution(results, class_names)
            return {"batch_detections": results}

        elif os.path.isfile(src):
            ext = os.path.splitext(src)[1].lower()
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

            if ext in video_exts:
                out_video = os.path.join(self.output_dir, "output_annotated.mp4")
                frame_dets = detector.infer_video(src, output_path=out_video)
                return {"frame_detections": frame_dets, "output_video": out_video}
            else:
                img = cv2.imread(src)
                if img is None:
                    raise RuntimeError(
                        f"Could not read image: '{src}'\n"
                        "Check the file is a valid image."
                    )
                dets = detector.infer_image(src)
                out_path = viz.save_result(
                    img, dets, class_names,
                    filename=f"result_{Path(src).stem}.jpg",
                )
                return {"detections": dets, "output_image": out_path}
        else:
            raise FileNotFoundError(
                f"Source not found: '{src}'\n"
                "Pass a valid image path, video path, or directory."
            )


# ============================================================
# ENTRY POINT
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object Detection — Section 3")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "infer"],
        default="train",
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=YOLO_MODEL_SIZE,
        help="YOLO weights: model-size string or path to .pt file.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to existing dataset root (skips Roboflow download).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input for inference: image / video / directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=LOG_DIR,
        help="Directory for output files.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    pipeline = DetectionPipeline(
        mode=args.mode,
        weights=args.weights,
        data_root=args.data_root,
        source=args.source,
        output_dir=args.output_dir,
    )
    result = pipeline.run()
    print(json.dumps(
        {k: v for k, v in result.items() if isinstance(v, (str, dict, float, int))},
        indent=2,
    ))


if __name__ == "__main__":
    main()
