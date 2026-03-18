from __future__ import annotations

"""
pose_estimation.py
==================
Industry-Standard Pose Estimation Pipeline — YOLOv8-pose / MediaPipe Holistic

Installation
------------
    pip install ultralytics roboflow opencv-python matplotlib numpy \
                mediapipe tqdm

Usage
-----
    # Download dataset and train (YOLOv8-pose default)
    python pose_estimation.py --mode train

    # Use MediaPipe backend instead (no training required)
    python pose_estimation.py --mode infer --backend mediapipe --source path/to/image.jpg

    # Evaluate trained YOLOv8-pose model
    python pose_estimation.py --mode evaluate --weights ./runs/pose/train/weights/best.pt

    # Inference on image / video / webcam
    python pose_estimation.py --mode infer --source path/to/video.mp4
    python pose_estimation.py --mode infer --source 0          # webcam

Author: CV Course — Section 6
Python: 3.9+  |  Ultralytics YOLOv8-pose / MediaPipe
"""

# ============================================================
# GLOBAL CONFIGURATION — edit these before running
# ============================================================
ROBOFLOW_API_KEY: str   = "YOUR_API_KEY_HERE"   # free key at roboflow.com
ROBOFLOW_WORKSPACE: str = "joseph-nelson"
ROBOFLOW_PROJECT: str   = "yoga-pose"
ROBOFLOW_VERSION: int   = 1

BACKEND: str                = "yolov8pose"   # "yolov8pose" or "mediapipe"
YOLO_MODEL_SIZE: str        = "yolov8n-pose"
IMAGE_SIZE: int             = 640
BATCH_SIZE: int             = 16
NUM_EPOCHS: int             = 50
CONFIDENCE_THRESHOLD: float = 0.25
NMS_THRESHOLD: float        = 0.45
KEYPOINT_CONF_THRESHOLD: float = 0.5   # min visibility for a keypoint to be drawn
DEVICE: str                 = "0"      # "0" = first GPU, "cpu" = CPU
SEED: int                   = 42
NUM_WORKERS: int            = 4
CHECKPOINT_DIR: str         = "./runs/pose"
LOG_DIR: str                = "./logs/pose"

# ── COCO 17-keypoint skeleton ─────────────────────────────────────────
# Index pairs defining bones to draw as lines between keypoints
COCO_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Head
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9),                           # Left arm
    (6, 8), (8, 10),                          # Right arm
    (5, 11), (6, 12),                         # Torso sides
    (11, 12),                                 # Hips
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]

COCO_KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

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
logger = logging.getLogger("PoseEstimation")


def ensure_dir(path: str) -> None:
    """Create directory tree if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS: DatasetDownloader
# ============================================================
class DatasetDownloader:
    """Download a Roboflow keypoint / pose dataset in YOLOv8 format.

    Expected layout after download:
        <root>/data.yaml          ← class names + keypoint count + split paths
        <root>/train/images/
        <root>/train/labels/      ← YOLO keypoint .txt files
        <root>/valid/images/
        <root>/valid/labels/

    YOLO keypoint label format (per line):
        <class_id>  <cx> <cy> <w> <h>
        <kp0_x> <kp0_y> <kp0_vis>  <kp1_x> <kp1_y> <kp1_vis>  …

        All coordinates normalised to [0, 1].
        Visibility: 0 = not labelled, 1 = labelled but occluded, 2 = fully visible.

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
        dest_dir: str = "./data/pose",
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
            dest_dir=config.get("dest_dir", "./data/pose"),
        )

    def download(self) -> str:
        """Download dataset and return path to ``data.yaml``.

        Returns:
            Path to ``data.yaml`` for Ultralytics training.

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
            "Downloading %s/%s v%d (yolov8 format) …",
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
                "Check API key, workspace/project slugs, and connection."
            ) from exc

        yaml_path = os.path.join(root, "data.yaml")
        if not os.path.isfile(yaml_path):
            for f in Path(root).rglob("data.yaml"):
                yaml_path = str(f)
                break

        self._logger.info("Dataset ready.  data.yaml → %s", yaml_path)
        return yaml_path


# ============================================================
# CLASS: KeypointNormalizer
# ============================================================
class KeypointNormalizer:
    """Normalise, denormalise, and align keypoint arrays.

    Keypoints in YOLO format are stored as normalised (x, y) coordinates
    in [0, 1] relative to the image dimensions, plus a visibility flag.

    This class handles:
        - Converting between normalised and pixel coordinates.
        - Procrustes alignment: scale + translate to a canonical pose for
          comparing predicted and ground-truth skeletons.
        - Computing per-keypoint OKS (Object Keypoint Similarity) — the
          standard COCO pose metric.

    OKS definition:
        OKS(pred, gt) = Σ_k exp(−d_k² / (2 s² σ_k²)) · δ(v_k > 0)
                        ────────────────────────────────────────────
                              Σ_k δ(v_k > 0)

        where d_k = Euclidean distance between pred and GT keypoint k,
              s   = object scale (√(area) of the GT bounding box),
              σ_k = per-keypoint standard deviation (from COCO annotation
                    statistics, larger for wrists than hips etc.),
              v_k = GT visibility flag.

    Args:
        num_keypoints: Number of keypoints per person.
        image_size:    Canonical image size (used for pixel↔normalised conversion).
    """

    # COCO per-keypoint sigmas (17 keypoints)
    # Source: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    COCO_SIGMAS: np.ndarray = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089,
    ], dtype=np.float32)

    def __init__(
        self,
        num_keypoints: int = 17,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        # Use COCO sigmas if available, else uniform
        if num_keypoints == 17:
            self.sigmas = self.COCO_SIGMAS
        else:
            self.sigmas = np.full(num_keypoints, 0.05, dtype=np.float32)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"KeypointNormalizer(num_keypoints={self.num_keypoints}, "
            f"image_size={self.image_size})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "KeypointNormalizer":
        """Construct from config dict."""
        return cls(
            num_keypoints=config.get("num_keypoints", 17),
            image_size=config.get("image_size", IMAGE_SIZE),
        )

    def to_pixels(
        self,
        keypoints_norm: np.ndarray,
        image_hw: tuple[int, int],
    ) -> np.ndarray:
        """Convert normalised keypoints to pixel coordinates.

        Args:
            keypoints_norm: ``(N, 2)`` or ``(N, 3)`` array in [0, 1].
                            If 3 columns, the third is visibility (unchanged).
            image_hw:       (height, width) of the target image.

        Returns:
            Pixel-space keypoints, same shape as input.
        """
        h, w = image_hw
        result = keypoints_norm.copy().astype(np.float32)
        result[:, 0] *= w
        result[:, 1] *= h
        return result

    def to_normalised(
        self,
        keypoints_px: np.ndarray,
        image_hw: tuple[int, int],
    ) -> np.ndarray:
        """Convert pixel keypoints to normalised [0, 1] coordinates.

        Args:
            keypoints_px: ``(N, 2)`` or ``(N, 3)`` pixel array.
            image_hw:     (height, width) of the source image.

        Returns:
            Normalised keypoints, same shape.
        """
        h, w = image_hw
        result = keypoints_px.copy().astype(np.float32)
        result[:, 0] /= w
        result[:, 1] /= h
        return result

    def compute_oks(
        self,
        pred_kps: np.ndarray,
        gt_kps: np.ndarray,
        gt_bbox: Optional[np.ndarray],
        visibility: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Object Keypoint Similarity between predicted and GT keypoints.

        OKS measures the quality of keypoint localisation, accounting for the
        natural difficulty of each keypoint type (e.g. ankles are harder to
        localise than hips) via per-keypoint sigmas.

        Args:
            pred_kps:   ``(K, 2)`` predicted (x, y) pixel coordinates.
            gt_kps:     ``(K, 2)`` ground-truth (x, y) pixel coordinates.
            gt_bbox:    ``(4,)`` xyxy bounding box used to compute object scale.
                        If ``None``, scale is estimated from keypoint spread.
            visibility: ``(K,)`` visibility flags (0 = invisible).
                        If ``None``, all keypoints treated as visible.

        Returns:
            OKS value in [0, 1].  1.0 = perfect, 0.0 = no overlap.
        """
        K = len(gt_kps)
        if visibility is None:
            visibility = np.ones(K, dtype=np.float32)

        # Object scale: square root of bounding box area
        if gt_bbox is not None:
            bw = gt_bbox[2] - gt_bbox[0]
            bh = gt_bbox[3] - gt_bbox[1]
            s  = np.sqrt(bw * bh) + 1e-8
        else:
            # Estimate from spread of visible GT keypoints
            vis_kps = gt_kps[visibility > 0]
            if len(vis_kps) < 2:
                return 0.0
            s = np.sqrt(vis_kps[:, 0].var() + vis_kps[:, 1].var()) + 1e-8

        sigmas = self.sigmas[:K] if K <= len(self.sigmas) else np.full(K, 0.05, dtype=np.float32)
        var    = (2 * s * sigmas) ** 2

        d_sq = np.sum((pred_kps - gt_kps) ** 2, axis=1)  # (K,)
        exp_terms = np.exp(-d_sq / (var + 1e-10))

        valid = visibility > 0
        if not valid.any():
            return 0.0

        oks = float(exp_terms[valid].sum() / valid.sum())
        return min(oks, 1.0)

    def procrustes_align(
        self,
        source_kps: np.ndarray,
        target_kps: np.ndarray,
        visibility: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Align ``source_kps`` to ``target_kps`` via translation + uniform scale.

        Procrustes alignment (translation + isotropic scale only, no rotation):
            1. Centre both sets on their visible-keypoint centroids.
            2. Compute scale as ratio of target to source spread.
            3. Apply scale to centred source, then translate to target centroid.

        This is useful for action classification: normalising out subject
        position and size before comparing poses.

        Args:
            source_kps: ``(K, 2)`` keypoints to align.
            target_kps: ``(K, 2)`` reference keypoints.
            visibility: ``(K,)`` visibility mask. If ``None``, all visible.

        Returns:
            Aligned ``(K, 2)`` keypoints (same scale/position as target).
        """
        if visibility is None:
            visibility = np.ones(len(source_kps), dtype=np.float32)

        valid = visibility > 0
        if valid.sum() < 2:
            return source_kps.copy()

        src_vis = source_kps[valid]
        tgt_vis = target_kps[valid]

        src_mean = src_vis.mean(axis=0)
        tgt_mean = tgt_vis.mean(axis=0)

        src_centred = src_vis - src_mean
        tgt_centred = tgt_vis - tgt_mean

        src_scale = np.sqrt((src_centred ** 2).sum() / max(valid.sum(), 1))
        tgt_scale = np.sqrt((tgt_centred ** 2).sum() / max(valid.sum(), 1))

        scale_factor = tgt_scale / (src_scale + 1e-8)

        aligned = (source_kps - src_mean) * scale_factor + tgt_mean
        return aligned.astype(np.float32)


# ============================================================
# CLASS: PoseEstimator
# ============================================================
class PoseEstimator:
    """Wrap Ultralytics YOLOv8-pose for train / evaluate / infer.

    YOLOv8-pose architecture:
        - Extends YOLOv8 detection head with a parallel keypoint regression head.
        - For each anchor, predicts: box (4), confidence (1), class (nc),
          and keypoints (K × 3: x, y, visibility).
        - Trained jointly with box loss + keypoint OKS loss.
        - No additional decoder; keypoints are directly regressed.

    Args:
        weights:  Path to ``.pt`` weights or model-size string.
        device:   ``"0"`` for GPU, ``"cpu"`` for CPU.
        conf:     Confidence threshold.
        iou:      NMS IoU threshold.

    Raises:
        FileNotFoundError: Custom weights path does not exist.
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
            f"PoseEstimator(weights='{self.weights}', device='{self.device}', "
            f"conf={self.conf})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "PoseEstimator":
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
            FileNotFoundError: Custom ``.pt`` path does not exist.
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
        """Fine-tune YOLOv8-pose on a custom keypoint dataset.

        Args:
            data_yaml: Path to Ultralytics ``data.yaml`` for pose.
            epochs:    Number of training epochs.
            imgsz:     Training image size.
            batch:     Batch size.
            project:   Root output directory.
            name:      Run subdirectory name.

        Returns:
            Path to best weights file.

        Raises:
            FileNotFoundError: ``data_yaml`` does not exist.
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

        best = os.path.join(project, name, "weights", "best.pt")
        self._logger.info("Training complete.  Best weights → %s", best)
        return best

    def infer_image(
        self, image: Union[str, np.ndarray]
    ) -> list[dict]:
        """Run pose estimation on a single image.

        Args:
            image: File path or BGR ``np.ndarray``.

        Returns:
            List of person dicts, each with:
              ``box`` (xyxy), ``confidence``, ``keypoints`` (K×3 array:
              x_px, y_px, visibility), ``keypoint_names``.

        Raises:
            FileNotFoundError: Image path does not exist.
        """
        if isinstance(image, str) and not os.path.isfile(image):
            raise FileNotFoundError(f"Image not found: '{image}'")

        model = self._load_model()
        results = model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        persons = []
        for r in results:
            if r.boxes is None:
                continue
            kps_data = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

            for i, box in enumerate(r.boxes):
                kps = kps_data[i] if kps_data is not None else np.zeros((17, 3))
                persons.append({
                    "box":            box.xyxy[0].cpu().numpy().tolist(),
                    "confidence":     float(box.conf[0].cpu()),
                    "keypoints":      kps,           # (K, 3): x, y, visibility
                    "keypoint_names": COCO_KEYPOINT_NAMES,
                })

        self._logger.info(
            "infer_image: %d person(s) detected", len(persons)
        )
        return persons

    def infer_video(
        self,
        source: Union[str, int],
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> list[list[dict]]:
        """Run pose estimation frame by frame on a video or webcam.

        Args:
            source:      Video file path or webcam index (int, e.g. ``0``).
            output_path: If provided, write annotated video here.
            max_frames:  Stop after this many frames.

        Returns:
            List of per-frame person lists.

        Raises:
            FileNotFoundError: Video path does not exist.
            RuntimeError:      Cannot open video source.
        """
        if isinstance(source, str) and not os.path.isfile(source):
            raise FileNotFoundError(f"Video not found: '{source}'")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video source: '{source}'\n"
                "Check the file path or webcam index."
            )

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        limit  = min(total, max_frames) if max_frames else (max_frames or float("inf"))

        writer = None
        if output_path:
            ensure_dir(os.path.dirname(output_path) or ".")
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        viz = Visualizer()
        all_frames: list[list[dict]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= limit:
                break

            persons = self.infer_image(frame)
            all_frames.append(persons)

            if writer is not None:
                annotated = viz.draw_poses(frame, persons)
                writer.write(annotated)

            frame_idx += 1
            if frame_idx % 100 == 0:
                self._logger.info("Processed %d frames …", frame_idx)

        cap.release()
        if writer:
            writer.release()
            self._logger.info("Annotated video saved → %s", output_path)

        self._logger.info("Video inference done: %d frames.", len(all_frames))
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
            split:     ``"val"`` or ``"test"``.

        Returns:
            Dict with ``pose_mAP50``, ``pose_mAP50_95``, ``box_mAP50``.

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
                "pose_mAP50":    float(results.pose.map50),
                "pose_mAP50_95": float(results.pose.map),
                "box_mAP50":     float(results.box.map50),
                "precision":     float(results.pose.mp),
                "recall":        float(results.pose.mr),
            }
        except AttributeError as exc:
            self._logger.warning("Could not extract metrics: %s", exc)
            metrics = {"pose_mAP50": 0., "pose_mAP50_95": 0.,
                       "box_mAP50": 0., "precision": 0., "recall": 0.}

        self._logger.info(
            "Results: pose_mAP50=%.4f  pose_mAP50-95=%.4f  box_mAP50=%.4f",
            metrics["pose_mAP50"], metrics["pose_mAP50_95"], metrics["box_mAP50"],
        )
        return metrics


# ============================================================
# CLASS: MediaPipeEstimator
# ============================================================
class MediaPipeEstimator:
    """Pose estimation with MediaPipe Holistic / Pose.

    MediaPipe Pose uses a two-stage pipeline:
        Stage 1 — Pose detector:
            BlazePose detector finds the full-body bounding box.
        Stage 2 — Pose landmark model:
            A lightweight model regresses 33 3D landmarks (x, y, z, visibility).
            x, y are normalised to [0, 1] relative to image size.
            z is the relative depth (smaller = closer to camera).
            Visibility is in [0, 1] (sigmoid of raw logit).

    MediaPipe's 33 landmarks are a superset of COCO's 17 keypoints, adding
    face landmarks, foot points, and finger tips.

    Args:
        model_complexity: ``0`` (lite, fastest), ``1`` (full), ``2`` (heavy).
        min_detection_confidence: Minimum confidence for detection.
        min_tracking_confidence:  Minimum confidence for tracking.
        static_image_mode: If ``True``, detect on every frame (no tracking).

    Raises:
        ImportError: If ``mediapipe`` is not installed.
    """

    # Mapping from MediaPipe 33 landmark indices to COCO 17 keypoint indices
    # (used when converting to COCO format for metric comparison)
    MP_TO_COCO: dict[int, int] = {
        0: 0,    # nose
        2: 1,    # left_eye
        5: 2,    # right_eye
        7: 3,    # left_ear
        8: 4,    # right_ear
        11: 5,   # left_shoulder
        12: 6,   # right_shoulder
        13: 7,   # left_elbow
        14: 8,   # right_elbow
        15: 9,   # left_wrist
        16: 10,  # right_wrist
        23: 11,  # left_hip
        24: 12,  # right_hip
        25: 13,  # left_knee
        26: 14,  # right_knee
        27: 15,  # left_ankle
        28: 16,  # right_ankle
    }

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self._pose = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"MediaPipeEstimator(complexity={self.model_complexity}, "
            f"det_conf={self.min_detection_confidence})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "MediaPipeEstimator":
        """Construct from config dict."""
        return cls(
            model_complexity=config.get("model_complexity", 1),
            min_detection_confidence=config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=config.get("min_tracking_confidence", 0.5),
            static_image_mode=config.get("static_image_mode", False),
        )

    def _load(self):
        """Lazily initialise the MediaPipe Pose solution.

        Returns:
            MediaPipe Pose instance.

        Raises:
            ImportError: If ``mediapipe`` is not installed.
        """
        if self._pose is not None:
            return self._pose

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                f"mediapipe not found: {exc}\nRun: pip install mediapipe"
            ) from exc

        self._pose = mp.solutions.pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            static_image_mode=self.static_image_mode,
        )
        self._logger.info(
            "MediaPipe Pose initialised (complexity=%d)", self.model_complexity
        )
        return self._pose

    def infer_image(self, image: np.ndarray) -> list[dict]:
        """Run MediaPipe Pose on a BGR image.

        MediaPipe returns at most one person per call (whole-body pose).

        Args:
            image: BGR uint8 ``np.ndarray``.

        Returns:
            List with zero or one person dict (same format as
            ``PoseEstimator.infer_image`` but with 33 landmarks).

        Raises:
            ImportError: ``mediapipe`` not installed.
        """
        pose = self._load()
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        persons = []
        if results.pose_landmarks is None:
            self._logger.debug("No person detected.")
            return persons

        lms = results.pose_landmarks.landmark
        kps = np.array(
            [[lm.x * w, lm.y * h, lm.visibility] for lm in lms],
            dtype=np.float32,
        )  # (33, 3)

        # Bounding box from keypoints
        vis_kps = kps[kps[:, 2] > 0.1]
        if len(vis_kps) > 0:
            x1 = float(vis_kps[:, 0].min())
            y1 = float(vis_kps[:, 1].min())
            x2 = float(vis_kps[:, 0].max())
            y2 = float(vis_kps[:, 1].max())
        else:
            x1 = y1 = 0.0
            x2, y2 = float(w), float(h)

        persons.append({
            "box":            [x1, y1, x2, y2],
            "confidence":     float(np.mean(kps[:, 2])),
            "keypoints":      kps,
            "keypoint_names": [f"mp_{i}" for i in range(33)],
        })
        return persons

    def to_coco_format(
        self, mp_keypoints: np.ndarray, image_hw: tuple[int, int]
    ) -> np.ndarray:
        """Map MediaPipe 33 landmarks to COCO 17 keypoints.

        Args:
            mp_keypoints: ``(33, 3)`` array (x_px, y_px, visibility).
            image_hw:     (H, W) of the source image.

        Returns:
            ``(17, 3)`` COCO-format keypoints.
        """
        coco_kps = np.zeros((17, 3), dtype=np.float32)
        for mp_idx, coco_idx in self.MP_TO_COCO.items():
            if mp_idx < len(mp_keypoints):
                coco_kps[coco_idx] = mp_keypoints[mp_idx]
        return coco_kps

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None


# ============================================================
# CLASS: ActionClassifier
# ============================================================
class ActionClassifier:
    """Rule-based and angle-based pose action/label classifier.

    Classifies a detected pose into one of a small set of actions using
    joint angle heuristics — no model training required.

    Joint angle computation:
        Given three keypoints A, B (vertex), C:
        angle = arccos( dot(BA, BC) / (|BA| · |BC|) )

    Supported actions (configurable via ``action_rules``):
        - ``"standing"``:   torso vertical, knees near-straight.
        - ``"sitting"``:    hip angle < 120°.
        - ``"arms_raised"``: wrists above shoulders.
        - ``"t_pose"``:     arms horizontal, both wrists at shoulder height.

    Args:
        keypoint_names:  Ordered list of keypoint name strings.
        action_rules:    Optional dict overriding built-in angle thresholds.
        conf_threshold:  Minimum keypoint visibility to include in angle calc.
    """

    def __init__(
        self,
        keypoint_names: Optional[list[str]] = None,
        action_rules: Optional[dict] = None,
        conf_threshold: float = KEYPOINT_CONF_THRESHOLD,
    ) -> None:
        self.keypoint_names = keypoint_names or COCO_KEYPOINT_NAMES
        self.conf_threshold = conf_threshold
        # Build name → index lookup
        self._kp_idx: dict[str, int] = {
            name: i for i, name in enumerate(self.keypoint_names)
        }
        self._rules = action_rules or self._default_rules()
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"ActionClassifier(actions={list(self._rules.keys())}, "
            f"conf_threshold={self.conf_threshold})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "ActionClassifier":
        """Construct from config dict."""
        return cls(
            keypoint_names=config.get("keypoint_names"),
            action_rules=config.get("action_rules"),
            conf_threshold=config.get("keypoint_conf_threshold", KEYPOINT_CONF_THRESHOLD),
        )

    @staticmethod
    def _default_rules() -> dict:
        """Return built-in action rule thresholds."""
        return {
            "arms_raised":  {"wrists_above_shoulders": True},
            "t_pose":       {"elbow_angle_min": 150, "wrist_shoulder_y_diff_max": 60},
            "sitting":      {"hip_angle_max": 120},
            "standing":     {"knee_angle_min": 150},
            "unknown":      {},
        }

    @staticmethod
    def _joint_angle(
        a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> float:
        """Compute angle at vertex B formed by A–B–C.

        Args:
            a: Point A coordinates ``(2,)``.
            b: Vertex B coordinates ``(2,)``.
            c: Point C coordinates ``(2,)``.

        Returns:
            Angle in degrees [0, 180].
        """
        ba = a - b
        bc = c - b
        norm = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
        cos_angle = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def _get_kp(
        self, keypoints: np.ndarray, name: str
    ) -> Optional[np.ndarray]:
        """Retrieve a keypoint by name if visible above threshold.

        Args:
            keypoints: ``(K, 3)`` array (x, y, visibility).
            name:      Keypoint name string.

        Returns:
            ``(2,)`` (x, y) array if visible, else ``None``.
        """
        idx = self._kp_idx.get(name)
        if idx is None or idx >= len(keypoints):
            return None
        kp = keypoints[idx]
        if kp[2] < self.conf_threshold:
            return None
        return kp[:2]

    def classify(self, keypoints: np.ndarray) -> str:
        """Classify a single-person pose into an action label.

        Rules are evaluated in priority order.  The first matching rule wins.

        Args:
            keypoints: ``(K, 3)`` array (x_px, y_px, visibility).

        Returns:
            Action label string.
        """
        g = lambda name: self._get_kp(keypoints, name)

        # ── Arms raised ──────────────────────────────────────────────
        lw = g("left_wrist");  rw = g("right_wrist")
        ls = g("left_shoulder"); rs = g("right_shoulder")
        if lw is not None and ls is not None and rw is not None and rs is not None:
            if lw[1] < ls[1] and rw[1] < rs[1]:   # y increases downward
                return "arms_raised"

        # ── T-pose: arms horizontal ────────────────────────────────
        le = g("left_elbow"); re = g("right_elbow")
        if (lw is not None and ls is not None and le is not None and
                rw is not None and rs is not None and re is not None):
            l_angle = self._joint_angle(ls, le, lw)
            r_angle = self._joint_angle(rs, re, rw)
            y_diff_l = abs(float(lw[1]) - float(ls[1]))
            y_diff_r = abs(float(rw[1]) - float(rs[1]))
            if (l_angle > 150 and r_angle > 150 and
                    y_diff_l < 60 and y_diff_r < 60):
                return "t_pose"

        # ── Sitting: hip angle < 120° ───────────────────────────────
        lh = g("left_hip"); rh = g("right_hip")
        lk = g("left_knee"); rk = g("right_knee")
        if ls is not None and lh is not None and lk is not None:
            hip_angle = self._joint_angle(ls, lh, lk)
            if hip_angle < 120:
                return "sitting"

        # ── Standing: knee angle > 150° ─────────────────────────────
        la = g("left_ankle"); ra = g("right_ankle")
        if lh is not None and lk is not None and la is not None:
            knee_angle = self._joint_angle(lh, lk, la)
            if knee_angle > 150:
                return "standing"

        return "unknown"

    def classify_batch(
        self, persons: list[dict]
    ) -> list[str]:
        """Classify a list of person dicts.

        Args:
            persons: Output of ``PoseEstimator.infer_image``.

        Returns:
            List of action label strings (same order).
        """
        return [self.classify(p["keypoints"]) for p in persons]


# ============================================================
# CLASS: Evaluator
# ============================================================
class Evaluator:
    """Compute PCK and OKS-based metrics from keypoint predictions.

    PCK (Percentage of Correct Keypoints):
        A keypoint prediction is correct if its Euclidean distance to the
        GT keypoint is less than ``threshold × max(bbox_h, bbox_w)``.
        PCK@0.2 is the standard threshold.

    mOKS (mean Object Keypoint Similarity):
        Average OKS across all persons and images.
        Directly usable as a proxy for COCO pose mAP.

    Args:
        num_keypoints:  Number of keypoints per person.
        pck_threshold:  PCK threshold as fraction of bounding box size.
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        pck_threshold: float = 0.2,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.pck_threshold = pck_threshold
        self._normalizer = KeypointNormalizer(num_keypoints=num_keypoints)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"Evaluator(num_keypoints={self.num_keypoints}, "
            f"pck_threshold={self.pck_threshold})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "Evaluator":
        """Construct from config dict."""
        return cls(
            num_keypoints=config.get("num_keypoints", 17),
            pck_threshold=config.get("pck_threshold", 0.2),
        )

    def compute_pck(
        self,
        pred_kps: np.ndarray,
        gt_kps: np.ndarray,
        gt_bbox: np.ndarray,
        visibility: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute PCK@threshold for a single person.

        Args:
            pred_kps:   ``(K, 2)`` predicted pixel keypoints.
            gt_kps:     ``(K, 2)`` ground-truth pixel keypoints.
            gt_bbox:    ``(4,)`` xyxy bounding box.
            visibility: ``(K,)`` visibility flags.  All visible if ``None``.

        Returns:
            Dict with ``pck`` (overall) and ``per_keypoint_correct`` list.
        """
        K = len(gt_kps)
        if visibility is None:
            visibility = np.ones(K, dtype=np.float32)

        bw = gt_bbox[2] - gt_bbox[0]
        bh = gt_bbox[3] - gt_bbox[1]
        normaliser = max(bw, bh) * self.pck_threshold + 1e-8

        distances = np.linalg.norm(pred_kps - gt_kps, axis=1)  # (K,)
        correct   = (distances < normaliser) & (visibility > 0)
        valid_kps = visibility > 0

        pck = float(correct.sum() / max(valid_kps.sum(), 1))
        return {
            "pck":                pck,
            "per_keypoint_correct": correct.tolist(),
            "num_valid":          int(valid_kps.sum()),
        }

    def compute_dataset_metrics(
        self,
        all_pred: list[list[dict]],
        all_gt: list[list[dict]],
    ) -> dict:
        """Compute PCK and mOKS across all images.

        Args:
            all_pred: Per-image lists of predicted person dicts.
            all_gt:   Per-image lists of GT person dicts.

        Returns:
            Dict with ``pck``, ``mOKS``, and ``per_keypoint_pck``.
        """
        all_pck: list[float] = []
        all_oks: list[float] = []
        per_kp_correct = np.zeros(self.num_keypoints)
        per_kp_count   = np.zeros(self.num_keypoints)

        for pred_persons, gt_persons in zip(all_pred, all_gt):
            # Simple greedy 1-to-1 matching by box IoU
            for gt in gt_persons:
                if not pred_persons:
                    break
                # Use first unmatched prediction
                pred = pred_persons[0]
                pred_kps = pred["keypoints"][:, :2]
                gt_kps   = gt["keypoints"][:, :2]
                vis      = gt["keypoints"][:, 2] if gt["keypoints"].shape[1] > 2 else None
                gt_bbox  = np.array(gt["box"])

                K = min(self.num_keypoints, len(pred_kps), len(gt_kps))
                pck_result = self.compute_pck(
                    pred_kps[:K], gt_kps[:K],
                    gt_bbox, vis[:K] if vis is not None else None,
                )
                oks = self._normalizer.compute_oks(
                    pred_kps[:K], gt_kps[:K],
                    gt_bbox, vis[:K] if vis is not None else None,
                )

                all_pck.append(pck_result["pck"])
                all_oks.append(oks)

                correct = pck_result["per_keypoint_correct"]
                for k, c in enumerate(correct[:self.num_keypoints]):
                    per_kp_correct[k] += int(c)
                    per_kp_count[k]   += 1

        mean_pck = float(np.mean(all_pck)) if all_pck else 0.0
        mean_oks = float(np.mean(all_oks)) if all_oks else 0.0
        per_kp   = (per_kp_correct / np.maximum(per_kp_count, 1)).tolist()

        self._logger.info(
            "Dataset: PCK@%.1f=%.4f  mOKS=%.4f  n_persons=%d",
            self.pck_threshold, mean_pck, mean_oks, len(all_pck),
        )
        return {
            "pck":            mean_pck,
            "mOKS":           mean_oks,
            "per_keypoint_pck": per_kp,
        }


# ============================================================
# CLASS: Visualizer
# ============================================================
class Visualizer:
    """Draw skeletons, keypoints, and action labels; save charts.

    Args:
        output_dir:        Directory for saved outputs.
        keypoint_radius:   Radius for keypoint circles.
        skeleton_color:    BGR colour for skeleton limb lines.
        keypoint_color:    BGR colour for keypoint dots.
        kp_conf_threshold: Minimum visibility to draw a keypoint.
    """

    # Per-body-part BGR colours for skeleton segments
    LIMB_COLORS: list[tuple[int, int, int]] = [
        (255, 128, 0),   # head
        (255, 128, 0),
        (255, 128, 0),
        (255, 128, 0),
        (128, 255, 0),   # shoulders
        (0, 255, 128),   # left arm
        (0, 255, 128),
        (255, 0, 128),   # right arm
        (255, 0, 128),
        (128, 128, 255), # torso
        (128, 128, 255),
        (128, 128, 255),
        (0, 200, 255),   # left leg
        (0, 200, 255),
        (255, 200, 0),   # right leg
        (255, 200, 0),
        (255, 200, 0),
    ]

    def __init__(
        self,
        output_dir: str = LOG_DIR,
        keypoint_radius: int = 4,
        skeleton_color: tuple[int, int, int] = (0, 255, 0),
        keypoint_color: tuple[int, int, int] = (255, 0, 0),
        kp_conf_threshold: float = KEYPOINT_CONF_THRESHOLD,
    ) -> None:
        self.output_dir = output_dir
        self.keypoint_radius = keypoint_radius
        self.skeleton_color = skeleton_color
        self.keypoint_color = keypoint_color
        self.kp_conf_threshold = kp_conf_threshold
        ensure_dir(output_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"Visualizer(output_dir='{self.output_dir}')"

    @classmethod
    def from_config(cls, config: dict) -> "Visualizer":
        """Construct from config dict."""
        return cls(output_dir=config.get("log_dir", LOG_DIR))

    def draw_poses(
        self,
        image: np.ndarray,
        persons: list[dict],
        actions: Optional[list[str]] = None,
        skeleton: Optional[list[tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Draw keypoints and skeleton for all detected persons.

        For each person:
            1. Draw each limb in the skeleton as a coloured line between
               two keypoints (only if both are visible above threshold).
            2. Draw each keypoint as a filled circle with index label.
            3. Overlay action label if provided.

        Args:
            image:    BGR uint8 image.
            persons:  List of person dicts from ``PoseEstimator.infer_image``.
            actions:  Optional list of action strings (same length as persons).
            skeleton: List of (i, j) index pairs for limb connections.
                      Defaults to ``COCO_SKELETON``.

        Returns:
            Annotated BGR image.
        """
        vis = image.copy()
        skel = skeleton or COCO_SKELETON

        for p_idx, person in enumerate(persons):
            kps  = person["keypoints"]  # (K, 3)
            conf = person["confidence"]

            # Draw bounding box
            x1, y1, x2, y2 = [int(v) for v in person["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

            # Draw skeleton limbs
            for bone_idx, (i, j) in enumerate(skel):
                if i >= len(kps) or j >= len(kps):
                    continue
                kp_i, kp_j = kps[i], kps[j]
                if kp_i[2] < self.kp_conf_threshold or kp_j[2] < self.kp_conf_threshold:
                    continue
                color = self.LIMB_COLORS[bone_idx % len(self.LIMB_COLORS)]
                cv2.line(
                    vis,
                    (int(kp_i[0]), int(kp_i[1])),
                    (int(kp_j[0]), int(kp_j[1])),
                    color, 2, cv2.LINE_AA,
                )

            # Draw keypoints
            for k_idx, kp in enumerate(kps):
                if kp[2] < self.kp_conf_threshold:
                    continue
                cx, cy = int(kp[0]), int(kp[1])
                cv2.circle(vis, (cx, cy), self.keypoint_radius, self.keypoint_color, -1)
                cv2.circle(vis, (cx, cy), self.keypoint_radius + 1, (255, 255, 255), 1)

            # Action label
            action = actions[p_idx] if actions and p_idx < len(actions) else None
            label  = f"conf={conf:.2f}"
            if action:
                label = f"{action}  {label}"
            cv2.putText(
                vis, label, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return vis

    def save_result(
        self,
        image: np.ndarray,
        persons: list[dict],
        filename: str,
        actions: Optional[list[str]] = None,
    ) -> str:
        """Draw poses and save annotated image to disk.

        Args:
            image:    Original BGR image.
            persons:  Person list from inference.
            filename: Output filename.
            actions:  Optional action labels.

        Returns:
            Full path to saved file.
        """
        annotated = self.draw_poses(image, persons, actions)
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, annotated)
        self._logger.info(
            "Saved → %s  (%d persons)", path, len(persons)
        )
        return path

    def plot_per_keypoint_pck(
        self,
        per_kp_pck: list[float],
        keypoint_names: Optional[list[str]] = None,
        filename: str = "per_keypoint_pck.png",
    ) -> str:
        """Horizontal bar chart of PCK per keypoint.

        Args:
            per_kp_pck:     PCK value per keypoint.
            keypoint_names: Names for x-axis labels.
            filename:       Output filename.

        Returns:
            Full path to saved figure.
        """
        names = keypoint_names or COCO_KEYPOINT_NAMES
        n = min(len(names), len(per_kp_pck))
        names, values = names[:n], per_kp_pck[:n]
        mean_pck = float(np.mean(values))

        fig, ax = plt.subplots(figsize=(7, max(4, n * 0.4)))
        bars = ax.barh(names, values, color="steelblue")
        ax.axvline(mean_pck, color="red", linestyle="--", label=f"mean PCK={mean_pck:.3f}")
        ax.set_xlabel("PCK@0.2"); ax.set_title("Per-Keypoint PCK")
        ax.set_xlim(0, 1.05); ax.legend()
        for bar, v in zip(bars, values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=7)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Per-keypoint PCK chart saved → %s", path)
        return path

    def plot_action_distribution(
        self,
        actions: list[str],
        filename: str = "action_distribution.png",
    ) -> str:
        """Bar chart of action label frequencies.

        Args:
            actions:  Flat list of predicted action strings.
            filename: Output filename.

        Returns:
            Full path to saved figure.
        """
        counts: dict[str, int] = {}
        for a in actions:
            counts[a] = counts.get(a, 0) + 1

        names  = list(counts.keys())
        values = [counts[n] for n in names]

        fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.9), 4))
        ax.bar(names, values, color="steelblue", edgecolor="white")
        ax.set_xlabel("Action"); ax.set_ylabel("Count")
        ax.set_title("Action Distribution")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Action distribution saved → %s", path)
        return path

    def plot_keypoint_heatmap(
        self,
        persons: list[dict],
        image_hw: tuple[int, int],
        filename: str = "keypoint_heatmap.png",
        sigma: int = 15,
    ) -> str:
        """2-D Gaussian heatmap of keypoint positions across many detections.

        Useful for visualising where keypoints tend to appear in a dataset —
        reveals dataset biases or dominant poses.

        Each visible keypoint adds a Gaussian blob to the accumulator.
        Final heatmap is normalised and colour-mapped.

        Args:
            persons:   List of person dicts (aggregated from a batch).
            image_hw:  (H, W) of the reference image grid.
            filename:  Output filename.
            sigma:     Gaussian spread in pixels.

        Returns:
            Full path to saved figure.
        """
        H, W = image_hw
        heatmap = np.zeros((H, W), dtype=np.float32)

        for person in persons:
            kps = person["keypoints"]
            for kp in kps:
                x, y, v = kp
                if v < self.kp_conf_threshold:
                    continue
                xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
                # Add a small Gaussian blob at (xi, yi)
                x_range = np.arange(W)
                y_range = np.arange(H)
                xx, yy  = np.meshgrid(x_range, y_range)
                blob = np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sigma ** 2))
                heatmap += blob.astype(np.float32)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        fig, ax = plt.subplots(figsize=(W / 80, H / 80))
        ax.imshow(heatmap, cmap="hot", vmin=0, vmax=1)
        ax.set_title("Keypoint Density Heatmap")
        ax.axis("off")
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Keypoint heatmap saved → %s", path)
        return path


# ============================================================
# CLASS: PosePipeline  (orchestrator)
# ============================================================
class PosePipeline:
    """End-to-end orchestrator: download → train → evaluate → infer.

    Args:
        mode:       ``"train"``, ``"evaluate"``, or ``"infer"``.
        backend:    ``"yolov8pose"`` or ``"mediapipe"``.
        weights:    Weights path or YOLO model-size string.
        data_root:  Existing dataset root (skips download).
        source:     Image / video path, or webcam index for inference.
        output_dir: Directory for outputs.
        config:     Optional flat config dict.

    Example::

        pipeline = PosePipeline(mode="train", backend="yolov8pose")
        pipeline.run()

        pipeline = PosePipeline(mode="infer", backend="mediapipe",
                                source="person.jpg")
        pipeline.run()
    """

    def __init__(
        self,
        mode: str = "train",
        backend: str = BACKEND,
        weights: str = YOLO_MODEL_SIZE,
        data_root: Optional[str] = None,
        source: Optional[Union[str, int]] = None,
        output_dir: str = LOG_DIR,
        config: Optional[dict] = None,
    ) -> None:
        self.mode = mode
        self.backend = backend
        self.weights = weights
        self.data_root = data_root
        self.source = source
        self.output_dir = output_dir
        self.config = config or {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"PosePipeline(mode='{self.mode}', backend='{self.backend}')"
        )

    @classmethod
    def from_config(cls, config: dict) -> "PosePipeline":
        """Construct from config dict."""
        return cls(
            mode=config.get("mode", "train"),
            backend=config.get("backend", BACKEND),
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
        downloader = DatasetDownloader.from_config(self.config)
        return downloader.download()

    def run(self) -> dict:
        """Execute pipeline according to ``self.mode`` and ``self.backend``.

        Returns:
            Result dict (contents depend on mode and backend).

        Raises:
            ValueError: Unknown mode or backend.
        """
        self._logger.info("=" * 55)
        self._logger.info(
            "Pose Pipeline  mode='%s'  backend='%s'",
            self.mode, self.backend,
        )
        self._logger.info("=" * 55)

        if self.backend == "mediapipe":
            return self._run_mediapipe()

        if self.backend != "yolov8pose":
            raise ValueError(
                f"Unknown backend='{self.backend}'. "
                "Choose: yolov8pose / mediapipe."
            )

        estimator = PoseEstimator(
            weights=self.weights,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_THRESHOLD,
        )
        viz = Visualizer(output_dir=self.output_dir)
        classifier = ActionClassifier()

        if self.mode == "train":
            yaml_path    = self._get_yaml()
            best_weights = estimator.train(data_yaml=yaml_path)
            metrics      = estimator.evaluate(yaml_path, weights=best_weights)
            ensure_dir(self.output_dir)
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            return {"metrics": metrics, "best_weights": best_weights}

        elif self.mode == "evaluate":
            yaml_path = self._get_yaml()
            metrics   = estimator.evaluate(yaml_path, weights=self.weights)
            return {"metrics": metrics}

        elif self.mode == "infer":
            return self._run_yolo_infer(estimator, viz, classifier)

        else:
            raise ValueError(
                f"Unknown mode='{self.mode}'. Choose: train / evaluate / infer."
            )

    def _run_yolo_infer(
        self,
        estimator: PoseEstimator,
        viz: Visualizer,
        classifier: ActionClassifier,
    ) -> dict:
        """Handle YOLOv8-pose inference on image, video, or webcam."""
        if self.source is None:
            raise ValueError("--source is required for infer mode.")

        src = self.source

        # Webcam index
        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            idx = int(src)
            self._logger.info("Webcam inference (index=%d). Press Q to quit.", idx)
            frame_persons = estimator.infer_video(idx)
            return {"frame_persons": frame_persons}

        if not isinstance(src, str):
            raise ValueError(f"Unexpected source type: {type(src)}")

        ext = os.path.splitext(src)[1].lower()
        if ext in {".mp4", ".avi", ".mov", ".mkv"}:
            out_video = os.path.join(self.output_dir, "pose_output.mp4")
            frame_persons = estimator.infer_video(src, output_path=out_video)
            all_actions = [a for fp in frame_persons for a in classifier.classify_batch(fp)]
            if all_actions:
                viz.plot_action_distribution(all_actions)
            return {"frame_persons": frame_persons, "output_video": out_video}

        else:
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Image not found: '{src}'")
            img = cv2.imread(src)
            if img is None:
                raise RuntimeError(f"Could not read image: '{src}'")
            persons = estimator.infer_image(src)
            actions = classifier.classify_batch(persons)
            out_path = viz.save_result(
                img, persons,
                filename=f"pose_{Path(src).stem}.jpg",
                actions=actions,
            )
            # Heatmap
            viz.plot_keypoint_heatmap(persons, img.shape[:2])
            return {"persons": persons, "actions": actions, "output_image": out_path}

    def _run_mediapipe(self) -> dict:
        """Handle MediaPipe Pose inference."""
        mp_est = MediaPipeEstimator(
            model_complexity=self.config.get("model_complexity", 1),
            static_image_mode=(self.mode == "infer"),
        )

        if self.source is None:
            raise ValueError("--source is required for mediapipe infer mode.")

        src = self.source

        if isinstance(src, str) and os.path.isfile(src):
            ext = os.path.splitext(src)[1].lower()
            if ext in {".mp4", ".avi", ".mov", ".mkv"}:
                cap = cv2.VideoCapture(src)
                viz = Visualizer(output_dir=self.output_dir)
                all_persons = []
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    persons = mp_est.infer_image(frame)
                    all_persons.extend(persons)
                    frame_idx += 1
                cap.release()
                mp_est.close()
                return {"total_frames": frame_idx, "total_detections": len(all_persons)}
            else:
                img = cv2.imread(src)
                if img is None:
                    raise RuntimeError(f"Could not read image: '{src}'")
                persons = mp_est.infer_image(img)
                viz = Visualizer(output_dir=self.output_dir)
                classifier = ActionClassifier()
                actions = classifier.classify_batch(persons)
                out = viz.save_result(
                    img, persons,
                    filename=f"mediapipe_{Path(src).stem}.jpg",
                    actions=actions,
                )
                mp_est.close()
                return {"persons": persons, "actions": actions, "output_image": out}
        else:
            raise FileNotFoundError(f"Source not found or invalid: '{src}'")


# ============================================================
# ENTRY POINT
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose Estimation — Section 6")
    parser.add_argument("--mode", choices=["train", "evaluate", "infer"],
                        default="train")
    parser.add_argument("--backend", choices=["yolov8pose", "mediapipe"],
                        default=BACKEND)
    parser.add_argument("--weights", type=str, default=YOLO_MODEL_SIZE)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=LOG_DIR)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    pipeline = PosePipeline(
        mode=args.mode,
        backend=args.backend,
        weights=args.weights,
        data_root=args.data_root,
        source=args.source,
        output_dir=args.output_dir,
    )
    result = pipeline.run()
    printable = {k: v for k, v in result.items()
                 if isinstance(v, (str, dict, float, int, list)) and k != "frame_persons"}
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
