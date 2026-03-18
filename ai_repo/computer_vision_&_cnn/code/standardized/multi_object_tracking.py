from __future__ import annotations

"""
multi_object_tracking.py
========================
Industry-Standard Multi-Object Tracking Pipeline — ByteTrack via Ultralytics + supervision

Installation
------------
    pip install ultralytics supervision roboflow opencv-python matplotlib \
                numpy scipy tqdm

Usage
-----
    # Track objects in a video (default: YOLOv8n detector + ByteTrack)
    python multi_object_tracking.py --source path/to/video.mp4

    # Use a custom-trained detector
    python multi_object_tracking.py --source video.mp4 --weights best.pt

    # Track from webcam
    python multi_object_tracking.py --source 0

    # Evaluate tracking on MOT-format ground truth
    python multi_object_tracking.py --mode evaluate --source video.mp4 \
                                    --gt-file path/to/gt.txt

    # Export tracks to JSON / CSV
    python multi_object_tracking.py --source video.mp4 --export json

Author: CV Course — Section 7
Python: 3.9+  |  Ultralytics YOLOv8 + supervision ByteTrack
"""

# ============================================================
# GLOBAL CONFIGURATION — edit these before running
# ============================================================
YOLO_MODEL_SIZE: str        = "yolov8n"          # nano detector (fast)
TRACKER: str                = "bytetrack"        # ByteTrack via supervision
IMAGE_SIZE: int             = 640
CONFIDENCE_THRESHOLD: float = 0.25
NMS_THRESHOLD: float        = 0.45
DEVICE: str                 = "0"                # "0" = GPU, "cpu" = CPU
SEED: int                   = 42
CHECKPOINT_DIR: str         = "./checkpoints"
LOG_DIR: str                = "./logs/tracking"

# ByteTrack hyperparameters
BT_TRACK_THRESH: float      = 0.25   # detection confidence to start a track
BT_TRACK_BUFFER: int        = 30     # frames to keep a lost track alive
BT_MATCH_THRESH: float      = 0.8    # IoU threshold for track-detection matching
BT_FRAME_RATE: int          = 30     # assumed video frame rate for Kalman motion model

# HOTA / MOTA evaluation thresholds
EVAL_IOU_THRESHOLD: float   = 0.5    # IoU overlap to count a detection as TP

# ============================================================
# IMPORTS
# ============================================================
import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
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
    import supervision as sv
except ImportError as exc:
    sys.exit(
        f"supervision import failed: {exc}\n"
        "Run: pip install supervision"
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
logger = logging.getLogger("MOT")


def ensure_dir(path: str) -> None:
    """Create directory tree if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS: DetectionResult
# ============================================================
class DetectionResult:
    """Structured container for a single-frame detection + tracking output.

    Stores the raw detection outputs from YOLO alongside the track IDs
    assigned by ByteTrack, enabling clean data flow between pipeline stages.

    Attributes:
        frame_id:    Frame index (0-based).
        boxes_xyxy:  ``(N, 4)`` float32 bounding boxes in pixel coordinates.
        confidences: ``(N,)``   float32 confidence scores.
        class_ids:   ``(N,)``   int32 class IDs.
        track_ids:   ``(N,)``   int32 track IDs (-1 = untracked).
        class_names: Dict mapping class id → name string.
    """

    def __init__(
        self,
        frame_id: int,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        track_ids: np.ndarray,
        class_names: Optional[dict[int, str]] = None,
    ) -> None:
        self.frame_id    = frame_id
        self.boxes_xyxy  = boxes_xyxy
        self.confidences = confidences
        self.class_ids   = class_ids
        self.track_ids   = track_ids
        self.class_names = class_names or {}

    def __repr__(self) -> str:
        return (
            f"DetectionResult(frame={self.frame_id}, "
            f"n={len(self.boxes_xyxy)}, "
            f"track_ids={self.track_ids.tolist()})"
        )

    def __len__(self) -> int:
        return len(self.boxes_xyxy)

    @classmethod
    def from_config(cls, config: dict) -> "DetectionResult":
        """Construct an empty DetectionResult from config dict."""
        return cls(
            frame_id=config.get("frame_id", 0),
            boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
            class_ids=np.empty(0, dtype=np.int32),
            track_ids=np.empty(0, dtype=np.int32),
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dict with frame_id, detections list (one entry per object).
        """
        detections = []
        for i in range(len(self)):
            detections.append({
                "track_id":   int(self.track_ids[i]),
                "class_id":   int(self.class_ids[i]),
                "class_name": self.class_names.get(int(self.class_ids[i]), ""),
                "confidence": float(self.confidences[i]),
                "box_xyxy":   self.boxes_xyxy[i].tolist(),
            })
        return {"frame_id": self.frame_id, "detections": detections}

    def filter_by_class(self, class_ids: list[int]) -> "DetectionResult":
        """Return a new DetectionResult keeping only specified class IDs.

        Args:
            class_ids: List of class IDs to retain.

        Returns:
            Filtered ``DetectionResult``.
        """
        mask = np.isin(self.class_ids, class_ids)
        return DetectionResult(
            frame_id=self.frame_id,
            boxes_xyxy=self.boxes_xyxy[mask],
            confidences=self.confidences[mask],
            class_ids=self.class_ids[mask],
            track_ids=self.track_ids[mask],
            class_names=self.class_names,
        )


# ============================================================
# CLASS: ByteTrackWrapper
# ============================================================
class ByteTrackWrapper:
    """Wrap supervision's ByteTrack for frame-by-frame tracking.

    ByteTrack Algorithm Overview:
        ByteTrack (Zhang et al., 2022) improves upon SORT by using *every*
        detection — including low-confidence ones — in a two-stage matching:

        Stage 1 — High-confidence associations:
            Keep detections with conf ≥ track_thresh.
            Match to existing tracks via IoU using the Hungarian algorithm.
            Update matched tracks; mark unmatched tracks as lost.

        Stage 2 — Low-confidence recovery:
            Detections with conf in [0.1, track_thresh) are used to recover
            lost tracks from Stage 1 (e.g. partially occluded objects that
            got a low-confidence detection).
            Match lost tracks to low-conf detections by IoU.

        Track lifecycle:
            New track: created when a detection is unmatched for > 1 frame.
            Active:    matched in the current frame.
            Lost:      unmatched; kept alive for track_buffer frames
                       using Kalman-filter-predicted position.
            Removed:   lost for > track_buffer frames.

        Kalman filter:
            State: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
            Constant-velocity motion model.
            Measurement: [cx, cy, aspect_ratio, height] from detector.

    Args:
        track_thresh:  Confidence threshold for high-quality detections.
        track_buffer:  Frames to keep a lost track alive.
        match_thresh:  IoU threshold for Stage 1 matching.
        frame_rate:    Video FPS (used to size the Kalman velocity model).
    """

    def __init__(
        self,
        track_thresh: float = BT_TRACK_THRESH,
        track_buffer: int   = BT_TRACK_BUFFER,
        match_thresh: float = BT_MATCH_THRESH,
        frame_rate: int     = BT_FRAME_RATE,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate   = frame_rate
        self._tracker: Optional[sv.ByteTracker] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"ByteTrackWrapper(track_thresh={self.track_thresh}, "
            f"track_buffer={self.track_buffer}, "
            f"match_thresh={self.match_thresh})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "ByteTrackWrapper":
        """Construct from config dict."""
        return cls(
            track_thresh=config.get("track_thresh", BT_TRACK_THRESH),
            track_buffer=config.get("track_buffer", BT_TRACK_BUFFER),
            match_thresh=config.get("match_thresh", BT_MATCH_THRESH),
            frame_rate=config.get("frame_rate", BT_FRAME_RATE),
        )

    def _init_tracker(self) -> sv.ByteTracker:
        """Lazily initialise supervision ByteTracker.

        Returns:
            Configured ``sv.ByteTracker`` instance.
        """
        if self._tracker is not None:
            return self._tracker

        try:
            self._tracker = sv.ByteTracker(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=self.frame_rate,
            )
        except TypeError:
            # Older supervision API fallback
            self._tracker = sv.ByteTracker(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=self.frame_rate,
            )

        self._logger.info(
            "ByteTracker initialised  thresh=%.2f  buffer=%d  match=%.2f  fps=%d",
            self.track_thresh, self.track_buffer, self.match_thresh, self.frame_rate,
        )
        return self._tracker

    def update(
        self,
        detections: sv.Detections,
        frame_shape: tuple[int, int],
    ) -> sv.Detections:
        """Update tracker state with new detections.

        Args:
            detections:  supervision ``Detections`` for the current frame.
            frame_shape: ``(H, W)`` of the current frame.

        Returns:
            Updated ``sv.Detections`` with ``tracker_id`` field populated.
        """
        tracker = self._init_tracker()
        tracked = tracker.update_with_detections(detections)
        self._logger.debug(
            "Tracker update: %d dets → %d tracks", len(detections), len(tracked)
        )
        return tracked

    def reset(self) -> None:
        """Reset tracker state (call between videos or evaluation runs)."""
        self._tracker = None
        self._logger.info("ByteTracker reset.")


# ============================================================
# CLASS: TrackHistory
# ============================================================
class TrackHistory:
    """Maintain per-track trajectory history for visualisation and analysis.

    Stores the centroid positions of each track over time, enabling trail
    visualisation (showing where each object has been), track lifetime
    statistics, and speed estimation.

    Args:
        max_trail_length: Maximum number of past positions to retain per track.
        fps:              Video frame rate (for speed calculation in px/sec).
    """

    def __init__(
        self,
        max_trail_length: int = 60,
        fps: float = 30.0,
    ) -> None:
        self.max_trail_length = max_trail_length
        self.fps = fps
        # track_id → list of (x, y) centroids
        self._trails: dict[int, list[tuple[float, float]]] = defaultdict(list)
        # track_id → first frame seen
        self._birth_frame: dict[int, int] = {}
        # track_id → last frame seen
        self._last_frame: dict[int, int] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"TrackHistory(active_tracks={len(self._trails)}, "
            f"max_trail={self.max_trail_length})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "TrackHistory":
        """Construct from config dict."""
        return cls(
            max_trail_length=config.get("max_trail_length", 60),
            fps=config.get("fps", 30.0),
        )

    def update(self, result: DetectionResult) -> None:
        """Ingest one frame's tracking results.

        Args:
            result: ``DetectionResult`` with populated track IDs.
        """
        for i, tid in enumerate(result.track_ids):
            if tid < 0:
                continue
            box = result.boxes_xyxy[i]
            cx  = float((box[0] + box[2]) / 2)
            cy  = float((box[1] + box[3]) / 2)

            trail = self._trails[tid]
            trail.append((cx, cy))
            if len(trail) > self.max_trail_length:
                trail.pop(0)

            if tid not in self._birth_frame:
                self._birth_frame[tid] = result.frame_id
            self._last_frame[tid] = result.frame_id

    def get_trail(self, track_id: int) -> list[tuple[float, float]]:
        """Return the centroid trail for a given track.

        Args:
            track_id: Track ID.

        Returns:
            List of ``(x, y)`` tuples, oldest first.
        """
        return list(self._trails.get(track_id, []))

    def track_lifetime(self, track_id: int) -> int:
        """Return the number of frames a track has been alive.

        Args:
            track_id: Track ID.

        Returns:
            Frame count.  0 if track not found.
        """
        if track_id not in self._birth_frame:
            return 0
        return self._last_frame[track_id] - self._birth_frame[track_id] + 1

    def estimate_speed(self, track_id: int, n_frames: int = 10) -> float:
        """Estimate instantaneous speed from the last N trail positions.

        Speed = mean Euclidean displacement per frame × FPS (pixels/second).

        Args:
            track_id: Track ID.
            n_frames: Number of recent frames to average over.

        Returns:
            Speed in pixels per second.
        """
        trail = self._trails.get(track_id, [])
        if len(trail) < 2:
            return 0.0

        recent = trail[-min(n_frames + 1, len(trail)):]
        dists  = [
            np.sqrt((recent[k][0] - recent[k-1][0])**2 +
                    (recent[k][1] - recent[k-1][1])**2)
            for k in range(1, len(recent))
        ]
        mean_disp = float(np.mean(dists))
        return mean_disp * self.fps

    def summary(self) -> dict:
        """Return summary statistics across all tracks.

        Returns:
            Dict with ``total_tracks``, ``mean_lifetime_frames``,
            ``max_lifetime_frames``, ``active_track_ids``.
        """
        all_ids = list(self._birth_frame.keys())
        lifetimes = [self.track_lifetime(tid) for tid in all_ids]
        return {
            "total_tracks":         len(all_ids),
            "mean_lifetime_frames": float(np.mean(lifetimes)) if lifetimes else 0.0,
            "max_lifetime_frames":  int(max(lifetimes)) if lifetimes else 0,
            "active_track_ids":     all_ids,
        }


# ============================================================
# CLASS: MOTEvaluator
# ============================================================
class MOTEvaluator:
    """Compute MOTA, IDF1, and ID-switch metrics from tracking results.

    Metrics:
        MOTA (Multiple Object Tracking Accuracy):
            MOTA = 1 - (FN + FP + IDSW) / GT
            Penalises missed detections (FN), false alarms (FP), and ID
            switches (IDSW, where a track changes its assigned GT identity).
            Range: (-inf, 1].  Higher is better.

        IDF1 (ID F1 Score):
            IDF1 = 2 × IDTP / (2 × IDTP + IDFP + IDFN)
            Measures how long the tracker correctly maintains a single ID
            for each ground-truth object.
            IDTP = frames where both localisation (IoU≥threshold) and ID
            assignment are correct.

        Precision / Recall:
            Precision = TP / (TP + FP)
            Recall    = TP / (TP + FN)

    Ground truth format (MOT Challenge):
        CSV with columns: frame, id, x, y, w, h, conf, class, visibility
        (conf=1 for GT entries; only rows with conf≥1 are used as GT)

    Args:
        iou_threshold:  Minimum IoU overlap to count a detection as TP.
        class_filter:   Optional list of class IDs to evaluate (None = all).
    """

    def __init__(
        self,
        iou_threshold: float = EVAL_IOU_THRESHOLD,
        class_filter: Optional[list[int]] = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.class_filter  = class_filter
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"MOTEvaluator(iou_threshold={self.iou_threshold}, "
            f"class_filter={self.class_filter})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "MOTEvaluator":
        """Construct from config dict."""
        return cls(
            iou_threshold=config.get("iou_threshold", EVAL_IOU_THRESHOLD),
            class_filter=config.get("class_filter"),
        )

    @staticmethod
    def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two xyxy boxes.

        Args:
            box_a: ``(4,)`` xyxy.
            box_b: ``(4,)`` xyxy.

        Returns:
            IoU float in [0, 1].
        """
        xi1 = max(box_a[0], box_b[0])
        yi1 = max(box_a[1], box_b[1])
        xi2 = min(box_a[2], box_b[2])
        yi2 = min(box_a[3], box_b[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1    = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
        a2    = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
        return float(inter / (a1 + a2 - inter + 1e-8))

    def load_gt(self, gt_path: str) -> dict[int, list[dict]]:
        """Parse a MOT-Challenge format ground truth file.

        Args:
            gt_path: Path to GT CSV (frame, id, x, y, w, h, conf, …).

        Returns:
            Dict mapping frame_id → list of GT dicts
            (``id``, ``box_xyxy`` as np.ndarray).

        Raises:
            FileNotFoundError: GT file not found.
        """
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(
                f"GT file not found: '{gt_path}'\n"
                "Provide a MOT-Challenge format ground truth CSV."
            )

        gt: dict[int, list[dict]] = defaultdict(list)
        with open(gt_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    frame = int(row[0])
                    obj_id = int(row[1])
                    x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                    conf = float(row[6]) if len(row) > 6 else 1.0
                except ValueError:
                    continue  # skip header rows
                if conf < 1:
                    continue  # ignore ignored/distractor regions
                box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
                gt[frame].append({"id": obj_id, "box_xyxy": box_xyxy})

        self._logger.info(
            "Loaded GT: %d frames, %d total objects",
            len(gt), sum(len(v) for v in gt.values()),
        )
        return dict(gt)

    def evaluate(
        self,
        all_results: list[DetectionResult],
        gt: dict[int, list[dict]],
    ) -> dict:
        """Compute MOTA, IDF1, precision, recall across all frames.

        Matching algorithm (greedy, per frame):
            1. For each frame, collect predicted boxes and GT boxes.
            2. Compute pairwise IoU matrix (pred × gt).
            3. Greedily match highest-IoU pairs above threshold.
            4. Matched predictions → TP; unmatched pred → FP; unmatched GT → FN.

        ID switch detection:
            A switch occurs when a previously matched GT object is now matched
            to a different track ID than in the last frame it was observed.

        Args:
            all_results: List of ``DetectionResult`` objects (one per frame).
            gt:          GT dict from ``load_gt``.

        Returns:
            Dict with ``MOTA``, ``IDF1``, ``precision``, ``recall``,
            ``num_id_switches``, ``num_fp``, ``num_fn``, ``num_gt``.
        """
        total_tp = total_fp = total_fn = total_gt = 0
        total_id_switches = 0
        id_idtp = id_idfp = id_idfn = 0

        # Track last known GT ID → track ID assignment per GT object
        gt_to_track: dict[int, int] = {}

        for result in all_results:
            frame_gt = gt.get(result.frame_id + 1, [])  # MOT frames are 1-indexed
            total_gt += len(frame_gt)

            if len(result) == 0:
                total_fn += len(frame_gt)
                continue

            pred_boxes = result.boxes_xyxy
            pred_tids  = result.track_ids
            gt_boxes   = [g["box_xyxy"] for g in frame_gt]
            gt_ids     = [g["id"]       for g in frame_gt]

            # Build IoU matrix
            n_pred, n_gt = len(pred_boxes), len(gt_boxes)
            iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
            for pi in range(n_pred):
                for gi in range(n_gt):
                    iou_mat[pi, gi] = self._box_iou(pred_boxes[pi], gt_boxes[gi])

            # Greedy matching (descending IoU)
            matched_pred = set()
            matched_gt   = set()
            pairs: list[tuple[int, int]] = []

            flat_order = np.argsort(iou_mat.ravel())[::-1]
            for flat_idx in flat_order:
                pi, gi = divmod(int(flat_idx), n_gt)
                if iou_mat[pi, gi] < self.iou_threshold:
                    break
                if pi in matched_pred or gi in matched_gt:
                    continue
                pairs.append((pi, gi))
                matched_pred.add(pi)
                matched_gt.add(gi)

            tp = len(pairs)
            fp = n_pred - tp
            fn = n_gt   - tp
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # ID switch detection + IDF1 accumulation
            for pi, gi in pairs:
                gt_id  = gt_ids[gi]
                tid    = int(pred_tids[pi])
                prev   = gt_to_track.get(gt_id)
                if prev is not None and prev != tid:
                    total_id_switches += 1
                gt_to_track[gt_id] = tid
                id_idtp += 1

            id_idfp += fp
            id_idfn += fn

        mota = (
            1.0 - (total_fn + total_fp + total_id_switches) / max(total_gt, 1)
        )
        idf1 = (
            2 * id_idtp / max(2 * id_idtp + id_idfp + id_idfn, 1)
        )
        precision = total_tp / max(total_tp + total_fp, 1)
        recall    = total_tp / max(total_tp + total_fn,  1)

        metrics = {
            "MOTA":             float(mota),
            "IDF1":             float(idf1),
            "precision":        float(precision),
            "recall":           float(recall),
            "num_tp":           total_tp,
            "num_fp":           total_fp,
            "num_fn":           total_fn,
            "num_gt":           total_gt,
            "num_id_switches":  total_id_switches,
        }

        self._logger.info(
            "MOTA=%.4f  IDF1=%.4f  P=%.4f  R=%.4f  IDSW=%d",
            mota, idf1, precision, recall, total_id_switches,
        )
        return metrics


# ============================================================
# CLASS: TrackExporter
# ============================================================
class TrackExporter:
    """Export tracking results to MOT-format CSV or JSON.

    MOT CSV format (compatible with py-motmetrics):
        frame, id, x, y, w, h, conf, -1, -1, -1

    JSON format:
        List of frame dicts, each with a list of detection dicts.

    Args:
        output_dir: Directory for exported files.
    """

    def __init__(self, output_dir: str = LOG_DIR) -> None:
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"TrackExporter(output_dir='{self.output_dir}')"

    @classmethod
    def from_config(cls, config: dict) -> "TrackExporter":
        """Construct from config dict."""
        return cls(output_dir=config.get("log_dir", LOG_DIR))

    def to_mot_csv(
        self,
        all_results: list[DetectionResult],
        filename: str = "tracks.csv",
    ) -> str:
        """Export tracking results in MOT Challenge CSV format.

        Format: frame, id, x, y, w, h, conf, -1, -1, -1

        Args:
            all_results: List of per-frame ``DetectionResult`` objects.
            filename:    Output filename.

        Returns:
            Full path to saved CSV.
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for result in all_results:
                for i in range(len(result)):
                    tid  = int(result.track_ids[i])
                    if tid < 0:
                        continue
                    x1, y1, x2, y2 = result.boxes_xyxy[i]
                    x, y = float(x1), float(y1)
                    w    = float(x2 - x1)
                    h    = float(y2 - y1)
                    conf = float(result.confidences[i])
                    # MOT format: frame is 1-indexed
                    writer.writerow([result.frame_id + 1, tid,
                                     round(x, 2), round(y, 2),
                                     round(w, 2), round(h, 2),
                                     round(conf, 4), -1, -1, -1])

        self._logger.info("MOT CSV exported → %s  (%d frames)", path, len(all_results))
        return path

    def to_json(
        self,
        all_results: list[DetectionResult],
        filename: str = "tracks.json",
    ) -> str:
        """Export tracking results as JSON.

        Args:
            all_results: List of per-frame ``DetectionResult`` objects.
            filename:    Output filename.

        Returns:
            Full path to saved JSON.
        """
        data = [r.to_dict() for r in all_results]
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._logger.info("JSON exported → %s  (%d frames)", path, len(all_results))
        return path

    def to_csv_summary(
        self,
        track_history: TrackHistory,
        filename: str = "track_summary.csv",
    ) -> str:
        """Export per-track summary statistics to CSV.

        Columns: track_id, lifetime_frames, mean_speed_px_per_sec.

        Args:
            track_history: ``TrackHistory`` instance after processing.
            filename:      Output filename.

        Returns:
            Full path to saved CSV.
        """
        path = os.path.join(self.output_dir, filename)
        summary = track_history.summary()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "lifetime_frames", "speed_px_s"])
            writer.writeheader()
            for tid in summary["active_track_ids"]:
                writer.writerow({
                    "track_id":        tid,
                    "lifetime_frames": track_history.track_lifetime(tid),
                    "speed_px_s":      round(track_history.estimate_speed(tid), 2),
                })
        self._logger.info("Track summary → %s", path)
        return path


# ============================================================
# CLASS: Visualizer
# ============================================================
class Visualizer:
    """Annotate frames with bounding boxes, track IDs, trails, and metrics.

    Args:
        output_dir:       Directory for saved outputs.
        trail_thickness:  Pixel width for trail lines.
        box_thickness:    Pixel width for bounding boxes.
        label_font_scale: OpenCV font scale for track ID labels.
    """

    # 20 visually distinct BGR colours cycling by track ID
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
        trail_thickness: int = 2,
        box_thickness: int = 2,
        label_font_scale: float = 0.55,
    ) -> None:
        self.output_dir       = output_dir
        self.trail_thickness  = trail_thickness
        self.box_thickness    = box_thickness
        self.label_font_scale = label_font_scale
        ensure_dir(output_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"Visualizer(output_dir='{self.output_dir}')"

    @classmethod
    def from_config(cls, config: dict) -> "Visualizer":
        """Construct from config dict."""
        return cls(output_dir=config.get("log_dir", LOG_DIR))

    def _color(self, track_id: int) -> tuple[int, int, int]:
        """Return a consistent BGR colour for a given track ID."""
        return self.PALETTE[track_id % len(self.PALETTE)]

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        track_history: Optional[TrackHistory] = None,
        show_conf: bool = True,
        show_speed: bool = False,
    ) -> np.ndarray:
        """Draw tracks, trails, and labels on a single frame.

        Args:
            frame:         BGR uint8 image.
            result:        ``DetectionResult`` for this frame.
            track_history: Optional history for trail drawing.
            show_conf:     If True, include confidence in label.
            show_speed:    If True, include speed estimate in label.

        Returns:
            Annotated BGR image.
        """
        vis = frame.copy()

        for i in range(len(result)):
            tid  = int(result.track_ids[i])
            if tid < 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in result.boxes_xyxy[i]]
            conf    = result.confidences[i]
            cls_id  = int(result.class_ids[i])
            name    = result.class_names.get(cls_id, str(cls_id))
            color   = self._color(tid)

            # Draw trail
            if track_history is not None:
                trail = track_history.get_trail(tid)
                for k in range(1, len(trail)):
                    pt1 = (int(trail[k-1][0]), int(trail[k-1][1]))
                    pt2 = (int(trail[k][0]),   int(trail[k][1]))
                    # Fade older trail segments
                    alpha = k / len(trail)
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(vis, pt1, pt2, faded, self.trail_thickness, cv2.LINE_AA)

            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, self.box_thickness)

            # Build label
            parts = [f"#{tid}", name]
            if show_conf:
                parts.append(f"{conf:.2f}")
            if show_speed and track_history:
                spd = track_history.estimate_speed(tid)
                parts.append(f"{spd:.0f}px/s")
            label = " ".join(parts)

            (tw, th), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale, 1
            )
            ly = max(y1, th + 4)
            cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw, ly + bl - 2), color, -1)
            cv2.putText(
                vis, label, (x1, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        # Frame stats overlay
        n_active = (result.track_ids >= 0).sum()
        cv2.putText(
            vis, f"Frame {result.frame_id}  Tracks: {n_active}",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return vis

    def save_frame(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        track_history: Optional[TrackHistory],
        filename: str,
    ) -> str:
        """Annotate and save a single frame.

        Args:
            frame:         BGR image.
            result:        Detection result for this frame.
            track_history: Optional trail history.
            filename:      Output filename.

        Returns:
            Full path to saved file.
        """
        annotated = self.annotate_frame(frame, result, track_history)
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, annotated)
        self._logger.info("Frame saved → %s", path)
        return path

    def plot_track_count_over_time(
        self,
        all_results: list[DetectionResult],
        filename: str = "track_count.png",
    ) -> str:
        """Line chart of active track count per frame.

        Args:
            all_results: List of per-frame results.
            filename:    Output filename.

        Returns:
            Full path to saved figure.
        """
        frames = [r.frame_id for r in all_results]
        counts = [(r.track_ids >= 0).sum() for r in all_results]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, counts, "b-", linewidth=1.2)
        ax.fill_between(frames, counts, alpha=0.2)
        ax.set_xlabel("Frame"); ax.set_ylabel("Active Tracks")
        ax.set_title("Active Track Count Over Time")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Track count chart saved → %s", path)
        return path

    def plot_track_lifetimes(
        self,
        track_history: TrackHistory,
        filename: str = "track_lifetimes.png",
    ) -> str:
        """Histogram of track lifetime (frames alive) distribution.

        Args:
            track_history: ``TrackHistory`` after processing.
            filename:      Output filename.

        Returns:
            Full path to saved figure.
        """
        summary = track_history.summary()
        ids = summary["active_track_ids"]
        lifetimes = [track_history.track_lifetime(tid) for tid in ids]

        if not lifetimes:
            self._logger.warning("No tracks to plot.")
            return ""

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(lifetimes, bins=max(10, len(lifetimes) // 5),
                color="steelblue", edgecolor="white")
        ax.axvline(float(np.mean(lifetimes)), color="red", linestyle="--",
                   label=f"mean={np.mean(lifetimes):.1f} frames")
        ax.set_xlabel("Lifetime (frames)"); ax.set_ylabel("Count")
        ax.set_title("Track Lifetime Distribution")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Track lifetime histogram saved → %s", path)
        return path

    def plot_metrics_bar(
        self,
        metrics: dict,
        filename: str = "mot_metrics.png",
    ) -> str:
        """Bar chart of key tracking metrics (MOTA, IDF1, P, R).

        Args:
            metrics:  Output of ``MOTEvaluator.evaluate``.
            filename: Output filename.

        Returns:
            Full path to saved figure.
        """
        keys   = ["MOTA", "IDF1", "precision", "recall"]
        values = [metrics.get(k, 0.0) for k in keys]

        colors = ["steelblue" if v >= 0 else "tomato" for v in values]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(keys, [max(v, 0) for v in values], color=colors)
        ax.set_ylim(0, 1.1)
        ax.set_title("MOT Metrics")
        ax.set_ylabel("Score")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self._logger.info("Metrics bar chart saved → %s", path)
        return path


# ============================================================
# CLASS: TrackingPipeline  (orchestrator)
# ============================================================
class TrackingPipeline:
    """End-to-end multi-object tracking orchestrator.

    Combines:
        - YOLOv8 object detector (detection).
        - ByteTrack (association + trajectory management).
        - TrackHistory (centroid trails, speed estimation).
        - Visualizer (annotated video output).
        - TrackExporter (CSV / JSON export).
        - MOTEvaluator (optional MOTA / IDF1 evaluation).

    Args:
        mode:        ``"track"`` or ``"evaluate"``.
        source:      Video path, image path, or webcam index.
        weights:     YOLO ``.pt`` path or model-size string.
        gt_file:     GT CSV path for ``"evaluate"`` mode.
        class_ids:   Optional list of class IDs to track (``None`` = all).
        export:      Output format: ``"json"``, ``"csv"``, ``"both"``, or ``None``.
        output_dir:  Directory for all outputs.
        config:      Optional flat config dict.

    Example::

        # Basic tracking
        pipeline = TrackingPipeline(source="video.mp4")
        pipeline.run()

        # Tracking with evaluation
        pipeline = TrackingPipeline(mode="evaluate", source="video.mp4",
                                    gt_file="gt.txt")
        pipeline.run()
    """

    def __init__(
        self,
        mode: str = "track",
        source: Union[str, int] = "",
        weights: str = YOLO_MODEL_SIZE,
        gt_file: Optional[str] = None,
        class_ids: Optional[list[int]] = None,
        export: Optional[str] = None,
        output_dir: str = LOG_DIR,
        config: Optional[dict] = None,
    ) -> None:
        self.mode       = mode
        self.source     = source
        self.weights    = weights
        self.gt_file    = gt_file
        self.class_ids  = class_ids
        self.export     = export
        self.output_dir = output_dir
        self.config     = config or {}
        self._logger    = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"TrackingPipeline(mode='{self.mode}', "
            f"source='{self.source}', weights='{self.weights}')"
        )

    @classmethod
    def from_config(cls, config: dict) -> "TrackingPipeline":
        """Construct from a flat config dictionary."""
        return cls(
            mode=config.get("mode", "track"),
            source=config.get("source", ""),
            weights=config.get("weights", YOLO_MODEL_SIZE),
            gt_file=config.get("gt_file"),
            class_ids=config.get("class_ids"),
            export=config.get("export"),
            output_dir=config.get("output_dir", LOG_DIR),
            config=config,
        )

    def _open_video(self) -> tuple[cv2.VideoCapture, int, float, int, int]:
        """Open the video source and return capture properties.

        Returns:
            Tuple of (cap, total_frames, fps, width, height).

        Raises:
            FileNotFoundError: Video file path does not exist.
            RuntimeError:      Cannot open video source.
        """
        src = self.source
        if isinstance(src, str) and src not in ("", "0") and not os.path.isfile(src):
            raise FileNotFoundError(
                f"Video file not found: '{src}'\n"
                "Provide a valid file path or webcam index."
            )

        cap = cv2.VideoCapture(int(src) if isinstance(src, str) and src.isdigit() else src)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video source: '{src}'\n"
                "Check the file path or webcam index."
            )

        fps    = cap.get(cv2.CAP_PROP_FPS) or float(BT_FRAME_RATE)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        self._logger.info(
            "Source: %s  %dx%d  %.1f fps  ~%d frames",
            src, width, height, fps, total,
        )
        return cap, total, fps, width, height

    def _load_detector(self) -> YOLO:
        """Load or download the YOLO detector.

        Returns:
            Loaded ``YOLO`` instance.

        Raises:
            FileNotFoundError: Custom weights path does not exist.
        """
        if self.weights.endswith(".pt") and not os.path.isfile(self.weights):
            raise FileNotFoundError(
                f"Weights file not found: '{self.weights}'\n"
                "Train a custom model first or use a model-size string like 'yolov8n'."
            )
        self._logger.info("Loading detector: %s", self.weights)
        return YOLO(self.weights)

    def run(self) -> dict:
        """Execute the pipeline.

        For ``"track"`` mode:
            1. Open video.
            2. Run YOLO detector on each frame.
            3. Pass detections to ByteTrack.
            4. Update TrackHistory.
            5. Annotate and write output video.
            6. Export tracks (if requested).
            7. Generate summary charts.

        For ``"evaluate"`` mode:
            Same as track, plus MOTA / IDF1 computation against GT.

        Returns:
            Dict with ``output_video``, ``track_summary``, and optionally ``metrics``.

        Raises:
            ValueError: Unknown mode.
        """
        self._logger.info("=" * 55)
        self._logger.info(
            "Tracking Pipeline  mode='%s'  source='%s'",
            self.mode, self.source,
        )
        self._logger.info("=" * 55)

        detector     = self._load_detector()
        tracker      = ByteTrackWrapper.from_config(self.config)
        history      = TrackHistory(fps=BT_FRAME_RATE)
        viz          = Visualizer(output_dir=self.output_dir)
        exporter     = TrackExporter(output_dir=self.output_dir)

        cap, total, fps, width, height = self._open_video()

        out_video_path = os.path.join(self.output_dir, "tracked_output.mp4")
        ensure_dir(self.output_dir)
        writer = cv2.VideoWriter(
            out_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        all_results: list[DetectionResult] = []
        frame_iter  = range(total) if total > 0 else iter(int, 1)  # infinite if webcam

        class_names: dict[int, str] = {}

        for frame_id in tqdm(
            frame_iter if total > 0 else iter(int, 1),
            total=total if total > 0 else None,
            desc="Tracking", ncols=90,
        ):
            ret, frame = cap.read()
            if not ret:
                break

            # ── Detect ────────────────────────────────────────────────
            det_results = detector.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )

            if not class_names:
                class_names = detector.names

            # Convert to supervision Detections
            sv_dets = sv.Detections.from_ultralytics(det_results[0])

            # Filter by class if requested
            if self.class_ids:
                mask    = np.isin(sv_dets.class_id, self.class_ids)
                sv_dets = sv_dets[mask]

            # ── Track ──────────────────────────────────────────────────
            tracked = tracker.update(sv_dets, frame.shape[:2])

            # Build DetectionResult
            if len(tracked) > 0:
                boxes   = tracked.xyxy.astype(np.float32)
                confs   = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
                cls_ids = tracked.class_id.astype(np.int32) if tracked.class_id is not None else np.zeros(len(tracked), dtype=np.int32)
                tids    = tracked.tracker_id.astype(np.int32) if tracked.tracker_id is not None else np.full(len(tracked), -1, dtype=np.int32)
            else:
                boxes   = np.empty((0, 4), dtype=np.float32)
                confs   = np.empty(0, dtype=np.float32)
                cls_ids = np.empty(0, dtype=np.int32)
                tids    = np.empty(0, dtype=np.int32)

            result = DetectionResult(
                frame_id=frame_id,
                boxes_xyxy=boxes,
                confidences=confs,
                class_ids=cls_ids,
                track_ids=tids,
                class_names=class_names,
            )
            all_results.append(result)
            history.update(result)

            # ── Annotate + write ───────────────────────────────────────
            annotated = viz.annotate_frame(frame, result, history, show_speed=True)
            writer.write(annotated)

        cap.release()
        writer.release()
        self._logger.info("Annotated video saved → %s", out_video_path)

        # ── Exports ────────────────────────────────────────────────────
        export_paths: dict[str, str] = {}
        if self.export in ("csv", "both"):
            export_paths["mot_csv"]      = exporter.to_mot_csv(all_results)
            export_paths["track_summary"] = exporter.to_csv_summary(history)
        if self.export in ("json", "both"):
            export_paths["json"] = exporter.to_json(all_results)

        # ── Charts ─────────────────────────────────────────────────────
        chart_paths: dict[str, str] = {}
        if all_results:
            chart_paths["track_count"]   = viz.plot_track_count_over_time(all_results)
            chart_paths["track_lifetime"] = viz.plot_track_lifetimes(history)

        # ── Evaluation ─────────────────────────────────────────────────
        result_dict: dict = {
            "output_video":  out_video_path,
            "track_summary": history.summary(),
            "export_paths":  export_paths,
            "chart_paths":   chart_paths,
        }

        if self.mode == "evaluate":
            if not self.gt_file:
                raise ValueError("--gt-file is required for evaluate mode.")
            evaluator = MOTEvaluator.from_config(self.config)
            gt        = evaluator.load_gt(self.gt_file)
            metrics   = evaluator.evaluate(all_results, gt)
            chart_paths["metrics"] = viz.plot_metrics_bar(metrics)

            # Save metrics JSON
            mpath = os.path.join(self.output_dir, "mot_metrics.json")
            with open(mpath, "w") as f:
                json.dump(metrics, f, indent=2)
            self._logger.info("Metrics → %s", mpath)

            result_dict["metrics"] = metrics

        self._logger.info(
            "Done. Total tracks: %d", history.summary()["total_tracks"]
        )
        return result_dict


# ============================================================
# ENTRY POINT
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Object Tracking — Section 7 (ByteTrack)"
    )
    parser.add_argument(
        "--mode", choices=["track", "evaluate"], default="track",
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--source", type=str, default="",
        help="Video path, image path, or webcam index (e.g. '0').",
    )
    parser.add_argument(
        "--weights", type=str, default=YOLO_MODEL_SIZE,
        help="YOLO weights: model-size string or path to .pt file.",
    )
    parser.add_argument(
        "--gt-file", type=str, default=None,
        help="MOT-format ground truth CSV for evaluation mode.",
    )
    parser.add_argument(
        "--class-ids", type=int, nargs="+", default=None,
        help="Class IDs to track (default: all classes).",
    )
    parser.add_argument(
        "--export", choices=["json", "csv", "both"], default=None,
        help="Export track data format.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=LOG_DIR,
        help="Directory for output files.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    pipeline = TrackingPipeline(
        mode=args.mode,
        source=args.source,
        weights=args.weights,
        gt_file=args.gt_file,
        class_ids=args.class_ids,
        export=args.export,
        output_dir=args.output_dir,
    )
    result = pipeline.run()
    printable = {
        k: v for k, v in result.items()
        if isinstance(v, (str, dict, float, int))
    }
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
