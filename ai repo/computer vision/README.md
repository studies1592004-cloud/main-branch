# Computer Vision

A complete, self-contained Computer Vision curriculum covering classical methods through modern deep learning. The folder is split into **theory** (slide decks) and **code** — which contains theory implementation scripts, industry-standard OOP pipelines, hands-on Colab notebooks, and a test suite.

---

## Folder Structure

```
computer_vision/
├── README.md
├── requirements.txt
├── theory/                              ← 12 lecture slide decks (367 slides total)
└── code/
    ├── theory implementation/           ← Section reference scripts (one per lecture)
    │   ├── cv_A_image_fundamentals.py
    │   ├── cv_B_filters_convolution.py
    │   ├── ...
    │   └── cv_L_geometry.py
    ├── standardized/                    ← Industry-standard OOP pipelines
    │   ├── classical_cv_pipeline.py
    │   ├── image_classification.py
    │   ├── object_detection.py
    │   ├── semantic_segmentation.py
    │   ├── instance_segmentation.py
    │   ├── pose_estimation.py
    │   └── multi_object_tracking.py
    ├── tests/                           ← pytest suite for all standardized pipelines
    │   ├── conftest.py
    │   ├── test_classical_cv_pipeline.py
    │   ├── test_image_classification.py
    │   ├── test_object_detection.py
    │   ├── test_semantic_segmentation.py
    │   ├── test_instance_segmentation.py
    │   ├── test_pose_estimation.py
    │   └── test_multi_object_tracking.py
    └── google colab notebooks/          ← Runnable notebooks + saved outputs
        ├── basics/
        ├── OpenCV/
        ├── object classification/       ← ResNet-50 & ViT-B/16 + Emotion dataset
        ├── object segmentation/
        ├── object tracking/
        └── pose estimation/
```

---

## Theory — Slide Decks

12 lecture decks covering the full CV stack from first principles to production models.

| Section | Topic | Slides |
|---------|-------|--------|
| A | Image Formation & Signals — pixels, colour spaces, Fourier, sampling | 40 |
| B | Image Filters & Convolution — kernels, Gaussian, sharpening, separability | 35 |
| C | Image Processing Fundamentals — histograms, morphology, thresholding, transforms | 29 |
| D | Classical Object Detection — HOG-SVM, Haar cascades, sliding window, NMS | 35 |
| E | Optical Flow & Motion — Lucas-Kanade, Farneback, Horn-Schunck, motion segmentation | 28 |
| F | Neural Network Basics — perceptron, backprop, activations, regularisation | 34 |
| G | Convolutional Neural Networks — conv layers, pooling, receptive field, architectures | 30 |
| H | Training Large Models & Optimization — SGD/Adam, LR schedules, batch norm, mixed precision | 28 |
| I | Classification & Transfer Learning — fine-tuning, feature extraction, ResNet, ViT | 27 |
| J | Object Detection (Full Evolution) — R-CNN → Fast/Faster → YOLO → DETR | 27 |
| K | Evaluation Metrics & Loss Functions — IoU, mAP, confusion matrix, Dice, focal loss | 28 |
| L | Keypoint Detection, Pose & Geometry — SIFT, stereo, homography, pose estimation | 26 |
| | **Total** | **367** |

---

## Code — Theory Implementation

One Python file per theory section inside `code/theory implementation/`. Each file is a standalone educational module demonstrating the core algorithms from that section using NumPy and OpenCV, with no external training dependencies.

| File | Section | Key Demonstrations |
|------|---------|--------------------|
| `cv_A_image_fundamentals.py` | A | Pixel manipulation, colour spaces, Fourier transform, sampling |
| `cv_B_filters_convolution.py` | B | Gaussian/Sobel/Laplacian kernels, separable filtering, frequency response |
| `cv_C_image_processing.py` | C | CLAHE, morphological ops, Otsu thresholding, Hough transform |
| `cv_D_classical_detection.py` | D | HOG feature extraction, sliding window, Haar-like features, NMS |
| `cv_E_optical_flow.py` | E | Lucas-Kanade sparse flow, Farneback dense flow, motion segmentation |
| `cv_F_neural_networks.py` | F | MLP from scratch, backprop, activations, XOR demo |
| `cv_G_cnns.py` | G | Conv2D from scratch, pooling, receptive field calculation |
| `cv_H_training.py` | H | SGD/Adam/AdaGrad, LR schedules, gradient clipping, batch norm |
| `cv_I_transfer_learning.py` | I | ResNet/ViT fine-tuning, feature extraction, layer freezing |
| `cv_J_object_detection.py` | J | Anchor generation, box encoding/decoding, IoU, YOLO inference |
| `cv_K_metrics_losses.py` | K | mAP, IoU, Dice, focal loss, confusion matrix, PR curves |
| `cv_L_geometry.py` | L | SIFT descriptors, homography, stereo disparity, essential matrix |

**Run any file directly:**
```bash
python "code/theory implementation/cv_A_image_fundamentals.py"
```

---

## Code — Standardized OOP Pipelines

Seven industry-grade pipelines in `code/standardized/`. Each file follows the same conventions:

- Full OOP with `__init__`, `__repr__`, `from_config()`, Google docstrings, type hints, logging
- Global config block at the top — edit once before running, no CLI flags needed for basic use
- Roboflow dataset integration (free API key at roboflow.com)
- Train → evaluate → infer modes via `--mode` flag
- Standalone — no imports between files

### 1. `classical_cv_pipeline.py`
Pure NumPy/OpenCV pipeline. No deep learning, no API key needed.

```bash
pip install numpy opencv-python matplotlib
python classical_cv_pipeline.py
```

**Classes:** `ImageLoader`, `FilteringProcessor`, `EdgeDetector`, `HistogramEqualizer`, `MorphologyProcessor`, `TemplateMatcher`, `HOGExtractor`, `OpticalFlowEstimator`, `Visualizer`, `ClassicalCVPipeline`

---

### 2. `image_classification.py`
Transfer learning with ResNet-50 / EfficientNet / ViT on a Roboflow dataset.

```bash
pip install torch torchvision roboflow matplotlib scikit-learn seaborn tqdm
python image_classification.py --mode fine_tuning
```

**Dataset:** Rock-Paper-Scissors (Roboflow Universe)
**Classes:** `DatasetDownloader`, `TransformBuilder`, `ClassificationDataset`, `ModelBuilder`, `Trainer`, `Evaluator`, `Visualizer`, `ClassificationPipeline`

---

### 3. `object_detection.py`
YOLOv8 object detection with from-scratch NMS and AP evaluation.

```bash
pip install ultralytics roboflow opencv-python matplotlib numpy scikit-learn seaborn tqdm
python object_detection.py --mode train
```

**Dataset:** Hard Hat Sample (Roboflow Universe)
**Classes:** `DatasetDownloader`, `Preprocessor`, `PostProcessor`, `Detector`, `Evaluator`, `Visualizer`, `DetectionPipeline`

---

### 4. `semantic_segmentation.py`
DeepLabV3+ (ResNet-101 + ASPP) and SegFormer-B2 with combined CE + Dice loss.

```bash
pip install torch torchvision roboflow opencv-python matplotlib scikit-learn seaborn tqdm transformers
python semantic_segmentation.py --backbone deeplabv3plus
python semantic_segmentation.py --backbone segformer
```

**Dataset:** Sidewalk Semantic (Roboflow Universe)
**Classes:** `DatasetDownloader`, `SegmentationDataset`, `ModelBuilder`, `SegmentationLoss`, `MetricsCalculator`, `Trainer`, `Visualizer`, `SegmentationPipeline`, `SegFormerWrapper`

---

### 5. `instance_segmentation.py`
YOLOv8-seg (32-prototype mask branch) and Mask R-CNN (ResNet-50-FPN, two-stage).

```bash
pip install ultralytics roboflow opencv-python matplotlib numpy tqdm torch torchvision
python instance_segmentation.py --mode train --backbone yolov8seg
python instance_segmentation.py --mode train --backbone maskrcnn
```

**Dataset:** Brain Tumor Detection (Roboflow Universe)
**Classes:** `DatasetDownloader`, `MaskProcessor`, `Segmentor`, `MaskRCNNSegmentor`, `Evaluator`, `Visualizer`, `InstanceSegPipeline`

---

### 6. `pose_estimation.py`
YOLOv8-pose (direct keypoint regression) and MediaPipe Holistic (33 BlazePose landmarks). Includes COCO OKS metric, Procrustes alignment, and rule-based action classification.

```bash
pip install ultralytics roboflow opencv-python matplotlib numpy mediapipe tqdm
python pose_estimation.py --mode train --backend yolov8pose
python pose_estimation.py --mode infer --backend mediapipe --source image.jpg
python pose_estimation.py --mode infer --source 0          # webcam
```

**Dataset:** Yoga Pose (Roboflow Universe)
**Classes:** `DatasetDownloader`, `KeypointNormalizer`, `PoseEstimator`, `MediaPipeEstimator`, `ActionClassifier`, `Evaluator`, `Visualizer`, `PosePipeline`

---

### 7. `multi_object_tracking.py`
ByteTrack via supervision — two-stage high/low-confidence matching with Kalman filter motion prediction. Includes MOTA/IDF1 evaluation and MOT-format export.

```bash
pip install ultralytics supervision opencv-python matplotlib numpy scipy tqdm
python multi_object_tracking.py --source video.mp4
python multi_object_tracking.py --source 0                   # webcam
python multi_object_tracking.py --source video.mp4 --export both
python multi_object_tracking.py --mode evaluate --source video.mp4 --gt-file gt.txt
```

**Classes:** `DetectionResult`, `ByteTrackWrapper`, `TrackHistory`, `MOTEvaluator`, `TrackExporter`, `Visualizer`, `TrackingPipeline`

---

## Code — Tests

`code/tests/` contains a full pytest suite covering all seven standardized pipelines — **263 tests across 7 files**.

```bash
pip install pytest numpy opencv-python torch torchvision
pytest code/tests/ -v

# Run a single file
pytest code/tests/test_multi_object_tracking.py -v

# Skip slow or GPU tests
pytest code/tests/ -m "not slow" -v
```

All external dependencies (Ultralytics, Roboflow, MediaPipe, supervision) are mocked so every test runs **without a GPU, API key, or internet connection**.

| Test File | Classes Tested | Tests |
|-----------|----------------|-------|
| `test_classical_cv_pipeline.py` | `ImageLoader`, `FilteringProcessor`, `EdgeDetector`, `HistogramEqualizer`, `MorphologyProcessor`, `TemplateMatcher`, `HOGExtractor`, `OpticalFlowEstimator`, `ClassicalCVPipeline` | 55 |
| `test_multi_object_tracking.py` | `DetectionResult`, `TrackHistory`, `MOTEvaluator`, `TrackExporter`, `Visualizer` | 56 |
| `test_instance_segmentation.py` | `MaskProcessor`, `Evaluator`, `Visualizer` | 36 |
| `test_pose_estimation.py` | `KeypointNormalizer`, `ActionClassifier`, `Evaluator`, `Visualizer` | 36 |
| `test_image_classification.py` | `TransformBuilder`, `ModelBuilder`, `Trainer`, `Evaluator`, `Visualizer` | 30 |
| `test_object_detection.py` | `Preprocessor`, `PostProcessor`, `Evaluator`, `Visualizer` | 28 |
| `test_semantic_segmentation.py` | `SegmentationLoss`, `MetricsCalculator`, `Visualizer` | 22 |
| **Total** | | **263** |

---

## Code — Google Colab Notebooks

Hands-on experiments and projects. Each subfolder contains notebooks plus saved predictions, models, and result artefacts.

### `basics/`
| Notebook | Topics |
|----------|--------|
| `cv basics.ipynb` | Images, colour spaces, basic ops |
| `histogram equalization.ipynb` | CLAHE, AHE, global HE |
| `morphological operations.ipynb` | Erosion, dilation, opening, closing, top-hat |
| `hog-svm & haar-cascade.ipynb` | HOG feature extraction, SVM classifier, Haar face detector |
| `farneback dense flow & lucas kanade tracking & horn schunck & motion segmentation.ipynb` | All four optical flow methods compared |
| `object tracking using lukas kanade.ipynb` | Feature point tracking across frames |
| `sort.ipynb` | SORT tracker (Kalman + Hungarian) from scratch |

### `OpenCV/`
14 structured notebooks covering the full OpenCV API: image I/O, annotation, mathematical operations, camera access, video writing, edge detection, image alignment, panorama stitching, HDR imaging, object tracking, face detection, TensorFlow object detection, and OpenPose.

### `object classification/`
Fine-tuned **ResNet-50** and **ViT-B/16** on the **Emotion Recognition** dataset (Roboflow Universe — facial expression classification across angry, sad, and other emotion classes). The full dataset is included locally under `Emotion-Recognition-1/` with train, valid, and test splits organised into per-class subfolders. Saved `.pth` weights, confusion matrices, and `results.json` with per-class metrics are included for both models.

### `object segmentation/`
YOLOv8 instance segmentation trained end-to-end. Includes full training run artefacts (loss curves, PR curves, confusion matrices, sample batch visualisations) across `training/`, `testing/`, and `validation/` splits, plus saved `best.pt` and `last.pt` weights.

### `object tracking/`
YOLOv8 detection + ByteTrack on a custom video. Annotated output video saved to `predictions/`.

### `pose estimation/`
MediaPipe Holistic and YOLOv8-pose on image and video. Both backends are compared with saved image and video predictions across separate output folders.

---

## Requirements

See `requirements.txt` in this folder. Install everything at once:

```bash
pip install -r requirements.txt
```

Or install only what you need per pipeline — each standardized file lists its own dependencies in the header docstring.

---

## Key Concepts Covered

**Classical CV** — convolution, Fourier analysis, morphological operations, histogram processing, HOG, SIFT, optical flow (Lucas-Kanade, Farneback, Horn-Schunck), homography, stereo vision, epipolar geometry.

**Deep Learning Foundations** — MLP from scratch, backpropagation, activations, batch normalisation, dropout, weight initialisation, mixed-precision training, gradient clipping.

**Modern Architectures** — ResNet, EfficientNet, ViT (Vision Transformer), DeepLabV3+ (ASPP + atrous convolutions), SegFormer (hierarchical transformer), Mask R-CNN (RPN + RoI-Align + mask head), YOLOv8 family (detection / segmentation / pose).

**Training & Evaluation** — transfer learning, fine-tuning, differential learning rates, cosine/poly LR schedules, early stopping, mAP@50:95, OKS, Dice, focal loss, MOTA, IDF1.

**Production Patterns** — OOP pipelines with `from_config()`, logging, type hints, Google docstrings, error handling, Roboflow dataset integration, MOT-format export, ByteTrack association.
