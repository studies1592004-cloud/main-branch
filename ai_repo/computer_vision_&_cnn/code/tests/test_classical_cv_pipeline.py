"""
tests/test_classical_cv_pipeline.py
====================================
Unit tests for code/standardized/classical_cv_pipeline.py

Run:
    pytest tests/test_classical_cv_pipeline.py -v

Dependencies (no GPU, no API key needed):
    pip install numpy opencv-python pytest
"""

import numpy as np
import cv2
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "standardized"))
from classical_cv_pipeline import (
    ImageLoader,
    FilteringProcessor,
    EdgeDetector,
    HistogramEqualizer,
    MorphologyProcessor,
    TemplateMatcher,
    HOGExtractor,
    OpticalFlowEstimator,
    Visualizer,
    ClassicalCVPipeline,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_image():
    """128×128 random uint8 grayscale image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (128, 128), dtype=np.uint8)


@pytest.fixture
def bgr_image():
    """128×128 random uint8 BGR image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def checkerboard():
    """64×64 high-contrast checkerboard — good for edge/template tests."""
    img = np.zeros((64, 64), dtype=np.uint8)
    for r in range(64):
        for c in range(64):
            if (r // 8 + c // 8) % 2 == 0:
                img[r, c] = 255
    return img


# ── ImageLoader ───────────────────────────────────────────────────────────────

class TestImageLoader:
    def test_repr(self):
        loader = ImageLoader()
        assert "ImageLoader" in repr(loader)

    def test_from_config(self):
        loader = ImageLoader.from_config({"color_mode": "gray"})
        assert loader.color_mode == "gray"

    def test_from_array_bgr(self, bgr_image):
        loader = ImageLoader(color_mode="bgr")
        img = loader.from_array(bgr_image)
        assert img.shape == bgr_image.shape
        assert img.dtype == np.uint8

    def test_from_array_gray(self, bgr_image):
        loader = ImageLoader(color_mode="gray")
        img = loader.from_array(bgr_image)
        assert img.ndim == 2

    def test_from_array_rgb(self, bgr_image):
        loader = ImageLoader(color_mode="rgb")
        img = loader.from_array(bgr_image)
        assert img.shape == bgr_image.shape
        # R and B channels should be swapped vs BGR
        assert not np.array_equal(img[:, :, 0], bgr_image[:, :, 0])

    def test_load_missing_file_raises(self):
        loader = ImageLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/image.png")


# ── FilteringProcessor ────────────────────────────────────────────────────────

class TestFilteringProcessor:
    def test_repr(self):
        fp = FilteringProcessor()
        assert "FilteringProcessor" in repr(fp)

    def test_from_config(self):
        fp = FilteringProcessor.from_config({"kernel_size": 7})
        assert fp.kernel_size == 7

    def test_gaussian_blur_shape(self, gray_image):
        fp = FilteringProcessor(kernel_size=5)
        blurred = fp.gaussian_blur(gray_image)
        assert blurred.shape == gray_image.shape
        assert blurred.dtype == np.uint8

    def test_gaussian_blur_reduces_variance(self, gray_image):
        """Blurring should reduce pixel variance."""
        fp = FilteringProcessor(kernel_size=15)
        blurred = fp.gaussian_blur(gray_image)
        assert blurred.astype(float).var() < gray_image.astype(float).var()

    def test_median_blur_shape(self, gray_image):
        fp = FilteringProcessor(kernel_size=5)
        result = fp.median_blur(gray_image)
        assert result.shape == gray_image.shape

    def test_sharpen_shape(self, gray_image):
        fp = FilteringProcessor()
        result = fp.sharpen(gray_image)
        assert result.shape == gray_image.shape

    def test_bilateral_filter_shape(self, gray_image):
        fp = FilteringProcessor()
        result = fp.bilateral_filter(gray_image)
        assert result.shape == gray_image.shape


# ── EdgeDetector ──────────────────────────────────────────────────────────────

class TestEdgeDetector:
    def test_repr(self):
        ed = EdgeDetector()
        assert "EdgeDetector" in repr(ed)

    def test_from_config(self):
        ed = EdgeDetector.from_config({"low_threshold": 30, "high_threshold": 90})
        assert ed.low_threshold == 30
        assert ed.high_threshold == 90

    def test_canny_output_shape(self, gray_image):
        ed = EdgeDetector()
        edges = ed.canny(gray_image)
        assert edges.shape == gray_image.shape
        assert edges.dtype == np.uint8

    def test_canny_detects_edges_on_checkerboard(self, checkerboard):
        ed = EdgeDetector(low_threshold=50, high_threshold=150)
        edges = ed.canny(checkerboard)
        # Checkerboard has many edges — should have substantial non-zero pixels
        assert edges.sum() > 0

    def test_sobel_output_shape(self, gray_image):
        ed = EdgeDetector()
        sobel = ed.sobel(gray_image)
        assert sobel.shape == gray_image.shape

    def test_laplacian_output_shape(self, gray_image):
        ed = EdgeDetector()
        lap = ed.laplacian(gray_image)
        assert lap.shape == gray_image.shape

    def test_canny_binary_values(self, gray_image):
        """Canny output must be binary (0 or 255)."""
        ed = EdgeDetector()
        edges = ed.canny(gray_image)
        unique = np.unique(edges)
        assert all(v in (0, 255) for v in unique)


# ── HistogramEqualizer ────────────────────────────────────────────────────────

class TestHistogramEqualizer:
    def test_repr(self):
        he = HistogramEqualizer()
        assert "HistogramEqualizer" in repr(he)

    def test_from_config(self):
        he = HistogramEqualizer.from_config({"clip_limit": 3.0})
        assert he.clip_limit == 3.0

    def test_equalize_global_shape(self, gray_image):
        he = HistogramEqualizer()
        result = he.equalize_global(gray_image)
        assert result.shape == gray_image.shape
        assert result.dtype == np.uint8

    def test_equalize_global_spreads_histogram(self):
        """After global equalisation, pixel range should be wider."""
        dark = np.full((64, 64), 50, dtype=np.uint8)
        dark[:32, :32] = 60   # small variation
        he = HistogramEqualizer()
        eq = he.equalize_global(dark)
        assert eq.max() - eq.min() >= dark.max() - dark.min()

    def test_clahe_output_shape(self, gray_image):
        he = HistogramEqualizer(clip_limit=2.0, tile_grid_size=(8, 8))
        result = he.clahe(gray_image)
        assert result.shape == gray_image.shape

    def test_equalize_color_shape(self, bgr_image):
        he = HistogramEqualizer()
        result = he.equalize_color(bgr_image)
        assert result.shape == bgr_image.shape


# ── MorphologyProcessor ───────────────────────────────────────────────────────

class TestMorphologyProcessor:
    def test_repr(self):
        mp = MorphologyProcessor()
        assert "MorphologyProcessor" in repr(mp)

    def test_from_config(self):
        mp = MorphologyProcessor.from_config({"kernel_size": 7})
        assert mp.kernel_size == 7

    def test_erode_reduces_white_area(self):
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[20:44, 20:44] = 255
        mp = MorphologyProcessor(kernel_size=5)
        eroded = mp.erode(binary, iterations=2)
        assert eroded.sum() <= binary.sum()

    def test_dilate_increases_white_area(self):
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[28:36, 28:36] = 255
        mp = MorphologyProcessor(kernel_size=5)
        dilated = mp.dilate(binary, iterations=2)
        assert dilated.sum() >= binary.sum()

    def test_opening_removes_small_noise(self):
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[10, 10] = 255   # isolated noise pixel
        binary[30:50, 30:50] = 255   # large blob
        mp = MorphologyProcessor(kernel_size=5)
        opened = mp.opening(binary)
        # Noise pixel should be removed
        assert opened[10, 10] == 0
        # Large blob should survive (partially)
        assert opened[35:45, 35:45].sum() > 0

    def test_closing_fills_gap(self):
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[20:44, 20:30] = 255
        binary[20:44, 34:44] = 255   # gap at cols 30–33
        mp = MorphologyProcessor(kernel_size=7)
        closed = mp.closing(binary)
        # Gap should be partially or fully filled
        assert closed[30, 31] == 255 or closed[30, 32] == 255

    def test_output_shape_preserved(self, gray_image):
        mp = MorphologyProcessor()
        for method in [mp.erode, mp.dilate, mp.opening, mp.closing]:
            result = method(gray_image)
            assert result.shape == gray_image.shape


# ── TemplateMatcher ───────────────────────────────────────────────────────────

class TestTemplateMatcher:
    def test_repr(self):
        tm = TemplateMatcher()
        assert "TemplateMatcher" in repr(tm)

    def test_from_config(self):
        tm = TemplateMatcher.from_config({"method": cv2.TM_SQDIFF})
        assert tm.method == cv2.TM_SQDIFF

    def test_exact_match_location(self):
        """Template extracted from a known location should match there."""
        rng = np.random.default_rng(0)
        image    = rng.integers(30, 200, (128, 128), dtype=np.uint8)
        template = image[40:60, 40:60].copy()
        tm = TemplateMatcher(method=cv2.TM_CCORR_NORMED)
        top_left, score = tm.match(image, template)
        assert top_left == (40, 40), f"Expected (40,40), got {top_left}"

    def test_exact_match_score_near_one(self):
        rng = np.random.default_rng(1)
        image    = rng.integers(30, 200, (128, 128), dtype=np.uint8)
        template = image[20:40, 50:70].copy()
        tm = TemplateMatcher(method=cv2.TM_CCORR_NORMED)
        _, score = tm.match(image, template)
        assert score > 0.99

    def test_returns_tuple(self, gray_image):
        tm = TemplateMatcher()
        template = gray_image[10:30, 10:30]
        result = tm.match(gray_image, template)
        assert isinstance(result, tuple) and len(result) == 2

    def test_template_larger_than_image_raises(self, gray_image):
        tm = TemplateMatcher()
        big_template = np.zeros((200, 200), dtype=np.uint8)
        with pytest.raises(Exception):
            tm.match(gray_image, big_template)


# ── HOGExtractor ──────────────────────────────────────────────────────────────

class TestHOGExtractor:
    def test_repr(self):
        hog = HOGExtractor()
        assert "HOGExtractor" in repr(hog)

    def test_from_config(self):
        hog = HOGExtractor.from_config({"orientations": 12, "pixels_per_cell": (8, 8)})
        assert hog.orientations == 12

    def test_descriptor_is_1d(self, gray_image):
        hog = HOGExtractor()
        desc = hog.extract(gray_image)
        assert desc.ndim == 1

    def test_descriptor_length_consistent(self, gray_image):
        """Same image should always produce the same length descriptor."""
        hog = HOGExtractor()
        d1 = hog.extract(gray_image)
        d2 = hog.extract(gray_image)
        assert len(d1) == len(d2)

    def test_descriptor_nonnegative(self, gray_image):
        """HOG descriptors use unsigned gradients — must be ≥ 0."""
        hog = HOGExtractor()
        desc = hog.extract(gray_image)
        assert (desc >= 0).all()

    def test_different_images_different_descriptors(self, gray_image):
        hog = HOGExtractor()
        rng = np.random.default_rng(99)
        other = rng.integers(0, 255, gray_image.shape, dtype=np.uint8)
        d1 = hog.extract(gray_image)
        d2 = hog.extract(other)
        assert not np.allclose(d1, d2)


# ── OpticalFlowEstimator ──────────────────────────────────────────────────────

class TestOpticalFlowEstimator:
    def test_repr(self):
        ofe = OpticalFlowEstimator()
        assert "OpticalFlowEstimator" in repr(ofe)

    def test_from_config(self):
        ofe = OpticalFlowEstimator.from_config({"max_corners": 50})
        assert ofe.max_corners == 50

    def test_lucas_kanade_tracks_points(self, gray_image):
        ofe = OpticalFlowEstimator(max_corners=20)
        # Shift frame1 by (3, 3) to create frame2
        frame1 = gray_image.copy()
        M = np.float32([[1, 0, 3], [0, 1, 3]])
        frame2 = cv2.warpAffine(frame1, M, (frame1.shape[1], frame1.shape[0]))
        pts1, pts2, status = ofe.lucas_kanade(frame1, frame2)
        # At least some points should be tracked
        tracked = (status.ravel() == 1).sum()
        assert tracked > 0

    def test_lucas_kanade_output_shapes(self, gray_image):
        ofe = OpticalFlowEstimator(max_corners=16)
        frame2 = gray_image.copy()
        pts1, pts2, status = ofe.lucas_kanade(gray_image, frame2)
        assert pts1.shape[-1] == 2
        assert pts2.shape[-1] == 2
        assert len(status) == len(pts1)

    def test_farneback_output_shape(self, gray_image):
        ofe = OpticalFlowEstimator()
        flow = ofe.farneback(gray_image, gray_image)
        assert flow.shape == (gray_image.shape[0], gray_image.shape[1], 2)

    def test_farneback_zero_motion_on_identical_frames(self, gray_image):
        ofe = OpticalFlowEstimator()
        flow = ofe.farneback(gray_image, gray_image)
        assert np.abs(flow).mean() < 0.5   # near-zero motion


# ── ClassicalCVPipeline ───────────────────────────────────────────────────────

class TestClassicalCVPipeline:
    def test_repr(self):
        pipe = ClassicalCVPipeline()
        assert "ClassicalCVPipeline" in repr(pipe)

    def test_from_config(self):
        pipe = ClassicalCVPipeline.from_config({"kernel_size": 3})
        assert pipe is not None

    def test_run_returns_dict(self, bgr_image, tmp_path):
        """Pipeline.run() on an in-memory image should return a result dict."""
        pipe = ClassicalCVPipeline()
        result = pipe.run_on_array(bgr_image)
        assert isinstance(result, dict)

    def test_components_accessible(self):
        pipe = ClassicalCVPipeline()
        assert hasattr(pipe, "loader")
        assert hasattr(pipe, "filter_proc")
        assert hasattr(pipe, "edge_detector")
