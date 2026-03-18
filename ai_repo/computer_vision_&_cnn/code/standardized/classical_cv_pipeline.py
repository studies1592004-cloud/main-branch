from __future__ import annotations
"""
classical_cv_pipeline.py
========================
Industry-Standard Classical Computer Vision Pipeline

Installation
------------
    pip install numpy opencv-python matplotlib scikit-learn requests

Usage
-----
    python classical_cv_pipeline.py                     # auto-downloads sample image
    python classical_cv_pipeline.py --image path/to/img.jpg
    python classical_cv_pipeline.py --image img.jpg --template tmpl.jpg

Author: CV Course — Section 1
Python: 3.9+
"""

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================
IMAGE_SIZE: int = 512          # resize longest edge to this before processing
GAUSSIAN_KERNEL_SIZE: int = 5  # must be odd
GAUSSIAN_SIGMA: float = 1.4
MEDIAN_KERNEL_SIZE: int = 5    # must be odd
CANNY_LOW_THRESH: float = 0.05  # as fraction of max gradient
CANNY_HIGH_THRESH: float = 0.15
HOG_CELL_SIZE: int = 8         # pixels per HOG cell
HOG_BLOCK_SIZE: int = 2        # cells per HOG block
HOG_N_BINS: int = 9            # orientation bins (0–180°)
LK_WIN_SIZE: tuple = (21, 21)  # Lucas-Kanade window
LK_MAX_LEVEL: int = 3          # pyramid levels
MORPH_KERNEL_SIZE: int = 5
OUTPUT_DIR: str = "./cv_output"

# ============================================================
# IMPORTS
# ============================================================

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ClassicalCV")


# ============================================================
# UTILITY: ensure output directory
# ============================================================
def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS: ImageLoader
# ============================================================
class ImageLoader:
    """Load, validate, and optionally resize an image from disk or URL.

    Args:
        max_size: Resize longest edge to this value (preserving aspect ratio).
            Pass ``None`` to disable resizing.

    Raises:
        FileNotFoundError: If the requested local path does not exist.
        RuntimeError: If OpenCV cannot decode the image data.
    """

    SAMPLE_URL: str = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "3/3f/Bikesgray.jpg/640px-Bikesgray.jpg"
    )
    SAMPLE_PATH: str = os.path.join(OUTPUT_DIR, "sample.jpg")

    def __init__(self, max_size: Optional[int] = IMAGE_SIZE) -> None:
        self.max_size = max_size
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"ImageLoader(max_size={self.max_size})"

    @classmethod
    def from_config(cls, config: dict) -> "ImageLoader":
        """Construct from a config dictionary.

        Args:
            config: Must contain optional key ``max_size`` (int).

        Returns:
            Configured ``ImageLoader`` instance.
        """
        return cls(max_size=config.get("max_size", IMAGE_SIZE))

    def load(self, path: Optional[str] = None) -> np.ndarray:
        """Load image as BGR uint8 NumPy array.

        Args:
            path: Filesystem path.  If ``None``, downloads a sample image.

        Returns:
            BGR image array, shape ``(H, W, 3)``.
        """
        if path is None:
            path = self._get_sample()

        path = str(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(
                f"OpenCV could not decode '{path}'. "
                "Check the file is a valid image format."
            )

        self._logger.info(
            "Loaded '%s'  shape=%s  dtype=%s", path, img.shape, img.dtype
        )

        if self.max_size is not None:
            img = self._resize(img)
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize so that the longest edge equals ``self.max_size``."""
        h, w = img.shape[:2]
        scale = self.max_size / max(h, w)
        if scale >= 1.0:
            return img
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self._logger.info("Resized to %s", resized.shape)
        return resized

    def _get_sample(self) -> str:
        """Download sample image if it is not already cached."""
        _ensure_dir(OUTPUT_DIR)
        if not os.path.isfile(self.SAMPLE_PATH):
            self._logger.info(
                "No image supplied — downloading sample from Wikipedia…"
            )
            try:
                urllib.request.urlretrieve(self.SAMPLE_URL, self.SAMPLE_PATH)
                self._logger.info("Sample saved to %s", self.SAMPLE_PATH)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download sample image: {exc}\n"
                    "Please supply an image with --image path/to/img.jpg"
                ) from exc
        else:
            self._logger.info("Using cached sample: %s", self.SAMPLE_PATH)
        return self.SAMPLE_PATH


# ============================================================
# CLASS: FilteringProcessor
# ============================================================
class FilteringProcessor:
    """Apply Gaussian and median filters from scratch (no cv2.GaussianBlur).

    Uses only ``numpy`` for convolution so the implementation is transparent.
    OpenCV is used only for ``medianBlur`` (median has no clean closed-form
    separable kernel — the sliding-histogram approach is non-trivial).

    Args:
        gaussian_ksize: Kernel size (odd integer).
        gaussian_sigma: Standard deviation for Gaussian kernel.
        median_ksize:   Kernel size for median filter (odd integer).
    """

    def __init__(
        self,
        gaussian_ksize: int = GAUSSIAN_KERNEL_SIZE,
        gaussian_sigma: float = GAUSSIAN_SIGMA,
        median_ksize: int = MEDIAN_KERNEL_SIZE,
    ) -> None:
        if gaussian_ksize % 2 == 0:
            raise ValueError("gaussian_ksize must be odd.")
        if median_ksize % 2 == 0:
            raise ValueError("median_ksize must be odd.")
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigma = gaussian_sigma
        self.median_ksize = median_ksize
        self._logger = logging.getLogger(self.__class__.__name__)
        self._kernel = self._build_gaussian_kernel()

    def __repr__(self) -> str:
        return (
            f"FilteringProcessor(g_ksize={self.gaussian_ksize}, "
            f"g_sigma={self.gaussian_sigma}, m_ksize={self.median_ksize})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "FilteringProcessor":
        """Construct from config dict."""
        return cls(
            gaussian_ksize=config.get("gaussian_ksize", GAUSSIAN_KERNEL_SIZE),
            gaussian_sigma=config.get("gaussian_sigma", GAUSSIAN_SIGMA),
            median_ksize=config.get("median_ksize", MEDIAN_KERNEL_SIZE),
        )

    def _build_gaussian_kernel(self) -> np.ndarray:
        """Build a 2-D Gaussian kernel.

        The 1-D kernel g(x) = exp(-x²/(2σ²)), then outer-product with itself.

        Returns:
            Normalised 2-D kernel, shape ``(k, k)``.
        """
        k = self.gaussian_ksize
        half = k // 2
        xs = np.arange(-half, half + 1, dtype=np.float64)
        g1d = np.exp(-(xs ** 2) / (2 * self.gaussian_sigma ** 2))
        kernel = np.outer(g1d, g1d)
        kernel /= kernel.sum()
        self._logger.debug("Gaussian kernel shape=%s  sum=%.6f", kernel.shape, kernel.sum())
        return kernel

    def _convolve2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2-D convolution with zero-padding (manual, channel-aware).

        Args:
            img:    Input image, shape ``(H, W)`` or ``(H, W, C)``.
            kernel: 2-D kernel, shape ``(kH, kW)``.

        Returns:
            Convolved array, same shape as ``img``, dtype float32.
        """
        kH, kW = kernel.shape
        pH, pW = kH // 2, kW // 2
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            squeeze = True
        else:
            squeeze = False

        H, W, C = img.shape
        padded = np.pad(
            img.astype(np.float32),
            ((pH, pH), (pW, pW), (0, 0)),
            mode="reflect",
        )
        out = np.zeros((H, W, C), dtype=np.float32)
        for c in range(C):
            for i in range(kH):
                for j in range(kW):
                    out[:, :, c] += (
                        kernel[i, j] * padded[i: i + H, j: j + W, c]
                    )
        if squeeze:
            out = out[:, :, 0]
        return out

    def gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur (from-scratch separable convolution).

        Args:
            img: BGR or grayscale uint8 image.

        Returns:
            Blurred image, dtype float32, same spatial size.
        """
        self._logger.info(
            "Gaussian blur  ksize=%d  sigma=%.2f", self.gaussian_ksize, self.gaussian_sigma
        )
        # Separable: apply 1-D kernel along rows then columns
        k = self.gaussian_ksize
        half = k // 2
        xs = np.arange(-half, half + 1, dtype=np.float64)
        g1d = np.exp(-(xs ** 2) / (2 * self.gaussian_sigma ** 2))
        g1d /= g1d.sum()

        row_kernel = g1d[np.newaxis, :]   # (1, k)
        col_kernel = g1d[:, np.newaxis]   # (k, 1)

        blurred = self._convolve2d(img, row_kernel)
        blurred = self._convolve2d(blurred, col_kernel)
        return blurred

    def median_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply median filter (OpenCV implementation, explained here).

        Median filtering replaces each pixel with the median of its
        neighbourhood.  Unlike mean/Gaussian, it removes salt-and-pepper
        noise while preserving edges because the median is a rank filter
        (not a linear convolution).

        Args:
            img: BGR or grayscale uint8 image.

        Returns:
            Median-filtered image, dtype uint8.
        """
        self._logger.info("Median blur  ksize=%d", self.median_ksize)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.medianBlur(img, self.median_ksize)


# ============================================================
# CLASS: EdgeDetector
# ============================================================
class EdgeDetector:
    """Sobel gradient maps and Canny edge detection — both from scratch.

    Args:
        canny_low:  Low threshold as fraction of max gradient magnitude.
        canny_high: High threshold as fraction of max gradient magnitude.
    """

    # 3×3 Sobel kernels
    _KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    _KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    def __init__(
        self,
        canny_low: float = CANNY_LOW_THRESH,
        canny_high: float = CANNY_HIGH_THRESH,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"EdgeDetector(canny_low={self.canny_low}, canny_high={self.canny_high})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "EdgeDetector":
        """Construct from config dict."""
        return cls(
            canny_low=config.get("canny_low", CANNY_LOW_THRESH),
            canny_high=config.get("canny_high", CANNY_HIGH_THRESH),
        )

    def _to_gray_float(self, img: np.ndarray) -> np.ndarray:
        """Convert to float32 grayscale in [0, 1]."""
        if img.ndim == 3:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        return gray.astype(np.float32) / 255.0

    def _convolve(self, gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Fast 2-D convolution via ``cv2.filter2D``."""
        return cv2.filter2D(gray, cv2.CV_32F, kernel)

    def sobel(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Sobel gradient magnitude and direction.

        Steps:
            1. Convert to float32 grayscale.
            2. Convolve with 3×3 Sobel-X and Sobel-Y kernels.
            3. Magnitude = sqrt(Gx² + Gy²).
            4. Direction = arctan2(Gy, Gx), quantised to 0/45/90/135°.

        Args:
            img: BGR or grayscale uint8 image.

        Returns:
            Tuple of (magnitude, direction_deg, gray) all float32.
        """
        self._logger.info("Sobel gradient computation")
        gray = self._to_gray_float(img)
        gx = self._convolve(gray, self._KX)
        gy = self._convolve(gray, self._KY)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        direction = np.rad2deg(np.arctan2(gy, gx)) % 180  # [0, 180)
        return magnitude, direction, gray

    def canny(self, img: np.ndarray) -> np.ndarray:
        """Canny edge detector — all 5 steps from scratch.

        Steps:
            1. Gaussian smoothing (5×5, σ=1.4).
            2. Sobel gradients.
            3. Non-maximum suppression (NMS) along gradient direction.
            4. Double-threshold: strong / weak / suppressed pixels.
            5. Hysteresis: keep weak pixels only if connected to strong.

        Args:
            img: BGR or grayscale uint8 image.

        Returns:
            Binary edge map, dtype uint8 (0 or 255).
        """
        self._logger.info(
            "Canny edge detection  low=%.2f  high=%.2f",
            self.canny_low,
            self.canny_high,
        )

        # 1. Gaussian smoothing
        fp = FilteringProcessor(gaussian_ksize=5, gaussian_sigma=1.4)
        gray = self._to_gray_float(img)
        smoothed = fp.gaussian_blur((gray * 255).astype(np.uint8))

        # 2. Sobel gradients
        gx = self._convolve(smoothed, self._KX)
        gy = self._convolve(smoothed, self._KY)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        direction = np.rad2deg(np.arctan2(gy, gx)) % 180

        # 3. Non-maximum suppression
        nms = self._non_max_suppression(magnitude, direction)

        # 4. Double threshold
        high = self.canny_high * nms.max()
        low = self.canny_low * nms.max()
        strong = (nms >= high).astype(np.uint8)
        weak = ((nms >= low) & (nms < high)).astype(np.uint8)

        # 5. Hysteresis tracking
        edges = self._hysteresis(strong, weak)
        return (edges * 255).astype(np.uint8)

    def _non_max_suppression(
        self, magnitude: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Thin edges: keep pixel only if it is a local maximum along the gradient.

        Gradient direction is quantised to 4 bins (0, 45, 90, 135°).
        Each pixel is compared to its two neighbours along that direction.
        If it is not the maximum, it is suppressed (set to 0).

        Args:
            magnitude: Gradient magnitude, shape (H, W).
            direction: Gradient direction in [0, 180), shape (H, W).

        Returns:
            Thinned magnitude, shape (H, W).
        """
        H, W = magnitude.shape
        out = np.zeros_like(magnitude)

        # Quantise direction to 0 / 45 / 90 / 135
        angle = direction.copy()
        angle[(angle >= 157.5) | (angle < 22.5)] = 0
        angle[(angle >= 22.5) & (angle < 67.5)] = 45
        angle[(angle >= 67.5) & (angle < 112.5)] = 90
        angle[(angle >= 112.5) & (angle < 157.5)] = 135

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                a = angle[i, j]
                m = magnitude[i, j]
                if a == 0:
                    n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
                elif a == 45:
                    n1, n2 = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
                elif a == 90:
                    n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
                else:  # 135
                    n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
                if m >= n1 and m >= n2:
                    out[i, j] = m
        return out

    def _hysteresis(
        self, strong: np.ndarray, weak: np.ndarray
    ) -> np.ndarray:
        """Edge tracking by hysteresis using BFS flood-fill.

        A weak pixel becomes a confirmed edge if it is 8-connected to a
        strong pixel.

        Args:
            strong: Binary map of strong edge pixels.
            weak:   Binary map of weak edge pixels.

        Returns:
            Binary edge map (0/1), same shape.
        """
        from collections import deque

        H, W = strong.shape
        edges = strong.copy()
        queue = deque(zip(*np.where(strong == 1)))
        visited = strong.astype(bool)

        while queue:
            r, c = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and not visited[nr, nc]
                        and weak[nr, nc] == 1
                    ):
                        edges[nr, nc] = 1
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        return edges


# ============================================================
# CLASS: HistogramEqualizer
# ============================================================
class HistogramEqualizer:
    """Global histogram equalisation from scratch using CDF mapping.

    For a grayscale image:
        1. Compute normalised histogram  h[v] = count(v) / N_pixels.
        2. Compute CDF:                  cdf[v] = sum_{u=0}^{v} h[u].
        3. Map:                          out[i,j] = round(255 * cdf[img[i,j]]).

    For colour images, equalise only the luminance channel in YCrCb space
    (otherwise colours shift).

    Args:
        colour_mode: ``"luminance"`` equalises Y channel only (recommended);
            ``"channel"`` equalises each BGR channel independently.
    """

    def __init__(self, colour_mode: str = "luminance") -> None:
        if colour_mode not in ("luminance", "channel"):
            raise ValueError("colour_mode must be 'luminance' or 'channel'.")
        self.colour_mode = colour_mode
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"HistogramEqualizer(colour_mode='{self.colour_mode}')"

    @classmethod
    def from_config(cls, config: dict) -> "HistogramEqualizer":
        """Construct from config dict."""
        return cls(colour_mode=config.get("colour_mode", "luminance"))

    def _equalise_gray(self, gray: np.ndarray) -> np.ndarray:
        """Equalise a single uint8 grayscale channel.

        Args:
            gray: 2-D uint8 array.

        Returns:
            Equalised uint8 array, same shape.
        """
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        cdf = hist.cumsum()
        # Normalise CDF to [0, 255]; skip leading zeros (masked CDF)
        cdf_min = cdf[cdf > 0][0]
        n_pixels = gray.size
        lut = np.round(
            (cdf - cdf_min) / (n_pixels - cdf_min) * 255
        ).astype(np.uint8)
        return lut[gray]

    def equalise(self, img: np.ndarray) -> np.ndarray:
        """Apply histogram equalisation.

        Args:
            img: BGR uint8 image (H, W, 3) or grayscale (H, W).

        Returns:
            Equalised uint8 image, same shape and channels.
        """
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            self._logger.info("HE: grayscale image")
            return self._equalise_gray(img)

        self._logger.info("HE: colour image, mode='%s'", self.colour_mode)

        if self.colour_mode == "luminance":
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = self._equalise_gray(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            out = img.copy()
            for c in range(3):
                out[:, :, c] = self._equalise_gray(img[:, :, c])
            return out


# ============================================================
# CLASS: MorphologyProcessor
# ============================================================
class MorphologyProcessor:
    """Morphological operations from scratch using NumPy sliding windows.

    Supported operations: erosion, dilation, opening, closing.

    The structuring element (SE) is a flat disk/square of ones.  For each
    pixel, erosion keeps the minimum value under the SE; dilation keeps
    the maximum.

    Args:
        kernel_size: Side length of the square structuring element.
        shape:       ``"rect"`` or ``"ellipse"``.
    """

    def __init__(
        self,
        kernel_size: int = MORPH_KERNEL_SIZE,
        shape: str = "rect",
    ) -> None:
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.shape = shape
        self._se = self._build_se()
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"MorphologyProcessor(kernel_size={self.kernel_size}, shape='{self.shape}')"

    @classmethod
    def from_config(cls, config: dict) -> "MorphologyProcessor":
        """Construct from config dict."""
        return cls(
            kernel_size=config.get("kernel_size", MORPH_KERNEL_SIZE),
            shape=config.get("shape", "rect"),
        )

    def _build_se(self) -> np.ndarray:
        """Build the structuring element.

        Returns:
            Boolean array shape ``(k, k)``.
        """
        k = self.kernel_size
        if self.shape == "rect":
            return np.ones((k, k), dtype=bool)
        # Ellipse / disk
        half = k // 2
        yy, xx = np.mgrid[-half: half + 1, -half: half + 1]
        return (xx ** 2 / (half + 0.5) ** 2 + yy ** 2 / (half + 0.5) ** 2) <= 1

    def _apply(self, img: np.ndarray, mode: str) -> np.ndarray:
        """Core sliding-window morphological operation.

        Args:
            img:  Grayscale float32 array ``(H, W)``.
            mode: ``"erode"`` or ``"dilate"``.

        Returns:
            Processed array, same shape, float32.
        """
        k = self.kernel_size
        half = k // 2
        H, W = img.shape
        padded = np.pad(img, half, mode="edge")
        out = np.empty_like(img)

        agg = np.min if mode == "erode" else np.max
        for i in range(H):
            for j in range(W):
                patch = padded[i: i + k, j: j + k]
                out[i, j] = agg(patch[self._se])
        return out

    def erode(self, img: np.ndarray) -> np.ndarray:
        """Binary/grayscale erosion.

        Shrinks bright regions: each output pixel = min of neighbourhood
        under the structuring element.

        Args:
            img: Grayscale uint8 image.

        Returns:
            Eroded image, uint8.
        """
        self._logger.info("Erosion  kernel=%d", self.kernel_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        result = self._apply(gray.astype(np.float32), "erode")
        return np.clip(result, 0, 255).astype(np.uint8)

    def dilate(self, img: np.ndarray) -> np.ndarray:
        """Grayscale dilation.

        Expands bright regions: each output pixel = max of neighbourhood.

        Args:
            img: Grayscale uint8 image.

        Returns:
            Dilated image, uint8.
        """
        self._logger.info("Dilation  kernel=%d", self.kernel_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        result = self._apply(gray.astype(np.float32), "dilate")
        return np.clip(result, 0, 255).astype(np.uint8)

    def opening(self, img: np.ndarray) -> np.ndarray:
        """Morphological opening = erosion then dilation.

        Removes small bright objects while preserving larger ones.

        Args:
            img: Grayscale uint8 image.

        Returns:
            Opened image, uint8.
        """
        self._logger.info("Opening")
        return self.dilate(self.erode(img).reshape(img.shape[:2]))

    def closing(self, img: np.ndarray) -> np.ndarray:
        """Morphological closing = dilation then erosion.

        Fills small dark holes while preserving structure.

        Args:
            img: Grayscale uint8 image.

        Returns:
            Closed image, uint8.
        """
        self._logger.info("Closing")
        return self.erode(self.dilate(img).reshape(img.shape[:2]))


# ============================================================
# CLASS: TemplateMatcher
# ============================================================
class TemplateMatcher:
    """Normalised Cross-Correlation (NCC) template matching from scratch.

    NCC(u, v) = (Σ (f(x,y)-f̄)(t(x-u,y-v)-t̄)) /
                sqrt(Σ(f(x,y)-f̄)² · Σ(t(x-u,y-v)-t̄)²)

    Range: [-1, 1].  1 = perfect match.  Invariant to additive and
    multiplicative intensity changes (brightness / contrast changes).

    Args:
        threshold: Minimum NCC score to consider a match.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"TemplateMatcher(threshold={self.threshold})"

    @classmethod
    def from_config(cls, config: dict) -> "TemplateMatcher":
        """Construct from config dict."""
        return cls(threshold=config.get("threshold", 0.7))

    def match(
        self, image: np.ndarray, template: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int], float]:
        """Slide template over image and compute NCC at every position.

        Args:
            image:    Grayscale or BGR uint8 image to search within.
            template: Grayscale or BGR uint8 template patch.

        Returns:
            Tuple of:
              - ncc_map:  Float32 array ``(H-tH+1, W-tW+1)`` of NCC scores.
              - best_loc: ``(col, row)`` of best match top-left corner.
              - best_score: NCC score at best location.

        Raises:
            ValueError: If template is larger than the image.
        """
        img_g = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image.copy()
        )
        tmpl_g = (
            cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            if template.ndim == 3
            else template.copy()
        )

        img_g = img_g.astype(np.float32)
        tmpl_g = tmpl_g.astype(np.float32)
        tH, tW = tmpl_g.shape
        iH, iW = img_g.shape

        if tH > iH or tW > iW:
            raise ValueError(
                f"Template ({tH}×{tW}) is larger than image ({iH}×{iW})."
            )

        self._logger.info(
            "NCC template matching  image=%s  template=%s", img_g.shape, tmpl_g.shape
        )

        # Zero-mean template
        t_mean = tmpl_g.mean()
        t_norm = tmpl_g - t_mean
        t_denom = np.sqrt((t_norm ** 2).sum()) + 1e-8

        ncc_map = np.zeros((iH - tH + 1, iW - tW + 1), dtype=np.float32)

        for r in range(iH - tH + 1):
            for c in range(iW - tW + 1):
                patch = img_g[r: r + tH, c: c + tW]
                p_mean = patch.mean()
                p_norm = patch - p_mean
                p_denom = np.sqrt((p_norm ** 2).sum()) + 1e-8
                ncc_map[r, c] = (p_norm * t_norm).sum() / (p_denom * t_denom)

        best_r, best_c = np.unravel_index(ncc_map.argmax(), ncc_map.shape)
        best_score = float(ncc_map[best_r, best_c])
        self._logger.info("Best NCC score=%.4f at (col=%d, row=%d)", best_score, best_c, best_r)
        return ncc_map, (best_c, best_r), best_score


# ============================================================
# CLASS: HOGExtractor
# ============================================================
class HOGExtractor:
    """Histogram of Oriented Gradients (HOG) from scratch.

    Algorithm (Dalal & Triggs, 2005):
        1. Normalise image gamma (optional).
        2. Compute pixel gradients (Sobel).
        3. Divide into cells of size ``cell_size × cell_size`` pixels.
        4. For each cell, compute a weighted histogram of gradient orientations
           (``n_bins`` bins covering 0–180°, unsigned gradients).
        5. Group cells into overlapping blocks (``block_size × block_size`` cells).
        6. L2-Hys normalise each block (clip to 0.2, re-normalise).
        7. Concatenate all block descriptors.

    The resulting descriptor is compatible with ``sklearn.svm.LinearSVC``.

    Args:
        cell_size:  Pixels per cell (square).
        block_size: Cells per block (square).
        n_bins:     Orientation bins (0–180°).
    """

    def __init__(
        self,
        cell_size: int = HOG_CELL_SIZE,
        block_size: int = HOG_BLOCK_SIZE,
        n_bins: int = HOG_N_BINS,
    ) -> None:
        self.cell_size = cell_size
        self.block_size = block_size
        self.n_bins = n_bins
        self._bin_edges = np.linspace(0, 180, n_bins + 1)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"HOGExtractor(cell_size={self.cell_size}, "
            f"block_size={self.block_size}, n_bins={self.n_bins})"
        )

    @classmethod
    def from_config(cls, config: dict) -> "HOGExtractor":
        """Construct from config dict."""
        return cls(
            cell_size=config.get("cell_size", HOG_CELL_SIZE),
            block_size=config.get("block_size", HOG_BLOCK_SIZE),
            n_bins=config.get("n_bins", HOG_N_BINS),
        )

    def _cell_histogram(self, mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
        """Build a weighted orientation histogram for one cell.

        Each gradient vote is distributed linearly between the two nearest
        bins (soft binning) weighted by its magnitude.

        Args:
            mag: Magnitude array ``(cell_size, cell_size)``.
            ang: Angle array ``(cell_size, cell_size)`` in [0, 180).

        Returns:
            1-D histogram, length ``n_bins``.
        """
        hist = np.zeros(self.n_bins, dtype=np.float32)
        bin_width = 180.0 / self.n_bins
        for r in range(mag.shape[0]):
            for c in range(mag.shape[1]):
                a = ang[r, c]
                m = mag[r, c]
                lo_bin = int(a / bin_width) % self.n_bins
                hi_bin = (lo_bin + 1) % self.n_bins
                lo_centre = (lo_bin + 0.5) * bin_width
                hi_weight = abs(a - lo_centre) / bin_width
                lo_weight = 1.0 - hi_weight
                hist[lo_bin] += m * lo_weight
                hist[hi_bin] += m * hi_weight
        return hist

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Extract HOG descriptor from an image.

        Args:
            img: BGR or grayscale uint8 image.

        Returns:
            1-D float32 descriptor. Length =
            ``n_blocks_y × n_blocks_x × block_size² × n_bins``.
        """
        # Convert to grayscale
        gray = (
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            if img.ndim == 3
            else img.copy().astype(np.uint8)
        )

        H, W = gray.shape
        cs = self.cell_size
        # Crop to multiple of cell_size
        H_crop = (H // cs) * cs
        W_crop = (W // cs) * cs
        gray = gray[:H_crop, :W_crop].astype(np.float32)

        # Gradients
        gx = cv2.filter2D(gray, cv2.CV_32F, np.array([[-1, 0, 1]], dtype=np.float32))
        gy = cv2.filter2D(gray, cv2.CV_32F, np.array([[-1], [0], [1]], dtype=np.float32))
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.rad2deg(np.arctan2(np.abs(gy), np.abs(gx)))  # unsigned [0,90] → remap

        # Full unsigned range [0, 180)
        angle = np.rad2deg(np.arctan2(gy, gx)) % 180

        # Build cell histograms
        n_cells_y = H_crop // cs
        n_cells_x = W_crop // cs
        cell_hists = np.zeros((n_cells_y, n_cells_x, self.n_bins), dtype=np.float32)

        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                m_cell = magnitude[cy * cs: (cy + 1) * cs, cx * cs: (cx + 1) * cs]
                a_cell = angle[cy * cs: (cy + 1) * cs, cx * cs: (cx + 1) * cs]
                cell_hists[cy, cx] = self._cell_histogram(m_cell, a_cell)

        # Block normalisation (L2-Hys)
        bs = self.block_size
        descriptor_parts = []
        for by in range(n_cells_y - bs + 1):
            for bx in range(n_cells_x - bs + 1):
                block = cell_hists[by: by + bs, bx: bx + bs].ravel()
                # L2-normalise
                norm = np.sqrt((block ** 2).sum() + 1e-8)
                block = block / norm
                # Clip (Hys)
                block = np.clip(block, 0, 0.2)
                # Re-normalise
                norm2 = np.sqrt((block ** 2).sum() + 1e-8)
                block = block / norm2
                descriptor_parts.append(block)

        descriptor = np.concatenate(descriptor_parts)
        self._logger.info(
            "HOG descriptor length=%d  cells=(%d,%d)", len(descriptor), n_cells_y, n_cells_x
        )
        return descriptor


# ============================================================
# CLASS: OpticalFlowEstimator
# ============================================================
class OpticalFlowEstimator:
    """Pyramidal Lucas-Kanade sparse optical flow (OpenCV, step-by-step).

    Lucas-Kanade (LK) assumes:
        1. Brightness constancy:  I(x,y,t) ≈ I(x+u,y+v,t+1).
        2. Small motion:          flow is approximately constant in a window W.
        3. Spatial coherence.

    For a window W centred at pixel p:
        Sum_{q∈W} [Ix(q)u + Iy(q)v + It(q)]² → minimised → 2×2 linear system.
        [Σ Ix²   Σ IxIy] [u]   [-Σ IxIt]
        [Σ IxIy  Σ Iy²] [v] = [-Σ IyIt]

    Pyramidal extension:
        Build image pyramid (Gaussian downscale by 2 per level).
        Estimate flow at coarsest level, propagate estimate to finer levels.
        This handles large motions where small-motion assumption fails.

    Args:
        win_size:   Lucas-Kanade integration window size.
        max_level:  Number of pyramid levels (0 = no pyramid).
        max_corners: Max number of Shi-Tomasi corners to track.
    """

    def __init__(
        self,
        win_size: tuple = LK_WIN_SIZE,
        max_level: int = LK_MAX_LEVEL,
        max_corners: int = 200,
    ) -> None:
        self.win_size = win_size
        self.max_level = max_level
        self.max_corners = max_corners
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"OpticalFlowEstimator(win_size={self.win_size}, "
            f"max_level={self.max_level})"
        )

    def track(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect corners in frame1 and track them to frame2.

        Args:
            frame1: First BGR frame.
            frame2: Second BGR frame (next in time).

        Returns:
            Tuple of:
              - pts1: detected points in frame1 ``(N, 1, 2)`` float32.
              - pts2: tracked points in frame2 ``(N, 1, 2)`` float32.
              - status: ``(N, 1)`` uint8; 1 = successfully tracked.
        """
        g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Shi-Tomasi corner detection (goodFeaturesToTrack)
        pts1 = cv2.goodFeaturesToTrack(
            g1,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=10,
        )

        if pts1 is None or len(pts1) == 0:
            self._logger.warning("No corners detected in frame1.")
            empty = np.empty((0, 1, 2), dtype=np.float32)
            return empty, empty, np.empty((0, 1), dtype=np.uint8)

        self._logger.info(
            "Detected %d corners in frame1. Running pyramidal LK (levels=%d, win=%s).",
            len(pts1),
            self.max_level,
            self.win_size,
        )

        lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        pts2, status, _err = cv2.calcOpticalFlowPyrLK(g1, g2, pts1, None, **lk_params)

        n_tracked = int(status.sum())
        self._logger.info("Tracked %d / %d points.", n_tracked, len(pts1))
        return pts1, pts2, status

    def draw_flow(
        self,
        frame: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        status: np.ndarray,
    ) -> np.ndarray:
        """Draw flow vectors (arrows) on a copy of ``frame``.

        Args:
            frame:  BGR image.
            pts1:   Source points ``(N, 1, 2)``.
            pts2:   Destination points ``(N, 1, 2)``.
            status: Track status ``(N, 1)``.

        Returns:
            Annotated BGR image.
        """
        vis = frame.copy()
        good1 = pts1[status.ravel() == 1].reshape(-1, 2)
        good2 = pts2[status.ravel() == 1].reshape(-1, 2)
        for (x1, y1), (x2, y2) in zip(good1.astype(int), good2.astype(int)):
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(vis, (x1, y1), 3, (0, 0, 255), -1)
        return vis


# ============================================================
# CLASS: Visualizer
# ============================================================
class Visualizer:
    """Compose and save a grid of result images.

    Args:
        output_dir: Directory where output images are saved.
        dpi:        Figure DPI.
    """

    def __init__(self, output_dir: str = OUTPUT_DIR, dpi: int = 120) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        _ensure_dir(output_dir)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"Visualizer(output_dir='{self.output_dir}', dpi={self.dpi})"

    def save_grid(
        self,
        panels: list[tuple[np.ndarray, str]],
        filename: str,
        cols: int = 4,
    ) -> str:
        """Arrange panels in a grid and save to disk.

        Args:
            panels:   List of (image_array, title) tuples.
                      Images may be BGR (3-ch), grayscale (2-ch), or float.
            filename: Output filename (relative to ``output_dir``).
            cols:     Number of columns in the grid.

        Returns:
            Full path to the saved file.
        """
        n = len(panels)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = np.array(axes).ravel()

        for ax, (img, title) in zip(axes, panels):
            if img.ndim == 3:
                # Convert BGR → RGB for matplotlib
                disp = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                ax.imshow(disp)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255 if img.max() > 1 else 1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        # Hide unused axes
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout(pad=1.0)
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self._logger.info("Saved grid → %s", out_path)
        return out_path


# ============================================================
# CLASS: ClassicalCVPipeline  (orchestrator)
# ============================================================
class ClassicalCVPipeline:
    """Top-level orchestrator that runs all classical CV operations.

    Instantiates each processing class, runs them in order, and
    produces a summary visualisation grid.

    Args:
        image_path:    Path to input image.  ``None`` downloads a sample.
        template_path: Path to template for matching.  ``None`` crops the
                       centre-quarter of the input image automatically.
        output_dir:    Directory for output files.
        config:        Optional dict to override global constants.
    """

    def __init__(
        self,
        image_path: Optional[str] = None,
        template_path: Optional[str] = None,
        output_dir: str = OUTPUT_DIR,
        config: Optional[dict] = None,
    ) -> None:
        self.image_path = image_path
        self.template_path = template_path
        self.output_dir = output_dir
        cfg = config or {}
        self._logger = logging.getLogger(self.__class__.__name__)

        self.loader = ImageLoader.from_config(cfg)
        self.filter = FilteringProcessor.from_config(cfg)
        self.edge = EdgeDetector.from_config(cfg)
        self.he = HistogramEqualizer.from_config(cfg)
        self.morph = MorphologyProcessor.from_config(cfg)
        self.matcher = TemplateMatcher.from_config(cfg)
        self.hog = HOGExtractor.from_config(cfg)
        self.flow = OpticalFlowEstimator()
        self.viz = Visualizer(output_dir=output_dir)

    def __repr__(self) -> str:
        return (
            f"ClassicalCVPipeline(image='{self.image_path}', "
            f"output='{self.output_dir}')"
        )

    @classmethod
    def from_config(cls, config: dict) -> "ClassicalCVPipeline":
        """Construct from a flat config dictionary."""
        return cls(
            image_path=config.get("image_path"),
            template_path=config.get("template_path"),
            output_dir=config.get("output_dir", OUTPUT_DIR),
            config=config,
        )

    def _auto_template(self, img: np.ndarray) -> np.ndarray:
        """Crop the centre 25% of image as an auto-template."""
        H, W = img.shape[:2]
        r0, r1 = H // 4, 3 * H // 4
        c0, c1 = W // 4, 3 * W // 4
        return img[r0:r1, c0:c1]

    def run(self) -> dict[str, np.ndarray]:
        """Execute the full pipeline and save result grids.

        Returns:
            Dictionary mapping result names to NumPy arrays.
        """
        self._logger.info("=" * 55)
        self._logger.info("Classical CV Pipeline  START")
        self._logger.info("=" * 55)

        # ── 1. Load image ─────────────────────────────────────
        img = self.loader.load(self.image_path)
        results: dict[str, np.ndarray] = {"original": img}

        # ── 2. Filtering ──────────────────────────────────────
        gauss = self.filter.gaussian_blur(img)
        gauss_u8 = np.clip(gauss, 0, 255).astype(np.uint8)
        median = self.filter.median_blur(img)
        results.update({"gaussian_blur": gauss_u8, "median_blur": median})
        self._logger.info("Filtering complete.")

        # ── 3. Edge detection ─────────────────────────────────
        sobel_mag, _, _ = self.edge.sobel(img)
        sobel_vis = np.clip(sobel_mag / sobel_mag.max() * 255, 0, 255).astype(np.uint8)
        canny_edges = self.edge.canny(img)
        results.update({"sobel": sobel_vis, "canny": canny_edges})
        self._logger.info("Edge detection complete.")

        # ── 4. Histogram equalisation ─────────────────────────
        he_result = self.he.equalise(img)
        results["histogram_eq"] = he_result
        self._logger.info("Histogram equalisation complete.")

        # ── 5. Morphological operations ───────────────────────
        eroded = self.morph.erode(img)
        dilated = self.morph.dilate(img)
        opened = self.morph.opening(img)
        closed = self.morph.closing(img)
        results.update({
            "erosion": eroded,
            "dilation": dilated,
            "opening": opened,
            "closing": closed,
        })
        self._logger.info("Morphological operations complete.")

        # ── 6. Template matching ──────────────────────────────
        template = (
            self.loader.load(self.template_path)
            if self.template_path and os.path.isfile(self.template_path)
            else self._auto_template(img)
        )
        # Ensure template is smaller than image
        iH, iW = img.shape[:2]
        tH, tW = template.shape[:2]
        if tH >= iH or tW >= iW:
            template = cv2.resize(template, (iW // 3, iH // 3))
            self._logger.warning("Template larger than image — resized to 1/3.")

        ncc_map, best_loc, best_score = self.matcher.match(img, template)
        match_vis = img.copy()
        tH, tW = template.shape[:2]
        cx, cy = best_loc
        cv2.rectangle(match_vis, (cx, cy), (cx + tW, cy + tH), (0, 255, 0), 2)
        cv2.putText(
            match_vis,
            f"NCC={best_score:.3f}",
            (cx, max(0, cy - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        results["template_match"] = match_vis
        self._logger.info("Template matching complete. Best NCC=%.4f", best_score)

        # ── 7. HOG features ───────────────────────────────────
        descriptor = self.hog.extract(img)
        # Render a HOG visualisation by reshaping descriptor to cell grid
        hog_vis = self._render_hog(img, descriptor)
        results["hog"] = hog_vis
        self._logger.info(
            "HOG extraction complete. Descriptor length=%d", len(descriptor)
        )

        # ── 8. Optical flow (two synthetic frames) ────────────
        frame2 = np.roll(img, 8, axis=1)  # simulate rightward motion
        pts1, pts2, status = self.flow.track(img, frame2)
        if len(pts1) > 0:
            flow_vis = self.flow.draw_flow(img, pts1, pts2, status)
        else:
            flow_vis = img.copy()
        results["optical_flow"] = flow_vis
        self._logger.info("Optical flow complete.")

        # ── 9. Visualise ──────────────────────────────────────
        panels = [
            (img,               "Original"),
            (gauss_u8,          "Gaussian Blur"),
            (median,            "Median Blur"),
            (sobel_vis,         "Sobel Magnitude"),
            (canny_edges,       "Canny Edges"),
            (he_result,         "Histogram Eq."),
            (eroded,            "Erosion"),
            (dilated,           "Dilation"),
            (opened,            "Opening"),
            (closed,            "Closing"),
            (match_vis,         "Template Match (NCC)"),
            (hog_vis,           "HOG Features"),
            (flow_vis,          "Optical Flow (LK)"),
        ]
        self.viz.save_grid(panels, "classical_cv_results.png", cols=4)

        self._logger.info("=" * 55)
        self._logger.info("Classical CV Pipeline  DONE")
        self._logger.info("Results saved to '%s'", self.output_dir)
        self._logger.info("=" * 55)
        return results

    def _render_hog(self, img: np.ndarray, descriptor: np.ndarray) -> np.ndarray:
        """Render gradient orientation histogram lines on the image.

        Draws a compass-rose of HOG bin orientations at each cell centre,
        weighted by the bin magnitude.  Provides visual intuition for
        what HOG 'sees' in the image.

        Args:
            img:        Original BGR image.
            descriptor: 1-D HOG descriptor (used only for logging; we
                        re-extract cell histograms here for visualisation).

        Returns:
            BGR image with HOG overlaid.
        """
        vis = img.copy()
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        cs = self.hog.cell_size
        H, W = gray.shape
        n_cells_y = H // cs
        n_cells_x = W // cs

        # Quick gradient recomputation for visualisation only
        gx = cv2.filter2D(gray, cv2.CV_32F,
                          np.array([[-1, 0, 1]], dtype=np.float32))
        gy = cv2.filter2D(gray, cv2.CV_32F,
                          np.array([[-1], [0], [1]], dtype=np.float32))
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.rad2deg(np.arctan2(gy, gx)) % 180

        bin_width = 180.0 / self.hog.n_bins
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                m_cell = magnitude[cy * cs: (cy + 1) * cs, cx * cs: (cx + 1) * cs]
                a_cell = angle[cy * cs: (cy + 1) * cs, cx * cs: (cx + 1) * cs]
                hist = np.zeros(self.hog.n_bins, dtype=np.float32)
                for r in range(cs):
                    for c in range(cs):
                        b = int(a_cell[r, c] / bin_width) % self.hog.n_bins
                        hist[b] += m_cell[r, c]
                # Normalise
                hist /= (hist.max() + 1e-8)

                centre_x = cx * cs + cs // 2
                centre_y = cy * cs + cs // 2
                half_len = (cs // 2) - 1

                for b in range(self.hog.n_bins):
                    ang_rad = np.deg2rad(b * bin_width + bin_width / 2)
                    length = int(hist[b] * half_len)
                    dx = int(length * np.cos(ang_rad))
                    dy = int(length * np.sin(ang_rad))
                    cv2.line(
                        vis,
                        (centre_x - dx, centre_y - dy),
                        (centre_x + dx, centre_y + dy),
                        (0, 200, 255),
                        1,
                    )
        return vis


# ============================================================
# ENTRY POINT
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical CV Pipeline — Section 1"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (downloads sample if omitted).")
    parser.add_argument("--template", type=str, default=None,
                        help="Path to template patch for NCC matching.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory for output files.")
    return parser.parse_args()


def main() -> None:
    """Run the classical CV pipeline."""
    args = _parse_args()
    pipeline = ClassicalCVPipeline(
        image_path=args.image,
        template_path=args.template,
        output_dir=args.output_dir,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
