"""
cv_B_filters_convolution.py
============================
Computer Vision Course — Section B: Image Filters & Convolution

Topics covered:
  B1 - Convolution mechanics: manual 2D convolution, padding, stride, output size
  B2 - Linear filters: box filter, Gaussian filter, separability speedup
  B3 - Non-linear filters: median, bilateral, non-local means
  B4 - Edge detection: Sobel, Prewitt, LoG (Laplacian of Gaussian)
  B5 - Canny edge detector: 5-step pipeline from scratch
  B6 - Frequency domain: DFT, low-pass, high-pass, unsharp masking

Dependencies: numpy, opencv-python, matplotlib, scipy
Install:  pip install numpy opencv-python matplotlib scipy
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_test_image(h=256, w=256):
    img = np.zeros((h, w), dtype=np.float32)
    # horizontal gradient
    img += np.linspace(0, 0.5, w)[np.newaxis, :]
    # vertical gradient
    img += np.linspace(0, 0.3, h)[:, np.newaxis]
    # sharp edge rectangle
    img[60:120, 80:180] += 0.5
    # circle
    cx, cy, r = 180, 80, 35
    yy, xx = np.ogrid[:h, :w]
    img[(xx - cx)**2 + (yy - cy)**2 < r**2] = 0.9
    return np.clip(img, 0, 1)


def save_grid(imgs, titles, filename, cmap="gray"):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if img.max() <= 1 else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  Saved: {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# B1 — CONVOLUTION MECHANICS
# ═════════════════════════════════════════════════════════════════════════════

def section_B1():
    print("\n── B1: Convolution Mechanics ──")

    # --- Manual 2D convolution (no padding) ---
    def conv2d(img, kernel, padding="same"):
        """
        Manual 2D discrete convolution (correlation in practice — kernel NOT flipped).
        padding: 'same' (zero-pad to preserve size) or 'valid' (no padding)
        """
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2

        if padding == "same":
            padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
        else:
            padded = img
            # output will be smaller
            ph, pw = 0, 0

        H, W = img.shape
        out_h = H if padding == "same" else H - kh + 1
        out_w = W if padding == "same" else W - kw + 1
        out = np.zeros((out_h, out_w), dtype=np.float64)

        for i in range(out_h):
            for j in range(out_w):
                patch = padded[i:i+kh, j:j+kw]
                out[i, j] = np.sum(patch * kernel)
        return out

    # Output size formula: O = floor((I + 2P - K) / S) + 1
    def output_size(I, K, P=0, S=1):
        return (I + 2*P - K) // S + 1

    print("  Output size formula O = floor((I+2P-K)/S)+1:")
    for I, K, P, S in [(28,3,0,1), (28,3,1,1), (28,3,0,2), (224,3,1,1)]:
        print(f"    I={I} K={K} P={P} S={S} → O={output_size(I,K,P,S)}")

    # Test manual conv with a 3x3 edge kernel
    img = make_test_image(64, 64)         # small for speed
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)

    out_manual  = conv2d(img, sobel_x, padding="same")
    out_scipy   = convolve(img, sobel_x)   # reference
    diff = np.abs(out_manual - out_scipy).max()
    print(f"  Manual vs scipy conv max diff: {diff:.6f}")

    # --- Parameter counting ---
    def conv_params(C_in, C_out, K, bias=True):
        return C_out * (C_in * K * K + (1 if bias else 0))

    for C_in, C_out, K in [(3,64,3), (64,128,3), (512,512,3)]:
        p = conv_params(C_in, C_out, K)
        print(f"    Conv({C_in}->{C_out}, K={K}): {p:,} params")

    print("  Done: B1 convolution mechanics")


# ═════════════════════════════════════════════════════════════════════════════
# B2 — LINEAR FILTERS
# ═════════════════════════════════════════════════════════════════════════════

def section_B2():
    print("\n── B2: Linear Filters ──")

    img = make_test_image()

    # --- Box filter (averaging) ---
    box3  = cv2.boxFilter(img, -1, (3, 3))
    box9  = cv2.boxFilter(img, -1, (9, 9))
    box21 = cv2.boxFilter(img, -1, (21, 21))

    # --- Gaussian filter ---
    def gaussian_kernel_1d(sigma):
        """1D Gaussian kernel truncated at ±3σ."""
        radius = int(3 * sigma)
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-x**2 / (2 * sigma**2))
        return k / k.sum()

    def gaussian_kernel_2d(sigma):
        """2D Gaussian kernel via outer product of two 1D kernels."""
        k1d = gaussian_kernel_1d(sigma)
        return np.outer(k1d, k1d)

    def gaussian_filter_separable(img, sigma):
        """
        Separable Gaussian: apply 1D filter along rows then columns.
        Complexity: O(H*W*2K) vs O(H*W*K^2) for 2D — speedup = K/2
        """
        k1d = gaussian_kernel_1d(sigma)
        # Apply along columns (axis=0)
        tmp = np.apply_along_axis(lambda row: np.convolve(row, k1d, "same"), 1, img)
        # Apply along rows (axis=1)
        out = np.apply_along_axis(lambda col: np.convolve(col, k1d, "same"), 0, tmp)
        return out

    gauss_s1  = gaussian_filter_separable(img, sigma=1.0)
    gauss_s3  = gaussian_filter_separable(img, sigma=3.0)
    gauss_cv2 = cv2.GaussianBlur(img, (0, 0), sigmaX=3.0)

    diff = np.abs(gauss_s3 - gauss_cv2).max()
    print(f"  Custom Gaussian vs cv2.GaussianBlur (sigma=3) max diff: {diff:.4f}")

    # --- Separability speedup ---
    for sigma in [1, 3, 7]:
        k = int(6 * sigma + 1) | 1       # kernel size
        ops_2d  = k * k
        ops_sep = 2 * k
        print(f"    sigma={sigma}: K={k}  2D ops={ops_2d}  sep ops={ops_sep}  "
              f"speedup={ops_2d/ops_sep:.1f}x")

    save_grid(
        [img, box3, box9, gauss_s1, gauss_s3],
        ["Original", "Box 3x3", "Box 9x9", "Gauss σ=1", "Gauss σ=3"],
        "B2_linear_filters.png"
    )
    print("  Done: B2 linear filters")


# ═════════════════════════════════════════════════════════════════════════════
# B3 — NON-LINEAR FILTERS
# ═════════════════════════════════════════════════════════════════════════════

def section_B3():
    print("\n── B3: Non-linear Filters ──")

    clean = make_test_image()

    # Add salt-and-pepper noise
    noisy = clean.copy()
    rng = np.random.rand(*clean.shape)
    noisy[rng < 0.04] = 0.0
    noisy[rng > 0.96] = 1.0

    # --- Median filter (best for S&P, preserves edges) ---
    median3 = cv2.medianBlur((noisy * 255).astype(np.uint8), 3).astype(np.float32) / 255
    median7 = cv2.medianBlur((noisy * 255).astype(np.uint8), 7).astype(np.float32) / 255

    # --- Bilateral filter (edge-preserving, smooths flat regions) ---
    # Params: d=diameter, sigmaColor (intensity space), sigmaSpace (spatial)
    noisy_u8 = (noisy * 255).astype(np.uint8)
    bilateral_s = cv2.bilateralFilter(noisy_u8, d=9, sigmaColor=30,  sigmaSpace=30)
    bilateral_h = cv2.bilateralFilter(noisy_u8, d=9, sigmaColor=75,  sigmaSpace=75)
    bilateral_s = bilateral_s.astype(np.float32) / 255
    bilateral_h = bilateral_h.astype(np.float32) / 255

    # --- Manual bilateral filter (small patch, for understanding) ---
    def bilateral_manual(img, d=5, sigma_s=2.0, sigma_r=0.1):
        """Naive O(H*W*d^2) bilateral filter for understanding."""
        h, w = img.shape
        r = d // 2
        out = np.zeros_like(img)
        padded = np.pad(img, r, mode="reflect")

        for i in range(h):
            for j in range(w):
                patch = padded[i:i+d, j:j+d]
                # Spatial Gaussian weights
                yy, xx = np.mgrid[-r:r+1, -r:r+1]
                w_s = np.exp(-(xx**2 + yy**2) / (2 * sigma_s**2))
                # Range (intensity) Gaussian weights
                diff = patch - img[i, j]
                w_r = np.exp(-diff**2 / (2 * sigma_r**2))
                w_total = w_s * w_r
                out[i, j] = np.sum(patch * w_total) / np.sum(w_total)
        return out

    # Run manual bilateral on small patch for speed demo
    patch = noisy[:32, :32]
    bilat_manual = bilateral_manual(patch, d=5, sigma_s=2.0, sigma_r=0.1)
    print(f"  Manual bilateral patch PSNR vs clean: "
          f"{10*np.log10(1/np.mean((clean[:32,:32]-bilat_manual)**2)):.2f} dB")

    # --- Non-Local Means (cv2 NLM) ---
    nlm = cv2.fastNlMeansDenoising(
        (noisy * 255).astype(np.uint8), None,
        h=10,                  # filter strength
        templateWindowSize=7,  # patch size
        searchWindowSize=21    # search area
    ).astype(np.float32) / 255

    def psnr(a, b):
        mse = np.mean((a - b) ** 2)
        return 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

    print(f"  PSNR noisy:    {psnr(clean, noisy):.2f} dB")
    print(f"  PSNR median3:  {psnr(clean, median3):.2f} dB")
    print(f"  PSNR bilateral:{psnr(clean, bilateral_h):.2f} dB")
    print(f"  PSNR NLM:      {psnr(clean, nlm):.2f} dB")

    save_grid(
        [noisy, median3, bilateral_h, nlm],
        ["Noisy (S&P 4%)", "Median 3x3", "Bilateral σ=75", "NLM h=10"],
        "B3_nonlinear_filters.png"
    )
    print("  Done: B3 non-linear filters")


# ═════════════════════════════════════════════════════════════════════════════
# B4 — EDGE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def section_B4():
    print("\n── B4: Edge Detection ──")

    img = make_test_image()
    img_u8 = (img * 255).astype(np.uint8)

    # --- Sobel ---
    Gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx) * 180 / np.pi   # degrees

    mag_norm = magnitude / magnitude.max()
    print(f"  Sobel magnitude range: [{magnitude.min():.1f}, {magnitude.max():.1f}]")

    # --- Prewitt kernels (manual) ---
    Kx_prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    Ky_prewitt = np.array([[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]], dtype=np.float32)
    Px = convolve(img.astype(np.float32), Kx_prewitt)
    Py = convolve(img.astype(np.float32), Ky_prewitt)
    prewitt_mag = np.sqrt(Px**2 + Py**2)
    prewitt_mag = prewitt_mag / prewitt_mag.max()

    # --- Laplacian of Gaussian (LoG) —— zero-crossings mark edges ---
    def log_kernel(sigma, size=None):
        """Laplacian of Gaussian kernel."""
        if size is None:
            size = int(6 * sigma + 1) | 1
        r = size // 2
        y, x = np.mgrid[-r:r+1, -r:r+1]
        s2 = sigma ** 2
        k = -1 / (np.pi * s2**2) * (1 - (x**2+y**2)/(2*s2)) * np.exp(-(x**2+y**2)/(2*s2))
        return k - k.mean()

    log_k = log_kernel(sigma=2.0)
    log_response = convolve(img, log_k)

    # Zero-crossing detection for LoG
    def zero_crossings(response, threshold=0.01):
        """Mark pixels where sign changes between neighbouring pixels."""
        pos = response > threshold
        neg = response < -threshold
        edges = np.zeros_like(response, dtype=np.uint8)
        # Check 4-connected neighbours
        edges[1:, :] |= (pos[1:, :] & neg[:-1, :]) | (neg[1:, :] & pos[:-1, :])
        edges[:, 1:] |= (pos[:, 1:] & neg[:, :-1]) | (neg[:, 1:] & pos[:, :-1])
        return edges.astype(np.float32)

    log_edges = zero_crossings(log_response, threshold=0.005)
    print(f"  LoG zero-crossings: {log_edges.sum():.0f} edge pixels")

    save_grid(
        [img, mag_norm, prewitt_mag, log_edges],
        ["Original", "Sobel magnitude", "Prewitt magnitude", "LoG zero-crossings"],
        "B4_edge_detection.png"
    )
    print("  Done: B4 edge detection")


# ═════════════════════════════════════════════════════════════════════════════
# B5 — CANNY EDGE DETECTOR (from scratch)
# ═════════════════════════════════════════════════════════════════════════════

def section_B5():
    print("\n── B5: Canny Edge Detector ──")

    img = make_test_image()

    def canny_from_scratch(img, sigma=1.0, low_thresh=0.05, high_thresh=0.15):
        """
        Full 5-step Canny pipeline:
        1. Gaussian smoothing
        2. Gradient magnitude and direction (Sobel)
        3. Non-maximum suppression (NMS)
        4. Double threshold
        5. Hysteresis edge tracking
        """
        # Step 1: Gaussian smoothing
        ksize = int(6 * sigma + 1) | 1
        smoothed = cv2.GaussianBlur(img, (ksize, ksize), sigma)

        # Step 2: Gradient (Sobel)
        Gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(Gx**2 + Gy**2)
        ang = np.arctan2(Gy, Gx) * 180 / np.pi
        ang[ang < 0] += 180       # map to [0, 180)

        # Step 3: Non-maximum suppression
        # Round angle to 0, 45, 90, 135 degrees and compare along gradient direction
        H, W = mag.shape
        nms = np.zeros_like(mag)
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                a = ang[i, j]
                m = mag[i, j]
                if   (0   <= a < 22.5) or (157.5 <= a < 180):
                    n1, n2 = mag[i, j-1], mag[i, j+1]        # horizontal
                elif 22.5 <= a < 67.5:
                    n1, n2 = mag[i-1, j+1], mag[i+1, j-1]    # diagonal /
                elif 67.5 <= a < 112.5:
                    n1, n2 = mag[i-1, j], mag[i+1, j]         # vertical
                else:
                    n1, n2 = mag[i-1, j-1], mag[i+1, j+1]    # diagonal \
                nms[i, j] = m if (m >= n1 and m >= n2) else 0

        # Step 4: Double threshold
        high = high_thresh * nms.max()
        low  = low_thresh  * nms.max()
        strong = (nms >= high).astype(np.uint8)          # definite edges
        weak   = ((nms >= low) & (nms < high)).astype(np.uint8)  # candidates

        # Step 5: Hysteresis — keep weak pixels connected to strong ones
        edges = strong.copy()
        changed = True
        while changed:
            changed = False
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    if weak[i, j] and edges[i-1:i+2, j-1:j+2].any():
                        edges[i, j] = 1
                        weak[i, j]  = 0
                        changed = True

        return smoothed, mag / mag.max(), nms / (nms.max() + 1e-8), edges.astype(np.float32)

    # Run from-scratch Canny
    smoothed, grad_mag, after_nms, edges_scratch = canny_from_scratch(img, sigma=1.0)

    # cv2.Canny for comparison
    img_u8 = (img * 255).astype(np.uint8)
    edges_cv2 = cv2.Canny(img_u8, threshold1=30, threshold2=90).astype(np.float32) / 255

    # Overlap check
    overlap = np.logical_and(edges_scratch > 0, edges_cv2 > 0).sum()
    total   = max(edges_scratch.sum(), edges_cv2.sum())
    print(f"  From-scratch Canny edge pixels:  {edges_scratch.sum():.0f}")
    print(f"  cv2.Canny edge pixels:           {edges_cv2.sum():.0f}")
    print(f"  Overlap (Dice):                  {2*overlap/(edges_scratch.sum()+edges_cv2.sum()):.2f}")

    save_grid(
        [img, grad_mag, after_nms, edges_scratch, edges_cv2],
        ["Original", "Gradient mag", "After NMS", "Canny (scratch)", "cv2.Canny"],
        "B5_canny.png"
    )
    print("  Done: B5 Canny edge detector")


# ═════════════════════════════════════════════════════════════════════════════
# B6 — FREQUENCY DOMAIN
# ═════════════════════════════════════════════════════════════════════════════

def section_B6():
    print("\n── B6: Frequency Domain Filtering ──")

    img = make_test_image()

    # --- DFT ---
    dft   = np.fft.fft2(img)
    dft_s = np.fft.fftshift(dft)          # shift DC to centre
    magnitude_spectrum = np.log1p(np.abs(dft_s))
    magnitude_spectrum /= magnitude_spectrum.max()

    H, W = img.shape
    cy, cx = H // 2, W // 2

    # --- Low-pass filter (remove high frequencies = smooth) ---
    def ideal_circle_mask(H, W, radius):
        yy, xx = np.ogrid[:H, :W]
        dist   = np.sqrt((xx - W//2)**2 + (yy - H//2)**2)
        return (dist <= radius).astype(np.float32)

    lp_mask   = ideal_circle_mask(H, W, radius=30)
    lp_dft    = dft_s * lp_mask
    lp_img    = np.real(np.fft.ifft2(np.fft.ifftshift(lp_dft)))
    lp_img    = np.clip(lp_img, 0, 1)

    # --- High-pass filter (keep high frequencies = edges) ---
    hp_mask   = 1 - lp_mask
    hp_dft    = dft_s * hp_mask
    hp_img    = np.real(np.fft.ifft2(np.fft.ifftshift(hp_dft)))
    hp_img    = (hp_img - hp_img.min()) / (hp_img.max() - hp_img.min() + 1e-8)

    # --- Gaussian low-pass filter in frequency domain ---
    sigma_f = 30.0
    yy, xx = np.ogrid[:H, :W]
    gauss_mask = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma_f**2))
    glp_dft  = dft_s * gauss_mask
    glp_img  = np.real(np.fft.ifft2(np.fft.ifftshift(glp_dft)))
    glp_img  = np.clip(glp_img, 0, 1)

    # --- Unsharp masking (sharpen = original + alpha * high-frequency detail) ---
    def unsharp_mask(img, sigma=2.0, alpha=1.5):
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        detail  = img - blurred
        return np.clip(img + alpha * detail, 0, 1)

    sharpened = unsharp_mask(img, sigma=2.0, alpha=1.5)
    print(f"  Unsharp mask — sharpened image range: [{sharpened.min():.3f}, {sharpened.max():.3f}]")

    # --- Frequency content comparison ---
    for label, image in [("Original", img), ("LP filtered", lp_img), ("Sharpened", sharpened)]:
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(image)))
        # Fraction of energy in high frequencies (outer 50% of radius)
        full_energy = (fft_mag ** 2).sum()
        mask_inner  = ideal_circle_mask(H, W, radius=min(H, W) // 4)
        inner_energy = (fft_mag ** 2 * mask_inner).sum()
        print(f"  {label:15s}: high-freq energy fraction = "
              f"{(full_energy - inner_energy) / full_energy:.3f}")

    save_grid(
        [img, magnitude_spectrum, lp_img, hp_img, glp_img, sharpened],
        ["Original", "DFT magnitude", "Ideal LP", "Ideal HP", "Gauss LP", "Unsharp"],
        "B6_frequency.png"
    )
    print("  Done: B6 frequency domain")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CV Section B — Image Filters & Convolution")
    print("=" * 60)
    np.random.seed(42)

    section_B1()   # convolution mechanics, output size, param count
    section_B2()   # box, Gaussian, separability
    section_B3()   # median, bilateral, NLM
    section_B4()   # Sobel, Prewitt, LoG
    section_B5()   # full Canny from scratch
    section_B6()   # DFT, low-pass, high-pass, unsharp masking

    print("\n✓ All Section B demos complete.")
    print("  Output images saved to current directory.")
