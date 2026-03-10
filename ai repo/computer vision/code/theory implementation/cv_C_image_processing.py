"""
cv_C_image_processing.py
=========================
Computer Vision Course — Section C: Image Processing Fundamentals

Topics covered:
  C1 - Histogram equalisation: global HE, CLAHE
  C2 - Morphological operations: erosion, dilation, opening, closing,
       top-hat, black-hat, hit-or-miss, skeleton
  C3 - Image degradation model: g = h*f + n, PSF types (Gaussian, motion)
  C4 - Image restoration: Wiener filter, inverse filter
  C5 - Denoising comparison: Gaussian, Median, Bilateral, NLM

Dependencies: numpy, opencv-python, matplotlib, scipy
Install:  pip install numpy opencv-python matplotlib scipy
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import convolve
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_test_image(h=256, w=256):
    img = np.zeros((h, w), dtype=np.float32)
    img[40:80,  30:120]  = 0.9   # bright rectangle
    img[100:180,60:200]  = 0.5   # medium rectangle
    img[180:220,100:160] = 0.3   # dark rectangle
    cx, cy, r = 200, 50, 35
    yy, xx = np.ogrid[:h, :w]
    img[(xx-cx)**2 + (yy-cy)**2 < r**2] = 1.0
    return np.clip(img, 0, 1)


def make_dark_image():
    """Low-contrast dark image for histogram equalisation demo."""
    img = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            img[i, j] = int((i / 256 * 0.4 + j / 256 * 0.3) * 100 + 30)
    return img


def save_grid(imgs, titles, filename, cmap="gray", norm=True):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        vmin, vmax = (0, 1) if (norm and img.max() <= 1.0) else (None, None)
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  Saved: {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# C1 — HISTOGRAM EQUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def section_C1():
    print("\n── C1: Histogram Equalisation ──")

    img_dark = make_dark_image()

    # --- Manual global histogram equalisation ---
    def hist_equalise(img):
        """
        HE pipeline:
        1. Compute normalised histogram (PMF)
        2. Compute CDF
        3. Map: T(r) = round((L-1) * CDF(r))
        """
        L = 256
        h_flat = img.flatten()

        # PMF
        hist, _ = np.histogram(h_flat, bins=L, range=(0, L))
        pmf = hist / h_flat.size

        # CDF
        cdf = np.cumsum(pmf)

        # Mapping
        lut = np.round((L - 1) * cdf).astype(np.uint8)
        return lut[img]

    eq_manual = hist_equalise(img_dark)
    eq_cv2    = cv2.equalizeHist(img_dark)

    diff = np.abs(eq_manual.astype(int) - eq_cv2.astype(int)).max()
    print(f"  Manual HE vs cv2.equalizeHist max diff: {diff}")

    # --- CLAHE (Contrast Limited Adaptive Histogram Equalisation) ---
    # Divides image into tiles, applies HE per tile, clips histogram to limit,
    # bilinear interpolation at tile borders to avoid blocking artefacts.
    clahe_mild   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_strong = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8))
    clahe_small  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

    eq_clahe_mild   = clahe_mild.apply(img_dark)
    eq_clahe_strong = clahe_strong.apply(img_dark)
    eq_clahe_small  = clahe_small.apply(img_dark)

    # --- Measure contrast improvement ---
    def rms_contrast(img):
        f = img.astype(np.float32) / 255.0
        return f.std()

    print(f"  RMS contrast original: {rms_contrast(img_dark):.4f}")
    print(f"  RMS contrast global HE:{rms_contrast(eq_manual):.4f}")
    print(f"  RMS contrast CLAHE 2.0:{rms_contrast(eq_clahe_mild):.4f}")
    print(f"  RMS contrast CLAHE 8.0:{rms_contrast(eq_clahe_strong):.4f}")

    # --- Histogram comparison ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    images = [img_dark, eq_manual, eq_clahe_mild, eq_clahe_strong]
    labels = ["Original", "Global HE", "CLAHE clip=2", "CLAHE clip=8"]
    for col, (im, lb) in enumerate(zip(images, labels)):
        axes[0, col].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(lb, fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].hist(im.flatten(), bins=64, range=(0,255), color="steelblue")
        axes[1, col].set_title(f"Histogram", fontsize=9)
    plt.tight_layout()
    plt.savefig("C1_histogram_eq.png", dpi=100)
    plt.close()
    print("  Saved: C1_histogram_eq.png")
    print("  Done: C1 histogram equalisation")


# ═════════════════════════════════════════════════════════════════════════════
# C2 — MORPHOLOGICAL OPERATIONS
# ═════════════════════════════════════════════════════════════════════════════

def section_C2():
    print("\n── C2: Morphological Operations ──")

    img = make_test_image()
    img_u8 = (img * 255).astype(np.uint8)
    # Binary version
    _, binary = cv2.threshold(img_u8, 127, 255, cv2.THRESH_BINARY)

    # --- Structuring elements ---
    se_rect   = cv2.getStructuringElement(cv2.MORPH_RECT,    (5, 5))
    se_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    se_cross  = cv2.getStructuringElement(cv2.MORPH_CROSS,   (5, 5))

    # --- Basic operations ---
    eroded    = cv2.erode(binary, se_rect, iterations=1)
    dilated   = cv2.dilate(binary, se_rect, iterations=1)
    opened    = cv2.morphologyEx(binary, cv2.MORPH_OPEN,    se_ellipse)  # remove small bright
    closed    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,   se_ellipse)  # fill small dark holes
    gradient  = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT,se_rect)     # dilate - erode = outline
    tophat    = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT,  se_rect)     # img - opened (bright detail on dark)
    blackhat  = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT,se_rect)     # closed - img (dark detail on bright)

    print(f"  Erosion  removes pixels: {(binary>0).sum() - (eroded>0).sum()}")
    print(f"  Dilation adds pixels:    {(dilated>0).sum() - (binary>0).sum()}")
    print(f"  Opening  removes pixels: {(binary>0).sum() - (opened>0).sum()}")
    print(f"  Closing  adds pixels:    {(closed>0).sum()  - (binary>0).sum()}")

    # --- Manual erosion for understanding ---
    def erode_manual(img_bin, se):
        """
        Binary erosion: a pixel stays ON only if ALL pixels under the
        structuring element (centred at that pixel) are ON.
        """
        h, w = img_bin.shape
        kh, kw = se.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(img_bin, ((ph,ph),(pw,pw)), mode="constant", constant_values=0)
        out = np.zeros_like(img_bin)
        se_bool = se.astype(bool)
        for i in range(h):
            for j in range(w):
                patch = padded[i:i+kh, j:j+kw]
                out[i, j] = 255 if np.all(patch[se_bool] == 255) else 0
        return out

    eroded_manual = erode_manual(binary, se_rect)
    diff = np.abs(eroded.astype(int) - eroded_manual.astype(int)).max()
    print(f"  Manual erosion vs cv2 max diff: {diff}")

    # --- Skeletonisation (iterative thinning) ---
    def skeletonise(img_bin):
        """Zhang-Suen thinning approximation via repeated erosion + subtraction."""
        skel  = np.zeros_like(img_bin)
        temp  = img_bin.copy()
        se_e  = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        while True:
            eroded_t = cv2.erode(temp, se_e)
            opened_t = cv2.dilate(eroded_t, se_e)
            diff_t   = cv2.subtract(temp, opened_t)
            skel     = cv2.bitwise_or(skel, diff_t)
            temp     = eroded_t.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skel

    skeleton = skeletonise(binary)
    print(f"  Skeleton pixels: {(skeleton>0).sum()} (vs original {(binary>0).sum()})")

    save_grid(
        [binary, eroded, dilated, opened, closed, gradient, tophat, skeleton],
        ["Binary", "Eroded", "Dilated", "Opened", "Closed", "Gradient", "Top-hat", "Skeleton"],
        "C2_morphology.png"
    )
    print("  Done: C2 morphological operations")


# ═════════════════════════════════════════════════════════════════════════════
# C3 — IMAGE DEGRADATION MODEL
# ═════════════════════════════════════════════════════════════════════════════

def section_C3():
    print("\n── C3: Image Degradation Model ──")

    clean = make_test_image()

    # --- PSF Types ---

    def gaussian_psf(size=21, sigma=3.0):
        """Gaussian blur PSF — models out-of-focus lens."""
        r = size // 2
        y, x = np.mgrid[-r:r+1, -r:r+1]
        k = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return k / k.sum()

    def motion_psf(length=20, angle=45.0):
        """
        Linear motion blur PSF — models camera shake.
        Creates a line kernel at given angle.
        """
        size = max(length, 1)
        k = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        angle_rad = np.deg2rad(angle)
        for i in range(size):
            offset = i - center
            x = center + int(round(offset * np.cos(angle_rad)))
            y = center + int(round(offset * np.sin(angle_rad)))
            if 0 <= x < size and 0 <= y < size:
                k[y, x] = 1
        s = k.sum()
        return k / s if s > 0 else k

    def defocus_psf(size=21, radius=8):
        """Disk (pillbox) PSF — models circular aperture defocus."""
        k = np.zeros((size, size), dtype=np.float32)
        cy, cx = size//2, size//2
        yy, xx = np.ogrid[:size, :size]
        k[(xx-cx)**2 + (yy-cy)**2 <= radius**2] = 1
        return k / k.sum()

    # --- Degradation model: g = h * f + n ---
    def degrade(img, psf, noise_sigma=0.01):
        """Apply PSF convolution + additive Gaussian noise."""
        blurred = fftconvolve(img, psf, mode="same")
        noise   = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        return np.clip(blurred + noise, 0, 1)

    psf_gauss  = gaussian_psf(size=21, sigma=3.0)
    psf_motion = motion_psf(length=20, angle=45)
    psf_defocus= defocus_psf(size=21, radius=8)

    degraded_gauss  = degrade(clean, psf_gauss,   noise_sigma=0.01)
    degraded_motion = degrade(clean, psf_motion,  noise_sigma=0.01)
    degraded_defocus= degrade(clean, psf_defocus, noise_sigma=0.01)

    def psnr(a, b):
        mse = np.mean((a-b)**2)
        return 10*np.log10(1/mse) if mse > 0 else float("inf")

    print(f"  PSNR after Gaussian blur  (sigma=3): {psnr(clean, degraded_gauss):.2f} dB")
    print(f"  PSNR after Motion blur    (len=20):  {psnr(clean, degraded_motion):.2f} dB")
    print(f"  PSNR after Defocus blur   (r=8):     {psnr(clean, degraded_defocus):.2f} dB")

    save_grid(
        [clean, degraded_gauss, degraded_motion, degraded_defocus],
        ["Clean", "Gaussian blur+noise", "Motion blur+noise", "Defocus blur+noise"],
        "C3_degradation.png"
    )
    print("  Done: C3 image degradation model")


# ═════════════════════════════════════════════════════════════════════════════
# C4 — IMAGE RESTORATION
# ═════════════════════════════════════════════════════════════════════════════

def section_C4():
    print("\n── C4: Image Restoration ──")

    from scipy.signal import fftconvolve

    clean = make_test_image()

    # Build a known PSF and degrade
    def gaussian_psf(size=21, sigma=2.0):
        r = size // 2
        y, x = np.mgrid[-r:r+1, -r:r+1]
        k = np.exp(-(x**2+y**2)/(2*sigma**2))
        return (k / k.sum()).astype(np.float32)

    psf    = gaussian_psf(size=21, sigma=2.0)
    noise_sigma = 0.02
    blurred = fftconvolve(clean, psf, mode="same")
    noisy   = np.clip(blurred + np.random.normal(0, noise_sigma, clean.shape), 0, 1)

    # --- Inverse filter (naive — amplifies noise horribly) ---
    def inverse_filter(degraded, psf):
        """
        Inverse filter in frequency domain:
        F_hat(u,v) = G(u,v) / H(u,v)
        Problem: H(u,v) near zero → massive noise amplification.
        """
        H, W = degraded.shape
        psf_padded = np.zeros_like(degraded)
        ph, pw = psf.shape[0]//2, psf.shape[1]//2
        psf_padded[:psf.shape[0], :psf.shape[1]] = psf
        psf_padded = np.roll(np.roll(psf_padded, -ph, axis=0), -pw, axis=1)

        G = np.fft.fft2(degraded)
        Hf = np.fft.fft2(psf_padded)
        # Add small epsilon to avoid division by zero
        F_hat = G / (Hf + 1e-6)
        return np.real(np.fft.ifft2(F_hat))

    # --- Wiener filter (optimal linear restoration) ---
    def wiener_filter(degraded, psf, K=0.01):
        """
        Wiener filter:
        F_hat = H* / (|H|^2 + K) * G
        K = noise-to-signal power ratio.
        K small → approaches inverse filter (sharp but noisy).
        K large → approaches zero (over-smoothed).
        Optimal K ≈ noise_power / signal_power.
        """
        H_size, W_size = degraded.shape
        psf_padded = np.zeros_like(degraded, dtype=np.float64)
        ph, pw = psf.shape[0]//2, psf.shape[1]//2
        psf_padded[:psf.shape[0], :psf.shape[1]] = psf
        psf_padded = np.roll(np.roll(psf_padded, -ph, axis=0), -pw, axis=1)

        G  = np.fft.fft2(degraded.astype(np.float64))
        Hf = np.fft.fft2(psf_padded)
        Hc = np.conj(Hf)
        H2 = np.abs(Hf) ** 2
        F_hat = (Hc / (H2 + K)) * G
        return np.real(np.fft.ifft2(F_hat))

    inv_restored   = np.clip(inverse_filter(noisy, psf), 0, 1)
    wien_tight     = np.clip(wiener_filter(noisy, psf, K=0.001), 0, 1)
    wien_optimal   = np.clip(wiener_filter(noisy, psf, K=0.01),  0, 1)
    wien_smooth    = np.clip(wiener_filter(noisy, psf, K=0.1),   0, 1)

    def psnr(a, b):
        mse = np.mean((a-b)**2)
        return 10*np.log10(1/mse) if mse > 0 else float("inf")

    print(f"  PSNR degraded:              {psnr(clean, noisy):.2f} dB")
    print(f"  PSNR inverse filter:        {psnr(clean, inv_restored):.2f} dB  (noisy!)")
    print(f"  PSNR Wiener K=0.001:        {psnr(clean, wien_tight):.2f} dB")
    print(f"  PSNR Wiener K=0.01:         {psnr(clean, wien_optimal):.2f} dB")
    print(f"  PSNR Wiener K=0.1:          {psnr(clean, wien_smooth):.2f} dB  (over-smoothed)")

    save_grid(
        [clean, noisy, inv_restored, wien_optimal],
        ["Clean", "Degraded (blur+noise)", "Inverse filter", "Wiener K=0.01"],
        "C4_restoration.png"
    )
    print("  Done: C4 image restoration")


# ═════════════════════════════════════════════════════════════════════════════
# C5 — DENOISING COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def section_C5():
    print("\n── C5: Denoising Comparison ──")

    clean = make_test_image()

    # Add Gaussian noise
    noise_sigma = 0.05
    noisy = np.clip(clean + np.random.normal(0, noise_sigma, clean.shape), 0, 1)
    noisy_u8 = (noisy * 255).astype(np.uint8)

    # --- 1. Gaussian denoising ---
    gauss_dn = cv2.GaussianBlur(noisy, (0, 0), sigmaX=1.5).astype(np.float32) / 255

    # --- 2. Median denoising ---
    median_dn = cv2.medianBlur(noisy_u8, 3).astype(np.float32) / 255

    # --- 3. Bilateral denoising ---
    bilateral_dn = cv2.bilateralFilter(noisy_u8, d=9, sigmaColor=50, sigmaSpace=50).astype(np.float32) / 255

    # --- 4. Non-local means ---
    nlm_dn = cv2.fastNlMeansDenoising(
        noisy_u8, None, h=15, templateWindowSize=7, searchWindowSize=21
    ).astype(np.float32) / 255

    # --- 5. Total Variation denoising (Chambolle's algorithm) ---
    def tv_denoise(img, weight=0.1, n_iter=50):
        """
        Chambolle's total variation denoising.
        Minimises: 0.5*||u - f||^2 + weight * TV(u)
        TV(u) = sum |grad(u)|  (promotes piecewise-constant images).
        """
        u = img.copy().astype(np.float64)
        px = np.zeros_like(u)
        py = np.zeros_like(u)
        tau = 0.25

        for _ in range(n_iter):
            # Gradient of u
            ux = np.roll(u, -1, axis=1) - u
            uy = np.roll(u, -1, axis=0) - u
            # Update dual variable p
            px_new = px + tau * ux
            py_new = py + tau * uy
            # Project onto unit ball
            norm = np.maximum(1.0, np.sqrt(px_new**2 + py_new**2) / weight)
            px = px_new / norm
            py = py_new / norm
            # Divergence of p
            div_p = (px - np.roll(px, 1, axis=1) +
                     py - np.roll(py, 1, axis=0))
            u = img - div_p

        return np.clip(u, 0, 1).astype(np.float32)

    tv_dn = tv_denoise(noisy, weight=0.1, n_iter=50)

    def psnr(a, b):
        mse = np.mean((a-b)**2)
        return 10*np.log10(1/mse) if mse > 0 else float("inf")

    print(f"  PSNR noisy:     {psnr(clean, noisy):.2f} dB")
    print(f"  PSNR Gaussian:  {psnr(clean, gauss_dn):.2f} dB")
    print(f"  PSNR Median:    {psnr(clean, median_dn):.2f} dB")
    print(f"  PSNR Bilateral: {psnr(clean, bilateral_dn):.2f} dB")
    print(f"  PSNR NLM:       {psnr(clean, nlm_dn):.2f} dB")
    print(f"  PSNR TV:        {psnr(clean, tv_dn):.2f} dB")

    save_grid(
        [noisy, gauss_dn, median_dn, bilateral_dn, nlm_dn, tv_dn],
        ["Noisy", "Gaussian σ=1.5", "Median 3x3", "Bilateral", "NLM h=15", "TV λ=0.1"],
        "C5_denoising.png"
    )
    print("  Done: C5 denoising comparison")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CV Section C — Image Processing Fundamentals")
    print("=" * 60)
    np.random.seed(42)

    section_C1()   # histogram equalisation, CLAHE
    section_C2()   # morphological operations
    section_C3()   # degradation model: PSF types
    section_C4()   # Wiener filter restoration
    section_C5()   # denoising comparison incl. TV

    print("\n✓ All Section C demos complete.")
    print("  Output images saved to current directory.")
