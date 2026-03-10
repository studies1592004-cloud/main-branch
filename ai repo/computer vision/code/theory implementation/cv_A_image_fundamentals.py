"""
cv_A_image_fundamentals.py
==========================
Computer Vision Course — Section A: Digital Image Fundamentals

Topics covered:
  A1 - Image representation: pixels, channels, dtypes, HWC vs CHW
  A2 - Colour space conversions: RGB, BGR, HSV, LAB, YCbCr, Grayscale
  A3 - Sampling & interpolation: nearest, bilinear, bicubic resize
  A4 - Quantisation: bit-depth reduction and dithering
  A5 - Noise models: Gaussian, Salt-and-Pepper, Poisson, Speckle
  A6 - Annotation formats: COCO (absolute), YOLO (normalised), Pascal VOC (XML)
       and RLE (run-length encoding) for binary masks

Dependencies: numpy, opencv-python, matplotlib
Install:  pip install numpy opencv-python matplotlib
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")           # headless — remove if you have a display
import matplotlib.pyplot as plt
import json
import struct
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_sample_image(h=256, w=256):
    """Create a synthetic RGB image with gradients + shapes for demos."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # horizontal red gradient
    img[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    # vertical green gradient
    img[:, :, 1] = np.tile(np.linspace(0, 255, h, dtype=np.uint8), (w, 1)).T
    # blue circle
    cx, cy, r = w // 2, h // 2, 60
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
    img[mask, 2] = 200
    return img


def show_and_save(imgs, titles, filename, cmap=None):
    """Display a row of images and save to disk."""
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  Saved: {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# A1 — IMAGE REPRESENTATION
# ═════════════════════════════════════════════════════════════════════════════

def section_A1():
    print("\n── A1: Image Representation ──")

    img_bgr = make_sample_image()          # shape (H, W, 3), dtype uint8

    # --- Pixel layout ---
    H, W, C = img_bgr.shape
    print(f"  Shape (H, W, C): {img_bgr.shape}")
    print(f"  dtype: {img_bgr.dtype}  |  min: {img_bgr.min()}  max: {img_bgr.max()}")

    # Single pixel access
    pixel = img_bgr[100, 150]             # (B, G, R) for OpenCV images
    print(f"  Pixel at (100,150) [BGR]: {pixel}")

    # --- HWC  ↔  CHW conversion (PyTorch uses CHW) ---
    img_chw = img_bgr.transpose(2, 0, 1)  # (C, H, W)
    img_hwc = img_chw.transpose(1, 2, 0)  # back to (H, W, C)
    assert np.array_equal(img_bgr, img_hwc), "Round-trip failed"
    print(f"  HWC shape: {img_bgr.shape}  →  CHW shape: {img_chw.shape}")

    # --- dtype conversions ---
    img_f32 = img_bgr.astype(np.float32) / 255.0      # [0,1]  float32
    img_f16 = img_f32.astype(np.float16)               # half precision
    img_u8  = (img_f32 * 255).clip(0, 255).astype(np.uint8)  # back to uint8
    print(f"  float32 range: [{img_f32.min():.2f}, {img_f32.max():.2f}]")

    # --- Individual channels ---
    B, G, R = cv2.split(img_bgr)          # each is (H, W)
    img_red_only = np.zeros_like(img_bgr)
    img_red_only[:, :, 2] = R             # only red channel populated

    show_and_save(
        [img_bgr, B, G, R],
        ["Original (BGR)", "Blue channel", "Green channel", "Red channel"],
        "A1_channels.png"
    )
    print("  Done: A1 image representation")


# ═════════════════════════════════════════════════════════════════════════════
# A2 — COLOUR SPACE CONVERSIONS
# ═════════════════════════════════════════════════════════════════════════════

def section_A2():
    print("\n── A2: Colour Spaces ──")

    img_bgr = make_sample_image()

    # --- OpenCV colour conversions ---
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # (H, W)
    img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)   # H:[0,179] S:[0,255] V:[0,255]
    img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)   # L:[0,255] a:[0,255] b:[0,255]
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb) # Y Cr Cb

    # --- Manual BGR → Grayscale (the formula) ---
    #   Gray = 0.114*B + 0.587*G + 0.299*R  (ITU-R BT.601 luma)
    img_f = img_bgr.astype(np.float32)
    gray_manual = (0.114 * img_f[:,:,0] +
                   0.587 * img_f[:,:,1] +
                   0.299 * img_f[:,:,2]).clip(0, 255).astype(np.uint8)
    diff = np.abs(gray_manual.astype(int) - img_gray.astype(int)).max()
    print(f"  Manual vs cv2 grayscale max diff: {diff}")  # should be ≤ 1

    # --- HSV colour thresholding example ---
    # Select pixels in a hue range (e.g., green hues: H 40-80)
    lower = np.array([40, 40, 40])
    upper = np.array([80, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower, upper)  # binary mask

    # --- LAB colour distance (ΔE) ---
    # LAB is perceptually uniform: ΔE ≈ perceptual colour difference
    lab1 = np.array([[[50, 0, 0]]], dtype=np.uint8)     # neutral grey
    lab2 = np.array([[[50, 60, 0]]], dtype=np.uint8)    # reddish
    delta_e = np.sqrt(np.sum((lab1.astype(float) - lab2.astype(float)) ** 2))
    print(f"  Delta-E between two LAB colours: {delta_e:.2f}")

    # --- YCbCr: separate luma from chroma ---
    Y, Cr, Cb = cv2.split(img_ycbcr)
    print(f"  YCbCr channels — Y range: [{Y.min()},{Y.max()}]  "
          f"Cb: [{Cb.min()},{Cb.max()}]  Cr: [{Cr.min()},{Cr.max()}]")

    show_and_save(
        [img_bgr, img_gray, img_hsv, img_lab],
        ["BGR", "Grayscale", "HSV", "LAB"],
        "A2_colorspaces.png"
    )
    print("  Done: A2 colour spaces")


# ═════════════════════════════════════════════════════════════════════════════
# A3 — SAMPLING & INTERPOLATION
# ═════════════════════════════════════════════════════════════════════════════

def section_A3():
    print("\n── A3: Sampling & Interpolation ──")

    img = make_sample_image(256, 256)

    # --- Downsample then upsample with different interpolation methods ---
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

    methods = {
        "Nearest":  cv2.INTER_NEAREST,
        "Bilinear": cv2.INTER_LINEAR,
        "Bicubic":  cv2.INTER_CUBIC,
        "Lanczos":  cv2.INTER_LANCZOS4,
    }

    results, titles = [small], ["Downsampled 64x64"]
    for name, flag in methods.items():
        up = cv2.resize(small, (256, 256), interpolation=flag)
        results.append(up)
        titles.append(name)

    show_and_save(results, titles, "A3_interpolation.png")

    # --- Manual bilinear interpolation (for understanding) ---
    def bilinear_sample(img, x, y):
        """Sample img at floating-point location (x, y) using bilinear interp."""
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)
        dx, dy = x - x0, y - y0
        return ((1 - dx) * (1 - dy) * img[y0, x0] +
                dx        * (1 - dy) * img[y0, x1] +
                (1 - dx) * dy        * img[y1, x0] +
                dx        * dy        * img[y1, x1])

    val = bilinear_sample(img[:, :, 0].astype(float), 127.5, 127.5)
    print(f"  Manual bilinear sample at (127.5, 127.5) red channel: {val:.2f}")

    # --- Nyquist: aliasing demo ---
    # High-frequency pattern — downsampling without anti-aliasing causes aliasing
    freq_img = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        if (i // 2) % 2 == 0:
            freq_img[:, i] = 255          # alternating black/white columns

    aliased  = cv2.resize(freq_img, (64, 64), interpolation=cv2.INTER_NEAREST)
    antialiased = cv2.resize(freq_img, (64, 64), interpolation=cv2.INTER_AREA)
    print("  Aliased downsample — unique values:", np.unique(aliased).tolist())
    print("  Anti-aliased downsample — unique values sample:", np.unique(antialiased)[:6].tolist())
    print("  Done: A3 sampling & interpolation")


# ═════════════════════════════════════════════════════════════════════════════
# A4 — QUANTISATION
# ═════════════════════════════════════════════════════════════════════════════

def section_A4():
    print("\n── A4: Quantisation ──")

    img = make_sample_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def quantise(img_f32, bits):
        """Reduce image to `bits` bits per channel."""
        levels = 2 ** bits
        quantised = np.floor(img_f32 * levels) / levels  # step function
        return quantised

    results, titles = [], []
    for bits in [8, 4, 2, 1]:
        q = quantise(gray, bits)
        results.append((q * 255).astype(np.uint8))
        titles.append(f"{bits}-bit ({2**bits} levels)")

    show_and_save(results, titles, "A4_quantisation.png", cmap="gray")

    # --- Quantisation error ---
    q4 = quantise(gray, 4)
    error = np.abs(gray - q4)
    print(f"  4-bit quantisation — max error: {error.max():.4f}  "
          f"mean error: {error.mean():.4f}")

    # --- Floyd-Steinberg dithering (reduces banding artefacts) ---
    def floyd_steinberg(img_f32):
        """Apply Floyd-Steinberg error-diffusion dithering for 1-bit output."""
        img = img_f32.copy()
        h, w = img.shape
        for y in range(h):
            for x in range(w):
                old = img[y, x]
                new = 1.0 if old >= 0.5 else 0.0
                img[y, x] = new
                err = old - new
                if x + 1 < w:            img[y,   x+1] += err * 7/16
                if y + 1 < h:
                    if x - 1 >= 0:        img[y+1, x-1] += err * 3/16
                    img[y+1, x]           += err * 5/16
                    if x + 1 < w:         img[y+1, x+1] += err * 1/16
        return img

    # Dithering on small patch for speed
    patch = gray[:64, :64]
    dithered = floyd_steinberg(patch.copy())
    print(f"  Floyd-Steinberg dithered image — unique values: {np.unique(dithered.round(2))}")
    print("  Done: A4 quantisation")


# ═════════════════════════════════════════════════════════════════════════════
# A5 — NOISE MODELS
# ═════════════════════════════════════════════════════════════════════════════

def section_A5():
    print("\n── A5: Noise Models ──")

    img = make_sample_image().astype(np.float32) / 255.0  # [0,1]

    # --- 1. Gaussian (Additive White Gaussian Noise) ---
    def gaussian_noise(img, sigma=0.05):
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img + noise, 0, 1)

    # --- 2. Salt-and-Pepper ---
    def salt_and_pepper(img, prob=0.05):
        out = img.copy()
        rng = np.random.rand(*img.shape[:2])
        out[rng < prob / 2]       = 0.0   # pepper
        out[rng > 1 - prob / 2]   = 1.0   # salt
        return out

    # --- 3. Poisson (shot noise — depends on signal level) ---
    def poisson_noise(img, scale=255.0):
        """Scale image to photon counts, sample Poisson, rescale."""
        counts = img * scale
        noisy  = np.random.poisson(counts).astype(np.float32) / scale
        return np.clip(noisy, 0, 1)

    # --- 4. Speckle (multiplicative noise, common in radar/ultrasound) ---
    def speckle_noise(img, sigma=0.1):
        noise = np.random.normal(1.0, sigma, img.shape).astype(np.float32)
        return np.clip(img * noise, 0, 1)

    noisy_gauss = gaussian_noise(img, sigma=0.05)
    noisy_sp    = salt_and_pepper(img, prob=0.05)
    noisy_pois  = poisson_noise(img, scale=255)
    noisy_speck = speckle_noise(img, sigma=0.1)

    def psnr(clean, noisy):
        mse = np.mean((clean - noisy) ** 2)
        if mse == 0:
            return float("inf")
        return 10 * np.log10(1.0 / mse)

    print(f"  PSNR Gaussian noise (sigma=0.05): {psnr(img, noisy_gauss):.2f} dB")
    print(f"  PSNR Salt-and-Pepper (p=0.05):   {psnr(img, noisy_sp):.2f} dB")
    print(f"  PSNR Poisson noise:               {psnr(img, noisy_pois):.2f} dB")
    print(f"  PSNR Speckle noise (sigma=0.1):   {psnr(img, noisy_speck):.2f} dB")

    def to_u8(x):
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    show_and_save(
        [to_u8(img), to_u8(noisy_gauss), to_u8(noisy_sp),
         to_u8(noisy_pois), to_u8(noisy_speck)],
        ["Clean", "Gaussian", "Salt&Pepper", "Poisson", "Speckle"],
        "A5_noise.png"
    )
    print("  Done: A5 noise models")


# ═════════════════════════════════════════════════════════════════════════════
# A6 — ANNOTATION FORMATS
# ═════════════════════════════════════════════════════════════════════════════

def section_A6():
    print("\n── A6: Annotation Formats ──")

    H, W = 480, 640   # image dimensions used in examples

    # ── COCO format (JSON, absolute pixel coordinates) ──────────────────────
    coco_annotation = {
        "image_id": 1,
        "category_id": 1,             # 1 = person
        "bbox": [120, 80, 200, 300],  # [x_min, y_min, width, height] in pixels
        "area": 200 * 300,
        "iscrowd": 0,
        "id": 42,
        # Segmentation polygon (list of [x,y,x,y,...] per polygon)
        "segmentation": [[120, 80, 320, 80, 320, 380, 120, 380]],
        # Keypoints: [x1,y1,v1, x2,y2,v2, ...] v=0 not labelled, 1 occluded, 2 visible
        "keypoints": [220, 90, 2,  # nose
                      215, 85, 2,  # left eye
                      225, 85, 2,  # right eye
                      210, 88, 1,  # left ear
                      230, 88, 1,  # right ear
                      ],
        "num_keypoints": 5
    }

    # Convert COCO bbox → corners
    x, y, bw, bh = coco_annotation["bbox"]
    x1, y1, x2, y2 = x, y, x + bw, y + bh
    print(f"  COCO bbox [x,y,w,h]: {[x,y,bw,bh]}")
    print(f"  → corners [x1,y1,x2,y2]: {[x1,y1,x2,y2]}")

    # ── YOLO format (txt, normalised [0,1], relative to image size) ──────────
    def coco_to_yolo(x, y, w, h, img_w, img_h):
        """Convert COCO bbox to YOLO normalised format."""
        x_c = (x + w / 2) / img_w
        y_c = (y + h / 2) / img_h
        w_n = w / img_w
        h_n = h / img_h
        return x_c, y_c, w_n, h_n

    yolo_box = coco_to_yolo(x, y, bw, bh, W, H)
    yolo_line = f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
    print(f"  YOLO line: {yolo_line}")

    def yolo_to_coco(x_c, y_c, w_n, h_n, img_w, img_h):
        """Inverse: YOLO normalised → COCO pixel bbox."""
        w   = w_n  * img_w
        h   = h_n  * img_h
        x   = (x_c * img_w) - w / 2
        y   = (y_c * img_h) - h / 2
        return int(x), int(y), int(w), int(h)

    recovered = yolo_to_coco(*yolo_box, W, H)
    print(f"  YOLO → COCO recovered: {list(recovered)}  (original: {[x,y,bw,bh]})")

    # ── Pascal VOC format (XML) ──────────────────────────────────────────────
    pascal_voc_xml = f"""<annotation>
  <filename>image001.jpg</filename>
  <size><width>{W}</width><height>{H}</height><depth>3</depth></size>
  <object>
    <name>person</name>
    <difficult>0</difficult>
    <bndbox>
      <xmin>{x1}</xmin><ymin>{y1}</ymin>
      <xmax>{x2}</xmax><ymax>{y2}</ymax>
    </bndbox>
  </object>
</annotation>"""
    print(f"  Pascal VOC XML (excerpt):\n{pascal_voc_xml[:200]}...")

    # ── RLE (Run-Length Encoding) for binary masks ───────────────────────────
    def encode_rle(binary_mask):
        """
        Encode a binary mask as COCO-style RLE.
        Returns counts of alternating 0-runs and 1-runs starting from 0.
        Mask is read column-major (Fortran order) — COCO standard.
        """
        flat = binary_mask.flatten(order="F").astype(np.uint8)
        counts = []
        current = 0
        count = 0
        for val in flat:
            if val == current:
                count += 1
            else:
                counts.append(count)
                count = 1
                current = val
        counts.append(count)
        if flat[0] == 1:          # COCO always starts with 0-run
            counts.insert(0, 0)
        return {"counts": counts, "size": list(binary_mask.shape)}

    def decode_rle(rle):
        """Decode COCO RLE back to binary mask."""
        h, w = rle["size"]
        flat = np.zeros(h * w, dtype=np.uint8)
        pos = 0
        current = 0
        for count in rle["counts"]:
            flat[pos:pos + count] = current
            pos += count
            current = 1 - current
        return flat.reshape((h, w), order="F")

    # Test RLE on a simple mask
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 2:7] = 1                           # 5x5 square of ones
    rle  = encode_rle(mask)
    back = decode_rle(rle)
    assert np.array_equal(mask, back), "RLE round-trip failed!"
    print(f"  RLE encoded 5x5 mask: counts={rle['counts'][:10]}...")
    print(f"  RLE decode round-trip: OK (arrays identical)")
    print("  Done: A6 annotation formats")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CV Section A — Digital Image Fundamentals")
    print("=" * 60)

    np.random.seed(42)

    section_A1()   # image representation, channels, dtype
    section_A2()   # colour spaces
    section_A3()   # sampling & interpolation
    section_A4()   # quantisation & dithering
    section_A5()   # noise models
    section_A6()   # annotation formats & RLE

    print("\n✓ All Section A demos complete.")
    print("  Output images saved to current directory.")
