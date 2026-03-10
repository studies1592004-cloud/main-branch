"""
cv_L_geometry.py
=================
Computer Vision Course — Section L: Keypoints, Pose, Geometry & Stereo

Topics covered:
  L1 - Keypoint representation: (x,y,v) format, heatmap encoding, OKS metric
  L2 - OpenPose: Part Affinity Fields (PAF), limb association, Hungarian matching
  L3 - HRNet: parallel multi-resolution branches, top-down pipeline
  L4 - Face/hand landmarks: 68-point face model, NME metric, MediaPipe overview
  L5 - 3D pose estimation: 2D→3D ambiguity, MPJPE metric, lifting networks
  L6 - Homography: 8 DoF, DLT via SVD, RANSAC robust estimation
  L7 - Epipolar geometry: fundamental matrix F, essential matrix E,
       8-point algorithm, epipolar line
  L8 - Stereo depth: disparity d=f*b/Z, rectification, monocular depth overview

Dependencies: numpy, opencv-python, matplotlib, scipy
Install:  pip install numpy opencv-python matplotlib scipy
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(filename, dpi=100):
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"  Saved: {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# L1 — KEYPOINT REPRESENTATION & OKS
# ═════════════════════════════════════════════════════════════════════════════

def section_L1():
    print("\n── L1: Keypoint Representation & OKS ──")

    """
    Keypoint format: (x, y, v)
      x, y: pixel coordinates
      v = 0: not labelled
      v = 1: labelled but occluded  (not visible, but annotated)
      v = 2: labelled and visible

    COCO 17-keypoint skeleton:
      0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
      5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
      9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
      13:left_knee  14:right_knee  15:left_ankle  16:right_ankle
    """

    COCO_KP_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # Simulated ground truth keypoints (x, y, v)
    gt_kps = np.array([
        [320, 80,  2],  # nose
        [310, 70,  2],  # left_eye
        [330, 70,  2],  # right_eye
        [300, 75,  1],  # left_ear (occluded)
        [340, 75,  2],  # right_ear
        [280, 130, 2],  # left_shoulder
        [360, 130, 2],  # right_shoulder
        [260, 180, 2],  # left_elbow
        [380, 175, 2],  # right_elbow
        [245, 230, 2],  # left_wrist
        [390, 225, 1],  # right_wrist (occluded)
        [290, 240, 2],  # left_hip
        [350, 240, 2],  # right_hip
        [285, 310, 2],  # left_knee
        [355, 308, 2],  # right_knee
        [280, 380, 2],  # left_ankle
        [360, 378, 2],  # right_ankle
    ], dtype=float)

    # Simulated prediction (slight noise)
    pred_kps = gt_kps.copy()
    pred_kps[:, :2] += np.random.randn(17, 2) * 5.0   # ±5px noise

    # --- OKS (Object Keypoint Similarity) ---
    """
    OKS measures similarity between predicted and GT keypoints, analogous to IoU for boxes.

    OKS = sum_i [exp(-d_i^2 / (2*s^2*k_i^2)) * delta(v_i>0)] / sum_i [delta(v_i>0)]

    where:
      d_i   = Euclidean distance between predicted and GT keypoint i
      s     = object scale (sqrt of bounding box area)
      k_i   = per-keypoint constant (larger k → more tolerance)
              COCO constants: eyes/ears 0.025, shoulders/hips 0.107, etc.
      v_i   = visibility flag (only evaluate where v_i > 0)

    OKS ∈ [0,1]. OKS=1: perfect prediction.
    mAP in COCO pose = mean AP over OKS thresholds [0.5:0.05:0.95]
    """

    # COCO per-keypoint sigmas (k = 2*sigma in OKS formula)
    COCO_SIGMAS = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035,  # nose, eyes, ears
        0.079, 0.079, 0.072, 0.072,          # shoulders, elbows
        0.062, 0.062, 0.107, 0.107,          # wrists, hips
        0.087, 0.087, 0.089, 0.089           # knees, ankles
    ])

    def compute_oks(pred, gt, sigmas, bbox_area):
        """
        Compute OKS between predicted and GT keypoints.
        pred, gt: (17, 3) arrays of (x, y, v)
        sigmas:   (17,) per-keypoint sigma values
        bbox_area: area of the person bounding box
        """
        s = np.sqrt(bbox_area)
        visible = gt[:, 2] > 0    # evaluate only visible/labelled keypoints

        dx = pred[:, 0] - gt[:, 0]
        dy = pred[:, 1] - gt[:, 1]
        d2 = dx**2 + dy**2

        # Exponent: -d^2 / (2 * s^2 * k^2),  k = 2*sigma
        k2 = (2 * sigmas) ** 2
        exp_term = np.exp(-d2 / (2 * s**2 * k2))

        oks = exp_term[visible].sum() / visible.sum()
        return oks

    # Bounding box area: rough estimate from GT keypoints
    x_min, x_max = gt_kps[gt_kps[:,2]>0, 0].min(), gt_kps[gt_kps[:,2]>0, 0].max()
    y_min, y_max = gt_kps[gt_kps[:,2]>0, 1].min(), gt_kps[gt_kps[:,2]>0, 1].max()
    bbox_area = (x_max - x_min) * (y_max - y_min)

    oks = compute_oks(pred_kps, gt_kps, COCO_SIGMAS, bbox_area)
    print(f"  OKS (±5px noise): {oks:.4f}  (1.0=perfect, ≥0.5 counts as correct)")

    # OKS vs noise level
    print(f"\n  OKS vs prediction noise level:")
    print(f"  {'Noise (px)':>10} {'OKS':>8}")
    for noise in [0, 2, 5, 10, 20, 50]:
        p = gt_kps.copy()
        p[:, :2] += np.random.randn(17, 2) * noise
        o = compute_oks(p, gt_kps, COCO_SIGMAS, bbox_area)
        print(f"  {noise:>10} {o:>8.4f}")

    # --- Heatmap encoding ---
    """
    Keypoints are encoded as Gaussian heatmaps for training CNNs:
      H[y, x] = exp(-((x-kx)^2 + (y-ky)^2) / (2*sigma^2))
    One heatmap per keypoint. Network output: (17, H/stride, W/stride).
    Prediction: argmax of heatmap → keypoint location.
    Soft-argmax (differentiable): weighted average of all positions.
    """

    def make_heatmap(kx, ky, H=64, W=64, sigma=2.0):
        """Generate a single Gaussian heatmap centred at (kx, ky)."""
        yy, xx = np.mgrid[:H, :W]
        heatmap = np.exp(-((xx - kx)**2 + (yy - ky)**2) / (2 * sigma**2))
        return heatmap

    # Make heatmaps for nose and left shoulder
    nose_hm    = make_heatmap(32, 8,  sigma=2.0)
    lshoulder_hm = make_heatmap(14, 26, sigma=2.0)

    # Decode: argmax
    def decode_heatmap(hm):
        idx = np.unravel_index(hm.argmax(), hm.shape)
        return idx[1], idx[0]   # x, y

    nx, ny = decode_heatmap(nose_hm)
    print(f"\n  Heatmap encode/decode: nose at (32,8) → decoded ({nx},{ny})")

    # Soft-argmax (differentiable alternative)
    def soft_argmax(hm):
        H, W = hm.shape
        yy, xx = np.mgrid[:H, :W]
        prob = hm / (hm.sum() + 1e-8)
        x = (prob * xx).sum()
        y = (prob * yy).sum()
        return x, y

    sx, sy = soft_argmax(nose_hm)
    print(f"  Soft-argmax: nose → ({sx:.2f},{sy:.2f})  (sub-pixel accurate)")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    # Draw skeleton
    canvas = np.zeros((450, 640, 3), dtype=np.uint8)
    SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for i, (x, y, v) in enumerate(gt_kps):
        if v > 0:
            color = (0, 255, 0) if v == 2 else (0, 165, 255)
            cv2.circle(canvas, (int(x), int(y)), 5, color, -1)
    for i, j in SKELETON:
        if gt_kps[i,2]>0 and gt_kps[j,2]>0:
            cv2.line(canvas, (int(gt_kps[i,0]),int(gt_kps[i,1])),
                     (int(gt_kps[j,0]),int(gt_kps[j,1])), (255,200,0), 2)
    axes[0].imshow(canvas)
    axes[0].set_title("Skeleton (green=visible, orange=occluded)")
    axes[0].axis("off")
    axes[1].imshow(nose_hm, cmap="hot")
    axes[1].set_title("Nose heatmap σ=2")
    axes[1].axis("off")
    axes[2].imshow(lshoulder_hm, cmap="hot")
    axes[2].set_title("L.shoulder heatmap")
    axes[2].axis("off")
    save_fig("L1_keypoints.png")
    print("  Done: L1 keypoint representation & OKS")


# ═════════════════════════════════════════════════════════════════════════════
# L2 — PART AFFINITY FIELDS (OPENPOSE)
# ═════════════════════════════════════════════════════════════════════════════

def section_L2():
    print("\n── L2: Part Affinity Fields (OpenPose) ──")

    """
    OpenPose (Cao et al. 2017) — bottom-up multi-person pose estimation.

    Two branches (both run on the full image):
      1. Confidence maps:   K heatmaps, one per keypoint type
      2. Part Affinity Fields (PAFs): 2×L vector fields, one per limb

    PAF for a limb connecting keypoints A→B:
      For each pixel p near the line segment AB:
        PAF(p) = unit vector from A to B  (direction of limb)
      For pixels far from AB:
        PAF(p) = (0, 0)

    Limb association:
      For candidate body part pair (di, ej):
        Integral along the line: ∫ PAF(p(u)) · e_AB du
        where p(u) = (1-u)*di + u*ej, u ∈ [0,1]
        High integral → di and ej belong to the same limb.

    Hungarian matching then assembles body parts into full skeletons.
    Complexity: O(1) with respect to number of people (bottom-up).
    """

    def make_paf(kp_a, kp_b, H=64, W=64, sigma=1.5):
        """
        Generate a Part Affinity Field (2 channels: ux, uy) for limb A→B.
        """
        ax, ay = kp_a
        bx, by = kp_b
        paf_x = np.zeros((H, W), dtype=np.float32)
        paf_y = np.zeros((H, W), dtype=np.float32)

        # Unit vector along limb
        limb_len = np.sqrt((bx-ax)**2 + (by-ay)**2) + 1e-8
        ux = (bx - ax) / limb_len
        uy = (by - ay) / limb_len

        # Perpendicular unit vector
        perp_x, perp_y = -uy, ux

        yy, xx = np.mgrid[:H, :W]
        # Projection along limb direction (0 to limb_len = on limb)
        proj_along = (xx - ax) * ux + (yy - ay) * uy
        # Perpendicular distance from limb line
        proj_perp  = np.abs((xx - ax) * perp_x + (yy - ay) * perp_y)

        # Pixels on the limb segment within sigma perpendicular width
        on_limb = (proj_along >= 0) & (proj_along <= limb_len) & (proj_perp <= sigma)
        paf_x[on_limb] = ux
        paf_y[on_limb] = uy

        return paf_x, paf_y

    def line_integral_paf(paf_x, paf_y, kp_a, kp_b, n_samples=10):
        """
        Compute line integral of PAF along the line segment A→B.
        High value → A and B are connected by this limb.
        """
        ax, ay = kp_a
        bx, by = kp_b
        limb_x = (bx - ax) / (np.sqrt((bx-ax)**2 + (by-ay)**2) + 1e-8)
        limb_y = (by - ay) / (np.sqrt((bx-ax)**2 + (by-ay)**2) + 1e-8)

        score = 0.0
        for u in np.linspace(0, 1, n_samples):
            px = int(ax + u * (bx - ax))
            py = int(ay + u * (by - ay))
            if 0 <= py < paf_y.shape[0] and 0 <= px < paf_x.shape[1]:
                score += paf_x[py, px] * limb_x + paf_y[py, px] * limb_y
        return score / n_samples

    # Simulate PAF for left upper arm (shoulder → elbow)
    shoulder = (14, 26)
    elbow    = (10, 40)
    paf_x, paf_y = make_paf(shoulder, elbow, H=64, W=64, sigma=1.5)

    # Test association: correct pair vs wrong pair
    score_correct = line_integral_paf(paf_x, paf_y, shoulder, elbow)
    score_wrong   = line_integral_paf(paf_x, paf_y, shoulder, (50, 26))

    print(f"  PAF line integral scores (shoulder→elbow limb):")
    print(f"    Correct pair  (shoulder→elbow):  {score_correct:.4f}  (should be ~1.0)")
    print(f"    Wrong pair    (shoulder→other):  {score_wrong:.4f}   (should be ~0.0)")

    # Hungarian matching demo for body part association
    """
    For each limb type, build cost matrix C[i,j] = line integral between
    candidate part i (type A) and candidate part j (type B).
    Use Hungarian algorithm to find optimal bipartite matching.
    """

    # Simulate cost matrix for 3 left-shoulders and 3 left-elbows
    # (in a multi-person image with 2 people visible)
    cost_matrix = np.array([
        [0.95, 0.12, 0.08],   # shoulder 0 matches elbow 0 best
        [0.10, 0.92, 0.15],   # shoulder 1 matches elbow 1 best
        [0.05, 0.08, 0.07],   # shoulder 2 is ambiguous (false detection)
    ])

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)   # maximise
    print(f"\n  Hungarian matching for limb association:")
    print(f"  Cost matrix (3 shoulders × 3 elbows):")
    print(f"  {cost_matrix}")
    print(f"  Optimal assignment: shoulders {row_ind.tolist()} → elbows {col_ind.tolist()}")
    for r, c in zip(row_ind, col_ind):
        status = "MATCH" if cost_matrix[r, c] > 0.3 else "WEAK (likely false detection)"
        print(f"    shoulder {r} → elbow {c}  score={cost_matrix[r,c]:.2f}  {status}")

    # Visualise PAF
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.sqrt(paf_x**2 + paf_y**2), cmap="hot", origin="upper")
    axes[0].set_title("PAF magnitude (limb region)")
    axes[0].axis("off")
    # Quiver plot
    step = 4
    H, W = paf_x.shape
    ys = np.arange(0, H, step); xs = np.arange(0, W, step)
    XX, YY = np.meshgrid(xs, ys)
    axes[1].imshow(np.zeros((H, W)), cmap="gray", origin="upper")
    axes[1].quiver(XX, YY, paf_x[::step, ::step], -paf_y[::step, ::step],
                   color="red", scale=15)
    axes[1].set_title("PAF direction (quiver)")
    axes[1].axis("off")
    save_fig("L2_paf.png")
    print("  Done: L2 Part Affinity Fields")


# ═════════════════════════════════════════════════════════════════════════════
# L3 — HRNET & TOP-DOWN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def section_L3():
    print("\n── L3: HRNet & Top-Down Pose Pipeline ──")

    """
    HRNet (Sun et al. 2019) — High-Resolution Network:
      Maintains high-resolution representations throughout the network.
      Parallel branches at different resolutions exchange information via fusions.

      Branch resolutions (for input 256×192):
        Branch 1: 64×48   (stride 4)
        Branch 2: 32×24   (stride 8)
        Branch 3: 16×12   (stride 16)
        Branch 4:  8×6    (stride 32)

      Each stage adds a lower-resolution branch.
      Multi-scale fusion: repeated upsampling + downsampling between branches.
      Output: keypoint heatmaps at Branch 1 resolution (64×48).

    Top-down pipeline:
      Step 1: Detect all persons with an object detector (YOLO, Faster R-CNN)
      Step 2: Crop and resize each person to fixed input size (256×192)
      Step 3: Run pose network on each crop independently
      Step 4: Map keypoint predictions back to original image coordinates

    vs Bottom-up (OpenPose):
      + Better accuracy per person (cropped, normalised input)
      - Speed proportional to number of people
    """

    # Simulate HRNet-like architecture param counts
    def hrnet_branch_params(in_ch, out_ch, n_blocks=4):
        """Approximate params for one HRNet branch (BasicBlock × n_blocks)."""
        params_per_block = 2 * (3*3*out_ch*out_ch + out_ch)   # 2 conv-BN per block
        shortcut = in_ch * out_ch if in_ch != out_ch else 0
        return n_blocks * params_per_block + shortcut

    branches = [
        ("Branch 1 (64×48)",  32, 32, 4),
        ("Branch 2 (32×24)",  64, 64, 4),
        ("Branch 3 (16×12)", 128, 128, 4),
        ("Branch 4 ( 8×6)", 256, 256, 4),
    ]
    total_branch_params = 0
    for name, in_ch, out_ch, n_blocks in branches:
        p = hrnet_branch_params(in_ch, out_ch, n_blocks)
        total_branch_params += p
        print(f"  {name}: ~{p/1e6:.2f}M params")

    # Final heatmap head
    n_keypoints = 17
    head_params = 1*1*32*n_keypoints   # 1×1 conv to keypoint channels
    print(f"  Heatmap head (1×1 conv → 17 ch): {head_params:,} params")
    print(f"  Total (approximate): ~{(total_branch_params + head_params)/1e6:.1f}M params")
    print(f"  (Actual HRNet-W32: ~28.5M params)")

    # Top-down coordinate transform
    def person_crop_transform(kp_img, person_bbox, crop_size=(192, 256)):
        """
        Transform keypoints from original image coords to crop coords.
        person_bbox: (x1, y1, x2, y2) in image pixels
        crop_size: (W, H) of crop
        """
        x1, y1, x2, y2 = person_bbox
        bw, bh = x2-x1, y2-y1
        scale_x = crop_size[0] / bw
        scale_y = crop_size[1] / bh
        kp_crop = kp_img.copy()
        kp_crop[:, 0] = (kp_img[:, 0] - x1) * scale_x
        kp_crop[:, 1] = (kp_img[:, 1] - y1) * scale_y
        return kp_crop

    def crop_to_image_transform(kp_crop, person_bbox, crop_size=(192, 256)):
        """Inverse transform: crop coords → image coords."""
        x1, y1, x2, y2 = person_bbox
        bw, bh = x2-x1, y2-y1
        scale_x = bw / crop_size[0]
        scale_y = bh / crop_size[1]
        kp_img = kp_crop.copy()
        kp_img[:, 0] = kp_crop[:, 0] * scale_x + x1
        kp_img[:, 1] = kp_crop[:, 1] * scale_y + y1
        return kp_img

    # Test round-trip
    person_bbox = (100, 50, 300, 450)
    kps_img = np.array([[150., 80., 2.], [180., 70., 2.], [250., 200., 2.]])
    kps_crop = person_crop_transform(kps_img, person_bbox)
    kps_back = crop_to_image_transform(kps_crop, person_bbox)
    roundtrip_err = np.abs(kps_img[:,:2] - kps_back[:,:2]).max()
    print(f"\n  Coordinate transform round-trip error: {roundtrip_err:.6f} px (should be 0)")

    # Benchmark
    print(f"\n  Pose estimation benchmark (COCO val, single-scale):")
    print(f"  {'Method':<22} {'Type':<12} {'AP':>6} {'Speed':>12}")
    print(f"  {'-'*58}")
    methods = [
        ("OpenPose",       "Bottom-up",  56.6, "~0.5 fps (GPU)"),
        ("SimpleBaseline", "Top-down",   72.2, "~20 fps"),
        ("HRNet-W32",      "Top-down",   74.4, "~15 fps"),
        ("HRNet-W48",      "Top-down",   75.1, "~10 fps"),
        ("ViTPose-B",      "Top-down",   75.8, "~25 fps"),
    ]
    for name, typ, ap, speed in methods:
        print(f"  {name:<22} {typ:<12} {ap:>6.1f} {speed:>12}")
    print("  Done: L3 HRNet & top-down pipeline")


# ═════════════════════════════════════════════════════════════════════════════
# L4 — FACE & HAND LANDMARKS
# ═════════════════════════════════════════════════════════════════════════════

def section_L4():
    print("\n── L4: Face & Hand Landmarks ──")

    """
    Face alignment keypoints:
      68-point model (dlib / 3DDFA):
        Points 1-17:  jaw contour
        Points 18-22: left eyebrow
        Points 23-27: right eyebrow
        Points 28-31: nose bridge
        Points 32-36: nose tip
        Points 37-42: left eye
        Points 43-48: right eye
        Points 49-60: outer lip
        Points 61-68: inner lip

    MediaPipe Face Mesh: 468 landmarks (dense mesh covering full face)
    MediaPipe Hands:      21 landmarks per hand (4 per finger + 1 palm)
    """

    # Simulate 68-point face landmarks
    def make_face_landmarks(cx=200, cy=200, scale=80):
        """Generate approximate 68 face landmark positions."""
        kps = []
        # Jaw (17 points)
        for i, angle in enumerate(np.linspace(-70, 70, 17)):
            r = np.deg2rad(angle)
            x = cx + scale * 1.1 * np.sin(r)
            y = cy + scale * 0.7 * (1 - 0.3*np.cos(r))
            kps.append((x, y))
        # Eyebrows (10)
        for x in np.linspace(cx-0.6*scale, cx-0.15*scale, 5):
            kps.append((x, cy - 0.45*scale))
        for x in np.linspace(cx+0.15*scale, cx+0.6*scale, 5):
            kps.append((x, cy - 0.45*scale))
        # Nose (9)
        for y in np.linspace(cy-0.3*scale, cy+0.1*scale, 4):
            kps.append((cx, y))
        for x in np.linspace(cx-0.25*scale, cx+0.25*scale, 5):
            kps.append((x, cy + 0.12*scale))
        # Eyes (12)
        for x in np.linspace(cx-0.55*scale, cx-0.15*scale, 6):
            kps.append((x, cy - 0.15*scale))
        for x in np.linspace(cx+0.15*scale, cx+0.55*scale, 6):
            kps.append((x, cy - 0.15*scale))
        # Lips (20)
        for x in np.linspace(cx-0.4*scale, cx+0.4*scale, 10):
            kps.append((x, cy + 0.35*scale))
        for x in np.linspace(cx-0.3*scale, cx+0.3*scale, 10):
            kps.append((x, cy + 0.42*scale))
        return np.array(kps[:68])

    gt_face  = make_face_landmarks(cx=200, cy=200, scale=80)
    pred_face = gt_face + np.random.randn(68, 2) * 3.0   # 3px noise

    # --- NME (Normalised Mean Error) ---
    """
    NME = (1/N) * sum_i ||pred_i - gt_i||_2 / d

    d = normalisation distance (prevents scale ambiguity):
      Inter-ocular:  distance between outer eye corners (points 37, 46 in 1-indexed)
      Inter-pupil:   distance between eye centres
      Bounding box diagonal: robust to occlusion

    NME < 0.05 (5%): good alignment
    NME < 0.03 (3%): excellent alignment
    """

    def nme(pred, gt, norm_idx_a=36, norm_idx_b=45):
        """
        NME using inter-ocular distance for normalisation.
        norm_idx_a, norm_idx_b: indices of outer eye corners (0-indexed).
        """
        norm_dist = np.linalg.norm(gt[norm_idx_a] - gt[norm_idx_b])
        point_dists = np.linalg.norm(pred - gt, axis=1)   # (N,)
        return point_dists.mean() / (norm_dist + 1e-8)

    nme_val = nme(pred_face, gt_face)
    print(f"  NME (3px noise, 68 points): {nme_val:.4f}  ({nme_val*100:.2f}% of inter-ocular distance)")

    # NME vs noise
    print(f"\n  NME vs noise level:")
    inter_ocular = np.linalg.norm(gt_face[36] - gt_face[45])
    print(f"  Inter-ocular distance: {inter_ocular:.1f}px")
    for noise_px in [0, 1, 2, 5, 10]:
        p = gt_face + np.random.randn(68, 2) * noise_px
        n = nme(p, gt_face)
        print(f"    noise={noise_px:2d}px  NME={n:.4f} ({n*100:.1f}%)")

    # Hand landmarks (MediaPipe 21-point model)
    """
    MediaPipe Hand 21 landmarks:
      0: WRIST
      1-4:   THUMB (CMC, MCP, IP, TIP)
      5-8:   INDEX (MCP, PIP, DIP, TIP)
      9-12:  MIDDLE (MCP, PIP, DIP, TIP)
      13-16: RING (MCP, PIP, DIP, TIP)
      17-20: PINKY (MCP, PIP, DIP, TIP)
    """
    print(f"\n  MediaPipe landmark counts:")
    print(f"    Face Mesh:  468 landmarks (dense face mesh)")
    print(f"    Pose:        33 landmarks (full body, includes 3D z)")
    print(f"    Hands:       21 landmarks per hand (4 per finger + wrist)")
    print(f"    Holistic:   pose + 2×hands + face in a single model pass")

    # Visualise face landmarks
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt_face[:,0],   gt_face[:,1],   s=10, c="green",  label="GT",  alpha=0.8)
    ax.scatter(pred_face[:,0], pred_face[:,1], s=10, c="red",    label="Pred", alpha=0.8)
    for i in range(68):
        ax.plot([gt_face[i,0], pred_face[i,0]],
                [gt_face[i,1], pred_face[i,1]], 'b-', lw=0.5, alpha=0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title(f"68-pt face landmarks  NME={nme_val:.3f}")
    ax.axis("off")
    save_fig("L4_face_landmarks.png")
    print("  Done: L4 face & hand landmarks")


# ═════════════════════════════════════════════════════════════════════════════
# L5 — 3D POSE ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def section_L5():
    print("\n── L5: 3D Pose Estimation ──")

    """
    2D → 3D ambiguity (depth ambiguity):
      Multiple 3D poses project to the same 2D image.
      A point at (X, Y, Z) projects to (fX/Z, fY/Z).
      Scaling (X,Y,Z) by any constant c gives the same 2D projection.
      → Additional constraints needed: temporal consistency, limb length priors.

    Lifting network (VideoPose3D, Martinez et al. 2017):
      Input:  2D pose sequence [(x1,y1), ..., (x17,y17)] per frame
      Output: 3D pose [(X1,Y1,Z1), ..., (X17,Y17,Z17)] in camera coordinates
      Architecture: temporal conv (TCN) or transformer over time window
    """

    # --- MPJPE (Mean Per Joint Position Error) ---
    """
    MPJPE = (1/N) * sum_i ||pred_3d_i - gt_3d_i||_2

    Units: millimetres (mm). Lower is better.
    State-of-art on Human3.6M: ~40mm (2023).

    Variants:
      PA-MPJPE: MPJPE after Procrustes alignment (removes global rotation/scale)
      N-MPJPE:  MPJPE after normalising bone lengths (removes subject shape)
    """

    def mpjpe(pred_3d, gt_3d):
        """Mean Per Joint Position Error in mm."""
        return np.linalg.norm(pred_3d - gt_3d, axis=-1).mean()

    def pa_mpjpe(pred_3d, gt_3d):
        """
        Procrustes-aligned MPJPE.
        Find optimal rotation R, scale s, translation t such that
        ||s*R*pred + t - gt||^2 is minimised, then compute MPJPE.
        """
        mu_p = pred_3d.mean(0); mu_g = gt_3d.mean(0)
        p = pred_3d - mu_p;     g = gt_3d - mu_g

        # Scale
        ss_p = (p**2).sum(); ss_g = (g**2).sum()
        scale = np.sqrt(ss_g / (ss_p + 1e-8))

        # Rotation via SVD
        M = g.T @ p
        U, S, Vt = np.linalg.svd(M)
        # Fix reflection
        d = np.linalg.det(U @ Vt)
        D = np.diag([1, 1, d])
        R = U @ D @ Vt

        pred_aligned = scale * (p @ R.T) + mu_g
        return np.linalg.norm(pred_aligned - gt_3d, axis=-1).mean()

    # Synthetic 3D pose (17 joints, 3D coords in mm)
    gt_3d = np.array([
        [0, 100, 0],    # nose
        [-20, 110, 5],  # left eye
        [20, 110, 5],   # right eye
        [-40, 105, 3],  # left ear
        [40, 105, 3],   # right ear
        [-120, 0, 0],   # left shoulder
        [120, 0, 0],    # right shoulder
        [-160, -100, 10],  # left elbow
        [160, -100, 10],   # right elbow
        [-170, -200, 20],  # left wrist
        [170, -200, 20],   # right wrist
        [-80, -280, 0],    # left hip
        [80, -280, 0],     # right hip
        [-85, -460, 5],    # left knee
        [85, -460, 5],     # right knee
        [-88, -640, 10],   # left ankle
        [88, -640, 10],    # right ankle
    ], dtype=float)

    # Add 3D prediction noise
    pred_3d_noisy = gt_3d + np.random.randn(17, 3) * 40.0   # 40mm noise

    # Add rotated prediction (to test PA-MPJPE)
    angle = np.deg2rad(15)
    R_small = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle),  np.cos(angle), 0],
                        [0, 0, 1]])
    pred_3d_rotated = (gt_3d @ R_small.T) + np.array([10, 5, 20])

    print(f"  3D Pose Evaluation (17 joints):")
    for name, pred in [("Noisy (40mm σ)", pred_3d_noisy),
                       ("Rotated+shifted", pred_3d_rotated)]:
        m  = mpjpe(pred, gt_3d)
        pa = pa_mpjpe(pred, gt_3d)
        print(f"    {name:<20}: MPJPE={m:.1f}mm  PA-MPJPE={pa:.1f}mm")

    # Project 3D → 2D (perspective projection)
    def project_3d_to_2d(kps_3d, f=1000, cx=320, cy=240):
        """Perspective projection: (X,Y,Z) → (f*X/Z + cx, f*Y/Z + cy)"""
        Z = kps_3d[:, 2] + 1500   # add depth offset so Z > 0
        x = f * kps_3d[:, 0] / Z + cx
        y = f * kps_3d[:, 1] / Z + cy
        return np.stack([x, y], axis=1)

    kps_2d = project_3d_to_2d(gt_3d)
    print(f"\n  2D projection (f=1000, depth offset 1500mm):")
    print(f"    Nose 3D: {gt_3d[0]}  → 2D: {kps_2d[0].round(1)}")

    print(f"\n  3D pose estimation benchmarks (Human3.6M):")
    print(f"  {'Method':<25} {'MPJPE':>8} {'PA-MPJPE':>10}")
    print(f"  {'-'*47}")
    benchmarks = [
        ("Martinez 2017",     51.8, 42.5),
        ("VideoPose3D",       46.8, 36.5),
        ("MotionBERT",        39.3, 30.2),
        ("D3DP",              38.4, 29.7),
    ]
    for name, m, pa in benchmarks:
        print(f"  {name:<25} {m:>8.1f} {pa:>10.1f}")
    print("  Done: L5 3D pose estimation")


# ═════════════════════════════════════════════════════════════════════════════
# L6 — HOMOGRAPHY: DLT + RANSAC
# ═════════════════════════════════════════════════════════════════════════════

def section_L6():
    print("\n── L6: Homography — DLT & RANSAC ──")

    """
    Homography H is a 3×3 matrix (8 DoF: 9 elements, scale=1) that maps
    points between two planar views:
      x' = H*x  (in homogeneous coordinates)
      [x', y', w']^T = H * [x, y, 1]^T
      final: (x'/w', y'/w')

    Applications: image stitching, planar AR, bird's-eye view transform.

    Direct Linear Transform (DLT):
      Each point correspondence (x↔x') gives 2 equations.
      4 correspondences → 8 equations → solve 8-DoF system.
      Stack into matrix A (2n×9), solve: Ah=0 via SVD.
      h = last column of V (in A = U Σ V^T).
    """

    def dlt_homography(pts_src, pts_dst):
        """
        Compute homography via DLT.
        pts_src, pts_dst: (N, 2) arrays, N >= 4.
        Returns H (3×3), normalised so H[2,2]=1.
        """
        N = pts_src.shape[0]
        A = []
        for i in range(N):
            x, y   = pts_src[i]
            xp, yp = pts_dst[i]
            # Two equations per correspondence
            A.append([-x, -y, -1,  0,  0,  0, x*xp, y*xp, xp])
            A.append([ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp])
        A = np.array(A)
        # SVD: solution is last row of V^T (smallest singular value)
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1]
        H = h.reshape(3, 3)
        return H / H[2, 2]

    def apply_homography(H, pts):
        """Apply homography H to points (N,2). Returns (N,2)."""
        N = pts.shape[0]
        pts_h = np.hstack([pts, np.ones((N, 1))])   # (N,3) homogeneous
        pts_t = (H @ pts_h.T).T                      # (N,3)
        return pts_t[:, :2] / pts_t[:, 2:3]         # dehomogenise

    # Test DLT: 4 point correspondences (planar scene under rotation)
    pts_src = np.array([[100., 100.], [300., 100.], [300., 250.], [100., 250.]])
    # Apply known homography (45° rotation + some shear)
    H_true = np.array([[0.9, -0.2, 30.],
                        [0.15, 0.85, 20.],
                        [0.0005, 0.0002, 1.]])
    pts_dst = apply_homography(H_true, pts_src)

    H_est = dlt_homography(pts_src, pts_dst)
    # Verify: reproject source points
    pts_reproj = apply_homography(H_est, pts_src)
    reproj_err = np.linalg.norm(pts_reproj - pts_dst, axis=1).mean()
    print(f"  DLT homography (4 exact points):")
    print(f"    Reprojection error: {reproj_err:.6f} px  (should be ~0)")
    print(f"    H_true[0]: {H_true[0].round(4)}")
    print(f"    H_est[0]:  {H_est[0].round(4)}")

    # --- RANSAC for robust homography estimation ---
    def ransac_homography(pts_src, pts_dst, n_iter=500, threshold=3.0):
        """
        RANSAC homography estimation.
        Each iteration: sample 4 correspondences, fit H, count inliers.
        Returns H with most inliers.

        Inlier criterion: reprojection error < threshold.
        """
        N = pts_src.shape[0]
        best_H       = None
        best_inliers = []

        for _ in range(n_iter):
            # Randomly sample 4 correspondences
            idx    = np.random.choice(N, 4, replace=False)
            sample_src = pts_src[idx]
            sample_dst = pts_dst[idx]

            try:
                H_cand = dlt_homography(sample_src, sample_dst)
            except np.linalg.LinAlgError:
                continue

            # Count inliers
            reproj  = apply_homography(H_cand, pts_src)
            errors  = np.linalg.norm(reproj - pts_dst, axis=1)
            inliers = np.where(errors < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_H       = H_cand

        # Refit on all inliers
        if best_H is not None and len(best_inliers) >= 4:
            best_H = dlt_homography(pts_src[best_inliers], pts_dst[best_inliers])

        return best_H, best_inliers

    # Add 8 clean correspondences + 4 outliers (25% outlier rate)
    N_pts = 12
    pts_s = np.random.rand(N_pts, 2) * 300 + 50
    pts_d = apply_homography(H_true, pts_s)
    pts_d[:4] += np.random.randn(4, 2) * 50    # outliers: 4 random noise points

    H_ransac, inliers = ransac_homography(pts_s, pts_d, n_iter=500, threshold=3.0)
    reproj_r = apply_homography(H_ransac, pts_s[inliers])
    err_ransac = np.linalg.norm(reproj_r - pts_d[inliers], axis=1).mean()

    print(f"\n  RANSAC homography ({N_pts} pts, 4 outliers = {4/N_pts*100:.0f}%):")
    print(f"    Inliers found: {len(inliers)}/{N_pts}  "
          f"(expected: {N_pts-4})")
    print(f"    Inlier reprojection error: {err_ransac:.4f} px")

    # Degrees of freedom
    print(f"\n  Homography degrees of freedom:")
    print(f"    H is 3×3 = 9 elements, but scale is irrelevant → 8 DoF")
    print(f"    Need minimum 4 point correspondences (2 equations each = 8)")
    print(f"    Compare: Translation=2DoF, Similarity=4DoF, Affine=6DoF, Homography=8DoF")
    print("  Done: L6 Homography DLT + RANSAC")


# ═════════════════════════════════════════════════════════════════════════════
# L7 — EPIPOLAR GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════

def section_L7():
    print("\n── L7: Epipolar Geometry ──")

    """
    Fundamental matrix F (3×3, rank 2, 7 DoF):
      Encodes the epipolar constraint for uncalibrated cameras:
        x'^T * F * x = 0  for all corresponding point pairs (x, x')
      Epipolar line in image 2 for point x in image 1: l' = F * x
      Epipolar line in image 1 for point x' in image 2: l = F^T * x'

    Essential matrix E (for calibrated cameras, 5 DoF):
      E = K'^T * F * K  where K, K' are intrinsic matrices
      Encodes relative rotation R and translation t:
        E = [t]_× * R  (t cross-product matrix times R)
      Decompose E → 4 possible (R,t) solutions → test with cheirality.

    8-point algorithm (Hartley):
      1. Normalise points (centre and scale for numerical stability)
      2. Build matrix A from point correspondences (8×9)
      3. SVD of A → F = last column of V reshaped to 3×3
      4. Enforce rank-2 constraint: SVD of F, set smallest σ to 0
      5. Denormalise: F = T'^T * F_norm * T
    """

    def normalise_points(pts):
        """
        Isotropic normalisation: translate centroid to origin, scale so
        mean distance from origin = sqrt(2).
        Returns normalised points and 3×3 normalisation matrix T.
        """
        mu = pts.mean(0)
        centred = pts - mu
        mean_dist = np.linalg.norm(centred, axis=1).mean()
        scale = np.sqrt(2) / (mean_dist + 1e-8)
        T = np.array([[scale, 0, -scale*mu[0]],
                      [0, scale, -scale*mu[1]],
                      [0,     0,           1]])
        pts_n = (T @ np.hstack([pts, np.ones((len(pts),1))]).T).T[:,:2]
        return pts_n, T

    def eight_point_algorithm(pts1, pts2):
        """
        Normalised 8-point algorithm for fundamental matrix estimation.
        """
        # Normalise
        p1n, T1 = normalise_points(pts1)
        p2n, T2 = normalise_points(pts2)

        # Build matrix A (N×9)
        N = pts1.shape[0]
        A = np.zeros((N, 9))
        for i in range(N):
            x, y   = p1n[i]
            xp, yp = p2n[i]
            A[i] = [xp*x, xp*y, xp, yp*x, yp*y, yp, x, y, 1]

        # SVD → F_norm
        _, _, Vt = np.linalg.svd(A)
        F_norm = Vt[-1].reshape(3, 3)

        # Enforce rank-2 (set smallest singular value to 0)
        U, S, Vt2 = np.linalg.svd(F_norm)
        S[2] = 0
        F_norm = U @ np.diag(S) @ Vt2

        # Denormalise
        F = T2.T @ F_norm @ T1
        return F / (F[2,2] + 1e-12)

    def epipolar_line(F, pt):
        """Epipolar line l = F*x in image 2 for point x in image 1."""
        x = np.array([pt[0], pt[1], 1.0])
        l = F @ x
        return l / (np.sqrt(l[0]**2 + l[1]**2) + 1e-8)

    def sampson_distance(F, pts1, pts2):
        """
        Sampson distance: first-order approximation of reprojection error.
        d_S = (x'^T F x)^2 / (Fx)_1^2 + (Fx)_2^2 + (F^T x')_1^2 + (F^T x')_2^2
        Should be ~0 for correct correspondences.
        """
        N = pts1.shape[0]
        p1h = np.hstack([pts1, np.ones((N,1))])
        p2h = np.hstack([pts2, np.ones((N,1))])
        Fx  = (F @ p1h.T).T    # (N,3)
        FTx = (F.T @ p2h.T).T  # (N,3)
        num = np.sum(p2h * Fx, axis=1)**2
        den = Fx[:,0]**2 + Fx[:,1]**2 + FTx[:,0]**2 + FTx[:,1]**2
        return num / (den + 1e-8)

    # Simulate stereo camera setup
    np.random.seed(7)
    N = 12
    # 3D points
    pts3d = np.random.randn(N, 3) * 2 + np.array([0, 0, 5])

    # Camera 1: identity pose
    K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])

    # Camera 2: translated 1 unit right + slight rotation
    angle = np.deg2rad(5)
    R2 = np.array([[np.cos(angle), 0, np.sin(angle)],
                   [0,             1, 0            ],
                   [-np.sin(angle),0, np.cos(angle)]])
    t2 = np.array([[-1.0], [0.0], [0.0]])
    P2 = K @ np.hstack([R2, t2])

    # Project to image planes
    def project(P, pts3d):
        pts_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])
        proj  = (P @ pts_h.T).T
        return proj[:, :2] / proj[:, 2:3]

    pts1 = project(P1, pts3d)
    pts2 = project(P2, pts3d)

    # Estimate F
    F_est = eight_point_algorithm(pts1, pts2)

    # Evaluate: Sampson distances should be small
    sd = sampson_distance(F_est, pts1, pts2)
    print(f"  8-point algorithm (N={N} correspondences):")
    print(f"    Sampson distance: mean={sd.mean():.4f}  max={sd.max():.4f}  (should be ~0)")
    print(f"    F matrix rank: {np.linalg.matrix_rank(F_est)} (should be 2)")
    print(f"    F[0]: {F_est[0].round(6)}")

    # Epipolar line
    test_pt = pts1[0]
    line = epipolar_line(F_est, test_pt)
    # Distance of corresponding point from epipolar line
    pt2_h = np.array([pts2[0,0], pts2[0,1], 1.0])
    dist  = abs(line @ pt2_h)
    print(f"\n  Epipolar line test:")
    print(f"    Point in image 1: {test_pt.round(2)}")
    print(f"    Epipolar line in image 2: {line.round(4)}  (ax+by+c=0)")
    print(f"    Corresponding point distance from line: {dist:.4f} px  (should be ~0)")

    # Essential matrix
    """
    E = K'^T * F * K
    SVD(E) → 2 possible R, 2 possible ±t → 4 solutions.
    Only one has all 3D points in front of both cameras (cheirality check).
    """
    E_est = K.T @ F_est @ K
    U, S, Vt = np.linalg.svd(E_est)
    print(f"\n  Essential matrix singular values: {S.round(4)}")
    print(f"  (Should be σ1≈σ2, σ3≈0 for valid E)")
    print("  Done: L7 epipolar geometry")


# ═════════════════════════════════════════════════════════════════════════════
# L8 — STEREO DEPTH & MONOCULAR DEPTH
# ═════════════════════════════════════════════════════════════════════════════

def section_L8():
    print("\n── L8: Stereo Depth & Monocular Depth ──")

    """
    Stereo depth formula:
      Z = f * b / d

    where:
      Z = depth (distance to camera in same units as b)
      f = focal length in pixels
      b = baseline (distance between camera centres, same units as Z)
      d = disparity = x_left - x_right  (pixel difference)

    Precision: ΔZ / Z ≈ (Z / (f*b)) * Δd
      → depth precision degrades quadratically with distance
      → wide baseline b and long focal length f improve precision
    """

    def stereo_depth(disparity, f, b):
        """Z = f * b / d  (disparity > 0)."""
        return f * b / (disparity + 1e-8)

    def stereo_disparity(Z, f, b):
        """d = f * b / Z"""
        return f * b / (Z + 1e-8)

    f = 800.0     # focal length (pixels)
    b = 0.12      # baseline (metres)

    print(f"  Stereo depth Z = f*b/d  (f={f}px, b={b}m):")
    print(f"  {'Disparity (px)':>16} {'Depth (m)':>12} {'Δdepth (1px err)':>18}")
    for d in [5, 10, 20, 50, 100, 200]:
        Z      = stereo_depth(d, f, b)
        Z_plus = stereo_depth(d+1, f, b)
        delta_Z = abs(Z - Z_plus)
        print(f"  {d:>16}  {Z:>12.3f}  {delta_Z:>18.4f}")

    # --- Stereo rectification overview ---
    """
    Stereo rectification transforms both images so that:
      - Epipolar lines become horizontal
      - Corresponding points lie on the same row
      - Disparity search becomes 1D (left→right scan only)

    After rectification: use block matching (BM) or semi-global BM (SGBM)
    to find disparity at each pixel.

    cv2.StereoSGBM_create() — semi-global block matching
      numDisparities: max disparity range (multiple of 16)
      blockSize:      matching window size (odd, 3–11)
    """

    # Simulate disparity map
    H, W = 64, 64
    np.random.seed(3)
    true_depth = np.zeros((H, W), dtype=np.float32)
    true_depth[10:40, 10:55] = 2.0   # near object at 2m
    true_depth[40:60, 20:60] = 5.0   # far region at 5m
    true_depth[true_depth == 0] = 10.0  # background at 10m

    true_disp = stereo_disparity(true_depth, f, b)
    noisy_disp = true_disp + np.random.normal(0, 0.3, true_disp.shape)
    noisy_disp = np.clip(noisy_disp, 0.5, None)
    noisy_depth = stereo_depth(noisy_disp, f, b)

    depth_err = np.abs(noisy_depth - true_depth).mean()
    print(f"\n  Simulated stereo depth (f={f}px, b={b}m):")
    print(f"    Near object (2m): disparity = {stereo_disparity(2.0,f,b):.1f}px")
    print(f"    Far object  (5m): disparity = {stereo_disparity(5.0,f,b):.1f}px")
    print(f"    Background (10m): disparity = {stereo_disparity(10.0,f,b):.1f}px")
    print(f"    Mean depth error (0.3px disp noise): {depth_err:.4f}m")

    # --- Monocular depth estimation overview ---
    """
    Monocular depth — single image to dense depth map.
    Ill-posed: no stereo baseline. Must learn scene priors.

    Key methods:
      MiDaS (Ranftl 2020):   relative depth, scale-invariant training.
                              Trained on 10 datasets with diverse content.
      DPT (Ranftl 2021):     Vision Transformer backbone for global context.
      Depth Anything (2024): Large-scale training (62M images), strong zero-shot.
      ZoeDepth (2023):       Metric depth (absolute values in metres).

    Metrics:
      AbsRel = (1/N) * sum |Z_pred - Z_gt| / Z_gt
      δ < 1.25: fraction of pixels where max(Z_pred/Z_gt, Z_gt/Z_pred) < 1.25
    """

    # Simulate monocular depth quality
    def abs_rel(pred, gt):
        return np.mean(np.abs(pred - gt) / (gt + 1e-8))

    def delta_threshold(pred, gt, threshold=1.25):
        ratio = np.maximum(pred / (gt + 1e-8), gt / (pred + 1e-8))
        return (ratio < threshold).mean()

    gt_depth = true_depth.copy()
    for name, noise in [("Good (low noise)", 0.1), ("Medium noise", 0.5), ("Bad", 2.0)]:
        pred = gt_depth * (1 + np.random.randn(*gt_depth.shape) * noise)
        pred = np.clip(pred, 0.1, 20)
        ar = abs_rel(pred, gt_depth)
        d1 = delta_threshold(pred, gt_depth, 1.25)
        print(f"  {name:<22}: AbsRel={ar:.4f}  δ<1.25={d1:.3f}")

    # SfM/SLAM note
    print(f"\n  Depth estimation methods summary:")
    print(f"  {'Method':<22} {'Type':<18} {'Scale':>8} {'Speed':>12}")
    print(f"  {'-'*65}")
    methods = [
        ("Stereo BM/SGBM",  "Passive stereo",    "Metric",    "Real-time"),
        ("Active depth",    "Structured light",  "Metric",    "Real-time"),
        ("LiDAR",           "Active ToF",        "Metric",    "Real-time"),
        ("SfM (COLMAP)",    "Multi-view",        "Up-to-scale","Minutes"),
        ("ORB-SLAM3",       "SLAM (monocular)",  "Up-to-scale","Real-time"),
        ("MiDaS",           "Monocular DNN",     "Relative",  "~30 fps"),
        ("Depth Anything",  "Monocular DNN",     "Relative",  "~25 fps"),
        ("ZoeDepth",        "Monocular DNN",     "Metric",    "~10 fps"),
    ]
    for name, typ, scale, speed in methods:
        print(f"  {name:<22} {typ:<18} {scale:>8} {speed:>12}")

    # Visualise depth map
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(true_depth, cmap="plasma")
    axes[0].set_title("Ground truth depth")
    axes[0].axis("off")
    axes[1].imshow(true_disp, cmap="viridis")
    axes[1].set_title("Disparity (d = f*b/Z)")
    axes[1].axis("off")
    axes[2].imshow(noisy_depth, cmap="plasma")
    axes[2].set_title("Depth from noisy disparity")
    axes[2].axis("off")
    save_fig("L8_stereo_depth.png")
    print("  Done: L8 stereo depth")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CV Section L — Keypoints, Pose, Geometry & Stereo")
    print("=" * 60)
    np.random.seed(42)

    section_L1()   # keypoint format, OKS, heatmap encode/decode, soft-argmax
    section_L2()   # Part Affinity Fields, line integral, Hungarian matching
    section_L3()   # HRNet architecture, top-down pipeline, coord transform
    section_L4()   # 68-pt face landmarks, NME metric, MediaPipe overview
    section_L5()   # MPJPE, PA-MPJPE, Procrustes alignment, 3D projection
    section_L6()   # DLT homography from scratch, RANSAC, 8 DoF
    section_L7()   # fundamental matrix, 8-point algorithm, Sampson distance
    section_L8()   # Z=fb/d, disparity map, monocular depth methods

    print("\n✓ All Section L demos complete.")
    print("  Output images: L1_keypoints.png  L2_paf.png")
    print("                 L4_face_landmarks.png  L8_stereo_depth.png")
