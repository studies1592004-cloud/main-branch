"""
cv_D_classical_detection.py
============================
Computer Vision Course — Section D: Classical Object Detection

Topics covered:
  D1 - Template matching: SSD, NCC, ZNCC
  D2 - Sliding window + image pyramid + NMS (IoU-based)
  D3 - HOG feature descriptor (from scratch)
  D4 - HOG + Linear SVM classifier pipeline
  D5 - Harris corner detector (from scratch)
  D6 - SIFT keypoints (cv2) + ORB keypoints + feature matching

Dependencies: numpy, opencv-python, matplotlib, scikit-learn, scikit-image
Install:  pip install numpy opencv-python matplotlib scikit-learn scikit-image
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_scene(h=256, w=256):
    """Synthetic grayscale scene with several shapes."""
    img = np.full((h, w), 180, dtype=np.uint8)
    cv2.rectangle(img, (30,  40), (80,  90),  60, -1)
    cv2.rectangle(img, (150, 60), (210, 120), 80, -1)
    cv2.circle(img, (180, 180), 30, 40, -1)
    cv2.rectangle(img, (60, 160), (130, 220), 100, -1)
    img = cv2.GaussianBlur(img, (3,3), 0)
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def save_grid(imgs, titles, filename):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  Saved: {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# D1 — TEMPLATE MATCHING
# ═════════════════════════════════════════════════════════════════════════════

def section_D1():
    print("\n── D1: Template Matching ──")

    scene    = make_scene()
    # Extract a template from the scene (the first rectangle)
    template = scene[42:88, 32:78].copy()   # 46×46 patch

    TH, TW = template.shape
    SH, SW = scene.shape

    # --- Sum of Squared Differences (SSD) ---
    def match_ssd(scene, template):
        TH, TW = template.shape
        SH, SW = scene.shape
        out = np.zeros((SH - TH + 1, SW - TW + 1), dtype=np.float64)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                diff = scene[i:i+TH, j:j+TW].astype(float) - template.astype(float)
                out[i, j] = np.sum(diff ** 2)
        return out

    # --- Normalised Cross-Correlation (NCC) ---
    def match_ncc(scene, template):
        """NCC in range [-1,+1]. +1 = perfect match."""
        TH, TW = template.shape
        SH, SW = scene.shape
        t = template.astype(float) - template.mean()
        t_norm = np.sqrt((t**2).sum()) + 1e-8
        out = np.zeros((SH-TH+1, SW-TW+1))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                patch = scene[i:i+TH, j:j+TW].astype(float)
                p = patch - patch.mean()
                p_norm = np.sqrt((p**2).sum()) + 1e-8
                out[i,j] = np.sum(p*t) / (p_norm * t_norm)
        return out

    # Use cv2 for speed (SSD and NCC)
    ssd_map = cv2.matchTemplate(scene, template, cv2.TM_SQDIFF)
    ncc_map = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)

    # Best match locations
    _, _, ssd_loc, _ = cv2.minMaxLoc(ssd_map)   # min for SSD
    _, _, _, ncc_loc = cv2.minMaxLoc(ncc_map)    # max for NCC

    print(f"  Template size: {template.shape}")
    print(f"  SSD best match location (top-left): {ssd_loc}")
    print(f"  NCC best match location (top-left): {ncc_loc}")
    print(f"  NCC peak value: {ncc_map.max():.4f} (1.0 = perfect)")

    # Draw matches
    scene_ssd = cv2.cvtColor(scene.copy(), cv2.COLOR_GRAY2BGR)
    scene_ncc = cv2.cvtColor(scene.copy(), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(scene_ssd, ssd_loc, (ssd_loc[0]+TW, ssd_loc[1]+TH), (0,255,0), 2)
    cv2.rectangle(scene_ncc, ncc_loc, (ncc_loc[0]+TW, ncc_loc[1]+TH), (0,0,255), 2)

    save_grid(
        [scene, template, ncc_map, scene_ncc],
        ["Scene", "Template", "NCC response map", "NCC match (red box)"],
        "D1_template_matching.png"
    )
    print("  Done: D1 template matching")


# ═════════════════════════════════════════════════════════════════════════════
# D2 — SLIDING WINDOW + IMAGE PYRAMID + NMS
# ═════════════════════════════════════════════════════════════════════════════

def section_D2():
    print("\n── D2: Sliding Window + Image Pyramid + NMS ──")

    # --- IoU (Intersection over Union) ---
    def iou(box1, box2):
        """
        box format: (x1, y1, x2, y2)
        Returns IoU in [0,1].
        """
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter   = inter_w * inter_h
        area1   = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2   = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union   = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    # --- Non-Maximum Suppression ---
    def nms(boxes, scores, iou_threshold=0.5):
        """
        Standard NMS.
        boxes:  list of (x1,y1,x2,y2)
        scores: list of confidence scores
        Returns indices of kept boxes.
        """
        if len(boxes) == 0:
            return []
        order  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        kept   = []
        while order:
            i = order.pop(0)
            kept.append(i)
            order = [j for j in order if iou(boxes[i], boxes[j]) < iou_threshold]
        return kept

    # Test NMS
    boxes  = [(10,10,60,60), (15,15,65,65), (12,12,58,58),   # cluster 1
              (100,100,160,160), (105,105,165,165)]            # cluster 2
    scores = [0.95, 0.80, 0.60,   0.90, 0.70]

    kept = nms(boxes, scores, iou_threshold=0.5)
    print(f"  NMS: {len(boxes)} boxes → {len(kept)} after NMS (kept indices {kept})")

    # Verify IoU formula
    box_a = (0, 0, 100, 100)
    box_b = (50, 50, 150, 150)
    print(f"  IoU of overlapping 100x100 boxes (50% overlap): {iou(box_a, box_b):.4f}")
    print(f"  IoU of identical boxes: {iou(box_a, box_a):.4f}")
    print(f"  IoU of non-overlapping boxes: {iou((0,0,10,10),(20,20,30,30)):.4f}")

    # --- Image pyramid ---
    def build_pyramid(img, scale=0.75, min_size=32):
        """Yield image at progressively smaller scales."""
        current = img.copy()
        while min(current.shape[:2]) >= min_size:
            yield current
            h = int(current.shape[0] * scale)
            w = int(current.shape[1] * scale)
            if h < min_size or w < min_size:
                break
            current = cv2.resize(current, (w, h), interpolation=cv2.INTER_AREA)

    scene = make_scene()
    pyramid_levels = list(build_pyramid(scene))
    print(f"  Pyramid levels: {[p.shape for p in pyramid_levels]}")

    # --- Sliding window demo (counts windows) ---
    def count_sliding_windows(img_shape, win_size=(32,32), stride=8):
        H, W = img_shape[:2]
        wh, ww = win_size
        count = 0
        for y in range(0, H - wh + 1, stride):
            for x in range(0, W - ww + 1, stride):
                count += 1
        return count

    total_windows = sum(count_sliding_windows(p.shape) for p in pyramid_levels)
    print(f"  Total sliding windows across pyramid (32x32, stride=8): {total_windows}")

    # --- Soft-NMS ---
    def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.01):
        """
        Soft-NMS: instead of removing boxes, decay their scores.
        score_new = score * exp(-IoU^2 / sigma)
        """
        boxes  = list(boxes)
        scores = list(scores)
        kept   = []
        while scores:
            i = int(np.argmax(scores))
            kept.append(boxes[i])
            best_box  = boxes.pop(i)
            best_score= scores.pop(i)
            new_scores = []
            new_boxes  = []
            for j, (b, s) in enumerate(zip(boxes, scores)):
                iou_val = iou(best_box, b)
                s_new   = s * np.exp(-iou_val**2 / sigma)
                if s_new >= score_threshold:
                    new_scores.append(s_new)
                    new_boxes.append(b)
            boxes  = new_boxes
            scores = new_scores
        return kept

    soft_kept = soft_nms(boxes, scores)
    print(f"  Soft-NMS: {len(soft_kept)} boxes kept")
    print("  Done: D2 sliding window + NMS")


# ═════════════════════════════════════════════════════════════════════════════
# D3 — HOG FEATURE DESCRIPTOR (from scratch)
# ═════════════════════════════════════════════════════════════════════════════

def section_D3():
    print("\n── D3: HOG Feature Descriptor ──")

    def compute_hog(img, cell_size=8, block_size=2, n_bins=9):
        """
        HOG (Histogram of Oriented Gradients) from scratch.
        Args:
            img:        grayscale uint8
            cell_size:  pixels per cell (square)
            block_size: cells per block side (square)
            n_bins:     orientation bins in [0,180)
        Returns:
            feature vector (1D numpy array)
        """
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        H, W = img.shape

        # Step 1: Compute gradients
        Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        angle     = np.arctan2(np.abs(Gy), np.abs(Gx)) * 180 / np.pi  # [0,180)

        # Step 2: Cell histograms (soft bin assignment)
        n_cells_y = H // cell_size
        n_cells_x = W // cell_size
        cell_hists = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)

        bin_width = 180.0 / n_bins
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                y0, y1 = cy * cell_size, (cy+1) * cell_size
                x0, x1 = cx * cell_size, (cx+1) * cell_size
                cell_mag = magnitude[y0:y1, x0:x1].flatten()
                cell_ang = angle[y0:y1, x0:x1].flatten()
                # Soft voting: split magnitude between two nearest bins
                bin_f  = cell_ang / bin_width
                bin_lo = np.floor(bin_f).astype(int) % n_bins
                bin_hi = (bin_lo + 1) % n_bins
                frac   = bin_f - np.floor(bin_f)
                np.add.at(cell_hists[cy, cx], bin_lo, cell_mag * (1 - frac))
                np.add.at(cell_hists[cy, cx], bin_hi, cell_mag * frac)

        # Step 3: Block normalisation (L2-Hys)
        blocks_y = n_cells_y - block_size + 1
        blocks_x = n_cells_x - block_size + 1
        hog_features = []

        for by in range(blocks_y):
            for bx in range(blocks_x):
                block = cell_hists[by:by+block_size, bx:bx+block_size, :].flatten()
                # L2 normalisation
                norm = np.sqrt(np.sum(block**2) + 1e-6)
                block = block / norm
                # Clamp (Hys): limit to 0.2
                block = np.clip(block, 0, 0.2)
                # Renormalise
                norm2 = np.sqrt(np.sum(block**2) + 1e-6)
                block = block / norm2
                hog_features.append(block)

        return np.concatenate(hog_features)

    # Test on a 64x64 patch
    scene  = make_scene()
    patch  = scene[:64, :64]
    feat   = compute_hog(patch, cell_size=8, block_size=2, n_bins=9)
    print(f"  HOG feature vector length: {len(feat)}")
    print(f"  HOG feature range: [{feat.min():.4f}, {feat.max():.4f}]")

    # Compare with skimage HOG
    try:
        from skimage.feature import hog as sk_hog
        feat_sk, vis = sk_hog(patch, orientations=9, pixels_per_cell=(8,8),
                              cells_per_block=(2,2), visualize=True)
        print(f"  skimage HOG length: {len(feat_sk)}")
        print(f"  Correlation with skimage: {np.corrcoef(feat, feat_sk)[0,1]:.4f}")
    except ImportError:
        print("  skimage not available for comparison")

    # Visualise HOG
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(patch, cmap="gray")
    axes[0].set_title("Patch")
    axes[0].axis("off")
    axes[1].bar(range(len(feat[:36])), feat[:36])
    axes[1].set_title("HOG feature (first 36 dims)")
    plt.tight_layout()
    plt.savefig("D3_hog.png", dpi=100)
    plt.close()
    print("  Saved: D3_hog.png")
    print("  Done: D3 HOG descriptor")


# ═════════════════════════════════════════════════════════════════════════════
# D4 — HOG + LINEAR SVM
# ═════════════════════════════════════════════════════════════════════════════

def section_D4():
    print("\n── D4: HOG + Linear SVM ──")

    from skimage.feature import hog as sk_hog

    WIN_SIZE = (64, 64)

    def extract_hog(img_gray):
        """Extract HOG feature from a grayscale window."""
        resized = cv2.resize(img_gray, WIN_SIZE)
        feat = sk_hog(resized, orientations=9, pixels_per_cell=(8,8),
                      cells_per_block=(2,2), block_norm="L2-Hys")
        return feat

    # --- Generate synthetic training data ---
    np.random.seed(42)
    n_pos, n_neg = 200, 400

    X, y = [], []
    for _ in range(n_pos):
        # Positive: patch with a vertical edge (simulate object)
        patch = np.random.randint(60, 120, WIN_SIZE, dtype=np.uint8)
        patch[:, WIN_SIZE[0]//2:] += np.random.randint(50, 100)
        patch = np.clip(patch, 0, 255).astype(np.uint8)
        X.append(extract_hog(patch))
        y.append(1)

    for _ in range(n_neg):
        # Negative: random texture (no clear edge)
        patch = np.random.randint(50, 200, WIN_SIZE, dtype=np.uint8)
        X.append(extract_hog(patch))
        y.append(0)

    X = np.array(X)
    y = np.array(y)
    print(f"  Training data: {X.shape[0]} samples, {X.shape[1]}-dim features")

    # --- Train LinearSVC ---
    from sklearn.model_selection import cross_val_score
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    LinearSVC(C=0.01, max_iter=2000))
    ])
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  LinearSVC 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)

    # --- Sliding window detection on test scene ---
    scene = make_scene(256, 256)
    stride = 16
    detections = []
    det_scores  = []

    for y0 in range(0, scene.shape[0] - WIN_SIZE[1] + 1, stride):
        for x0 in range(0, scene.shape[1] - WIN_SIZE[0] + 1, stride):
            window = scene[y0:y0+WIN_SIZE[1], x0:x0+WIN_SIZE[0]]
            feat   = extract_hog(window).reshape(1, -1)
            score  = clf.decision_function(feat)[0]
            if score > 0.5:
                detections.append((x0, y0, x0+WIN_SIZE[0], y0+WIN_SIZE[1]))
                det_scores.append(score)

    print(f"  Raw detections (score>0.5): {len(detections)}")

    # Apply NMS
    def nms(boxes, scores, iou_threshold=0.5):
        if not boxes: return []
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        kept  = []
        while order:
            i = order.pop(0)
            kept.append(i)
            order = [j for j in order
                     if _iou(boxes[i], boxes[j]) < iou_threshold]
        return kept

    def _iou(b1, b2):
        xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
        xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
        inter=max(0,xi2-xi1)*max(0,yi2-yi1)
        a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        union=a1+a2-inter
        return inter/union if union>0 else 0

    kept = nms(detections, det_scores)
    print(f"  After NMS: {len(kept)} detections")
    print("  Done: D4 HOG + LinearSVM")


# ═════════════════════════════════════════════════════════════════════════════
# D5 — HARRIS CORNER DETECTOR (from scratch)
# ═════════════════════════════════════════════════════════════════════════════

def section_D5():
    print("\n── D5: Harris Corner Detector ──")

    scene = make_scene()

    def harris_corners(img, k=0.04, sigma=1.0, threshold=0.01):
        """
        Harris corner detector from scratch.
        1. Compute image gradients Ix, Iy
        2. Compute structure tensor M = [[Ix^2, IxIy],[IxIy, Iy^2]]
        3. Smooth M components with Gaussian
        4. Compute response R = det(M) - k * trace(M)^2
        5. Threshold and non-maximum suppression
        """
        # Step 1: Gradients
        img_f = img.astype(np.float32)
        Ix = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

        # Step 2 & 3: Smoothed structure tensor components
        ksize = int(6*sigma+1) | 1
        Ixx = cv2.GaussianBlur(Ix*Ix, (ksize,ksize), sigma)
        Ixy = cv2.GaussianBlur(Ix*Iy, (ksize,ksize), sigma)
        Iyy = cv2.GaussianBlur(Iy*Iy, (ksize,ksize), sigma)

        # Step 4: Harris response R = det(M) - k*trace(M)^2
        det_M   = Ixx * Iyy - Ixy * Ixy
        trace_M = Ixx + Iyy
        R = det_M - k * trace_M ** 2

        # Step 5: Threshold
        R_norm = R / (R.max() + 1e-8)
        corners = np.zeros_like(R_norm)
        corners[R_norm > threshold] = R_norm[R_norm > threshold]

        # Non-maximum suppression (local max in 5x5 window)
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(corners, size=5)
        corners_nms = ((corners == local_max) & (corners > 0)).astype(np.float32)
        corner_pts  = np.argwhere(corners_nms > 0)[:, ::-1]  # (x,y) format

        return R_norm, corners_nms, corner_pts

    R, corners, pts = harris_corners(scene)

    # cv2 Harris for comparison
    harris_cv2 = cv2.cornerHarris(scene.astype(np.float32), blockSize=2, ksize=3, k=0.04)
    harris_cv2 = cv2.dilate(harris_cv2, None)
    n_cv2 = (harris_cv2 > 0.01 * harris_cv2.max()).sum()

    print(f"  Harris corners found (scratch): {len(pts)}")
    print(f"  Harris corners found (cv2):     {n_cv2}")
    print(f"  R response range: [{R.min():.4f}, {R.max():.4f}]")
    print(f"  R >> 0 → corner. R << 0 → edge. R ≈ 0 → flat region.")

    # Visualise
    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    for (x, y) in pts[:50]:
        cv2.circle(scene_color, (int(x), int(y)), 3, (0, 0, 255), -1)

    save_grid(
        [scene, R, corners, scene_color],
        ["Scene", "Harris R response", "Corner map (NMS)", "Detected corners"],
        "D5_harris.png"
    )
    print("  Done: D5 Harris corner detector")


# ═════════════════════════════════════════════════════════════════════════════
# D6 — SIFT & ORB FEATURE MATCHING
# ═════════════════════════════════════════════════════════════════════════════

def section_D6():
    print("\n── D6: SIFT & ORB Feature Matching ──")

    scene = make_scene()

    # Simulate a second view: slight rotation + translation
    M = cv2.getRotationMatrix2D((128, 128), angle=10, scale=0.95)
    M[0, 2] += 10    # translate x
    M[1, 2] += 5     # translate y
    scene2 = cv2.warpAffine(scene, M, (scene.shape[1], scene.shape[0]),
                             borderMode=cv2.BORDER_REFLECT)

    # --- SIFT ---
    try:
        sift = cv2.SIFT_create(nfeatures=500)
        kp1_sift, des1_sift = sift.detectAndCompute(scene,  None)
        kp2_sift, des2_sift = sift.detectAndCompute(scene2, None)

        # BFMatcher with L2 distance and ratio test
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches_raw = bf.knnMatch(des1_sift, des2_sift, k=2)
        # Lowe's ratio test: keep match if best distance < 0.75 * second best
        good_sift = [m for m, n in matches_raw if m.distance < 0.75 * n.distance]
        print(f"  SIFT keypoints: {len(kp1_sift)} | {len(kp2_sift)}")
        print(f"  SIFT good matches (ratio test 0.75): {len(good_sift)}")
        sift_available = True
    except cv2.error:
        print("  SIFT not available (requires opencv-contrib or non-free module)")
        sift_available = False

    # --- ORB (patent-free, fast) ---
    orb = cv2.ORB_create(nfeatures=500)
    kp1_orb, des1_orb = orb.detectAndCompute(scene,  None)
    kp2_orb, des2_orb = orb.detectAndCompute(scene2, None)

    # BFMatcher with Hamming distance (ORB uses binary descriptors)
    bf_ham = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_orb = bf_ham.match(des1_orb, des2_orb) if des1_orb is not None and des2_orb is not None else []
    matches_orb = sorted(matches_orb, key=lambda m: m.distance)
    good_orb = [m for m in matches_orb if m.distance < 64]   # threshold for binary descriptor

    print(f"  ORB keypoints: {len(kp1_orb)} | {len(kp2_orb)}")
    print(f"  ORB good matches (distance < 64): {len(good_orb)}")

    # SIFT: 128-D float descriptor; ORB: 32-byte (256-bit) binary descriptor
    if des1_orb is not None:
        print(f"  ORB descriptor: {des1_orb.shape[1]} bytes = {des1_orb.shape[1]*8} bits per keypoint")

    # --- Homography estimation from matches ---
    if len(good_orb) >= 4:
        src_pts = np.float32([kp1_orb[m.queryIdx].pt for m in good_orb]).reshape(-1,1,2)
        dst_pts = np.float32([kp2_orb[m.trainIdx].pt for m in good_orb]).reshape(-1,1,2)
        H_est, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = mask.ravel().sum() if mask is not None else 0
        print(f"  Homography RANSAC inliers: {inliers}/{len(good_orb)}")

    # Visualise ORB matches
    match_img = cv2.drawMatches(scene, kp1_orb, scene2, kp2_orb,
                                good_orb[:20], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    save_grid([match_img], ["ORB matches (top-20)"], "D6_feature_matching.png")

    print("  Done: D6 SIFT & ORB feature matching")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CV Section D — Classical Object Detection")
    print("=" * 60)
    np.random.seed(42)

    section_D1()   # template matching
    section_D2()   # sliding window, pyramid, IoU, NMS, Soft-NMS
    section_D3()   # HOG descriptor from scratch
    section_D4()   # HOG + LinearSVM pipeline
    section_D5()   # Harris corner detector from scratch
    section_D6()   # SIFT, ORB, Lowe's ratio test, homography RANSAC

    print("\n✓ All Section D demos complete.")
    print("  Output images saved to current directory.")
