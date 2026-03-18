"""
cv_E_optical_flow.py
=====================
Computer Vision Course — Section E: Optical Flow & Motion

Topics covered:
  E1 - Optical flow basics: brightness constancy constraint equation (OFCE)
  E2 - Lucas-Kanade (LK) optical flow from scratch
  E3 - Pyramidal Lucas-Kanade + KLT tracker
  E4 - Horn-Schunck global optical flow from scratch
  E5 - Dense optical flow with cv2.FarnebackOpticalFlow
  E6 - Flow visualisation: HSV colour wheel, quiver plot

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

def make_frame(h=128, w=128, t=0):
    """
    Synthetic moving scene:
    - Background gradient
    - A bright rectangle that translates with time
    - A circle that translates in a different direction
    """
    img = np.zeros((h, w), dtype=np.float32)
    # background gradient
    img += np.linspace(0.1, 0.3, w)[np.newaxis, :]

    # moving rectangle (dx=2px/frame, dy=1px/frame)
    rx0 = int(20 + t * 2) % w
    ry0 = int(30 + t * 1) % h
    rx1 = min(rx0 + 30, w)
    ry1 = min(ry0 + 20, h)
    img[ry0:ry1, rx0:rx1] = 0.9

    # moving circle (dx=-1px/frame, dy=2px/frame)
    cx = int(90 - t * 1) % w
    cy = int(40 + t * 2) % h
    yy, xx = np.ogrid[:h, :w]
    img[(xx - cx)**2 + (yy - cy)**2 < 15**2] = 0.6

    return np.clip(img, 0, 1)


def flow_to_hsv(flow):
    """
    Visualise optical flow as an HSV colour image.
    Hue = direction, Saturation = 1, Value = magnitude.
    """
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    angle  = np.arctan2(fy, fx)         # [-pi, pi]
    mag    = np.sqrt(fx**2 + fy**2)

    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)   # [0,179]
    sat = np.full((h, w), 255, dtype=np.uint8)
    val = np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255).astype(np.uint8)

    hsv = np.stack([hue, sat, val], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_grid(imgs, titles, filename):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1 if img.max() <= 1 else None)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  Saved: {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# E1 — BRIGHTNESS CONSTANCY CONSTRAINT
# ═════════════════════════════════════════════════════════════════════════════

def section_E1():
    print("\n── E1: Brightness Constancy Constraint ──")

    """
    OFCE (Optical Flow Constraint Equation):
        I(x,y,t) = I(x+u*dt, y+v*dt, t+dt)

    First-order Taylor expansion:
        I_x * u + I_y * v + I_t = 0

    Where:
        I_x = dI/dx  (spatial gradient in x)
        I_y = dI/dy  (spatial gradient in y)
        I_t = dI/dt  (temporal gradient)
        (u, v) = optical flow vector at (x,y)

    This is ONE equation in TWO unknowns (u,v) → aperture problem.
    The solution lies on a constraint LINE in (u,v) space.
    """

    I1 = make_frame(t=0)
    I2 = make_frame(t=1)

    # Compute spatial and temporal gradients
    I1_f = (I1 * 255).astype(np.float32)
    I2_f = (I2 * 255).astype(np.float32)

    Ix = cv2.Sobel((I1_f + I2_f) / 2, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel((I1_f + I2_f) / 2, cv2.CV_32F, 0, 1, ksize=3)
    It = I2_f - I1_f

    print(f"  Frame 1 shape: {I1.shape}, dtype: {I1.dtype}")
    print(f"  Ix range: [{Ix.min():.1f}, {Ix.max():.1f}]")
    print(f"  Iy range: [{Iy.min():.1f}, {Iy.max():.1f}]")
    print(f"  It range: [{It.min():.1f}, {It.max():.1f}]")

    # OFCE residual at a pixel (should be ~0 for correct flow)
    px, py = 35, 32          # pixel inside the moving rectangle
    u_gt, v_gt = 2.0, 1.0    # known ground truth flow
    ofce_residual = Ix[py,px]*u_gt + Iy[py,px]*v_gt + It[py,px]
    print(f"  OFCE residual at rect pixel (GT flow u=2,v=1): {ofce_residual:.2f}  (close to 0 = consistent)")

    # Aperture problem: show constraint line at one pixel
    ix_val = float(Ix[py, px])
    iy_val = float(Iy[py, px])
    it_val = float(It[py, px])
    # Constraint: ix*u + iy*v = -it  → line in (u,v) space
    u_range = np.linspace(-5, 5, 100)
    if abs(iy_val) > 0.01:
        v_line = (-it_val - ix_val * u_range) / iy_val
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(u_range, v_line, 'b-', label="Constraint line")
        ax.plot(u_gt, v_gt, 'r*', markersize=12, label="GT flow (2,1)")
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_xlabel("u (horizontal flow)")
        ax.set_ylabel("v (vertical flow)")
        ax.set_title("Aperture problem: flow lies on constraint line")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("E1_aperture_problem.png", dpi=100)
        plt.close()
        print("  Saved: E1_aperture_problem.png")

    print("  Done: E1 brightness constancy constraint")


# ═════════════════════════════════════════════════════════════════════════════
# E2 — LUCAS-KANADE OPTICAL FLOW (from scratch)
# ═════════════════════════════════════════════════════════════════════════════

def section_E2():
    print("\n── E2: Lucas-Kanade Optical Flow ──")

    def lucas_kanade(I1, I2, win_size=7):
        """
        Lucas-Kanade optical flow (dense, for every pixel).

        Assumption: flow (u,v) is constant within a window W of size win_size.
        For all pixels (xi,yi) in W:
            Ix_i * u + Iy_i * v = -It_i

        Stack as: A * [u,v]^T = b  where A is Nx2, b is Nx1
        Least-squares solution: [u,v] = (A^T A)^-1 A^T b

        A^T A = [[sum(Ix^2), sum(Ix*Iy)],
                 [sum(Ix*Iy), sum(Iy^2)]]   ← structure tensor M
        A^T b = [-sum(Ix*It), -sum(Iy*It)]

        Solution exists when M is invertible (det(M) > threshold).
        Well-conditioned when eigenvalues of M are both large
        (i.e., the pixel is a corner, not a flat region or edge).
        """
        I1f = I1.astype(np.float32)
        I2f = I2.astype(np.float32)
        avg = (I1f + I2f) / 2

        # Gradients
        Ix = cv2.Sobel(avg, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(avg, cv2.CV_32F, 0, 1, ksize=3)
        It = I2f - I1f

        H, W = I1.shape
        u = np.zeros((H, W), dtype=np.float32)
        v = np.zeros((H, W), dtype=np.float32)
        r = win_size // 2

        # Pre-compute products
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        Ixt = Ix * It
        Iyt = Iy * It

        # Sum over window using box filter (much faster than nested loops)
        ksize = (win_size, win_size)
        SIxx = cv2.boxFilter(Ixx, -1, ksize, normalize=False)
        SIxy = cv2.boxFilter(Ixy, -1, ksize, normalize=False)
        SIyy = cv2.boxFilter(Iyy, -1, ksize, normalize=False)
        SIxt = cv2.boxFilter(Ixt, -1, ksize, normalize=False)
        SIyt = cv2.boxFilter(Iyt, -1, ksize, normalize=False)

        # Solve 2x2 system at each pixel
        det = SIxx * SIyy - SIxy * SIxy
        valid = np.abs(det) > 1e-6

        u[valid] = (-SIyy[valid] * SIxt[valid] + SIxy[valid] * SIyt[valid]) / det[valid]
        v[valid] = ( SIxy[valid] * SIxt[valid] - SIxx[valid] * SIyt[valid]) / det[valid]

        return np.stack([u, v], axis=-1)

    I1 = (make_frame(t=0) * 255).astype(np.float32)
    I2 = (make_frame(t=1) * 255).astype(np.float32)

    flow_lk = lucas_kanade(I1, I2, win_size=7)

    # Evaluate at known moving region
    rect_u = flow_lk[32:52, 22:52, 0]
    rect_v = flow_lk[32:52, 22:52, 1]
    print(f"  LK flow at moving rectangle:")
    print(f"    Mean u: {rect_u.mean():.2f} (GT: 2.0)")
    print(f"    Mean v: {rect_v.mean():.2f} (GT: 1.0)")

    flow_vis = flow_to_hsv(flow_lk)
    save_grid(
        [I1/255, I2/255, flow_vis],
        ["Frame 1", "Frame 2", "LK flow (colour-coded)"],
        "E2_lucas_kanade.png"
    )
    print("  Done: E2 Lucas-Kanade optical flow")
    return flow_lk


# ═════════════════════════════════════════════════════════════════════════════
# E3 — PYRAMIDAL LK + KLT TRACKER
# ═════════════════════════════════════════════════════════════════════════════

def section_E3():
    print("\n── E3: Pyramidal LK + KLT Tracker ──")

    # --- Pyramidal LK (cv2 version) for sparse feature tracking ---
    I1 = (make_frame(t=0) * 255).astype(np.uint8)
    I2 = (make_frame(t=1) * 255).astype(np.uint8)

    # Detect good features to track (Shi-Tomasi corners)
    feature_params = dict(
        maxCorners    = 50,
        qualityLevel  = 0.01,
        minDistance   = 7,
        blockSize     = 7
    )
    pts1 = cv2.goodFeaturesToTrack(I1, mask=None, **feature_params)

    # LK optical flow parameters
    lk_params = dict(
        winSize    = (15, 15),
        maxLevel   = 3,           # pyramid levels
        criteria   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # Track features from frame 1 to frame 2
    pts2, status, error = cv2.calcOpticalFlowPyrLK(I1, I2, pts1, None, **lk_params)

    # Keep only successfully tracked points
    good1 = pts1[status.ravel() == 1]
    good2 = pts2[status.ravel() == 1]

    print(f"  Features detected: {len(pts1)}")
    print(f"  Features tracked:  {len(good1)} (status==1)")

    # Compute flow vectors
    flow_vecs = good2 - good1
    u_tracked = flow_vecs[:, 0, 0]
    v_tracked = flow_vecs[:, 0, 1]
    print(f"  Mean tracked flow: u={u_tracked.mean():.2f}, v={v_tracked.mean():.2f}")
    print(f"  Median tracked flow: u={np.median(u_tracked):.2f}, v={np.median(v_tracked):.2f}")

    # Visualise tracked points
    vis = cv2.cvtColor(I2, cv2.COLOR_GRAY2BGR)
    for (x1,y1), (x2,y2) in zip(good1.reshape(-1,2), good2.reshape(-1,2)):
        cv2.arrowedLine(vis,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0), 1, tipLength=0.3)
        cv2.circle(vis, (int(x1), int(y1)), 2, (0, 0, 255), -1)

    # --- KLT Tracker across multiple frames ---
    print(f"\n  KLT multi-frame tracking:")
    frames = [(make_frame(t=i) * 255).astype(np.uint8) for i in range(6)]
    tracks = {i: [pts1[i].flatten().tolist()] for i in range(len(pts1))}
    current_pts = pts1.copy()

    for frame_idx in range(1, len(frames)):
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            frames[frame_idx - 1], frames[frame_idx], current_pts, None, **lk_params
        )
        for i, (st, pt) in enumerate(zip(status.ravel(), next_pts)):
            if st == 1 and i < len(tracks):
                tracks[i].append(pt.flatten().tolist())
        current_pts = next_pts[status.ravel() == 1].reshape(-1, 1, 2)

    track_lengths = [len(v) for v in tracks.values()]
    print(f"    Avg track length over 6 frames: {np.mean(track_lengths):.1f}")

    save_grid([I1, vis], ["Frame 1", "KLT tracked (arrows=flow)"], "E3_klt_tracker.png")
    print("  Done: E3 Pyramidal LK + KLT tracker")


# ═════════════════════════════════════════════════════════════════════════════
# E4 — HORN-SCHUNCK GLOBAL OPTICAL FLOW (from scratch)
# ═════════════════════════════════════════════════════════════════════════════

def section_E4():
    print("\n── E4: Horn-Schunck Optical Flow ──")

    def horn_schunck(I1, I2, alpha=1.0, n_iter=100):
        """
        Horn-Schunck global optical flow.

        Minimises energy:
          E(u,v) = integral[ (Ix*u + Iy*v + It)^2 + alpha*(|grad u|^2 + |grad v|^2) ] dx dy

        Euler-Lagrange equations → iterative update (Gauss-Seidel):
          u^(k+1) = u_avg^k - Ix * (Ix*u_avg^k + Iy*v_avg^k + It) / (alpha^2 + Ix^2 + Iy^2)
          v^(k+1) = v_avg^k - Iy * (Ix*u_avg^k + Iy*v_avg^k + It) / (alpha^2 + Ix^2 + Iy^2)

        alpha: smoothness weight. Large alpha → smoother flow, less data fidelity.

        Averaging kernel for Laplacian (neighbourhood average):
          [1/12, 1/6, 1/12]
          [1/6,   0,  1/6 ]
          [1/12, 1/6, 1/12]
        """
        I1f = I1.astype(np.float64)
        I2f = I2.astype(np.float64)
        avg = (I1f + I2f) / 2.0

        # Gradients
        Ix = cv2.Sobel(avg, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(avg, cv2.CV_64F, 0, 1, ksize=3)
        It = I2f - I1f

        # Averaging kernel (neighbourhood average for Laplacian)
        avg_kernel = np.array([[1/12, 1/6, 1/12],
                                [1/6,   0,  1/6 ],
                                [1/12, 1/6, 1/12]], dtype=np.float64)

        u = np.zeros_like(I1f)
        v = np.zeros_like(I1f)

        denom = alpha**2 + Ix**2 + Iy**2 + 1e-8

        for iteration in range(n_iter):
            # Compute neighbourhood averages
            u_avg = convolve(u, avg_kernel)
            v_avg = convolve(v, avg_kernel)

            # Common numerator term
            num = Ix * u_avg + Iy * v_avg + It

            # Update
            u = u_avg - Ix * num / denom
            v = v_avg - Iy * num / denom

        return np.stack([u.astype(np.float32), v.astype(np.float32)], axis=-1)

    # Run on small image for speed
    I1 = (make_frame(64, 64, t=0) * 255).astype(np.float32)
    I2 = (make_frame(64, 64, t=1) * 255).astype(np.float32)

    flow_hs_weak   = horn_schunck(I1, I2, alpha=0.5,  n_iter=100)
    flow_hs_strong = horn_schunck(I1, I2, alpha=5.0,  n_iter=100)

    # Evaluate at moving rectangle pixels
    for name, flow in [("alpha=0.5", flow_hs_weak), ("alpha=5.0", flow_hs_strong)]:
        rect_u = flow[14:22, 10:30, 0]
        rect_v = flow[14:22, 10:30, 1]
        print(f"  HS {name}: mean flow at rect: u={rect_u.mean():.2f} v={rect_v.mean():.2f} (GT: 2,1)")

    # Endpoint error (EPE): average L2 distance between predicted and GT flow
    gt_u = np.zeros((64, 64), dtype=np.float32)
    gt_v = np.zeros((64, 64), dtype=np.float32)
    gt_u[14:22, 10:30] = 2.0   # rectangle moves right 2px
    gt_v[14:22, 10:30] = 1.0   # rectangle moves down 1px

    for name, flow in [("HS alpha=0.5", flow_hs_weak), ("HS alpha=5.0", flow_hs_strong)]:
        epe = np.sqrt((flow[...,0]-gt_u)**2 + (flow[...,1]-gt_v)**2).mean()
        print(f"  EPE {name}: {epe:.3f} px")

    vis_hs = flow_to_hsv(flow_hs_strong)
    save_grid(
        [I1/255, I2/255, vis_hs],
        ["Frame 1", "Frame 2", "Horn-Schunck flow"],
        "E4_horn_schunck.png"
    )
    print("  Done: E4 Horn-Schunck optical flow")


# ═════════════════════════════════════════════════════════════════════════════
# E5 — DENSE OPTICAL FLOW (Farneback) + VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def section_E5():
    print("\n── E5: Dense Optical Flow (Farneback) ──")

    I1 = (make_frame(128, 128, t=0) * 255).astype(np.uint8)
    I2 = (make_frame(128, 128, t=1) * 255).astype(np.uint8)

    # --- Farneback dense optical flow ---
    # Approximates neighbourhood by polynomial expansion,
    # then estimates flow from the displacement of these polynomials.
    flow_fb = cv2.calcOpticalFlowFarneback(
        I1, I2,
        flow     = None,
        pyr_scale= 0.5,    # pyramid scale
        levels   = 3,      # pyramid levels
        winsize  = 15,     # averaging window size
        iterations = 3,
        poly_n   = 5,      # neighbourhood size for polynomial
        poly_sigma = 1.2,  # Gaussian std for polynomial weights
        flags    = 0
    )

    print(f"  Farneback flow shape: {flow_fb.shape}")
    print(f"  Flow u range: [{flow_fb[...,0].min():.2f}, {flow_fb[...,0].max():.2f}]")
    print(f"  Flow v range: [{flow_fb[...,1].min():.2f}, {flow_fb[...,1].max():.2f}]")

    # Flow statistics
    magnitude = np.sqrt(flow_fb[...,0]**2 + flow_fb[...,1]**2)
    print(f"  Mean magnitude: {magnitude.mean():.3f} px")
    print(f"  Max  magnitude: {magnitude.max():.3f} px")

    # --- Quiver plot ---
    step = 8
    H, W = I1.shape
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    XX, YY = np.meshgrid(xs, ys)
    UU = flow_fb[::step, ::step, 0]
    VV = flow_fb[::step, ::step, 1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(I1, cmap="gray")
    axes[0].set_title("Frame 1")
    axes[0].axis("off")

    axes[1].imshow(I2, cmap="gray")
    axes[1].quiver(XX, YY, UU, VV, color="red", scale=50, scale_units="inches", width=0.004)
    axes[1].set_title("Dense flow (quiver)")
    axes[1].axis("off")

    flow_vis = flow_to_hsv(flow_fb)
    axes[2].imshow(cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Flow (HSV: hue=dir, val=mag)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("E5_dense_flow.png", dpi=100)
    plt.close()
    print("  Saved: E5_dense_flow.png")

    # --- Flow decomposition: global motion vs local motion ---
    # Global motion estimate: median of all flow vectors (robust)
    global_u = np.median(flow_fb[..., 0])
    global_v = np.median(flow_fb[..., 1])
    residual = flow_fb - np.array([global_u, global_v])
    residual_mag = np.sqrt(residual[...,0]**2 + residual[...,1]**2)

    print(f"  Global motion (median): u={global_u:.2f}, v={global_v:.2f}")
    print(f"  Residual (local) flow mean magnitude: {residual_mag.mean():.3f}")
    print("  Done: E5 dense optical flow")


# ═════════════════════════════════════════════════════════════════════════════
# E6 — FLOW METRICS & COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def section_E6():
    print("\n── E6: Flow Metrics & Comparison ──")

    I1 = (make_frame(64, 64, t=0) * 255).astype(np.float32)
    I2 = (make_frame(64, 64, t=1) * 255).astype(np.float32)

    # Ground truth flow
    gt_flow = np.zeros((64, 64, 2), dtype=np.float32)
    gt_flow[14:22, 10:30, 0] = 2.0   # rectangle: u=2
    gt_flow[14:22, 10:30, 1] = 1.0   # rectangle: v=1
    # circle roughly: u=-1, v=2
    cx, cy = 58, 8
    yy, xx = np.ogrid[:64, :64]
    circle_mask = (xx - cx)**2 + (yy - cy)**2 < 15**2
    gt_flow[circle_mask, 0] = -1.0
    gt_flow[circle_mask, 1] =  2.0

    # LK flow (use box-filter version)
    def lk_dense(I1, I2, win=7):
        avg = (I1 + I2) / 2
        Ix = cv2.Sobel(avg, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(avg, cv2.CV_32F, 0, 1, ksize=3)
        It = I2 - I1
        k = (win, win)
        SIxx = cv2.boxFilter(Ix*Ix, -1, k, normalize=False)
        SIxy = cv2.boxFilter(Ix*Iy, -1, k, normalize=False)
        SIyy = cv2.boxFilter(Iy*Iy, -1, k, normalize=False)
        SIxt = cv2.boxFilter(Ix*It, -1, k, normalize=False)
        SIyt = cv2.boxFilter(Iy*It, -1, k, normalize=False)
        det = SIxx*SIyy - SIxy**2
        valid = np.abs(det) > 1e-6
        u = np.zeros_like(I1); v = np.zeros_like(I1)
        u[valid] = (-SIyy[valid]*SIxt[valid] + SIxy[valid]*SIyt[valid]) / det[valid]
        v[valid] = ( SIxy[valid]*SIxt[valid] - SIxx[valid]*SIyt[valid]) / det[valid]
        return np.stack([u, v], axis=-1)

    flow_lk_dense = lk_dense(I1, I2)

    # Farneback
    flow_fb = cv2.calcOpticalFlowFarneback(
        I1.astype(np.uint8), I2.astype(np.uint8), None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Metrics
    def epe(pred, gt):
        """Average Endpoint Error (L2 distance per pixel)."""
        return np.sqrt(((pred - gt)**2).sum(axis=-1)).mean()

    def angular_error(pred, gt):
        """Average angular error in degrees."""
        dot   = (pred[...,0]*gt[...,0] + pred[...,1]*gt[...,1] + 1)
        mag_p = np.sqrt(pred[...,0]**2 + pred[...,1]**2 + 1)
        mag_g = np.sqrt(gt[...,0]**2   + gt[...,1]**2   + 1)
        cosine= np.clip(dot / (mag_p * mag_g), -1, 1)
        return np.degrees(np.arccos(cosine)).mean()

    print(f"  {'Method':<15} {'EPE (px)':>10} {'AE (deg)':>10}")
    print(f"  {'-'*37}")
    for name, flow in [("LK (scratch)", flow_lk_dense), ("Farneback", flow_fb)]:
        e = epe(flow, gt_flow)
        a = angular_error(flow, gt_flow)
        print(f"  {name:<15} {e:>10.3f} {a:>10.3f}")

    # Colour wheel legend
    h_wheel, w_wheel = 64, 64
    cx_w, cy_w = w_wheel // 2, h_wheel // 2
    yy, xx = np.mgrid[:h_wheel, :w_wheel]
    fx = (xx - cx_w).astype(np.float32)
    fy = (yy - cy_w).astype(np.float32)
    wheel_flow = np.stack([fx, fy], axis=-1)
    wheel_vis  = flow_to_hsv(wheel_flow)

    save_grid(
        [flow_to_hsv(gt_flow), flow_to_hsv(flow_lk_dense), flow_to_hsv(flow_fb), wheel_vis],
        ["GT flow", "LK flow", "Farneback flow", "Colour wheel"],
        "E6_flow_comparison.png"
    )
    print("  Done: E6 flow metrics & comparison")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CV Section E — Optical Flow & Motion")
    print("=" * 60)
    np.random.seed(42)

    section_E1()   # OFCE derivation, aperture problem
    section_E2()   # Lucas-Kanade from scratch (dense)
    section_E3()   # Pyramidal LK + KLT sparse tracker
    section_E4()   # Horn-Schunck from scratch
    section_E5()   # Farneback dense flow + quiver plot
    section_E6()   # EPE & angular error metrics

    print("\n✓ All Section E demos complete.")
    print("  Output images saved to current directory.")
