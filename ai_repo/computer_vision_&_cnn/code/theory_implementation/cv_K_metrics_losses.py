"""
cv_K_metrics_losses.py
=======================
Computer Vision Course — Section K: Evaluation Metrics & Loss Functions

Topics covered:
  K1 - Classification metrics: confusion matrix, P/R/F1, ROC-AUC, macro/micro/weighted
  K2 - Detection metrics: AP (area under P-R curve), mAP, COCO AP@[.5:.95]
  K3 - Segmentation metrics: mIoU, Dice, Panoptic Quality
  K4 - Image quality metrics: PSNR, SSIM, LPIPS, FID
  K5 - Classification losses: cross-entropy, label smoothing, knowledge distillation
  K6 - Regression losses: SmoothL1, IoU→GIoU→DIoU→CIoU hierarchy
  K7 - Distribution Focal Loss (DFL), Task-Aligned Learning (TAL) assignment
  K8 - Segmentation losses: binary CE + Dice combined, Tversky, total loss weighting

Dependencies: numpy, matplotlib, scikit-learn, scipy
Install:  pip install numpy matplotlib scikit-learn scipy
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                              average_precision_score, confusion_matrix)
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════════════
# K1 — CLASSIFICATION METRICS
# ═════════════════════════════════════════════════════════════════════════════

def section_K1():
    print("\n── K1: Classification Metrics ──")

    # --- Binary metrics ---
    y_true = np.array([1,1,1,1,1, 0,0,0,0,0, 1,1,1, 0,0])
    y_pred = np.array([1,1,1,0,0, 0,0,1,1,1, 1,0,0, 0,1])

    TP = ((y_pred==1) & (y_true==1)).sum()
    TN = ((y_pred==0) & (y_true==0)).sum()
    FP = ((y_pred==1) & (y_true==0)).sum()
    FN = ((y_pred==0) & (y_true==1)).sum()

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)       # = sensitivity = TPR
    f1        = 2 * precision * recall / (precision + recall)
    specificity = TN / (TN + FP)     # = TNR

    print(f"  Confusion matrix: TP={TP} TN={TN} FP={FP} FN={FN}")
    print(f"  Precision: {precision:.3f}  (of all predicted positive, fraction truly positive)")
    print(f"  Recall:    {recall:.3f}  (of all actual positive, fraction we found)")
    print(f"  F1:        {f1:.3f}  (harmonic mean of P and R)")
    print(f"  Specificity:{specificity:.3f}  (of all actual negative, fraction correctly rejected)")

    # Precision-Recall tradeoff
    print(f"\n  P/R tradeoff with threshold (simulated scores):")
    scores = np.array([0.9,0.8,0.75,0.6,0.55, 0.4,0.35,0.7,0.65,0.3, 0.85,0.5,0.45, 0.2,0.6])
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        y_p = (scores >= thresh).astype(int)
        tp = ((y_p==1)&(y_true==1)).sum()
        fp = ((y_p==1)&(y_true==0)).sum()
        fn = ((y_p==0)&(y_true==1)).sum()
        p  = tp/(tp+fp) if tp+fp>0 else 0
        r  = tp/(tp+fn) if tp+fn>0 else 0
        f  = 2*p*r/(p+r) if p+r>0 else 0
        print(f"  {thresh:>10.1f} {p:>10.3f} {r:>10.3f} {f:>8.3f}")

    # ROC-AUC
    auc = roc_auc_score(y_true, scores)
    print(f"\n  ROC-AUC: {auc:.4f}  (random=0.5, perfect=1.0)")

    # --- Multiclass: macro vs micro vs weighted ---
    np.random.seed(0)
    n_classes = 4
    y_true_mc = np.random.randint(0, n_classes, 100)
    y_pred_mc = y_true_mc.copy()
    flip = np.random.choice(100, 25, replace=False)
    y_pred_mc[flip] = np.random.randint(0, n_classes, 25)

    cm = confusion_matrix(y_true_mc, y_pred_mc)
    per_class_p = np.diag(cm) / (cm.sum(axis=0) + 1e-8)
    per_class_r = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    per_class_f1= 2*per_class_p*per_class_r / (per_class_p+per_class_r+1e-8)
    support     = cm.sum(axis=1)

    macro_f1    = per_class_f1.mean()
    micro_f1    = np.diag(cm).sum() / cm.sum()
    weighted_f1 = (per_class_f1 * support).sum() / support.sum()

    print(f"\n  Multiclass F1 averaging (4 classes, 100 samples):")
    for i in range(n_classes):
        print(f"    Class {i}: P={per_class_p[i]:.3f}  R={per_class_r[i]:.3f}  "
              f"F1={per_class_f1[i]:.3f}  support={support[i]}")
    print(f"  Macro   F1: {macro_f1:.3f}  (unweighted mean — treats all classes equally)")
    print(f"  Micro   F1: {micro_f1:.3f}  (global TP/FP/FN — dominated by frequent classes)")
    print(f"  Weighted F1:{weighted_f1:.3f}  (weighted by support — accounts for imbalance)")
    print("  Done: K1 classification metrics")


# ═════════════════════════════════════════════════════════════════════════════
# K2 — DETECTION METRICS: AP & mAP
# ═════════════════════════════════════════════════════════════════════════════

def section_K2():
    print("\n── K2: Detection Metrics — AP & mAP ──")

    """
    Average Precision (AP):
      1. Sort predictions by confidence (descending)
      2. Compute precision and recall at each threshold
      3. AP = area under the precision-recall curve
         (using interpolation: precision = max precision at recall >= r)
      4. A prediction is TP if IoU with a GT box >= threshold (usually 0.5)

    mAP = mean AP over all classes
    COCO mAP = mean AP over IoU thresholds [0.5, 0.55, ..., 0.95]
    """

    def compute_ap(recall, precision):
        """
        Compute AP using the 11-point interpolation (PASCAL VOC 2007) or
        all-points interpolation (VOC 2010+, COCO).
        We use all-points here.
        """
        # Append sentinel values
        mrec = np.concatenate([[0.], recall, [1.]])
        mpre = np.concatenate([[0.], precision, [0.]])
        # Make precision monotonically decreasing
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        # Find points where recall changes
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        # AP = sum of rectangular areas
        ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
        return ap

    # Simulate detection results for one class
    # 10 GT boxes, 15 detections sorted by confidence
    n_gt = 10
    np.random.seed(7)
    # matched[i] = True if detection i is a TP (matched to a GT box)
    matched = np.array([1,1,0,1,1,0,0,1,0,1, 0,1,0,0,1], dtype=bool)
    confidences = np.sort(np.random.rand(15))[::-1]

    # Compute P-R curve
    tp_cumsum = np.cumsum(matched)
    fp_cumsum = np.cumsum(~matched)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall    = tp_cumsum / n_gt

    ap = compute_ap(recall, precision)
    print(f"  Detection AP example (1 class, 10 GT boxes, 15 preds):")
    print(f"    AP = {ap:.4f}")
    print(f"    Final recall: {recall[-1]:.2f}  ({tp_cumsum[-1]}/{n_gt} GT found)")

    # sklearn cross-check
    ap_sk = average_precision_score(matched.astype(int), confidences)
    print(f"    sklearn AP: {ap_sk:.4f}  (may differ due to confidence ordering)")

    # mAP over multiple classes
    np.random.seed(42)
    n_classes = 20
    class_aps = np.random.uniform(0.3, 0.9, n_classes)
    mAP_50 = class_aps.mean()
    print(f"\n  PASCAL VOC mAP@0.5 (20 classes): {mAP_50:.4f}")

    # COCO mAP: averaged over IoU thresholds 0.5:0.05:0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    coco_aps = []
    for iou_thresh in iou_thresholds:
        # AP decreases at higher IoU thresholds (harder to match)
        scale = 1.0 - 0.5 * (iou_thresh - 0.5)
        coco_aps.append(class_aps * scale)
    coco_mAP = np.mean([ap.mean() for ap in coco_aps])
    print(f"  COCO mAP@[.5:.95] (10 IoU thresholds × 20 classes): {coco_mAP:.4f}")
    print(f"  COCO also reports: AP50={mAP_50:.3f}  AP75={coco_aps[5].mean():.3f}")

    # Plot P-R curve
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.step(recall, precision, where='post', color='blue')
    ax.fill_between(recall, precision, alpha=0.2)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall curve  AP={ap:.3f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("K2_pr_curve.png", dpi=100)
    plt.close()
    print("  Saved: K2_pr_curve.png")
    print("  Done: K2 detection metrics")


# ═════════════════════════════════════════════════════════════════════════════
# K3 — SEGMENTATION METRICS
# ═════════════════════════════════════════════════════════════════════════════

def section_K3():
    print("\n── K3: Segmentation Metrics ──")

    # --- mIoU (mean Intersection over Union) ---
    """
    Per-class IoU:
      IoU_c = TP_c / (TP_c + FP_c + FN_c)
            = intersection / union

    mIoU = mean of IoU_c over all C classes.
    Also called Jaccard Index.

    Dice coefficient (F1 score for segmentation):
      Dice_c = 2*TP_c / (2*TP_c + FP_c + FN_c)
    Relationship: Dice = 2*IoU / (1 + IoU)
    """

    def segmentation_metrics(pred_mask, true_mask, num_classes):
        """
        Compute per-class IoU and Dice for semantic segmentation.
        pred_mask, true_mask: (H, W) integer arrays in [0, num_classes)
        """
        ious, dices = [], []
        for c in range(num_classes):
            pred_c = (pred_mask == c)
            true_c = (true_mask == c)
            tp = (pred_c & true_c).sum()
            fp = (pred_c & ~true_c).sum()
            fn = (~pred_c & true_c).sum()
            iou  = tp / (tp + fp + fn + 1e-8)
            dice = 2*tp / (2*tp + fp + fn + 1e-8)
            ious.append(iou)
            dices.append(dice)
        return np.array(ious), np.array(dices)

    # Synthetic segmentation masks
    H, W = 64, 64
    true_mask = np.zeros((H, W), dtype=int)
    true_mask[10:40, 10:40] = 1   # class 1
    true_mask[30:60, 30:60] = 2   # class 2

    # Predicted mask: slight offset
    pred_mask = np.zeros((H, W), dtype=int)
    pred_mask[12:42, 12:42] = 1   # shifted by 2px
    pred_mask[32:62, 32:62] = 2   # shifted by 2px

    ious, dices = segmentation_metrics(pred_mask, true_mask, num_classes=3)
    miou  = ious.mean()
    mdice = dices.mean()

    print(f"  Per-class metrics (3 classes, 2px shift):")
    for c, (iou, dice) in enumerate(zip(ious, dices)):
        print(f"    Class {c}: IoU={iou:.4f}  Dice={dice:.4f}")
    print(f"  mIoU = {miou:.4f}")
    print(f"  mDice = {mdice:.4f}")
    print(f"  Relationship: Dice ≈ 2*IoU/(1+IoU) = "
          f"{2*ious[1]/(1+ious[1]):.4f} ≈ {dices[1]:.4f}")

    # Boundary IoU (boundary-sensitive metric)
    def boundary_iou(pred, gt, dilation_ratio=0.02):
        """
        Boundary IoU: compute IoU only on pixels near the boundary.
        More sensitive to contour quality than standard IoU.
        """
        from scipy.ndimage import binary_erosion
        d = max(1, int(dilation_ratio * max(pred.shape)))
        struct = np.ones((d*2+1, d*2+1), dtype=bool)
        pred_boundary = pred ^ binary_erosion(pred, struct)
        gt_boundary   = gt   ^ binary_erosion(gt,   struct)
        tp = (pred_boundary & gt_boundary).sum()
        fp = (pred_boundary & ~gt_boundary).sum()
        fn = (~pred_boundary & gt_boundary).sum()
        return tp / (tp + fp + fn + 1e-8)

    biou = boundary_iou(pred_mask == 1, true_mask == 1)
    print(f"  Boundary IoU (class 1): {biou:.4f}  (penalises rough boundaries)")

    # Panoptic Quality
    """
    Panoptic Quality (PQ) = SQ × RQ
    SQ (Segmentation Quality) = average IoU of matched segments
    RQ (Recognition Quality) = F1 of matched vs unmatched segments
    PQ ∈ [0,1] — penalises both bad segmentation AND missed instances
    """
    sq = 0.72   # simulated
    rq = 0.81
    pq = sq * rq
    print(f"\n  Panoptic Quality: SQ={sq:.2f} × RQ={rq:.2f} = PQ={pq:.4f}")
    print("  Done: K3 segmentation metrics")


# ═════════════════════════════════════════════════════════════════════════════
# K4 — IMAGE QUALITY METRICS
# ═════════════════════════════════════════════════════════════════════════════

def section_K4():
    print("\n── K4: Image Quality Metrics ──")

    # --- PSNR ---
    def psnr(img1, img2, max_val=1.0):
        """
        Peak Signal-to-Noise Ratio (dB).
        PSNR = 10 * log10(MAX^2 / MSE)
        Higher = better. Typical: >30dB good, >40dB excellent.
        Limitation: doesn't correlate perfectly with perceptual quality.
        """
        mse = np.mean((img1.astype(float) - img2.astype(float))**2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(max_val**2 / mse)

    # --- SSIM ---
    def ssim(img1, img2, k1=0.01, k2=0.03, sigma=1.5, max_val=1.0):
        """
        Structural Similarity Index (SSIM).
        Compares luminance, contrast, and structure.
        SSIM = (2μ1μ2 + C1)(2σ12 + C2) / ((μ1²+μ2²+C1)(σ1²+σ2²+C2))
        Range: [-1, 1].  1 = identical.
        """
        C1 = (k1 * max_val) ** 2
        C2 = (k2 * max_val) ** 2

        mu1 = gaussian_filter(img1.astype(float), sigma)
        mu2 = gaussian_filter(img2.astype(float), sigma)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = gaussian_filter(img1.astype(float)**2, sigma) - mu1_sq
        sigma2_sq = gaussian_filter(img2.astype(float)**2, sigma) - mu2_sq
        sigma12   = gaussian_filter(img1.astype(float)*img2.astype(float), sigma) - mu1_mu2

        numerator   = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / (denominator + 1e-8)
        return ssim_map.mean()

    # Create test images
    np.random.seed(42)
    H, W = 64, 64
    original = np.random.rand(H, W).astype(np.float32)
    noisy    = np.clip(original + np.random.normal(0, 0.1, (H, W)), 0, 1).astype(np.float32)
    blurred  = gaussian_filter(original, sigma=2.0).astype(np.float32)
    shifted  = np.roll(original, 2, axis=0)   # spatially shifted — same MSE as noisy but different structure

    print(f"  {'Distortion':<15} {'PSNR (dB)':>10} {'SSIM':>8}")
    print(f"  {'-'*36}")
    for name, img in [("Identical", original), ("Noisy", noisy),
                      ("Blurred", blurred), ("Shifted", shifted)]:
        p = psnr(original, img)
        s = ssim(original, img)
        p_str = f"{p:.2f}" if p != float('inf') else "∞"
        print(f"  {name:<15} {p_str:>10} {s:>8.4f}")

    # LPIPS note (perceptual metric, requires deep network)
    print(f"\n  LPIPS (Learned Perceptual Image Patch Similarity):")
    print(f"    Uses VGG/AlexNet feature distances — better perceptual correlation than PSNR/SSIM.")
    print(f"    LPIPS ≈ 0: perceptually identical.  LPIPS >> 0: clearly different.")
    print(f"    Not implemented here (requires pretrained VGG); use pip install lpips.")

    # FID note
    print(f"\n  FID (Fréchet Inception Distance) — for generative models:")
    print(f"    Compares distributions of real vs generated images in Inception feature space.")
    print(f"    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2*sqrt(Σ_r * Σ_g))")
    print(f"    Lower FID = better. FID < 10 = high quality. FID > 100 = poor quality.")

    # Simulate FID computation skeleton
    def compute_fid_from_features(real_feats, fake_feats):
        """Compute FID given feature arrays (N, D)."""
        mu_r, mu_g = real_feats.mean(0), fake_feats.mean(0)
        sigma_r = np.cov(real_feats.T)
        sigma_g = np.cov(fake_feats.T)
        diff = mu_r - mu_g
        # Matrix square root via eigendecomposition (simplified)
        vals, vecs = np.linalg.eigh(sigma_r @ sigma_g)
        sqrt_term  = vecs @ np.diag(np.sqrt(np.abs(vals))) @ vecs.T
        fid = diff @ diff + np.trace(sigma_r + sigma_g - 2*sqrt_term)
        return float(np.real(fid))

    real_feats = np.random.randn(200, 64)   # simulated Inception features
    good_feats = real_feats + np.random.randn(200, 64) * 0.1   # close distribution
    bad_feats  = np.random.randn(200, 64) * 2.0                # far distribution

    fid_good = compute_fid_from_features(real_feats, good_feats)
    fid_bad  = compute_fid_from_features(real_feats, bad_feats)
    print(f"  FID (good generator): {fid_good:.2f}")
    print(f"  FID (bad  generator): {fid_bad:.2f}")
    print("  Done: K4 image quality metrics")


# ═════════════════════════════════════════════════════════════════════════════
# K5 — CLASSIFICATION LOSSES
# ═════════════════════════════════════════════════════════════════════════════

def section_K5():
    print("\n── K5: Classification Losses ──")

    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    # --- Cross-entropy ---
    def cross_entropy(logits, y_true):
        probs = softmax(logits)
        n = len(y_true)
        return -np.mean(np.log(probs[np.arange(n), y_true] + 1e-8))

    # --- Label smoothing ---
    def label_smoothing_loss(logits, y_true, eps=0.1):
        """
        Label smoothing:
        Instead of hard one-hot target, use soft target:
          y_smooth[c] = (1-eps) if c == true_class else eps/(C-1)

        Effect:
          - Prevents overconfident predictions (logit gaps become smaller)
          - Acts as regularisation
          - Improves calibration
        Common value: eps = 0.1
        """
        n, C = logits.shape
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))  # log softmax
        smooth_target = np.full((n, C), eps / (C - 1))
        smooth_target[np.arange(n), y_true] = 1.0 - eps
        return -(smooth_target * log_probs).sum(axis=-1).mean()

    # --- Knowledge Distillation (KD) loss ---
    def kd_loss(student_logits, teacher_logits, y_true, T=4.0, alpha=0.5):
        """
        Knowledge Distillation (Hinton et al. 2015):
          L = alpha * L_CE(student, y_hard) + (1-alpha) * T^2 * KL(student||teacher)

        T = temperature: soften distributions to reveal dark knowledge.
        Large T → softer, more informative targets from teacher.
        alpha: balance between hard labels and soft labels.
        T^2 factor: compensates for reduced gradient magnitude at high T.
        """
        n = len(y_true)
        # Hard label loss
        ce = cross_entropy(student_logits, y_true)
        # Soft label loss (KL divergence between softened distributions)
        student_soft = softmax(student_logits / T)
        teacher_soft = softmax(teacher_logits / T)
        kl = np.mean(np.sum(teacher_soft * np.log(teacher_soft / (student_soft + 1e-8) + 1e-8), axis=-1))
        return alpha * ce + (1 - alpha) * T**2 * kl, ce, kl

    # Test on synthetic logits
    np.random.seed(42)
    n, C = 16, 10
    logits      = np.random.randn(n, C)
    y_true      = np.random.randint(0, C, n)
    teacher_logits = logits + np.random.randn(n, C) * 0.3  # teacher slightly better

    ce_loss = cross_entropy(logits, y_true)
    ls_loss = label_smoothing_loss(logits, y_true, eps=0.1)
    kd, hard, soft = kd_loss(logits, teacher_logits, y_true, T=4.0, alpha=0.5)

    print(f"  Cross-entropy loss:      {ce_loss:.4f}")
    print(f"  Label smoothing (ε=0.1): {ls_loss:.4f}  (slightly higher, prevents overconfidence)")
    print(f"  Knowledge distillation:  {kd:.4f}  (CE={hard:.3f}, KL={soft:.3f}, T=4, α=0.5)")

    # Temperature effect on softened distributions
    print(f"\n  Temperature scaling effect on softmax distribution:")
    logit_ex = np.array([3.0, 1.0, 0.5, -1.0])
    for T in [1, 2, 4, 10]:
        p = softmax(logit_ex / T)
        print(f"    T={T:2d}: {p.round(3)}  entropy={(-p*np.log(p+1e-8)).sum():.3f}")
    print("  Done: K5 classification losses")


# ═════════════════════════════════════════════════════════════════════════════
# K6 — REGRESSION LOSSES: SmoothL1 → CIoU HIERARCHY
# ═════════════════════════════════════════════════════════════════════════════

def section_K6():
    print("\n── K6: Box Regression Losses ──")

    def box_area(box):
        return max(0, box[2]-box[0]) * max(0, box[3]-box[1])

    def iou_val(b1, b2):
        xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
        xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
        inter=max(0,xi2-xi1)*max(0,yi2-yi1)
        union=box_area(b1)+box_area(b2)-inter
        return inter/(union+1e-8)

    def giou_loss(pred, gt):
        iou  = iou_val(pred, gt)
        cx1=min(pred[0],gt[0]); cy1=min(pred[1],gt[1])
        cx2=max(pred[2],gt[2]); cy2=max(pred[3],gt[3])
        c_area=(cx2-cx1)*(cy2-cy1)
        xi1=max(pred[0],gt[0]); yi1=max(pred[1],gt[1])
        xi2=min(pred[2],gt[2]); yi2=min(pred[3],gt[3])
        inter=max(0,xi2-xi1)*max(0,yi2-yi1)
        union=box_area(pred)+box_area(gt)-inter
        giou = iou - (c_area-union)/(c_area+1e-8)
        return 1 - giou

    def diou_loss(pred, gt):
        iou  = iou_val(pred, gt)
        cx_p=(pred[0]+pred[2])/2; cy_p=(pred[1]+pred[3])/2
        cx_g=(gt[0]+gt[2])/2;     cy_g=(gt[1]+gt[3])/2
        d2=(cx_p-cx_g)**2+(cy_p-cy_g)**2
        ex1=min(pred[0],gt[0]); ey1=min(pred[1],gt[1])
        ex2=max(pred[2],gt[2]); ey2=max(pred[3],gt[3])
        c2=(ex2-ex1)**2+(ey2-ey1)**2+1e-8
        return 1-(iou-d2/c2)

    def ciou_loss(pred, gt):
        """
        CIoU = DIoU - α*v
        v = (4/π²) * (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))²
            measures aspect ratio consistency
        α = v / (1 - IoU + v)   (trade-off weight)
        """
        iou  = iou_val(pred, gt)
        cx_p=(pred[0]+pred[2])/2; cy_p=(pred[1]+pred[3])/2
        cx_g=(gt[0]+gt[2])/2;     cy_g=(gt[1]+gt[3])/2
        d2=(cx_p-cx_g)**2+(cy_p-cy_g)**2
        ex1=min(pred[0],gt[0]); ey1=min(pred[1],gt[1])
        ex2=max(pred[2],gt[2]); ey2=max(pred[3],gt[3])
        c2=(ex2-ex1)**2+(ey2-ey1)**2+1e-8
        w_p=pred[2]-pred[0]; h_p=pred[3]-pred[1]
        w_g=gt[2]-gt[0];     h_g=gt[3]-gt[1]
        v = (4/np.pi**2) * (np.arctan(w_g/(h_g+1e-8)) - np.arctan(w_p/(h_p+1e-8)))**2
        alpha_v = v / (1-iou+v+1e-8)
        return 1-(iou - d2/c2 - alpha_v*v)

    # Test cases
    gt   = (10, 10, 50, 40)
    pred_good  = (11, 11, 49, 41)    # near-perfect
    pred_offset= (20, 10, 60, 40)    # shifted right
    pred_aspect= (10, 15, 50, 35)    # wrong aspect ratio

    print(f"  GT box: {gt}")
    print(f"  {'Prediction':<20} {'IoU':>6} {'GIoU-L':>8} {'DIoU-L':>8} {'CIoU-L':>8}")
    print(f"  {'-'*58}")
    for name, pred in [("Near-perfect", pred_good),
                       ("Shifted right", pred_offset),
                       ("Wrong aspect", pred_aspect)]:
        i  = iou_val(pred, gt)
        gl = giou_loss(pred, gt)
        dl = diou_loss(pred, gt)
        cl = ciou_loss(pred, gt)
        print(f"  {name:<20} {i:>6.3f} {gl:>8.4f} {dl:>8.4f} {cl:>8.4f}")

    print(f"\n  Loss hierarchy: IoU → GIoU → DIoU → CIoU")
    print(f"    IoU loss:  1 - IoU. Zero gradient when no overlap.")
    print(f"    GIoU loss: adds enclosing box penalty. Gradient when no overlap.")
    print(f"    DIoU loss: adds centre distance penalty. Faster convergence.")
    print(f"    CIoU loss: adds aspect ratio penalty. Best all-round.")
    print("  Done: K6 box regression losses")


# ═════════════════════════════════════════════════════════════════════════════
# K7 — DISTRIBUTION FOCAL LOSS (DFL) + TASK-ALIGNED LEARNING (TAL)
# ═════════════════════════════════════════════════════════════════════════════

def section_K7():
    print("\n── K7: DFL & Task-Aligned Learning ──")

    """
    Distribution Focal Loss (DFL — Li et al. 2020):
      Instead of predicting a single scalar distance (e.g., to left edge),
      predict a DISTRIBUTION over discrete distance bins [0, 1, ..., n].
      This makes the regression smoother and captures ambiguity.

      Target: a value y ∈ [y_floor, y_ceil]
      Distribute y between two adjacent bins:
        weight_floor = y_ceil - y          (proportion at floor)
        weight_ceil  = y - y_floor         (proportion at ceil)
      DFL = -( w_floor * log(p_floor) + w_ceil * log(p_ceil) )

      Final distance = sum_i (i * p_i)   (expected value)
    """

    def dfl_target(y, n_bins=16):
        """
        Compute DFL soft target (floor/ceil distribution) for value y.
        y should be in [0, n_bins-1].
        """
        y_floor = int(np.floor(y))
        y_ceil  = min(y_floor + 1, n_bins - 1)
        w_floor = y_ceil - y
        w_ceil  = y - y_floor
        target  = np.zeros(n_bins)
        target[y_floor] = w_floor
        target[y_ceil]  = w_ceil
        return target

    def dfl_loss(pred_logits, y_true, n_bins=16):
        """DFL loss: cross-entropy with soft target at floor/ceil bins."""
        target = dfl_target(y_true, n_bins)
        log_probs = pred_logits - np.log(np.exp(pred_logits).sum())  # log-softmax
        loss = -(target * log_probs).sum()
        # Expected value prediction
        probs = np.exp(log_probs)
        pred_val = (probs * np.arange(n_bins)).sum()
        return loss, pred_val

    n_bins = 16
    y_true = 5.7   # true distance to edge (in bins)
    pred_logits = np.random.randn(n_bins)

    loss, pred_val = dfl_loss(pred_logits, y_true, n_bins)
    target = dfl_target(y_true, n_bins)

    print(f"  DFL example (y_true={y_true}, n_bins={n_bins}):")
    print(f"    Soft target:  bin 5 → {target[5]:.2f}, bin 6 → {target[6]:.2f}")
    print(f"    DFL loss:     {loss:.4f}")
    print(f"    Predicted val:{pred_val:.3f}  (expected value over distribution)")

    # --- Task-Aligned Learning (TAL) assignment ---
    """
    TAL (Xu et al. 2022, used in YOLOv8):
    Assigns GT boxes to anchors/cells using a combined score:

      score_ij = cls_score_i(c_j)^α × IoU(pred_i, gt_j)^β

    where i = prediction, j = GT box.
    α controls class score weight, β controls IoU weight.
    Typical: α=0.5, β=6.0 (IoU matters more).

    For each GT, keep top-k predictions with highest scores.
    Conflicts (same cell assigned to multiple GTs) → keep highest score.
    """

    def tal_assignment(cls_scores, ious, alpha=0.5, beta=6.0, top_k=10):
        """
        TAL assignment.
        cls_scores: (N, n_gt)   — cls score of each prediction for each GT class
        ious:       (N, n_gt)   — IoU between each prediction and each GT box
        Returns: assignment array (N,) — index of assigned GT, or -1 if unassigned
        """
        N, n_gt = cls_scores.shape
        # Combined alignment score
        scores = (cls_scores ** alpha) * (ious ** beta)   # (N, n_gt)
        assignment = np.full(N, -1, dtype=int)
        for j in range(n_gt):
            # Top-k predictions for this GT
            top_idx = np.argsort(scores[:, j])[-top_k:]
            assignment[top_idx] = j
        # Resolve conflicts: cell assigned to multiple GTs → keep highest score
        for i in range(N):
            if assignment[i] != -1:
                gts = np.where(
                    np.any(np.isin(np.argsort(scores, axis=0)[-top_k:].T, [i]), axis=1)
                )[0]
                if len(gts) > 1:
                    assignment[i] = gts[np.argmax(scores[i, gts])]
        return assignment

    # Demo
    N, n_gt = 50, 3
    cls_scores = np.random.rand(N, n_gt)
    ious       = np.random.rand(N, n_gt) * 0.8 + 0.1

    assignment = tal_assignment(cls_scores, ious, alpha=0.5, beta=6.0, top_k=10)
    n_assigned = (assignment >= 0).sum()
    print(f"\n  TAL assignment ({N} predictions, {n_gt} GTs, top_k=10):")
    for j in range(n_gt):
        n_j = (assignment == j).sum()
        print(f"    GT {j}: assigned to {n_j} predictions")
    print(f"  Total assigned: {n_assigned}  Unassigned: {N - n_assigned}")
    print("  Done: K7 DFL & TAL")


# ═════════════════════════════════════════════════════════════════════════════
# K8 — SEGMENTATION LOSSES
# ═════════════════════════════════════════════════════════════════════════════

def section_K8():
    print("\n── K8: Segmentation Losses ──")

    # --- Binary Cross-Entropy for segmentation ---
    def bce_seg(pred, target):
        """Pixel-wise BCE: each pixel is independently classified."""
        pred = np.clip(pred, 1e-7, 1-1e-7)
        return -(target * np.log(pred) + (1-target) * np.log(1-pred)).mean()

    # --- Dice loss ---
    def dice_loss(pred, target, smooth=1.0):
        """
        Dice loss = 1 - Dice coefficient
        Dice = 2 * |A ∩ B| / (|A| + |B|)
             = 2*sum(pred*target) / (sum(pred) + sum(target))

        Advantages over BCE:
          - Handles class imbalance naturally (small objects don't get swamped)
          - Directly optimises the segmentation metric (Dice/F1)
        """
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    # --- Tversky loss (generalises Dice) ---
    def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1.0):
        """
        Tversky loss — asymmetric generalisation of Dice:
          T = TP / (TP + α*FP + β*FN)
        α + β = 1.
        β > 0.5: penalises FN more → better recall (useful for detecting rare objects)
        When α=β=0.5: reduces to Dice.
        """
        tp = (pred * target).sum()
        fp = (pred * (1-target)).sum()
        fn = ((1-pred) * target).sum()
        return 1 - (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)

    # --- Combined loss (most common in practice) ---
    def combined_seg_loss(pred, target, w_bce=0.5, w_dice=0.5):
        """BCE + Dice combined loss. Both are complementary:
        BCE: pixel-level accuracy, gradient everywhere.
        Dice: region overlap, handles imbalance.
        """
        return w_bce * bce_seg(pred, target) + w_dice * dice_loss(pred, target)

    # Test on synthetic prediction/target
    H, W = 64, 64
    target = np.zeros((H, W), dtype=np.float32)
    target[20:45, 20:45] = 1.0   # small object (10% of image)

    # Perfect prediction
    pred_perfect = target.copy()
    # Near-miss: slight offset
    pred_near = np.zeros_like(target)
    pred_near[22:47, 22:47] = 0.9
    # Overconfident background
    pred_random = np.random.rand(H, W).astype(np.float32) * 0.3

    print(f"  Segmentation losses comparison (small object = 10% of image):")
    print(f"  {'Prediction':<20} {'BCE':>8} {'Dice-L':>8} {'Tversky-L':>10} {'Combined':>10}")
    print(f"  {'-'*62}")
    for name, pred in [("Perfect", pred_perfect),
                       ("Near-miss (+2px)", pred_near),
                       ("Random low", pred_random)]:
        b  = bce_seg(np.clip(pred, 1e-7, 1-1e-7), target)
        d  = dice_loss(pred, target)
        tv = tversky_loss(pred, target, alpha=0.3, beta=0.7)
        c  = combined_seg_loss(np.clip(pred, 1e-7, 1-1e-7), target)
        print(f"  {name:<20} {b:>8.4f} {d:>8.4f} {tv:>10.4f} {c:>10.4f}")

    # Weighted multi-class loss
    print(f"\n  Multi-task detection loss weighting (Faster R-CNN style):")
    print(f"    L_total = L_rpn_cls + L_rpn_reg + L_det_cls + L_det_reg")
    print(f"    L_total = 1.0 * L_cls + 1.0 * L_reg  (equal weights, common default)")
    print(f"\n  YOLOv8 loss weights:")
    print(f"    L_total = 7.5 * L_box + 0.5 * L_cls + 1.5 * L_dfl")
    print(f"    (box regression dominates — critical for AP)")
    print("  Done: K8 segmentation losses")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CV Section K — Evaluation Metrics & Loss Functions")
    print("=" * 60)

    section_K1()   # P/R/F1, ROC-AUC, macro/micro/weighted
    section_K2()   # AP from P-R curve, mAP, COCO AP@[.5:.95]
    section_K3()   # mIoU, Dice, Boundary IoU, Panoptic Quality
    section_K4()   # PSNR, SSIM, LPIPS note, FID from features
    section_K5()   # CE, label smoothing, knowledge distillation
    section_K6()   # IoU→GIoU→DIoU→CIoU loss hierarchy
    section_K7()   # DFL distribution target, TAL assignment
    section_K8()   # BCE, Dice, Tversky, combined segmentation loss

    print("\n✓ All Section K demos complete.")
