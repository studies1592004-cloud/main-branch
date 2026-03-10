"""
cv_J_object_detection.py
=========================
Computer Vision Course — Section J: Object Detection

Topics covered:
  J1 - IoU, NMS, Soft-NMS, anchor design
  J2 - Two-stage detectors: R-CNN → Fast → Faster R-CNN; RoI Align; 4 losses
  J3 - Feature Pyramid Network (FPN): bottom-up, top-down, lateral connections
  J4 - YOLO family: YOLOv1 grid, v1→v8 progression, anchor-free head
  J5 - SSD: multi-scale anchors, default boxes, 8732 total anchors
  J6 - Focal Loss: derivation, effect on easy/hard examples
  J7 - FCOS: per-pixel prediction, centerness, anchor-free
       DETR: Hungarian matching, bipartite loss

Dependencies: torch, numpy, matplotlib
Install:  pip install torch numpy matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════════════
# J1 — IoU, NMS, ANCHOR DESIGN
# ═════════════════════════════════════════════════════════════════════════════

def section_J1():
    print("\n── J1: IoU, NMS, Anchor Design ──")

    # --- IoU variants ---
    def iou(b1, b2):
        """Standard IoU. Boxes in (x1,y1,x2,y2) format."""
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def giou(b1, b2):
        """
        Generalised IoU: adds penalty for non-overlapping boxes.
        GIoU = IoU - |C minus (A∪B)| / |C|
        where C is the smallest enclosing box of A and B.
        Range: [-1, 1].  GIoU=IoU when boxes overlap perfectly.
        """
        iou_val = iou(b1, b2)
        # Enclosing box
        cx1 = min(b1[0], b2[0]); cy1 = min(b1[1], b2[1])
        cx2 = max(b1[2], b2[2]); cy2 = max(b1[3], b2[3])
        c_area = (cx2-cx1) * (cy2-cy1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
        xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
        inter = max(0,xi2-xi1)*max(0,yi2-yi1)
        union = a1+a2-inter
        return iou_val - (c_area - union) / (c_area + 1e-7)

    def diou(b1, b2):
        """
        Distance IoU: adds normalised centre-distance penalty.
        DIoU = IoU - (d^2 / c^2)
        d = distance between box centres
        c = diagonal of enclosing box
        """
        iou_val = iou(b1, b2)
        cx1 = (b1[0]+b1[2])/2; cy1 = (b1[1]+b1[3])/2
        cx2 = (b2[0]+b2[2])/2; cy2 = (b2[1]+b2[3])/2
        d2  = (cx1-cx2)**2 + (cy1-cy2)**2
        # Enclosing box diagonal
        ex1=min(b1[0],b2[0]); ey1=min(b1[1],b2[1])
        ex2=max(b1[2],b2[2]); ey2=max(b1[3],b2[3])
        c2  = (ex2-ex1)**2 + (ey2-ey1)**2 + 1e-7
        return iou_val - d2/c2

    b1 = (10, 10, 60, 60)
    b2 = (30, 30, 80, 80)
    b3 = (70, 70, 120, 120)   # no overlap

    print(f"  Overlapping boxes ({b1}) vs ({b2}):")
    print(f"    IoU:  {iou(b1,b2):.4f}")
    print(f"    GIoU: {giou(b1,b2):.4f}")
    print(f"    DIoU: {diou(b1,b2):.4f}")

    print(f"  Non-overlapping boxes ({b1}) vs ({b3}):")
    print(f"    IoU:  {iou(b1,b3):.4f}  ← can't distinguish 'how far apart'")
    print(f"    GIoU: {giou(b1,b3):.4f}  ← negative, penalises separation")
    print(f"    DIoU: {diou(b1,b3):.4f}  ← penalises centre distance")

    # --- NMS ---
    def nms(boxes, scores, iou_thresh=0.5):
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        kept  = []
        while order:
            i = order.pop(0)
            kept.append(i)
            order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
        return kept

    boxes  = [(10,10,60,60),(12,12,62,62),(14,14,58,58),(100,100,150,150),(102,102,152,152)]
    scores = [0.95, 0.80, 0.60, 0.92, 0.70]
    kept   = nms(boxes, scores)
    print(f"\n  NMS: {len(boxes)} boxes → {len(kept)} kept (indices {kept})")

    # --- Anchor design ---
    def generate_anchors(stride, scales, ratios, base_size=256):
        """
        Generate anchor boxes centred at a single location.
        scales: sqrt of area relative to base_size^2
        ratios: width/height ratios
        Returns (x1, y1, x2, y2) relative to cell centre.
        """
        anchors = []
        for scale in scales:
            area = (base_size * scale) ** 2
            for ratio in ratios:
                w = np.sqrt(area / ratio)
                h = area / w
                anchors.append((-w/2, -h/2, w/2, h/2))
        return np.array(anchors)

    # Standard Faster R-CNN anchors
    scales = [0.5, 1.0, 2.0]
    ratios = [0.5, 1.0, 2.0]
    anchors = generate_anchors(stride=16, scales=scales, ratios=ratios)
    print(f"\n  Faster R-CNN anchor shapes ({len(scales)} scales × {len(ratios)} ratios = {len(anchors)} per location):")
    for i, (x1,y1,x2,y2) in enumerate(anchors):
        w, h = x2-x1, y2-y1
        print(f"    Anchor {i+1}: {w:.0f}×{h:.0f}  ratio={w/h:.2f}")

    # Total anchors across a feature map
    for feat_size, n_anchors in [(50*50, 9), (25*25, 9), (13*13, 9)]:
        print(f"    {int(feat_size**0.5):2d}×{int(feat_size**0.5):2d} feature map × {n_anchors} anchors = {feat_size*n_anchors:,} total")

    print("  Done: J1 IoU, NMS, anchors")


# ═════════════════════════════════════════════════════════════════════════════
# J2 — TWO-STAGE DETECTORS
# ═════════════════════════════════════════════════════════════════════════════

def section_J2():
    print("\n── J2: Two-Stage Detectors ──")

    """
    R-CNN family evolution:
      R-CNN (2014):     selective search → warp → CNN per region. ~47s/image
      Fast R-CNN (2015):CNN on full image → RoI Pool on features. ~2s/image
      Faster R-CNN (2016): replace selective search with RPN. ~0.2s/image

    Faster R-CNN has 4 losses:
      L_rpn_cls:  binary CE (object vs background) for each anchor
      L_rpn_reg:  smooth-L1 for anchor → proposal box regression
      L_det_cls:  CE over N classes for each RoI proposal
      L_det_reg:  smooth-L1 for proposal → final box regression
    """

    # --- RoI Align (Mask R-CNN improvement over RoI Pool) ---
    def roi_align_1d(feature_line, x_start, x_end, output_size):
        """
        1D RoI Align: bilinear interpolation instead of hard quantisation.
        Eliminates the misalignment caused by integer rounding in RoI Pool.
        """
        L = len(feature_line)
        out = []
        for i in range(output_size):
            # Sample at regular intervals (output_size+1 intervals → output_size bins)
            x = x_start + (i + 0.5) * (x_end - x_start) / output_size
            x = np.clip(x, 0, L - 1)
            lo = int(np.floor(x))
            hi = min(lo + 1, L - 1)
            frac = x - lo
            val = (1 - frac) * feature_line[lo] + frac * feature_line[hi]
            out.append(val)
        return np.array(out)

    # Demonstrate alignment improvement
    feature = np.array([1., 3., 2., 5., 4., 2., 1., 3.])
    # RoI from x=1.7 to x=5.3 — fractional coordinates
    x_start, x_end = 1.7, 5.3
    align_out = roi_align_1d(feature, x_start, x_end, output_size=3)
    # RoI Pool would quantise: floor(1.7)=1 to floor(5.3)=5
    pool_out = np.array([feature[1:3].max(), feature[3:5].max(), feature[5:6].max()])

    print(f"  RoI Align vs RoI Pool (1D demo):")
    print(f"    Feature:    {feature}")
    print(f"    RoI:        x=[{x_start},{x_end}]")
    print(f"    RoI Align:  {align_out.round(3)}")
    print(f"    RoI Pool:   {pool_out.round(3)}")

    # --- Smooth-L1 loss (Huber-like for box regression) ---
    def smooth_l1(x, beta=1.0):
        """
        SmoothL1 (Huber loss variant):
          |x| < beta: 0.5 * x^2 / beta     (quadratic — stable for small errors)
          |x| >= beta: |x| - 0.5 * beta    (linear  — robust to outliers)
        """
        return np.where(np.abs(x) < beta,
                        0.5 * x**2 / beta,
                        np.abs(x) - 0.5 * beta)

    errors = np.array([-3., -1., -0.5, 0., 0.5, 1., 3.])
    sl1    = smooth_l1(errors, beta=1.0)
    l2     = 0.5 * errors**2
    l1     = np.abs(errors)
    print(f"\n  Smooth-L1 vs L1 vs L2 at various errors:")
    print(f"  {'error':>6} {'Smooth-L1':>10} {'L1':>8} {'L2':>8}")
    for e, s, l1v, l2v in zip(errors, sl1, l1, l2):
        print(f"  {e:>6.1f} {s:>10.3f} {l1v:>8.3f} {l2v:>8.3f}")

    # --- Faster R-CNN summary ---
    print(f"\n  Faster R-CNN architecture summary:")
    print(f"    Backbone (ResNet-50 + FPN)")
    print(f"    RPN: 3×3 conv → cls head (2k scores) + reg head (4k offsets)")
    print(f"         k=9 anchors per location. NMS → top-300 proposals.")
    print(f"    RoI Align: map proposals to 7×7 feature grids")
    print(f"    Detection head: 2× FC → cls (C+1) + reg (4C)")
    print(f"    Losses: L = L_rpn_cls + L_rpn_reg + L_det_cls + L_det_reg")
    print("  Done: J2 two-stage detectors")


# ═════════════════════════════════════════════════════════════════════════════
# J3 — FEATURE PYRAMID NETWORK (FPN)
# ═════════════════════════════════════════════════════════════════════════════

def section_J3():
    print("\n── J3: Feature Pyramid Network (FPN) ──")

    """
    FPN (Lin et al. 2017) builds a multi-scale feature pyramid with:
      Bottom-up pathway:   standard forward pass through backbone (C2..C5)
      Top-down pathway:    upsample from C5 down, add lateral connections
      Lateral connections: 1×1 conv on bottom-up → merge with top-down

    Each pyramid level P_i has 256 channels.
    Larger features (P2) → small objects; smaller features (P5) → large objects.

    Strides: P2=4, P3=8, P4=16, P5=32 (×2 per level)
    """

    class LateralBlock(nn.Module):
        """Single FPN lateral connection + top-down merge."""
        def __init__(self, in_channels, out_channels=256):
            super().__init__()
            self.lateral = nn.Conv2d(in_channels, out_channels, 1)
            self.output  = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        def forward(self, bottom_up, top_down=None):
            lateral = self.lateral(bottom_up)
            if top_down is not None:
                top_down_up = F.interpolate(top_down, size=lateral.shape[-2:], mode='nearest')
                lateral = lateral + top_down_up
            return self.output(lateral)

    class FPN(nn.Module):
        """Minimal FPN operating on 4 backbone feature levels."""
        def __init__(self, in_channels_list=(256, 512, 1024, 2048), out_channels=256):
            super().__init__()
            self.blocks = nn.ModuleList([
                LateralBlock(c, out_channels) for c in in_channels_list
            ])

        def forward(self, features):
            """features: [C2, C3, C4, C5] from bottom-up."""
            # Top-down pass (from C5 to C2)
            outputs = []
            top_down = None
            for feat, block in zip(reversed(features), reversed(self.blocks)):
                out = block(feat, top_down)
                top_down = out
                outputs.append(out)
            return list(reversed(outputs))   # [P2, P3, P4, P5]

    # Simulate backbone feature maps (ResNet-50 output channels & spatial sizes)
    backbone_features = [
        torch.randn(1, 256,  56, 56),   # C2
        torch.randn(1, 512,  28, 28),   # C3
        torch.randn(1, 1024, 14, 14),   # C4
        torch.randn(1, 2048,  7,  7),   # C5
    ]

    fpn = FPN()
    with torch.no_grad():
        pyramid = fpn(backbone_features)

    print(f"  FPN output pyramid:")
    level_names = ['P2', 'P3', 'P4', 'P5']
    for name, feat in zip(level_names, pyramid):
        stride = 224 // feat.shape[-1]   # approximate stride for 224 input
        print(f"    {name}: shape={tuple(feat.shape)}  stride≈{stride}  "
              f"detects objects ~{stride*2}–{stride*8}px")

    total_params = sum(p.numel() for p in fpn.parameters())
    print(f"  FPN params: {total_params:,}")

    # Anchor assignment to pyramid levels (by object size)
    print(f"\n  Anchor size → pyramid level assignment (RetinaNet rule):")
    print(f"    Object area < 32²  → P2 (stride 4,  finest detail)")
    print(f"    Object area < 64²  → P3 (stride 8)")
    print(f"    Object area < 128² → P4 (stride 16)")
    print(f"    Object area < 256² → P5 (stride 32)")
    print(f"    Object area ≥ 256² → P6 (stride 64, extra level in RetinaNet)")
    print("  Done: J3 FPN")


# ═════════════════════════════════════════════════════════════════════════════
# J4 — YOLO FAMILY
# ═════════════════════════════════════════════════════════════════════════════

def section_J4():
    print("\n── J4: YOLO Family ──")

    """
    YOLOv1 (Redmon 2016):
      - Single pass: divide image into S×S grid (S=7)
      - Each cell predicts B=2 boxes and C=20 class probabilities
      - Output tensor: S×S×(B*5 + C) = 7×7×30
      - Loss: λ_coord * box_loss + obj_loss + λ_noobj * noobj_loss + class_loss
      - λ_coord=5, λ_noobj=0.5 (up-weight localisation, down-weight no-object)
    """

    # YOLOv1 output tensor
    S, B, C = 7, 2, 20
    output_size = S * S * (B * 5 + C)
    print(f"  YOLOv1 output tensor: {S}×{S}×({B}*5+{C}) = {S}×{S}×{B*5+C} = {output_size} values")

    # Each cell predicts: [x, y, w, h, conf] × B + [p_1,...,p_C]
    print(f"  Per cell: {B} boxes × 5 values + {C} classes")
    print(f"  Total predictions: {S*S*B} boxes  (before NMS)")

    # --- YOLOv1 loss components ---
    def yolo_loss_demo(pred, target, lambda_coord=5.0, lambda_noobj=0.5):
        """
        Simplified YOLOv1 loss for a single cell with one anchor.
        pred/target: [x, y, w, h, conf, *class_probs]
        """
        px, py, pw, ph, pconf = pred[:5]
        tx, ty, tw, th, tconf = target[:5]
        has_obj = tconf > 0

        # Coordinate loss (only for cells with objects, λ_coord=5)
        coord_loss = lambda_coord * has_obj * (
            (px - tx)**2 + (py - ty)**2 +
            (np.sqrt(pw) - np.sqrt(tw))**2 +   # sqrt of w,h — penalises small box errors more
            (np.sqrt(ph) - np.sqrt(th))**2
        )

        # Confidence loss
        obj_loss   = has_obj * (pconf - tconf)**2
        noobj_loss = lambda_noobj * (1 - has_obj) * pconf**2

        # Class loss (only with object)
        class_loss = has_obj * np.sum((pred[5:] - target[5:])**2)

        total = coord_loss + obj_loss + noobj_loss + class_loss
        return total, coord_loss, obj_loss, noobj_loss, class_loss

    n_classes = 3
    pred   = np.array([0.5, 0.5, 0.3, 0.4, 0.8] + [0.7, 0.1, 0.2])
    target = np.array([0.45, 0.48, 0.32, 0.38, 1.0] + [1.0, 0.0, 0.0])
    total, coord, obj, noobj, cls = yolo_loss_demo(pred, target)
    print(f"\n  YOLOv1 loss (with object):")
    print(f"    coord={coord:.4f}  obj={obj:.4f}  noobj={noobj:.4f}  class={cls:.4f}  total={total:.4f}")

    # --- YOLO progression ---
    print(f"\n  YOLO family progression:")
    print(f"  {'Version':<12} {'Year':>6} {'Backbone':<20} {'Key innovation'}")
    print(f"  {'-'*70}")
    yolos = [
        ("YOLOv1",  2016, "Darknet",        "Unified detection, 7×7 grid, end-to-end"),
        ("YOLOv2",  2017, "Darknet-19",     "Anchor boxes, batch norm, multi-scale train"),
        ("YOLOv3",  2018, "Darknet-53",     "3-scale predictions, residual connections"),
        ("YOLOv4",  2020, "CSPDarknet53",   "Mosaic aug, CIoU, CBAM attention, PANet"),
        ("YOLOv5",  2020, "CSP",            "PyTorch, AutoAnchor, focus layer"),
        ("YOLOv6",  2022, "EfficientRep",   "Decoupled head, RepVGG backbone"),
        ("YOLOv7",  2022, "ELAN",           "Extended layer aggregation, aux loss"),
        ("YOLOv8",  2023, "CSPDarknet",     "Anchor-free, decoupled head, task-aligned"),
    ]
    for name, year, backbone, key in yolos:
        print(f"  {name:<12} {year:>6} {backbone:<20} {key}")

    # --- Anchor-free YOLO head (v8 style) ---
    """
    YOLOv8 anchor-free head:
      For each grid cell (i,j) at stride s:
        box:  predict (x,y,w,h) directly (no anchor priors needed)
        cls:  predict class probabilities (sigmoid, binary each)
      Distribution Focal Loss (DFL): predict distribution over
      discrete distances rather than raw offsets → smoother gradients.
    """
    print(f"\n  YOLOv8 anchor-free head:")
    print(f"    Input: 80×80 feature map (640px / stride 8)")
    print(f"    Per cell: 4 (box) + 80 (cls) = 84 values")
    print(f"    Total outputs: 80×80×84 + 40×40×84 + 20×20×84 = {80*80*84+40*40*84+20*20*84:,}")
    print("  Done: J4 YOLO family")


# ═════════════════════════════════════════════════════════════════════════════
# J5 — SSD
# ═════════════════════════════════════════════════════════════════════════════

def section_J5():
    print("\n── J5: SSD — Single Shot MultiBox Detector ──")

    """
    SSD (Liu et al. 2016):
      Uses multiple feature maps at different scales.
      Default boxes (anchors) of different aspect ratios at each location.
      Predicts class scores + box offsets for each default box.

    SSD-300 default box counts:
      conv4_3:  38×38 × 4 = 5,776
      fc7:      19×19 × 6 = 2,166
      conv6_2:  10×10 × 6 =   600
      conv7_2:   5×5  × 6 =   150
      conv8_2:   3×3  × 4 =    36
      conv9_2:   1×1  × 4 =     4
      Total:                 8,732
    """

    ssd_layers = [
        ("conv4_3",  38, 4),
        ("fc7",      19, 6),
        ("conv6_2",  10, 6),
        ("conv7_2",   5, 6),
        ("conv8_2",   3, 4),
        ("conv9_2",   1, 4),
    ]

    total_anchors = 0
    print(f"  SSD-300 anchor counts:")
    print(f"  {'Layer':<12} {'Size':>6} {'Anchors/loc':>12} {'Total':>8}")
    print(f"  {'-'*45}")
    for name, size, k in ssd_layers:
        count = size * size * k
        total_anchors += count
        print(f"  {name:<12} {size:>4}×{size:<4} {k:>12} {count:>8,}")
    print(f"  {'Total':<12} {'':>6} {'':>12} {total_anchors:>8,}")

    # Default box generation (aspect ratios + extra scale)
    def default_boxes(s_k, s_k1, aspect_ratios):
        """
        Generate default box widths/heights for one SSD layer.
        s_k, s_k1: current and next scale (as fraction of image size)
        """
        boxes = []
        # Standard aspect ratios
        for ar in aspect_ratios:
            w = s_k * np.sqrt(ar)
            h = s_k / np.sqrt(ar)
            boxes.append((w, h))
        # Extra box at geometric mean of adjacent scales
        s_prime = np.sqrt(s_k * s_k1)
        boxes.append((s_prime, s_prime))
        return boxes

    scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
    print(f"\n  Default box shapes for conv4_3 (scale={scales[0]:.2f}):")
    boxes = default_boxes(scales[0], scales[1], aspect_ratios=[1, 2, 0.5])
    for i, (w, h) in enumerate(boxes):
        print(f"    Box {i+1}: {w:.3f}×{h:.3f}  (as fraction of image)")

    # Matching strategy: assign default box to GT if IoU > 0.5
    print(f"\n  SSD matching strategy:")
    print(f"    1. For each GT box: assign best-matching default box (ensures coverage)")
    print(f"    2. For each default box with IoU > 0.5 with any GT: assign as positive")
    print(f"    3. All remaining → negative (background)")
    print(f"    4. Hard negative mining: sort negatives by loss, keep top 3× positives")
    print("  Done: J5 SSD")


# ═════════════════════════════════════════════════════════════════════════════
# J6 — FOCAL LOSS
# ═════════════════════════════════════════════════════════════════════════════

def section_J6():
    print("\n── J6: Focal Loss ──")

    """
    Problem: one-stage detectors evaluate ~100K anchor boxes per image.
    ~99% are easy negatives (background). Their loss dominates, overwhelming
    the signal from the rare hard positives.

    Focal Loss (Lin et al. RetinaNet 2017):
      FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
      p_t = p if y=1, else (1-p)   (probability of correct class)
      (1-p_t)^γ = modulating factor (down-weights easy examples)
      α_t       = class balancing weight

    γ=0: standard cross-entropy
    γ=2: easy examples (p_t≈0.9) get (1-0.9)^2=0.01 weight
         hard examples (p_t≈0.5) get (1-0.5)^2=0.25 weight
         ratio: 25× more attention to hard examples
    """

    def focal_loss(p, y, gamma=2.0, alpha=0.25):
        """Binary focal loss."""
        p = np.clip(p, 1e-8, 1-1e-8)
        p_t     = np.where(y == 1, p, 1 - p)
        alpha_t = np.where(y == 1, alpha, 1 - alpha)
        return -alpha_t * (1 - p_t)**gamma * np.log(p_t)

    def bce(p, y):
        p = np.clip(p, 1e-8, 1-1e-8)
        return -(y * np.log(p) + (1-y) * np.log(1-p))

    # Example: 10,000 background (easy) + 100 foreground (hard)
    np.random.seed(42)
    p_bg = np.random.beta(8, 2, 10000)   # background: model confident (p~0.9 for background)
    p_fg = np.random.beta(2, 2, 100)     # foreground: model uncertain

    y_bg = np.zeros(10000)
    y_fg = np.ones(100)

    # Background: model predicts LOW probability of object → p_t = 1 - p_bg ≈ 0.1
    bce_bg = bce(p_bg, y_bg)
    fl_bg  = focal_loss(p_bg, y_bg, gamma=2)

    # Foreground: model uncertain → p_t = p_fg ≈ 0.5
    bce_fg = bce(p_fg, y_fg)
    fl_fg  = focal_loss(p_fg, y_fg, gamma=2)

    print(f"  Class imbalance: 10,000 bg  vs  100 fg (100:1 ratio)")
    print(f"  {'Metric':<30} {'BCE':>10} {'Focal (γ=2)':>12}")
    print(f"  {'-'*55}")
    print(f"  {'Total bg loss (10K)':30} {bce_bg.sum():>10.2f} {fl_bg.sum():>12.2f}")
    print(f"  {'Total fg loss (100)':30} {bce_fg.sum():>10.2f} {fl_fg.sum():>12.2f}")
    print(f"  {'bg / fg ratio':30} {bce_bg.sum()/bce_fg.sum():>10.1f}x {fl_bg.sum()/fl_fg.sum():>12.1f}x")
    print(f"  → Focal loss reduces bg dominance by {bce_bg.sum()/bce_fg.sum() / (fl_bg.sum()/fl_fg.sum()):.0f}×")

    # Gamma sweep
    print(f"\n  Effect of gamma on easy (p_t=0.95) vs hard (p_t=0.5) example weighting:")
    print(f"  {'γ':>4} {'weight easy':>12} {'weight hard':>12} {'ratio':>8}")
    print(f"  {'-'*42}")
    for gamma in [0, 0.5, 1, 2, 5]:
        w_easy = (1 - 0.95)**gamma
        w_hard = (1 - 0.50)**gamma
        print(f"  {gamma:>4} {w_easy:>12.4f} {w_hard:>12.4f} {w_hard/w_easy:>8.1f}×")

    # Plot
    p_range = np.linspace(0.01, 0.99, 200)
    fig, ax = plt.subplots(figsize=(7, 4))
    y_ones = np.ones_like(p_range)
    ax.plot(p_range, bce(p_range, y_ones), 'k-', lw=2, label='CE (γ=0)')
    for gamma in [0.5, 1, 2, 5]:
        fl = focal_loss(p_range, y_ones, gamma=gamma, alpha=1.0)
        ax.plot(p_range, fl, label=f'Focal γ={gamma}')
    ax.set_xlabel('p_t (probability of correct class)')
    ax.set_ylabel('Loss')
    ax.set_title('Focal Loss vs Cross-Entropy')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 5)
    plt.tight_layout()
    plt.savefig("J6_focal_loss.png", dpi=100)
    plt.close()
    print("  Saved: J6_focal_loss.png")
    print("  Done: J6 Focal Loss")


# ═════════════════════════════════════════════════════════════════════════════
# J7 — FCOS + DETR
# ═════════════════════════════════════════════════════════════════════════════

def section_J7():
    print("\n── J7: FCOS (Anchor-Free) + DETR ──")

    # --- FCOS ---
    """
    FCOS (Tian et al. 2019) — Fully Convolutional One-Stage detector:
      Predicts per PIXEL (no anchors at all):
        (l, r, t, b): distances from pixel to left/right/top/bottom of GT box
        centerness:   sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
                      ∈ (0,1], peaks at box centre, falls off at edges
        class:        C-dim class probabilities

      During inference: score = class_score * centerness_score
      (centerness down-weights off-centre, low-quality boxes)
    """

    def centerness(l, r, t, b):
        """Centerness score for a single pixel's prediction."""
        return np.sqrt(
            (min(l, r) / (max(l, r) + 1e-8)) *
            (min(t, b) / (max(t, b) + 1e-8))
        )

    # GT box: (x1=10, y1=10, x2=50, y2=40)  size 40×30
    box = (10, 10, 50, 40)
    test_pixels = [
        (30, 25, "centre"),
        (15, 15, "near top-left corner"),
        (10, 10, "at corner"),
        (20, 20, "inside but off-centre"),
    ]

    print(f"  FCOS centerness for GT box {box}:")
    for px, py, desc in test_pixels:
        l = px - box[0]; r = box[2] - px
        t = py - box[1]; b = box[3] - py
        if l >= 0 and r >= 0 and t >= 0 and b >= 0:
            c = centerness(l, r, t, b)
        else:
            c = 0.0   # outside box
        print(f"    pixel ({px:2d},{py:2d}) [{desc}]: l={l} r={r} t={t} b={b}  centerness={c:.3f}")

    # FCOS head structure
    print(f"\n  FCOS head (per FPN level):")
    print(f"    4× (3×3 conv+ReLU) shared trunk")
    print(f"    → cls branch:  3×3 → sigmoid (C classes, binary each)")
    print(f"    → reg branch:  3×3 → exp(x) (l,r,t,b distances, always positive)")
    print(f"    → ctr branch:  3×3 → sigmoid (centerness)")
    print(f"    Params (per level): ~same as RetinaNet head but no anchor hyperparams")

    # --- DETR ---
    """
    DETR (Carion et al. 2020) — Detection Transformer:
      1. CNN backbone → flattened spatial features
      2. Transformer encoder: self-attention over all spatial positions
      3. Transformer decoder: N=100 learned object queries attend to encoder output
      4. Each query → (class, box) prediction independently
      5. Hungarian matching: find optimal bipartite assignment between
         N predictions and M GT boxes (M≤N). Unmatched → "no object" class.
      6. Loss only on matched pairs.

    Hungarian matching cost:
      C(pred_i, gt_j) = -p_i(c_j) + λ_iou * (1 - IoU(b_i, b_j)) + λ_L1 * |b_i - b_j|
    """

    def hungarian_matching_demo():
        """
        Tiny Hungarian matching example.
        3 predictions, 2 GT boxes.
        """
        from itertools import permutations

        # Simulated costs: cost[i,j] = cost of assigning pred i to GT j
        cost = np.array([
            [0.2, 0.9],   # pred 0 is close to GT 0
            [0.8, 0.3],   # pred 1 is close to GT 1
            [0.5, 0.6],   # pred 2 is mediocre for both
        ])

        # Try all assignments of 2 GTs to 3 preds (choose 2 from 3)
        from itertools import combinations
        best_cost = float('inf')
        best_assignment = None
        for pred_indices in combinations(range(3), 2):
            for gt_perm in permutations(range(2)):
                c = sum(cost[pi, gi] for pi, gi in zip(pred_indices, gt_perm))
                if c < best_cost:
                    best_cost = c
                    best_assignment = list(zip(pred_indices, gt_perm))

        return best_assignment, best_cost

    assignment, cost = hungarian_matching_demo()
    print(f"\n  DETR Hungarian matching (3 preds, 2 GTs):")
    print(f"    Cost matrix: pred×GT")
    print(f"      [[0.2, 0.9], [0.8, 0.3], [0.5, 0.6]]")
    print(f"    Optimal assignment: {assignment}  (pred→GT)  total cost={cost:.2f}")
    print(f"    Pred 2 → 'no object' (unmatched, uses ∅ class label)")

    print(f"\n  DETR vs FCOS vs YOLO comparison:")
    print(f"  {'Model':<10} {'Anchors':>8} {'Post-proc':>10} {'Training':>10} {'Notes'}")
    print(f"  {'-'*65}")
    models = [
        ("YOLO",  "Yes",  "NMS",     "Fast",   "Anchor hyperparams needed"),
        ("FCOS",  "No",   "NMS",     "Fast",   "Centerness branch"),
        ("DETR",  "No",   "None",    "Slow",   "Hungarian match, Transformer"),
        ("D-DETR","No",   "None",    "Medium", "Deformable attention, 10× faster"),
    ]
    for name, anchors, post, train, note in models:
        print(f"  {name:<10} {anchors:>8} {post:>10} {train:>10}  {note}")
    print("  Done: J7 FCOS + DETR")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(f"CV Section J — Object Detection  ({DEVICE})")
    print("=" * 60)

    section_J1()   # IoU variants, NMS, anchor generation
    section_J2()   # R-CNN family, RoI Align, Smooth-L1, 4 losses
    section_J3()   # FPN: bottom-up + top-down + lateral
    section_J4()   # YOLO v1–v8 progression, anchor-free head
    section_J5()   # SSD: 8732 anchors, default box generation
    section_J6()   # Focal Loss: derivation, gamma sweep, plot
    section_J7()   # FCOS centerness, DETR Hungarian matching

    print("\n✓ All Section J demos complete.")
