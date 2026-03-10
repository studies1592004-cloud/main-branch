"""
cv_G_cnns.py
=============
Computer Vision Course — Section G: Convolutional Neural Networks

Topics covered:
  G1 - Conv layer mechanics: weight sharing, param count, output shape
  G2 - Pooling: max pooling, average pooling, global average pooling
  G3 - Receptive field computation: stacked 3x3 convs, dilated convolutions
  G4 - Classic architectures: LeNet-5, AlexNet (param counts, layer-by-layer)
  G5 - VGG & Inception: 3x3 rationale, 1x1 bottleneck cost reduction
  G6 - ResNet: residual block, bottleneck block, ResNet-18/50 from scratch
  G7 - Modern architectures: MobileNetV2 (depthwise separable), custom CNN

Dependencies: torch, torchvision, numpy, matplotlib
Install:  pip install torch torchvision numpy matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model, input_shape):
    """Print layer-by-layer output shapes."""
    hooks, shapes = [], []

    def hook_fn(module, inp, out):
        name = type(module).__name__
        shapes.append((name, tuple(out.shape[1:])))   # exclude batch dim

    for m in model.modules():
        if not isinstance(m, nn.Sequential) and list(m.children()) == []:
            hooks.append(m.register_forward_hook(hook_fn))

    x = torch.zeros(1, *input_shape)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()
    return shapes


# ═════════════════════════════════════════════════════════════════════════════
# G1 — CONV LAYER MECHANICS
# ═════════════════════════════════════════════════════════════════════════════

def section_G1():
    print("\n── G1: Conv Layer Mechanics ──")

    # Output size formula
    def conv_out_size(H, K, P, S, D=1):
        """H=input, K=kernel, P=padding, S=stride, D=dilation"""
        K_eff = D * (K - 1) + 1    # effective kernel size with dilation
        return (H + 2*P - K_eff) // S + 1

    print("  Output size O = floor((H + 2P - D*(K-1) - 1) / S) + 1:")
    for H, K, P, S, D in [(224,3,1,1,1), (224,3,0,2,1), (56,3,1,1,2), (28,1,0,1,1)]:
        O = conv_out_size(H, K, P, S, D)
        print(f"    H={H} K={K} P={P} S={S} D={D} → O={O}")

    # Parameter count
    def conv_params(C_in, C_out, K, groups=1, bias=True):
        # weight: (C_out, C_in/groups, K, K)
        w = C_out * (C_in // groups) * K * K
        b = C_out if bias else 0
        return w + b

    print(f"\n  Parameter counts:")
    tests = [
        (3,   64,  3, 1, "First conv (ImageNet)"),
        (64,  128, 3, 1, "VGG block"),
        (256, 256, 3, 1, "Standard mid-layer"),
        (256, 256, 1, 1, "1x1 conv (channel mixing)"),
        (256, 256, 3, 256, "Depthwise 3x3 (groups=C_in)"),
    ]
    for C_in, C_out, K, G, label in tests:
        p = conv_params(C_in, C_out, K, G)
        print(f"    {label:<35}: {p:>10,} params")

    # Weight sharing demo
    conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    x    = torch.randn(1, 3, 224, 224)
    y    = conv(x)
    print(f"\n  Conv2d(3→64, K=3, P=1):")
    print(f"    Input shape:   {tuple(x.shape)}")
    print(f"    Output shape:  {tuple(y.shape)}")
    print(f"    Weight shape:  {tuple(conv.weight.shape)}")
    print(f"    Parameters:    {conv.weight.numel() + conv.bias.numel():,}")
    print(f"    Weight sharing: same {conv.weight.numel():,} weights applied at all {224*224:,} positions")

    # 1x1 convolution (channel projection / mixing)
    conv1x1 = nn.Conv2d(256, 64, kernel_size=1)
    x256    = torch.randn(1, 256, 28, 28)
    y64     = conv1x1(x256)
    print(f"\n  1x1 Conv (256→64): {tuple(x256.shape)} → {tuple(y64.shape)}")
    print(f"    Params: {count_params(conv1x1)[0]:,}  (no spatial mixing, only channel mixing)")
    print("  Done: G1 conv layer mechanics")


# ═════════════════════════════════════════════════════════════════════════════
# G2 — POOLING
# ═════════════════════════════════════════════════════════════════════════════

def section_G2():
    print("\n── G2: Pooling Layers ──")

    x = torch.randn(1, 64, 56, 56)

    # Max pooling
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    y_max    = max_pool(x)
    print(f"  MaxPool2d(k=2, s=2): {tuple(x.shape)} → {tuple(y_max.shape)}")

    # Average pooling
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    y_avg    = avg_pool(x)
    print(f"  AvgPool2d(k=2, s=2): {tuple(x.shape)} → {tuple(y_avg.shape)}")

    # Global Average Pooling (GAP) — replaces flatten + FC in modern nets
    gap = nn.AdaptiveAvgPool2d((1, 1))
    y_gap = gap(x).squeeze(-1).squeeze(-1)   # (1, 64)
    print(f"  GlobalAvgPool:       {tuple(x.shape)} → {tuple(y_gap.shape)}  (spatial collapsed)")

    # Adaptive average pooling (resize to any target)
    adapt = nn.AdaptiveAvgPool2d((7, 7))
    y_adapt = adapt(x)
    print(f"  AdaptiveAvgPool(7x7):{tuple(x.shape)} → {tuple(y_adapt.shape)}")

    # Manual max pooling (for understanding)
    def max_pool_manual(x_np, k=2, s=2):
        """Max pooling on a 2D array."""
        H, W = x_np.shape
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        out = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                out[i,j] = x_np[i*s:i*s+k, j*s:j*s+k].max()
        return out

    patch = np.array([[1,3,2,4],
                      [5,6,7,8],
                      [3,2,1,0],
                      [1,2,3,4]], dtype=float)
    pooled = max_pool_manual(patch, k=2, s=2)
    print(f"\n  Manual MaxPool2 example:")
    print(f"    Input:\n{patch}")
    print(f"    Output:\n{pooled}")

    # Strided conv vs max pool
    stride_conv = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
    y_stride    = stride_conv(x)
    print(f"\n  Strided conv(s=2): {tuple(x.shape)} → {tuple(y_stride.shape)}")
    print(f"  MaxPool(s=2):      {tuple(x.shape)} → {tuple(y_max.shape)}")
    print(f"  Strided conv params: {count_params(stride_conv)[0]:,}  (learns downsampling)")
    print(f"  MaxPool params:      0  (no learning)")
    print("  Done: G2 pooling")


# ═════════════════════════════════════════════════════════════════════════════
# G3 — RECEPTIVE FIELD
# ═════════════════════════════════════════════════════════════════════════════

def section_G3():
    print("\n── G3: Receptive Field ──")

    """
    Receptive field (RF) of a stacked conv network:
      RF_L = 1 + sum_{l=1}^{L} (K_l - 1) * prod_{j=1}^{l-1} S_j

    For L layers of 3x3 conv (K=3, S=1):
      RF = 1 + L*2 = 2L + 1

    For max-pool or stride S=2 between groups:
      RF grows faster.
    """

    def compute_rf(layers):
        """
        layers: list of (kernel_size, stride, dilation)
        Returns receptive field size.
        """
        rf = 1
        total_stride = 1
        for K, S, D in layers:
            K_eff = D * (K - 1) + 1
            rf += (K_eff - 1) * total_stride
            total_stride *= S
        return rf

    # Stacked 3x3 convs
    print("  RF of stacked 3x3 convs (S=1):")
    for n_layers in [1, 2, 3, 5, 10]:
        layers = [(3, 1, 1)] * n_layers
        rf = compute_rf(layers)
        print(f"    {n_layers:2d} layers → RF = {rf:3d}  (formula: 2*{n_layers}+1={2*n_layers+1})")

    # 3x3 with stride-2 pooling
    print("\n  RF of VGG-like blocks (3x3 conv x2 then pool):")
    vgg_like = [(3,1,1),(3,1,1),(1,2,1)] * 5  # 5 blocks, pool between blocks
    running_layers = []
    for block in range(5):
        running_layers += [(3,1,1),(3,1,1),(1,2,1)]
        rf = compute_rf(running_layers)
        print(f"    After block {block+1}: RF = {rf}")

    # Dilated convolutions
    print("\n  Dilated 3x3 convs (K_eff = D*(K-1)+1):")
    for D in [1, 2, 4, 8, 16]:
        K_eff = D * (3 - 1) + 1
        print(f"    Dilation D={D:2d}: effective kernel = {K_eff:2d}x{K_eff:2d}")

    # Dilated stack (1,2,4,8) — used in DeepLab / WaveNet
    dilated_layers = [(3,1,1),(3,1,2),(3,1,4),(3,1,8)]
    rf_dilated = compute_rf(dilated_layers)
    print(f"\n  Dilated stack [D=1,2,4,8]: RF = {rf_dilated}")
    print(f"  Same RF from 4 standard 3x3: {compute_rf([(3,1,1)]*4)}")

    print("  Done: G3 receptive field")


# ═════════════════════════════════════════════════════════════════════════════
# G4 — LENET-5 & ALEXNET
# ═════════════════════════════════════════════════════════════════════════════

def section_G4():
    print("\n── G4: LeNet-5 & AlexNet ──")

    # ── LeNet-5 (LeCun 1998) ─────────────────────────────────────────────────
    class LeNet5(nn.Module):
        """
        LeNet-5: designed for 32x32 grayscale images (MNIST).
        Architecture: C1(6,5x5) → S2(pool) → C3(16,5x5) → S4(pool) → C5(120,5x5) → F6(84) → Output(10)
        """
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5),      # C1: 32→28, 6 maps
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2, stride=2),  # S2: 28→14
                nn.Conv2d(6, 16, kernel_size=5),     # C3: 14→10, 16 maps
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2, stride=2),  # S4: 10→5
                nn.Conv2d(16, 120, kernel_size=5),   # C5: 5→1, 120 maps
                nn.Tanh(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(120, 84),
                nn.Tanh(),
                nn.Linear(84, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    lenet = LeNet5()
    total, _ = count_params(lenet)
    print(f"  LeNet-5 total params: {total:,}")
    x = torch.randn(1, 1, 32, 32)
    shapes = model_summary(lenet, (1, 32, 32))
    for name, shape in shapes:
        print(f"    {name:<20}: {shape}")

    # ── AlexNet (Krizhevsky 2012) ─────────────────────────────────────────────
    class AlexNet(nn.Module):
        """
        AlexNet: ImageNet 1000-class classifier.
        5 Conv innovations over LeNet:
          1. ReLU (not tanh) — faster training
          2. Dropout — regularisation
          3. Data augmentation
          4. Multi-GPU training
          5. Local Response Normalisation (LRN)
        """
        def __init__(self, num_classes=1000):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 11, stride=4, padding=0),   # 224→55
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2),                    # 55→27
                nn.Conv2d(96, 256, 5, padding=2),             # 27→27
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2),                    # 27→13
                nn.Conv2d(256, 384, 3, padding=1),            # 13→13
                nn.ReLU(),
                nn.Conv2d(384, 384, 3, padding=1),            # 13→13
                nn.ReLU(),
                nn.Conv2d(384, 256, 3, padding=1),            # 13→13
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2),                    # 13→6
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256*6*6, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    alexnet = AlexNet()
    total, _ = count_params(alexnet)
    print(f"\n  AlexNet total params: {total:,}")
    print("  Done: G4 LeNet-5 & AlexNet")


# ═════════════════════════════════════════════════════════════════════════════
# G5 — VGG & INCEPTION
# ═════════════════════════════════════════════════════════════════════════════

def section_G5():
    print("\n── G5: VGG & Inception ──")

    # --- 3x3 vs 5x5 cost comparison ---
    """
    Two 3x3 convs have same RF as one 5x5, but fewer params:
      1x 5x5 conv:   5*5*C*C     = 25*C^2
      2x 3x3 convs:  2*(3*3*C*C) = 18*C^2
      Saving: 28% fewer parameters
    """
    C = 256
    cost_5x5 = 5*5*C*C
    cost_2x3x3 = 2*(3*3*C*C)
    print(f"  5x5 conv C={C}: {cost_5x5:,} params")
    print(f"  2x 3x3 conv:   {cost_2x3x3:,} params  ({100*(1-cost_2x3x3/cost_5x5):.0f}% fewer)")

    # VGG block
    def vgg_block(in_channels, out_channels, n_convs=2):
        layers = []
        for i in range(n_convs):
            in_c = in_channels if i == 0 else out_channels
            layers += [nn.Conv2d(in_c, out_channels, 3, padding=1),
                       nn.ReLU()]
        layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    block = vgg_block(64, 128, n_convs=2)
    total, _ = count_params(block)
    print(f"\n  VGG block (64→128, 2 convs + pool): {total:,} params")

    # --- Inception module with 1x1 bottleneck ---
    """
    Naive Inception: branch with 5x5 conv from 128ch.
      Cost: 5*5 * 128 * 32 = 819,200

    Inception with 1x1 bottleneck (32 intermediate ch):
      1x1 conv:  1*1 * 128 * 16 =  2,048
      5x5 conv:  5*5 * 16  * 32 = 12,800
      Total: 14,848  → 98% cheaper
    """

    class InceptionModule(nn.Module):
        def __init__(self, in_ch, ch1x1, ch3x3_reduce, ch3x3,
                     ch5x5_reduce, ch5x5, ch_pool):
            super().__init__()
            # Branch 1: 1x1 conv
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, ch1x1, 1), nn.ReLU())
            # Branch 2: 1x1 reduce → 3x3
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_ch, ch3x3_reduce, 1), nn.ReLU(),
                nn.Conv2d(ch3x3_reduce, ch3x3, 3, padding=1), nn.ReLU())
            # Branch 3: 1x1 reduce → 5x5
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_ch, ch5x5_reduce, 1), nn.ReLU(),
                nn.Conv2d(ch5x5_reduce, ch5x5, 5, padding=2), nn.ReLU())
            # Branch 4: MaxPool → 1x1
            self.branch4 = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_ch, ch_pool, 1), nn.ReLU())

        def forward(self, x):
            return torch.cat([self.branch1(x), self.branch2(x),
                               self.branch3(x), self.branch4(x)], dim=1)

    inception = InceptionModule(192, 64, 96, 128, 16, 32, 32)
    total_out = 64 + 128 + 32 + 32   # concatenated output channels
    total, _ = count_params(inception)
    print(f"\n  Inception module (192 in → {total_out} out): {total:,} params")

    # Cost analysis
    naive_5x5_cost = 5*5*192*32
    bottleneck_cost = 1*1*192*16 + 5*5*16*32
    print(f"  Naive 5x5 branch:      {naive_5x5_cost:,} ops")
    print(f"  Bottleneck 5x5 branch: {bottleneck_cost:,} ops  "
          f"({100*(1-bottleneck_cost/naive_5x5_cost):.0f}% cheaper)")
    print("  Done: G5 VGG & Inception")


# ═════════════════════════════════════════════════════════════════════════════
# G6 — RESNET
# ═════════════════════════════════════════════════════════════════════════════

def section_G6():
    print("\n── G6: ResNet ──")

    # --- Basic Residual Block ---
    class BasicBlock(nn.Module):
        """
        ResNet basic block (used in ResNet-18/34).
        y = F(x, {Wi}) + x
        F = Conv-BN-ReLU-Conv-BN
        Gradient: dL/dx = dL/dy * (dF/dx + I)
        The '+I' identity term prevents vanishing gradients.
        """
        expansion = 1

        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)

            # Shortcut: 1x1 conv to match dimensions when stride>1 or channels change
            self.shortcut = nn.Sequential()
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)     # residual connection
            return F.relu(out)

    # --- Bottleneck Block ---
    class BottleneckBlock(nn.Module):
        """
        ResNet bottleneck (used in ResNet-50/101/152).
        1x1 → 3x3 → 1x1 (expand channels by 4x at output)
        Saves computation by reducing channels before 3x3.
        """
        expansion = 4

        def __init__(self, in_ch, mid_ch, stride=1):
            super().__init__()
            out_ch = mid_ch * self.expansion
            self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
            self.bn1   = nn.BatchNorm2d(mid_ch)
            self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(mid_ch)
            self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
            self.bn3   = nn.BatchNorm2d(out_ch)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out = out + self.shortcut(x)
            return F.relu(out)

    # --- ResNet-18 ---
    class ResNet18(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
            self.layer1 = nn.Sequential(BasicBlock(64,  64),  BasicBlock(64,  64))
            self.layer2 = nn.Sequential(BasicBlock(64,  128, stride=2), BasicBlock(128, 128))
            self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
            self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))
            self.pool   = nn.AdaptiveAvgPool2d((1,1))
            self.fc     = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x); x = self.layer2(x)
            x = self.layer3(x); x = self.layer4(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    resnet18 = ResNet18()
    total, _ = count_params(resnet18)
    print(f"  ResNet-18 total params: {total:,}")

    # Forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = resnet18(x)
    print(f"  Input: {tuple(x.shape)} → Output: {tuple(out.shape)}")

    # Param counts per block type
    bb = BasicBlock(64, 64)
    bn = BottleneckBlock(64, 64)
    bb_total, _ = count_params(bb)
    bn_total, _ = count_params(bn)
    print(f"\n  BasicBlock(64,64)      params: {bb_total:,}")
    print(f"  BottleneckBlock(64,64) params: {bn_total:,}")

    # Architecture table
    print(f"\n  ResNet variants:")
    print(f"    {'Model':<15} {'Blocks':<25} {'Params':>12}")
    print(f"    {'-'*55}")
    variants = [
        ("ResNet-18",  "[2,2,2,2] Basic",      "11.2M"),
        ("ResNet-34",  "[3,4,6,3] Basic",      "21.8M"),
        ("ResNet-50",  "[3,4,6,3] Bottleneck", "25.6M"),
        ("ResNet-101", "[3,4,23,3] Bottleneck","44.5M"),
        ("ResNet-152", "[3,8,36,3] Bottleneck","60.2M"),
    ]
    for name, blocks, params in variants:
        print(f"    {name:<15} {blocks:<25} {params:>12}")
    print("  Done: G6 ResNet")


# ═════════════════════════════════════════════════════════════════════════════
# G7 — MOBILENETV2 (DEPTHWISE SEPARABLE CONVOLUTIONS)
# ═════════════════════════════════════════════════════════════════════════════

def section_G7():
    print("\n── G7: MobileNetV2 — Depthwise Separable Conv ──")

    """
    Depthwise Separable Conv = Depthwise Conv + Pointwise (1x1) Conv
    Depthwise: 1 filter per input channel (groups=C_in)
    Pointwise: 1x1 conv to mix channels

    Cost comparison vs standard conv:
      Standard:    K*K*C_in*C_out
      DW + PW:     K*K*C_in + C_in*C_out
      Ratio:       1/C_out + 1/K^2

    For K=3, C_out=256:
      Ratio = 1/256 + 1/9 ≈ 0.115  →  ~8.7x cheaper
    """

    def cost_standard(C_in, C_out, K):
        return K * K * C_in * C_out

    def cost_dw_sep(C_in, C_out, K):
        dw = K * K * C_in           # depthwise
        pw = C_in * C_out           # pointwise (1x1)
        return dw + pw

    for C_in, C_out, K in [(3,32,3), (64,128,3), (256,256,3)]:
        std = cost_standard(C_in, C_out, K)
        dws = cost_dw_sep(C_in, C_out, K)
        print(f"  ({C_in}→{C_out}, K={K}): standard={std:,}  DW-Sep={dws:,}  "
              f"speedup={std/dws:.1f}x")

    # --- Inverted Residual Block (MobileNetV2) ---
    class InvertedResidual(nn.Module):
        """
        MobileNetV2 block:
        1x1 expand → Depthwise 3x3 → 1x1 project (no ReLU on last 1x1)
        Expansion factor t: temporarily widens channels before DW conv.
        Residual added when stride=1 and in_ch==out_ch.
        """
        def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
            super().__init__()
            mid_ch = in_ch * expand_ratio
            self.use_res = (stride == 1 and in_ch == out_ch)
            layers = []
            if expand_ratio != 1:
                layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                           nn.BatchNorm2d(mid_ch), nn.ReLU6()]
            layers += [
                nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                          padding=1, groups=mid_ch, bias=False),  # depthwise
                nn.BatchNorm2d(mid_ch), nn.ReLU6(),
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),          # pointwise
                nn.BatchNorm2d(out_ch)                             # no ReLU
            ]
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            if self.use_res:
                return x + self.conv(x)
            return self.conv(x)

    block = InvertedResidual(32, 16, stride=1, expand_ratio=6)
    total, _ = count_params(block)
    x = torch.randn(1, 32, 28, 28)
    with torch.no_grad():
        out = block(x)
    print(f"\n  InvertedResidual(32→16, t=6): params={total:,}  "
          f"in={tuple(x.shape)} out={tuple(out.shape)}")

    # Architecture comparison table
    print(f"\n  Efficient CNN architecture comparison:")
    print(f"    {'Model':<20} {'Params':>10} {'Top-1 (ImageNet)':>18} {'Key idea'}")
    print(f"    {'-'*75}")
    models = [
        ("VGG-16",       "138M",  "71.5%", "Deep 3x3 stacks"),
        ("ResNet-50",    "25.6M", "76.1%", "Residual connections"),
        ("MobileNetV2",  "3.4M",  "72.0%", "Depthwise + inverted residual"),
        ("EfficientNet-B0","5.3M","77.1%", "Compound scaling"),
        ("ConvNeXt-T",   "28M",   "82.1%", "Modernised ResNet"),
        ("ViT-B/16",     "86M",   "81.8%", "Pure attention, no conv"),
    ]
    for name, params, top1, idea in models:
        print(f"    {name:<20} {params:>10} {top1:>18}  {idea}")
    print("  Done: G7 MobileNetV2 & comparison")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(f"CV Section G — CNNs  (device: {DEVICE})")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    section_G1()   # conv mechanics, weight sharing, 1x1 conv
    section_G2()   # max/avg/global/adaptive pooling
    section_G3()   # receptive field formula, dilated convs
    section_G4()   # LeNet-5 (60K params) + AlexNet (60M params)
    section_G5()   # VGG 3x3 rationale, Inception bottleneck
    section_G6()   # ResNet-18 from scratch, residual block, bottleneck
    section_G7()   # MobileNetV2 depthwise separable, inverted residual

    print("\n✓ All Section G demos complete.")
