"""
cv_H_training.py
=================
Computer Vision Course — Section H: Training Large Models

Topics covered:
  H1 - Full PyTorch training loop with all best practices
  H2 - Learning rate schedules: StepLR, CosineAnnealing, OneCycleLR, warmup
  H3 - Batch size scaling: linear scaling rule, gradient accumulation
  H4 - Mixed precision training: AMP with autocast + GradScaler
  H5 - Gradient clipping: norm clipping, value clipping
  H6 - Data augmentation: standard stack, Mixup, CutMix
  H7 - Training diagnostics: loss curves, gradient norm monitoring
  H8 - Distributed training: DDP setup template (single-machine)

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
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_tiny_dataset(n=512, C=3, H=32, W=32, num_classes=10):
    """Synthetic image dataset for training demos."""
    X = torch.randn(n, C, H, W)
    y = torch.randint(0, num_classes, (n,))
    return TensorDataset(X, y)


def make_tiny_model(num_classes=10):
    """Tiny CNN for training demos."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),                                      # 32→16
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64*4*4, 128), nn.ReLU(),
        nn.Linear(128, num_classes)
    )


# ═════════════════════════════════════════════════════════════════════════════
# H1 — FULL PYTORCH TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def section_H1():
    print("\n── H1: Full PyTorch Training Loop ──")

    dataset    = make_tiny_dataset(n=256)
    val_dataset= make_tiny_dataset(n=64)
    train_loader = DataLoader(dataset,     batch_size=32, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model     = make_tiny_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=5)

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(5):
        # ── Training phase ──────────────────────────────────────────────────
        model.train()                        # enables dropout, BN uses batch stats
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimiser.zero_grad()            # clear accumulated gradients

            logits = model(images)           # forward pass
            loss   = criterion(logits, labels)  # compute loss

            loss.backward()                  # backward pass (compute gradients)
            optimiser.step()                 # update weights

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ── Validation phase ────────────────────────────────────────────────
        model.eval()                         # disables dropout, BN uses running stats
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():                # no gradient computation needed
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(images)
                val_loss += criterion(logits, labels).item()
                preds  = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc  = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        current_lr = optimiser.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/5: train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  lr={current_lr:.6f}")

    # Save checkpoint
    checkpoint = {
        'epoch':       5,
        'model_state': model.state_dict(),
        'optim_state': optimiser.state_dict(),
        'val_loss':    val_losses[-1],
    }
    torch.save(checkpoint, '/tmp/tiny_model_ckpt.pth')
    print(f"  Checkpoint saved: {list(checkpoint.keys())}")

    # Load checkpoint
    loaded = torch.load('/tmp/tiny_model_ckpt.pth', map_location=DEVICE, weights_only=True)
    model2 = make_tiny_model().to(DEVICE)
    model2.load_state_dict(loaded['model_state'])
    print(f"  Checkpoint loaded successfully. Epoch: {loaded['epoch']}")
    print("  Done: H1 training loop")


# ═════════════════════════════════════════════════════════════════════════════
# H2 — LEARNING RATE SCHEDULES
# ═════════════════════════════════════════════════════════════════════════════

def section_H2():
    print("\n── H2: Learning Rate Schedules ──")

    model = make_tiny_model()
    n_epochs = 30
    base_lr  = 0.1

    schedules = {}

    # 1. StepLR: multiply by gamma every step_size epochs
    opt = torch.optim.SGD(model.parameters(), lr=base_lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    lrs = []
    for _ in range(n_epochs):
        lrs.append(opt.param_groups[0]['lr'])
        sch.step()
    schedules["StepLR (gamma=0.1, step=10)"] = lrs

    # 2. MultiStepLR
    opt = torch.optim.SGD(model.parameters(), lr=base_lr)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10,20], gamma=0.1)
    lrs = []
    for _ in range(n_epochs):
        lrs.append(opt.param_groups[0]['lr'])
        sch.step()
    schedules["MultiStepLR [10,20]"] = lrs

    # 3. Cosine Annealing
    opt = torch.optim.SGD(model.parameters(), lr=base_lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)
    lrs = []
    for _ in range(n_epochs):
        lrs.append(opt.param_groups[0]['lr'])
        sch.step()
    schedules["CosineAnnealing"] = lrs

    # 4. Linear warmup + cosine decay (manual)
    def warmup_cosine_schedule(epoch, warmup_epochs=5, total_epochs=30,
                                base_lr=0.1, min_lr=1e-4):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs        # linear warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))

    lrs = [warmup_cosine_schedule(e) for e in range(n_epochs)]
    schedules["Warmup(5) + Cosine"] = lrs

    # 5. OneCycleLR
    opt    = torch.optim.SGD(model.parameters(), lr=base_lr)
    loader = DataLoader(make_tiny_dataset(256), batch_size=32)
    sch    = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=base_lr, steps_per_epoch=len(loader), epochs=n_epochs)
    lrs = []
    for _ in range(n_epochs * len(loader)):
        lrs.append(opt.param_groups[0]['lr'])
        sch.step()
    # Downsample to n_epochs points for plotting
    lrs_downsampled = lrs[::len(loader)][:n_epochs]
    schedules["OneCycleLR"] = lrs_downsampled

    # Print summary
    for name, lrs in schedules.items():
        print(f"  {name:<30}: start={lrs[0]:.4f} mid={lrs[n_epochs//2]:.4f} end={lrs[-1]:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, lrs in schedules.items():
        ax.plot(lrs, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedules")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("H2_lr_schedules.png", dpi=100)
    plt.close()
    print("  Saved: H2_lr_schedules.png")
    print("  Done: H2 learning rate schedules")


# ═════════════════════════════════════════════════════════════════════════════
# H3 — BATCH SIZE & GRADIENT ACCUMULATION
# ═════════════════════════════════════════════════════════════════════════════

def section_H3():
    print("\n── H3: Batch Size & Gradient Accumulation ──")

    """
    Linear Scaling Rule (Goyal et al. 2017):
      When multiplying batch size by k, multiply LR by k.
      Reasoning: k× larger batch → each step sees k× more gradients →
      equivalent to k× smaller learning rate with the original batch.

    Gradient Accumulation:
      When GPU memory limits batch size, accumulate gradients over
      accum_steps mini-batches before doing one optimiser step.
      Effective batch size = mini_batch_size * accum_steps
    """

    # Linear scaling rule demo
    base_bs = 32
    base_lr = 0.1
    print(f"  Linear scaling rule (base: bs={base_bs}, lr={base_lr}):")
    for factor in [1, 2, 4, 8, 16]:
        bs = base_bs * factor
        lr = base_lr * factor
        print(f"    bs={bs:4d} → lr={lr:.3f}")

    # --- Gradient accumulation ---
    model     = make_tiny_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = make_tiny_dataset(n=128)
    # Small mini-batches (simulating GPU memory constraint)
    mini_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    accum_steps = 4        # accumulate 4 steps → effective batch = 8*4 = 32
    effective_bs = 8 * accum_steps
    print(f"\n  Gradient accumulation: mini_bs=8, accum_steps={accum_steps}, "
          f"effective_bs={effective_bs}")

    model.train()
    optimiser.zero_grad()
    total_loss = 0.0
    n_updates  = 0

    for step, (images, labels) in enumerate(mini_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # IMPORTANT: divide loss by accum_steps to normalise gradients
        logits = model(images)
        loss   = criterion(logits, labels) / accum_steps
        loss.backward()                    # gradients ACCUMULATE in .grad

        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            optimiser.step()               # one weight update every accum_steps
            optimiser.zero_grad()
            n_updates += 1

    print(f"  Completed: {step+1} mini-batches, {n_updates} actual weight updates")
    print(f"  Average loss: {total_loss / (step+1):.4f}")
    print("  Done: H3 gradient accumulation")


# ═════════════════════════════════════════════════════════════════════════════
# H4 — MIXED PRECISION TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def section_H4():
    print("\n── H4: Mixed Precision Training (AMP) ──")

    """
    Automatic Mixed Precision (AMP):
    - Forward pass in FP16 (faster, less memory)
    - Loss scaling to prevent FP16 underflow
    - Gradients unscaled, params updated in FP32

    FP32: 1 sign + 8 exp + 23 mantissa = 32 bits. Range: ~1e-38 to 3.4e+38
    FP16: 1 sign + 5 exp + 10 mantissa = 16 bits. Range: ~6e-8  to 65504
    BF16: 1 sign + 8 exp + 7  mantissa = 16 bits. Same range as FP32, less precision.

    GradScaler:
      scale loss before backward → prevent FP16 gradient underflow
      unscale before clip/step   → recover true gradients
      update scale for next step → dynamic adjustment
    """

    model     = make_tiny_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    dataset    = make_tiny_dataset(n=128)
    loader     = DataLoader(dataset, batch_size=32)

    model.train()
    for step, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # --- Standard AMP training loop ---
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            # Forward pass runs in FP16 on CUDA, FP32 on CPU
            logits = model(images)
            loss   = criterion(logits, labels)

        optimiser.zero_grad()
        scaler.scale(loss).backward()      # backward with scaled loss
        scaler.unscale_(optimiser)         # unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)             # update (skips if inf/NaN)
        scaler.update()                    # adjust scale factor

    print(f"  AMP training: {step+1} steps completed")
    print(f"  GradScaler scale: {scaler.get_scale():.1f}")

    # dtype check
    for name, param in list(model.named_parameters())[:3]:
        print(f"  Param '{name}': dtype={param.dtype}  (always FP32 in master weights)")

    # Memory comparison
    x_fp32 = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    x_fp16 = x_fp32.half()
    x_bf16 = x_fp32.to(torch.bfloat16)
    print(f"\n  Memory per 224x224x3 activation:")
    print(f"    FP32: {x_fp32.element_size() * x_fp32.numel() / 1024:.1f} KB")
    print(f"    FP16: {x_fp16.element_size() * x_fp16.numel() / 1024:.1f} KB  (2x smaller)")
    print(f"    BF16: {x_bf16.element_size() * x_bf16.numel() / 1024:.1f} KB  (2x smaller, better range)")
    print("  Done: H4 mixed precision")


# ═════════════════════════════════════════════════════════════════════════════
# H5 — GRADIENT CLIPPING
# ═════════════════════════════════════════════════════════════════════════════

def section_H5():
    print("\n── H5: Gradient Clipping ──")

    """
    Gradient norm clipping:
      g_norm = ||g||_2 = sqrt(sum of all squared gradients)
      if g_norm > max_norm:
          g = g * (max_norm / g_norm)   ← scale down to max_norm
      else:
          g unchanged

    Typical max_norm values: 0.1 (tight) to 5.0 (loose). Default: 1.0.
    Gradient value clipping: clip each gradient element to [-clip_val, +clip_val]
    """

    def manual_clip_grad_norm(parameters, max_norm):
        """Manual implementation of gradient norm clipping."""
        params = [p for p in parameters if p.grad is not None]
        total_norm = torch.sqrt(sum(p.grad.data.norm(2)**2 for p in params))
        clip_coef  = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in params:
                p.grad.data.mul_(clip_coef)
        return total_norm.item()

    model     = make_tiny_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Compute gradients
    x, y = torch.randn(16, 3, 32, 32).to(DEVICE), torch.randint(0,10,(16,)).to(DEVICE)
    loss  = criterion(model(x), y)
    loss.backward()

    # Measure before clipping
    norm_before = torch.sqrt(sum(p.grad.norm(2)**2
                                 for p in model.parameters()
                                 if p.grad is not None)).item()

    # Clip with max_norm=1.0
    norm_clipped = manual_clip_grad_norm(model.parameters(), max_norm=1.0)

    norm_after = torch.sqrt(sum(p.grad.norm(2)**2
                                for p in model.parameters()
                                if p.grad is not None)).item()

    print(f"  Gradient norm before clipping: {norm_before:.4f}")
    print(f"  Gradient norm after  clipping: {norm_after:.4f}  (capped at 1.0)")

    # Compare with PyTorch built-in
    loss.backward()   # recompute (gradients were modified above)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    norm_pytorch = torch.sqrt(sum(p.grad.norm(2)**2
                                  for p in model.parameters()
                                  if p.grad is not None)).item()
    print(f"  PyTorch clip_grad_norm: {norm_pytorch:.4f}")

    # When to use what
    print(f"\n  Clipping guidelines:")
    print(f"    max_norm=0.1: RNN/LSTM, very tight control")
    print(f"    max_norm=1.0: default for transformers")
    print(f"    max_norm=5.0: loose, only catches extreme spikes")
    print(f"    Value clip:   less common, breaks gradient direction")
    print("  Done: H5 gradient clipping")


# ═════════════════════════════════════════════════════════════════════════════
# H6 — DATA AUGMENTATION: MIXUP & CUTMIX
# ═════════════════════════════════════════════════════════════════════════════

def section_H6():
    print("\n── H6: Data Augmentation — Mixup & CutMix ──")

    # --- Mixup ---
    def mixup(x, y, alpha=0.4, num_classes=10):
        """
        Mixup augmentation (Zhang et al. 2018).
        Creates a convex combination of two training examples:
          x_mix  = lambda * x_i + (1-lambda) * x_j
          y_mix  = lambda * y_i + (1-lambda) * y_j
        lambda ~ Beta(alpha, alpha)
        Loss: lambda * CE(pred, y_i) + (1-lambda) * CE(pred, y_j)
        """
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        idx = torch.randperm(batch_size)
        x_mix = lam * x + (1 - lam) * x[idx]
        y_a   = y
        y_b   = y[idx]
        return x_mix, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # --- CutMix ---
    def rand_bbox(size, lam):
        """Generate a random bounding box with area proportion (1-lam)."""
        W, H = size[-1], size[-2]
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w//2, 0, W)
        y1 = np.clip(cy - cut_h//2, 0, H)
        x2 = np.clip(cx + cut_w//2, 0, W)
        y2 = np.clip(cy + cut_h//2, 0, H)
        return x1, y1, x2, y2

    def cutmix(x, y, alpha=1.0):
        """
        CutMix augmentation (Yun et al. 2019).
        Replaces a rectangular region of one image with a patch from another.
        Labels mixed proportionally to cut area.
          lam_adjusted = 1 - (cut_area / total_area)
        """
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        idx = torch.randperm(batch_size)
        x1, y1, x2, y2 = rand_bbox(x.size(), lam)
        x_cut = x.clone()
        x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam_adjusted = 1 - (y2-y1)*(x2-x1) / (x.size(-1)*x.size(-2))
        return x_cut, y, y[idx], lam_adjusted

    # Demo
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    x_mix, y_a, y_b, lam = mixup(x, y, alpha=0.4)
    print(f"  Mixup: lambda={lam:.3f}  "
          f"input_range=[{x.min():.2f},{x.max():.2f}]  "
          f"mixed_range=[{x_mix.min():.2f},{x_mix.max():.2f}]")

    x_cut, y_a2, y_b2, lam2 = cutmix(x, y, alpha=1.0)
    print(f"  CutMix: lambda={lam2:.3f}  labels_a={y_a2.tolist()}  labels_b={y_b2.tolist()}")

    # RandAugment simulation
    """
    RandAugment (Cubuk et al. 2020):
    - Sample N transformations from a fixed list (N=2 typical)
    - Apply each with magnitude M (M=9 typical, range 0-30)
    - Eliminates need for separate augmentation policy search
    """
    augmentation_ops = [
        "AutoContrast", "Equalize", "Rotate", "Solarize",
        "Color", "Posterize", "Contrast", "Brightness",
        "Sharpness", "ShearX", "ShearY", "TranslateX", "TranslateY"
    ]
    N, M = 2, 9
    selected = np.random.choice(augmentation_ops, size=N, replace=False)
    print(f"\n  RandAugment (N={N}, M={M}): selected ops = {selected.tolist()}")

    print("  Done: H6 data augmentation")


# ═════════════════════════════════════════════════════════════════════════════
# H7 — TRAINING DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════

def section_H7():
    print("\n── H7: Training Diagnostics ──")

    model     = make_tiny_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = make_tiny_dataset(n=256)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    train_losses = []
    grad_norms   = []
    weight_norms = []
    lrs          = []

    for epoch in range(10):
        model.train()
        epoch_loss = 0.0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()

            # Monitor gradient norm before clipping
            g_norm = torch.sqrt(sum(
                p.grad.norm(2)**2
                for p in model.parameters()
                if p.grad is not None
            )).item()
            grad_norms.append(g_norm)

            # Monitor weight norm
            w_norm = torch.sqrt(sum(
                p.data.norm(2)**2
                for p in model.parameters()
            )).item()
            weight_norms.append(w_norm)

            optimiser.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        lrs.append(optimiser.param_groups[0]['lr'])

    print(f"  Tracked over 10 epochs:")
    print(f"    Loss:        {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    print(f"    Grad norm:   mean={np.mean(grad_norms):.4f}  max={np.max(grad_norms):.4f}")
    print(f"    Weight norm: mean={np.mean(weight_norms):.4f}")

    # Diagnostic patterns
    print(f"\n  Loss curve diagnostics:")
    print(f"    Loss not decreasing  → LR too low, wrong architecture, data issue")
    print(f"    Loss diverges (NaN)  → LR too high, no gradient clipping, bad init")
    print(f"    Train<<Val loss      → overfitting → add regularisation, more data")
    print(f"    Train≈Val, both high → underfitting → larger model, more epochs, lower LR")
    print(f"    Grad norm spikes     → exploding gradients → clip, lower LR")
    print(f"    Grad norm → 0        → vanishing gradients → check init, use ResNet/BN")

    # Plot diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(train_losses); axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
    axes[1].plot(grad_norms);   axes[1].set_title("Gradient Norms"); axes[1].set_xlabel("Step")
    axes[2].plot(weight_norms); axes[2].set_title("Weight Norms");  axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("H7_diagnostics.png", dpi=100)
    plt.close()
    print("  Saved: H7_diagnostics.png")
    print("  Done: H7 training diagnostics")


# ═════════════════════════════════════════════════════════════════════════════
# H8 — DISTRIBUTED TRAINING TEMPLATE
# ═════════════════════════════════════════════════════════════════════════════

def section_H8():
    print("\n── H8: Distributed Training (DDP Template) ──")

    """
    DDP (DistributedDataParallel) — the standard for multi-GPU training.

    How it works:
    1. Each GPU gets a full copy of the model.
    2. Each GPU processes a different batch (data parallelism).
    3. After backward, gradients are all-reduced across GPUs (average).
    4. All GPU model copies stay synchronised.

    Bandwidth: each GPU sends + receives (N-1)/N * total_gradient_size.
    All-reduce ring algorithm: O(2*(N-1)/N) ≈ O(2) for large N.
    """

    ddp_template = '''
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

def setup(rank, world_size):
    """Initialise process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model   = YourModel().cuda(rank)
    model   = DDP(model, device_ids=[rank])   # wrap with DDP

    dataset = YourDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader  = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)    # shuffle differently each epoch
        for x, y in loader:
            x, y = x.cuda(rank), y.cuda(rank)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()          # all-reduce happens here automatically
            optimiser.step()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
'''

    print(f"  DDP template structure:")
    print(f"    setup()    → init_process_group, set_device")
    print(f"    DDP(model) → wraps model, hooks into backward for all-reduce")
    print(f"    DistributedSampler → ensures non-overlapping data per GPU")
    print(f"    sampler.set_epoch(epoch) → re-shuffle each epoch correctly")
    print(f"    loss.backward() → triggers all-reduce automatically")

    print(f"\n  Parallelism strategies:")
    print(f"    Data Parallel (DDP):     replicate model, split data. Standard choice.")
    print(f"    Model Parallel:          split model layers across GPUs. For huge models.")
    print(f"    Pipeline Parallel:       split model into stages, pipeline micro-batches.")
    print(f"    Tensor Parallel:         split individual tensors (e.g., Megatron-LM).")
    print(f"    ZeRO (DeepSpeed):        shard optimizer state + gradients + params.")

    # Simulate gradient all-reduce cost
    model_params = sum(p.numel() for p in make_tiny_model().parameters())
    bytes_fp32   = model_params * 4        # 4 bytes per float32
    n_gpus       = 4
    # Ring all-reduce: each GPU sends+receives 2*(N-1)/N * gradient_size
    comm_factor  = 2 * (n_gpus - 1) / n_gpus
    comm_bytes   = comm_factor * bytes_fp32
    print(f"\n  DDP communication cost estimate ({n_gpus} GPUs):")
    print(f"    Model params:       {model_params:,}")
    print(f"    Gradient size (FP32): {bytes_fp32/1024:.1f} KB")
    print(f"    All-reduce factor:    {comm_factor:.3f}")
    print(f"    Per-GPU comm bytes:   {comm_bytes/1024:.1f} KB per step")
    print("  Done: H8 distributed training template")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(f"CV Section H — Training Large Models  (device: {DEVICE})")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    section_H1()   # full training loop with checkpointing
    section_H2()   # LR schedules: step, cosine, warmup+cosine, OneCycle
    section_H3()   # linear scaling rule + gradient accumulation
    section_H4()   # AMP with autocast + GradScaler
    section_H5()   # gradient norm clipping (manual + PyTorch)
    section_H6()   # Mixup, CutMix, RandAugment
    section_H7()   # loss/grad/weight norm monitoring
    section_H8()   # DDP template + parallelism strategies

    print("\n✓ All Section H demos complete.")
