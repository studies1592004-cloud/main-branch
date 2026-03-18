"""
cv_I_transfer_learning.py
==========================
Computer Vision Course — Section I: Classification & Transfer Learning

Topics covered:
  I1 - Classification pipeline: preprocessing, top-1/top-5 accuracy
  I2 - Why transfer learning works: feature hierarchy analysis
  I3 - Feature extraction: freeze backbone, replace head, precompute features
  I4 - Fine-tuning strategies: differential LR, gradual unfreezing
  I5 - Linear probing: evaluate frozen features
  I6 - Domain adaptation: covariate shift, BN adaptation, pseudo-labels
  I7 - CLIP: contrastive training, zero-shot inference, prompt engineering

Dependencies: torch, torchvision, numpy, matplotlib, scikit-learn
Install:  pip install torch torchvision numpy matplotlib scikit-learn
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_tiny_cnn(num_classes=10, pretrained_init=True):
    """Small CNN that simulates a pretrained backbone + head."""
    backbone = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
    )
    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    return backbone, head


def make_dataset(n=200, C=3, H=32, W=32, num_classes=10, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, C, H, W)
    y = torch.randint(0, num_classes, (n,))
    return TensorDataset(X, y)


def train_one_epoch(backbone, head, loader, optimiser, criterion):
    backbone.train(); head.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimiser.zero_grad()
        feats  = backbone(x)
        logits = head(feats.flatten(1) if feats.dim() > 2 else feats)
        loss   = criterion(logits, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(backbone, head, loader):
    backbone.eval(); head.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        feats  = backbone(x)
        logits = head(feats.flatten(1) if feats.dim() > 2 else feats)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ═════════════════════════════════════════════════════════════════════════════
# I1 — CLASSIFICATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def section_I1():
    print("\n── I1: Classification Pipeline ──")

    # --- ImageNet normalisation (standard preprocessing) ---
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def preprocess(img_tensor):
        """
        Standard ImageNet preprocessing:
        1. Resize to 256, centre-crop to 224
        2. ToTensor: HWC uint8 [0,255] → CHW float32 [0,1]
        3. Normalise: (x - mean) / std  per channel
        """
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD ).view(3, 1, 1)
        return (img_tensor - mean) / std

    def denormalise(img_tensor):
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD ).view(3, 1, 1)
        return img_tensor * std + mean

    x = torch.rand(3, 224, 224)
    x_norm = preprocess(x)
    x_back = denormalise(x_norm)
    print(f"  Original range:     [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Normalised range:   [{x_norm.min():.3f}, {x_norm.max():.3f}]")
    print(f"  Round-trip error:   {(x - x_back).abs().max():.2e}")

    # --- Top-1 and Top-5 accuracy ---
    def topk_accuracy(logits, targets, k=(1, 5)):
        """
        Compute top-k accuracy.
        A prediction is correct if the true label is in the top-k predictions.
        """
        results = {}
        maxk = max(k)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                                    # (maxk, batch)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        for ki in k:
            results[f"top{ki}"] = correct[:ki].any(dim=0).float().mean().item()
        return results

    logits  = torch.randn(32, 1000)
    targets = torch.randint(0, 1000, (32,))
    accs = topk_accuracy(logits, targets, k=(1, 5))
    print(f"\n  Random baseline (1000 classes):")
    print(f"    Top-1: {accs['top1']:.3f}  (expected ~0.001)")
    print(f"    Top-5: {accs['top5']:.3f}  (expected ~0.005)")

    # Simulate a good model: true class gets high score
    logits_good = torch.randn(32, 1000) * 0.1
    for i in range(32):
        logits_good[i, targets[i]] += 5.0
    accs_good = topk_accuracy(logits_good, targets)
    print(f"  Good model (true class +5 bias):")
    print(f"    Top-1: {accs_good['top1']:.3f}  Top-5: {accs_good['top5']:.3f}")

    # --- Confusion matrix (3-class example) ---
    from sklearn.metrics import confusion_matrix
    y_true = [0,0,0,1,1,1,2,2,2,0,1,2]
    y_pred = [0,0,1,1,1,0,2,2,1,0,1,2]
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion matrix (3 classes):")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    print(f"    {cm[2]}")
    print(f"  Per-class accuracy: {cm.diagonal() / cm.sum(axis=1)}")
    print("  Done: I1 classification pipeline")


# ═════════════════════════════════════════════════════════════════════════════
# I2 — WHY TRANSFER LEARNING WORKS
# ═════════════════════════════════════════════════════════════════════════════

def section_I2():
    print("\n── I2: Why Transfer Learning Works ──")

    """
    Deep CNNs learn a hierarchy of features:
      Early layers:  low-level (edges, colours, textures) — universal, transferable
      Mid layers:    mid-level (object parts, patterns)   — somewhat domain-specific
      Late layers:   high-level (class-specific features) — task-specific

    Evidence: visualising CNN filters shows:
      Layer 1: Gabor-like edge detectors, colour blobs  (same across all CNNs)
      Layer 2: corners, T-junctions, basic textures
      Layer 3: textures, repeated patterns
      Layer 4: object parts (wheels, faces, legs)
      Layer 5+: full objects, class-discriminative features
    """

    backbone, head = make_tiny_cnn(num_classes=10)
    backbone = backbone.to(DEVICE)
    head     = head.to(DEVICE)

    # Analyse activation statistics at each layer
    dataset = make_dataset(n=64)
    loader  = DataLoader(dataset, batch_size=64)
    x_batch, _ = next(iter(loader))
    x_batch = x_batch.to(DEVICE)

    print(f"  Feature statistics through backbone layers:")
    print(f"  {'Layer':<25} {'Shape':<20} {'Mean':>8} {'Std':>8} {'Sparsity':>10}")
    print(f"  {'-'*73}")

    hooks, activations = [], {}
    layer_names = []
    def make_hook(name):
        def hook(mod, inp, out):
            activations[name] = out.detach()
        return hook

    # Register hooks on conv layers
    for name, mod in backbone.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ReLU, nn.AdaptiveAvgPool2d)):
            hooks.append(mod.register_forward_hook(make_hook(name)))
            layer_names.append(name)

    backbone.eval()
    with torch.no_grad():
        _ = backbone(x_batch)

    for name, act in list(activations.items())[:8]:
        shape = tuple(act.shape[1:])
        mean  = act.mean().item()
        std   = act.std().item()
        sparsity = (act == 0).float().mean().item()
        print(f"  {name:<25} {str(shape):<20} {mean:>8.3f} {std:>8.3f} {sparsity:>10.3f}")

    for h in hooks:
        h.remove()

    # Feature reuse across tasks
    print(f"\n  Why early features transfer well:")
    print(f"    Edge detectors appear in ALL vision tasks (medical, satellite, microscopy)")
    print(f"    Retraining them from scratch on small datasets → random/noisy weights")
    print(f"    Keeping pretrained early layers → stable, generic low-level features")
    print(f"\n  Transferability by layer (ResNet-50 trained on ImageNet):")
    print(f"    {'Layer':<10} {'Transferability':<20} {'Recommendation'}")
    print(f"    {'-'*60}")
    for layer, trans, rec in [
        ("layer1", "High (universal)",     "Always freeze"),
        ("layer2", "High (universal)",     "Freeze or low LR"),
        ("layer3", "Medium (textures)",    "Low LR 1e-5"),
        ("layer4", "Low (task-specific)",  "Normal LR 1e-4"),
        ("fc",     "None (new task)",      "Train from scratch"),
    ]:
        print(f"    {layer:<10} {trans:<20} {rec}")
    print("  Done: I2 why transfer works")


# ═════════════════════════════════════════════════════════════════════════════
# I3 — FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def section_I3():
    print("\n── I3: Feature Extraction ──")

    """
    Feature extraction: freeze all backbone weights, only train the new head.
    1. Set requires_grad=False on all backbone parameters
    2. Replace final layer(s) with new head matching target classes
    3. Only pass head parameters to optimiser

    Advantage: fast (only one small layer trains), stable, works with tiny datasets.
    Disadvantage: backbone features may not be optimal for new domain.
    """

    backbone, _ = make_tiny_cnn(num_classes=10)
    backbone = backbone.to(DEVICE)

    # Step 1: Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    print(f"  Frozen backbone params: {frozen_params:,}")

    # Step 2: New head for target task (5 classes instead of 10)
    new_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 5),
    ).to(DEVICE)

    trainable = sum(p.numel() for p in new_head.parameters())
    print(f"  Trainable head params:  {trainable:,}")
    print(f"  Ratio:                  {trainable/(frozen_params+trainable)*100:.1f}% of total")

    # Step 3: Optimiser only sees head parameters
    optimiser = torch.optim.Adam(new_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Verify backbone weights don't change
    w_before = list(backbone.parameters())[0].data.clone()

    dataset = make_dataset(n=128, num_classes=5)
    loader  = DataLoader(dataset, batch_size=32)

    backbone.eval()   # important: keep BN in eval mode when frozen
    new_head.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimiser.zero_grad()
        with torch.no_grad():
            feats = backbone(x)              # no grad through backbone
        logits = new_head(feats.flatten(1) if feats.dim() > 2 else feats)
        loss   = criterion(logits, y)
        loss.backward()
        optimiser.step()

    w_after = list(backbone.parameters())[0].data
    weight_changed = not torch.allclose(w_before, w_after)
    print(f"  Backbone weights changed: {weight_changed}  (should be False)")

    # --- Precompute features (cache backbone outputs) ---
    """
    When backbone is frozen, you can precompute all features once
    and only train the head on cached features. Much faster.
    """
    print(f"\n  Precomputing features...")
    all_feats, all_labels = [], []
    backbone.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            feats = backbone(x).cpu()
            all_feats.append(feats.flatten(1))
            all_labels.append(y)

    feats_cache  = torch.cat(all_feats)
    labels_cache = torch.cat(all_labels)
    print(f"  Cached features: {feats_cache.shape}  labels: {labels_cache.shape}")
    print(f"  Precomputing once → {len(loader)} forward passes → 1 cached pass per epoch")
    print("  Done: I3 feature extraction")


# ═════════════════════════════════════════════════════════════════════════════
# I4 — FINE-TUNING
# ═════════════════════════════════════════════════════════════════════════════

def section_I4():
    print("\n── I4: Fine-Tuning Strategies ──")

    backbone, head = make_tiny_cnn(num_classes=5)
    backbone = backbone.to(DEVICE)
    head     = head.to(DEVICE)

    # Replace last FC of head to match new task
    in_features = 128 * 4 * 4
    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 256), nn.ReLU(),
        nn.Linear(256, 5)
    ).to(DEVICE)

    dataset    = make_dataset(n=256, num_classes=5)
    val_data   = make_dataset(n=64,  num_classes=5, seed=1)
    loader     = DataLoader(dataset,  batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    criterion  = nn.CrossEntropyLoss()

    # ── Strategy 1: Head-only (freeze backbone completely) ───────────────────
    for p in backbone.parameters():
        p.requires_grad = False
    opt1 = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(3):
        train_one_epoch(backbone, head, loader, opt1, criterion)
    acc1 = evaluate(backbone, head, val_loader)
    print(f"  Strategy 1 (head-only):          val_acc={acc1:.3f}")

    # ── Strategy 2: Full fine-tuning (unfreeze everything) ───────────────────
    for p in backbone.parameters():
        p.requires_grad = True
    opt2 = torch.optim.Adam([
        {'params': backbone.parameters(), 'lr': 1e-4},
        {'params': head.parameters(),     'lr': 1e-3},
    ])
    for _ in range(3):
        train_one_epoch(backbone, head, loader, opt2, criterion)
    acc2 = evaluate(backbone, head, val_loader)
    print(f"  Strategy 2 (full fine-tune):     val_acc={acc2:.3f}")

    # ── Strategy 3: Differential LR (different LR per layer group) ──────────
    """
    Differential LR rationale:
    - Early layers: pretrained features are good → very small LR (don't destroy them)
    - Late layers:  more task-specific → larger LR
    - New head:     random init → largest LR

    Typical values (ImageNet pretrained ResNet):
      layer1: lr = base_lr * 0.01   (1e-5 if base=1e-3)
      layer2: lr = base_lr * 0.05
      layer3: lr = base_lr * 0.1
      layer4: lr = base_lr * 0.5
      head:   lr = base_lr * 1.0    (1e-3)
    """
    named_layers = list(backbone.named_children())
    base_lr = 1e-3
    lr_multipliers = np.linspace(0.01, 0.5, len(named_layers))

    param_groups = []
    for (name, layer), mult in zip(named_layers, lr_multipliers):
        param_groups.append({'params': layer.parameters(), 'lr': base_lr * mult})
        print(f"    {name:<15}: lr = {base_lr * mult:.2e}")
    param_groups.append({'params': head.parameters(), 'lr': base_lr})
    print(f"    {'head':<15}: lr = {base_lr:.2e}")

    opt3 = torch.optim.Adam(param_groups)
    for _ in range(3):
        train_one_epoch(backbone, head, loader, opt3, criterion)
    acc3 = evaluate(backbone, head, val_loader)
    print(f"  Strategy 3 (differential LR):   val_acc={acc3:.3f}")

    # Gradual unfreezing
    print(f"\n  Gradual unfreezing schedule:")
    print(f"    Epoch 1-3:  train head only (backbone frozen)")
    print(f"    Epoch 4-6:  unfreeze last backbone block + head")
    print(f"    Epoch 7-9:  unfreeze last 2 backbone blocks + head")
    print(f"    Epoch 10+:  unfreeze all (full fine-tuning)")
    print(f"  Advantage: head first learns good features, then backbone adapts gently")
    print("  Done: I4 fine-tuning")


# ═════════════════════════════════════════════════════════════════════════════
# I5 — LINEAR PROBING
# ═════════════════════════════════════════════════════════════════════════════

def section_I5():
    print("\n── I5: Linear Probing ──")

    """
    Linear probing: freeze the entire backbone, train ONLY a single linear layer.
    Logistic regression on frozen features = linear probe.

    Why use it:
    - Strict evaluation of feature quality (no fine-tuning can compensate)
    - Standard evaluation protocol for self-supervised learning (SimCLR, DINO, etc.)
    - Fast: features computed once, then sklearn LogisticRegression

    Rule of thumb:
    - Good pretrained features: linear probe >> random init
    - Self-supervised claim valid only if linear probe shows good accuracy
    """

    backbone, _ = make_tiny_cnn(num_classes=10)
    backbone = backbone.to(DEVICE)
    backbone.eval()

    # Extract features
    def extract_features(backbone, dataset):
        loader = DataLoader(dataset, batch_size=64)
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                f = backbone(x).cpu().numpy().reshape(len(x), -1)
                feats.append(f); labels.append(y.numpy())
        return np.vstack(feats), np.concatenate(labels)

    train_ds = make_dataset(n=400, num_classes=10)
    val_ds   = make_dataset(n=100, num_classes=10, seed=99)

    X_train, y_train = extract_features(backbone, train_ds)
    X_val,   y_val   = extract_features(backbone, val_ds)

    print(f"  Feature shape: {X_train.shape}  (n_samples, feature_dim)")

    # Linear probe with sklearn (normalise features first)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_train_s, y_train)
    acc = accuracy_score(y_val, clf.predict(X_val_s))
    print(f"  Linear probe accuracy (random init backbone): {acc:.3f}")
    print(f"  Chance level (10 classes):                    {1/10:.3f}")

    # Compare: linear probe vs full fine-tuning vs head-only
    print(f"\n  Comparison of evaluation strategies:")
    print(f"  {'Strategy':<30} {'Notes'}")
    print(f"  {'-'*70}")
    rows = [
        ("Linear probe",           "1 linear layer, frozen backbone. Pure feature quality."),
        ("Feature extract + MLP",  "2-3 layer head, frozen backbone. Easy to overfit."),
        ("Partial fine-tuning",    "Unfreeze last N layers. Balance of speed/accuracy."),
        ("Full fine-tuning",       "All layers train. Best accuracy, risk of overfitting."),
        ("Training from scratch",  "No pretrain. Needs large dataset. Slow."),
    ]
    for name, note in rows:
        print(f"  {name:<30} {note}")
    print("  Done: I5 linear probing")


# ═════════════════════════════════════════════════════════════════════════════
# I6 — DOMAIN ADAPTATION
# ═════════════════════════════════════════════════════════════════════════════

def section_I6():
    print("\n── I6: Domain Adaptation ──")

    """
    Types of domain shift:
      Covariate shift:  P(X) changes, P(Y|X) same. e.g., daytime→nighttime photos.
      Label shift:      P(Y) changes, P(X|Y) same. e.g., class prevalence changes.
      Concept shift:    P(Y|X) changes. e.g., "cat" means different thing in new domain.

    Methods:
      1. Supervised fine-tuning:   gold standard if target labels available
      2. BN statistics adaptation: update running mean/var on target domain (no labels needed)
      3. Pseudo-labelling:         predict labels on target, re-train on high-confidence ones
      4. DANN (adversarial):       domain classifier + gradient reversal layer
    """

    backbone, head = make_tiny_cnn(num_classes=5)
    backbone = backbone.to(DEVICE)
    head     = head.to(DEVICE)

    # Simulate source and target domains (different input statistics)
    # Source: mean=0, std=1  (e.g., daytime images)
    # Target: mean=0.5, std=0.5  (e.g., nighttime images — brighter shifted)
    source_data = make_dataset(n=200, num_classes=5, seed=0)
    target_data = TensorDataset(
        torch.randn(200, 3, 32, 32) * 0.5 + 0.5,   # shifted distribution
        torch.randint(0, 5, (200,))
    )

    # --- Method 1: BN Statistics Adaptation ---
    """
    BatchNorm layers store running_mean and running_var computed on training data.
    If test domain differs, these stats are wrong → poor performance.
    Fix: re-run the model in train mode on target domain data (no label needed)
    to update running stats. Only BN layers change; weights unchanged.
    """
    def adapt_bn(model, target_loader, n_batches=10):
        """Adapt BatchNorm statistics to target domain."""
        model.train()    # enables BN to update running stats
        with torch.no_grad():
            for i, (x, _) in enumerate(target_loader):
                if i >= n_batches:
                    break
                _ = model(x.to(DEVICE))  # forward pass updates running stats
        model.eval()

    target_loader = DataLoader(target_data, batch_size=32)
    adapt_bn(backbone, target_loader, n_batches=5)
    print(f"  BN adaptation: updated running stats on target domain (no labels needed)")

    # --- Method 2: Pseudo-labelling ---
    """
    1. Train on labelled source domain
    2. Run inference on unlabelled target domain
    3. Keep high-confidence predictions as pseudo-labels (threshold τ)
    4. Retrain on source labels + pseudo-labels
    """
    backbone.eval()
    head.eval()
    pseudo_labels = []
    pseudo_x      = []
    threshold = 0.7

    with torch.no_grad():
        for x, _ in target_loader:
            x     = x.to(DEVICE)
            feats  = backbone(x)
            logits = head(feats.flatten(1) if feats.dim() > 2 else feats)
            probs  = F.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            mask = conf > threshold
            if mask.any():
                pseudo_x.append(x[mask].cpu())
                pseudo_labels.append(preds[mask].cpu())

    if pseudo_x:
        px = torch.cat(pseudo_x)
        py = torch.cat(pseudo_labels)
        print(f"  Pseudo-labelling (τ={threshold}): {len(py)}/{len(target_data)} "
              f"samples accepted ({len(py)/len(target_data)*100:.0f}%)")
        conf_mean = probs.max(dim=1).values.mean().item()
        print(f"  Mean confidence of accepted samples: {conf_mean:.3f}")
    else:
        print(f"  Pseudo-labelling: 0 samples exceeded threshold {threshold}")

    # DANN description
    print(f"\n  DANN (Domain-Adversarial Neural Network) structure:")
    print(f"    Backbone → [Feature extractor]")
    print(f"             ↓                    ↓")
    print(f"    Label classifier        Domain classifier")
    print(f"    (minimise label loss)   (with gradient reversal layer)")
    print(f"    GRL: during backward, flip sign of gradient from domain classifier")
    print(f"    Effect: backbone learns features that are INDISTINGUISHABLE across domains")
    print("  Done: I6 domain adaptation")


# ═════════════════════════════════════════════════════════════════════════════
# I7 — CLIP: CONTRASTIVE LANGUAGE-IMAGE PRETRAINING
# ═════════════════════════════════════════════════════════════════════════════

def section_I7():
    print("\n── I7: CLIP — Contrastive Language-Image Pretraining ──")

    """
    CLIP (Radford et al. 2021):
    - Train image encoder + text encoder jointly on 400M (image, text) pairs
    - Loss: contrastive — maximise cosine similarity of matching pairs,
            minimise similarity of non-matching pairs (N^2 - N pairs per batch)
    - At inference: compare image embedding to text embeddings of class descriptions
    - Zero-shot: no task-specific training needed

    Contrastive loss (InfoNCE / NT-Xent):
      For a batch of N pairs, build NxN similarity matrix S:
        S[i,j] = dot(image_i, text_j) / temperature
      Loss_image = mean CrossEntropy(S[i,:], target=i)    (for each image i)
      Loss_text  = mean CrossEntropy(S[:,j], target=j)    (for each text j)
      L = (Loss_image + Loss_text) / 2
    """

    def cosine_similarity_matrix(A, B):
        """
        Compute NxN cosine similarity matrix between rows of A and B.
        A: (N, D),  B: (N, D)  →  result: (N, N)
        """
        A_norm = F.normalize(torch.tensor(A, dtype=torch.float32), dim=1)
        B_norm = F.normalize(torch.tensor(B, dtype=torch.float32), dim=1)
        return (A_norm @ B_norm.T).numpy()

    def clip_contrastive_loss(image_embeds, text_embeds, temperature=0.07):
        """
        InfoNCE contrastive loss for a batch.
        image_embeds: (N, D)  text_embeds: (N, D)
        Both are L2-normalised.
        """
        N = len(image_embeds)
        img  = F.normalize(image_embeds, dim=1)
        txt  = F.normalize(text_embeds,  dim=1)
        logits = (img @ txt.T) / temperature   # (N, N)
        targets = torch.arange(N)              # diagonal = correct pairs
        loss_i = F.cross_entropy(logits, targets)        # image→text
        loss_t = F.cross_entropy(logits.T, targets)      # text→image
        return (loss_i + loss_t) / 2

    # Simulate embeddings
    torch.manual_seed(42)
    N, D = 8, 512

    # Perfectly aligned pairs (image[i] should match text[i])
    shared = torch.randn(N, D)
    image_embeds_good = shared + torch.randn(N, D) * 0.1
    text_embeds_good  = shared + torch.randn(N, D) * 0.1

    # Random (unaligned) embeddings
    image_embeds_rand = torch.randn(N, D)
    text_embeds_rand  = torch.randn(N, D)

    loss_good = clip_contrastive_loss(image_embeds_good, text_embeds_good)
    loss_rand = clip_contrastive_loss(image_embeds_rand, text_embeds_rand)
    print(f"  Contrastive loss (aligned pairs): {loss_good:.4f}")
    print(f"  Contrastive loss (random):        {loss_rand:.4f}")
    print(f"  Minimum possible loss:            {np.log(1):.4f}  (perfect alignment)")
    print(f"  Random baseline (N={N}):          {np.log(N):.4f}")

    # --- Zero-shot inference simulation ---
    """
    Zero-shot CLIP inference:
    1. Encode image → image_embedding
    2. Encode class descriptions → text_embeddings
       e.g., "a photo of a cat", "a photo of a dog", ...
    3. Predict class = argmax cosine_similarity(image_embed, text_embeds)
    """
    classes = ["cat", "dog", "car", "airplane", "bird"]
    prompts = [f"a photo of a {c}" for c in classes]
    print(f"\n  Zero-shot prompts:")
    for p in prompts:
        print(f"    '{p}'")

    # Simulate: image of a cat → similar to "cat" text embedding
    np.random.seed(7)
    D = 128
    # Simulated class text embeddings
    text_embeds_cls = np.random.randn(len(classes), D)
    text_embeds_cls /= np.linalg.norm(text_embeds_cls, axis=1, keepdims=True)

    # Image embedding close to class 0 (cat)
    img_embed = text_embeds_cls[0] + np.random.randn(D) * 0.2
    img_embed /= np.linalg.norm(img_embed)

    sims = text_embeds_cls @ img_embed
    pred_idx = sims.argmax()
    print(f"\n  Zero-shot prediction: '{classes[pred_idx]}' (similarity={sims[pred_idx]:.3f})")
    print(f"  All similarities: {dict(zip(classes, sims.round(3)))}")

    # Prompt engineering
    print(f"\n  Prompt engineering examples:")
    prompt_variants = [
        "a photo of a {}",
        "a blurry photo of a {}",
        "a photo of the small {}",
        "a {} in the wild",
        "art of a {}",
    ]
    for tmpl in prompt_variants:
        print(f"    '{tmpl.format('cat')}'")
    print(f"  Ensemble: average text embeddings across prompt variants → +1-3% accuracy")

    # ImageNet zero-shot accuracy
    print(f"\n  CLIP ImageNet benchmark:")
    print(f"    CLIP ViT-L/14:    76.2% top-1 (zero-shot, no ImageNet training)")
    print(f"    ResNet-50:        76.1% top-1 (fully supervised on ImageNet)")
    print(f"    Linear probe:     85.4% (CLIP features + linear head, still no ImageNet fine-tune)")
    print("  Done: I7 CLIP")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(f"CV Section I — Classification & Transfer Learning  ({DEVICE})")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    section_I1()   # preprocessing, top-k accuracy, confusion matrix
    section_I2()   # feature hierarchy, hook analysis
    section_I3()   # feature extraction, freeze, precompute cache
    section_I4()   # head-only, full fine-tune, differential LR, unfreezing
    section_I5()   # linear probing with sklearn LogisticRegression
    section_I6()   # BN adaptation, pseudo-labelling, DANN
    section_I7()   # CLIP contrastive loss, zero-shot, prompt engineering

    print("\n✓ All Section I demos complete.")