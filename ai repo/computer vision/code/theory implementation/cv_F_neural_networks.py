"""
cv_F_neural_networks.py
========================
Computer Vision Course — Section F: Neural Network Basics

Topics covered:
  F1 - Perceptron & MLP: forward pass, z = Wx + b, stacking layers
  F2 - Activation functions: sigmoid, tanh, ReLU, Leaky ReLU, GELU, Swish
  F3 - Loss functions: MSE, MAE, cross-entropy, focal loss
  F4 - Backpropagation: chain rule, manual gradient computation
  F5 - Optimisers: SGD, Momentum, Adam — all from scratch in NumPy
  F6 - Weight initialisation: Xavier, He — variance analysis
  F7 - Regularisation: L1/L2, dropout with inverted scaling
  F8 - Batch normalisation: full algorithm, train vs inference

All implementations are in pure NumPy — no PyTorch/TensorFlow.
A small XOR and MNIST-subset problem are used as tests.

Dependencies: numpy, matplotlib, scikit-learn (for data only)
Install:  pip install numpy matplotlib scikit-learn
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════════════
# F1 — PERCEPTRON & MLP FORWARD PASS
# ═════════════════════════════════════════════════════════════════════════════

def section_F1():
    print("\n── F1: Perceptron & MLP ──")

    # --- Single neuron (perceptron) ---
    # z = w^T x + b,  output = activation(z)
    def perceptron(x, w, b, activation=lambda z: (z > 0).astype(float)):
        z = np.dot(w, x) + b
        return activation(z), z

    x = np.array([0.5, -0.3, 0.8])
    w = np.array([0.2,  0.9, -0.4])
    b = 0.1
    out, z = perceptron(x, w, b)
    print(f"  Perceptron: z={z:.4f}, binary output={out}")

    # --- MLP layer: y = activation(Wx + b) ---
    class LinearLayer:
        def __init__(self, in_features, out_features):
            # He initialisation (for ReLU networks)
            self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
            self.b = np.zeros(out_features)

        def forward(self, x):
            """x: (batch, in_features)  →  (batch, out_features)"""
            self.x_cache = x
            return x @ self.W.T + self.b   # (batch, out)

    # Build a simple 3-layer MLP: 784 → 256 → 128 → 10
    layer_dims = [784, 256, 128, 10]
    layers = [LinearLayer(layer_dims[i], layer_dims[i+1])
              for i in range(len(layer_dims)-1)]

    # Count parameters
    total_params = sum(l.W.size + l.b.size for l in layers)
    print(f"  MLP 784→256→128→10 total params: {total_params:,}")
    for i, l in enumerate(layers):
        print(f"    Layer {i+1}: W={l.W.shape}  b={l.b.shape}  "
              f"params={l.W.size+l.b.size:,}")

    # Forward pass (batch of 8 samples)
    x_batch = np.random.randn(8, 784)
    out = x_batch
    for i, l in enumerate(layers):
        out = l.forward(out)
        out = np.maximum(0, out)   # ReLU (except last layer)
    print(f"  Forward pass output shape: {out.shape}  (8 samples, 10 logits)")
    print("  Done: F1 perceptron & MLP")


# ═════════════════════════════════════════════════════════════════════════════
# F2 — ACTIVATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def section_F2():
    print("\n── F2: Activation Functions ──")

    z = np.linspace(-5, 5, 300)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_grad(z):
        s = sigmoid(z)
        return s * (1 - s)                  # max 0.25 at z=0

    def tanh_act(z):
        return np.tanh(z)

    def tanh_grad(z):
        return 1 - np.tanh(z)**2            # max 1.0 at z=0

    def relu(z):
        return np.maximum(0, z)

    def relu_grad(z):
        return np.array(z > 0, dtype=float)  # 0 or 1

    def leaky_relu(z, alpha=0.01):
        return np.where(np.array(z) > 0, z, alpha * np.array(z))

    def leaky_relu_grad(z, alpha=0.01):
        return np.where(np.array(z) > 0, 1.0, alpha)

    def gelu(z):
        """GELU ≈ 0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3)))"""
        return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

    def swish(z, beta=1.0):
        return z * sigmoid(beta * z)

    activations = {
        "Sigmoid":    (sigmoid,    sigmoid_grad),
        "Tanh":       (tanh_act,   tanh_grad),
        "ReLU":       (relu,       relu_grad),
        "Leaky ReLU": (leaky_relu, leaky_relu_grad),
        "GELU":       (gelu,       None),
        "Swish":      (swish,      None),
    }

    print(f"  {'Activation':<12} {'f(0)':>8} {'f(-2)':>8} {'f(2)':>8}  {'grad(0)':>8}")
    print(f"  {'-'*52}")
    for name, (fn, gn) in activations.items():
        f0 = fn(0.0)
        fm = fn(-2.0)
        fp = fn(2.0)
        g0 = gn(0.0) if gn else float("nan")
        print(f"  {name:<12} {f0:>8.4f} {fm:>8.4f} {fp:>8.4f}  {g0:>8.4f}")

    # Vanishing gradient with sigmoid: after 10 layers, gradient ~ 0.25^10
    sigmoid_grad_chain = 0.25 ** 10
    print(f"\n  Sigmoid gradient after 10 layers (worst case): {sigmoid_grad_chain:.2e}")
    print(f"  → ReLU gradient after 10 layers (best case):  1.0 (no vanishing)")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (name, (fn, gn)) in zip(axes.flatten(), activations.items()):
        ax.plot(z, fn(z), 'b-', lw=2, label="f(z)")
        if gn:
            ax.plot(z, gn(z), 'r--', lw=1.5, label="f'(z)")
        ax.axhline(0, c='k', lw=0.5)
        ax.axvline(0, c='k', lw=0.5)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(-2, 3)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("F2_activations.png", dpi=100)
    plt.close()
    print("  Saved: F2_activations.png")
    print("  Done: F2 activation functions")


# ═════════════════════════════════════════════════════════════════════════════
# F3 — LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def section_F3():
    print("\n── F3: Loss Functions ──")

    # --- Regression losses ---
    def mse_loss(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def mae_loss(y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    def huber_loss(y_pred, y_true, delta=1.0):
        diff = np.abs(y_pred - y_true)
        return np.mean(np.where(diff < delta,
                                0.5 * diff**2,
                                delta * (diff - 0.5 * delta)))

    # --- Classification losses ---
    def softmax(logits):
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def cross_entropy(logits, y_true_idx):
        """
        Cross-entropy for multiclass:
        L = -log(p_true_class)
        p = softmax(logits)
        """
        probs = softmax(logits)
        n = len(y_true_idx)
        # Gather probabilities of true classes
        p_true = probs[np.arange(n), y_true_idx]
        return -np.mean(np.log(p_true + 1e-8))

    def binary_cross_entropy(p_pred, y_true):
        p = np.clip(p_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def focal_loss(p_pred, y_true, gamma=2.0, alpha=0.25):
        """
        Focal Loss: FL = -alpha * (1 - p_t)^gamma * log(p_t)
        Down-weights easy examples (high p_t) by (1-p_t)^gamma.
        gamma=0: standard BCE.  gamma=2: easy examples get ~100x less weight.
        """
        p = np.clip(p_pred, 1e-8, 1 - 1e-8)
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
        return -np.mean(alpha_t * (1 - p_t)**gamma * np.log(p_t))

    # Test values
    y_true = np.array([0, 1, 1, 0, 1], dtype=float)
    y_pred_good = np.array([0.05, 0.95, 0.90, 0.10, 0.85])
    y_pred_bad  = np.array([0.60, 0.40, 0.45, 0.70, 0.35])

    print(f"  BCE (good predictions): {binary_cross_entropy(y_pred_good, y_true):.4f}")
    print(f"  BCE (bad  predictions): {binary_cross_entropy(y_pred_bad,  y_true):.4f}")
    print(f"  Focal γ=2 (good):       {focal_loss(y_pred_good, y_true, gamma=2):.4f}")
    print(f"  Focal γ=2 (bad):        {focal_loss(y_pred_bad,  y_true, gamma=2):.4f}")

    # Focal loss effect: easy vs hard example
    p_easy = 0.95    # model is confident and correct
    p_hard = 0.55    # model is uncertain
    for gamma in [0, 1, 2, 5]:
        w_easy = (1 - p_easy) ** gamma
        w_hard = (1 - p_hard) ** gamma
        print(f"  gamma={gamma}: easy weight={w_easy:.4f}  hard weight={w_hard:.4f}  "
              f"ratio={w_hard/w_easy:.1f}x")

    # Multiclass cross-entropy
    batch_logits = np.random.randn(16, 10)   # 16 samples, 10 classes
    labels = np.random.randint(0, 10, 16)
    ce = cross_entropy(batch_logits, labels)
    print(f"\n  Cross-entropy (random 10-class): {ce:.4f}  (expected ~log(10)={np.log(10):.4f})")

    print("  Done: F3 loss functions")


# ═════════════════════════════════════════════════════════════════════════════
# F4 — BACKPROPAGATION
# ═════════════════════════════════════════════════════════════════════════════

def section_F4():
    print("\n── F4: Backpropagation ──")

    """
    Manual backpropagation through a 2-layer MLP:
      Input x → Linear(2,4) → ReLU → Linear(4,1) → Sigmoid → BCE loss

    Chain rule: dL/dW1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dW1
    """

    # ── Forward pass layers ──────────────────────────────────────────────────
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def relu(z):
        return np.maximum(0, z)

    def bce(p, y):
        p = np.clip(p, 1e-8, 1-1e-8)
        return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

    # Network weights (small 2→4→1 net)
    W1 = np.random.randn(4, 2) * 0.5
    b1 = np.zeros(4)
    W2 = np.random.randn(1, 4) * 0.5
    b2 = np.zeros(1)

    def forward(x, y):
        """Forward pass; returns loss and all intermediate values."""
        z1 = x @ W1.T + b1          # (N, 4)
        a1 = relu(z1)                # (N, 4)
        z2 = a1 @ W2.T + b2         # (N, 1)
        a2 = sigmoid(z2)             # (N, 1)
        loss = bce(a2, y.reshape(-1,1))
        return loss, z1, a1, z2, a2

    def backward(x, y, z1, a1, z2, a2):
        """
        Backward pass — manual chain rule.
        Returns gradients dW1, db1, dW2, db2.
        """
        n = x.shape[0]
        y = y.reshape(-1, 1)

        # dL/da2: gradient of BCE w.r.t sigmoid output
        dL_da2 = (a2 - y) / n           # (N,1)  [BCE + sigmoid combined]

        # dL/dz2 = dL/da2 * da2/dz2  (sigmoid gradient = a2*(1-a2))
        dL_dz2 = dL_da2 * a2 * (1 - a2)   # (N,1)

        # dL/dW2, dL/db2
        dW2 = dL_dz2.T @ a1             # (1,4)
        db2 = dL_dz2.sum(axis=0)        # (1,)

        # dL/da1 = dL/dz2 * dz2/da1 = dL/dz2 * W2
        dL_da1 = dL_dz2 @ W2            # (N,4)

        # dL/dz1 = dL/da1 * da1/dz1 (ReLU gradient)
        dL_dz1 = dL_da1 * (z1 > 0).astype(float)   # (N,4)

        # dL/dW1, dL/db1
        dW1 = dL_dz1.T @ x              # (4,2)
        db1 = dL_dz1.sum(axis=0)        # (4,)

        return dW1, db1, dW2, db2

    # XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    Y = np.array([0, 1, 1, 0], dtype=float)

    # Numerical gradient check
    loss0, z1,a1,z2,a2 = forward(X, Y)
    dW1, db1, dW2, db2 = backward(X, Y, z1, a1, z2, a2)

    # Check dW2[0,0] numerically
    eps = 1e-5
    W2_plus = W2.copy(); W2_plus[0,0] += eps
    W2_orig = W2.copy()
    W2[:] = W2_plus
    l_plus, *_ = forward(X, Y)
    W2_minus = W2_orig.copy(); W2_minus[0,0] -= eps
    W2[:] = W2_minus
    l_minus, *_ = forward(X, Y)
    W2[:] = W2_orig
    num_grad = (l_plus - l_minus) / (2 * eps)
    print(f"  Gradient check dW2[0,0]:")
    print(f"    Analytical: {dW2[0,0]:.6f}")
    print(f"    Numerical:  {num_grad:.6f}")
    print(f"    Relative error: {abs(dW2[0,0]-num_grad)/(abs(num_grad)+1e-8):.2e}")

    # Train XOR with manual backprop
    lr = 0.5
    losses = []
    for step in range(3000):
        loss, z1, a1, z2, a2 = forward(X, Y)
        dW1_g, db1_g, dW2_g, db2_g = backward(X, Y, z1, a1, z2, a2)
        W1 -= lr * dW1_g
        b1 -= lr * db1_g
        W2 -= lr * dW2_g
        b2 -= lr * db2_g
        if step % 300 == 0:
            losses.append(loss)

    loss_final, *_, a2_final = forward(X, Y)
    preds = (a2_final.flatten() > 0.5).astype(int)
    print(f"  XOR after 3000 steps: loss={loss_final:.4f}  "
          f"predictions={preds.tolist()}  GT={Y.astype(int).tolist()}")
    print(f"  Accuracy: {(preds == Y.astype(int)).mean()*100:.0f}%")
    print("  Done: F4 backpropagation")


# ═════════════════════════════════════════════════════════════════════════════
# F5 — OPTIMISERS
# ═════════════════════════════════════════════════════════════════════════════

def section_F5():
    print("\n── F5: Optimisers ──")

    """
    Minimise a simple 2D Rosenbrock function:
      f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    Gradient:
      df/dx = -2(1-x) - 400x(y-x^2)
      df/dy = 200(y-x^2)
    Minimum at (1,1).
    """

    def rosenbrock(params):
        x, y = params
        return (1 - x)**2 + 100 * (y - x**2)**2

    def rosenbrock_grad(params):
        x, y = params
        gx = -2*(1-x) - 400*x*(y-x**2)
        gy = 200*(y-x**2)
        return np.array([gx, gy])

    # --- SGD ---
    class SGD:
        def __init__(self, lr=0.001):
            self.lr = lr
        def step(self, params, grad):
            return params - self.lr * grad

    # --- SGD with Momentum ---
    class SGDMomentum:
        def __init__(self, lr=0.001, momentum=0.9):
            self.lr = lr; self.momentum = momentum
            self.v = None
        def step(self, params, grad):
            if self.v is None: self.v = np.zeros_like(params)
            self.v = self.momentum * self.v - self.lr * grad
            return params + self.v

    # --- Adam ---
    class Adam:
        """
        Adam update:
          m = beta1*m + (1-beta1)*g          (1st moment, biased)
          v = beta2*v + (1-beta2)*g^2        (2nd moment, biased)
          m_hat = m / (1-beta1^t)            (bias-corrected)
          v_hat = v / (1-beta2^t)            (bias-corrected)
          params -= lr * m_hat / (sqrt(v_hat) + eps)
        """
        def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
            self.lr=lr; self.beta1=beta1; self.beta2=beta2; self.eps=eps
            self.m=None; self.v=None; self.t=0
        def step(self, params, grad):
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
            m_hat  = self.m / (1 - self.beta1**self.t)
            v_hat  = self.v / (1 - self.beta2**self.t)
            return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    # --- AdamW (Adam + decoupled weight decay) ---
    class AdamW:
        def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
            self.lr=lr; self.beta1=beta1; self.beta2=beta2
            self.eps=eps; self.wd=wd
            self.m=None; self.v=None; self.t=0
        def step(self, params, grad):
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
            m_hat  = self.m / (1 - self.beta1**self.t)
            v_hat  = self.v / (1 - self.beta2**self.t)
            # Weight decay applied directly to params (not via gradient)
            params = params * (1 - self.lr * self.wd)
            return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def run_optimiser(opt, n_steps=500, start=(-1.5, 1.5)):
        params = np.array(start, dtype=float)
        history = [rosenbrock(params)]
        for _ in range(n_steps):
            grad   = rosenbrock_grad(params)
            params = opt.step(params, grad)
            history.append(rosenbrock(params))
        return params, history

    results = {}
    for name, opt in [("SGD",      SGD(lr=0.002)),
                      ("Momentum", SGDMomentum(lr=0.002, momentum=0.9)),
                      ("Adam",     Adam(lr=0.05)),
                      ("AdamW",    AdamW(lr=0.05, wd=0.01))]:
        final_params, hist = run_optimiser(opt)
        results[name] = hist
        dist = np.sqrt((final_params[0]-1)**2 + (final_params[1]-1)**2)
        print(f"  {name:<10}: final loss={hist[-1]:.4f}  "
              f"dist to (1,1)={dist:.4f}  params={final_params.round(3)}")

    # Plot convergence
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, hist in results.items():
        ax.semilogy(hist, label=name)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Optimiser convergence on Rosenbrock function")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("F5_optimisers.png", dpi=100)
    plt.close()
    print("  Saved: F5_optimisers.png")
    print("  Done: F5 optimisers")


# ═════════════════════════════════════════════════════════════════════════════
# F6 — WEIGHT INITIALISATION
# ═════════════════════════════════════════════════════════════════════════════

def section_F6():
    print("\n── F6: Weight Initialisation ──")

    """
    Goal: keep variance of activations and gradients ~1 across layers.

    Xavier (Glorot) init — for tanh/sigmoid:
      Var(W) = 2 / (fan_in + fan_out)
      W ~ Uniform(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)))

    He init — for ReLU:
      Var(W) = 2 / fan_in
      W ~ Normal(0, sqrt(2/fan_in))

    Why He for ReLU: ReLU kills half the neurons (negative outputs → 0),
    so we need 2x larger variance to compensate.
    """

    def zero_init(fan_in, fan_out):
        return np.zeros((fan_out, fan_in))

    def random_init(fan_in, fan_out, scale=0.01):
        return np.random.randn(fan_out, fan_in) * scale

    def xavier_init(fan_in, fan_out):
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_out, fan_in) * std

    def he_init(fan_in, fan_out):
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(fan_out, fan_in) * std

    def relu(x): return np.maximum(0, x)
    def tanh(x): return np.tanh(x)

    # Simulate forward pass through 10 layers, measure activation variance
    n_layers = 10
    fan = 256
    x0  = np.random.randn(1000, fan)   # batch of 1000

    print(f"  {'Init':<12} {'Activation':<10} {'Layer 1 var':>12} "
          f"{'Layer 5 var':>12} {'Layer 10 var':>12}")
    print(f"  {'-'*60}")

    for init_name, init_fn, act_fn, act_name in [
        ("Random 0.01", random_init,  relu, "ReLU"),
        ("Xavier",      xavier_init,  tanh, "Tanh"),
        ("He",          he_init,      relu, "ReLU"),
    ]:
        x = x0.copy()
        vars_ = []
        for layer in range(n_layers):
            W = init_fn(fan, fan)
            x = act_fn(x @ W.T)
            if layer in (0, 4, 9):
                vars_.append(x.var())
        print(f"  {init_name:<12} {act_name:<10} "
              f"{vars_[0]:>12.4f} {vars_[1]:>12.4f} {vars_[2]:>12.4f}")

    # Variance analysis for a single layer
    for fan_in in [64, 256, 1024]:
        w_he    = he_init(fan_in, fan_in)
        w_xavier= xavier_init(fan_in, fan_in)
        print(f"  fan_in={fan_in:4d}: "
              f"He var={w_he.var():.4f} (target={2/fan_in:.4f})  "
              f"Xavier var={w_xavier.var():.4f} (target={2/(2*fan_in):.4f})")

    print("  Done: F6 weight initialisation")


# ═════════════════════════════════════════════════════════════════════════════
# F7 — REGULARISATION
# ═════════════════════════════════════════════════════════════════════════════

def section_F7():
    print("\n── F7: Regularisation ──")

    # --- L1 and L2 regularisation ---
    """
    L2 (weight decay): loss += lambda/2 * sum(W^2)
      gradient: dL/dW += lambda * W   → W decays toward 0 each step

    L1 (lasso): loss += lambda * sum(|W|)
      gradient: dL/dW += lambda * sign(W)   → promotes exact zeros (sparse weights)
    """

    def l2_penalty(W, lam):
        return 0.5 * lam * np.sum(W**2)

    def l2_grad(W, lam):
        return lam * W

    def l1_penalty(W, lam):
        return lam * np.sum(np.abs(W))

    def l1_grad(W, lam):
        return lam * np.sign(W)

    W = np.array([ 2.0, -1.5, 0.1, -0.05, 3.0])
    lam = 0.1

    print(f"  W = {W}")
    print(f"  L2 penalty: {l2_penalty(W, lam):.4f}  gradient: {l2_grad(W, lam)}")
    print(f"  L1 penalty: {l1_penalty(W, lam):.4f}  gradient: {l1_grad(W, lam)}")

    # Simulate L1 vs L2 weight update (one step)
    W_l2_updated = W - 0.1 * l2_grad(W, lam)
    W_l1_updated = W - 0.1 * l1_grad(W, lam)
    print(f"  After one step (lr=0.1):")
    print(f"    L2: {W_l2_updated.round(4)}")
    print(f"    L1: {W_l1_updated.round(4)}  ← small weights get zero faster")

    # --- Dropout ---
    """
    Training: randomly zero out each neuron with probability p (drop rate).
    Scale surviving neurons by 1/(1-p) (inverted dropout) so expected
    activation is unchanged at test time.
    Test: use all neurons, no scaling needed.
    """

    def dropout(x, p=0.5, training=True):
        """
        Inverted dropout.
        p = probability of DROPPING a neuron.
        """
        if not training:
            return x                         # no dropout at test time
        mask = (np.random.rand(*x.shape) > p).astype(float)
        return x * mask / (1 - p)           # scale by 1/(1-p)

    x = np.ones((4, 8))   # batch of 4, 8-dim features

    np.random.seed(0)
    x_train = dropout(x, p=0.5, training=True)
    x_test  = dropout(x, p=0.5, training=False)

    print(f"\n  Dropout p=0.5:")
    print(f"    Train mean: {x_train.mean():.3f}  (should be ~1.0 due to inverted scaling)")
    print(f"    Test  mean: {x_test.mean():.3f}   (no dropout)")
    print(f"    Train sparsity: {(x_train==0).mean()*100:.0f}% zeros")

    # Expected value stays the same: E[x_drop] = (1-p) * x/1/(1-p) = x
    expected_match = abs(x_train.mean() - 1.0) < 0.5  # loose check
    print(f"    Expected value preserved: {expected_match}")
    print("  Done: F7 regularisation")


# ═════════════════════════════════════════════════════════════════════════════
# F8 — BATCH NORMALISATION
# ═════════════════════════════════════════════════════════════════════════════

def section_F8():
    print("\n── F8: Batch Normalisation ──")

    """
    Batch Norm (train):
      mu    = mean(x, axis=batch)                    # (features,)
      sigma = sqrt(var(x, axis=batch) + eps)         # (features,)
      x_hat = (x - mu) / sigma                       # normalised
      y     = gamma * x_hat + beta                   # learnable scale+shift

    Running statistics for inference:
      running_mean = momentum * running_mean + (1-momentum) * mu
      running_var  = momentum * running_var  + (1-momentum) * var

    Batch Norm (inference):
      x_hat = (x - running_mean) / sqrt(running_var + eps)
      y     = gamma * x_hat + beta
    """

    class BatchNorm1d:
        def __init__(self, num_features, eps=1e-5, momentum=0.1):
            self.eps      = eps
            self.momentum = momentum
            self.gamma    = np.ones(num_features)    # learnable scale
            self.beta     = np.zeros(num_features)   # learnable shift
            # Running statistics for inference
            self.running_mean = np.zeros(num_features)
            self.running_var  = np.ones(num_features)
            self.training = True

        def forward(self, x):
            """x: (batch, features)"""
            if self.training:
                mu  = x.mean(axis=0)                     # (features,)
                var = x.var(axis=0)                      # (features,)
                # Update running stats
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
                x_hat = (x - mu) / np.sqrt(var + self.eps)
            else:
                x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

            return self.gamma * x_hat + self.beta

    # Test: input with non-zero mean and large variance
    batch = np.random.randn(32, 64) * 5 + 3    # mean~3, std~5
    bn = BatchNorm1d(64)

    out_train = bn.forward(batch)
    print(f"  Input:  mean={batch.mean():.2f}  std={batch.std():.2f}")
    print(f"  BN out: mean={out_train.mean():.4f}  std={out_train.std():.4f}  "
          f"(should be ~0 and ~1)")

    # Switch to inference mode
    bn.training = False
    out_infer = bn.forward(batch)
    print(f"  Inference BN: mean={out_infer.mean():.4f}  std={out_infer.std():.4f}")

    # --- Layer Norm (normalise over features, not batch) ---
    class LayerNorm:
        def __init__(self, normalised_shape, eps=1e-5):
            self.eps   = eps
            self.gamma = np.ones(normalised_shape)
            self.beta  = np.zeros(normalised_shape)

        def forward(self, x):
            """x: (batch, features) — normalise over last axis"""
            mu  = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1,  keepdims=True)
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            return self.gamma * x_hat + self.beta

    ln  = LayerNorm(64)
    out_ln = ln.forward(batch)
    per_sample_mean = out_ln.mean(axis=-1)
    per_sample_std  = out_ln.std(axis=-1)
    print(f"\n  Layer Norm: per-sample mean range [{per_sample_mean.min():.4f}, "
          f"{per_sample_mean.max():.4f}]  (should be ~0)")
    print(f"  Layer Norm: per-sample std  range [{per_sample_std.min():.4f}, "
          f"{per_sample_std.max():.4f}]  (should be ~1)")

    print(f"\n  BN vs LN vs GN vs IN comparison:")
    print(f"    BatchNorm: normalise over (N,H,W) per channel. Needs large batch.")
    print(f"    LayerNorm: normalise over (C,H,W) per sample. Works with batch=1.")
    print(f"    GroupNorm: normalise over (C/G,H,W) per sample. Stable for small batches.")
    print(f"    InstanceNorm: normalise over (H,W) per sample per channel. For style transfer.")
    print("  Done: F8 batch normalisation")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CV Section F — Neural Network Basics (NumPy only)")
    print("=" * 60)
    np.random.seed(42)

    section_F1()   # perceptron, MLP forward pass, param count
    section_F2()   # all activations + vanishing gradient demo
    section_F3()   # MSE, BCE, focal loss
    section_F4()   # manual backprop + gradient check + XOR training
    section_F5()   # SGD, Momentum, Adam, AdamW on Rosenbrock
    section_F6()   # Xavier vs He init: variance analysis
    section_F7()   # L1/L2 regularisation + inverted dropout
    section_F8()   # BatchNorm + LayerNorm from scratch

    print("\n✓ All Section F demos complete.")
