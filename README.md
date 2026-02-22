# ComfyUI_QSSIM

A ComfyUI custom node that computes **QSSIM** (Quaternion Structural Similarity Index Measure) — a color-aware image quality metric that treats each RGB pixel as a pure quaternion.

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_QSSIM/blob/main/ScrShot%203.png)

## Why QSSIM?

Standard SSIM converts color images to grayscale or evaluates each channel independently, discarding cross-channel color relationships. QSSIM addresses this by representing every pixel as a **pure quaternion**:

```
q = R·i + G·j + B·k
```

This single mathematical object encodes all three color channels simultaneously, so the similarity measure inherently captures **inter-channel color correlations** that SSIM misses.

## How It Works

QSSIM decomposes similarity into three components, each computed in the quaternion domain:

```
QSSIM(x, y) = l(x,y) · c(x,y) · s(x,y)
```

| Component | Formula | Description |
|-----------|---------|-------------|
| **l** — Luminance | `(2‖μ₁‖‖μ₂‖ + C₁) / (‖μ₁‖² + ‖μ₂‖² + C₁)` | Compares quaternion norms of local means |
| **c** — Contrast | `(2σ₁σ₂ + C₂) / (σ₁² + σ₂² + C₂)` | Compares total color variance across channels |
| **s** — Structure | `(σ₁₂ + C₃) / (σ₁σ₂ + C₃)` | Compares cross-channel covariance structure |

### Key mathematical identities used

For pure quaternions `p = p_R·i + p_G·j + p_B·k` and `q = q_R·i + q_G·j + q_B·k`:

- **Quaternion norm**: `‖q‖² = q_R² + q_G² + q_B²`
- **Real part of product**: `Re(conj(p)·q) = p_R·q_R + p_G·q_G + p_B·q_B`
  → This means the quaternion covariance reduces to the **sum of per-channel covariances**, computed efficiently without explicit quaternion arithmetic.
- **Quaternion variance**: `σ² = Var(R) + Var(G) + Var(B)`

Local statistics are computed with a **2D Gaussian window** (default: 11×11, σ = 1.5), identical to the original SSIM paper.

## Installation

Clone or copy this folder into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone <repo-url> ComfyUI_QSSIM
```

Then restart ComfyUI. No additional dependencies are required beyond what ComfyUI already provides (PyTorch).

## Node

**Category:** `image/quality`
**Display name:** `QSSIM Quality Score`

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `image1` | IMAGE | Yes | Reference image |
| `image2` | IMAGE | Yes | Image to compare against the reference |
| `window_size` | INT | No | Gaussian window size (odd, default: 11, range: 3–21) |
| `sigma` | FLOAT | No | Gaussian standard deviation (default: 1.5, range: 0.1–5.0) |

### Outputs

| Name | Type | Range | Description |
|------|------|-------|-------------|
| `qssim_score` | FLOAT | [−1, 1] | QSSIM score. `1.0` = identical images. |

The score is also displayed directly on the node in the ComfyUI canvas.

### Handling mismatched inputs

- If `image2` has a different spatial resolution than `image1`, it is bilinearly resized to match.
- Grayscale images (1 channel) are broadcast to 3 channels before computation.
- Alpha channels (4-channel RGBA) are silently dropped; only RGB is used.

## Score Interpretation

| QSSIM range | Interpretation |
|-------------|----------------|
| ≈ 1.0 | Images are perceptually identical |
| 0.9 – 1.0 | High similarity, minor differences |
| 0.5 – 0.9 | Moderate similarity |
| 0.0 – 0.5 | Low similarity |
| < 0.0 | Strong structural dissimilarity |

## Parameters Guide

| Parameter | Lower value | Higher value |
|-----------|-------------|--------------|
| `window_size` | Finer local detail, faster | Coarser local statistics, slower |
| `sigma` | Sharper Gaussian weighting | Smoother, broader Gaussian weighting |

The defaults (`window_size=11`, `sigma=1.5`) match the original SSIM paper settings and work well for most use cases.

## File Structure

```
ComfyUI_QSSIM/
├── __init__.py      # Node registration
├── qssim_node.py    # QSSIM algorithm + QSSIMNode class
└── README.md
```

## License

MIT
