"""
QSSIM (Quaternion Structural Similarity Index Measure) Node for ComfyUI

각 RGB 픽셀을 순수 사원수(pure quaternion)로 표현:
    q = R·i + G·j + B·k

이를 통해 채널 간 색상 상관관계를 보존하면서 구조적 유사도를 측정합니다.

수학적 기반:
- 사원수 평균: μ = E[q]  (채널별 평균을 사원수 허수부로 구성)
- 사원수 분산: σ² = E[|q - μ|²] = Var(R) + Var(G) + Var(B)
- 사원수 공분산: σ₁₂ = E[Re(conj(q₁ - μ₁)·(q₂ - μ₂))]
                      = Cov(R₁,R₂) + Cov(G₁,G₂) + Cov(B₁,B₂)
- 사원수 노름: |μ|² = μR² + μG² + μB²

QSSIM = l(x,y) · c(x,y) · s(x,y)
  l: 밝기 비교  (2|μ₁||μ₂| + C₁) / (|μ₁|² + |μ₂|² + C₁)
  c: 대비 비교  (2σ₁σ₂ + C₂)     / (σ₁² + σ₂² + C₂)
  s: 구조 비교  (σ₁₂ + C₃)       / (σ₁σ₂ + C₃)
"""

import torch
import torch.nn.functional as F


def _gaussian_kernel_2d(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """정규화된 2D 가우시안 커널 생성."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)  # outer product → [H, W]
    return kernel


def _apply_filter(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """각 채널에 2D 필터를 독립적으로 적용. img: [B, C, H, W]"""
    B, C, H, W = img.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2
    k = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, kH, kW)
    return F.conv2d(img, k, padding=(pad_h, pad_w), groups=C)


def compute_qssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> float:
    """
    두 RGB 이미지 사이의 QSSIM 값을 계산합니다.

    Args:
        img1: 기준 이미지  [B, C, H, W], 값 범위 [0, 1]
        img2: 비교 이미지  [B, C, H, W], 값 범위 [0, 1]
        window_size: 가우시안 윈도우 크기 (홀수)
        sigma:       가우시안 표준편차
        K1, K2:      수치 안정성 상수

    Returns:
        QSSIM 스코어 (float). 범위: [-1, 1], 1이면 완전히 동일.
    """
    assert img1.shape == img2.shape, (
        f"두 이미지의 크기가 달라서 QSSIM을 계산할 수 없습니다. "
        f"img1={tuple(img1.shape)}, img2={tuple(img2.shape)}"
    )

    L = 1.0
    C1 = (K1 * L) ** 2  # 밝기 안정 상수
    C2 = (K2 * L) ** 2  # 대비 안정 상수
    C3 = C2 / 2.0       # 구조 안정 상수

    device = img1.device
    kernel = _gaussian_kernel_2d(window_size, sigma, device)

    # ── 사원수 평균 (채널별 국소 평균) ──────────────────────────────────────
    # mu1, mu2: [B, C, H, W]
    mu1 = _apply_filter(img1, kernel)
    mu2 = _apply_filter(img2, kernel)

    # |μ₁|² = μR² + μG² + μB²  →  [B, H, W]
    mu1_sq_norm = (mu1 ** 2).sum(dim=1)
    mu2_sq_norm = (mu2 ** 2).sum(dim=1)

    # dot(μ₁, μ₂) = Re(conj(μ₁)·μ₂)  →  [B, H, W]
    mu1_mu2_dot = (mu1 * mu2).sum(dim=1)

    # |μ₁|, |μ₂|  →  [B, H, W]
    mu1_norm = mu1_sq_norm.sqrt()
    mu2_norm = mu2_sq_norm.sqrt()

    # ── 사원수 분산 ──────────────────────────────────────────────────────────
    # E[|q|²] = E[R²] + E[G²] + E[B²]  →  [B, H, W]
    e_img1_sq = _apply_filter(img1 ** 2, kernel).sum(dim=1)
    e_img2_sq = _apply_filter(img2 ** 2, kernel).sum(dim=1)

    # σ² = E[|q|²] - |E[q]|²
    sigma1_sq = (e_img1_sq - mu1_sq_norm).clamp(min=0.0)
    sigma2_sq = (e_img2_sq - mu2_sq_norm).clamp(min=0.0)

    sigma1 = sigma1_sq.sqrt()
    sigma2 = sigma2_sq.sqrt()

    # ── 사원수 공분산 ─────────────────────────────────────────────────────────
    # E[dot(q₁, q₂)] = E[R₁R₂] + E[G₁G₂] + E[B₁B₂]  →  [B, H, W]
    e_img1_img2 = _apply_filter(img1 * img2, kernel).sum(dim=1)

    # σ₁₂ = E[dot(q₁,q₂)] - dot(μ₁,μ₂)
    sigma12 = e_img1_img2 - mu1_mu2_dot

    # ── QSSIM 성분 계산 ──────────────────────────────────────────────────────
    # l: 밝기 비교 — 사원수 노름 기반
    luminance = (2.0 * mu1_norm * mu2_norm + C1) / (mu1_sq_norm + mu2_sq_norm + C1)

    # c: 대비 비교 — 사원수 분산 기반
    contrast = (2.0 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)

    # s: 구조 비교 — 사원수 공분산 기반
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    # QSSIM map [B, H, W] → 스칼라
    qssim_map = luminance * contrast * structure
    return qssim_map.mean().item()


class QSSIMNode:
    """
    QSSIM (Quaternion SSIM) 화질 평가 노드

    두 컬러 이미지를 입력받아 사원수 기반 구조적 유사도(QSSIM)를 계산합니다.
    표준 SSIM과 달리 RGB 채널을 사원수 허수부로 표현하여
    채널 간 색상 상관관계를 보존하는 화질 지표를 산출합니다.

    출력 범위: [-1, 1]  (1에 가까울수록 두 이미지가 동일)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
            "optional": {
                "window_size": ("INT", {
                    "default": 11,
                    "min": 3,
                    "max": 21,
                    "step": 2,
                    "tooltip": "가우시안 윈도우 크기 (홀수)",
                }),
                "sigma": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "가우시안 표준편차",
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("qssim_score",)
    FUNCTION = "evaluate"
    CATEGORY = "image/quality"
    OUTPUT_NODE = True

    def evaluate(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
    ):
        """
        Args:
            image1: 기준 이미지  [B, H, W, C]  (ComfyUI 표준 포맷)
            image2: 비교 이미지  [B, H, W, C]
            window_size: 가우시안 윈도우 크기
            sigma:       가우시안 표준편차

        Returns:
            (qssim_score,): QSSIM 스코어 tuple
        """
        # ── 홀수 강제 ────────────────────────────────────────────────────────
        if window_size % 2 == 0:
            window_size += 1

        # ── ComfyUI [B,H,W,C] → PyTorch [B,C,H,W] ────────────────────────
        img1 = image1.permute(0, 3, 1, 2).contiguous()
        img2 = image2.permute(0, 3, 1, 2).contiguous()

        # ── 그레이스케일 확장 (1채널 → 3채널) ────────────────────────────────
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
        if img2.shape[1] == 1:
            img2 = img2.repeat(1, 3, 1, 1)

        # ── RGB만 사용 (알파채널 제거) ─────────────────────────────────────
        img1 = img1[:, :3, :, :]
        img2 = img2[:, :3, :, :]

        # ── 크기 불일치 시 img2를 img1 크기로 리사이즈 ───────────────────────
        if img1.shape[2:] != img2.shape[2:]:
            img2 = F.interpolate(
                img2, size=img1.shape[2:], mode="bilinear", align_corners=False
            )

        score = compute_qssim(img1, img2, window_size=window_size, sigma=sigma)

        print(f"[QSSIM] Score: {score:.6f}")

        return {"ui": {"text": [f"QSSIM: {score:.6f}"]}, "result": (score,)}
