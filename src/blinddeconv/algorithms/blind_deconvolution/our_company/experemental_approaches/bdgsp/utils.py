"""
utils.py — Helper functions for BDGSP (Babacan et al., ECCV 2012).

Reference:
    S. D. Babacan, R. Molina, M. N. Do, A. K. Katsaggelos,
    "Bayesian Blind Deconvolution with General Sparse Image Priors,"
    ECCV 2012.

Contents
--------
    * FFT-domain circular convolution helpers (`psf2otf`, `apply_K`,
      `apply_KT`, `apply_filter`).
    * High-pass derivative filters used in the filter-space formulation.
    * Super-Gaussian prior kernels  κ(ν) = ρ'(ν)/ν.
    * Image pyramid utilities (build, up/down sampling).
    * Kernel utilities: bilinear resize, center of mass centering,
      non-negative + simplex projection.
    * Edge-taper padding for mitigating circular-convolution boundary
      artefacts.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from numpy.fft import fft2, ifft2


# ════════════════════════════════════════════════════════════════════════════
#  FFT / convolution primitives
# ════════════════════════════════════════════════════════════════════════════
def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a small point-spread function to an optical transfer function
    of the requested output shape (MATLAB ``psf2otf`` semantics).

    The PSF is zero-padded then circularly shifted so that the kernel
    centre lies at pixel (0, 0), which makes FFT-based convolution
    equivalent to ``scipy.signal.convolve2d(..., mode='same')``.
    """
    psf = np.asarray(psf, dtype=np.float64)
    ph, pw = psf.shape
    oh, ow = shape
    pad = np.zeros(shape, dtype=np.float64)
    pad[:ph, :pw] = psf
    # Shift so the kernel centre is at (0, 0).
    pad = np.roll(pad, -(ph // 2), axis=0)
    pad = np.roll(pad, -(pw // 2), axis=1)
    return fft2(pad)


def apply_K(k: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Circular convolution  y = k ⊗ x  (the forward operator T_k x)."""
    K = psf2otf(k, x.shape)
    return np.real(ifft2(K * fft2(x)))


def apply_KT(k: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Adjoint of the circular convolution  T_kᵀ x  (correlation with k)."""
    K = psf2otf(k, x.shape)
    return np.real(ifft2(np.conj(K) * fft2(x)))


def apply_filter(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Circular filter application  f ⊗ x  (alias of :func:`apply_K`)."""
    return apply_K(f, x)


# ════════════════════════════════════════════════════════════════════════════
#  High-pass filters and super-Gaussian prior kernels
# ════════════════════════════════════════════════════════════════════════════
def gradient_filters(order: int = 2) -> List[np.ndarray]:
    """
    High-pass derivative filters used as ``{f_γ}`` in the filter-space
    formulation of Sec. 2 of the paper.

    ``order == 1`` returns only horizontal and vertical first derivatives
    (very sensitive to axis alignment, prone to cross-shaped kernel
    artefacts for diagonal blurs).

    ``order == 2`` (default) additionally returns second-order derivatives
    ``f_xx``, ``f_yy``, ``f_xy``.  This closely mirrors the filter banks
    used by Levin et al. / Fergus et al. and is essential for recovering
    non-axis-aligned blurs (diagonal motion, disk defocus, …).
    """
    fx = np.array([[1.0, -1.0]], dtype=np.float64)
    fy = np.array([[1.0], [-1.0]], dtype=np.float64)
    if order <= 1:
        return [fx, fy]
    fxx = np.array([[1.0, -2.0, 1.0]], dtype=np.float64)
    fyy = np.array([[1.0], [-2.0], [1.0]], dtype=np.float64)
    fxy = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    # Scale second-order derivatives down so their squared norm matches
    # the first-order ones and σ² stays consistent across filters.
    return [fx, fy, fxx / np.sqrt(6.0), fyy / np.sqrt(6.0), fxy / 2.0]


def xi_from_nu(
    nu: np.ndarray,
    prior: str = "log",
    p: float = 0.8,
    sigma_r: float = 0.9,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Compute  E[ξ_γ(i)] = ρ'(ν)/ν  for the chosen super-Gaussian prior.

    ``eps`` acts as a floor on ``ν²`` and therefore as an upper bound on
    ξ (critical for the log prior where ``ξ → 1/ν² → ∞`` at flat pixels).
    Without this floor, the trace-correction term ``trace(C_xγ)`` that
    prevents the kernel from collapsing to a delta vanishes at flat
    regions, and the algorithm deteriorates to MAP with its well-known
    delta-kernel trivial minimum (paper §3.2).

    Mapping follows Fig. 1 / Table of the paper:

    =============  ===================================================
    ``prior``      κ(ν) = ρ'(ν)/ν
    =============  ===================================================
    ``"gaussian"`` 1
    ``"lp"``       |ν|^(p-2)                (default ``p = 0.8``)
    ``"log"``      1 / (ν² + ε)             (default; best performer)
    ``"exp"``      exp(-ν² / (2 σ_r))       (default ``σ_r = 0.9``)
    =============  ===================================================
    """
    prior = prior.lower()
    nu2 = nu ** 2 + eps
    if prior == "gaussian":
        return np.ones_like(nu)
    if prior == "log":
        return 1.0 / nu2
    if prior == "lp":
        return np.power(np.sqrt(nu2), p - 2.0)
    if prior == "exp":
        return np.exp(-nu2 / (2.0 * sigma_r))
    raise ValueError(f"Unknown prior '{prior}'.")


# ════════════════════════════════════════════════════════════════════════════
#  Kernel utilities
# ════════════════════════════════════════════════════════════════════════════
def project_simplex(k: np.ndarray) -> np.ndarray:
    """
    Project a kernel onto { k : k ≥ 0, Σ k = 1 }.

    Uses the simple non-negativity + renormalisation rule that is standard
    in the blind-deconvolution literature.  Guarantees a valid PSF as long
    as the input has at least one positive entry.
    """
    k = np.maximum(k, 0.0)
    s = k.sum()
    if s > 0:
        k = k / s
    else:
        k = np.zeros_like(k)
        k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
    return k


def center_kernel(k: np.ndarray) -> np.ndarray:
    """Shift the centre of mass of ``k`` to the geometric centre."""
    h, w = k.shape
    s = k.sum()
    if s <= 0:
        return k
    ys, xs = np.mgrid[0:h, 0:w]
    cy = (ys * k).sum() / s
    cx = (xs * k).sum() / s
    dy = int(round(h / 2.0 - 0.5 - cy))
    dx = int(round(w / 2.0 - 0.5 - cx))
    return np.roll(np.roll(k, dy, axis=0), dx, axis=1)


def resize_kernel(k: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """
    Bilinear resize of a kernel to ``new_shape`` and re-projection onto the
    simplex.  Used when propagating the kernel between pyramid levels.
    """
    from scipy.ndimage import zoom  # local import: keep top imports light

    oh, ow = k.shape
    nh, nw = new_shape
    if (oh, ow) == (nh, nw):
        return project_simplex(k.copy())
    zy = nh / float(oh)
    zx = nw / float(ow)
    out = zoom(k, (zy, zx), order=1, mode="constant", cval=0.0)
    # Ensure output has exactly the requested shape.
    out = out[:nh, :nw]
    if out.shape != (nh, nw):
        pad_h = nh - out.shape[0]
        pad_w = nw - out.shape[1]
        out = np.pad(out, ((0, max(0, pad_h)), (0, max(0, pad_w))))
    return project_simplex(out)


# ════════════════════════════════════════════════════════════════════════════
#  Pyramid helpers
# ════════════════════════════════════════════════════════════════════════════
def _odd(x: float, min_val: int = 3) -> int:
    """Round ``x`` to the nearest odd integer, lower-bounded by ``min_val``."""
    x = int(max(min_val, round(x)))
    return x if x % 2 == 1 else x + 1


def build_pyramid_sizes(
    img_shape: Tuple[int, int],
    kernel_size: int,
    min_kernel: int = 3,
    scale: float = 1.0 / np.sqrt(2.0),
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Return a list of (image_shape, kernel_size) pairs from coarsest to
    finest, indexed so that successive levels have kernel sizes
    linearly increasing from ``min_kernel`` (odd) up to
    ``kernel_size`` (odd).

    The number of levels is chosen such that going coarser by
    ``scale`` per step brings the kernel down to ``min_kernel``.
    """
    min_k = _odd(min_kernel, min_val=3)
    max_k = _odd(kernel_size, min_val=min_k)
    if max_k <= min_k:
        return [(img_shape, max_k)]

    # Number of levels: how many ``scale`` steps take max_k → min_k.
    n_levels = int(np.ceil(np.log(min_k / max_k) / np.log(scale))) + 1
    n_levels = max(2, n_levels)

    levels: List[Tuple[Tuple[int, int], int]] = []
    H, W = img_shape
    for l in range(n_levels):
        # l = 0 is the finest level (full resolution), l = n_levels-1 coarsest.
        s = scale ** l
        h = max(int(round(H * s)), 2 * min_k)
        w = max(int(round(W * s)), 2 * min_k)
        k = _odd(max_k * s, min_val=min_k)
        k = min(k, max_k)
        levels.append(((h, w), k))
    # Coarsest first.
    return list(reversed(levels))


def resize_image(img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a 2-D image."""
    from scipy.ndimage import zoom

    h, w = img.shape
    nh, nw = new_shape
    if (h, w) == (nh, nw):
        return img.copy()
    return zoom(img, (nh / float(h), nw / float(w)), order=1, mode="reflect")


# ════════════════════════════════════════════════════════════════════════════
#  Boundary handling
# ════════════════════════════════════════════════════════════════════════════
def edgetaper(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Replication of MATLAB's ``edgetaper``: blend the image with its blurred
    replica near the borders so circular-convolution wrap-around does not
    inject high-frequency seams into the kernel estimate.

    This step is essential for reducing the axis-aligned cross artefacts
    that appear in kernel estimation when FFT-based convolution is used
    without proper boundary handling.
    """
    h, w = img.shape
    ph, pw = psf.shape
    # 1-D autocorrelations of the PSF, normalised and clipped to [0, 1].
    ax = psf.sum(axis=0)
    ay = psf.sum(axis=1)
    if ax.sum() > 0:
        ax = ax / ax.sum()
    if ay.sum() > 0:
        ay = ay / ay.sum()
    ac_x = np.correlate(ax, ax, mode="full")
    ac_y = np.correlate(ay, ay, mode="full")
    # Weighting profile that is 0 deep inside the image and grows near the
    # borders, reaching ~1 at the edge.
    wx_profile = np.zeros(w, dtype=np.float64)
    wy_profile = np.zeros(h, dtype=np.float64)
    Lx = min(pw, w // 2)
    Ly = min(ph, h // 2)
    if Lx > 0:
        base_x = ac_x[len(ac_x) // 2 :][:Lx]
        base_x = base_x / max(base_x.max(), 1e-12)
        wx_profile[:Lx] = 1.0 - base_x
        wx_profile[-Lx:] = 1.0 - base_x[::-1]
    if Ly > 0:
        base_y = ac_y[len(ac_y) // 2 :][:Ly]
        base_y = base_y / max(base_y.max(), 1e-12)
        wy_profile[:Ly] = 1.0 - base_y
        wy_profile[-Ly:] = 1.0 - base_y[::-1]
    weight = np.outer(wy_profile, wx_profile)
    blurred = apply_K(psf, img)
    return weight * img + (1.0 - weight) * blurred


def edgetaper_pad(img: np.ndarray, pad: int) -> np.ndarray:
    """Reflect-pad an image by ``pad`` pixels on each side."""
    if pad <= 0:
        return img.copy()
    return np.pad(img, pad, mode="reflect")


def crop_center(img: np.ndarray, pad: int) -> np.ndarray:
    """Crop ``pad`` pixels from each side (inverse of :func:`edgetaper_pad`)."""
    if pad <= 0:
        return img.copy()
    return img[pad:-pad, pad:-pad].copy()


# ════════════════════════════════════════════════════════════════════════════
#  Convenience
# ════════════════════════════════════════════════════════════════════════════
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a uint8/float image to ``float64`` grayscale in [0, 1]."""
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3 and x.shape[2] == 3:
        x = 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]
    elif x.ndim == 3:
        x = x[..., 0]
    if x.max() > 1.5:
        x = x / 255.0
    return x


def compute_filtered(y: np.ndarray, filters: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Return ``[f_γ ⊗ y for f_γ in filters]`` using circular convolution."""
    return [apply_filter(f, y) for f in filters]
