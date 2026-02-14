"""
Utility functions for Fractional-Order Blind Image Deconvolution.

Mathematical infrastructure for:
  - Grünwald–Letnikov (GL) fractional-order differentiation
  - Patch-wise Minimal Pixels (PMP) prior computation
  - Multi-scale Gaussian pyramid construction
  - PSF / OTF conversions, gradient operators, shrinkage operators

References
----------
[1] Wu, T., Wan, S., Feng, C., Zhang, H., Zeng, T.
    "Blind Image Deconvolution: When Patch-wise Minimal Pixels Prior
    Meets Fractional-Order Method."
    Journal of Mathematical Imaging and Vision, 2024.
    DOI: 10.1007/s10851-024-01221-x

[2] Pan, X., Ye, Y., Wang, J., Gao, X., Zhou, X.
    "Noncausal fractional directional differentiator and blind
    deconvolution: motion blur estimation."
    Multimedia Tools and Applications, 73(3), 1485–1506, 2014.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import minimum_filter, gaussian_filter
from scipy.special import gamma as gamma_func
from typing import Tuple, List, NamedTuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPSILON = 1e-12


# ---------------------------------------------------------------------------
# Named tuples for structured data
# ---------------------------------------------------------------------------
class FractionalOperators(NamedTuple):
    """Pre-computed Fourier-domain fractional gradient operators."""
    F_Cx: np.ndarray        # FFT of horizontal GL filter
    F_Cy: np.ndarray        # FFT of vertical GL filter
    F_frac_sq: np.ndarray   # |F_Cx|^2 + |F_Cy|^2


class IntegerGradientOperators(NamedTuple):
    """Pre-computed Fourier-domain first-order gradient operators."""
    F_dx: np.ndarray        # FFT of horizontal difference filter
    F_dy: np.ndarray        # FFT of vertical difference filter
    F_grad_sq: np.ndarray   # |F_dx|^2 + |F_dy|^2


# ===================================================================
#  PSF / OTF conversions
# ===================================================================
def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a Point-Spread Function (PSF) to an Optical Transfer Function (OTF).

    The PSF is zero-padded to *shape* and circularly shifted so that the
    centre of the PSF lands at the origin (0, 0) before the FFT.

    Parameters
    ----------
    psf : ndarray, shape (kh, kw)
        Spatial-domain blur kernel.
    shape : (H, W)
        Target output size (image dimensions).

    Returns
    -------
    otf : ndarray, complex, shape (H, W)
        Frequency-domain transfer function.
    """
    kh, kw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:kh, :kw] = psf
    # Circular shift so that the kernel centre goes to (0, 0)
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return fft2(padded)


def otf2psf(otf: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert an OTF back to a PSF of size *out_shape*.

    Parameters
    ----------
    otf : ndarray, complex
        Frequency-domain transfer function.
    out_shape : (kh, kw)
        Desired spatial extent of the output PSF.

    Returns
    -------
    psf : ndarray, shape (kh, kw)
    """
    kh, kw = out_shape
    full = np.real(ifft2(otf))
    full = np.roll(full, kh // 2, axis=0)
    full = np.roll(full, kw // 2, axis=1)
    return full[:kh, :kw]


# ===================================================================
#  Shrinkage / thresholding operators
# ===================================================================
def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    r"""
    Proximal operator of the :math:`\ell_1` norm (soft thresholding).

    .. math::
        \mathrm{shrink}(x, \tau) = \mathrm{sign}(x)\,\max(|x| - \tau,\, 0)

    Parameters
    ----------
    x : ndarray
    thresh : float  (>= 0)

    Returns
    -------
    ndarray, same shape as *x*.
    """
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


# ===================================================================
#  Grünwald–Letnikov fractional derivative
# ===================================================================
def grunwald_letnikov_weights(alpha: float, L: int) -> np.ndarray:
    r"""
    Compute the truncated Grünwald–Letnikov (GL) coefficients of order
    :math:`\alpha`.

    The recurrence ([2], Sec. 2; [1], Eq. 7) is

    .. math::
        c_0^{(\alpha)} = 1, \qquad
        c_j^{(\alpha)} = \Bigl(1 - \frac{\alpha + 1}{j}\Bigr)\,
                         c_{j-1}^{(\alpha)}, \quad j = 1, 2, \dots

    These are the coefficients of the finite-difference approximation

    .. math::
        D^{\alpha} f[n] \approx \sum_{l=0}^{L-1} c_l^{(\alpha)}\, f[n - l].

    Parameters
    ----------
    alpha : float
        Fractional order (typically 1 < alpha < 2).
    L : int
        Truncation length (number of taps).

    Returns
    -------
    c : ndarray, shape (L,)
        GL coefficients ``[c_0, c_1, ..., c_{L-1}]``.

    Notes
    -----
    * For :math:`\alpha = 1`: ``c = [1, -1, 0, ...]`` (first difference).
    * For :math:`\alpha = 2`: ``c = [1, -2, 1, 0, ...]`` (second difference).
    """
    c = np.zeros(L, dtype=np.float64)
    c[0] = 1.0
    for j in range(1, L):
        c[j] = (1.0 - (alpha + 1.0) / j) * c[j - 1]
    return c


def build_fractional_filter_2d(
    alpha: float, L: int, shape: Tuple[int, int], axis: int
) -> np.ndarray:
    r"""
    Embed GL coefficients into a 2-D array suitable for ``fft2``.

    The filter is placed along *axis* starting at the origin so that

    .. math::
        F\{D_{x}^{\alpha} u\} = \hat{C}_x \cdot \hat{u}

    with ``\hat{C}_x = \mathrm{fft2}(\text{filter})``.

    Parameters
    ----------
    alpha : float
    L : int
        Truncation length (capped to image dimension along *axis*).
    shape : (H, W)
    axis : int
        0 → vertical (y) derivative, 1 → horizontal (x) derivative.

    Returns
    -------
    filt : ndarray, shape (H, W)
    """
    H, W = shape
    dim = H if axis == 0 else W
    L_eff = min(L, dim)
    c = grunwald_letnikov_weights(alpha, L_eff)

    filt = np.zeros(shape, dtype=np.float64)
    if axis == 1:
        filt[0, :L_eff] = c
    else:
        filt[:L_eff, 0] = c
    return filt


def precompute_fractional_operators(
    shape: Tuple[int, int], alpha: float, L: int = 10
) -> FractionalOperators:
    r"""
    Pre-compute the FFTs of the fractional gradient operators and their
    combined squared magnitude.

    .. math::
        |\hat{C}_x|^2 + |\hat{C}_y|^2

    Parameters
    ----------
    shape : (H, W)
    alpha : float
        Fractional order.
    L : int
        GL truncation length (default 10).

    Returns
    -------
    FractionalOperators
        Named tuple ``(F_Cx, F_Cy, F_frac_sq)``.
    """
    filt_x = build_fractional_filter_2d(alpha, L, shape, axis=1)
    filt_y = build_fractional_filter_2d(alpha, L, shape, axis=0)
    F_Cx = fft2(filt_x)
    F_Cy = fft2(filt_y)
    F_frac_sq = np.abs(F_Cx) ** 2 + np.abs(F_Cy) ** 2
    return FractionalOperators(F_Cx, F_Cy, F_frac_sq)


# ===================================================================
#  Integer-order (first-difference) gradient operators
# ===================================================================
def precompute_gradient_operators(
    shape: Tuple[int, int],
) -> IntegerGradientOperators:
    r"""
    Pre-compute FFTs of the first-order forward-difference operators
    :math:`\partial_x`, :math:`\partial_y`.

    .. math::
        \partial_x u[i,j] = u[i, j+1] - u[i, j]

    Returns
    -------
    IntegerGradientOperators
        Named tuple ``(F_dx, F_dy, F_grad_sq)``.
    """
    H, W = shape
    dx = np.zeros(shape, dtype=np.float64)
    dx[0, 0] = -1.0
    dx[0, 1] = 1.0
    dy = np.zeros(shape, dtype=np.float64)
    dy[0, 0] = -1.0
    dy[1, 0] = 1.0
    F_dx = fft2(dx)
    F_dy = fft2(dy)
    F_grad_sq = np.abs(F_dx) ** 2 + np.abs(F_dy) ** 2
    return IntegerGradientOperators(F_dx, F_dy, F_grad_sq)


def compute_spatial_gradient(
    u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Spatial first-order forward differences (circular boundary).

    .. math::
        \partial_x u[i,j] = u[i, j{+}1] - u[i, j]

    Returns
    -------
    (dx, dy) : tuple of ndarray
    """
    dx = np.roll(u, -1, axis=1) - u
    dy = np.roll(u, -1, axis=0) - u
    return dx, dy


# ===================================================================
#  Patch-wise Minimal Pixels (PMP) prior
# ===================================================================
def compute_pmp_map(
    u: np.ndarray, patch_size: int = 5
) -> np.ndarray:
    r"""
    Compute the Patch-wise Minimal Pixels (PMP) map.

    For each pixel :math:`(i, j)` the PMP value is the minimum intensity
    in the surrounding :math:`p \times p` patch ([1], Sec. 3.2):

    .. math::
        \mathrm{PMP}(u)(i, j) = \min_{(s, t)\,\in\,\Omega_p(i,j)} u(s, t)

    This is equivalent to a morphological erosion with a flat
    :math:`p \times p` structuring element.

    Parameters
    ----------
    u : ndarray, shape (H, W)
        Grayscale image in [0, 1].
    patch_size : int
        Side length of the square patch (must be odd for symmetry).

    Returns
    -------
    pmp : ndarray, shape (H, W)
    """
    return minimum_filter(u, size=patch_size, mode='reflect')


def predict_edges_with_pmp(
    u: np.ndarray,
    grad_threshold_percentile: float = 94.0,
    patch_size: int = 5,
    pmp_gamma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Predict salient edges for kernel estimation, guided by the PMP prior.

    Procedure ([1], Sec. 3.2):
    1. Compute spatial gradients :math:`\partial_x u`, :math:`\partial_y u`.
    2. Compute the PMP map and derive a weight
       :math:`w = \exp(-\gamma \cdot \mathrm{PMP}(u))`.
       Low PMP values (near dark pixels) yield *high* weight, meaning
       those edges are likely genuine structure rather than blur artefacts.
    3. Weight the gradient magnitudes and retain only those above a
       percentile threshold.

    Parameters
    ----------
    u : ndarray  (H, W)
    grad_threshold_percentile : float
        Percentile (0–100) of weighted gradient magnitude above which
        edges are kept.
    patch_size : int
        PMP patch side length.
    pmp_gamma : float
        Decay rate for the PMP weight map.

    Returns
    -------
    (pred_dx, pred_dy) : tuple of ndarray
        Predicted (thresholded) gradient maps.
    """
    dx, dy = compute_spatial_gradient(u)
    mag = np.sqrt(dx ** 2 + dy ** 2 + EPSILON)

    # PMP-based weighting
    pmp = compute_pmp_map(u, patch_size)
    weight = np.exp(-pmp_gamma * pmp)

    weighted_mag = mag * weight
    tau = np.percentile(weighted_mag, grad_threshold_percentile)

    mask = weighted_mag >= tau
    pred_dx = dx * mask
    pred_dy = dy * mask
    return pred_dx, pred_dy


# ===================================================================
#  Multi-scale pyramid utilities
# ===================================================================
def build_scale_list(
    max_kernel_dim: int, min_kernel_dim: int = 3, scale_ratio: float = 1.5
) -> List[float]:
    r"""
    Compute the sequence of image-scale factors for coarse-to-fine
    kernel estimation.

    At the coarsest level the effective kernel width is approximately
    *min_kernel_dim* pixels; at the finest it equals *max_kernel_dim*.

    Parameters
    ----------
    max_kernel_dim : int
        Largest kernel dimension at the finest (original) scale.
    min_kernel_dim : int
        Approximate kernel size at the coarsest level.
    scale_ratio : float
        Geometric ratio between successive scales (> 1).

    Returns
    -------
    scales : list of float
        Image down-scale factors, coarse → fine.  The last entry is 1.0.
    """
    if max_kernel_dim <= min_kernel_dim:
        return [1.0]

    num_scales = int(
        np.ceil(np.log(max_kernel_dim / min_kernel_dim) / np.log(scale_ratio))
    ) + 1
    num_scales = max(num_scales, 2)

    # Geometric sequence of effective kernel widths, coarse → fine
    factors = np.logspace(
        np.log10(min_kernel_dim / max_kernel_dim), 0.0, num_scales
    )
    return factors.tolist()


def downscale_image(
    image: np.ndarray, scale: float
) -> np.ndarray:
    """
    Down-scale *image* by *scale* using Gaussian anti-aliasing.

    Parameters
    ----------
    image : ndarray (H, W)
    scale : float in (0, 1]

    Returns
    -------
    ndarray
    """
    if scale >= 1.0 - 1e-8:
        return image.copy()

    # Anti-alias with Gaussian whose sigma matches the scale
    sigma = 0.5 / scale
    smoothed = gaussian_filter(image, sigma=sigma, mode='reflect')

    H, W = image.shape
    new_H = max(1, int(np.round(H * scale)))
    new_W = max(1, int(np.round(W * scale)))

    # Bilinear interpolation via meshgrid
    row_idx = np.linspace(0, H - 1, new_H)
    col_idx = np.linspace(0, W - 1, new_W)
    row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing='ij')

    # Use floor/ceil for bilinear interpolation
    r0 = np.floor(row_grid).astype(int)
    r1 = np.minimum(r0 + 1, H - 1)
    c0 = np.floor(col_grid).astype(int)
    c1 = np.minimum(c0 + 1, W - 1)

    dr = row_grid - r0
    dc = col_grid - c0

    out = (
        smoothed[r0, c0] * (1 - dr) * (1 - dc)
        + smoothed[r1, c0] * dr * (1 - dc)
        + smoothed[r0, c1] * (1 - dr) * dc
        + smoothed[r1, c1] * dr * dc
    )
    return out


def resize_kernel(
    kernel: np.ndarray, new_shape: Tuple[int, int]
) -> np.ndarray:
    r"""
    Resize a blur kernel to *new_shape* using bilinear interpolation,
    then re-normalise to unit sum.

    Parameters
    ----------
    kernel : ndarray (kh_old, kw_old)
    new_shape : (kh_new, kw_new)

    Returns
    -------
    ndarray (kh_new, kw_new)
    """
    kh_old, kw_old = kernel.shape
    kh_new, kw_new = new_shape

    if (kh_old, kw_old) == (kh_new, kw_new):
        return kernel.copy()

    row_idx = np.linspace(0, kh_old - 1, kh_new)
    col_idx = np.linspace(0, kw_old - 1, kw_new)
    rg, cg = np.meshgrid(row_idx, col_idx, indexing='ij')

    r0 = np.floor(rg).astype(int)
    r1 = np.minimum(r0 + 1, kh_old - 1)
    c0 = np.floor(cg).astype(int)
    c1 = np.minimum(c0 + 1, kw_old - 1)
    dr = rg - r0
    dc = cg - c0

    out = (
        kernel[r0, c0] * (1 - dr) * (1 - dc)
        + kernel[r1, c0] * dr * (1 - dc)
        + kernel[r0, c1] * (1 - dr) * dc
        + kernel[r1, c1] * dr * dc
    )
    out = np.maximum(out, 0.0)
    s = out.sum()
    if s > EPSILON:
        out /= s
    return out


# ===================================================================
#  Kernel post-processing
# ===================================================================
def threshold_kernel(
    kernel: np.ndarray, rel_threshold: float = 0.05
) -> np.ndarray:
    r"""
    Remove small / noisy entries from a kernel estimate and
    re-normalise.

    Parameters
    ----------
    kernel : ndarray
    rel_threshold : float
        Fraction of the maximum value below which entries are zeroed.

    Returns
    -------
    ndarray
    """
    k = kernel.copy()
    k[k < rel_threshold * k.max()] = 0.0
    k = np.maximum(k, 0.0)
    s = k.sum()
    if s > EPSILON:
        k /= s
    else:
        # Fall back to a delta kernel
        k = np.zeros_like(kernel)
        k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
    return k


def center_kernel(kernel: np.ndarray) -> np.ndarray:
    """
    Shift a kernel so that its centre of mass coincides with the
    geometric centre of the array.

    Parameters
    ----------
    kernel : ndarray (kh, kw)

    Returns
    -------
    ndarray (kh, kw)
    """
    kh, kw = kernel.shape
    total = kernel.sum()
    if total < EPSILON:
        return kernel.copy()

    yy, xx = np.mgrid[:kh, :kw]
    cy = np.sum(yy * kernel) / total
    cx = np.sum(xx * kernel) / total

    shift_y = int(np.round(kh / 2.0 - cy))
    shift_x = int(np.round(kw / 2.0 - cx))

    return np.roll(np.roll(kernel, shift_y, axis=0), shift_x, axis=1)


def make_initial_kernel(
    kernel_shape: Tuple[int, int],
) -> np.ndarray:
    r"""
    Create an initial Gaussian kernel for the coarsest pyramid level.

    The standard deviation is set to ``max(kh, kw) / 6`` so that
    the kernel energy is well inside the support.

    Parameters
    ----------
    kernel_shape : (kh, kw)

    Returns
    -------
    ndarray (kh, kw)
    """
    kh, kw = kernel_shape
    sig = max(kh, kw) / 6.0
    yy, xx = np.ogrid[-(kh // 2): kh - kh // 2,
                       -(kw // 2): kw - kw // 2]
    h = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sig ** 2))
    h /= h.sum()
    return h
