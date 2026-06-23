"""
Utility functions for Variational Bayesian Sparse Kernel-Based
Blind Image Deconvolution with Student's-t Priors (VBSKB-BID-STP).

Based on:
    Tzikas D.G., Likas A.C., Galatsanos N.P. (2009).
    "Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution
     With Student's-t Priors." IEEE Trans. Image Process., 18(1), 200–208.

    Chantas G., Galatsanos N., Likas A., Saunders M. (2008).
    "Variational Bayesian Image Restoration Based on a Product of
     t-Distributions Image Prior." IEEE Trans. Image Process., 17(10), 1795–1805.

Provides:
    - FFT helpers (psf2otf, otf2psf).
    - RBF basis construction (Eq. (4)–(7) in Tzikas 2009).
    - Gradient filter operators Q^k (Sec. III-B in Tzikas 2009;
      fan-filters from Chantas 2008 [26]).
    - Cross-correlation matrices via FFT (Eq. (54) in Tzikas 2009).
    - Spectral covariance approximation (Eq. (54)–(55) in Tzikas 2009).
"""

import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPSILON = 1e-12


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Convert a point-spread function to an optical transfer function.

    The PSF is zero-padded to *shape* and circularly shifted so that its
    centre lands at index (0, 0) before taking the 2-D DFT.

    Parameters
    ----------
    psf : ndarray, shape (kh, kw)
        Point-spread function (kernel).
    shape : (H, W)
        Target image dimensions.

    Returns
    -------
    otf : ndarray, shape (H, W), complex
        Optical transfer function (frequency-domain PSF).
    """
    kh, kw = psf.shape
    buf = np.zeros(shape, dtype=psf.dtype)
    buf[:kh, :kw] = psf
    # Circular shift: centre of the PSF → origin
    buf = np.roll(buf, -(kh // 2), axis=0)
    buf = np.roll(buf, -(kw // 2), axis=1)
    return fft2(buf)


def otf2psf(otf: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """Convert an OTF back to a real-space PSF of size *out_shape*."""
    buf = np.real(ifft2(otf))
    kh, kw = out_shape
    buf = np.roll(buf, kh // 2, axis=0)
    buf = np.roll(buf, kw // 2, axis=1)
    return buf[:kh, :kw]


# ---------------------------------------------------------------------------
# RBF basis  (Tzikas 2009, Eq. (4)–(7))
# ---------------------------------------------------------------------------

def build_rbf_basis(kernel_shape: Tuple[int, int],
                    sigma_phi_sq: float = 0.1) -> np.ndarray:
    r"""Construct the Gaussian RBF basis matrix Φ ∈ R^{L×M}.

    Each column *m* corresponds to an RBF centred on pixel *c_m* of the
    kernel support.  Rows index the same set of pixels (so L = M = kh·kw).

        Φ_{l,m} = exp( −‖p_l − c_m‖² / (2 σ_φ²) )      (Eq. 6)

    Parameters
    ----------
    kernel_shape : (kh, kw)
        Spatial dimensions of the PSF support.
    sigma_phi_sq : float
        Bandwidth σ_φ² of the Gaussian RBF (default 0.1, Sec. IV-D).

    Returns
    -------
    Phi : ndarray, shape (L, M)  with L = M = kh*kw
    """
    kh, kw = kernel_shape
    # Integer grid of pixel-centre coordinates inside the PSF support
    ys = np.arange(kh, dtype=np.float64)
    xs = np.arange(kw, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')  # (kh, kw)
    coords = np.column_stack([grid_y.ravel(), grid_x.ravel()])  # (L, 2)

    # Squared pairwise distances  (L, L)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (L, L, 2)
    dist_sq = np.sum(diff ** 2, axis=-1)                        # (L, L)
    Phi = np.exp(-dist_sq / (2.0 * sigma_phi_sq))
    return Phi


# ---------------------------------------------------------------------------
# Gradient filter operators Q^k  (Tzikas 2009 Sec. III-B; Chantas 2008)
# ---------------------------------------------------------------------------

def build_gradient_filters(
    shape: Tuple[int, int],
    num_filters: int = 2
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Precompute gradient filters Q^k in the frequency domain.

    Parameters
    ----------
    shape : (H, W)
        Image dimensions.
    num_filters : {2, 4}
        K = 2 → horizontal & vertical first-order differences.
        K = 4 → add diagonal fan-filters (Chantas 2008, ref. [26]).

    Returns
    -------
    Q_fft : list of ndarray, length K
        FFT of each filter kernel (complex, shape (H, W)).
    Q_fft_sq : list of ndarray, length K
        |Q̂^k|² for each filter (real, shape (H, W)).
    """
    H, W = shape
    Q_fft: List[np.ndarray] = []
    Q_fft_sq: List[np.ndarray] = []

    # Q^1: horizontal forward difference  [−1, 1]
    d_h = np.zeros(shape)
    d_h[0, 0] = -1.0
    d_h[0, 1] = 1.0
    F_h = fft2(d_h)
    Q_fft.append(F_h)
    Q_fft_sq.append(np.abs(F_h) ** 2)

    # Q^2: vertical forward difference
    d_v = np.zeros(shape)
    d_v[0, 0] = -1.0
    d_v[1, 0] = 1.0
    F_v = fft2(d_v)
    Q_fft.append(F_v)
    Q_fft_sq.append(np.abs(F_v) ** 2)

    if num_filters >= 4:
        # Q^3: diagonal NW→SE  (Chantas 2008, fan-filter)
        d_d1 = np.zeros(shape)
        d_d1[0, 0] = 1.0
        d_d1[1, 1] = -1.0
        F_d1 = fft2(d_d1)
        Q_fft.append(F_d1)
        Q_fft_sq.append(np.abs(F_d1) ** 2)

        # Q^4: diagonal NE→SW
        d_d2 = np.zeros(shape)
        d_d2[0, 1] = 1.0
        d_d2[1, 0] = -1.0
        F_d2 = fft2(d_d2)
        Q_fft.append(F_d2)
        Q_fft_sq.append(np.abs(F_d2) ** 2)

    return Q_fft, Q_fft_sq


# ---------------------------------------------------------------------------
# PSF-support offset helpers
# ---------------------------------------------------------------------------

def _kernel_offsets(kernel_shape: Tuple[int, int]) -> np.ndarray:
    """Return (L, 2) array of (row, col) offsets for PSF pixels,
    centred so that the middle pixel has offset (0, 0)."""
    kh, kw = kernel_shape
    ys = np.arange(kh) - kh // 2
    xs = np.arange(kw) - kw // 2
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    return np.column_stack([gy.ravel(), gx.ravel()])  # (L, 2)


# ---------------------------------------------------------------------------
# Cross-correlation matrices via FFT  (Tzikas 2009 Eq. (54))
# ---------------------------------------------------------------------------

def compute_cross_correlation_matrix(
    mu_f: np.ndarray,
    kernel_shape: Tuple[int, int]
) -> np.ndarray:
    r"""Compute F̄^T F̄ ∈ R^{L×L} from the image mean μ_f.

    [F̄^T F̄]_{m,m'} = R_{μ_f}(p_m − p_{m'})                (Eq. 54, first term)

    where R_{μ_f} = IFFT(|FFT(μ_f)|²)  (autocorrelation via FFT).

    Parameters
    ----------
    mu_f : ndarray (H, W)
    kernel_shape : (kh, kw)

    Returns
    -------
    FtF : ndarray (L, L)
    """
    H, W = mu_f.shape
    offsets = _kernel_offsets(kernel_shape)          # (L, 2)

    F_mu = fft2(mu_f)
    R_mu = np.real(ifft2(np.abs(F_mu) ** 2))        # (H, W)

    # Vectorised pairwise differences: (L,1) − (1,L) → (L,L) index arrays
    dy = (offsets[:, 0, np.newaxis] - offsets[np.newaxis, :, 0]) % H
    dx = (offsets[:, 1, np.newaxis] - offsets[np.newaxis, :, 1]) % W
    return R_mu[dy, dx]


def compute_cross_correlation_vector(
    mu_f: np.ndarray,
    g: np.ndarray,
    kernel_shape: Tuple[int, int]
) -> np.ndarray:
    r"""Compute F̄^T g ∈ R^L  (cross-correlation of μ_f with observation g).

    [F̄^T g]_m = [xcorr(μ_f, g)](p_m)

    In the frequency domain:
        xcorr(μ_f, g) = IFFT( conj(FFT(μ_f)) · FFT(g) )

    Parameters
    ----------
    mu_f : ndarray (H, W)
    g    : ndarray (H, W)
    kernel_shape : (kh, kw)

    Returns
    -------
    Ftg : ndarray (L,)
    """
    H, W = mu_f.shape
    offsets = _kernel_offsets(kernel_shape)          # (L, 2)

    xcorr = np.real(ifft2(np.conj(fft2(mu_f)) * fft2(g)))  # (H, W)

    dy = offsets[:, 0] % H
    dx = offsets[:, 1] % W
    return xcorr[dy, dx]


# ---------------------------------------------------------------------------
# Covariance matrix D_f  (Tzikas 2009 Eq. (54), second term)
# ---------------------------------------------------------------------------

def compute_covariance_matrix_Df(
    r_f: np.ndarray,
    kernel_shape: Tuple[int, int]
) -> np.ndarray:
    r"""Build D_f ∈ R^{L×L} from the autocovariance r_f.

    [D_f]_{m,m'} = N · r_f(p_m − p_{m'})       (Eq. 54, second term)

    Parameters
    ----------
    r_f : ndarray (H, W)
        Autocovariance function  r_f = IFFT(Σ̂_f).
    kernel_shape : (kh, kw)

    Returns
    -------
    D_f : ndarray (L, L)
    """
    H, W = r_f.shape
    N = H * W
    offsets = _kernel_offsets(kernel_shape)          # (L, 2)

    # Vectorised pairwise differences: (L,1) − (1,L) → (L,L) index arrays
    dy = (offsets[:, 0, np.newaxis] - offsets[np.newaxis, :, 0]) % H
    dx = (offsets[:, 1, np.newaxis] - offsets[np.newaxis, :, 1]) % W
    return N * r_f[dy, dx]


# ---------------------------------------------------------------------------
# Spectral covariance approximations  (Tzikas 2009 Eq. (54)–(55))
# ---------------------------------------------------------------------------

def spectral_covariance(
    h_fft: np.ndarray,
    gamma_bar: np.ndarray,
    Q_fft_sq: List[np.ndarray],
    beta: float,
    prior_weight: float = 1.0,
) -> np.ndarray:
    r"""Spectral approximation of the image posterior covariance Σ_f.

    Σ̂_f(ω) = ( ⟨β⟩ |ĥ(ω)|² + (1/P) Σ_k γ̄^k |Q̂^k(ω)|² )⁻¹

    prior_weight = 1/P (Chantas 2008).  For Tzikas 2009, prior_weight = 1.0.

    Parameters
    ----------
    h_fft : ndarray (H, W), complex
    gamma_bar : array-like, length K
    Q_fft_sq : list of ndarray (H, W), length K
    beta : float
    prior_weight : float

    Returns
    -------
    Sigma_f_hat : ndarray (H, W), real
    """
    denom = beta * np.abs(h_fft) ** 2
    for k, gbar in enumerate(gamma_bar):
        denom = denom + prior_weight * gbar * Q_fft_sq[k]
    return 1.0 / (denom + EPSILON)


def spectral_diagonal(
    Q_fft_sq_k: np.ndarray,
    Sigma_f_hat: np.ndarray
) -> float:
    r"""Scalar approximation of diag(Q^k Σ_f (Q^k)^T).

    c_k = (1/N) Σ_ω |Q̂^k(ω)|² Σ̂_f(ω)                    (below Eq. 49)

    This gives a uniform (per-pixel constant) estimate of the diagonal
    entries [Q^k Σ_f (Q^k)^T]_{jj} ≈ c_k.

    Parameters
    ----------
    Q_fft_sq_k : ndarray (H, W)
    Sigma_f_hat : ndarray (H, W)

    Returns
    -------
    c_k : float
    """
    return float(np.mean(Q_fft_sq_k * Sigma_f_hat))


def compute_autocovariance(Sigma_f_hat: np.ndarray) -> np.ndarray:
    """Autocovariance of the image posterior:  r_f = IFFT(Σ̂_f)."""
    return np.real(ifft2(Sigma_f_hat))


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def build_initial_psf(kernel_shape: Tuple[int, int],
                      sigma_h_sq: float = 3.0) -> np.ndarray:
    """Gaussian PSF initialization (Tzikas 2009, Sec. IV-D).

    Parameters
    ----------
    kernel_shape : (kh, kw)
    sigma_h_sq : float
        Variance σ_h² of the initializing Gaussian (default 3).

    Returns
    -------
    h0 : ndarray (kh, kw), normalised to sum = 1.
    """
    kh, kw = kernel_shape
    cy, cx = kh // 2, kw // 2
    ys = np.arange(kh, dtype=np.float64) - cy
    xs = np.arange(kw, dtype=np.float64) - cx
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    h0 = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma_h_sq))
    h0 /= h0.sum()
    return h0


def initial_wiener(g: np.ndarray,
                   h: np.ndarray,
                   beta: float,
                   gamma_bar: np.ndarray,
                   Q_fft_sq: List[np.ndarray],
                   prior_weight: float = 1.0) -> np.ndarray:
    r"""Wiener-filter initialisation for μ_f  (Sec. IV-D / Eq. 60).

    μ_f = IFFT( conj(Ĥ)·Ĝ / (|Ĥ|² + (1/(βP)) Σ_k γ̄^k |Q̂^k|²) )

    Parameters
    ----------
    g : ndarray (H, W)
    h : ndarray (kh, kw)
    beta : float
    gamma_bar : ndarray (K,)
    Q_fft_sq : list of ndarray (H, W)
    prior_weight : float
        1.0 for Tzikas 2009; 1/K for Chantas 2008.

    Returns
    -------
    mu_f : ndarray (H, W)
    """
    H, W = g.shape
    G = fft2(g)
    Hf = psf2otf(h, (H, W))
    Hf_conj = np.conj(Hf)
    Hf_sq = np.abs(Hf) ** 2

    denom = Hf_sq
    for k in range(len(Q_fft_sq)):
        denom = denom + prior_weight * (gamma_bar[k] / (beta + EPSILON)) * Q_fft_sq[k]
    mu_f = np.real(ifft2(Hf_conj * G / (denom + EPSILON)))
    return mu_f


# ---------------------------------------------------------------------------
# Edge tapering  (mitigates circular-convolution boundary artefacts)
# ---------------------------------------------------------------------------

def edge_taper(g: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    """Apply MATLAB-style edge tapering to reduce boundary ringing.

    The image is blended with a locally blurred version at the borders
    using a smooth weight mask derived from the PSF autocorrelation.

    Parameters
    ----------
    g : ndarray (H, W)
        Input image in [0, 1].
    kernel_shape : (kh, kw)
        PSF support size (used to determine taper width).

    Returns
    -------
    g_tapered : ndarray (H, W)
    """
    H, W = g.shape
    kh, kw = kernel_shape
    # Half-widths for taper region
    ph, pw = kh // 2, kw // 2

    # Build 1-D cosine taper [0 → 1] over the border region
    taper_h = np.ones(H, dtype=np.float64)
    taper_w = np.ones(W, dtype=np.float64)
    if ph > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(ph) / ph))
        taper_h[:ph] = ramp
        taper_h[-ph:] = ramp[::-1]
    if pw > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(pw) / pw))
        taper_w[:pw] = ramp
        taper_w[-pw:] = ramp[::-1]

    mask = taper_h[:, None] * taper_w[None, :]   # (H, W) in [0, 1]

    # Blend: g_tapered = mask * g + (1 - mask) * mean(g)
    g_mean = float(np.mean(g))
    g_tapered = mask * g + (1.0 - mask) * g_mean
    return g_tapered


