import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple, List

EPSILON = 1e-12

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:

    kh, kw = psf.shape
    buf = np.zeros(shape, dtype=psf.dtype)
    buf[:kh, :kw] = psf

    buf = np.roll(buf, -(kh // 2), axis=0)
    buf = np.roll(buf, -(kw // 2), axis=1)
    return fft2(buf)

def otf2psf(otf: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:

    buf = np.real(ifft2(otf))
    kh, kw = out_shape
    buf = np.roll(buf, kh // 2, axis=0)
    buf = np.roll(buf, kw // 2, axis=1)
    return buf[:kh, :kw]

def build_rbf_basis(kernel_shape: Tuple[int, int],
                    sigma_phi_sq: float = 0.1) -> np.ndarray:

    kh, kw = kernel_shape

    ys = np.arange(kh, dtype=np.float64)
    xs = np.arange(kw, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
    coords = np.column_stack([grid_y.ravel(), grid_x.ravel()])

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    Phi = np.exp(-dist_sq / (2.0 * sigma_phi_sq))
    return Phi

def build_gradient_filters(
    shape: Tuple[int, int],
    num_filters: int = 2
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    H, W = shape
    Q_fft: List[np.ndarray] = []
    Q_fft_sq: List[np.ndarray] = []

    d_h = np.zeros(shape)
    d_h[0, 0] = -1.0
    d_h[0, 1] = 1.0
    F_h = fft2(d_h)
    Q_fft.append(F_h)
    Q_fft_sq.append(np.abs(F_h) ** 2)

    d_v = np.zeros(shape)
    d_v[0, 0] = -1.0
    d_v[1, 0] = 1.0
    F_v = fft2(d_v)
    Q_fft.append(F_v)
    Q_fft_sq.append(np.abs(F_v) ** 2)

    if num_filters >= 4:

        d_d1 = np.zeros(shape)
        d_d1[0, 0] = 1.0
        d_d1[1, 1] = -1.0
        F_d1 = fft2(d_d1)
        Q_fft.append(F_d1)
        Q_fft_sq.append(np.abs(F_d1) ** 2)

        d_d2 = np.zeros(shape)
        d_d2[0, 1] = 1.0
        d_d2[1, 0] = -1.0
        F_d2 = fft2(d_d2)
        Q_fft.append(F_d2)
        Q_fft_sq.append(np.abs(F_d2) ** 2)

    return Q_fft, Q_fft_sq

def _kernel_offsets(kernel_shape: Tuple[int, int]) -> np.ndarray:

    kh, kw = kernel_shape
    ys = np.arange(kh) - kh // 2
    xs = np.arange(kw) - kw // 2
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    return np.column_stack([gy.ravel(), gx.ravel()])

def compute_cross_correlation_matrix(
    mu_f: np.ndarray,
    kernel_shape: Tuple[int, int]
) -> np.ndarray:

    H, W = mu_f.shape
    offsets = _kernel_offsets(kernel_shape)

    F_mu = fft2(mu_f)
    R_mu = np.real(ifft2(np.abs(F_mu) ** 2))

    dy = (offsets[:, 0, np.newaxis] - offsets[np.newaxis, :, 0]) % H
    dx = (offsets[:, 1, np.newaxis] - offsets[np.newaxis, :, 1]) % W
    return R_mu[dy, dx]

def compute_cross_correlation_vector(
    mu_f: np.ndarray,
    g: np.ndarray,
    kernel_shape: Tuple[int, int]
) -> np.ndarray:

    H, W = mu_f.shape
    offsets = _kernel_offsets(kernel_shape)

    xcorr = np.real(ifft2(np.conj(fft2(mu_f)) * fft2(g)))

    dy = offsets[:, 0] % H
    dx = offsets[:, 1] % W
    return xcorr[dy, dx]

def compute_covariance_matrix_Df(
    r_f: np.ndarray,
    kernel_shape: Tuple[int, int]
) -> np.ndarray:

    H, W = r_f.shape
    N = H * W
    offsets = _kernel_offsets(kernel_shape)

    dy = (offsets[:, 0, np.newaxis] - offsets[np.newaxis, :, 0]) % H
    dx = (offsets[:, 1, np.newaxis] - offsets[np.newaxis, :, 1]) % W
    return N * r_f[dy, dx]

def spectral_covariance(
    h_fft: np.ndarray,
    gamma_bar: np.ndarray,
    Q_fft_sq: List[np.ndarray],
    beta: float,
    prior_weight: float = 1.0,
) -> np.ndarray:

    denom = beta * np.abs(h_fft) ** 2
    for k, gbar in enumerate(gamma_bar):
        denom = denom + prior_weight * gbar * Q_fft_sq[k]
    return 1.0 / (denom + EPSILON)

def spectral_diagonal(
    Q_fft_sq_k: np.ndarray,
    Sigma_f_hat: np.ndarray
) -> float:

    return float(np.mean(Q_fft_sq_k * Sigma_f_hat))

def compute_autocovariance(Sigma_f_hat: np.ndarray) -> np.ndarray:

    return np.real(ifft2(Sigma_f_hat))

def build_initial_psf(kernel_shape: Tuple[int, int],
                      sigma_h_sq: float = 3.0) -> np.ndarray:

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

def edge_taper(g: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:

    H, W = g.shape
    kh, kw = kernel_shape

    ph, pw = kh // 2, kw // 2

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

    mask = taper_h[:, None] * taper_w[None, :]

    g_mean = float(np.mean(g))
    g_tapered = mask * g + (1.0 - mask) * g_mean
    return g_tapered
