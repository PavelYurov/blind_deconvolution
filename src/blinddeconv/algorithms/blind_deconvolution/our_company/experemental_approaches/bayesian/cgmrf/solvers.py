import numpy as np
from .utils import (
    psf_to_otf,
    fft_convolve,
    apply_tv_precision_operator,
    tv_quadratic_form,
    gradient_power_spectrum,
    extract_centered_kernel,
    project_kernel,
    compute_mm_weights,
    hutchinson_trace_estimate,
    spectral_trace,
    spectral_log_det,
    forward_diff_h,
    forward_diff_v,
)

def compute_tv_weights(f: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    return compute_mm_weights(f, epsilon)

def solve_image_vb(y: np.ndarray,
                   h: np.ndarray,
                   f_init: np.ndarray,
                   alpha: float,
                   beta: float,
                   w: np.ndarray,
                   cg_max_iter: int = 50,
                   cg_tol: float = 1e-6,
                   n_trace_probes: int = 5,
                   rng: np.random.Generator = None,
                   ) -> tuple:

    shape = y.shape
    H_otf = psf_to_otf(h, shape)
    H_otf_sq = np.abs(H_otf)**2

    Dh_sq, Dv_sq = gradient_power_spectrum(shape)

    w_mean = np.mean(w)
    Q_spec = w_mean * (Dh_sq + Dv_sq)

    M_diag = beta * H_otf_sq + alpha * Q_spec
    M_inv = 1.0 / np.maximum(M_diag, 1e-12)

    def matvec_A(v: np.ndarray) -> np.ndarray:
        Fv = np.fft.fft2(v)
        HtHv = np.real(np.fft.ifft2(H_otf_sq * Fv))
        Qv = apply_tv_precision_operator(v, w)
        return beta * HtHv + alpha * Qv

    def precondition(v: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(M_inv * np.fft.fft2(v)))

    F_y = np.fft.fft2(y)
    b = beta * np.real(np.fft.ifft2(np.conj(H_otf) * F_y))

    f = f_init.copy()
    r = b - matvec_A(f)
    z = precondition(r)
    p = z.copy()
    rz = np.sum(r * z)

    for _ in range(cg_max_iter):
        Ap = matvec_A(p)
        pAp = np.sum(p * Ap)
        if pAp <= 1e-30:
            break

        step = rz / pAp
        f = f + step * p
        r = r - step * Ap

        if np.sqrt(np.sum(r**2)) < cg_tol:
            break

        z = precondition(r)
        rz_new = np.sum(r * z)
        if rz_new <= 1e-30:
            break

        beta_cg = rz_new / rz
        p = z + beta_cg * p
        rz = rz_new

    f = np.clip(f, 0.0, 1.0)

    if n_trace_probes > 0:
        def spectral_Ainv(v): return precondition(v)
        def matvec_Q(v): return apply_tv_precision_operator(v, w)
        def matvec_HtH(v): return np.real(np.fft.ifft2(H_otf_sq * np.fft.fft2(v)))

        tr_Sigma_Q = hutchinson_trace_estimate(spectral_Ainv, shape, matvec_Q, n_trace_probes, rng)
        tr_Sigma_HtH = hutchinson_trace_estimate(spectral_Ainv, shape, matvec_HtH, n_trace_probes, rng)
    else:
        tr_Sigma_Q = spectral_trace(H_otf_sq, Q_spec, alpha, beta, Q_spec)
        tr_Sigma_HtH = spectral_trace(H_otf_sq, Q_spec, alpha, beta, H_otf_sq)

    log_det_Sigma = -spectral_log_det(H_otf_sq, Q_spec, alpha, beta)

    info = {
        'tr_Sigma_Q': max(0.0, float(tr_Sigma_Q)),
        'tr_Sigma_HtH': max(0.0, float(tr_Sigma_HtH)),
        'log_det_Sigma': float(log_det_Sigma),
    }

    return f, info

def solve_kernel_vb(y: np.ndarray,
                    f: np.ndarray,
                    h_shape: tuple,
                    delta_h: float,
                    beta: float,
                    threshold_ratio: float = 0.05) -> tuple:

    shape = y.shape
    kh, kw = h_shape
    N_pixels = float(shape[0] * shape[1])
    K_pixels = float(kh * kw)

    dy_h = forward_diff_h(y)
    dy_v = forward_diff_v(y)
    df_h = forward_diff_h(f)
    df_v = forward_diff_v(f)

    F_df_h = np.fft.fft2(df_h)
    F_df_v = np.fft.fft2(df_v)
    F_dy_h = np.fft.fft2(dy_h)
    F_dy_v = np.fft.fft2(dy_v)

    F_df_sq = np.abs(F_df_h)**2 + np.abs(F_df_v)**2

    numer = np.conj(F_df_h) * F_dy_h + np.conj(F_df_v) * F_dy_v

    denom = beta * F_df_sq + delta_h

    H_est_freq = beta * numer / np.maximum(denom, 1e-12)
    h_full = np.real(np.fft.ifft2(H_est_freq))

    h_raw = extract_centered_kernel(h_full, h_shape)

    max_val = np.max(h_raw)
    if max_val > 0:

        h_raw[h_raw < threshold_ratio * max_val] = 0.0

    h_energy = float(np.sum(h_raw**2))

    full_grid_trace = np.sum(1.0 / np.maximum(denom, 1e-12))
    h_cov_trace = float(full_grid_trace) * (K_pixels / N_pixels)

    h_proj = project_kernel(h_raw)

    info = {
        'h_energy': h_energy,
        'h_cov_trace': h_cov_trace,
    }

    return h_proj, info

def update_hyperparameters_vb(y: np.ndarray,
                              f: np.ndarray,
                              h: np.ndarray,
                              w: np.ndarray,
                              alpha: float,
                              beta: float,
                              delta_h: float,
                              tr_Sigma_Q: float,
                              tr_Sigma_HtH: float,
                              h_energy: float,
                              h_cov_trace: float) -> tuple:

    N = float(y.size)
    K = float(h.size)

    fQf = tv_quadratic_form(f, w)
    denom_alpha = fQf + tr_Sigma_Q
    alpha_new = N / max(denom_alpha, 1e-10)

    residual = y - fft_convolve(f, h)
    res_sq = float(np.sum(residual**2))
    denom_beta = res_sq + tr_Sigma_HtH
    beta_new = N / max(denom_beta, 1e-10)

    denom_delta = h_energy + h_cov_trace
    delta_h_new = K / max(denom_delta, 1e-10)

    alpha_new = float(np.clip(alpha_new, 1e-3, 1e6))
    beta_new = float(np.clip(beta_new, 1.0, 1e7))
    delta_h_new = float(np.clip(delta_h_new, 1e-4, 1e8))

    return alpha_new, beta_new, delta_h_new
