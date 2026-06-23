import numpy as np
from typing import Tuple

from .utils import (
    conv2,
    rot180,
    grad_tv,
    resize_2d,
    build_pyramid,
)

def prida_single_scale(
    f: np.ndarray,
    u: np.ndarray,
    k: np.ndarray,
    lambda_val: float,
    n_iters: int,
    kernel_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:

    _EPS_FLOAT = np.finfo(np.float64).eps
    _CLAMP     = 1000.0
    _STEP_INIT = 1e-3

    for _ in range(n_iters):

        residual = conv2(u, k, mode='valid') - f
        k_rot    = rot180(k)
        grad_data_u = conv2(residual, k_rot, mode='full')

        tv_term = grad_tv(u)

        grad_u = grad_data_u - lambda_val * tv_term

        max_u  = np.max(np.abs(u))
        max_gu = np.max(np.abs(grad_u))
        step_u = _STEP_INIT * max_u / max(1e-31, max_gu)

        u_new = u - step_u * grad_u

        residual_k = conv2(u, k, mode='valid') - f
        u_rot      = rot180(u)
        grad_k     = conv2(u_rot, residual_k, mode='valid')

        max_k  = np.max(np.abs(k))
        max_gk = np.max(np.abs(grad_k))
        step_k = _STEP_INIT * max_k / max(1e-31, max_gk)

        eta = step_k / (k + _EPS_FLOAT)

        exp_term = np.exp(-eta * grad_k)
        exp_term = np.minimum(exp_term, _CLAMP)

        k_new = k * exp_term
        k_sum = np.sum(k_new)
        if k_sum > 0:
            k_new /= k_sum
        else:
            k_new = np.ones_like(k) / k.size

        u = u_new
        k = k_new

    return u, k

def coarse_to_fine(
    image: np.ndarray,
    kernel_shape: Tuple[int, int],
    lambda_val: float,
    n_iters: int = 1000,
    lambda_multiplier: float = 1.9,
    max_lambda: float = 0.11,
    scale_multiplier: float = 1.1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    MK, NK = kernel_shape

    pad_top  = MK // 2
    pad_left = NK // 2
    u = np.pad(image, ((pad_top, pad_top), (pad_left, pad_left)), mode='edge')

    k = np.ones((MK, NK), dtype=np.float64) / (MK * NK)

    pyramid = build_pyramid(
        image, MK, NK,
        lambda_val, lambda_multiplier, max_lambda, scale_multiplier,
    )
    n_scales = len(pyramid)

    if verbose:
        print(f"[PRIDA] Pyramid: {n_scales} scale(s), "
              f"Image: {image.shape[0]}×{image.shape[1]}, "
              f"Kernel: {MK}×{NK}")

    for idx in range(n_scales - 1, -1, -1):
        level = pyramid[idx]
        f_s  = level['image']
        M_s  = level['M']
        N_s  = level['N']
        MK_s = level['MK']
        NK_s = level['NK']
        lam_s = level['lambda']

        u_target_h = M_s + MK_s - 1
        u_target_w = N_s + NK_s - 1
        u = resize_2d(u, (u_target_h, u_target_w), order=1)
        k = resize_2d(k, (MK_s, NK_s), order=1)

        k_sum = np.sum(k)
        if k_sum > 0:
            k /= k_sum
        else:
            k = np.ones((MK_s, NK_s), dtype=np.float64) / (MK_s * NK_s)

        if verbose:
            print(f"  Scale {n_scales - idx}/{n_scales}: "
                  f"img {M_s}×{N_s}, ker {MK_s}×{NK_s}, λ={lam_s:.5f}")

        u, k = prida_single_scale(
            f_s, u, k, lam_s, n_iters, (MK_s, NK_s),
        )

    return u, k
