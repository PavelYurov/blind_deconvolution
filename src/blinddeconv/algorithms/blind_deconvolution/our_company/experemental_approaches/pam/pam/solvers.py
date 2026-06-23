import numpy as np

from .utils import (
    conv2fft,
    conv2_matlab,
    grad_tv_cc,
    gamma_correction,
    imresize,
    build_pyramid,
)

_FFT_THRESHOLD = 41 ** 2

def _padarray_replicate(f: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:

    if f.ndim == 2:
        return np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        return np.pad(f, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')

def _conv2(a: np.ndarray, b: np.ndarray, mode: str = 'full') -> np.ndarray:

    if b.size > _FFT_THRESHOLD:
        return conv2fft(a, b, mode)
    else:
        return conv2_matlab(a, b, mode)

def blind(f: np.ndarray, MK: int, NK: int,
          lam: float = 3e-4,
          u: np.ndarray = None,
          k: np.ndarray = None,
          iters: int = 1000,
          visualize: bool = False) -> tuple:

    squeeze_out = (f.ndim == 2)
    if f.ndim == 2:
        f = f[:, :, np.newaxis]

    M, N, C = f.shape

    if u is None:
        u = _padarray_replicate(f, MK // 2, NK // 2)
    elif u.ndim == 2 and C > 1:
        u = u[:, :, np.newaxis]

    if k is None:
        k = np.ones((MK, NK), dtype=np.float64) / (MK * NK)

    if u.ndim == 2:
        u = u[:, :, np.newaxis]

    MU = M + MK - 1
    NU = N + NK - 1

    k_rot = np.rot90(k, 2)

    for it in range(1, iters + 1):

        gradudata = np.zeros((MU, NU, C), dtype=np.float64)
        for c in range(C):

            residual = _conv2(u[:, :, c], k, 'valid') - f[:, :, c]
            gradudata[:, :, c] = _conv2(residual, k_rot, 'full')

        gradu = gradudata - lam * grad_tv_cc(u)

        sf = 5e-3 * np.max(u) / max(1e-31, np.max(np.abs(gradu)))
        u = u - sf * gradu

        gradk = np.zeros((MK, NK), dtype=np.float64)
        for c in range(C):

            if k.size > _FFT_THRESHOLD:
                inner = conv2fft(u[:, :, c], k, 'valid') - f[:, :, c]
            else:
                inner = conv2_matlab(u[:, :, c], k, 'valid') - f[:, :, c]
            gradk += conv2fft(np.rot90(u[:, :, c], 2), inner, 'valid')

        sh = 1e-3 * np.max(k) / max(1e-31, np.max(np.abs(gradk)))
        k = k - sh * gradk

        k = k * (k > 0)
        k_sum = k.sum()
        if k_sum > 0:
            k = k / k_sum

        k_rot = np.rot90(k, 2)

    if squeeze_out:
        u = u[:, :, 0]

    return u, k

def dec(f: np.ndarray, k: np.ndarray,
        lam: float = 3e-4,
        u: np.ndarray = None,
        iters: int = 1000,
        visualize: bool = False) -> np.ndarray:

    squeeze_out = (f.ndim == 2)
    if f.ndim == 2:
        f = f[:, :, np.newaxis]

    M, N, C = f.shape
    MK, NK = k.shape

    if u is None:
        u = _padarray_replicate(f, MK // 2, NK // 2)
    elif u.ndim == 2 and C > 1:
        u = u[:, :, np.newaxis]
    if u.ndim == 2:
        u = u[:, :, np.newaxis]

    MU = M + MK - 1
    NU = N + NK - 1

    k_rot = np.rot90(k, 2)

    for it in range(1, iters + 1):
        gradudata = np.zeros((MU, NU, C), dtype=np.float64)
        for c in range(C):

            if k.size > _FFT_THRESHOLD:
                residual = conv2fft(u[:, :, c], k, 'valid') - f[:, :, c]
                gradudata[:, :, c] = conv2fft(residual, k_rot, 'full')
            else:
                residual = conv2_matlab(u[:, :, c], k, 'valid') - f[:, :, c]
                gradudata[:, :, c] = conv2_matlab(residual, k_rot, 'full')

        gradu = gradudata - lam * grad_tv_cc(u)

        sf = 5e-3 * np.max(u) / max(1e-31, np.max(np.abs(gradu)))
        u = u - sf * gradu

    if squeeze_out:
        u = u[:, :, 0]

    return u

def coarse_to_fine(f: np.ndarray, MK: int, NK: int,
                   blind_iters: int = 1000,
                   visualize: bool = False,
                   final_lambda: float = 3e-4,
                   lambda_multiplier: float = 1.9,
                   max_lambda: float = 0.11,
                   kernel_size_multiplier: float = 1.1,
                   interp_method: str = 'bicubic') -> tuple:

    u = _padarray_replicate(f, MK // 2, NK // 2)
    k = np.ones((MK, NK), dtype=np.float64) / (MK * NK)

    fp, Mp, Np, MKp, NKp, lambdas, num_scales = build_pyramid(
        f, MK, NK, final_lambda, lambda_multiplier,
        interp_method, kernel_size_multiplier, max_lambda,
    )

    for scale_idx in range(num_scales - 1, -1, -1):
        Ms = Mp[scale_idx]
        Ns = Np[scale_idx]
        MKs = MKp[scale_idx]
        NKs = NKp[scale_idx]

        u_target_h = Ms + MKs - 1
        u_target_w = Ns + NKs - 1
        u = imresize(u, (u_target_h, u_target_w), method=interp_method)

        k = imresize(k, (MKs, NKs), method=interp_method)

        k = k * (k > 0)
        k_sum = k.sum()
        if k_sum > 0:
            k = k / k_sum

        fs = fp[scale_idx]
        lam = lambdas[scale_idx]

        if visualize:
            print(f"scale: {scale_idx + 1}  lambda: {lam:.6f}  "
                  f"MKs: {MKs}  NKs: {NKs}  iters: {blind_iters}")

        u_pre_blind = u.copy()

        _, k = blind(fs, MKs, NKs,
                     lam=lam, u=u, k=k,
                     iters=blind_iters, visualize=visualize)

        u = dec(fs, k, lam=lam, u=u_pre_blind,
                iters=blind_iters, visualize=visualize)

    return u, k

def deblur(f: np.ndarray, MK: int, NK: int,
           lam: float = 3e-4,
           iters: int = 1000,
           gamma_correct: bool = False,
           gamma: float = 1.0,
           visualize: bool = False) -> tuple:

    f = f.astype(np.float64)
    if f.max() > 1.0:
        f = f / 255.0

    M, N = f.shape[:2]
    if M % 2 == 0:
        f = f[:-1, ...]
    if N % 2 == 0:
        f = f[:, :-1, ...]

    if gamma_correct:
        f = gamma_correction(f, gamma)

    u, k = coarse_to_fine(
        f, MK, NK,
        blind_iters=iters,
        visualize=visualize,
        final_lambda=lam,
        lambda_multiplier=1.9,
        max_lambda=0.11,
        kernel_size_multiplier=1.1,
        interp_method='bicubic',
    )

    return u, k
