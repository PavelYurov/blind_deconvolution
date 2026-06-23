import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from typing import Tuple, List, Dict

def conv2(img: np.ndarray, kernel: np.ndarray, mode: str = 'full') -> np.ndarray:

    return fftconvolve(img, kernel, mode=mode)

def rot180(arr: np.ndarray) -> np.ndarray:

    return arr[::-1, ::-1].copy()

def grad_tv(f: np.ndarray) -> np.ndarray:

    eps = 1e-3

    f_shift_down = np.pad(f[1:, :], ((0, 1), (0, 0)), mode='edge')
    dx_fwd = f_shift_down - f

    f_shift_right = np.pad(f[:, 1:], ((0, 0), (0, 1)), mode='edge')
    dy_fwd = f_shift_right - f

    f_up = np.pad(f[:-1, :], ((1, 0), (0, 0)), mode='edge')

    f_left = np.pad(f[:, :-1], ((0, 0), (1, 0)), mode='edge')

    src_dx_jm1 = f[1:, :-1]
    f_down_left = np.pad(src_dx_jm1, ((0, 1), (1, 0)), mode='edge')
    dx_at_jminus1 = f_down_left - f_left

    src_dy_im1 = f[:-1, 1:]
    f_up_right = np.pad(src_dy_im1, ((1, 0), (0, 1)), mode='edge')
    dy_at_iminus1 = f_up_right - f_up

    dx_bwd = f - f_up
    dy_bwd = f - f_left

    phi_ij = np.maximum(np.sqrt(dx_fwd ** 2 + dy_fwd ** 2), eps)

    phi_im1_j = np.maximum(np.sqrt(dx_bwd ** 2 + dy_at_iminus1 ** 2), eps)

    phi_i_jm1 = np.maximum(np.sqrt(dx_at_jminus1 ** 2 + dy_bwd ** 2), eps)

    result = ((dx_fwd + dy_fwd) / phi_ij
              - dx_bwd / phi_im1_j
              - dy_bwd / phi_i_jm1)

    return result

def ensure_odd(n: int) -> int:

    return n if n % 2 != 0 else n - 1

def resize_2d(
    arr: np.ndarray,
    target_shape: Tuple[int, int],
    order: int = 1,
) -> np.ndarray:

    if arr.shape == tuple(target_shape):
        return arr.copy()
    factors = (target_shape[0] / arr.shape[0],
               target_shape[1] / arr.shape[1])
    return zoom(arr, factors, order=order)

def build_pyramid(
    image: np.ndarray,
    mk: int,
    nk: int,
    lambda_val: float,
    lambda_multiplier: float,
    max_lambda: float,
    scale_multiplier: float,
) -> List[Dict]:

    H, W = image.shape
    smallest = 3

    n_scales = 1
    mk_cur, nk_cur, lam_cur = float(mk), float(nk), lambda_val

    while (mk_cur > smallest
           and nk_cur > smallest
           and lam_cur * lambda_multiplier < max_lambda):
        n_scales += 1
        mk_cur = round(mk_cur / scale_multiplier)
        nk_cur = round(nk_cur / scale_multiplier)
        mk_cur = max(ensure_odd(int(mk_cur)), smallest)
        nk_cur = max(ensure_odd(int(nk_cur)), smallest)
        lam_cur *= lambda_multiplier

    pyramid: List[Dict] = [None] * n_scales
    pyramid[0] = {
        'image': image.copy(),
        'M': H, 'N': W,
        'MK': mk, 'NK': nk,
        'lambda': lambda_val,
    }

    for s in range(1, n_scales):
        prev = pyramid[s - 1]
        lam_s = prev['lambda'] * lambda_multiplier

        mk_s = int(round(prev['MK'] / scale_multiplier))
        nk_s = int(round(prev['NK'] / scale_multiplier))
        mk_s = ensure_odd(mk_s)
        nk_s = ensure_odd(nk_s)

        if nk_s == prev['NK']:
            nk_s -= 2
        if mk_s == prev['MK']:
            mk_s -= 2

        mk_s = max(mk_s, smallest)
        nk_s = max(nk_s, smallest)

        factor_m = prev['MK'] / mk_s
        factor_n = prev['NK'] / nk_s
        m_s = ensure_odd(int(round(prev['M'] / factor_m)))
        n_s = ensure_odd(int(round(prev['N'] / factor_n)))

        img_s = resize_2d(image, (m_s, n_s), order=1)

        pyramid[s] = {
            'image': img_s,
            'M': m_s, 'N': n_s,
            'MK': mk_s, 'NK': nk_s,
            'lambda': lam_s,
        }

    return pyramid
