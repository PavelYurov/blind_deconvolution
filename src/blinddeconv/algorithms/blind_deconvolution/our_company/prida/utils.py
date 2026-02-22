"""
Utility functions for the PRIDA blind deconvolution algorithm.

Provides low-level primitives for:
    - 2D convolution (FFT-based, matching MATLAB/OpenCV conv2 semantics).
    - Isotropic Total Variation gradient (finite-difference discretization).
    - Multi-scale image–kernel pyramid construction.
    - Image resizing and dimension utilities.

Reference
Ravi, S. N., Mehta, R., & Singh, V. (2018).
"Robust Blind Deconvolution via Mirror Descent."
arXiv:1803.08137 [cs.CV].

Original C++ implementation: main.cpp by Tianyi Shan (2018).
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from typing import Tuple, List, Dict

#  Convolution Primitives
def conv2(img: np.ndarray, kernel: np.ndarray, mode: str = 'full') -> np.ndarray:
    """
    2D convolution via FFT.

    Equivalent to MATLAB ``conv2(img, kernel, mode)`` and to the OpenCV-based
    ``conv2()`` in the original C++ code (which flips the kernel and applies
    ``filter2D``, i.e.  correlation with a flipped kernel = convolution).

    Parameters
    img : np.ndarray
        Input 2D array (image or feature map).
    kernel : np.ndarray
        Convolution kernel (2D).
    mode : {'full', 'valid'}
        ``'full'``  — output shape = ``img.shape + kernel.shape - 1``.
        ``'valid'`` — output shape = ``max(shapes) - min(shapes) + 1``
        (only positions where both arrays fully overlap).

    Returns
    np.ndarray
        Convolution result.
    """
    return fftconvolve(img, kernel, mode=mode)


def rot180(arr: np.ndarray) -> np.ndarray:
    """
    Rotate a 2D array by 180 degrees (flip both axes).

    Equivalent to ``cv::ROTATE_180`` in OpenCV or
    ``np.flipud(np.fliplr(arr))``.

    Parameters
    arr : np.ndarray
        2D array.

    Returns
    np.ndarray
        Rotated copy (contiguous in memory).
    """
    return arr[::-1, ::-1].copy()

#  Total Variation Gradient

def grad_tv(f: np.ndarray) -> np.ndarray:
    r"""
    Negative gradient of the isotropic Total Variation functional.

    Computes  :math:`\operatorname{div}(\nabla f / |\nabla f|)`, the
    Euler–Lagrange operator for the isotropic TV norm:

    .. math::
        \mathrm{TV}(u)
        = \sum_{i,j} \sqrt{(D_x^{+} u_{i,j})^2 + (D_y^{+} u_{i,j})^2}

    where :math:`D_x^{+}, D_y^{+}` are forward finite differences.
    The result equals :math:`-\partial \mathrm{TV}/\partial u`, i.e. the
    *smoothing direction* used in the image update step.

    Boundary conditions are Neumann (replicate / zero-gradient), implemented
    via ``np.pad(..., mode='edge')``.

    The discretization yields three gradient magnitudes:

    * :math:`\Phi(i,j)`:  forward differences at :math:`(i,j)`.
    * :math:`\Phi(i{-}1,j)`:  backward-x and cross-y differences.
    * :math:`\Phi(i,j{-}1)`:  cross-x and backward-y differences.

    Assembly (Eq. 4 in Ravi et al., 2018 / gradTVcc in main.cpp):

    .. math::
        -\frac{\partial\mathrm{TV}}{\partial u_{i,j}}
        = \frac{D_x^{+} + D_y^{+}}{\Phi(i,j)}
          - \frac{D_x^{-}}{\Phi(i{-}1,j)}
          - \frac{D_y^{-}}{\Phi(i,j{-}1)}

    Parameters
    f : np.ndarray, shape (H, W)
        Input grayscale image (float64).

    Returns
    np.ndarray, shape (H, W)
        :math:`-\partial \mathrm{TV}/\partial u`.

    Reference
    C++ source: ``gradTVcc()`` in main.cpp, lines 100–185.
    """
    eps = 1e-3  # avoids division by zero in flat regions

    # Forward differences at position (i, j)
    # D_x^+[i,j] = f[i+1,j] - f[i,j],  Neumann BC means D = 0 at bottom
    f_shift_down = np.pad(f[1:, :], ((0, 1), (0, 0)), mode='edge')
    dx_fwd = f_shift_down - f

    # D_y^+[i,j] = f[i,j+1] - f[i,j],  Neumann BC means D = 0 at right
    f_shift_right = np.pad(f[:, 1:], ((0, 0), (0, 1)), mode='edge')
    dy_fwd = f_shift_right - f

    # Shifted values for backward / cross terms
    # f[i-1, j] with replicate at top row
    f_up = np.pad(f[:-1, :], ((1, 0), (0, 0)), mode='edge')
    # f[i, j-1] with replicate at left column
    f_left = np.pad(f[:, :-1], ((0, 0), (1, 0)), mode='edge')

    # Cross-differences for shifted magnitudes
    # D_x^+[i, j-1] = f[i+1, j-1] - f[i, j-1]
    #   Source: f[1:, :-1]  (shape (H-1)x(W-1)), pad bottom+left → (H, W)
    src_dx_jm1 = f[1:, :-1]
    f_down_left = np.pad(src_dx_jm1, ((0, 1), (1, 0)), mode='edge')
    dx_at_jminus1 = f_down_left - f_left

    # D_y^+[i-1, j] = f[i-1, j+1] - f[i-1, j]
    #   Source: f[:-1, 1:]  (shape (H-1)x(W-1)), pad top+right → (H, W)
    src_dy_im1 = f[:-1, 1:]
    f_up_right = np.pad(src_dy_im1, ((1, 0), (0, 1)), mode='edge')
    dy_at_iminus1 = f_up_right - f_up

    # Backward differences at (i, j)
    dx_bwd = f - f_up     # f[i,j] - f[i-1,j]  (= D_x^+(i-1,j) from i's view)
    dy_bwd = f - f_left   # f[i,j] - f[i,j-1]  (= D_y^+(i,j-1) from j's view)

    # Gradient magnitudes Φ
    # Φ(i, j)   — forward differences at (i, j)
    phi_ij = np.maximum(np.sqrt(dx_fwd ** 2 + dy_fwd ** 2), eps)

    # Φ(i-1, j) — uses D_x^+(i-1,j) = dx_bwd  and  D_y^+(i-1,j) = dy_at_iminus1
    phi_im1_j = np.maximum(np.sqrt(dx_bwd ** 2 + dy_at_iminus1 ** 2), eps)

    # Φ(i, j-1) — uses D_x^+(i,j-1) = dx_at_jminus1  and  D_y^+(i,j-1) = dy_bwd
    phi_i_jm1 = np.maximum(np.sqrt(dx_at_jminus1 ** 2 + dy_bwd ** 2), eps)

    # Assemble: -∂TV/∂u  =  div(∇f / |∇f|)
    result = ((dx_fwd + dy_fwd) / phi_ij
              - dx_bwd / phi_im1_j
              - dy_bwd / phi_i_jm1)

    return result

#  Dimension / Resize Helpers

def ensure_odd(n: int) -> int:
    """Subtract 1 from *n* if it is even so that kernel support is symmetric."""
    return n if n % 2 != 0 else n - 1


def resize_2d(
    arr: np.ndarray,
    target_shape: Tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """
    Resize a 2D array to *target_shape* via spline interpolation.

    Parameters
    arr : np.ndarray, shape (H, W)
        Source array.
    target_shape : (int, int)
        Desired (rows, cols).
    order : int
        Interpolation order (1 = bilinear, matching ``cv::INTER_LINEAR``).

    Returns
    np.ndarray
    """
    if arr.shape == tuple(target_shape):
        return arr.copy()
    factors = (target_shape[0] / arr.shape[0],
               target_shape[1] / arr.shape[1])
    return zoom(arr, factors, order=order)

#  Multi-Scale Pyramid

def build_pyramid(
    image: np.ndarray,
    mk: int,
    nk: int,
    lambda_val: float,
    lambda_multiplier: float,
    max_lambda: float,
    scale_multiplier: float,
) -> List[Dict]:
    """
    Construct a coarse-to-fine Gaussian pyramid for multi-scale blind
    deconvolution.

    At each coarser level the kernel size is reduced by *scale_multiplier*,
    the image is downscaled proportionally, and λ is increased by
    *lambda_multiplier* (stronger regularisation at coarser scales prevents
    convergence to the trivial no-blur solution).

    Pyramid construction terminates when the kernel would become smaller
    than 3 px or λ would exceed *max_lambda*.

    Parameters
    image : np.ndarray, shape (H, W)
        Blurred input image (float64, range [0, 1]).
    mk, nk : int
        Kernel height and width at the finest (original) scale.
    lambda_val : float
        TV regularisation weight at the finest scale.
    lambda_multiplier : float
        Factor by which λ grows per coarser level.
    max_lambda : float
        Upper bound for λ; no new level is added once exceeded.
    scale_multiplier : float
        Divisor applied to kernel dimensions per coarser level.

    Returns
    list of dict
        Pyramid levels ordered finest → coarsest.
        Each dict: ``{'image', 'M', 'N', 'MK', 'NK', 'lambda'}``.

    Reference
    C++ source: ``buildPyramid()`` in main.cpp, lines 295–400.
    Ravi et al. (2018), Sec. 4 / Perrone & Favaro (2016), Sec. 4.2.
    """
    H, W = image.shape
    smallest = 3

    # Phase 1. determine number of scales
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

    # 2. fill pyramid array
    pyramid: List[Dict] = [None] * n_scales  # type: ignore[list-item]
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

        # Prevent stagnation: if dimension didn't actually shrink, force -2
        if nk_s == prev['NK']:
            nk_s -= 2
        if mk_s == prev['MK']:
            mk_s -= 2

        mk_s = max(mk_s, smallest)
        nk_s = max(nk_s, smallest)

        # Down-scale image proportionally to kernel size reduction
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
