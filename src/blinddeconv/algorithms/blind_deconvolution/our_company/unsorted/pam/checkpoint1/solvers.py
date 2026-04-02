"""
solvers.py

Core solver functions for PAM (Perrone-Favaro) TV blind deconvolution.

Ported from MATLAB code by Daniele Perrone and Paolo Favaro.
Reference:
    D. Perrone and P. Favaro: "Total Variation Blind Deconvolution:
    The Devil is in the Details", CVPR 2014.
    Technical Report: perrone2014tvTR.pdf

Contains:
    blind            — joint u,k gradient descent  (lib/blind.m, Algorithm 1)
    dec              — non-blind deconvolution      (lib/dec.m)
    coarse_to_fine   — multi-scale pyramid scheme   (lib/coarseToFine.m)
    deblur           — top-level entry point        (lib/deblur.m)

MATLAB → Python conversion notes:
    ─────────────────────────────────────────────────────────────────────
    conv2(u, k, 'valid'):
        Both MATLAB conv2 and scipy convolve2d perform true convolution
        (kernel is flipped). Output size matches: (M-Mk+1, N-Nk+1).

    conv2fft(a, b, mode):
        Custom FFT convolution. Ported in utils.py.

    rot90(k, 2):
        MATLAB rot90(k,2) = np.rot90(k, 2).  Both rotate 180°.

    numel(k) > 41^2:
        Switch threshold for FFT vs direct convolution.
        → k.size > 41**2

    padarray(f, [p1 p2], 'replicate'):
        → np.pad(f, ..., mode='edge')
        For 2D: np.pad(f, ((p1,p1),(p2,p2)), mode='edge')
        For 3D: np.pad(f, ((p1,p1),(p2,p2),(0,0)), mode='edge')

    max(u(:)):
        Global maximum of all elements.
        → u.max() or np.max(u)

    max(1e-31, max(abs(gradu(:)))):
        → max(1e-31, np.max(np.abs(gradu)))

    k.*(k>0):
        Element-wise multiply by boolean mask (clamp negatives to 0).
        → k * (k > 0)  or  np.maximum(k, 0)

    k/sum(k(:)):
        → k / k.sum()

    MATLAB size(f) for 2D returns [M, N, 1] when checking C.
        For a 2D Python array f.ndim==2, there is no third dim.
        We handle this by expanding to 3D internally when needed.
"""

import numpy as np

from .utils import (
    conv2fft,
    conv2_matlab,
    grad_tv_cc,
    gamma_correction,
    imresize,
    build_pyramid,
)


# Threshold for switching from direct conv2 to FFT convolution.
# MATLAB: numel(k) > 41^2
_FFT_THRESHOLD = 41 ** 2


def _padarray_replicate(f: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """
    Replicate-boundary padding matching MATLAB padarray(f,[p1 p2],'replicate').

    Works for 2D (M,N) and 3D (M,N,C) arrays.
    """
    if f.ndim == 2:
        return np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        return np.pad(f, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')


def _conv2(a: np.ndarray, b: np.ndarray, mode: str = 'full') -> np.ndarray:
    """
    2D convolution choosing FFT or direct depending on kernel size.

    Matches MATLAB logic:
        if numel(k) > 41^2
            conv2fft(...)
        else
            conv2(...)
    """
    if b.size > _FFT_THRESHOLD:
        return conv2fft(a, b, mode)
    else:
        return conv2_matlab(a, b, mode)


# ═════════════════════════════════════════════════════════════════════════════
# blind  (from lib/blind.m — Algorithm 1)
# ═════════════════════════════════════════════════════════════════════════════

def blind(f: np.ndarray, MK: int, NK: int,
          lam: float = 3e-4,
          u: np.ndarray = None,
          k: np.ndarray = None,
          iters: int = 1000,
          visualize: bool = False) -> tuple:
    """
    Joint blind estimation of sharp image u and kernel k via gradient
    descent.  Algorithm 1 from Perrone & Favaro (CVPR 2014).

    Key insight: TV is MAXIMISED (subtracted from data gradient), not
    minimised.  This prevents the trivial delta-kernel solution.

    Parameters
    ----------
    f : (M, N) or (M, N, C) blurry image (float64, [0,1])
    MK, NK : kernel height, width
    lam : TV regularisation weight (lambda)
    u : initial sharp image estimate.
        Default: f padded with replicate boundary.
    k : initial PSF estimate.
        Default: ones(MK,NK)/(MK*NK).
    iters : number of gradient descent iterations.
    visualize : if True, print progress info.

    Returns
    -------
    u : (MU, NU) or (MU, NU, C) estimated sharp image
    k : (MK, NK) estimated PSF
    """
    # ── Ensure 3D for uniform processing ─────────────────────────────
    squeeze_out = (f.ndim == 2)
    if f.ndim == 2:
        f = f[:, :, np.newaxis]

    M, N, C = f.shape

    # ── Defaults ─────────────────────────────────────────────────────
    if u is None:
        u = _padarray_replicate(f, MK // 2, NK // 2)
    elif u.ndim == 2 and C > 1:
        u = u[:, :, np.newaxis]

    if k is None:
        k = np.ones((MK, NK), dtype=np.float64) / (MK * NK)

    # Ensure 3D u
    if u.ndim == 2:
        u = u[:, :, np.newaxis]

    # ── Size of sharp image ──────────────────────────────────────────
    MU = M + MK - 1
    NU = N + NK - 1

    k_rot = np.rot90(k, 2)

    for it in range(1, iters + 1):
        # ── Update sharp image u ─────────────────────────────────────
        gradudata = np.zeros((MU, NU, C), dtype=np.float64)
        for c in range(C):
            # MATLAB: conv2(conv2(u(:,:,c), k, 'valid') - f(:,:,c), rot90(k,2), 'full')
            residual = _conv2(u[:, :, c], k, 'valid') - f[:, :, c]
            gradudata[:, :, c] = _conv2(residual, k_rot, 'full')

        # MATLAB: gradu = gradudata - lambda * gradTVcc(u)
        gradu = gradudata - lam * grad_tv_cc(u)

        # Adaptive step size (normalized steepest descent)
        # MATLAB: sf = 5e-3 * max(u(:)) / max(1e-31, max(abs(gradu(:))))
        sf = 5e-3 * np.max(u) / max(1e-31, np.max(np.abs(gradu)))
        u = u - sf * gradu

        # ── Update kernel k ──────────────────────────────────────────
        gradk = np.zeros((MK, NK), dtype=np.float64)
        for c in range(C):
            # Inner convolution: different threshold logic in MATLAB
            # For the inner conv, MATLAB always uses conv2fft for the outer
            # but conv2 or conv2fft for the inner based on kernel size.
            #
            # MATLAB:
            #   if numel(k) > 41^2
            #       gradk += conv2fft(rot90(u(:,:,c),2),
            #                         conv2fft(u(:,:,c), k, 'valid') - f(:,:,c), 'valid')
            #   else
            #       gradk += conv2fft(rot90(u(:,:,c),2),
            #                         conv2(u(:,:,c), k, 'valid') - f(:,:,c), 'valid')
            #
            # Note: outer conv is ALWAYS conv2fft in MATLAB code regardless
            # of the threshold. Only the inner differs.
            if k.size > _FFT_THRESHOLD:
                inner = conv2fft(u[:, :, c], k, 'valid') - f[:, :, c]
            else:
                inner = conv2_matlab(u[:, :, c], k, 'valid') - f[:, :, c]
            gradk += conv2fft(np.rot90(u[:, :, c], 2), inner, 'valid')

        # Adaptive step size for kernel
        # MATLAB: sh = 1e-3 * max(k(:)) / max(1e-31, max(abs(gradk(:))))
        sh = 1e-3 * np.max(k) / max(1e-31, np.max(np.abs(gradk)))
        k = k - sh * gradk

        # ── Kernel projection ────────────────────────────────────────
        # MATLAB: k = k.*(k>0); k = k/sum(k(:));
        k = k * (k > 0)
        k_sum = k.sum()
        if k_sum > 0:
            k = k / k_sum

        # Update rotated kernel for next iteration
        k_rot = np.rot90(k, 2)

    if squeeze_out:
        u = u[:, :, 0]

    return u, k


# ═════════════════════════════════════════════════════════════════════════════
# dec  (from lib/dec.m — non-blind deconvolution)
# ═════════════════════════════════════════════════════════════════════════════

def dec(f: np.ndarray, k: np.ndarray,
        lam: float = 3e-4,
        u: np.ndarray = None,
        iters: int = 1000,
        visualize: bool = False) -> np.ndarray:
    """
    Non-blind deconvolution via TV-regularised gradient descent.

    Same gradient descent as blind() but WITHOUT kernel update.
    Used to refine the image estimate after each pyramid scale.

    Matches MATLAB dec.m by Perrone & Favaro.

    Parameters
    ----------
    f : (M, N) or (M, N, C) blurry image
    k : (MK, NK) known PSF
    lam : TV regularisation weight
    u : initial sharp image estimate.
        Default: f padded with replicate boundary.
    iters : number of gradient descent iterations.
    visualize : unused (kept for interface compatibility).

    Returns
    -------
    u : (MU, NU) or (MU, NU, C) deconvolved image
    """
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
            # MATLAB: conv2(conv2(u(:,:,c), k, 'valid') - f(:,:,c), rot90(k,2), 'full')
            if k.size > _FFT_THRESHOLD:
                residual = conv2fft(u[:, :, c], k, 'valid') - f[:, :, c]
                gradudata[:, :, c] = conv2fft(residual, k_rot, 'full')
            else:
                residual = conv2_matlab(u[:, :, c], k, 'valid') - f[:, :, c]
                gradudata[:, :, c] = conv2_matlab(residual, k_rot, 'full')

        gradu = gradudata - lam * grad_tv_cc(u)

        # MATLAB: sf = 5e-3*max(u(:))/max(1e-31,max(abs(gradu(:))))
        sf = 5e-3 * np.max(u) / max(1e-31, np.max(np.abs(gradu)))
        u = u - sf * gradu

    if squeeze_out:
        u = u[:, :, 0]

    return u


# ═════════════════════════════════════════════════════════════════════════════
# coarse_to_fine  (from lib/coarseToFine.m)
# ═════════════════════════════════════════════════════════════════════════════

def coarse_to_fine(f: np.ndarray, MK: int, NK: int,
                   blind_iters: int = 1000,
                   visualize: bool = False,
                   final_lambda: float = 3e-4,
                   lambda_multiplier: float = 1.9,
                   max_lambda: float = 0.11,
                   kernel_size_multiplier: float = 1.1,
                   interp_method: str = 'bicubic') -> tuple:
    """
    Multi-scale coarse-to-fine blind deconvolution.

    Builds a pyramid of scales, then iterates from coarsest to finest.
    At each scale: (1) blind estimation of u,k; (2) non-blind refinement of u.

    Matches MATLAB coarseToFine.m by Daniele Perrone.

    Parameters
    ----------
    f : (M, N) or (M, N, C) blurry image (float64)
    MK, NK : kernel height, width at finest scale
    blind_iters : iterations per blind() call
    visualize : print progress
    final_lambda : λ at finest scale
    lambda_multiplier : factor to increase λ at coarser scales
    max_lambda : upper bound for λ
    kernel_size_multiplier : kernel shrink factor per coarser level
    interp_method : interpolation method

    Returns
    -------
    u : estimated sharp image
    k : estimated PSF
    """
    # ── Initial estimates ────────────────────────────────────────────
    u = _padarray_replicate(f, MK // 2, NK // 2)
    k = np.ones((MK, NK), dtype=np.float64) / (MK * NK)

    # ── Build pyramid ────────────────────────────────────────────────
    fp, Mp, Np, MKp, NKp, lambdas, num_scales = build_pyramid(
        f, MK, NK, final_lambda, lambda_multiplier,
        interp_method, kernel_size_multiplier, max_lambda,
    )

    # ── Multiscale processing: coarsest → finest ─────────────────────
    # MATLAB: for scale = scales:-1:1
    # In buildPyramid.m, index 1 = finest, index `scales` = coarsest.
    # Our build_pyramid returns lists where index 0 = finest, index num_scales-1 = coarsest.
    for scale_idx in range(num_scales - 1, -1, -1):
        Ms = Mp[scale_idx]
        Ns = Np[scale_idx]
        MKs = MKp[scale_idx]
        NKs = NKp[scale_idx]

        # Resize current estimates to this scale
        # MATLAB: imresize(u, [Ms+MKs-1, Ns+NKs-1], ...)
        u_target_h = Ms + MKs - 1
        u_target_w = Ns + NKs - 1
        u = imresize(u, (u_target_h, u_target_w), method=interp_method)

        # MATLAB: imresize(k, [MKs NKs], ...)
        k = imresize(k, (MKs, NKs), method=interp_method)
        # Kernel projection
        k = k * (k > 0)
        k_sum = k.sum()
        if k_sum > 0:
            k = k / k_sum

        fs = fp[scale_idx]
        lam = lambdas[scale_idx]

        if visualize:
            print(f"scale: {scale_idx + 1}  lambda: {lam:.6f}  "
                  f"MKs: {MKs}  NKs: {NKs}  iters: {blind_iters}")

        # (1) Blind estimation
        u, k = blind(fs, MKs, NKs,
                      lam=lam, u=u, k=k,
                      iters=blind_iters, visualize=visualize)

        # (2) Non-blind refinement
        u = dec(fs, k, lam=lam, u=u,
                iters=blind_iters, visualize=visualize)

    return u, k


# ═════════════════════════════════════════════════════════════════════════════
# deblur  (from lib/deblur.m — top-level entry point)
# ═════════════════════════════════════════════════════════════════════════════

def deblur(f: np.ndarray, MK: int, NK: int,
           lam: float = 3e-4,
           iters: int = 1000,
           gamma_correct: bool = False,
           gamma: float = 1.0,
           visualize: bool = False) -> tuple:
    """
    Top-level blind deconvolution.

    Pipeline:
        1. Normalise to float64 [0, 1].
        2. Ensure odd image dimensions (crop if even).
        3. Optional gamma correction.
        4. Coarse-to-fine blind deconvolution.

    Matches MATLAB deblur.m by Daniele Perrone.

    Parameters
    ----------
    f : input blurry image (uint8/uint16/float)
    MK, NK : PSF kernel height, width
    lam : TV regularisation weight.
          Typical: 3e-4 .. 6e-4.  Noisy images: 1e-3 .. 3e-3.
    iters : iterations per blind/non-blind call at each scale.
    gamma_correct : whether to apply gamma correction.
    gamma : gamma exponent (used only if gamma_correct is True).
    visualize : print diagnostic information.

    Returns
    -------
    u : (M', N') or (M', N', C) estimated sharp image (float64, [may exceed 0,1])
    k : (MK, NK) estimated PSF
    """
    # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
    # MATLAB: f = im2double(f)
    f = f.astype(np.float64)
    if f.max() > 1.0:
        f = f / 255.0

    # ── 2. Ensure odd dimensions ─────────────────────────────────────
    # MATLAB: if mod(M,2)==0, f=f(1:end-1,:,:); end
    M, N = f.shape[:2]
    if M % 2 == 0:
        f = f[:-1, ...]
    if N % 2 == 0:
        f = f[:, :-1, ...]

    # ── 3. Gamma correction ──────────────────────────────────────────
    if gamma_correct:
        f = gamma_correction(f, gamma)

    # ── 4. Coarse-to-fine blind deconvolution ────────────────────────
    # MATLAB coarse-to-fine parameters:
    #   ctf_params.lambdaMultiplier = 1.9;
    #   ctf_params.maxLambda = 1.1e-1;
    #   ctf_params.finalLambda = lambda;
    #   ctf_params.kernelSizeMultiplier = 1.1;
    #   ctf_params.interpolationMethod = 'bicubic';
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
