"""
utils.py

Utility functions for the LIP (Logarithmic Image Prior) blind deconvolution.

Ported from MATLAB code by D. Perrone, P. Favaro (2014).
Reference:
    D. Perrone, R. Diethelm, P. Favaro: "Blind Deconvolution via
    Lower-Bounded Logarithmic Image Priors", EMMCVPR 2015.

MATLAB → Python conversion notes:
    - convn(u,k,'valid')            → scipy.signal.fftconvolve(u, k, mode='valid')
      Both perform true convolution (kernel flip + slide).
    - convn(u,k,'full')             → scipy.signal.fftconvolve(u, k, mode='full')
    - padarray(f,[p,q],'replicate') → np.pad(f, ((p,p),(q,q)), mode='edge')
      MATLAB 'replicate' = NumPy 'edge' (nearest boundary value).
    - rot90(k,2)                    → np.rot90(k, 2)
    - imresize(img,[M,N],'bicubic') → scipy.ndimage.zoom(img, factors, order=3)
      MATLAB uses Keys' cubic (a=-0.5) with antialiasing on downsample;
      scipy uses B-spline order 3.  Minor numerical differences expected.
    - im2double(f)                  → f.astype(np.float64); if uint8: / 255.0
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import zoom


# ── Convolution wrappers ─────────────────────────────────────────────────────
#
# MATLAB's convn and Python's fftconvolve both implement TRUE convolution
# (i.e. the kernel is flipped).  The 'valid'/'full' size conventions are
# identical:
#   valid → (M - MK + 1,  N - NK + 1)   requires M >= MK, N >= NK
#   full  → (M + MK - 1,  N + NK - 1)

def convn_valid(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    2D convolution, 'valid' output size.
    Equivalent to MATLAB:  convn(u, k, 'valid')
    """
    return fftconvolve(u, k, mode='valid')


def convn_full(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    2D convolution, 'full' output size.
    Equivalent to MATLAB:  convn(u, k, 'full')
    """
    return fftconvolve(u, k, mode='full')


# ── Array / padding operations ───────────────────────────────────────────────

def pad_replicate(f: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """
    Pad with replicate (nearest-edge) boundary condition.
    Equivalent to MATLAB:  padarray(f, [pad_h, pad_w], 'replicate')

    Pads *pad_h* rows top & bottom, *pad_w* cols left & right.
    """
    return np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')


def shft(u: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Shifted finite difference with zero boundary conditions.

    Computes:
        result[i, j] = u[i + dy, j + dx] - u[i, j]   at valid positions,
        result[i, j] = 0                               at boundary positions.

    This is the local ``shft()`` function inside MATLAB's ``gradTVEM.m``.

    MATLAB (1-indexed):
        us = zeros(M, N, C);
        us(max(1-dy,1):min(M,M-dy), max(1-dx,1):min(N,N-dx), :) = ...
            u(max(dy+1,1):min(dy+M,M), max(dx+1,1):min(dx+N,N), :) - ...
            u(max(1-dy,1):min(M,M-dy), max(1-dx,1):min(N,N-dx), :);

    Converted to 0-indexed Python slicing (verified for all sign combinations
    of dx, dy including 0).

    Parameters
    ----------
    u  : (M, N) array
    dx : horizontal pixel shift (positive → right)
    dy : vertical  pixel shift (positive → down)

    Returns
    -------
    us : (M, N) array  — shifted difference, zeros at boundaries
    """
    M, N = u.shape
    us = np.zeros_like(u)

    # Destination slice
    r0 = max(-dy, 0)
    r1 = min(M, M - dy)
    c0 = max(-dx, 0)
    c1 = min(N, N - dx)

    # Source (shifted) slice
    sr0 = max(dy, 0)
    sr1 = min(dy + M, M)
    sc0 = max(dx, 0)
    sc1 = min(dx + N, N)

    us[r0:r1, c0:c1] = u[sr0:sr1, sc0:sc1] - u[r0:r1, c0:c1]
    return us


# ── Image preprocessing ──────────────────────────────────────────────────────

def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gamma correction:  Ic = I ** gamma.
    Equivalent to MATLAB:  gammaCorrection.m
    """
    return np.power(img, gamma)


def make_size_odd(f: np.ndarray) -> np.ndarray:
    """
    Trim image so both spatial dimensions are odd.

    Equivalent to MATLAB:
        if (mod(M,2)==0), f = f(1:end-1, :, :); end
        if (mod(N,2)==0), f = f(:, 1:end-1, :); end

    Removes the LAST row / column (same as MATLAB's 1:end-1).
    """
    if f.shape[0] % 2 == 0:
        f = f[:-1, :]
    if f.shape[1] % 2 == 0:
        f = f[:, :-1]
    return f


def imresize_matlab(img: np.ndarray, target_shape: tuple,
                    order: int = 3) -> np.ndarray:
    """
    Resize 2-D image to *target_shape* = (rows, cols).

    Approximates  MATLAB:  imresize(img, [M N], 'Method', 'bicubic')

    Differences from MATLAB:
        * MATLAB uses Keys' cubic kernel (a = -0.5) with antialiasing on
          downsample.  Here we use ``scipy.ndimage.zoom`` with B-spline
          order 3 (no antialiasing).  The minor numerical differences do
          not affect algorithm convergence.

    If ``scipy.ndimage.zoom`` produces an output shape that differs from
    *target_shape* by ±1 pixel (rounding), the result is trimmed or
    edge-padded to match exactly.
    """
    th, tw = int(target_shape[0]), int(target_shape[1])
    oh, ow = img.shape[:2]

    if oh == th and ow == tw:
        return img.copy()

    zoom_y = th / oh
    zoom_x = tw / ow
    result = zoom(img, (zoom_y, zoom_x), order=order)

    # Fix possible ±1 pixel mismatch from zoom rounding
    rh, rw = result.shape[:2]
    if rh > th:
        result = result[:th, :]
    elif rh < th:
        result = np.pad(result, ((0, th - rh), (0, 0)), mode='edge')

    rw = result.shape[1]
    if rw > tw:
        result = result[:, :tw]
    elif rw < tw:
        result = np.pad(result, ((0, 0), (0, tw - rw)), mode='edge')

    return result


# ── PSF / OTF utilities (needed for non-blind step) ──────────────────────────

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert point-spread function to optical transfer function.
    Equivalent to MATLAB:  psf2otf(psf, shape)

    1. Zero-pad PSF into an array of *shape*.
    2. Circularly shift so the PSF centre lands at index (0, 0).
    3. Return the 2-D FFT.
    """
    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf
    # Shift centre of the PSF to (0, 0)
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def get_gradient_operators(shape: tuple):
    """
    Forward-difference gradient operators in Fourier domain.

    Returns
    -------
    OTF_dx, OTF_dy, conj(OTF_dx), conj(OTF_dy)
    """
    kx = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]], dtype=np.float64)
    ky = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0]], dtype=np.float64)
    OTF_dx = psf2otf(kx, shape)
    OTF_dy = psf2otf(ky, shape)
    return OTF_dx, OTF_dy, np.conj(OTF_dx), np.conj(OTF_dy)


# ── Non-blind deconvolution filters ──────────────────────────────────────────
#
# These replace the missing ``deconvSps`` from Levin et al. that the MATLAB
# code refers to but does not include.

def wiener_filter(img: np.ndarray, kernel: np.ndarray,
                  noise_snr: float = 0.01) -> np.ndarray:
    """
    Wiener non-blind deconvolution.

    Model:  f = k ⊛ u + n
    Closed-form (Fourier):
        Û = conj(K) / (|K|² + snr) · F
    """
    H, W = img.shape
    otf = psf2otf(kernel, (H, W))
    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (np.abs(otf) ** 2 + noise_snr)) * F_img
    return np.real(np.fft.ifft2(F_res))


def tikhonov_filter(img: np.ndarray, kernel: np.ndarray,
                    alpha: float = 0.01) -> np.ndarray:
    """
    Non-blind Tikhonov-regularised deconvolution (1st-order gradient penalty).

    Solves  min_u  ||k ⊛ u − f||² + α·||∇u||²
    in closed form via FFT.
    """
    H, W = img.shape
    otf = psf2otf(kernel, (H, W))
    OTF_dx, OTF_dy, _, _ = get_gradient_operators((H, W))
    reg_term = np.abs(OTF_dx) ** 2 + np.abs(OTF_dy) ** 2
    denominator = np.abs(otf) ** 2 + alpha * reg_term
    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (denominator + 1e-12)) * F_img
    return np.real(np.fft.ifft2(F_res))


# ── Edge tapering (anti-ringing for FFT deconvolution) ───────────────────────

def edgetaper(img: np.ndarray, kernel: np.ndarray,
              n_tapers: int = 3) -> np.ndarray:
    """
    Taper image edges toward a blurred version to suppress FFT ringing.

    Mimics MATLAB's ``edgetaper(I, PSF)`` applied *n_tapers* times.

    Algorithm:
        1. Compute the autocorrelation of the PSF at its OWN size
           (not at image size) via ``fftconvolve(psf, psf[::-1,::-1], 'full')``.
           Result has shape (2*kh-1, 2*kw-1) with peak at the centre.
        2. Take the centre column/row profiles (length 2*k-1 each).
           The first half of each profile (length k-1) gives the taper
           ramp from small → ~1.
        3. Build a 1-D blending weight per dimension: 1 everywhere
           except for a (k-1)-wide ramp at each image border.
        4. The 2-D weight is the outer product of the two 1-D weights.
        5. Blend:  J = α·I + (1−α)·blur(I),  where α = 1 in the
           interior (original preserved) and α → 0 at the borders
           (replaced by the circularly-blurred version).

    Parameters
    ----------
    img      : (H, W) input image (typically padded before calling)
    kernel   : (kh, kw) PSF / blur kernel
    n_tapers : number of successive taper applications (default 3)

    Returns
    -------
    tapered : (H, W) edge-tapered image
    """
    H, W = img.shape
    kh, kw = kernel.shape

    # Autocorrelation of PSF at its own size → (2kh-1, 2kw-1)
    acf = fftconvolve(kernel, kernel[::-1, ::-1], mode='full')
    acf_max = acf.max()
    if acf_max > 0:
        acf /= acf_max

    # Centre column / row profiles of the small autocorrelation
    cy, cx = kh - 1, kw - 1          # centre indices
    z_col = acf[:, cx]               # length (2*kh - 1)
    z_row = acf[cy, :]               # length (2*kw - 1)

    # Build 1-D blending weights (1 in interior, ramp at borders)
    beta_y = np.ones(H, dtype=np.float64)
    beta_x = np.ones(W, dtype=np.float64)

    half_ky = kh - 1                  # number of taper pixels per side
    if half_ky > 0:
        taper = z_col[:half_ky]       # ramp from small → near-1
        n = min(len(taper), H // 2)   # guard against tiny images
        beta_y[:n] = taper[:n]
        beta_y[-n:] = taper[:n][::-1]

    half_kx = kw - 1
    if half_kx > 0:
        taper = z_row[:half_kx]
        n = min(len(taper), W // 2)
        beta_x[:n] = taper[:n]
        beta_x[-n:] = taper[:n][::-1]

    # 2-D weight (outer product): 1 at interior, small at borders
    alpha = beta_y[:, np.newaxis] * beta_x[np.newaxis, :]

    # Circular blur via FFT (same as MATLAB's edgetaper)
    otf = psf2otf(kernel, (H, W))

    result = img.copy()
    for _ in range(n_tapers):
        blurred = np.real(np.fft.ifft2(otf * np.fft.fft2(result)))
        result = alpha * result + (1.0 - alpha) * blurred

    return result


# ── Padding / cropping for non-blind step ────────────────────────────────────

def pad_image(img: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """Symmetric-pad image by the full kernel size on each side."""
    pad_h = kernel_shape[0]
    pad_w = kernel_shape[1]
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')


def crop_image(img: np.ndarray, original_shape: tuple,
               kernel_shape: tuple) -> np.ndarray:
    """Crop padded image back to original dimensions."""
    pad_h = kernel_shape[0]
    pad_w = kernel_shape[1]
    h, w = original_shape
    return img[pad_h:pad_h + h, pad_w:pad_w + w]
