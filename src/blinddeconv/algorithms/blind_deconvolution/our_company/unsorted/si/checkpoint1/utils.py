"""
utils.py

Utility functions for the Shift-Invariant Blind Deblurring algorithm.

Ported from MATLAB code by Hua Cheng (2011).
Reference:
    Hua Cheng, "Shift-Invariant Deblurring", part of Super-Resolution
    Project (SR.pdf), 2011.

Relies on works by:
    - Perona & Malik (ref 17): Nonlinear anisotropic diffusion
    - Osher & Rudin (ref 18): Shock filters
    - Xu & Jia (ref 16): Edge selection for kernel estimation

MATLAB -> Python conversion notes:
    ─────────────────────────────────────────────────────────────────────
    fspecial('gaussian', ksize, sigma):
        Produces a ksize x ksize Gaussian kernel normalised to sum=1.
        The grid is centred at the middle pixel.
        -> Manual construction with the same grid and normalisation.

    padarray(I, [p p], 'replicate', 'both'):
        Replicates the nearest edge value.
        -> np.pad(I, ((p,p),(p,p)), mode='edge')

    edgetaper(Im, kernel):
        Tapers image edges toward a blurred version using the kernel
        auto-correlation to define blending weights.
        -> Manual implementation matching MATLAB behaviour.

    imresize(I, scale) / imresize(I, scale, 'bicubic'):
        MATLAB default is bicubic with anti-aliasing.
        -> cv2.resize with INTER_AREA (down) / INTER_CUBIC (up).
           Falls back to scipy.ndimage.zoom when cv2 is unavailable.

    MATLAB I(:,[1 1:nx-1]) — Neumann BC shift right:
        -> np.hstack([I[:, 0:1], I[:, :-1]])

    MATLAB I(:,[2:nx nx]) — Neumann BC shift left:
        -> np.hstack([I[:, 1:], I[:, -1:]])

    MATLAB atan(x) = np.arctan(x).  Both return pi/2 for +Inf, NaN
    for NaN.

    MATLAB find(k <= 0) -> boolean indexing k[k <= 0] = 0.

    MATLAB 1-based indexing:
        sam(count)  ->  sam[count - 1]   (0-based)

    conv2(A, B, 'valid') with symmetric B:
        -> scipy.signal.convolve2d(A, B, mode='valid')
           For symmetric B (e.g. ones), convolution = correlation.
"""

import numpy as np
from scipy.signal import convolve2d, fftconvolve


# ═════════════════════════════════════════════════════════════════════════════
# PSF <-> OTF conversions
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert PSF to OTF.  Equivalent to MATLAB psf2otf(psf, shape).

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at (0, 0).
    3. Return fft2.

    MATLAB circshift amounts: -floor(size(psf)/2) per dimension.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    # Circular shift: move PSF centre to (0, 0)
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: tuple) -> np.ndarray:
    """
    Convert OTF back to PSF.  Equivalent to MATLAB otf2psf(otf, psf_size).

    1. ifft2 -> real part.
    2. Circular shift by +floor(psf_size/2) per dimension.
    3. Crop to psf_size.
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


# ═════════════════════════════════════════════════════════════════════════════
# Gaussian kernel  —  fspecial('gaussian', ksize, sigma)
# ═════════════════════════════════════════════════════════════════════════════

def fspecial_gaussian(ksize: int, sigma: float) -> np.ndarray:
    """
    Create a Gaussian kernel identical to MATLAB
    fspecial('gaussian', ksize, sigma).

    Returns
    -------
    kernel : (ksize, ksize) float64 array, sum = 1.
    """
    half = ksize // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


# ═════════════════════════════════════════════════════════════════════════════
# edgetaper  —  MATLAB edgetaper(img, psf)
# ═════════════════════════════════════════════════════════════════════════════

def edgetaper(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Taper image edges toward a blurred version to suppress FFT boundary
    ringing.  Mimics one call to MATLAB ``edgetaper(img, kernel)``.

    Algorithm:
        1. Auto-correlate the kernel -> acf (normalised to [0, 1]).
        2. Extract centre row/column of acf as 1-D taper profiles.
        3. Build a 2-D blending map alpha in [0, 1] (1 = keep original).
        4. Blur the image with the kernel via FFT.
        5. Return  alpha * img  +  (1 - alpha) * blurred.
    """
    H, W = img.shape[:2]
    kh, kw = kernel.shape

    # Auto-correlation of kernel
    acf = fftconvolve(kernel, kernel[::-1, ::-1], mode='full')
    acf_max = acf.max()
    if acf_max > 0:
        acf /= acf_max

    # Centre indices of auto-correlation  (size = 2*k - 1)
    cy, cx = kh - 1, kw - 1

    # 1-D taper profiles
    z_col = acf[:, cx]   # vertical
    z_row = acf[cy, :]   # horizontal

    beta_y = np.ones(H, dtype=np.float64)
    beta_x = np.ones(W, dtype=np.float64)

    half_ky = kh - 1
    if half_ky > 0:
        taper = z_col[:half_ky]
        n = min(len(taper), H // 2)
        beta_y[:n] = taper[:n]
        beta_y[-n:] = taper[:n][::-1]

    half_kx = kw - 1
    if half_kx > 0:
        taper = z_row[:half_kx]
        n = min(len(taper), W // 2)
        beta_x[:n] = taper[:n]
        beta_x[-n:] = taper[:n][::-1]

    # 2-D blending weight
    alpha_map = beta_y[:, np.newaxis] * beta_x[np.newaxis, :]

    # Blur image via FFT
    otf = psf2otf(kernel, (H, W))
    blurred = np.real(np.fft.ifft2(otf * np.fft.fft2(img)))

    return alpha_map * img + (1.0 - alpha_map) * blurred


# ═════════════════════════════════════════════════════════════════════════════
# imresize  —  MATLAB imresize(I, scale [, method])
# ═════════════════════════════════════════════════════════════════════════════

def imresize(img: np.ndarray, scale: float,
             method: str = 'bicubic') -> np.ndarray:
    """
    Resize a 2-D image, matching MATLAB ``imresize`` as closely as possible.

    Parameters
    ----------
    img    : 2-D float64 array.
    scale  : scaling factor (< 1 = downsample, > 1 = upsample).
    method : 'bicubic' (default) or 'bilinear'.

    Notes
    -----
    - Prefers cv2 (INTER_AREA for downscaling, INTER_CUBIC for up).
    - Falls back to scipy.ndimage.zoom when cv2 is unavailable.
    - MATLAB uses anti-aliased bicubic by default; cv2.INTER_AREA is the
      closest for downscaling.
    """
    new_h = max(1, round(img.shape[0] * scale))
    new_w = max(1, round(img.shape[1] * scale))
    try:
        import cv2
        if scale < 1.0:
            return cv2.resize(img, (new_w, new_h),
                              interpolation=cv2.INTER_AREA)
        else:
            interp = (cv2.INTER_CUBIC if method == 'bicubic'
                      else cv2.INTER_LINEAR)
            return cv2.resize(img, (new_w, new_h), interpolation=interp)
    except ImportError:
        from scipy.ndimage import zoom
        order = 3 if method == 'bicubic' else 1
        return zoom(img, scale, order=order)


# ═════════════════════════════════════════════════════════════════════════════
# Circular first-order gradient  (as used in deblur.m)
# ═════════════════════════════════════════════════════════════════════════════

def gradient_circular(I: np.ndarray):
    """
    Circular first-order finite differences exactly as in deblur.m::

        MATLAB:
            Im_x = [Im(1:xim-1,:)-Im(2:xim,:); Im(xim,:)-Im(1,:)];
            Im_y = [Im(:,1:yim-1)-Im(:,2:yim), Im(:,yim)-Im(:,1)];

    Returns
    -------
    I_x, I_y : arrays of the same shape as *I*.
    """
    # I_x: row i minus row i+1, with circular wrap on last row
    I_x = np.vstack([I[:-1, :] - I[1:, :],
                     I[-1:, :] - I[0:1, :]])
    # I_y: col j minus col j+1, with circular wrap on last column
    I_y = np.hstack([I[:, :-1] - I[:, 1:],
                     I[:, -1:] - I[:, 0:1]])
    return I_x, I_y


# ═════════════════════════════════════════════════════════════════════════════
# Perona–Malik nonlinear diffusion  (perona_malik.m)
# ═════════════════════════════════════════════════════════════════════════════

def perona_malik(I: np.ndarray, n_iter: int) -> np.ndarray:
    """
    Perona–Malik nonlinear anisotropic diffusion.
    Exact port of MATLAB ``perona_malik.m``.

    Reference:
        P. Perona, J. Malik, "Scale-space and edge detection using
        anisotropic diffusion", IEEE TPAMI 12(7), 1990.

    Parameters
    ----------
    I      : (H, W) input image (float64).
    n_iter : number of diffusion iterations.

    Returns
    -------
    I_pm : (H, W) diffused image.

    Notes
    -----
    MATLAB code processes only the interior pixels (rows 2..end-1,
    cols 2..end-1 in 1-based).  Border pixels are unchanged.  The
    diffusion coefficient threshold *k* is set to 5 % of the maximum
    directional difference (not absolute value).
    ``lambda = 1/4`` ensures stability for the 4-connected stencil.
    """
    lam = 0.25   # lambda = 1/4
    I_pm = I.copy()

    for _ in range(n_iter):
        inner = I_pm[1:-1, 1:-1]

        # Directional differences (matching MATLAB variable names)
        # MATLAB: I_n = I_pm(idxx-1, idxy) - I_pm(idxx, idxy)
        I_n = I_pm[:-2, 1:-1] - inner   # north: (i-1,j) - (i,j)
        I_s = I_pm[2:,  1:-1] - inner   # south: (i+1,j) - (i,j)
        I_e = I_pm[1:-1, :-2] - inner   # east:  (i,j-1) - (i,j)
        I_w = I_pm[1:-1, 2:]  - inner   # west:  (i,j+1) - (i,j)

        # Adaptive threshold  (MATLAB: k = 0.05 * max of all maxima)
        k_val = 0.05 * max(I_n.max(), I_s.max(), I_e.max(), I_w.max())

        # Diffusion coefficients  c(x) = 1 / (1 + |x| / k)
        C_n = 1.0 / (1.0 + np.abs(I_n) / k_val)
        C_s = 1.0 / (1.0 + np.abs(I_s) / k_val)
        C_e = 1.0 / (1.0 + np.abs(I_e) / k_val)
        C_w = 1.0 / (1.0 + np.abs(I_w) / k_val)

        I_pm[1:-1, 1:-1] += lam * (C_n * I_n + C_s * I_s +
                                    C_e * I_e + C_w * I_w)

    return I_pm


# ═════════════════════════════════════════════════════════════════════════════
# Shock filter — Osher & Rudin  (shock_filter.m)
# ═════════════════════════════════════════════════════════════════════════════

def shock_filter(I0: np.ndarray, n_iter: int,
                 dt: float = 0.1, h: float = 1.0) -> np.ndarray:
    """
    Osher–Rudin shock filter.
    Exact port of MATLAB ``shock_filter.m`` (method = 'org').

    Reference:
        S.J. Osher and L.I. Rudin, "Feature-oriented image enhancement
        using shock filters", SIAM J. Numer. Anal. 27, 1990.

    PDE:
        I_t = -sign(I_nn) · |∇I| / h

    where I_nn is the second derivative in the gradient direction,
    and |∇I| uses the minmod limiter.

    Parameters
    ----------
    I0     : (ny, nx) input image (float64).
    n_iter : number of evolution steps.
    dt     : time step (default 0.1).
    h      : spatial grid step (default 1.0).

    Returns
    -------
    I : (ny, nx) shock-filtered image.

    Notes
    -----
    Neumann (replicate) boundary conditions are used throughout,
    matching the MATLAB code exactly:
        I(:,[1  1:nx-1])  ->  np.hstack([I[:, 0:1], I[:, :-1]])
        I(:,[2:nx  nx])   ->  np.hstack([I[:, 1:],  I[:, -1:]])
        I([1  1:ny-1],:)  ->  np.vstack([I[0:1, :], I[:-1, :]])
        I([2:ny  ny],:)   ->  np.vstack([I[1:, :],  I[-1:, :]])
    """
    I = I0.copy()

    for _ in range(n_iter):
        # ── Neumann-BC shifted copies ───────────────────────────────────
        I_left  = np.hstack([I[:, 0:1],  I[:, :-1]])   # shift right
        I_right = np.hstack([I[:, 1:],   I[:, -1:]])   # shift left
        I_up    = np.vstack([I[0:1, :],  I[:-1, :]])   # shift down
        I_down  = np.vstack([I[1:, :],   I[-1:, :]])   # shift up

        # ── Forward / backward differences ──────────────────────────────
        I_mx = I - I_left       # backward x
        I_px = I_right - I      # forward  x
        I_my = I - I_up         # backward y
        I_py = I_down - I       # forward  y

        # ── Central differences ─────────────────────────────────────────
        I_x = (I_mx + I_px) / 2.0
        I_y = (I_my + I_py) / 2.0

        # ── Minmod operator for |∇I| ───────────────────────────────────
        Dx = np.minimum(np.abs(I_mx), np.abs(I_px))
        Dx[I_mx * I_px < 0] = 0.0
        Dy = np.minimum(np.abs(I_my), np.abs(I_py))
        Dy[I_my * I_py < 0] = 0.0

        # ── Second derivatives ──────────────────────────────────────────
        I_xx = I_right + I_left - 2.0 * I
        I_yy = I_down  + I_up  - 2.0 * I

        # I_xy = (I_x([2:ny ny],:) - I_x([1 1:ny-1],:)) / 2
        I_x_down = np.vstack([I_x[1:, :],  I_x[-1:, :]])
        I_x_up   = np.vstack([I_x[0:1, :], I_x[:-1, :]])
        I_xy = (I_x_down - I_x_up) / 2.0

        # ── Absolute gradient (minmod-limited) ──────────────────────────
        a_grad_I = np.sqrt(Dx ** 2 + Dy ** 2)

        # ── Second derivative in gradient direction ─────────────────────
        dl = 1e-8   # MATLAB: dl = 0.00000001
        denom = np.abs(I_x) ** 2 + np.abs(I_y) ** 2 + dl
        I_nn = (I_xx * np.abs(I_x) ** 2 +
                2.0 * I_xy * I_x * I_y +
                I_yy * np.abs(I_y) ** 2) / denom

        # Fix zero-gradient pixels  (MATLAB: I_nn(ind) = I_xx(ind))
        a2_grad_I = np.abs(I_x) + np.abs(I_y)
        zero_mask = (a2_grad_I == 0)
        I_nn[zero_mask] = I_xx[zero_mask]

        # ── Evolution step ──────────────────────────────────────────────
        I_t = -np.sign(I_nn) * a_grad_I / h
        I = I + dt * I_t

    return I


# ═════════════════════════════════════════════════════════════════════════════
# M_compute  —  mask of significant structures  (M_compute.m)
# ═════════════════════════════════════════════════════════════════════════════

def m_compute(Im_x: np.ndarray, Im_y: np.ndarray,
              ks_val: int, is_first_scale: bool,
              tau_r: float):
    """
    Compute binary mask *M* selecting pixels where the blurred image
    contains significant directional structure (r-map thresholding).

    Exact port of MATLAB ``M_compute.m``.

    Reference:
        Xu & Jia edge-selection strategy (ref 16 in SR.pdf).

    For each valid pixel (i, j), the r-map is::

        r_x = Σ I_x / (Σ |I_x| + 0.5)
        r_y = Σ I_y / (Σ |I_y| + 0.5)            (sums over ksize × ksize window)
        rmap = sqrt(r_x² + r_y²)

    On the **first scale**, the threshold τ_r is chosen adaptively by
    binning r-map values into four angular sectors (by θ = atan(r_x/r_y)),
    sorting each sector's values descending, and picking the *count*-th
    value, where count = ceil(0.5 · sqrt(ksize² · Ix · Iy)).  τ_r is the
    minimum across sectors, ensuring all edge orientations are represented.

    On **subsequent scales**, the threshold decays: τ_r /= 1.1.

    The window sums are computed via 2-D convolution with a ones kernel
    (mathematically equivalent to the sliding-column optimisation in the
    original MATLAB code).

    Parameters
    ----------
    Im_x, Im_y     : (Ix, Iy) gradient images of the blurred input.
    ks_val          : half-kernel-size at the current pyramid level.
    is_first_scale  : True for the coarsest level (idx == 0).
    tau_r           : threshold carried from the previous level
                      (ignored when *is_first_scale* is True).

    Returns
    -------
    M     : (Ix, Iy) binary mask (float64, values 0 or 1).
    tau_r : updated threshold.
    """
    Ix, Iy = Im_x.shape
    ksize = 2 * ks_val + 1
    ratio = 1.1

    # ── Window sums via 2-D convolution (mode='valid') ──────────────────
    # Equivalent to MATLAB sliding-column accumulation.
    # convolve2d flips the kernel, but ones is symmetric → no effect.
    box = np.ones((ksize, ksize), dtype=np.float64)

    sum_x      = convolve2d(Im_x,         box, mode='valid')
    sumabs_x   = convolve2d(np.abs(Im_x), box, mode='valid')
    sum_y      = convolve2d(Im_y,         box, mode='valid')
    sumabs_y   = convolve2d(np.abs(Im_y), box, mode='valid')

    # r-map components for valid region
    rmap_x = sum_x / (sumabs_x + 0.5)
    rmap_y = sum_y / (sumabs_y + 0.5)
    rmap_valid = np.sqrt(rmap_x ** 2 + rmap_y ** 2)

    # Full-size r-map (zero outside valid region)
    rmap = np.zeros((Ix, Iy), dtype=np.float64)
    rmap[ks_val:Ix - ks_val, ks_val:Iy - ks_val] = rmap_valid

    if is_first_scale:
        # ── Angle-sector adaptive threshold ─────────────────────────────
        # theta = atan(rmap_x / rmap_y),  matches MATLAB atan(a/b)
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.arctan(rmap_x / rmap_y)

        flat_rmap  = rmap_valid.ravel()
        flat_theta = theta.ravel()

        # Four sectors matching MATLAB if/elseif/else chain:
        #   angle1:  [-pi/2, -pi/4]
        #   angle2:  (-pi/4,  0]
        #   angle3:  (0,      pi/4]
        #   angle4:  else  (covers (pi/4, pi/2] and NaN)
        mask1 = (-np.pi / 2 <= flat_theta) & (flat_theta <= -np.pi / 4)
        mask2 = (-np.pi / 4 <  flat_theta) & (flat_theta <= 0)
        mask3 = (0           <  flat_theta) & (flat_theta <= np.pi / 4)
        mask4 = ~(mask1 | mask2 | mask3)

        angles = [flat_rmap[mask1], flat_rmap[mask2],
                  flat_rmap[mask3], flat_rmap[mask4]]

        # MATLAB: count = ceil(0.5 * sqrt((2*ks(idx)+1)^2 * Ix * Iy))
        count = int(np.ceil(0.5 * np.sqrt(ksize ** 2 * Ix * Iy)))

        tau_values = []
        for sam in angles:
            sorted_sam = np.sort(sam)[::-1]       # descending
            # MATLAB 1-based: sam(count)  ->  Python 0-based: [count-1]
            tau_values.append(sorted_sam[count - 1])

        tau_r = float(min(tau_values))
    else:
        tau_r = tau_r / ratio

    # ── Binary mask ─────────────────────────────────────────────────────
    M = np.zeros((Ix, Iy), dtype=np.float64)
    M[ks_val:Ix - ks_val, ks_val:Iy - ks_val] = \
        (rmap_valid >= tau_r).astype(np.float64)

    return M, tau_r


# ═════════════════════════════════════════════════════════════════════════════
# H_compute  —  mask of significant shock-filter edges  (H_compute.m)
# ═════════════════════════════════════════════════════════════════════════════

def h_compute(Ish_x: np.ndarray, Ish_y: np.ndarray,
              M: np.ndarray, ks_val: int,
              is_first_scale: bool, tau_s: float):
    """
    Compute binary mask *H* selecting pixels with significant edges in
    the shock-filtered image.

    Exact port of MATLAB ``H_compute.m``.

    H(i,j) = 1  iff  M(i,j) · ||∇I_sh(i,j)||₂  ≥  τ_s

    On the **first scale**, τ_s is chosen adaptively by binning gradient
    magnitudes ||∇I_sh|| into four angular sectors (by
    θ = atan(I_sh_x / I_sh_y)), selecting the top-*count* value in each
    sector (count = ceil(20 · ksize)), and taking the minimum.

    On **subsequent scales**, τ_s /= 1.1.

    Parameters
    ----------
    Ish_x, Ish_y    : (Ix, Iy) gradient images of shock-filtered estimate.
    M                : (Ix, Iy) binary mask from ``m_compute``.
    ks_val           : half-kernel-size at the current pyramid level.
    is_first_scale   : True for the coarsest level.
    tau_s            : threshold carried from the previous level
                       (ignored when *is_first_scale* is True).

    Returns
    -------
    H     : (Ix, Iy) binary mask (float64, values 0 or 1).
    tau_s : updated threshold.
    """
    Ix, Iy = Ish_x.shape
    ratio = 1.1

    # Gradient magnitude — valid region only
    # MATLAB: Delta_Ish = zeros(Ix, Iy); only fill ks+1:Ix-ks, ks+1:Iy-ks
    sl = (slice(ks_val, Ix - ks_val), slice(ks_val, Iy - ks_val))
    Delta_Ish = np.zeros((Ix, Iy), dtype=np.float64)
    Delta_Ish[sl] = np.sqrt(Ish_x[sl] ** 2 + Ish_y[sl] ** 2)

    if is_first_scale:
        ksize = 2 * ks_val + 1

        valid_Ish_x = Ish_x[sl]
        valid_Ish_y = Ish_y[sl]
        valid_delta = Delta_Ish[sl]

        # theta = atan(Ish_x / Ish_y),  matches MATLAB
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.arctan(valid_Ish_x / valid_Ish_y)

        flat_delta = valid_delta.ravel()
        flat_theta = theta.ravel()

        mask1 = (-np.pi / 2 <= flat_theta) & (flat_theta <= -np.pi / 4)
        mask2 = (-np.pi / 4 <  flat_theta) & (flat_theta <= 0)
        mask3 = (0           <  flat_theta) & (flat_theta <= np.pi / 4)
        mask4 = ~(mask1 | mask2 | mask3)

        angles = [flat_delta[mask1], flat_delta[mask2],
                  flat_delta[mask3], flat_delta[mask4]]

        # MATLAB: count = ceil(20 * (2*ks(idx)+1))
        count = int(np.ceil(20 * ksize))

        tau_values = []
        for sam in angles:
            sorted_sam = np.sort(sam)[::-1]       # descending
            tau_values.append(sorted_sam[count - 1])

        tau_s = float(min(tau_values))
    else:
        tau_s = tau_s / ratio

    # ── Binary mask:  H(i,j) = 1  iff  M·||∇I_sh|| >= tau_s ────────────
    H = np.zeros((Ix, Iy), dtype=np.float64)
    H[sl] = (M[sl] * Delta_Ish[sl] >= tau_s).astype(np.float64)

    return H, tau_s
