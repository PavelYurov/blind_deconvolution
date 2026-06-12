"""
utils.py

Utility functions for the HTP (Heavy-Tailed Priors) blind deconvolution
algorithm.

Ported from MATLAB code accompanying the paper:
    J. Kotera, F. Sroubek, P. Milanfar,
    "Blind Deconvolution Using Alternating Maximum a Posteriori Estimation
     with Heavy-tailed Priors", CAIP 2013.

MATLAB в†’ Python conversion notes (CRITICAL differences):
    в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    fft2(X, M, N):
        MATLAB pads X with zeros to (M, N) and computes 2-D FFT.
        в†’ np.fft.fft2(X, s=(M, N))

    ifft2 returns complex; MATLAB uses real(ifft2(.)) at output sites.
        в†’ np.real(np.fft.ifft2(.))

    imresize(I, scale, 'method'):
        MATLAB defaults to 'bicubic' WITH anti-aliasing on downscale.
        For 'lanczos3' kernel it uses a 3-lobe sinc with a=3.
        OpenCV INTER_LANCZOS4 has 4 lobes (a=4) вЂ” NOT identical.
        в†’ We implement separable resampling with explicit kernels and
          MATLAB's antialiasing rule (kernel width scales with 1/scale
          when scale < 1).

    edgetaper(I, PSF):
        Tapers the borders of I with a blurred version of itself,
        weighted by 1 - normalized autocorrelation of PSF.
        в†’ Custom implementation matching MATLAB exactly (separable
          autocorrelation projection).

    bwmorph(BW, 'clean'):
        Removes isolated foreground pixels (no 8-connected neighbors).
        в†’ scipy.ndimage.label with structure=np.ones((3,3)) and drop
          components of size 1.

    mat2gray(A):
        Linear contrast stretch to [0, 1] using min/max of A.

    im2col(A, [m n], 'sliding'):
        Returns one column per sliding (m Г— n) window of A in
        column-major order.  Used in calculate_mse for shift-invariant
        PSF MSE.
        в†’ Vectorized via stride tricks / explicit construction matching
          MATLAB column-major flattening of each window.

    fzero(f, [a b]):
        Root finder on bracketing interval.
        в†’ scipy.optimize.brentq.
"""

from __future__ import annotations

from typing import Tuple, Callable

import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import label


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# Image normalization  (MCrestoration.m в†’ simpnormimg)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def simpnormimg(G: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize the input image so that intensity values lie in [0, 1].

    MATLAB equivalent:
        lb = min(G(:)); ub = max(G(:));
        v  = ub - lb;   m  = lb;
        I  = (double(G) - m) / v;

    Returns
    -------
    I : float64 image in [0, 1]
    m : minimum value used as offset
    v : (max - min) used as scale  (1.0 if image is constant)
    """
    G = np.asarray(G, dtype=np.float64)
    lb = float(G.min())
    ub = float(G.max())
    v = ub - lb
    if v == 0.0:
        v = 1.0
    I = (G - lb) / v
    return I, lb, v


def denormimg(U: np.ndarray, m: float, v: float) -> np.ndarray:
    """Inverse of simpnormimg: U * v + m."""
    return U * v + m


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# ROI extraction  (MCrestoration.m в†’ getROI)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def get_roi(G: np.ndarray, win: Tuple[int, int]) -> np.ndarray:
    """
    Select the central window of size `win` from image G.

    For RGB images the green channel (index 1) is used (MATLAB code uses
    `cind = 2` in 1-based indexing).  For grayscale images, the only
    channel is returned.

    If the image is smaller than the requested window in any axis,
    the window is shrunk to the image size on that axis.
    """
    G = np.asarray(G)
    if G.ndim == 3 and G.shape[2] > 1:
        ch = G[..., 1]  # green
    elif G.ndim == 3:
        ch = G[..., 0]
    else:
        ch = G

    isize = ch.shape
    win = (min(win[0], isize[0]), min(win[1], isize[1]))
    margin = ((isize[0] - win[0]) // 2, (isize[1] - win[1]) // 2)
    return ch[margin[0]:margin[0] + win[0],
              margin[1]:margin[1] + win[1]].copy()


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# mat2gray  (linear contrast stretch to [0, 1])
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def mat2gray(A: np.ndarray) -> np.ndarray:
    """
    MATLAB mat2gray(A): rescale intensities to [0, 1] using min/max of A.
    Constant arrays map to 0 (consistent with MATLAB).
    """
    A = np.asarray(A, dtype=np.float64)
    lo = float(A.min())
    hi = float(A.max())
    if hi == lo:
        return np.zeros_like(A)
    return (A - lo) / (hi - lo)


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# bwmorph 'clean'  (remove isolated foreground pixels)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def bwmorph_clean(BW: np.ndarray) -> np.ndarray:
    """
    Equivalent of MATLAB bwmorph(BW, 'clean'):
    set to 0 every foreground pixel that has no 8-connected foreground
    neighbour.
    """
    BW = np.asarray(BW, dtype=bool)
    # 8-connected count of neighbours
    structure = np.ones((3, 3), dtype=bool)
    labeled, _ = label(BW, structure=structure)
    if labeled.max() == 0:
        return BW.copy()
    counts = np.bincount(labeled.ravel())
    isolated = counts == 1
    isolated[0] = False  # background label
    out = BW.copy()
    out[isolated[labeled]] = False
    return out


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# PSF centering  (PSFestimaLnoRgrad.m в†’ centerPSF)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def center_psf(H: np.ndarray, thresh: float) -> np.ndarray:
    """
    Threshold and re-center a PSF inside its support window.

    Steps (matching MATLAB centerPSF):
      1. mat2gray-stretch H to [0, 1].
      2. Form mask m = (h_stretched >= thresh).
      3. Remove isolated pixels via bwmorph 'clean' (preserve delta-like).
      4. Find bounding box of mask; compute new top-left corner so the
         mask is centered in the original window.
      5. Shift H accordingly (with zero padding); keep window size.
      6. Normalize sum to 1.

    Works on a single 2-D PSF (we only handle the monochannel case used
    in PSFestimaLnoRgrad).
    """
    H = np.asarray(H, dtype=np.float64)
    hsize = np.array(H.shape[:2])

    # MATLAB centerPSF originally uses ``h = mat2gray(H)`` (a linear contrast
    # stretch).  For PSFs that contain a small negative noise floor (after
    # ``H = real(ifft2(FH))[:hsize, :hsize]`` is unconstrained in sign) this
    # is *equivalent to adding a DC bias* to every pixel before sum-normalising
    # to 1, which dilutes the bright peaks with a uniform background.  On
    # complex curved PSFs (b-splines, comet/hook traces) this is what makes
    # the kernel look "muddy" and lets bbox-centering snap to a near-square
    # mask.
    #
    # We deviate slightly from MATLAB by using ``H_pos = max(H, 0)`` instead
    # of ``mat2gray(H)``: it still removes negatives and gives an unbiased
    # support mask, but preserves the relative magnitudes of the bright
    # pixels.  Same DSP intent (PSF >= 0), better numerics on heavy-tailed
    # / asymmetric kernels.
    h = np.maximum(H, 0.0)
    m_max = h.max()
    if m_max <= 0:
        s = h.sum()
        return h / s if s != 0 else h
    # Threshold relative to the max (mat2gray-style) but applied to H_pos.
    m = h >= (thresh * m_max)
    m2 = bwmorph_clean(m)
    if m2.any():
        m = m2  # preserve delta-like PSFs only if cleaning removes everything

    if not m.any():
        # Nothing above threshold вЂ” just normalize what we have.
        s = h.sum()
        return h / s if s != 0 else h

    sum1 = m.any(axis=0)  # along columns в†’ shape (W,)
    sum2 = m.any(axis=1)  # along rows    в†’ shape (H,)
    L = np.array([np.argmax(sum2),                     # first nonzero row
                  np.argmax(sum1)])                    # first nonzero col
    R = np.array([len(sum2) - 1 - np.argmax(sum2[::-1]),
                  len(sum1) - 1 - np.argmax(sum1[::-1])])

    # MATLAB:  topleft_1based = fix((L1 + R1 + 1 - hsize) / 2)
    # with L1, R1 1-based  ->  L1 = L0+1, R1 = R0+1.
    # 0-based:  topleft_0based = fix((L0 + R0 + 3 - hsize) / 2) - 1
    #
    # IMPORTANT:  MATLAB `fix` truncates toward zero.  Python `//` floors.
    # For negative non-integer arguments these DIFFER by 1, producing a
    # 1-pixel ghost shift on PSFs whose mass is in the upper/left half
    # (e.g. asymmetric chevron / V-shaped motion kernels).  Use np.fix.
    val = (L + R + 3 - hsize) / 2.0
    topleft = np.fix(val).astype(np.int64) - 1  # may be negative

    # Source slice from the existing data:
    src_r0 = max(int(topleft[0]), 0)
    src_c0 = max(int(topleft[1]), 0)
    src_r1 = min(int(topleft[0] + hsize[0]), int(hsize[0]))
    src_c1 = min(int(topleft[1] + hsize[1]), int(hsize[1]))

    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return h / h.sum() if h.sum() != 0 else h

    cropped = h[src_r0:src_r1, src_c0:src_c1]   # cropped from H_pos

    # Pre-padding (top/left) and post-padding (bottom/right)
    pad_pre = np.maximum(-topleft, 0).astype(int)
    out = np.zeros_like(h)
    out[pad_pre[0]:pad_pre[0] + cropped.shape[0],
        pad_pre[1]:pad_pre[1] + cropped.shape[1]] = cropped

    # ──────────────────────────────────────────────────────────────────────
    # Sub-pass: integer-pixel MASS-CENTROID alignment.
    #
    # The bbox shift above puts the SUPPORT center at the window center.
    # For asymmetric / curved kernels (b-splines, comet, hook, dendric)
    # the *centroid* of the PSF mass can still differ from the bbox center
    # by 1‑3 pixels.  Since fft_cg_sr_al uses
    #     hshift = δ(kh//2, kw//2)
    # to anchor the convolution origin, any centroid offset translates 1:1
    # into a translation of the recovered image.  We therefore do one more
    # integer-pixel shift to put the *mass centroid* at (kh//2, kw//2).
    # This stays inside the HTP framework (it's just an extra term in the
    # same centerPSF routine) and dramatically reduces image drift on
    # asymmetric PSFs without changing the algorithm itself.
    # ──────────────────────────────────────────────────────────────────────
    s = out.sum()
    if s == 0:
        return out
    kh, kw = out.shape
    ys = np.arange(kh, dtype=np.float64)[:, None]
    xs = np.arange(kw, dtype=np.float64)[None, :]
    yc = (out * ys).sum() / s
    xc = (out * xs).sum() / s
    sy = int(round(kh // 2 - yc))
    sx = int(round(kw // 2 - xc))
    if sy != 0 or sx != 0:
        shifted = np.zeros_like(out)
        src_r0 = max(0, -sy);  src_r1 = min(kh, kh - sy)
        src_c0 = max(0, -sx);  src_c1 = min(kw, kw - sx)
        dst_r0 = max(0, sy);   dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = max(0, sx);   dst_c1 = dst_c0 + (src_c1 - src_c0)
        if src_r1 > src_r0 and src_c1 > src_c0:
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = out[src_r0:src_r1, src_c0:src_c1]
        out = shifted

    s = out.sum()
    if s != 0:
        out = out / s
    return out


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# calculate_mse  (PSFestimaLnoRgrad.m в†’ calculateMSE)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def calculate_mse(h: np.ndarray, hs: np.ndarray) -> float:
    """
    Shift-invariant MSE between estimated PSF h and ground-truth hs.

    Replicates MATLAB:
        h = h/sum(h(:))*sum(hs(:));
        R = im2col(h,[size(hs,1) size(hs,2)],'sliding');
        s = sqrt(sum((R - hs(:)).^2,1));
        r = s(ceil(prod(i(1:2))/2));   % center column

    The MATLAB code returns s evaluated at the central sliding window,
    NOT the minimum (despite the function name).  We replicate this
    exactly.
    """
    h = np.asarray(h, dtype=np.float64)
    hs = np.asarray(hs, dtype=np.float64)
    sh = np.array(h.shape)
    shs = np.array(hs.shape)

    sum_h = h.sum()
    if sum_h != 0:
        h = h / sum_h * hs.sum()

    n = sh - shs + 1  # number of windows along each axis
    if np.any(n < 1):
        # Fall back to a direct rms on the overlap region
        return float(np.sqrt(((h - hs) ** 2).sum()))

    n_total = int(np.prod(n))
    # MATLAB reshape(hs, [], 1) with column-major flattening:
    hs_col = hs.flatten(order='F')

    # MATLAB im2col(...,'sliding') ordering: windows are enumerated in
    # column-major (Fortran) order; within each window the patch is also
    # flattened in column-major order.
    # Center column index (1-based): ceil(n_total/2); 0-based: ceil/2 - 1
    center_idx = int(np.ceil(n_total / 2)) - 1
    cj = center_idx // n[0]   # column-major: row-of-window cycles fastest
    ci = center_idx % n[0]

    window = h[ci:ci + shs[0], cj:cj + shs[1]].flatten(order='F')
    return float(np.sqrt(((window - hs_col) ** 2).sum()))


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# fft2 with zero-padding  (MATLAB fft2(X, M, N))
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def fft2_pad(X: np.ndarray, M: int, N: int) -> np.ndarray:
    """Equivalent of MATLAB fft2(X, M, N): zero-pad X to (M, N) then FFT."""
    return np.fft.fft2(X, s=(M, N))


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# Lp-norm prior shrinkage  (asetupLnormPrior.m)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def setup_lp_prior(q: float, alpha: float, beta: float) -> Callable[
        [np.ndarray, np.ndarray], np.ndarray]:
    """
    Build the half-quadratic shrinkage operator for the prior

        phi(s) = alpha * |s|^q     for |s| > u_star
        phi(s) = (beta/2) * s^2    for |s| <= u_star

    Returns a callable fh(DU, normDU) -> V matching MATLAB asetupLnormPrior:
        V = DU .* (normDU - k) ./ normDU      where normDU > u_star
        V = 0                                  elsewhere

    For 0 < q < 1 the thresholds (u_star, v_star) are found numerically
    (no closed form), exactly mirroring MATLAB fzero usage.
    """
    if q == 1.0:
        v_star = 0.0
        u_star = alpha / beta
    elif q == 0.0:
        v_star = np.sqrt(2.0 * alpha / beta)
        u_star = v_star
    else:
        ratio = alpha / beta
        # leftmarker: zero of -v + ratio*v^(q-1)*(1-q)*q
        f1 = lambda v: -v + ratio * (v ** (q - 1)) * (1.0 - q) * q
        leftmarker = brentq(f1, np.finfo(float).eps, 10.0)
        # v_star: zero of -0.5*v^2 + ratio*v^q*(1-q)
        f2 = lambda v: -0.5 * v * v + ratio * (v ** q) * (1.0 - q)
        v_star = brentq(f2, leftmarker, 10.0)
        u_star = v_star + ratio * q * (v_star ** (q - 1))

    k = u_star - v_star

    def fh(DU: np.ndarray, normDU: np.ndarray) -> np.ndarray:
        V = np.zeros_like(DU)
        m = normDU > u_star
        # Avoid division by zero: m guarantees normDU > u_star >= 0,
        # but if u_star == 0 keep a safe path.
        nDp = normDU[m]
        # Where normDU is exactly 0 it cannot satisfy m unless u_star<0,
        # which never happens here.
        V[m] = DU[m] * (nDp - k) / nDp
        return V

    return fh


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# Lanczos3 / Bicubic separable resampling  (MATLAB imresize)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _kernel_cubic(x: np.ndarray) -> np.ndarray:
    """MATLAB imresize cubic kernel (a = -0.5)."""
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    f = ((1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1.0)
         + (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0)
           * ((absx > 1.0) & (absx <= 2.0)))
    return f


def _kernel_lanczos3(x: np.ndarray) -> np.ndarray:
    """3-lobe Lanczos kernel: sinc(x)*sinc(x/3) for |x|<3, else 0."""
    f = np.zeros_like(x, dtype=np.float64)
    m = np.abs(x) < 3.0
    xm = x[m]
    # Use np.sinc which is sin(pi*x)/(pi*x)
    f[m] = np.sinc(xm) * np.sinc(xm / 3.0)
    return f


_KERNEL_WIDTHS = {
    'bicubic': 4.0,
    'cubic':   4.0,
    'lanczos3': 6.0,
}

_KERNEL_FUNCS = {
    'bicubic':  _kernel_cubic,
    'cubic':    _kernel_cubic,
    'lanczos3': _kernel_lanczos3,
}


def _contributions(in_length: int, out_length: int, scale: float,
                   kernel: Callable, kernel_width: float):
    """
    MATLAB imresize 'contributions' along one axis.  Returns
    (weights, indices) where for each output sample i, the value is
        sum_j weights[i, j] * input[indices[i, j]]
    with replicate boundary (clamped indices).

    Anti-aliasing: when scale < 1 the kernel is stretched by 1/scale,
    matching MATLAB's default antialiasing=True behaviour.
    """
    # MATLAB: u = (x/scale + 0.5*(1 - 1/scale))   with x = 1..out_length
    # Convert to 0-based:
    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)

    if scale < 1.0:
        kernel_width_eff = kernel_width / scale
        kernel_eff = lambda t: scale * kernel(scale * t)
    else:
        kernel_width_eff = kernel_width
        kernel_eff = kernel

    # Left-most contributing input sample (1-based MATLAB):
    left = np.floor(u - kernel_width_eff / 2.0)
    P = int(np.ceil(kernel_width_eff)) + 2
    # indices (1-based), shape (out_length, P)
    indices = left[:, None] + np.arange(P, dtype=np.float64)[None, :]
    # weights at those positions
    weights = kernel_eff(u[:, None] - indices)
    # normalize per-row
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum[weights_sum == 0] = 1.0
    weights = weights / weights_sum

    # Convert MATLAB 1-based indices to 0-based and apply MIRROR-REFLECT
    # boundary, exactly as MATLAB imresize does:
    #   aux = [1:N, N:-1:1]  (length 2N)   indices_clamped = aux(mod(idx-1, 2N)+1)
    indices = indices.astype(np.int64) - 1  # 0-based, may be negative or >= in_length
    period = 2 * in_length
    mirror = np.concatenate([np.arange(in_length),
                             np.arange(in_length - 1, -1, -1)])
    indices = mirror[np.mod(indices, period)]

    # Drop columns that are all zero weights (MATLAB does this)
    keep = np.any(weights != 0, axis=0)
    weights = weights[:, keep]
    indices = indices[:, keep]

    return weights, indices


def _resize_along_dim(A: np.ndarray, dim: int,
                      weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Apply 1-D resampling along the given axis."""
    # Move target dim to position 0, gather, weight, sum, restore.
    A = np.moveaxis(A, dim, 0)
    gathered = A[indices, ...]                 # (out, P, ...)
    # weights shape (out, P) вЂ” broadcast over remaining dims
    w_shape = (weights.shape[0], weights.shape[1]) + (1,) * (gathered.ndim - 2)
    out = (gathered * weights.reshape(w_shape)).sum(axis=1)
    return np.moveaxis(out, 0, dim)


def imresize_matlab(A: np.ndarray, scale, method: str = 'bicubic') -> np.ndarray:
    """
    Port of MATLAB imresize(A, scale, method) with anti-aliasing.

    Parameters
    ----------
    A : 2-D or 3-D array (last axis = channels).
    scale : float (uniform scale) or 2-tuple/list (sy, sx) or output
        size (h, w) when both are integers > 8 вЂ” for our use-cases we
        only need uniform float scale, but tuples are also supported.
    method : 'bicubic' (= 'cubic') or 'lanczos3'.

    Returns
    -------
    Resized array of dtype float64.
    """
    A = np.asarray(A, dtype=np.float64)
    if method not in _KERNEL_FUNCS:
        raise ValueError(f'Unsupported method: {method}')

    in_h, in_w = A.shape[:2]

    if np.isscalar(scale):
        sy = sx = float(scale)
        # MATLAB imresize uses round() for scalar scales
        out_h = max(1, int(np.round(in_h * sy)))
        out_w = max(1, int(np.round(in_w * sx)))
    else:
        scale = list(scale)
        if (len(scale) == 2 and isinstance(scale[0], (int, np.integer))
                and isinstance(scale[1], (int, np.integer))):
            out_h, out_w = int(scale[0]), int(scale[1])
            sy = out_h / in_h
            sx = out_w / in_w
        else:
            sy, sx = float(scale[0]), float(scale[1])
            out_h = max(1, int(np.round(in_h * sy)))
            out_w = max(1, int(np.round(in_w * sx)))

    kfn = _KERNEL_FUNCS[method]
    kw = _KERNEL_WIDTHS[method]

    # MATLAB resizes along the dimension with the smaller scale first
    # (more aggressive low-pass) for better quality.  We mirror that.
    order = (0, 1) if sy <= sx else (1, 0)
    sizes = {0: (in_h, out_h, sy), 1: (in_w, out_w, sx)}

    out = A
    for dim in order:
        in_len, out_len, s = sizes[dim]
        if in_len == out_len and s == 1.0:
            continue
        weights, indices = _contributions(in_len, out_len, s, kfn, kw)
        out = _resize_along_dim(out, dim, weights, indices)

    return out


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# edgetaper  (MATLAB edgetaper)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def edgetaper_matlab(I: np.ndarray, PSF: np.ndarray) -> np.ndarray:
    """
    Port of MATLAB edgetaper(I, PSF).

    Algorithm (matches MATLAB exactly):
        For each axis k:
            proj  = sum(PSF, along the OTHER axis)              # length pk
            z     = ifft(|fft(proj, N-1)|^2)                    # length N-1
                    (= circular autocorrelation, peak at lag 0)
            z     = [z; z(1)]                                   # length N
                    (peak now at indices 0 AND N-1, i.e. at the EDGES)
            beta  = 1 - z / max(z)                              # в‰€1 interior, в†’0 edges
        alpha   = outer(beta_y, beta_x)                          # в‰€1 interior, в†’0 borders
        otf     = psf2otf(PSF, size(I))                          # centered convolution
        blurred = real( ifft2( fft2(I) .* otf ) )
        out     = alpha .* I + (1 - alpha) .* blurred

    The crucial MATLAB-specific detail is the *circular* autocorrelation
    via FFT, padded to N-1 and then closed with one extra sample so the
    peak lies at the edges of the length-N array (NOT centered).
    """
    I = np.asarray(I, dtype=np.float64)
    PSF = np.asarray(PSF, dtype=np.float64)
    s = PSF.sum()
    if s != 0:
        PSF = PSF / s

    if I.ndim == 2:
        return _edgetaper_2d(I, PSF)
    out = np.empty_like(I)
    for c in range(I.shape[2]):
        out[..., c] = _edgetaper_2d(I[..., c], PSF)
    return out


def _edgetaper_alpha_1d(proj: np.ndarray, N: int) -> np.ndarray:
    """
    1-D edgetaper weight of length N built from a PSF projection.

    Matches MATLAB:
        z = abs(fft(proj, N-1)).^2;
        z = real(ifft(z));
        z = [z; z(1)];          % length N, peak at 0 and N-1
        beta = 1 - z/max(z);    % в‰€1 in middle, в†’0 at edges
    """
    if N <= 1:
        return np.zeros(max(N, 0), dtype=np.float64)
    L = N - 1
    F = np.fft.fft(proj, L)
    z = np.real(np.fft.ifft(np.abs(F) ** 2))
    z = np.concatenate([z, z[:1]])  # length N
    zmax = z.max()
    if zmax <= 0:
        return np.ones(N, dtype=np.float64)
    return 1.0 - z / zmax


def _psf2otf_centered(PSF: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """psf2otf(PSF, shape): zero-pad, circular shift so center -> (0,0), FFT."""
    ph, pw = PSF.shape
    H, W = shape
    padded = np.zeros((H, W), dtype=np.float64)
    padded[:ph, :pw] = PSF
    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return np.fft.fft2(padded)


def _edgetaper_2d(I: np.ndarray, PSF: np.ndarray) -> np.ndarray:
    H, W = I.shape
    proj_y = PSF.sum(axis=1)             # length ph (vertical projection)
    proj_x = PSF.sum(axis=0)             # length pw (horizontal projection)

    beta_y = _edgetaper_alpha_1d(proj_y, H)
    beta_x = _edgetaper_alpha_1d(proj_x, W)
    alpha = np.outer(beta_y, beta_x)     # в‰€1 interior, в†’0 borders
    alpha = np.clip(alpha, 0.0, 1.0)

    OTF = _psf2otf_centered(PSF, (H, W))
    blurred = np.real(np.fft.ifft2(np.fft.fft2(I) * OTF))

    return alpha * I + (1.0 - alpha) * blurred
