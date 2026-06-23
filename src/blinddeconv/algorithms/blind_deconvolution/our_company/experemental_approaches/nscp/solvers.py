"""
solvers.py

Core solver functions for NSCP (Novel Sparse Channel Prior) blind deconvolution.

Ported from the Python implementation by D. Yang.
Reference:
    D. Yang, X. Wu, H. Yin: "Blind Image Deblurring via a Novel Sparse
    Channel Prior", Mathematics, 2022.
    https://www.mdpi.com/2227-7390/10/8/1238

Contains:
    update_l       — Frequency-domain closed-form solution for the latent
                     image l  (Eq. 18).
    update_kernel  — Frequency-domain closed-form solution for the blur
                     kernel k in gradient space  (Eq. 22).
    final_restore  — Non-blind Wiener-filter deconvolution for the final
                     sharp image  (simplified Algorithm 1).
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from .utils import pad_kernel_centered, extract_kernel_center

EPS = 1e-8


# ═════════════════════════════════════════════════════════════════════════════
# Helper
# ═════════════════════════════════════════════════════════════════════════════

def _crop_center(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Crop the centre (out_h × out_w) region from *img*."""
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    y0 = cy - out_h // 2
    x0 = cx - out_w // 2
    return img[y0 : y0 + out_h, x0 : x0 + out_w]


def _pad_img(img: np.ndarray, h2: int, w2: int) -> np.ndarray:
    """Zero-pad *img* to size (h2, w2[, C])."""
    if img.ndim == 3:
        out = np.zeros((h2, w2, img.shape[2]), dtype=np.float32)
        out[: img.shape[0], : img.shape[1], :] = img
    else:
        out = np.zeros((h2, w2), dtype=np.float32)
        out[: img.shape[0], : img.shape[1]] = img
    return out


# ═════════════════════════════════════════════════════════════════════════════
# update_l  —  Eq. (18)
#
#         F(k)* · F(b)  +  λ · F(g)  +  ξ · F(p)
#   l = ───────────────────────────────────────────────
#        |F(k)|²  +  λ · (|F(∇h)|² + |F(∇v)|²)  +  ξ
#
# where  F(g) = F(∇h)* · F(g_h)  +  F(∇v)* · F(g_v)
#
# Implementation uses linear-convolution padding (H2 = H+kh-1, W2 = W+kw-1)
# and crops the valid centre after IFFT.
# ═════════════════════════════════════════════════════════════════════════════

def update_l(
    l: np.ndarray,
    k: np.ndarray,
    b: np.ndarray,
    g: tuple,
    p: np.ndarray,
    lam: float,
    xi: float,
) -> np.ndarray:
    """
    Frequency-domain solve for the latent image (Eq. 18).

    Parameters
    ----------
    l   : current latent image estimate, (H, W) or (H, W, C), float32
    k   : blur kernel, (kh, kw)
    b   : blurred observation, same shape as *l*
    g   : tuple (g_h, g_v) — thresholded gradient auxiliary variables
    p   : dark-channel auxiliary variable, same shape as *l*
    lam : weight λ  (gradient fidelity)
    xi  : weight ξ  (dark-channel fidelity)

    Returns
    -------
    l_new : updated latent image, same (H, W[, C]) as *b*, clipped to [0, 1]
    """
    # Spatial dimensions
    if b.ndim == 3:
        H, W, _C = b.shape
    else:
        H, W = b.shape
    fft_axes = (0, 1)

    kh, kw = k.shape
    H2 = H + kh - 1
    W2 = W + kw - 1

    # ── Pad images & priors ──────────────────────────────────────────────
    bpad = _pad_img(b, H2, W2)
    ppad = _pad_img(p, H2, W2)

    Fb = fft2(bpad, s=(H2, W2), axes=fft_axes)
    Fp = fft2(ppad, s=(H2, W2), axes=fft_axes)

    # ── Kernel FFT (centred padding → ifftshift) ────────────────────────
    kpad = pad_kernel_centered(k, (H2, W2))
    Fk = fft2(kpad, s=(H2, W2))
    if b.ndim == 3:
        Fk = Fk[:, :, np.newaxis]

    # ── Derivative filters (forward difference convention) ──────────────
    # Forward diff: result[m] = x[m+1] - x[m].
    # As circular filter: d[0] = -1, d[N-1] = +1
    # This matches compute_gradients() which also uses forward difference.
    # (The original code used backward diff [-1,+1] at [0,1] which did NOT
    #  match the forward-diff gradient, causing a 1-pixel phase shift.)
    Dh_sp = np.zeros((H2, W2), dtype=np.float32)
    Dh_sp[0, 0] = -1.0
    if W2 > 1:
        Dh_sp[0, -1] = 1.0

    Dv_sp = np.zeros((H2, W2), dtype=np.float32)
    Dv_sp[0, 0] = -1.0
    if H2 > 1:
        Dv_sp[-1, 0] = 1.0

    FDh = fft2(Dh_sp, s=(H2, W2))
    FDv = fft2(Dv_sp, s=(H2, W2))

    if b.ndim == 3:
        FDh = FDh[:, :, np.newaxis]
        FDv = FDv[:, :, np.newaxis]

    # ── Gradient FFTs ───────────────────────────────────────────────────
    # No roll needed: the derivative filter now uses the same forward-
    # difference convention as compute_gradients().  (The original code
    # applied np.roll(-1) to compensate for a backward-diff / forward-diff
    # mismatch, but the roll itself was incorrect — it shifted the prior
    # by 1 pixel, producing diagonal streak artefacts.)
    g_h, g_v = g

    gh_pad = _pad_img(g_h, H2, W2)
    gv_pad = _pad_img(g_v, H2, W2)

    Fgh = fft2(gh_pad, s=(H2, W2), axes=fft_axes)
    Fgv = fft2(gv_pad, s=(H2, W2), axes=fft_axes)

    # ── Closed-form solve ────────────────────────────────────────────────
    Fg = np.conj(FDh) * Fgh + np.conj(FDv) * Fgv

    numerator = np.conj(Fk) * Fb + lam * Fg + xi * Fp
    denominator = (
        np.abs(Fk) ** 2
        + lam * (np.abs(FDh) ** 2 + np.abs(FDv) ** 2)
        + xi
    )
    denominator = np.maximum(denominator, 1e-2)

    Fl = numerator / denominator
    l_full = np.real(ifft2(Fl, axes=fft_axes))

    # ── Crop & clamp ─────────────────────────────────────────────────────
    # All inputs (b, p, g) were zero-padded at the top-left corner.
    # The kernel was centred then ifftshifted to (0,0).  Therefore the
    # FFT solution is aligned with the top-left corner of the padded
    # array — exactly the same convention DCP uses (S = S[:H, :W]).
    # The original code used _crop_center which extracted the MIDDLE
    # of the array, shifting the result by ~(kh//2, kw//2) pixels and
    # destroying the alignment between l and b for kernel estimation.
    l_new = l_full[:H, :W]
    if l_new.ndim == 3 and b.ndim == 3 and l_new.shape[2] != b.shape[2]:
        l_new = l_new[:, :, :b.shape[2]]  # safety
    l_new = np.clip(l_new, 0.0, 1.0)
    return l_new


# ═════════════════════════════════════════════════════════════════════════════
# update_kernel  —  Eq. (22)
#
#          F(∇l)* · F(∇b)
#   k = ────────────────────
#        F(∇l)* · F(∇l) + γ
#
# Estimation is performed in the gradient domain (Eq. 21) which helps
# suppress ringing artefacts and eliminate noise.
# ═════════════════════════════════════════════════════════════════════════════

def update_kernel(
    l: np.ndarray,
    b: np.ndarray,
    gamma: float,
    image_shape: tuple,
    prev_k: np.ndarray = None,
) -> np.ndarray:
    """
    Frequency-domain solve for the blur kernel (Eq. 22).

    Parameters
    ----------
    l           : current latent image, (H, W) or (H, W, C)
    b           : blurred observation, same shape as *l*
    gamma       : L2 regularisation weight for the kernel
    image_shape : (H, W) of the image at the current pyramid level
    prev_k      : previous kernel estimate (used for target crop size)

    Returns
    -------
    k_small : estimated compact kernel, normalised
    """
    H, W = image_shape[:2]

    # Local gradient (forward difference, same convention as model code)
    def _get_grad(img):
        gh = np.zeros_like(img)
        gv = np.zeros_like(img)
        gh[:, :-1] = img[:, 1:] - img[:, :-1]
        gv[:-1, :] = img[1:, :] - img[:-1, :]
        return gh, gv

    grad_l_h, grad_l_v = _get_grad(l)
    grad_b_h, grad_b_v = _get_grad(b)

    # FFT on spatial axes
    fft_axes = (0, 1)
    Flh = fft2(grad_l_h, axes=fft_axes)
    Flv = fft2(grad_l_v, axes=fft_axes)
    Fbh = fft2(grad_b_h, axes=fft_axes)
    Fbv = fft2(grad_b_v, axes=fft_axes)

    # Numerator & denominator (per channel)
    num_ch = np.conj(Flh) * Fbh + np.conj(Flv) * Fbv
    denom_ch = np.conj(Flh) * Flh + np.conj(Flv) * Flv

    # Sum across colour channels (if any)
    if l.ndim == 3:
        numerator = np.sum(num_ch, axis=2)
        denominator = np.sum(denom_ch, axis=2)
    else:
        numerator = num_ch
        denominator = denom_ch

    denominator += gamma
    Fk = numerator / (denominator + EPS)

    k_full = np.real(ifft2(Fk, axes=fft_axes))

    # Extract compact kernel (crop around centre)
    expected = prev_k.shape if prev_k is not None else None
    k_small = extract_kernel_center(k_full, expected_size=expected)

    # Safety: if kernel vanished, fall back to previous or delta
    if k_small.sum() <= 1e-12:
        if prev_k is not None and prev_k.sum() > 1e-12:
            k_small = prev_k.copy()
        else:
            k_small = np.zeros((3, 3), dtype=np.float32)
            k_small[1, 1] = 1.0

    return k_small


# ═════════════════════════════════════════════════════════════════════════════
# final_restore  —  Wiener-filter non-blind deconvolution
#
# Simplified substitute for Algorithm 1 of the paper (which averages
# hyper-Laplacian and TV restorations).  The original code uses a
# straightforward Wiener filter:
#
#          Conj(K) · G
#   F = ─────────────────
#        |K|² + SNR_const
#
# with  SNR_const ≈ 0.015.
# ═════════════════════════════════════════════════════════════════════════════

def final_restore(
    img: np.ndarray,
    kernel: np.ndarray,
    snr_const: float = 0.015,
) -> np.ndarray:
    """
    Non-blind Wiener-filter deconvolution.

    Parameters
    ----------
    img       : blurred image, (H, W) or (H, W, C), float32 in [0, 1]
    kernel    : estimated blur kernel (kh, kw)
    snr_const : noise-to-signal constant (higher → smoother).
                Default 0.015.

    Returns
    -------
    result : deconvolved image, same shape, clipped to [0, 1]
    """
    # Normalise kernel
    kernel = kernel / (np.sum(kernel) + 1e-8)

    H, W = img.shape[:2]
    kh, kw = kernel.shape

    # Pad kernel to image size, centred, then ifftshift
    k_pad = np.zeros((H, W), dtype=np.float32)
    y_off = (H - kh) // 2
    x_off = (W - kw) // 2
    k_pad[y_off : y_off + kh, x_off : x_off + kw] = kernel
    k_pad = ifftshift(k_pad)

    K_f = fft2(k_pad)
    denom = np.abs(K_f) ** 2 + snr_const

    def _apply_wiener(channel_img):
        Y_f = fft2(channel_img)
        numer = np.conj(K_f) * Y_f
        return np.real(ifft2(numer / denom))

    if img.ndim == 3:
        final = np.zeros_like(img)
        for c in range(img.shape[2]):
            final[:, :, c] = _apply_wiener(img[:, :, c])
    else:
        final = _apply_wiener(img)

    return np.clip(final, 0.0, 1.0)
