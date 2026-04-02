"""
utils.py

Utility functions for Variational Bayesian Pansharpening / Super-Resolution.

Ported from MATLAB code:
    Pérez-Bueno, F., Vega, M., Mateos, J., Molina, R., & Katsaggelos, A. K. (2020).
    Variational Bayesian Pansharpening with Super-Gaussian Sparse Image Priors.
    Sensors, 20(18), 5308.

    M. Vega, J. Mateos, R. Molina, and A. K. Katsaggelos, "Super resolution of
    multispectral images using TV image models," in International Conference on
    Knowledge-Based and Intelligent Information and Engineering Systems, 2008,
    pp. 408-415.

MATLAB -> Python porting notes:
    - Indexing: MATLAB is 1-based, Python is 0-based.
    - Array order: MATLAB is column-major (Fortran), NumPy default is row-major (C).
      For the polyphase block-FFT representation, all 4D arrays use shape
      [low_nr, low_nc, bkn2, bkn2] with standard C-order.
    - reshape(v', N*M, 1) in MATLAB (transpose + col-major flatten) is equivalent
      to v.flatten() in Python (row-major flatten).
    - circshift(A, [r, c])  ->  np.roll(np.roll(A, r, axis=0), c, axis=1)
    - fft2/ifft2/conj       ->  np.fft.fft2 / np.fft.ifft2 / np.conj

Contains:
    Block-FFT infrastructure for polyphase decimation/interpolation model:
        im_decomp, im_comp, blk_fft2, blk_ifft2, blk_fd_conv,
        blk_fd_trace, blk_fft2_DtD, blk_DtY

    FFT-based operator representations:
        cent_nucleus2fft, Tcent_nucleus2fft,
        cent_nucleus2blk_fft2, Tcent_nucleus2blk_fft2

    Circular gradient operators:
        circ_gradient2, Tcirc_gradient2

    Multi-band vector <-> image conversions:
        convertToMBVec, convertToMBImg

    Prior model functions:
        compute_Wpvb, getfilters, getkappa, weight_log, weight_lp

    Observation model helpers:
        calclam, image_normalize, image_denormalize, get_psf
"""

import numpy as np
from numpy.fft import fft2, ifft2

_EPS = np.finfo(np.float64).eps


# ═════════════════════════════════════════════════════════════════════════════
# Polyphase decomposition / composition  (im_decomp.m / im_comp.m)
# ═════════════════════════════════════════════════════════════════════════════

def im_decomp(img, bknr, bknc):
    """Polyphase decomposition of a 2-D image.

    Splits an [nr, nc] image into [low_nr, low_nc, bknr*bknc, 1] polyphase
    components by extracting every bknr-th row / bknc-th column.

    Parameters
    ----------
    img  : (nr, nc) array
    bknr : int — vertical block (decimation) factor
    bknc : int — horizontal block (decimation) factor

    Returns
    -------
    out : (low_nr, low_nc, bknr*bknc, 1) real array
    """
    nr, nc = img.shape[:2]
    low_nr = int(np.ceil(nr / bknr))
    low_nc = int(np.ceil(nc / bknc))

    pad_r = bknr * low_nr - nr
    pad_c = bknc * low_nc - nc
    if pad_r > 0 or pad_c > 0:
        img = np.pad(img, ((0, pad_r), (0, pad_c)))

    bkn2 = bknr * bknc
    out = np.zeros((low_nr, low_nc, bkn2, 1))
    pos = 0
    for spr in range(bknr):
        for spc in range(bknc):
            out[:, :, pos, 0] = img[spr::bknr, spc::bknc]
            pos += 1
    return out


def im_comp(blk, bknr, bknc):
    """Inverse polyphase composition.

    Reconstructs an [nr, nc] image from its [low_nr, low_nc, bkn2, 1]
    polyphase representation.

    Parameters
    ----------
    blk  : (low_nr, low_nc, bkn2, 1) array
    bknr : int — vertical block factor
    bknc : int — horizontal block factor

    Returns
    -------
    out : (low_nr*bknr, low_nc*bknc) real array
    """
    low_nr, low_nc = blk.shape[:2]
    nr = low_nr * bknr
    nc = low_nc * bknc
    out = np.zeros((nr, nc))
    pos = 0
    for spr in range(bknr):
        for spc in range(bknc):
            out[spr::bknr, spc::bknc] = np.real(blk[:, :, pos, 0])
            pos += 1
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Block FFT / iFFT  (blk_fft2.m / blk_ifft2.m)
# ═════════════════════════════════════════════════════════════════════════════

def blk_fft2(data):
    """Block-wise 2-D FFT over the first two dimensions.

    Parameters
    ----------
    data : (low_nr, low_nc, p, q) real or complex array

    Returns
    -------
    out : (low_nr, low_nc, p, q) complex array
    """
    out = np.zeros_like(data, dtype=complex)
    for i in range(data.shape[2]):
        for j in range(data.shape[3]):
            out[:, :, i, j] = fft2(data[:, :, i, j])
    return out


def blk_ifft2(data):
    """Block-wise 2-D inverse FFT (returns real part).

    Parameters
    ----------
    data : (low_nr, low_nc, p, q) complex array

    Returns
    -------
    out : (low_nr, low_nc, p, q) real array
    """
    out = np.zeros(data.shape)
    for i in range(data.shape[2]):
        for j in range(data.shape[3]):
            out[:, :, i, j] = np.real(ifft2(data[:, :, i, j]))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Block Fourier-domain arithmetic  (blk_fd_conv.m / blk_fd_trace.m / etc.)
# ═════════════════════════════════════════════════════════════════════════════

def blk_fd_conv(Op1, Op2):
    """Block Fourier-domain matrix multiplication (convolution).

    Performs element-wise "matrix multiply" over the block indices (axes 2,3)
    while the spatial-frequency axes (0,1) are point-wise.

    Op1 : (S1, S2, p, q)
    Op2 : (S1, S2, q, r)
    Out : (S1, S2, p, r)
    """
    assert Op1.shape[3] == Op2.shape[2], (
        f"Inner block dimensions must match: Op1[...,{Op1.shape[3]}] vs Op2[...,{Op2.shape[2]}]"
    )
    assert Op1.shape[:2] == Op2.shape[:2], "Spatial dimensions must match"

    dt = complex if (np.iscomplexobj(Op1) or np.iscomplexobj(Op2)) else float
    out = np.zeros((Op2.shape[0], Op2.shape[1], Op1.shape[2], Op2.shape[3]),
                   dtype=dt)
    for i in range(out.shape[2]):
        for j in range(out.shape[3]):
            for k in range(Op1.shape[3]):
                out[:, :, i, j] += Op1[:, :, i, k] * Op2[:, :, k, j]
    return out


def blk_fd_trace(m):
    """Scalar trace of a block Fourier-domain matrix.

    m : (S1, S2, p, p) block matrix.
    Returns real scalar  sum_{i,j} trace(m[i,j,:,:]).
    """
    tr = 0.0 + 0.0j
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            tr += np.trace(m[i, j, :, :])

    if abs(np.imag(tr)) > 1e-3:
        pass  # numerical noise in complex block-FFT ops; MATLAB takes real() implicitly
    return float(np.real(tr))


def blk_fft2_DtD(nr, nc, bknr, bknc):
    """D^T D operator in block-FFT representation.

    D is the regular decimation-by-(bknr, bknc) operator.
    D^T D selects only the (0,0) polyphase component.

    Always called  with iy=1, ix=1 in MATLAB (=index 0 in Python).

    Returns
    -------
    out : (low_nr, low_nc, bkn2, bkn2) real array
    """
    low_nr = int(np.ceil(nr / bknr))
    low_nc = int(np.ceil(nc / bknc))
    bkn2 = bknr * bknc
    out = np.zeros((low_nr, low_nc, bkn2, bkn2))
    out[:, :, 0, 0] = 1.0
    return out


def blk_DtY(Y, bknr, bknc):
    """D^T Y : up-sampling by zero-insertion.

    Places the LR observation Y into the (0,0) polyphase slot.

    Parameters
    ----------
    Y    : (low_nr, low_nc) 2-D observation
    bknr : int — vertical decimation factor
    bknc : int — horizontal decimation factor

    Returns
    -------
    out : (low_nr, low_nc, bkn2, 1) real array
    """
    low_nr, low_nc = Y.shape[:2]
    bkn2 = bknr * bknc
    out = np.zeros((low_nr, low_nc, bkn2, 1))
    out[:, :, 0, 0] = Y
    return out


# ═════════════════════════════════════════════════════════════════════════════
# PSF <-> FFT conversions  (cent_nucleus2fft.m / Tcent_nucleus2fft.m)
# ═════════════════════════════════════════════════════════════════════════════

def cent_nucleus2fft(spkernel, nr, nc):
    """Optical Transfer Function of a centered convolution kernel.

    Places the kernel with its center at position (0, 0) of an (nr, nc)
    array (with circular wrap-around for negative indices) and takes fft2.

    Parameters
    ----------
    spkernel : 2-D array — spatial-domain convolution kernel
    nr, nc   : int — target image dimensions

    Returns
    -------
    H : (nr, nc) complex array — OTF
    """
    spkernel = np.asarray(spkernel, dtype=np.float64)
    kh, kw = spkernel.shape

    # Crop kernel if larger than image (rare edge case)
    if kh > nr or kw > nc:
        fac_orig = spkernel.sum()
        if kh > nr:
            c = (kh - nr) // 2
            spkernel = spkernel[c:c + nr, :]
        kh, kw = spkernel.shape
        if kw > nc:
            c = (kw - nc) // 2
            spkernel = spkernel[:, c:c + nc]
        kh, kw = spkernel.shape
        fac_new = spkernel.sum()
        if abs(fac_new) > _EPS:
            spkernel *= fac_orig / fac_new

    cr, cc = kh // 2, kw // 2
    h = np.zeros((nr, nc))
    for i in range(kh):
        for j in range(kw):
            h[(i - cr) % nr, (j - cc) % nc] = spkernel[i, j]
    return fft2(h)


def Tcent_nucleus2fft(spkernel, nr, nc):
    """Adjoint (conjugate-transpose) OTF.

    For a real kernel, the adjoint convolution corresponds to
    convolution with the flipped kernel, whose OTF is conj(H).
    """
    return np.conj(cent_nucleus2fft(spkernel, nr, nc))


# ═════════════════════════════════════════════════════════════════════════════
# Kernel -> block-FFT  (cent_nucleus2blk_fft2.m / Tcent_nucleus2blk_fft2.m)
# ═════════════════════════════════════════════════════════════════════════════

def cent_nucleus2blk_fft2(spkernel, nr, nc, bknr, bknc):
    """Convolution kernel -> polyphase block-FFT representation.

    Converts a spatial-domain kernel into the block circulant
    representation used by the decimation observation model.

    Returns
    -------
    H : (low_nr, low_nc, bkn2, bkn2) complex array
    """
    low_nr = int(np.ceil(nr / bknr))
    low_nc = int(np.ceil(nc / bknc))
    bkn2 = bknr * bknc

    H = np.zeros((low_nr, low_nc, bkn2, bkn2))

    # Centered impulse response
    cent_nuc = np.real(ifft2(cent_nucleus2fft(spkernel, nr, nc)))

    col_idx = 0
    inter = cent_nuc.copy()
    for jr in range(bknr):
        for jc in range(bknc):
            col = im_decomp(inter, bknr, bknc)
            H[:, :, :, col_idx] = col[:, :, :, 0]
            col_idx += 1
            inter = np.roll(inter, 1, axis=1)      # circshift([0 1])
        cent_nuc = np.roll(cent_nuc, 1, axis=0)     # circshift([1 0])
        inter = cent_nuc.copy()

    return blk_fft2(H)


def Tcent_nucleus2blk_fft2(spkernel, nr, nc, bknr, bknc):
    """Adjoint kernel -> polyphase block-FFT representation.

    Like cent_nucleus2blk_fft2 but for H^T.
    """
    low_nr = int(np.ceil(nr / bknr))
    low_nc = int(np.ceil(nc / bknc))
    bkn2 = bknr * bknc

    H = np.zeros((low_nr, low_nc, bkn2, bkn2))

    cent_nuc = np.real(ifft2(Tcent_nucleus2fft(spkernel, nr, nc)))

    col_idx = 0
    inter = cent_nuc.copy()
    for jr in range(bknr):
        for jc in range(bknc):
            col = im_decomp(inter, bknr, bknc)
            H[:, :, :, col_idx] = col[:, :, :, 0]
            col_idx += 1
            inter = np.roll(inter, 1, axis=1)
        cent_nuc = np.roll(cent_nuc, 1, axis=0)
        inter = cent_nuc.copy()

    return blk_fft2(H)


# ═════════════════════════════════════════════════════════════════════════════
# Circular gradient operators  (circ_gradient2.m / Tcirc_gradient2.m)
# ═════════════════════════════════════════════════════════════════════════════

def circ_gradient2(f):
    """Circular (periodic) backward-difference gradient.

    Returns
    -------
    dfh : (n, m) — horizontal differences  f[:,j] - f[:,j+1]  (wrap)
    dfv : (n, m) — vertical differences    f[i,:] - f[i+1,:]  (wrap)
    """
    dfh = np.concatenate([f[:, :-1] - f[:, 1:],
                          (f[:, -1] - f[:, 0])[:, np.newaxis]], axis=1)
    dfv = np.concatenate([f[:-1, :] - f[1:, :],
                          (f[-1, :] - f[0, :])[np.newaxis, :]], axis=0)
    return dfh, dfv


def Tcirc_gradient2(f):
    """Adjoint (transpose) of the circular gradient.

    Returns
    -------
    dfhT : (n, m) — adjoint horizontal
    dfvT : (n, m) — adjoint vertical
    """
    dfhT = np.concatenate([(f[:, 0] - f[:, -1])[:, np.newaxis],
                           f[:, 1:] - f[:, :-1]], axis=1)
    dfvT = np.concatenate([(f[0, :] - f[-1, :])[np.newaxis, :],
                           f[1:, :] - f[:-1, :]], axis=0)
    return dfhT, dfvT


# ═════════════════════════════════════════════════════════════════════════════
# Multi-band vector <-> image conversions  (convertToMBVec.m / convertToMBImg.m)
# ═════════════════════════════════════════════════════════════════════════════

def convertToMBVec(x):
    """Flatten [N, M, nb] multi-band image to a 1-D vector for PCG.

    Band-by-band row-major flattening (matches MATLAB transpose + col-major).
    """
    if x.ndim == 2:
        return x.flatten()
    N, M, nb = x.shape
    parts = [x[:, :, i].flatten() for i in range(nb)]
    return np.concatenate(parts)


def convertToMBImg(x, Ng, Mg, nb):
    """Reshape a flat vector back to [Ng, Mg, nb] multi-band image.

    Inverse of convertToMBVec.
    """
    pix = Ng * Mg
    y = np.zeros((Ng, Mg, nb))
    for i in range(nb):
        y[:, :, i] = x[i * pix:(i + 1) * pix].reshape(Ng, Mg)
    return y


# ═════════════════════════════════════════════════════════════════════════════
# Weight / Prior model functions
# ═════════════════════════════════════════════════════════════════════════════

def compute_Wpvb(u, p, epsW):
    """Weights for the TV prior IRLS majorisation.

    W = 2 / (p * (epsW + u^((p-1)/p)))

    Used in TVME_Sens:  u = Dh(y)^2 + Dv(y)^2,  p = 2.
    """
    exponent = (p - 1.0) / p
    return 2.0 / (p * (epsW + u ** exponent))


def weight_log(u):
    """Weight function kappa(u) for the Super-Gaussian *log* prior.

    kappa(u) = 1 / (eps + (|u| + eps) * |u|)

    See Table 1 in Pérez-Bueno 2020.
    """
    val = _EPS + (np.abs(u) + _EPS) * np.abs(u)
    return 1.0 / val


def weight_lp(u, p):
    """Weight function kappa(u) for the Super-Gaussian *lp* prior.

    kappa(u) = 1 / (eps + |u|^(2-p))

    See Table 1 in Pérez-Bueno 2020.
    """
    val = _EPS + np.abs(u) ** (2 - p)
    return 1.0 / val


def getfilters(filtersetname):
    """Return a list of first-order difference kernels for the SG prior.

    Parameters
    ----------
    filtersetname : str
        'none' — identity (1 filter)
        'fohv' — horizontal + vertical differences (2 filters)
        'fo'   — H + V + 2 diagonals (4 filters)

    Returns
    -------
    list of 2-D ndarrays
    """
    sq2 = np.sqrt(2.0)
    if filtersetname == 'fohv':
        return [
            np.array([[0.0, 1.0, -1.0]]) / sq2,
            np.array([[0.0], [1.0], [-1.0]]) / sq2,
        ]
    elif filtersetname == 'fo':
        return [
            np.array([[0.0, 1.0, -1.0]]) / sq2,
            np.array([[0.0], [1.0], [-1.0]]) / sq2,
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float) / sq2,
            np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype=float) / sq2,
        ]
    else:   # 'none' or unknown
        return [np.array([[1.0]])]


def getkappa(sg_prior_name, parameter=None):
    """Return (kappa_f, rho_f, alpha_f) triplet for a Super-Gaussian prior.

    Parameters
    ----------
    sg_prior_name : str — 'log' or 'lp'
    parameter     : float or None — required for 'lp' (the exponent p)

    Returns
    -------
    [kappa_f, rho_f, alpha_f] — three callables
    """
    if sg_prior_name == 'log':
        kappa_f = weight_log
        rho_f = lambda nu: np.log(np.abs(nu) + _EPS)
        alpha_f = lambda val: 1.0 + 1.0 / val
    elif sg_prior_name == 'lp':
        if parameter is None:
            raise ValueError("'lp' prior requires a parameter p")
        p = parameter
        kappa_f = lambda u, _p=p: weight_lp(u, _p)
        rho_f = lambda nu, _p=p: np.abs(nu) ** _p
        alpha_f = lambda val, _p=p: 1.0 / (_EPS + _p * val)
    else:
        raise ValueError(f"Unknown SG prior name: {sg_prior_name!r}")

    return [kappa_f, rho_f, alpha_f]


# ═════════════════════════════════════════════════════════════════════════════
# Observation model helpers
# ═════════════════════════════════════════════════════════════════════════════

def calclam(hires, pan):
    """Estimate lambda coefficients: pan ≈ sum_b lambda_b * hires_b.

    Solves the least-squares system  M lambda = r  where:
        M_ij = sum(hires_i .* hires_j)
        r_i  = sum(hires_i .* pan)

    For a single-band image returns [1.0].

    Parameters
    ----------
    hires : (H, W, nbands) or (H, W) — MS image (same resolution as pan)
    pan   : (H, W) — panchromatic image

    Returns
    -------
    lamb : (nbands,) array — non-negative, normalized to sum=1
    """
    if hires.ndim == 2:
        return np.array([1.0])

    nbands = hires.shape[2]
    if nbands == 1:
        return np.array([1.0])

    M = np.zeros((nbands, nbands))
    r = np.zeros(nbands)

    for ib in range(nbands):
        for jb in range(nbands):
            M[ib, jb] = np.sum(hires[:, :, ib] * hires[:, :, jb])
        r[ib] = np.sum(hires[:, :, ib] * pan)

    lamb = np.linalg.solve(M, r)
    lamb[lamb <= _EPS] = 0.0
    pos = lamb > _EPS
    if np.any(pos):
        lamb[pos] /= np.sum(lamb[pos])
    return lamb


def image_normalize(Y, x):
    """Per-band mean normalisation of the MS and PAN images.

    Parameters
    ----------
    Y : (lr_h, lr_w, nbands) or (lr_h, lr_w) — LR multi-spectral image
    x : (hr_h, hr_w) — HR panchromatic image

    Returns
    -------
    Y_norm, x_norm, facY, facx
    """
    Y = np.asarray(Y, dtype=np.float64).copy()
    x = np.asarray(x, dtype=np.float64).copy()

    if Y.ndim == 2:
        Y = Y[:, :, np.newaxis]

    nbands = Y.shape[2]
    lr_h, lr_w = Y.shape[:2]
    hr_h, hr_w = x.shape[:2]

    facY = np.zeros(nbands)
    for i in range(nbands):
        facY[i] = np.sum(Y[:, :, i]) / (lr_h * lr_w)
        if facY[i] > _EPS:
            Y[:, :, i] /= facY[i]

    facx = np.sum(x) / (hr_h * hr_w)
    if facx > _EPS:
        x /= facx

    return Y, x, facY, facx


def image_denormalize(Y, facY):
    """Reverse per-band mean normalisation (back to original pixel range).

    Parameters
    ----------
    Y    : (h, w, nbands) or (h, w) — normalised image
    facY : (nbands,) array — normalisation factors from image_normalize

    Returns
    -------
    Y_out : image in original range
    """
    Y = np.asarray(Y, dtype=np.float64).copy()
    Y[Y < _EPS] = 0.0

    if Y.ndim == 2:
        Y *= facY[0]
    else:
        for i in range(Y.shape[2]):
            Y[:, :, i] *= facY[i]
    return Y


def get_psf(ratio, sensor='none'):
    """Generate PSF kernel(s) for the degradation model.

    Parameters
    ----------
    ratio  : int — decimation ratio (HR_size / LR_size)
    sensor : str — sensor name: 'none', 'QB', 'IKONOS', 'GeoEye1', 'WV2'

    Returns
    -------
    For sensor='none' : single (2*ratio-1, 2*ratio-1) box-filter array.
    For known sensors : list of per-band Gaussian PSF arrays.
    """
    known_sensors = {
        'QB':      [0.34, 0.32, 0.30, 0.22],
        'IKONOS':  [0.26, 0.28, 0.29, 0.28],
        'GeoEye1': [0.23, 0.23, 0.23, 0.23],
        'WV2':     [0.35] * 7 + [0.27],
    }

    if sensor in known_sensors:
        GNyq = known_sensors[sensor]
        N = 41
        fcut = 1.0 / ratio
        psfs = []
        for g in GNyq:
            alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(g)))
            H = _fspecial_gaussian(N, alpha)
            H /= H.sum()
            psfs.append(H)
        return psfs

    # Default: box (averaging) filter — no padding needed;
    # cent_nucleus2blk_fft2 handles embedding into the block representation.
    psf = np.ones((ratio, ratio), dtype=np.float64) / (ratio * ratio)
    return psf


def _fspecial_gaussian(size, sigma):
    """Gaussian filter kernel (equivalent to MATLAB fspecial('gaussian')).

    Parameters
    ----------
    size  : int — kernel width/height (square)
    sigma : float — standard deviation

    Returns
    -------
    h : (size, size) normalised Gaussian kernel
    """
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    h = np.outer(g, g)
    h /= h.sum()
    return h
