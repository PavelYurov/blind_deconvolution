import numpy as np
from scipy.special import erfcx, erfc, gammaln
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import zoom
from numpy.fft import fft2, ifft2

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:

    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape[:2]
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return fft2(padded)

def delta_kernel(s: int) -> np.ndarray:

    if s % 2 == 0:
        s = s + 1
    out = np.zeros((s, s), dtype=np.float64)
    c = s // 2
    out[c, c] = 1.0
    return out

def rgb2gray_rob(rgb: np.ndarray, saturation_level: float = 250.0) -> np.ndarray:

    r = np.asarray(rgb, dtype=np.float64)
    sat_mask = (
        (r[:, :, 0] > saturation_level)
        | (r[:, :, 1] > saturation_level)
        | (r[:, :, 2] > saturation_level)
    )

    T_row = np.array([0.29893602, 0.58704307, 0.11402090])
    gray = r[:, :, 0] * T_row[0] + r[:, :, 1] * T_row[1] + r[:, :, 2] * T_row[2]
    gray = np.clip(gray, 0.0, 255.0)

    gray[sat_mask] = 255.0
    return gray

def invDel2(isize: int) -> np.ndarray:

    K = np.zeros((isize, isize), dtype=np.float64)
    c = isize // 2

    K[c - 1, c - 1] = -4.0
    K[c, c - 1] = 1.0
    K[c - 1, c] = 1.0
    K[c - 2, c - 1] = 1.0
    K[c - 1, c - 2] = 1.0

    Khat = fft2(K)

    zero_mask = (Khat == 0)
    Khat_safe = np.where(zero_mask, 1.0, Khat)
    invKhat = np.where(zero_mask, 0.0, 1.0 / Khat_safe)

    invK = np.real(ifft2(invKhat))
    invK = -invK

    shift_kernel = np.zeros((3, 3), dtype=np.float64)
    shift_kernel[0, 0] = 1.0
    invK = convolve2d(invK, shift_kernel, mode='same', boundary='fill')

    return invK

def reconsEdge3(dx: np.ndarray, dy: np.ndarray,
                invKhat: np.ndarray = None):

    sx, sy = dx.shape
    mxsize = max(sx, sy)

    if invKhat is None:
        invK = invDel2(2 * mxsize)
        invKhat = fft2(invK)

    imX = convolve2d(dx, np.array([[-1, 1, 0]], dtype=np.float64),
                     mode='same', boundary='fill')

    imY = convolve2d(dy, np.array([[-1], [1], [0]], dtype=np.float64),
                     mode='same', boundary='fill')

    imS = imX + imY
    imShat = fft2(imS, s=(2 * mxsize, 2 * mxsize))
    im = np.real(ifft2(imShat * invKhat))

    im = im[mxsize:mxsize + sx, mxsize:mxsize + sy]
    return im, invKhat

def normMDpdf(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:

    mu = mu.ravel()
    nDims = x.shape[0]
    nPoints = x.shape[1]

    i_sig = np.linalg.inv(sig)
    det_sig = np.linalg.det(sig)
    d = ((2 * np.pi) ** (-nDims / 2.0)) / np.sqrt(det_sig)

    tt = x - mu[:, np.newaxis]
    ttt = i_sig @ tt
    e = np.sum(tt * ttt, axis=0)

    y = d * np.exp(-0.5 * e)
    return y

def clip_image(im: np.ndarray, minval: float, maxval: float) -> np.ndarray:

    return np.clip(im, minval, maxval)

def _histeq_mapping(gray_in: np.ndarray, target_hist: np.ndarray):

    target_cdf = np.cumsum(target_hist).astype(np.float64)
    if target_cdf[-1] > 0:
        target_cdf /= target_cdf[-1]

    gray_uint8 = np.clip(np.round(gray_in * 255.0), 0, 255).astype(np.int32)
    input_hist = np.bincount(gray_uint8.ravel(), minlength=256).astype(np.float64)
    input_cdf = np.cumsum(input_hist)
    if input_cdf[-1] > 0:
        input_cdf /= input_cdf[-1]

    T = np.zeros(256, dtype=np.float64)
    for i in range(256):
        j = np.searchsorted(target_cdf, input_cdf[i])
        j = min(j, 255)
        T[i] = j / 256.0

    J = T[gray_uint8]
    return J, T

def histmatch(in_img: np.ndarray, reference: np.ndarray) -> np.ndarray:

    in_f = np.asarray(in_img, dtype=np.float64)
    ref = np.asarray(reference)

    if in_f.ndim == 3 and in_f.shape[2] != 1:
        gray_in = 0.2989 * in_f[:, :, 0] + 0.5870 * in_f[:, :, 1] + 0.1140 * in_f[:, :, 2]
    else:
        gray_in = in_f if in_f.ndim == 2 else in_f[:, :, 0]

    if ref.ndim == 3 and ref.shape[2] != 1:
        gray_ref = (0.2989 * ref[:, :, 0] + 0.5870 * ref[:, :, 1] + 0.1140 * ref[:, :, 2]).astype(np.float64)
    else:
        gray_ref = ref.astype(np.float64) if ref.ndim == 2 else ref[:, :, 0].astype(np.float64)

    hist_reference = np.bincount(
        np.clip(np.round(gray_ref).astype(np.int32).ravel(), 0, 255),
        minlength=256
    ).astype(np.float64)

    _, T = _histeq_mapping(gray_in, hist_reference)

    nch = in_f.shape[2] if in_f.ndim == 3 else 1
    if in_f.ndim == 2:
        in_f = in_f[:, :, np.newaxis]

    out = np.zeros_like(in_f)
    for a in range(nch):
        q = in_f[:, :, a]

        x_knots = np.arange(256) / 256.0
        qm = np.interp(q.ravel(), x_knots, T)
        out[:, :, a] = (256.0 * qm.reshape(q.shape))

    out = np.clip(out, 0, 255).astype(np.uint8)
    if nch == 1:
        out = out[:, :, 0]
    return out

def fix_image(in_img: np.ndarray, reference: np.ndarray) -> np.ndarray:

    SPACING = 0.05

    ref_im = reference.astype(np.float64)
    ref_max = ref_im.max()
    if ref_max > 0:
        ref_im = ref_im / ref_max

    x_bins = np.arange(0, 1.0 + SPACING, SPACING)
    hist_ref, _ = np.histogram(ref_im.ravel(), bins=np.append(x_bins, np.inf))
    hist_ref = hist_ref[:len(x_bins)]

    m = in_img.min()
    in_shift = in_img - m
    in_max = in_shift.max()
    if in_max > 0:
        in_norm = in_shift / in_max
    else:
        in_norm = in_shift

    out, _ = _histeq_mapping(in_norm, hist_ref)
    return out

def automatic_patch_selector(im: np.ndarray, patch_size: int,
                             weight: float,
                             sat_mask: np.ndarray):

    SMOOTH_SIGMA = 3

    II, JJ = im.shape

    yy, xx = np.mgrid[0:II, 0:JJ]
    xx = xx - round(JJ / 2)
    yy = yy - round(II / 2)
    centre_weight_mask = np.exp(-weight / (JJ ** 2) * (xx ** 2 + yy ** 2))

    II2 = II * 2
    JJ2 = JJ * 2

    dk = delta_kernel(patch_size)
    centre_weight_mask = np.real(
        ifft2(fft2(centre_weight_mask, s=(II2, JJ2))
              * fft2(dk, s=(II2, JJ2)))
    )

    pmask = np.ones((patch_size, patch_size), dtype=np.float64) / (patch_size ** 2)

    ei2 = np.real(ifft2(fft2(im ** 2, s=(II2, JJ2)) * fft2(pmask, s=(II2, JJ2))))
    mu2 = np.real(ifft2(fft2(im, s=(II2, JJ2)) * fft2(pmask, s=(II2, JJ2)))) ** 2
    w = ei2 - mu2

    q = np.real(ifft2(fft2(sat_mask.astype(np.float64), s=(II2, JJ2))
                      * fft2(pmask, s=(II2, JJ2))))

    mean_im = im.mean()
    combined = centre_weight_mask * w / (q * mean_im ** 2 + 1.0)

    from scipy.ndimage import gaussian_filter
    combined_smooth = np.real(
        ifft2(
            fft2(combined, s=(II2, JJ2))
            * fft2(
                _fspecial_gaussian(8, SMOOTH_SIGMA, (II2, JJ2)),
            )
        )
    )

    combined_crop = combined_smooth[patch_size - 1:II, patch_size - 1:JJ]

    mm = np.argmax(combined_crop)
    sy, sx = np.unravel_index(mm, combined_crop.shape)

    patch_location = np.array([sx, sy])

    py = sy
    px = sx
    out_im = im[py:py + patch_size, px:px + patch_size]

    return out_im, patch_location

def _fspecial_gaussian(hsize: int, sigma: float,
                       fft_shape: tuple = None) -> np.ndarray:

    half = hsize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]

    if y.shape[0] > hsize:
        y = y[:hsize, :hsize]
        x = x[:hsize, :hsize]
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()

    if fft_shape is not None:
        padded = np.zeros(fft_shape, dtype=np.float64)
        padded[:g.shape[0], :g.shape[1]] = g
        return padded

    return g

def GaussianMixtures1D(x: np.ndarray, nComponents: int):

    MAX_ITERATIONS = 100
    LIKELIHOOD_CHANGE_THRESHOLD = 1e-5

    x = x.ravel().astype(np.float64)
    nPoints = len(x)

    mu = np.zeros((1, nComponents), dtype=np.float64)
    sigma = np.zeros((1, 1, nComponents), dtype=np.float64)
    weight = np.ones(nComponents, dtype=np.float64) / nComponents

    for a in range(nComponents):
        sigma[0, 0, a] = 1e6 - np.random.rand() * 1e6
        if sigma[0, 0, a] <= 0:
            sigma[0, 0, a] = 1.0

    sigma[0, 0, 0] = 1e6

    resp = np.zeros((nComponents, nPoints), dtype=np.float64)
    likelihoods = np.zeros((nComponents, nPoints), dtype=np.float64)
    log_likelihood_list = []
    delta_lh = np.inf

    for iteration in range(MAX_ITERATIONS):

        for c in range(nComponents):
            s = sigma[0, 0, c]
            if s <= 0:
                s = 1e-10
            normaliser = 1.0 / np.sqrt(2 * np.pi * s)
            offset = x - mu[0, c]
            exponent = offset ** 2 / s
            likelihoods[c, :] = weight[c] * normaliser * np.exp(-0.5 * exponent)

        total = np.sum(likelihoods, axis=0)
        total = np.maximum(total, 1e-300)
        ll = np.mean(np.log(total))
        log_likelihood_list.append(ll)

        if iteration > 0:
            delta_lh = log_likelihood_list[-1] - log_likelihood_list[-2]

        for c in range(nComponents):
            resp[c, :] = likelihoods[c, :] / total

        for c in range(nComponents):
            total_resp_c = np.sum(resp[c, :])
            if total_resp_c < 1e-10:
                total_resp_c = 1e-10

            weight[c] = total_resp_c / nPoints
            mu[0, c] = 0.0

            offset = x - mu[0, c]
            u = np.sqrt(resp[c, :])
            new_sigma = np.sum((u * offset) ** 2) / total_resp_c
            sigma[0, 0, c] = new_sigma + 1e-5

        if iteration < 9:
            sigma[0, 0, 0] = 1e6

        if delta_lh < LIKELIHOOD_CHANGE_THRESHOLD and iteration > 0:
            break

    return mu, sigma, weight, log_likelihood_list

def edgetaper(im: np.ndarray, kernel: np.ndarray,
              n_tapers: int = 1) -> np.ndarray:

    kh, kw = kernel.shape
    ac = convolve2d(kernel, kernel[::-1, ::-1], mode='full')
    ac = ac / ac.max()

    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    taper_y = ac[:, cx]
    taper_x = ac[cy, :]

    result = im.astype(np.float64).copy()
    for _ in range(n_tapers):
        if result.ndim == 2:
            blurred = _fft_convolve_same(result, kernel)
        else:
            blurred = np.stack(
                [_fft_convolve_same(result[:, :, c], kernel)
                 for c in range(result.shape[2])],
                axis=2
            )

        H, W = result.shape[:2]
        alpha_y = np.ones(H, dtype=np.float64)
        alpha_x = np.ones(W, dtype=np.float64)

        half_ky = len(taper_y) // 2
        half_kx = len(taper_x) // 2
        for i in range(min(half_ky, H)):
            v = taper_y[half_ky - i]
            alpha_y[i] = min(alpha_y[i], v)
            alpha_y[H - 1 - i] = min(alpha_y[H - 1 - i], v)
        for j in range(min(half_kx, W)):
            v = taper_x[half_kx - j]
            alpha_x[j] = min(alpha_x[j], v)
            alpha_x[W - 1 - j] = min(alpha_x[W - 1 - j], v)

        alpha = alpha_y[:, np.newaxis] * alpha_x[np.newaxis, :]
        if result.ndim == 3:
            alpha = alpha[:, :, np.newaxis]

        result = alpha * result + (1.0 - alpha) * blurred

    return result

def _fft_convolve_same(im: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    out = fftconvolve(im, kernel, mode='same')
    return out

def train_ensemble_get(c: int, dimensions: np.ndarray, x: np.ndarray) -> np.ndarray:

    c_idx = c - 1
    if c_idx > 0:
        start = int(np.sum(dimensions[:c_idx, 0] * dimensions[:c_idx, 1]))
    else:
        start = 0
    n_rows = int(dimensions[c_idx, 0])
    n_cols = int(dimensions[c_idx, 1])
    return x[start:start + n_rows * n_cols].reshape(n_rows, n_cols)

def train_ensemble_put(c: int, dimensions: np.ndarray,
                       x: np.ndarray, cx: np.ndarray) -> np.ndarray:

    c_idx = c - 1
    if c_idx > 0:
        start = int(np.sum(dimensions[:c_idx, 0] * dimensions[:c_idx, 1]))
    else:
        start = 0
    n_rows = int(dimensions[c_idx, 0])
    n_cols = int(dimensions[c_idx, 1])
    x_out = x.copy()
    x_out[start:start + n_rows * n_cols] = cx.reshape(n_rows * n_cols)
    return x_out

def train_ensemble_get_lambda(c: int, dimensions: np.ndarray,
                              log_lambda_x: np.ndarray) -> np.ndarray:

    c_idx = c - 1
    if c_idx > 0:
        start = int(np.sum(np.prod(dimensions[:c_idx, :3], axis=1)))
    else:
        start = 0
    n = int(np.prod(dimensions[c_idx, :3]))
    d0 = int(dimensions[c_idx, 0])
    d1 = int(dimensions[c_idx, 1])
    d2 = int(dimensions[c_idx, 2])
    return log_lambda_x[start:start + n].reshape(d0, d1, d2)

def train_ensemble_put_lambda(c: int, dimensions: np.ndarray,
                              log_lambda_x: np.ndarray,
                              c_log_lambda_x: np.ndarray) -> np.ndarray:

    c_idx = c - 1
    if c_idx > 0:
        start = int(np.sum(np.prod(dimensions[:c_idx, :3], axis=1)))
    else:
        start = 0
    n = int(np.prod(dimensions[c_idx, :3]))
    out = log_lambda_x.copy()
    out[start:start + n] = c_log_lambda_x.ravel()
    return out

def train_ensemble_rectified5(x1: np.ndarray, x2: np.ndarray,
                              dist_type: int):

    x2 = np.maximum(np.asarray(x2, dtype=np.float64), 1e-300)
    x1 = np.asarray(x1, dtype=np.float64)

    if dist_type == 0:

        mx = x1 / x2
        mx2 = x1 ** 2 / x2 ** 2 + 1.0 / x2
        Hx = -0.5 + 0.5 * np.log(x2)

    elif dist_type == 1 or dist_type == 2:

        sqrt_2x2 = np.sqrt(2.0 * x2)
        t = -x1 / sqrt_2x2
        erf_table = erfcx(t)

        mask_low = (t <= 25)

        safe_erf = np.where(erf_table == 0, 1e-300, erf_table)

        safe_x1 = np.where(np.abs(x1) < 1e-100, np.copysign(1e-100, x1 + 1e-300), x1)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):

            mx_low = x1 / x2 + np.sqrt(2.0 / (np.pi * x2)) / safe_erf
            mx2_low = (x1 ** 2 / x2 ** 2 + 1.0 / x2
                        + 2.0 * x1 / x2 / np.sqrt(2.0 * np.pi * x2) / safe_erf)

            mx_high = (-1.0 / safe_x1 + 2.0 * x2 / safe_x1 ** 3
                        - 10.0 * x2 ** 2 / safe_x1 ** 5)
            mx2_high = (2.0 / safe_x1 ** 2 - 10.0 * x2 / safe_x1 ** 4
                         + 74.0 * x2 ** 2 / safe_x1 ** 6)

        mx = np.where(mask_low, mx_low, mx_high)
        mx2 = np.where(mask_low, mx2_low, mx2_high)

        erfc_clamped = np.maximum(erfc(np.minimum(t, 25.0)), 1e-300)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            Hx_low = (-np.log(erfc_clamped)
                       + 0.5 * np.log(2.0 * x2 / np.pi) - 0.5
                       + x1 / np.sqrt(2.0 * np.pi * x2) / safe_erf)
            Hx_high = (np.log(np.maximum(np.abs(safe_x1), 1e-300)) - 1.0
                        + 2.0 * x2 / safe_x1 ** 2
                        - 15.0 * x2 ** 2 / safe_x1 ** 4 / 2.0
                        + 148.0 * x2 ** 3 / safe_x1 ** 6 / 3.0)

        Hx = np.where(t < 25, Hx_low, Hx_high)

        if dist_type == 2:
            Hx = Hx + 0.5 * np.log(np.pi / 2.0)

    elif dist_type == 3:

        mx = np.tanh(x1)
        mx2 = np.ones_like(x1)
        Hx = x1 * mx - np.abs(x1) - np.log(1.0 + np.exp(-2.0 * np.abs(x1))) + np.log(2.0)

    elif dist_type == 4:

        sqrt_2x2 = np.sqrt(2.0 * x2)
        t = -x1 / sqrt_2x2
        erf_table = erfcx(t)

        mask_low = (t <= 25)
        safe_erf = np.where(erf_table == 0, 1e-300, erf_table)
        safe_x1 = np.where(np.abs(x1) < 1e-100, np.copysign(1e-100, x1 + 1e-300), x1)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            mx_low = x1 / x2 + np.sqrt(2.0 / (np.pi * x2)) / safe_erf
            mx2_low = (x1 ** 2 / x2 ** 2 + 1.0 / x2
                        + 2.0 * x1 / x2 / np.sqrt(2.0 * np.pi * x2) / safe_erf)
            mx_high = (-1.0 / safe_x1 + 2.0 * x2 / safe_x1 ** 3
                        - 10.0 * x2 ** 2 / safe_x1 ** 5)
            mx2_high = (2.0 / safe_x1 ** 2 - 10.0 * x2 / safe_x1 ** 4
                         + 74.0 * x2 ** 2 / safe_x1 ** 6)

        mx = np.where(mask_low, mx_low, mx_high)
        mx2 = np.where(mask_low, mx2_low, mx2_high)

        erfc_clamped = np.maximum(erfc(np.minimum(t, 25.0)), 1e-300)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            Hx_low = (-np.log(erfc_clamped)
                       + 0.5 * np.log(2.0 * x2 / np.pi) - 0.5
                       + x1 / np.sqrt(2.0 * np.pi * x2) / safe_erf)
            Hx_high = (np.log(np.maximum(np.abs(safe_x1), 1e-300)) - 1.0
                        + 2.0 * x2 / safe_x1 ** 2
                        - 15.0 * x2 ** 2 / safe_x1 ** 4 / 2.0
                        + 148.0 * x2 ** 3 / safe_x1 ** 6 / 3.0)

        Hx = np.where(t < 25, Hx_low, Hx_high)
        Hx = Hx + 0.5 * np.log(np.pi / 2.0)

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    return Hx, mx, mx2

def move_level(mx: np.ndarray, me: np.ndarray,
               K: int, L: int, M: int, N: int,
               mode: str = 'matlab_bilinear',
               resize_step: float = np.sqrt(2),
               center: bool = False) -> tuple:

    if center:
        me = me / me.sum()
        rows = np.arange(me.shape[0])
        cols = np.arange(me.shape[1])
        mu_y = np.sum(rows * me.sum(axis=1))
        mu_x = np.sum(cols * me.sum(axis=0))

        offset_y = round(me.shape[0] // 2 - mu_y)
        offset_x = round(me.shape[1] // 2 - mu_x)

        shift_kernel = np.zeros((abs(offset_y) * 2 + 1,
                                 abs(offset_x) * 2 + 1), dtype=np.float64)
        shift_kernel[abs(offset_y) + offset_y,
                     abs(offset_x) + offset_x] = 1.0

        me = convolve2d(me, shift_kernel, mode='same', boundary='fill')

        if mx.ndim == 3:
            for c in range(mx.shape[2]):
                mx[:, :, c] = convolve2d(
                    mx[:, :, c], shift_kernel[::-1, ::-1],
                    mode='same', boundary='fill'
                )
        else:
            mx = convolve2d(mx, shift_kernel[::-1, ::-1],
                            mode='same', boundary='fill')

    if mx.ndim == 2:
        zoom_y = M / mx.shape[0]
        zoom_x = N / mx.shape[1]
        mx_new = zoom(mx, (zoom_y, zoom_x), order=1)
    else:
        zoom_y = M / mx.shape[0]
        zoom_x = N / mx.shape[1]
        mx_new = np.stack(
            [zoom(mx[:, :, c], (zoom_y, zoom_x), order=1)
             for c in range(mx.shape[2])],
            axis=2
        )

    zoom_ky = K / me.shape[0]
    zoom_kx = L / me.shape[1]
    me_new = zoom(me, (zoom_ky, zoom_kx), order=1)

    if mx_new.ndim == 2:
        mx_new = mx_new[:M, :N]
    else:
        mx_new = mx_new[:M, :N, :]
    me_new = me_new[:K, :L]

    me_sum = me_new.sum()
    if me_sum > 0:
        me_new = me_new / me_sum

    return mx_new, me_new

def imresize(im: np.ndarray, scale_or_shape, method: str = 'bilinear') -> np.ndarray:

    import cv2

    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
    }

    if isinstance(scale_or_shape, (int, float)):
        scale = float(scale_or_shape)
        new_h = max(1, int(round(im.shape[0] * scale)))
        new_w = max(1, int(round(im.shape[1] * scale)))
    else:
        new_h, new_w = int(scale_or_shape[0]), int(scale_or_shape[1])

    interp = interp_map.get(method, cv2.INTER_LINEAR)

    is_downscale = (new_h < im.shape[0]) or (new_w < im.shape[1])
    if is_downscale and method != 'nearest':
        interp = cv2.INTER_AREA

    im_f64 = im.astype(np.float64)

    return cv2.resize(im_f64, (new_w, new_h), interpolation=interp)

_DEFAULT_PRIORS_STREET_4 = [
    {'pi': np.array([[0.3424340310728903, 0.40767651333052146, 0.07812420385914433, 0.1717652517374289]]),
     'gamma': np.array([[0.29802001130266303, 3.0573335190267334, 0.0014491231427730406, 0.01704592730142977]])},
    {'pi': np.array([[0.3190705041263619, 0.09596426181604158, 0.3879939885023601, 0.19697124555521656]]),
     'gamma': np.array([[0.6353124313785059, 0.0028742740247449322, 5.9569282007663, 0.03574942831297822]])},
    {'pi': np.array([[0.09976787553775796, 0.3548257534999502, 0.21415238959909377, 0.33125398136318857]]),
     'gamma': np.array([[0.004147895133134799, 9.91287418753391, 0.052356814834793375, 1.0760367256192498]])},
    {'pi': np.array([[0.0806457697881914, 0.3531087529161156, 0.33509045272367893, 0.23115502457201323]]),
     'gamma': np.array([[0.0050454695734232, 16.54712911833657, 1.5114534596710867, 0.0603056841619064]])},
    {'pi': np.array([[0.3837669289986711, 0.2549862266831338, 0.05017782885585161, 0.31106901546234117]]),
     'gamma': np.array([[2.1968983384539866, 0.0629588203406576, 0.004338338495067549, 31.45072207040538]])},
    {'pi': np.array([[0.2361579503716776, 0.02410007746499708, 0.3889560888676661, 0.3507858832956606]]),
     'gamma': np.array([[0.0494504494719423, 0.002400522386224243, 27.772678182031825, 1.3818254410186852]])},
    {'pi': np.array([[0.3750836558156682, 0.2278563970606603, 0.3907337897732923, 0.006326157350377128]]),
     'gamma': np.array([[30.87423860368161, 0.03422103832974795, 1.3676396026917672, 0.0006615342462320538]])},
    {'pi': np.array([[0.36546849898726225, 0.4444363146286955, 0.003820813210408265, 0.18627437317363424]]),
     'gamma': np.array([[0.665507621417974, 21.146669814425323, 0.00031931372931303, 0.023160504210950776]])},
]

def get_default_priors(prior_name: str = 'street',
                       num_components: int = 4) -> list:

    if num_components != 4:
        raise ValueError(
            f"Built-in priors only available for 4 components, got {num_components}")
    if prior_name == 'street':
        return [dict(p) for p in _DEFAULT_PRIORS_STREET_4]
    raise ValueError(f"Unknown prior name: {prior_name!r}. Available: 'street'")

def load_matlab_priors(mat_path: str) -> list:

    import scipy.io as sio
    mat = sio.loadmat(mat_path)
    if 'priors' not in mat:
        raise KeyError(f"No 'priors' variable found in {mat_path}")
    raw = mat['priors']
    result = []
    for b in range(raw.shape[1]):
        result.append({
            'pi': np.asarray(raw[0, b]['pi'], dtype=np.float64),
            'gamma': np.asarray(raw[0, b]['gamma'], dtype=np.float64),
        })
    return result

def estimate_priors_from_images(images: list, num_components: int,
                                num_scales: int,
                                gradient_type: str = 'haar') -> list:

    SCALE_STEP = np.sqrt(2)
    MAX_IM_SIZE = 700
    STEP_SIZE = 1

    x_bins = np.arange(-200, 201, STEP_SIZE).astype(np.float64)

    priors = []
    for b in range(num_scales):
        scale = SCALE_STEP ** (-b)
        b_all = []

        for im in images:
            if im.ndim == 3:
                im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
            im = im.astype(np.float64)

            imy, imx = im.shape
            scale_factor = MAX_IM_SIZE / max(imx, imy)
            im = imresize(im, scale_factor, 'bilinear')

            if gradient_type == 'haar':
                b_x = convolve2d(im, np.array([[1, -1]], dtype=np.float64),
                                 mode='valid')
                b_y = convolve2d(im, np.array([[1], [-1]], dtype=np.float64),
                                 mode='valid')
                if scale != 1.0:
                    b_x = imresize(b_x, scale, 'bilinear')
                    b_y = imresize(b_y, scale, 'bilinear')
            else:
                raise NotImplementedError("Steerable pyramid not implemented")

            b_all.extend(b_x.ravel().tolist())
            b_all.extend(b_y.ravel().tolist())

        b_all = np.array(b_all, dtype=np.float64)
        mu, sigma, weight, ll = GaussianMixtures1D(b_all, num_components)

        prior_entry = {
            'pi': weight.reshape(1, -1).copy(),
            'gamma': (1.0 / sigma[0, 0, :]).reshape(1, -1).copy(),
        }
        priors.append(prior_entry)

    return priors

def create_greenspan_settings(**kwargs) -> dict:

    bfilt = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    lo_filt = np.outer(bfilt, bfilt)

    S = {
        'lo_filt': lo_filt,
        'c': 0.4,
        's': 5,
        'bp': 1,
        'factor': 1,
    }
    S.update(kwargs)
    return S

def greenspan(im: np.ndarray, S: dict):

    z = 2 ** S['factor']
    lo_filt = S['lo_filt']

    im_smooth = convolve2d(im, lo_filt, mode='same', boundary='symm')
    L1 = im - im_smooth

    target_shape = (z * im.shape[0], z * im.shape[1])
    L0 = imresize(L1, target_shape, 'bilinear') * (z ** 2)

    maxL0 = np.abs(L0).max() if np.abs(L0).max() > 0 else 1.0
    L0 = S['s'] * clip_image(L0, -(1 - S['c']) * maxL0, (1 - S['c']) * maxL0)

    if S['bp']:
        L0_smooth = convolve2d(L0, lo_filt, mode='same', boundary='symm')
        L0 = L0 - L0_smooth

    en = imresize(im, target_shape, 'bilinear') * (z ** 2) + L0
    return en, L0
