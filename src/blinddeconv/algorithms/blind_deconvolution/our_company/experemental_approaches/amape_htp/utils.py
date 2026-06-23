import numpy as np
import cv2
import math

from numpy.fft import fft2, ifft2

def copy_mat_2_mat_zeros(a, b_shape, rows, cols):

    b = np.zeros(b_shape, dtype=np.float64)
    b[:rows - 1, :cols - 1] = a[:rows - 1, :cols - 1]
    return b

def copy_mat_2_cmat_zeros(a, b_shape, rows, cols):

    b = np.zeros(b_shape, dtype=np.complex128)
    b[:rows - 1, :cols - 1] = a[:rows - 1, :cols - 1].astype(np.complex128)
    return b

def copy_mat_2_cmat(a, b_shape, rows, cols):

    b = np.zeros(b_shape, dtype=np.complex128)
    b[:rows - 1, :cols - 1] = a[:rows - 1, :cols - 1].astype(np.complex128)
    return b

def copy_cmat_2_cmat_zeros(a, b_shape, rows, cols):

    b = np.zeros(b_shape, dtype=np.complex128)
    b[:rows - 1, :cols - 1] = a[:rows - 1, :cols - 1]
    return b

def power(x, y):

    assert x != 0.0
    return math.exp(y * math.log(x))

def sfix(d):

    if d > 0:
        return int(math.floor(d))
    return int(math.ceil(d))

def sumabs2(m):

    return np.sum(np.real(np.conj(m) * m))

def normalizeImage(m):

    min_val = m.min()
    max_val = m.max()
    scale = max_val - min_val
    if scale < 0.0:
        scale *= -1.0
    assert scale != 0.0
    m_out = (m - min_val) / scale
    return m_out, min_val, max_val

def matNormalize(h):

    h_out = h.copy()
    s = h_out.sum()
    if s > 0.0:
        h_out /= s
    return h_out

def mat2gray(m):

    m_out, _, _ = normalizeImage(m)
    return m_out

def _get_interpolation_type(rsize, csize):

    if 8 <= rsize and 8 <= csize:
        return cv2.INTER_LANCZOS4
    elif 4 <= rsize and 4 <= csize:
        return cv2.INTER_CUBIC
    else:
        return cv2.INTER_LINEAR

def imresize_to_shape(in_mat, sx, sy):

    rsize = in_mat.shape[0]
    csize = in_mat.shape[1]
    interp = _get_interpolation_type(rsize, csize)

    src = in_mat.astype(np.float64)

    dst = cv2.resize(src, (sy, sx), interpolation=interp)
    return dst

def imresize_by_factor(in_mat, factor):

    rsize = in_mat.shape[0]
    csize = in_mat.shape[1]
    interp = _get_interpolation_type(rsize, csize)

    src = in_mat.astype(np.float64)

    dst = cv2.resize(src, None, fx=factor, fy=factor, interpolation=interp)
    return dst

def getROI(m, maxROIsize_r, maxROIsize_c):

    m_r = m.shape[0]
    m_c = m.shape[1]
    margin_r = 0
    margin_c = 0

    if m_r > maxROIsize_r:
        margin_r = int(math.floor((m_r - maxROIsize_r) / 2))
        m_r = maxROIsize_r
    if m_c > maxROIsize_c:
        margin_c = int(math.floor((m_c - maxROIsize_c) / 2))
        m_c = maxROIsize_c

    end_r = margin_r + m_r + margin_r
    end_c = margin_c + m_c + margin_c

    end_r = min(end_r, m.shape[0])
    end_c = min(end_c, m.shape[1])
    return m[margin_r:end_r, margin_c:end_c].copy()

def createROI(tmp, L, MSlevels):

    if L != MSlevels:
        factor = 1.0
        for _ in range(L, MSlevels):
            factor *= 0.5
        tmp2 = imresize_by_factor(tmp, factor)
        return tmp2
    return tmp.copy()

def doublePSF(h):

    hx = h.shape[0]
    hy = h.shape[1]
    return imresize_to_shape(h, hx * 2, hy * 2)

def get_margins_mat(m, threshold):

    rows, cols = m.shape

    m_left = 1
    m_right = rows
    m_top = 1
    m_bottom = cols

    found = False
    for x in range(rows):
        for y in range(cols):
            if m[x, y] > threshold:
                m_top = x + 1
                found = True
                break
        if found:
            break

    found = False
    for x in range(rows - 1, -1, -1):
        for y in range(cols):
            if m[x, y] > threshold:
                m_bottom = x + 1
                found = True
                break
        if found:
            break

    found = False
    for y in range(cols):
        for x in range(rows):
            if m[x, y] > threshold:
                m_left = y + 1
                found = True
                break
        if found:
            break

    found = False
    for y in range(cols - 1, -1, -1):
        for x in range(rows):
            if m[x, y] > threshold:
                m_right = y + 1
                found = True
                break
        if found:
            break

    return m_left, m_right, m_top, m_bottom

def make_mask_mat(m, threshold):

    return np.where(m > threshold, 1.0, 0.0)

def bwmorph_clean(mask):

    m_x, m_y = mask.shape
    mask_out = mask.copy()

    temp = np.zeros((m_x + 2, m_y + 2), dtype=np.float64)
    temp[1:m_x + 1, 1:m_y + 1] = mask

    for x in range(1, m_x + 1):
        for y in range(1, m_y + 1):
            if (temp[x - 1, y - 1] == 0.0 and
                temp[x - 1, y] == 0.0 and
                temp[x - 1, y + 1] == 0.0 and
                temp[x, y - 1] == 0.0 and
                temp[x, y + 1] == 0.0 and
                temp[x + 1, y - 1] == 0.0 and
                temp[x + 1, y] == 0.0 and
                    temp[x + 1, y + 1] == 0.0):
                mask_out[x - 1, y - 1] = 0.0

    return mask_out

def set_Vh(vh, rows, cols):

    vh_out = np.zeros_like(vh)
    tmp = vh[:rows - 1, :cols - 1].copy()
    tmp[tmp < 0.0] = 0.0
    vh_out[:rows - 1, :cols - 1] = tmp
    return vh_out

def centerPSF(H, centering_threshold):

    H = mat2gray(H)
    mask = make_mask_mat(H, centering_threshold)
    mask = bwmorph_clean(mask)
    m_left, m_right, m_top, m_bottom = get_margins_mat(
        mask, centering_threshold
    )

    rows, cols = H.shape

    topleft_x = m_top
    topleft_y = m_left

    begin_x = max(topleft_x, 1) - 1
    end_x = min(topleft_x + rows - 1, rows) - 1
    begin_y = max(topleft_y, 1) - 1
    end_y = min(topleft_y + cols - 1, cols) - 1

    shift_x = (begin_x + end_x - rows) // 2
    shift_y = (begin_y + end_y - cols) // 2

    tmp = np.zeros((rows, cols), dtype=np.float64)

    for x in range(begin_x, end_x + 1):
        for y in range(begin_y, end_y + 1):
            if H[x, y] <= centering_threshold:
                tmp[x, y] = 0.0
            else:
                tmp[x, y] = H[x, y]

    H = tmp.copy()
    s = H.sum()
    if s != 0.0:
        H /= s

    return H

def set_vrange(m):

    return m.min(), m.max()

def uConstr(m, vrange_min, vrange_max):

    m_out = m.copy()
    re = np.real(m_out)
    mask_lo = re < vrange_min
    mask_hi = re > vrange_max
    m_out[mask_lo] = complex(vrange_min, 0.0)
    m_out[mask_hi] = complex(vrange_max, 0.0)
    return m_out

def snx(x, alpha, beta, q):

    return -x + (1.0 - q) * q * alpha / beta * power(x, q - 1.0)

def snx2(x, alpha, beta, q):

    return -x * x / 2.0 + (1.0 - q) * alpha / beta * power(x, q)

def dsnx2(x, alpha, beta, q):

    return snx(x, alpha, beta, q)

def newton(f, df, alpha, beta, q, x0, eps1, eps2, max_iter):

    x1 = x0
    x = x1
    ind = 0
    sw = 0

    while sw == 0 and ind >= 0:
        sw = 1
        ind += 1

        g = f(x1, alpha, beta, q)

        if abs(g) > eps2:
            if ind <= max_iter:
                dg = df(x1, alpha, beta, q)
                if abs(dg) > eps2:
                    x = x1 - g / dg
                    if abs(x - x1) > eps1 and abs(x - x1) > eps1 * abs(x):
                        x1 = x
                        sw = 0
                else:
                    ind = -1
            else:
                ind = -1

    return x, ind

def asetLnorm(q, alpha, beta):

    if q == 1.0:
        v_star = 0.0
        u_star = alpha / beta
    elif q == 0.0:
        v_star = math.sqrt(2.0 * alpha / beta)
        u_star = math.sqrt(2.0 * alpha / beta)
    else:
        eps1 = 2.0e-16
        eps2 = 2.0e-16
        max_iter = 100
        x0 = 0.1

        v_star, _ = newton(snx2, dsnx2, alpha, beta, q,
                           x0, eps1, eps2, max_iter)
        u_star = v_star + alpha / beta * q * power(v_star, q - 1.0)

    return v_star, u_star

def aLn(DU, normDU, v_star, u_star):

    k = u_star - v_star

    V = np.zeros_like(DU)
    mask = normDU > u_star

    safe_norm = np.where(mask, normDU, 1.0)
    V[mask] = DU[mask] * (safe_norm[mask] - k) / safe_norm[mask]

    return V

def create_gaussian_kernel(psf_x, psf_y, sigma):

    psf = np.zeros((psf_x, psf_y), dtype=np.float64)
    factor = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    hh = psf_x / 2.0
    hw = psf_y / 2.0

    for x in range(psf_x):
        tmp_x = (x - hh) / sigma
        for y in range(psf_y):
            tmp_y = (y - hw) / sigma
            psf[x, y] = math.exp(-(tmp_x * tmp_x + tmp_y * tmp_y) / 2.0) * factor

    s = psf.sum()
    if s > 0.0:
        psf /= s
    return psf

def normalized_autocorrelation(psf):

    psf_x, psf_y = psf.shape
    xpsf_x = 2 * psf_x - 1
    xpsf_y = 2 * psf_y - 1
    xpsf = np.zeros((xpsf_x, xpsf_y), dtype=np.float64)

    max_val = -np.inf
    for x in range(xpsf_x):
        for y in range(xpsf_y):
            c = 0.0
            for u in range(psf_x):
                for v in range(psf_y):
                    ix = x - u
                    iy = y - v
                    if 0 <= ix < psf_x and 0 <= iy < psf_y:
                        c += psf[u, v] * psf[psf_x - 1 - ix, psf_y - 1 - iy]
            xpsf[x, y] = c
            if c > max_val:
                max_val = c

    if max_val != 0.0:
        xpsf /= max_val

    return xpsf

def create_weights(img_x, img_y, psf):

    psf_x, psf_y = psf.shape
    mergin_x = psf_x - 1
    mergin_y = psf_y - 1
    w_x = img_x
    w_y = img_y

    xpsf = normalized_autocorrelation(psf)

    weights = np.zeros((w_x, w_y), dtype=np.float64)

    for x in range(w_x):
        for y in range(w_y):
            if x < mergin_x:
                if y < mergin_y:

                    weights[x, y] = xpsf[x, y]
                elif mergin_y <= y < (w_y - mergin_y):

                    weights[x, y] = xpsf[x, mergin_y]
                else:

                    weights[x, y] = xpsf[x, y - (w_y - 2 * mergin_y) + 1]
            elif mergin_x <= x < (w_x - mergin_x):
                if y < mergin_y:

                    weights[x, y] = xpsf[mergin_x, y]
                elif mergin_y <= y < (w_y - mergin_y):

                    weights[x, y] = 1.0
                else:

                    weights[x, y] = xpsf[mergin_x, y - (w_y - 2 * mergin_y) + 1]
            else:
                if y < mergin_y:

                    weights[x, y] = xpsf[x - (w_x - 2 * mergin_x) + 1, y]
                elif mergin_y <= y < (w_y - mergin_y):

                    weights[x, y] = xpsf[x - (w_x - 2 * mergin_x) + 1, mergin_y]
                else:

                    weights[x, y] = xpsf[
                        x - (w_x - 2 * mergin_x) + 1,
                        y - (w_y - 2 * mergin_y) + 1
                    ]

    return weights

def create_blurred(img, psf):

    img_x, img_y = img.shape
    psf_x, psf_y = psf.shape

    m_x = ((psf_x + 1) >> 1) - 1
    m_y = ((psf_y + 1) >> 1) - 1

    blurred = np.zeros_like(img)

    for x in range(img_x):
        for y in range(img_y):

            if (psf_x < x <= (img_x - psf_x)) and (psf_y < y <= (img_y - psf_y)):
                blurred[x, y] = img[x, y]
            else:
                val = 0.0 + 0.0j
                for i in range(psf_x):
                    for j in range(psf_y):
                        u = i - m_x
                        v = j - m_y
                        if 0 <= (x + u) < img_x and 0 <= (y + v) < img_y:
                            val += psf[i, j] * img[x + u, y + v]
                blurred[x, y] = val

    return blurred

def edgetaper(img):

    img_x, img_y = img.shape

    psf_x = 10
    psf_y = 10
    psf = create_gaussian_kernel(psf_x, psf_y, 1.0)

    blurred = create_blurred(img, psf)
    weights = create_weights(img_x, img_y, psf)

    edge = np.zeros_like(img)
    for x in range(img_x):
        for y in range(img_y):
            if weights[x, y] != 1.0:
                edge[x, y] = img[x, y] * weights[x, y] + blurred[x, y] * (1.0 - weights[x, y])
            else:
                edge[x, y] = img[x, y]

    return edge
