import numpy as np
from scipy.signal import convolve2d as _scipy_convolve2d

def _matlab_round(x):

    return int(np.floor(x + 0.5))

def conv2fft(a: np.ndarray, b: np.ndarray, mode: str = 'full') -> np.ndarray:

    Nx1, Nx2 = a.shape
    NKx1, NKx2 = b.shape

    full_shape = (Nx1 + NKx1 - 1, Nx2 + NKx2 - 1)

    ahat = np.fft.fftshift(np.fft.fftn(a, s=full_shape))
    bhat = np.fft.fftshift(np.fft.fftn(b, s=full_shape))

    chat = ahat * bhat

    c = np.real(np.fft.ifftn(np.fft.ifftshift(chat)))

    if mode == 'same':

        r0 = NKx1 // 2
        c0 = NKx2 // 2
        c = c[r0:r0 + Nx1, c0:c0 + Nx2]
    elif mode == 'valid':

        c = c[NKx1 - 1:Nx1, NKx2 - 1:Nx2]

    return c

def conv2_matlab(a: np.ndarray, b: np.ndarray,
                 mode: str = 'full') -> np.ndarray:

    return _scipy_convolve2d(a, b, mode=mode)

def grad_tv_cc(f: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:

    squeeze_out = (f.ndim == 2)
    if f.ndim == 2:
        f = f[:, :, np.newaxis]

    fxforw = np.concatenate([f[1:, :, :], f[-1:, :, :]], axis=0) - f

    fyforw = np.concatenate([f[:, 1:, :], f[:, -1:, :]], axis=1) - f

    fxback = f - np.concatenate([f[:1, :, :], f[:-1, :, :]], axis=0)

    fyback = f - np.concatenate([f[:, :1, :], f[:, :-1, :]], axis=1)

    f_down = np.concatenate([f[1:, :, :], f[-1:, :, :]], axis=0)
    f_left = np.concatenate([f[:, :1, :], f[:, :-1, :]], axis=1)
    f_down_left = np.concatenate(
        [f_down[:, :1, :], f_down[:, :-1, :]], axis=1
    )
    fxmixd = f_down_left - f_left

    f_up = np.concatenate([f[:1, :, :], f[:-1, :, :]], axis=0)
    f_up_right = np.concatenate(
        [f_up[:, 1:, :], f_up[:, -1:, :]], axis=1
    )
    fymixd = f_up_right - f_up

    divTV = np.zeros_like(f)
    for cc in range(f.shape[2]):
        fxf = fxforw[:, :, cc]
        fyf = fyforw[:, :, cc]
        fxb = fxback[:, :, cc]
        fyb = fyback[:, :, cc]
        fxm = fxmixd[:, :, cc]
        fym = fymixd[:, :, cc]

        divTV[:, :, cc] = (
            (fxf + fyf)
            / np.maximum(epsilon, np.sqrt(fxf ** 2 + fyf ** 2))
            - fxb
            / np.maximum(epsilon, np.sqrt(fxb ** 2 + fym ** 2))
            - fyb
            / np.maximum(epsilon, np.sqrt(fxm ** 2 + fyb ** 2))
        )

    if squeeze_out:
        divTV = divTV[:, :, 0]

    return divTV

def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:

    return np.power(image, gamma)

def imresize(image: np.ndarray, target_size: tuple,
             method: str = 'bicubic') -> np.ndarray:

    from skimage.transform import resize as sk_resize

    order_map = {'bicubic': 3, 'bilinear': 1, 'nearest': 0}
    order = order_map.get(method, 3)

    is_downsampling = (target_size[0] < image.shape[0]
                       or target_size[1] < image.shape[1])

    if image.ndim == 3:
        out_shape = (target_size[0], target_size[1], image.shape[2])
    else:
        out_shape = target_size

    return sk_resize(
        image, out_shape, order=order,
        anti_aliasing=is_downsampling,
        preserve_range=True,
        mode='edge',
    )

def build_pyramid(f: np.ndarray, MK: int, NK: int,
                  final_lambda: float, lambda_multiplier: float,
                  interp_method: str = 'bicubic',
                  scale_multiplier: float = 1.1,
                  largest_lambda: float = 0.11):

    M, N = f.shape[:2]
    smallest_scale = 3

    fp = [f]
    Mp = [M]
    Np = [N]
    MKp = [MK]
    NKp = [NK]
    lambdas = [final_lambda]

    while (MKp[-1] > smallest_scale
           and NKp[-1] > smallest_scale
           and lambdas[-1] * lambda_multiplier < largest_lambda):

        new_lambda = lambdas[-1] * lambda_multiplier

        new_MK = _matlab_round(MKp[-1] / scale_multiplier)
        new_NK = _matlab_round(NKp[-1] / scale_multiplier)

        if new_MK % 2 == 0:
            new_MK -= 1
        if new_NK % 2 == 0:
            new_NK -= 1

        if new_NK == NKp[-1]:
            new_NK -= 2
        if new_MK == MKp[-1]:
            new_MK -= 2

        if new_NK < smallest_scale:
            new_NK = smallest_scale
        if new_MK < smallest_scale:
            new_MK = smallest_scale

        factor_M = MKp[-1] / new_MK
        factor_N = NKp[-1] / new_NK

        new_M = _matlab_round(Mp[-1] / factor_M)
        new_N = _matlab_round(Np[-1] / factor_N)

        if new_M % 2 == 0:
            new_M -= 1
        if new_N % 2 == 0:
            new_N -= 1

        resized = imresize(f, (new_M, new_N), method=interp_method)

        fp.append(resized)
        Mp.append(new_M)
        Np.append(new_N)
        MKp.append(new_MK)
        NKp.append(new_NK)
        lambdas.append(new_lambda)

    num_scales = len(fp)
    return fp, Mp, Np, MKp, NKp, lambdas, num_scales

def comp_upto_shift(I1: np.ndarray, I2: np.ndarray):

    from scipy.interpolate import RegularGridInterpolator

    maxshift = 5

    shifts = np.arange(-5, 5.25, 0.25)

    I2c = I2[15:-15, 15:-15].copy()

    I1c = I1[15 - maxshift:I1.shape[0] - 15 + maxshift,
             15 - maxshift:I1.shape[1] - 15 + maxshift].copy()

    N1, N2 = I2c.shape

    x_1d = np.arange(1 - maxshift, N2 + maxshift + 1, dtype=np.float64)
    y_1d = np.arange(1 - maxshift, N1 + maxshift + 1, dtype=np.float64)

    interp_func = RegularGridInterpolator(
        (y_1d, x_1d), I1c.astype(np.float64),
        method='linear', bounds_error=False, fill_value=np.nan,
    )

    gx0, gy0 = np.meshgrid(
        np.arange(1, N2 + 1, dtype=np.float64),
        np.arange(1, N1 + 1, dtype=np.float64),
    )

    ssdem = np.full((len(shifts), len(shifts)), np.inf)
    for i, si in enumerate(shifts):
        for j, sj in enumerate(shifts):
            gxn = gx0 + si
            gyn = gy0 + sj
            pts = np.stack([gyn.ravel(), gxn.ravel()], axis=-1)
            tI1_flat = interp_func(pts)
            tI1_tmp = tI1_flat.reshape(N1, N2)
            ssdem[i, j] = np.nansum((tI1_tmp - I2c) ** 2)

    ssde = ssdem.min()
    idx = np.unravel_index(ssdem.argmin(), ssdem.shape)

    gxn = gx0 + shifts[idx[0]]
    gyn = gy0 + shifts[idx[1]]
    pts = np.stack([gyn.ravel(), gxn.ravel()], axis=-1)
    tI1 = interp_func(pts).reshape(N1, N2)

    return ssde, tI1
