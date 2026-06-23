import numpy as np
from scipy.ndimage import convolve as _ndconvolve
from numpy.fft import ifft2, fftshift

__all__ = ['act_denoise']


def _choose_num_scales(H, W):


    return max(2, min(4, int(np.ceil(np.log2(min(H, W)))) - 2))


def _udct_pad_multiple(num_scales):


    return 1 << max(num_scales - 1, 0)


def _make_udct(H, W, num_scales=None):


    from curvelets.numpy import UDCT

    if num_scales is None:
        num_scales = _choose_num_scales(H, W)

    return UDCT(shape=(H, W), num_scales=num_scales, transform_kind='real')


def _compute_curvelet_noise_rootpsd(fft_psd, udct_op):


    kernel_noise = fftshift(ifft2(np.sqrt(fft_psd.astype(np.complex128)))).real

    c_struct = udct_op.forward(kernel_noise)

    rootpsd = []
    for scale in c_struct:
        dirs = []
        for direction in scale:
            wedges = []
            for wedge in direction:
                rms = float(np.sqrt(np.mean(np.abs(wedge) ** 2)))
                wedges.append(rms)
            dirs.append(wedges)
        rootpsd.append(dirs)
    return rootpsd


def _ml_estimator(noisy_coeffs, noise_rootpsd, noise_type):


    if noise_type == 'white':

        k = np.ones((7, 7), dtype=np.float64) / 48.0
        k[3, 3] = 0.0
    else:

        k = np.ones((31, 31), dtype=np.float64) / 960.0
        k[15, 15] = 0.0


    power = (np.abs(noisy_coeffs) ** 2).astype(np.float64)
    local_var = _ndconvolve(power, k, mode='wrap')


    clean_var = local_var - noise_rootpsd ** 2
    clean_var = np.maximum(clean_var, 0.0)

    return np.sqrt(clean_var)


def _apply_act(c_struct, rootpsd, threshold_setting, noise_type):


    nscales = len(c_struct)
    denoised = []

    for J in range(nscales):
        dirs = []
        for D in range(len(c_struct[J])):
            wedges = []
            for W in range(len(c_struct[J][D])):
                coeff = c_struct[J][D][W].copy()


                if J == 0:
                    wedges.append(coeff)
                    continue

                sigma_n = rootpsd[J][D][W]
                mag = np.abs(coeff)

                if threshold_setting in ('s', 'h'):

                    clean_std = _ml_estimator(coeff, sigma_n, noise_type)
                    safe_std = np.maximum(clean_std, 1e-10)

                    if threshold_setting == 's':

                        threshold = np.sqrt(2.0) * (sigma_n ** 2) / safe_std
                        threshold = np.where(clean_std > 0, threshold, np.inf)


                        shrunk = np.maximum(mag - threshold, 0.0)
                        coeff = np.where(
                            mag > 1e-30,
                            coeff * (shrunk / mag),
                            np.zeros_like(coeff),
                        )

                    else:

                        is_finest = float(J == nscales - 1)
                        threshold = ((3.0 + is_finest) * (sigma_n ** 2)
                                     / (np.sqrt(2.0) * safe_std))
                        threshold = np.where(clean_std > 0, threshold, np.inf)

                        coeff = coeff * (mag > threshold)

                else:
                    is_finest = float(J == nscales - 1)
                    threshold = (3.0 + is_finest) * sigma_n

                    coeff = coeff * (mag > threshold)

                wedges.append(coeff)
            dirs.append(wedges)
        denoised.append(dirs)

    return denoised


def act_denoise(image, noise_var=None, threshold_setting='s'):


    if threshold_setting not in ('s', 'h', 'ksigma'):
        raise ValueError(
            f"threshold_setting='{threshold_setting}': "
            f"choose from 's', 'h', 'ksigma'")

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {img.shape}")
    H, W = img.shape


    num_scales = _choose_num_scales(H, W)
    pad_mult = _udct_pad_multiple(num_scales)
    Hp = H + (-H) % pad_mult
    Wp = W + (-W) % pad_mult
    pad_h = Hp - H
    pad_w = Wp - W
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    N = Hp * Wp


    udct_op = _make_udct(Hp, Wp, num_scales=num_scales)


    c_struct = udct_op.forward(img)


    blind = noise_var is None
    if blind:


        mads = []
        for direction in c_struct[-1]:
            for wedge in direction:
                vals = wedge.real.ravel() if np.iscomplexobj(wedge)\
                    else wedge.ravel()
                mads.append(
                    np.median(np.abs(vals - np.median(vals))) / 0.6745)
        noise_std = float(np.median(mads))
        noise_var = noise_std ** 2


    scalar_var = (np.isscalar(noise_var)
                  or (isinstance(noise_var, np.ndarray)
                      and noise_var.size == 1))
    if scalar_var:
        sigma2 = float(np.ravel(noise_var)[0]
                        if not np.isscalar(noise_var)
                        else noise_var)
        fft_psd = np.full((Hp, Wp), sigma2 * N, dtype=np.float64)
        noise_type = 'white'
    else:
        fft_psd = np.asarray(noise_var, dtype=np.float64)
        if fft_psd.shape == (H, W) and (pad_h or pad_w):
            fft_psd = np.pad(fft_psd, ((0, pad_h), (0, pad_w)), mode='wrap')
        if fft_psd.shape != (Hp, Wp):
            raise ValueError(
                f"FFT-PSD shape {fft_psd.shape} != image ({Hp}, {Wp})")

        psd_range = float(fft_psd.max() - fft_psd.min())
        noise_type = 'white' if psd_range < 0.015 * N else 'colored'


    rootpsd = _compute_curvelet_noise_rootpsd(fft_psd, udct_op)


    denoised_struct = _apply_act(
        c_struct, rootpsd, threshold_setting, noise_type)


    denoised = udct_op.backward(denoised_struct)
    if np.iscomplexobj(denoised):
        denoised = denoised.real


    denoised = denoised[:H, :W]

    info = {
        'noise_type': noise_type,
        'noise_var': sigma2 if scalar_var else 'fft_psd',
        'threshold_setting': threshold_setting,
        'blind': blind,
    }
    return denoised, info
