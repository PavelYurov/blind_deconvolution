from __future__ import annotations

import numpy as np
from scipy.signal import correlate2d


def check_input_dim(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[..., np.newaxis]
    return array


def check_output_dim(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[2] == 1:
        return array[..., 0]
    return array


def _check_output(shape: tuple[int, ...], output: np.ndarray | None, init: bool = False) -> np.ndarray:
    if output is None:
        output = np.zeros(shape, dtype=float)
    else:
        if output.shape != shape:
            raise ValueError("Shapes mismatch")
        if init:
            output[...] = 0.0
    return output


def project_simplex(y: np.ndarray, norm: float = 1.0) -> np.ndarray:
    y_flat = np.asarray(y, dtype=float).ravel()
    n = y_flat.size
    if n == 0:
        return y_flat.reshape(y.shape)

    sorted_idx = np.argsort(y_flat)[::-1]
    y_sorted = y_flat[sorted_idx]

    cumulative = 0.0
    theta = 0.0
    found = False
    for i in range(n - 1):
        cumulative += float(y_sorted[i])
        theta = (cumulative - norm) / float(i + 1)
        if theta >= float(y_sorted[i + 1]):
            found = True
            break
    if not found:
        theta = (float(y_sorted.sum()) - norm) / float(n)

    projected = np.clip(y_flat - theta, 0.0, np.inf)
    return projected.reshape(y.shape)


def _reflect_pad_2d(array: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    if pad_y == 0 and pad_x == 0:
        return array
    return np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")


def convolve2d(input: np.ndarray, kernel: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
    input_ = check_input_dim(np.asarray(input, dtype=float))
    kernel = np.asarray(kernel, dtype=float)
    out = _check_output(np.asarray(input).shape, output, init=False)
    out_ = check_input_dim(out)

    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2

    for channel in range(input_.shape[2]):
        padded = _reflect_pad_2d(input_[..., channel], pad_y, pad_x)
        out_[..., channel] = correlate2d(padded, kernel, mode="valid")
    return out


def dx(image: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
    image_ = check_input_dim(np.asarray(image, dtype=float))
    out = _check_output(np.asarray(image).shape, output)
    out_ = check_input_dim(out)

    out_[..., :-1, :] = image_[:, 1:, :] - image_[:, :-1, :]
    if image_.shape[1] >= 2:
        out_[:, -1, :] = -out_[:, -2, :]
    else:
        out_[:, -1, :] = 0.0
    return out


def dx_b(image: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
    image_ = check_input_dim(np.asarray(image, dtype=float))
    out = _check_output(np.asarray(image).shape, output)
    out_ = check_input_dim(out)

    out_[:, 1:, :] = image_[:, 1:, :] - image_[:, :-1, :]
    if image_.shape[1] >= 2:
        out_[:, 0, :] = -out_[:, 1, :]
    else:
        out_[:, 0, :] = 0.0
    return out


def dy(image: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
    out = _check_output(np.asarray(image).shape, output)
    return dx(np.swapaxes(image, 0, 1), np.swapaxes(out, 0, 1)).swapaxes(0, 1)


def dy_b(image: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
    out = _check_output(np.asarray(image).shape, output)
    return dx_b(np.swapaxes(image, 0, 1), np.swapaxes(out, 0, 1)).swapaxes(0, 1)


def norm2(mat: np.ndarray, mask: np.ndarray | None = None) -> float:
    mat_ = check_input_dim(np.asarray(mat, dtype=float))
    if mask is None:
        return float(np.sum(mat_ * mat_))
    mask = np.asarray(mask).astype(bool)
    masked = mat_[mask, :]
    return float(np.sum(masked * masked))


def _log_phi(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    two_pi_inv = 1.0 / (2.0 * np.pi)
    numerator = b * (a ** (b - 1.0)) * np.sin(np.pi / b) * two_pi_inv
    denominator = (a**b) + (np.abs(x) ** b)
    return np.log(numerator / denominator)


def _dlog_phi(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    apowb = a**b
    absx = np.abs(x)
    xpowb = absx**b
    den = apowb + xpowb
    two_pi_inv = 1.0 / (2.0 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -b * b * apowb / a * np.sin(np.pi / b) * two_pi_inv * xpowb / (x * den * den)
    out = np.where(x == 0, 0.0, out)
    return out


def global_prior(dL: np.ndarray, a: float, b: float) -> float:
    return float(np.sum(_log_phi(check_input_dim(dL), a, b)))


def grad_global_prior_x(dL: np.ndarray, a: float, b: float, output: np.ndarray | None = None) -> np.ndarray:
    dL_ = check_input_dim(np.asarray(dL, dtype=float))
    out = _check_output(np.asarray(dL).shape, output, init=True)
    out_ = check_input_dim(out)

    m = dL_.shape[1]
    if m == 0:
        return out

    r = _dlog_phi(dL_[:, :-1, :], a, b)
    out_[:, :-1, :] -= r
    out_[:, 1:, :] += r

    if m >= 2:
        r_last = _dlog_phi(dL_[:, m - 1 : m, :], a, b)
        out_[:, m - 1 : m, :] -= r_last
        out_[:, m - 2 : m - 1, :] -= r_last
    return out


def grad_global_prior_y(dL: np.ndarray, a: float, b: float, output: np.ndarray | None = None) -> np.ndarray:
    out = _check_output(np.asarray(dL).shape, output, init=True)
    return grad_global_prior_x(np.swapaxes(dL, 0, 1), a, b, np.swapaxes(out, 0, 1)).swapaxes(0, 1)


def local_prior(I: np.ndarray, J: np.ndarray, M: np.ndarray) -> float:
    I_ = check_input_dim(np.asarray(I, dtype=float))
    J_ = check_input_dim(np.asarray(J, dtype=float))
    if I_.shape != J_.shape:
        raise ValueError("Shapes mismatch")
    M = np.asarray(M).astype(bool)
    diff = I_ - J_
    return float(np.sum((diff[M, :]) ** 2))


def grad_local_prior_x(
    dxL: np.ndarray, dxI: np.ndarray, M: np.ndarray, output: np.ndarray | None = None
) -> np.ndarray:
    out = _check_output(np.asarray(dxL).shape, output)
    dx_b(np.asarray(dxL, dtype=float) - np.asarray(dxI, dtype=float), out)
    out = out * -2.0
    if out.ndim == 3:
        return out * np.asarray(M)[..., np.newaxis]
    return out * np.asarray(M)


def grad_local_prior_y(
    dyL: np.ndarray, dyI: np.ndarray, M: np.ndarray, output: np.ndarray | None = None
) -> np.ndarray:
    out = _check_output(np.asarray(dyL).shape, output)
    return grad_local_prior_x(
        np.swapaxes(dyL, 0, 1),
        np.swapaxes(dyI, 0, 1),
        np.swapaxes(M, 0, 1),
        np.swapaxes(out, 0, 1),
    ).swapaxes(0, 1)


def grad_L(P: np.ndarray, R: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    R_ = check_input_dim(np.asarray(R, dtype=float))
    out = _check_output(np.asarray(R).shape, output)
    out_ = check_input_dim(out)

    kernel = P[::-1, ::-1]
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    for channel in range(R_.shape[2]):
        padded = _reflect_pad_2d(R_[..., channel], pad_y, pad_x)
        out_[..., channel] = 2.0 * correlate2d(padded, kernel, mode="valid")
    return out


def grad_P(pshape: tuple[int, int], L: np.ndarray, R: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
    npk, npl = int(pshape[0]), int(pshape[1])
    L_ = check_input_dim(np.asarray(L, dtype=float))
    R_ = check_input_dim(np.asarray(R, dtype=float))
    if L_.shape != R_.shape:
        raise ValueError("Shapes mismatch")

    out = _check_output((npk, npl) if np.asarray(L).ndim == 2 else (npk, npl, L_.shape[2]), output)
    out_ = check_input_dim(out)

    pad_y = npk // 2
    pad_x = npl // 2
    for channel in range(L_.shape[2]):
        padded = _reflect_pad_2d(L_[..., channel], pad_y, pad_x)
        for k in range(npk):
            for l in range(npl):
                shift_y = pad_y - k
                shift_x = pad_x - l
                start_y = pad_y + shift_y
                start_x = pad_x + shift_x
                window = padded[
                    start_y : start_y + R_.shape[0],
                    start_x : start_x + R_.shape[1],
                ]
                out_[k, l, channel] = 2.0 * float(np.sum(window * R_[..., channel]))
    return out
