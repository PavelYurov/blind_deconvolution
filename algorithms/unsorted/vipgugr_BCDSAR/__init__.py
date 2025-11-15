from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve2d

from ..base import DeconvolutionAlgorithm

Array2D = np.ndarray

_EPS = np.finfo(np.float64).eps


def _psf2otf(psf: Array2D, shape: Tuple[int, int]) -> Array2D:
    padded = np.zeros(shape, dtype=np.float64)
    psf_shape = psf.shape
    padded[:psf_shape[0], :psf_shape[1]] = psf
    for axis, axis_size in enumerate(psf_shape):
        padded = np.roll(padded, -axis_size // 2, axis=axis)
    return np.fft.fft2(padded)


def _intensities_to_od(image: Array2D) -> Array2D:
    safe = np.array(image, dtype=np.float64, copy=True)
    mask = safe <= 0.0
    if np.any(mask):
        positives = safe[~mask]
        replacement = positives.min() if positives.size else _EPS
        safe[mask] = replacement
    return -np.log10(safe)


def _od_to_intensities(od: Array2D) -> Array2D:
    return np.power(10.0, -od, dtype=np.float64)


def _image_to_matrix(image: Array2D) -> Array2D:
    m, n, c = image.shape
    return np.reshape(image, (m * n, c), order='F').T


def _matrix_to_image(matrix: Array2D, shape: Tuple[int, int]) -> Array2D:
    m, n = shape
    c, total = matrix.shape
    if total != m * n:
        raise ValueError('Matrix size does not match image dimensions.')
    return np.reshape(matrix.T, (m, n, c), order='F')


def _computing_es_zs(YT: Array2D, CT: Array2D, M: Array2D) -> Tuple[np.ndarray, np.ndarray]:
    ns = CT.shape[0]
    nc = YT.shape[0]
    tamm = YT.shape[1]
    eminus = np.zeros((tamm, nc, ns), dtype=np.float64)
    zminus = np.zeros((tamm, ns), dtype=np.float64)
    residual = YT - M @ CT
    for s in range(ns):
        aux = residual + np.outer(M[:, s], CT[s, :])
        eminus[:, :, s] = aux.T
        zminus[:, s] = (M[:, s].T @ aux).T
    return zminus, eminus


def _alpha_update(CT: Array2D, FTF: Array2D, SigmaC: Array2D, shape: Tuple[int, int]) -> np.ndarray:
    ns = CT.shape[0]
    alpha = np.zeros(ns, dtype=np.float64)
    pixels = shape[0] * shape[1]
    for s in range(ns):
        ct_map = np.reshape(CT[s, :], shape, order='F')
        tmp = np.fft.fft2(ct_map)
        tmp = np.conj(tmp) * FTF * tmp
        norma = np.real(tmp.sum()) / pixels
        sigma_map = np.reshape(SigmaC[:, s], shape, order='F')
        traza = np.real((sigma_map * FTF).sum())
        alpha[s] = pixels / (norma + traza + _EPS)
    return alpha


def _beta_update(YT: Array2D, CT: Array2D, SigmaC: Array2D, M: Array2D, SigmaM: Array2D) -> float:
    tamm = YT.shape[1]
    residual = YT - M @ CT
    norma = np.sum(residual * residual)
    traza_diag = np.sum(SigmaC, axis=0)
    m_col_norms = np.sum(M * M, axis=0)
    traza1 = np.sum(traza_diag * m_col_norms)
    ct_col_norms = np.sum(CT * CT, axis=1)
    traza2 = np.sum((ct_col_norms + traza_diag) * (3.0 * SigmaM))
    denominator = norma + traza1 + traza2 + _EPS
    return 3.0 * tamm / denominator


def _gamma_update(M: Array2D, SigmaM: Array2D, RM: Array2D) -> np.ndarray:
    diff = M - RM
    aux = diff.T @ diff
    return 3.0 / (np.diag(aux) + 3.0 * SigmaM + _EPS)


def _color_vector_update(
    YT: Array2D,
    CT: Array2D,
    SigmaC: Array2D,
    M: Array2D,
    RM: Array2D,
    beta: float,
    gamma: np.ndarray,
) -> Tuple[Array2D, np.ndarray]:
    ns = CT.shape[0]
    SigmaM = np.zeros(ns, dtype=np.float64)
    for s in range(ns):
        _, eminus = _computing_es_zs(YT, CT, M)
        expC2 = float(CT[s, :] @ CT[s, :].T + np.sum(SigmaC[:, s]))
        SigmaM[s] = 1.0 / (beta * expC2 + gamma[s] + _EPS)
        contribution = CT[s, :] @ eminus[:, :, s]
        M[:, s] = SigmaM[s] * (beta * contribution + gamma[s] * RM[:, s])
        scale = np.linalg.norm(M[:, s]) + _EPS
        M[:, s] /= scale
        SigmaM[s] /= scale ** 2
    return M, SigmaM


def _conc_update(
    YT: Array2D,
    CT: Array2D,
    M: Array2D,
    SigmaM: Array2D,
    FTF: Array2D,
    beta: float,
    alpha: np.ndarray,
) -> Tuple[Array2D, Array2D]:
    ns = CT.shape[0]
    shape = FTF.shape
    pixels = shape[0] * shape[1]
    SigmaC = np.zeros((pixels, ns), dtype=np.float64)
    for s in range(ns):
        zminus, _ = _computing_es_zs(YT, CT, M)
        expM2 = float(M[:, s].T @ M[:, s] + 3.0 * SigmaM[s])
        denom = beta * expM2 + alpha[s] * FTF
        auxSigmaC = 1.0 / (denom + _EPS)
        z_map = np.reshape(zminus[:, s], shape, order='F')
        Fz = np.fft.fft2(z_map)
        cs = np.fft.ifft2(beta * auxSigmaC * Fz)
        CT[s, :] = np.real(cs).reshape(pixels, order='F')
        SigmaC[:, s] = np.real(auxSigmaC).reshape(pixels, order='F')
    return CT, SigmaC


def _load_default_reference(ns: int) -> np.ndarray:
    mat_path = Path(__file__).resolve().parent / 'source' / 'MLandini.mat'
    data = loadmat(mat_path)
    rm = np.asarray(data['RM'], dtype=np.float64)
    if rm.shape[1] < ns:
        raise ValueError('Reference matrix does not have enough columns for the requested number of stains.')
    return rm[:, :ns]


class VipgugrBCDSAR(DeconvolutionAlgorithm):
    def __init__(
        self,
        tolerance: float = 2.0e-5,
        max_iter: int = 100,
        num_stains: int = 2,
        reference_matrix: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__('BCDHE')
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)
        self.num_stains = int(num_stains)
        self.reference_matrix = None if reference_matrix is None else np.asarray(reference_matrix, dtype=np.float64).copy()
        self._last_concentrations: Optional[np.ndarray] = None
        self._last_color_matrix: Optional[np.ndarray] = None

    def change_param(self, param: Any):
        if isinstance(param, dict):
            if 'tolerance' in param and param['tolerance'] is not None:
                self.tolerance = float(param['tolerance'])
            if 'max_iter' in param and param['max_iter'] is not None:
                self.max_iter = int(param['max_iter'])
            if 'num_stains' in param and param['num_stains'] is not None:
                self.num_stains = int(param['num_stains'])
            if 'reference_matrix' in param:
                value = param['reference_matrix']
                self.reference_matrix = None if value is None else np.asarray(value, dtype=np.float64).copy()
            return super().change_param(param)
        return super().change_param(param)

    def _prepare_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.dtype]:
        original_dtype = image.dtype
        work = image.astype(np.float64, copy=False)
        if work.ndim == 2:
            work = np.repeat(work[:, :, None], 3, axis=2)
        if work.shape[2] != 3:
            raise ValueError('BCDHE expects an RGB image.')
        if work.max() > 1.5:
            work = work / 255.0
        return work, original_dtype

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        working, original_dtype = self._prepare_image(image)
        start = time()

        m, n, _ = working.shape
        pixels = m * n
        ns = int(self.num_stains)
        if ns <= 0:
            raise ValueError('Number of stains must be positive.')

        if self.reference_matrix is not None:
            RM = np.asarray(self.reference_matrix, dtype=np.float64)
            if RM.shape[0] != 3:
                raise ValueError('Reference matrix must have three rows (RGB).')
            if RM.shape[1] < ns:
                raise ValueError('Reference matrix does not have enough columns.')
            RM = RM[:, :ns]
        else:
            RM = _load_default_reference(ns)

        od_image = _intensities_to_od(working)
        YT = _image_to_matrix(od_image)

        CT, *_ = np.linalg.lstsq(RM, YT, rcond=None)
        CT = np.maximum(CT, _EPS)
        M = RM.copy()
        SigmaM = np.zeros(ns, dtype=np.float64)
        SigmaC = np.zeros((pixels, ns), dtype=np.float64)

        Fn = np.array([[0.0, -0.25, 0.0],
                       [-0.25, 1.0, -0.25],
                       [0.0, -0.25, 0.0]], dtype=np.float64)
        FTFn = convolve2d(Fn, Fn, mode='full')
        FTF = _psf2otf(FTFn, (m, n))

        CT_prev = CT.copy()
        tolerance = float(self.tolerance)
        max_iter = int(self.max_iter)

        iteration = 1
        convergence = np.full(ns, tolerance + 1.0, dtype=np.float64)

        while iteration <= max_iter and np.any(convergence > tolerance):
            beta = _beta_update(YT, CT, SigmaC, M, SigmaM)
            alpha = _alpha_update(CT, FTF, SigmaC, (m, n))
            gamma = _gamma_update(M, SigmaM, RM)
            M, SigmaM = _color_vector_update(YT, CT, SigmaC, M, RM, beta, gamma)
            CT, SigmaC = _conc_update(YT, CT, M, SigmaM, FTF, beta, alpha)

            diff = CT - CT_prev
            denom = np.sum(CT_prev * CT_prev, axis=1) + _EPS
            convergence = np.sum(diff * diff, axis=1) / denom
            CT_prev = CT.copy()
            iteration += 1

        CT = np.maximum(CT, _EPS)
        self.timer = time() - start

        reconstructed_od = M @ CT
        reconstructed_od_image = _matrix_to_image(reconstructed_od, (m, n))
        reconstructed = _od_to_intensities(reconstructed_od_image)
        reconstructed = np.clip(reconstructed, 0.0, 1.0)
        if np.issubdtype(original_dtype, np.integer):
            restored = np.clip(reconstructed * 255.0, 0, 255).round().astype(original_dtype)
        else:
            restored = reconstructed.astype(original_dtype, copy=False)

        self._last_concentrations = CT.copy()
        self._last_color_matrix = M.copy()

        kernel = M.astype(np.float32)
        return restored, kernel

    def get_param(self):
        return [
            ('tolerance', self.tolerance),
            ('max_iter', self.max_iter),
            ('num_stains', self.num_stains),
            ('reference_matrix', None if self.reference_matrix is None else self.reference_matrix.copy()),
        ]

__all__ = ['VipgugrBCDSAR']

__all__ = ["VipgugrBCDSAR"]
