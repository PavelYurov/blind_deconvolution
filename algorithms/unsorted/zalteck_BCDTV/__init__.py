from __future__ import annotations

import math
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import LinearOperator, cg

from ..base import DeconvolutionAlgorithm

Array2D = np.ndarray

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _intensities_to_od(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 1e-6, None)
    return -np.log10(clipped)


def _od_to_intensities(od: np.ndarray) -> np.ndarray:
    return np.power(10.0, -od)


def _psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    psf = np.asarray(psf, dtype=np.float64)
    otf = np.zeros(shape, dtype=np.float64)
    h, w = psf.shape
    otf[:h, :w] = psf
    otf = np.roll(otf, -h // 2, axis=0)
    otf = np.roll(otf, -w // 2, axis=1)
    return np.fft.fftn(otf)


def _circ_gradient(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dfh = np.empty_like(f, dtype=np.float64)
    dfv = np.empty_like(f, dtype=np.float64)

    dfh[:, :-1] = f[:, :-1] - f[:, 1:]
    dfh[:, -1] = f[:, -1] - f[:, 0]

    dfv[:-1, :] = f[:-1, :] - f[1:, :]
    dfv[-1, :] = f[-1, :] - f[0, :]
    return dfh, dfv


def _tcirc_gradient(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dfhT = np.empty_like(f, dtype=np.float64)
    dfvT = np.empty_like(f, dtype=np.float64)

    dfhT[:, 0] = f[:, 0] - f[:, -1]
    dfhT[:, 1:] = f[:, 1:] - f[:, :-1]

    dfvT[0, :] = f[0, :] - f[-1, :]
    dfvT[1:, :] = f[1:, :] - f[:-1, :]
    return dfhT, dfvT


def _computing_es_zs(YT: np.ndarray, CT: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ns = CT.shape[0]
    nc, tamm = YT.shape
    eminus = np.zeros((tamm, nc, ns), dtype=np.float64)
    zminus = np.zeros((tamm, ns), dtype=np.float64)

    YT_minus = YT - M @ CT
    for s in range(ns):
        aux = YT_minus + M[:, s : s + 1] @ CT[s : s + 1, :]
        eminus[:, :, s] = aux.T
        zminus[:, s] = (M[:, s].T @ aux).ravel()
    return zminus, eminus


def _beta_update(YT: np.ndarray, CT: np.ndarray, SigmaC: np.ndarray, M: np.ndarray, SigmaM: np.ndarray) -> float:
    tamm = YT.shape[1]
    residual = YT - M @ CT
    norma = float(np.trace(residual @ residual.T))

    sigma_sums = SigmaC.sum(axis=0)
    trazadiagonal = np.diag(sigma_sums)
    traza1 = float(np.trace(M.T @ M @ trazadiagonal))
    traza2 = float(np.trace((CT @ CT.T + trazadiagonal) @ np.diag(3.0 * SigmaM)))
    denom = norma + traza1 + traza2 + 1e-12
    return float(3.0 * tamm / denom)


def _alpha_update_tv(
    CT: np.ndarray,
    d_sum_real: np.ndarray,
    SigmaC: np.ndarray,
    eps_w: float,
    m: int,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ns = CT.shape[0]
    tamm = m * n
    alpha = np.zeros(ns, dtype=np.float64)
    WT = np.zeros((ns, tamm), dtype=np.float64)
    denom_sum = float(np.sum(d_sum_real))

    for s in range(ns):
        sigma_scalar = float(np.mean(SigmaC[:, s])) if SigmaC.size else 0.0
        traza = sigma_scalar * denom_sum
        cs = CT[s, :].reshape(m, n)
        Dhy, Dvy = _circ_gradient(cs)
        v = Dhy ** 2 + Dvy ** 2 + traza / max(tamm, 1)
        v = np.maximum(v, 0.0)
        tmp = np.sqrt(v)
        W = 1.0 / (eps_w + tmp)
        WT[s, :] = W.reshape(-1)
        normaptraza = 2.0 * float(np.sum(tmp))
        alpha[s] = tamm / (normaptraza + np.finfo(np.float64).eps)
    return alpha, WT


def _gamma_update(M: np.ndarray, SigmaM: np.ndarray, RM: np.ndarray) -> np.ndarray:
    aux = M - RM + 1e-12
    aux2 = aux.T @ aux
    diag_term = np.diag(aux2 + np.diag(3.0 * SigmaM))
    return 3.0 / (diag_term + 1e-12)


def _color_vector_update(
    YT: np.ndarray,
    CT: np.ndarray,
    SigmaC: np.ndarray,
    M: np.ndarray,
    RM: np.ndarray,
    beta: float,
    gamma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ns = CT.shape[0]
    SigmaM = np.zeros(ns, dtype=np.float64)

    for s in range(ns):
        _, eminus = _computing_es_zs(YT, CT, M)
        expC2 = float(CT[s, :] @ CT[s, :].T + np.sum(SigmaC[:, s]))
        SigmaM[s] = 1.0 / (beta * expC2 + gamma[s])
        update_vec = beta * (CT[s, :] @ eminus[:, :, s]) + gamma[s] * RM[:, s]
        M[:, s] = SigmaM[s] * update_vec
        scalefactor = float(np.linalg.norm(M[:, s]))
        if scalefactor < 1e-12:
            scalefactor = 1.0
        M[:, s] = M[:, s] / scalefactor
        SigmaM[s] = SigmaM[s] / (scalefactor ** 2)
    return M, SigmaM


def _conc_update_tv(
    YT: np.ndarray,
    CT: np.ndarray,
    M: np.ndarray,
    SigmaM: np.ndarray,
    d_sum_real: np.ndarray,
    beta: float,
    alpha: np.ndarray,
    m: int,
    n: int,
    WT: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ns = CT.shape[0]
    tamm = YT.shape[1]
    SigmaC = np.zeros((tamm, ns), dtype=np.float64)
    zminus, _ = _computing_es_zs(YT, CT, M)

    eps_cs = 1.0e-7
    itmax_cs = 50

    for s in range(ns):
        W = WT[s, :].reshape(m, n)
        expM2 = float(M[:, s] @ M[:, s] + 3.0 * SigmaM[s])
        denom = beta * expM2 + alpha[s] * np.mean(W) * d_sum_real
        auxSigmaC = 1.0 / (denom + 1e-12)
        SigmaC[:, s] = auxSigmaC.reshape(-1)

        inv_cov_fix = beta * expM2
        indep_term = beta * zminus[:, s]

        def matvec(vec: np.ndarray) -> np.ndarray:
            cs_map = vec.reshape(m, n)
            Dhcs, Dvcs = _circ_gradient(cs_map)
            F2, _ = _tcirc_gradient(W * Dhcs)
            temp, F3 = _tcirc_gradient(W * Dvcs)
            prior_term = alpha[s] * (F2 + F3)
            result = inv_cov_fix * cs_map + prior_term
            return result.reshape(-1)

        A = LinearOperator((tamm, tamm), matvec=matvec, dtype=np.float64)
        cs0 = CT[s, :].copy()
        cs, info = cg(A, indep_term, x0=cs0, rtol=eps_cs, atol=0.0, maxiter=itmax_cs)
        if info != 0 or cs is None:
            cs = cs0
        CT[s, :] = cs
    return CT, SigmaC


def _reshape_concentrations(CT: np.ndarray, m: int, n: int) -> np.ndarray:
    ns = CT.shape[0]
    return CT.reshape(ns, m, n)


def _load_reference(reference: Optional[np.ndarray]) -> np.ndarray:
    if reference is not None:
        return np.asarray(reference, dtype=np.float64)
    mat_path = SOURCE_ROOT / "RMImageSet.mat"
    mat = loadmat(mat_path)
    return np.asarray(mat["RM"], dtype=np.float64)


class ZalteckBCDTV(DeconvolutionAlgorithm):
    """TV-based blind color deconvolution ported from the BCDTV MATLAB code."""

    def __init__(
        self,
        max_iterations: int = 10,
        min_iterations: int = 3,
        convergence_tol: float = 1e-5,
        reference_matrix: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__("BCDHETV")
        self.max_iterations = max(1, int(max_iterations))
        self.min_iterations = max(1, int(min_iterations))
        self.convergence_tol = float(convergence_tol)
        self.reference_matrix = _load_reference(reference_matrix)
        self._last_matrix: Optional[np.ndarray] = None
        self._last_concentrations: Optional[np.ndarray] = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        if "max_iterations" in param and param["max_iterations"] is not None:
            self.max_iterations = max(1, int(param["max_iterations"]))
        if "min_iterations" in param and param["min_iterations"] is not None:
            self.min_iterations = max(1, int(param["min_iterations"]))
        if "convergence_tol" in param and param["convergence_tol"] is not None:
            self.convergence_tol = float(param["convergence_tol"])
        if "reference_matrix" in param and param["reference_matrix"] is not None:
            self.reference_matrix = _load_reference(param["reference_matrix"])
        return super().change_param(param)

    def get_param(self):
        return [
            ("max_iterations", self.max_iterations),
            ("min_iterations", self.min_iterations),
            ("convergence_tol", self.convergence_tol),
            ("reference_matrix_shape", None if self.reference_matrix is None else self.reference_matrix.shape),
            ("last_matrix", None if self._last_matrix is None else self._last_matrix.tolist()),
        ]

    def process(self, image: Array2D) -> Tuple[Array2D, Array2D]:
        if image is None:
            raise ValueError("Input image is None.")

        arr = np.asarray(image)
        original_dtype = arr.dtype
        float_image = arr.astype(np.float64, copy=False)
        if float_image.ndim == 2:
            float_image = np.stack([float_image] * 3, axis=-1)
        if float_image.ndim != 3 or float_image.shape[2] != 3:
            raise ValueError("Expected an image with 3 channels after conversion.")

        if float_image.max() > 1.5:
            float_image = float_image / 255.0
        float_image = np.clip(float_image, 1e-6, 1.0)

        m, n, nc = float_image.shape
        if nc != 3:
            raise ValueError("BCDTV expects 3-channel input data.")

        YT = _intensities_to_od(float_image).reshape(-1, nc).T
        RM = np.asarray(self.reference_matrix, dtype=np.float64)
        ns = RM.shape[1]

        # Initial concentration estimate via least squares
        CT, *_ = np.linalg.lstsq(RM, YT, rcond=None)
        CT = np.clip(CT, np.finfo(np.float64).eps, None)
        M = RM.copy()
        SigmaM = np.zeros(ns, dtype=np.float64)
        SigmaC = np.zeros((m * n, ns), dtype=np.float64)

        term = self.convergence_tol
        min_iters = min(self.min_iterations, self.max_iterations)
        max_iters = self.max_iterations
        epsW = float(np.mean(CT)) * 1.0e-6 if CT.size else 1.0e-6
        epsW = max(epsW, 1.0e-10)

        DhtDh = _psf2otf(np.array([[-1.0, 2.0, -1.0]], dtype=np.float64), (m, n))
        DvtDv = _psf2otf(np.array([[-1.0], [2.0], [-1.0]], dtype=np.float64), (m, n))
        d_sum_real = np.real(DhtDh + DvtDv)

        prev_CT = CT.copy()
        start = time()
        for iteration in range(1, max_iters + 1):
            beta = _beta_update(YT, CT, SigmaC, M, SigmaM)
            alpha, WT = _alpha_update_tv(CT, d_sum_real, SigmaC, epsW, m, n)
            gamma = _gamma_update(M, SigmaM, RM)
            M, SigmaM = _color_vector_update(YT, CT, SigmaC, M, RM, beta, gamma)
            CT, SigmaC = _conc_update_tv(YT, CT, M, SigmaM, d_sum_real, beta, alpha, m, n, WT)

            diff = np.sum((CT - prev_CT) ** 2, axis=1)
            denom = np.sum(prev_CT ** 2, axis=1) + 1e-12
            conv = diff / denom
            prev_CT = CT.copy()
            if iteration >= min_iters and np.all(conv < term):
                break

        self.timer = time() - start

        CT = np.clip(CT, np.finfo(np.float64).eps, None)
        concentrations = _reshape_concentrations(CT, m, n)
        self._last_concentrations = concentrations
        self._last_matrix = M.copy()

        # Reconstruct first stain and convert to grayscale output
        stain_index = 0
        recon_od = np.outer(M[:, stain_index], CT[stain_index, :]).reshape(3, m, n)
        recon_od = np.transpose(recon_od, (1, 2, 0))
        recon_rgb = _od_to_intensities(recon_od)
        restored = np.mean(recon_rgb, axis=2)
        restored = np.clip(restored, 0.0, 1.0)

        if np.issubdtype(original_dtype, np.integer):
            restored = np.round(restored * 255.0).astype(original_dtype)
        else:
            restored = restored.astype(original_dtype, copy=False)

        return restored, M.astype(np.float64)

    def get_kernel(self) -> Array2D | None:
        return None if self._last_matrix is None else self._last_matrix.copy()


__all__ = ["ZalteckBCDTV"]
