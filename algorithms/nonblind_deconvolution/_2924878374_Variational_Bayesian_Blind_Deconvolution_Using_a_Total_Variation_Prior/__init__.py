from __future__ import annotations

from time import time
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg

from ...base import DeconvolutionAlgorithm

KernelSpec = Union[int, Tuple[int, int], Iterable[int]]


def _normalize_kernel_shape(kernel: KernelSpec) -> Tuple[int, int]:
    if isinstance(kernel, int):
        if kernel <= 0:
            raise ValueError("Kernel size must be positive.")
        return (int(kernel), int(kernel))

    if isinstance(kernel, tuple):
        if len(kernel) != 2:
            raise ValueError("Kernel tuple must have two elements.")
        h, w = kernel
        if h <= 0 or w <= 0:
            raise ValueError("Kernel dimensions must be positive.")
        return (int(h), int(w))

    try:
        values = tuple(int(v) for v in kernel)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("Unsupported kernel specification.") from exc

    if len(values) == 0:
        raise ValueError("Kernel specification can not be empty.")
    if len(values) == 1:
        return _normalize_kernel_shape(values[0])

    h, w = values[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Kernel dimensions must be positive.")
    return (int(h), int(w))


def _create_uniform_kernel(shape: Tuple[int, int]) -> np.ndarray:
    kernel = np.ones(shape, dtype=np.float32)
    total = float(kernel.sum())
    if total == 0:
        raise ValueError("Kernel sum can not be zero.")
    kernel /= total
    return kernel


def _prepare_image(image: np.ndarray) -> Tuple[np.ndarray, np.dtype, Optional[int]]:
    original_dtype = image.dtype
    channels: Optional[int] = None

    if image.ndim == 3 and image.shape[2] != 1:
        channels = int(image.shape[2])
        working = image.mean(axis=2)
    else:
        working = np.squeeze(image)

    working = working.astype(np.float32, copy=False)
    if working.size and float(working.max()) > 1.5:
        working = working / 255.0

    if working.ndim != 2:
        raise ValueError("Poisson TV algorithms expect a 2-D grayscale image.")

    return working, original_dtype, channels


def _restore_dtype(restored: np.ndarray, original_dtype: np.dtype, channels: Optional[int]) -> np.ndarray:
    clipped = np.clip(restored, 0.0, 1.0)

    if np.issubdtype(original_dtype, np.integer):
        output = (clipped * 255.0).round().astype(original_dtype)
    else:
        output = clipped.astype(original_dtype, copy=False)

    if channels is not None and channels > 1:
        output = np.repeat(output[..., None], channels, axis=2)

    return output


def _precompute_psf_fft(psf: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    psf = np.asarray(psf, dtype=np.float32)
    pad_h = image_shape[0] - psf.shape[0]
    pad_w = image_shape[1] - psf.shape[1]

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel can not be larger than the image.")

    psf_padded = np.pad(psf, ((0, pad_h), (0, pad_w)), mode="constant")
    shift = np.array(psf.shape) // 2
    psf_padded = np.roll(psf_padded, -shift, axis=(0, 1))
    return fft2(psf_padded)


def _apply_psf_fft_cached(image: np.ndarray, psf_fft: np.ndarray) -> np.ndarray:
    image_fft = fft2(image)
    result_fft = image_fft * psf_fft
    restored = ifft2(result_fft)
    return np.real(restored).astype(np.float32)


def _sparse_laplacian(shape: Tuple[int, int]):
    h, w = shape
    n = h * w

    main = -4.0 * np.ones(n, dtype=np.float32)
    if n > 1:
        east = np.ones(n - 1, dtype=np.float32)
        west = np.ones(n - 1, dtype=np.float32)
        east[np.arange(1, n) % w == 0] = 0.0
        west[np.arange(n - 1) % w == 0] = 0.0
    else:
        east = np.zeros(0, dtype=np.float32)
        west = np.zeros(0, dtype=np.float32)

    if h > 1:
        north = np.ones(n - w, dtype=np.float32)
        south = np.ones(n - w, dtype=np.float32)
    else:
        north = np.zeros(0, dtype=np.float32)
        south = np.zeros(0, dtype=np.float32)

    diagonals = [main, east, west, south, north]
    offsets = [0, 1, -1, w, -w]
    return diags(diagonals, offsets, shape=(n, n), dtype=np.float32, format="csr")


def _conjugate_gradient(lin_op, rhs, x0, maxiter, tol):
    try:
        return cg(
            lin_op,
            rhs,
            x0=x0,
            maxiter=maxiter,
            rtol=tol,
            atol=0.0,
        )
    except TypeError:
        return cg(
            lin_op,
            rhs,
            x0=x0,
            maxiter=maxiter,
            tol=tol,
        )


class TV1DeconvolutionAlgorithm(DeconvolutionAlgorithm):
    """Total variation deconvolution with a fixed blur kernel."""

    def __init__(
        self,
        kernel_size: KernelSpec = (9, 9),
        max_iter: int = 20,
        reg_param: float = 1e-2,
        cg_tol: float = 1e-4,
        cg_max_iter: int = 200,
        convergence_tol: float = 1e-3,
    ) -> None:
        super().__init__("PoissonTV-TV1")
        self.kernel_shape = _normalize_kernel_shape(kernel_size)
        self.max_iter = int(max_iter)
        self.reg_param = float(reg_param)
        self.cg_tol = float(cg_tol)
        self.cg_max_iter = int(cg_max_iter)
        self.convergence_tol = float(convergence_tol)
        self._kernel_override: Optional[np.ndarray] = None
        self._last_cg_info: Optional[int] = None
        self._last_kernel: Optional[np.ndarray] = None

    def _initial_kernel(self) -> np.ndarray:
        if self._kernel_override is not None:
            kernel = np.asarray(self._kernel_override, dtype=np.float32)
            if kernel.ndim != 2:
                raise ValueError("Initial kernel must be a 2-D array.")
            if kernel.sum() <= 0:
                raise ValueError("Initial kernel must have a positive sum.")
            self.kernel_shape = tuple(int(v) for v in kernel.shape)
            kernel = kernel / float(kernel.sum())
            return kernel.astype(np.float32)
        return _create_uniform_kernel(self.kernel_shape)

    def change_param(self, param: Any):
        if not isinstance(param, dict):
            return super().change_param(param)

        if "kernel_size" in param and param["kernel_size"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_size"])
        if "kernel_shape" in param and param["kernel_shape"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_shape"])
        if "max_iter" in param and param["max_iter"] is not None:
            self.max_iter = int(param["max_iter"])
        if "iterations" in param and param["iterations"] is not None:
            self.max_iter = int(param["iterations"])
        if "reg_param" in param and param["reg_param"] is not None:
            self.reg_param = float(param["reg_param"])
        if "cg_tol" in param and param["cg_tol"] is not None:
            self.cg_tol = float(param["cg_tol"])
        if "cg_max_iter" in param and param["cg_max_iter"] is not None:
            self.cg_max_iter = int(param["cg_max_iter"])
        if "convergence_tol" in param and param["convergence_tol"] is not None:
            self.convergence_tol = float(param["convergence_tol"])
        if "initial_kernel" in param and param["initial_kernel"] is not None:
            self._kernel_override = np.asarray(param["initial_kernel"], dtype=np.float32)
        if "kernel" in param and param["kernel"] is not None:
            self._kernel_override = np.asarray(param["kernel"], dtype=np.float32)

        return super().change_param(param)

    def process(self, image: np.ndarray):
        working, original_dtype, channels = _prepare_image(image)
        kernel = self._initial_kernel()
        laplacian = _sparse_laplacian(working.shape)
        psf_fft = _precompute_psf_fft(kernel, working.shape)

        current = working.copy()
        start = time()

        for _ in range(self.max_iter):
            def matvec(x_vec: np.ndarray) -> np.ndarray:
                vec = np.asarray(x_vec, dtype=np.float32)
                img_view = vec.reshape(working.shape)
                blurred = _apply_psf_fft_cached(img_view, psf_fft).ravel()
                if self.reg_param <= 0.0:
                    return blurred
                regularized = laplacian.dot(vec)
                return blurred + self.reg_param * regularized

            lin_op = LinearOperator(
                shape=(working.size, working.size),
                matvec=matvec,
                dtype=np.float32,
            )

            solution, info = _conjugate_gradient(
                lin_op,
                working.ravel(),
                current.ravel(),
                self.cg_max_iter,
                self.cg_tol,
            )
            self._last_cg_info = info
            if solution is None:
                break

            candidate = np.asarray(solution, dtype=np.float32).reshape(working.shape)
            delta = np.linalg.norm(candidate - current) / (np.linalg.norm(current) + 1e-8)
            current = candidate

            if delta < self.convergence_tol:
                break

        self.timer = time() - start
        self._last_kernel = kernel.astype(np.float32)

        restored = _restore_dtype(current, original_dtype, channels)
        return restored, self._last_kernel

    def get_param(self):
        return [
            ("kernel_shape", self.kernel_shape),
            ("max_iter", self.max_iter),
            ("reg_param", self.reg_param),
            ("cg_tol", self.cg_tol),
            ("cg_max_iter", self.cg_max_iter),
            ("convergence_tol", self.convergence_tol),
        ]


class TV2DeconvolutionAlgorithm(TV1DeconvolutionAlgorithm):
    """Alternating minimisation with total variation priors for image and kernel."""

    def __init__(
        self,
        kernel_size: KernelSpec = (9, 9),
        max_iter: int = 20,
        reg_param: float = 1e-2,
        cg_tol: float = 1e-4,
        cg_max_iter: int = 200,
        convergence_tol: float = 1e-3,
        gamma: float = 1e-2,
        minimize_max_iter: int = 30,
        minimize_ftol: float = 1e-2,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            max_iter=max_iter,
            reg_param=reg_param,
            cg_tol=cg_tol,
            cg_max_iter=cg_max_iter,
            convergence_tol=convergence_tol,
        )
        self.name = "PoissonTV-TV2"
        self.gamma = float(gamma)
        self.minimize_max_iter = int(minimize_max_iter)
        self.minimize_ftol = float(minimize_ftol)

    def change_param(self, param: Any):
        result = super().change_param(param)
        if isinstance(param, dict):
            if "gamma" in param and param["gamma"] is not None:
                self.gamma = float(param["gamma"])
            if "minimize_max_iter" in param and param["minimize_max_iter"] is not None:
                self.minimize_max_iter = int(param["minimize_max_iter"])
            if "minimize_ftol" in param and param["minimize_ftol"] is not None:
                self.minimize_ftol = float(param["minimize_ftol"])
        return result

    def process(self, image: np.ndarray):
        working, original_dtype, channels = _prepare_image(image)
        kernel = self._initial_kernel()
        laplacian = _sparse_laplacian(working.shape)

        current = working.copy()
        start = time()

        for _ in range(self.max_iter):
            psf_fft = _precompute_psf_fft(kernel, working.shape)

            def matvec(x_vec: np.ndarray) -> np.ndarray:
                vec = np.asarray(x_vec, dtype=np.float32)
                img_view = vec.reshape(working.shape)
                blurred = _apply_psf_fft_cached(img_view, psf_fft).ravel()
                if self.reg_param <= 0.0:
                    return blurred
                regularized = laplacian.dot(vec)
                return blurred + self.reg_param * regularized

            lin_op = LinearOperator(
                shape=(working.size, working.size),
                matvec=matvec,
                dtype=np.float32,
            )

            solution, info = _conjugate_gradient(
                lin_op,
                working.ravel(),
                current.ravel(),
                self.cg_max_iter,
                self.cg_tol,
            )
            self._last_cg_info = info
            if solution is None:
                break

            candidate = np.asarray(solution, dtype=np.float32).reshape(working.shape)
            delta = np.linalg.norm(candidate - current) / (np.linalg.norm(current) + 1e-8)
            current = candidate

            def objective(h_flat: np.ndarray) -> float:
                h = np.asarray(h_flat, dtype=np.float32).reshape(self.kernel_shape)
                h = np.clip(h, 0.0, None)
                total = float(h.sum())
                if total <= 0:
                    h = _create_uniform_kernel(self.kernel_shape)
                else:
                    h = h / total
                psf_fft_local = _precompute_psf_fft(h, working.shape)
                prediction = _apply_psf_fft_cached(current, psf_fft_local)
                data_term = np.sum((working - prediction) ** 2, dtype=np.float64)
                grad_y, grad_x = np.gradient(h)
                smoothness = self.gamma * (np.sum(grad_y ** 2) + np.sum(grad_x ** 2))
                return float(data_term + smoothness)

            result = minimize(
                objective,
                kernel.ravel(),
                method="L-BFGS-B",
                options={"maxiter": self.minimize_max_iter, "ftol": self.minimize_ftol},
            )
            kernel_candidate = np.asarray(result.x, dtype=np.float32).reshape(self.kernel_shape)
            kernel_candidate = np.clip(kernel_candidate, 0.0, None)
            total = float(kernel_candidate.sum())
            if total <= 0:
                kernel = _create_uniform_kernel(self.kernel_shape)
            else:
                kernel = kernel_candidate / total

            if delta < self.convergence_tol:
                break

        self.timer = time() - start
        self._last_kernel = kernel.astype(np.float32)

        restored = _restore_dtype(current, original_dtype, channels)
        return restored, self._last_kernel

    def get_param(self):
        base = super().get_param()
        base.extend(
            [
                ("gamma", self.gamma),
                ("minimize_max_iter", self.minimize_max_iter),
                ("minimize_ftol", self.minimize_ftol),
            ]
        )
        return base


class _2924878374VariationalBayesianBlindDeconvolutionUsingATotalVariationPrior(DeconvolutionAlgorithm):
    """Facade that exposes TV1/TV2 variants via the common framework interface."""

    def __init__(
        self,
        variant: str = "tv2",
        kernel_size: KernelSpec = (9, 9),
        max_iter: int = 20,
        reg_param: float = 1e-2,
        gamma: float = 1e-2,
        cg_tol: float = 1e-4,
        cg_max_iter: int = 200,
        convergence_tol: float = 1e-3,
        minimize_max_iter: int = 30,
        minimize_ftol: float = 1e-2,
    ) -> None:
        super().__init__("PoissonTV")
        self.variant = str(variant)
        self.kernel_size = _normalize_kernel_shape(kernel_size)
        self.max_iter = int(max_iter)
        self.reg_param = float(reg_param)
        self.gamma = float(gamma)
        self.cg_tol = float(cg_tol)
        self.cg_max_iter = int(cg_max_iter)
        self.convergence_tol = float(convergence_tol)
        self.minimize_max_iter = int(minimize_max_iter)
        self.minimize_ftol = float(minimize_ftol)
        self._tv1 = TV1DeconvolutionAlgorithm(
            kernel_size=self.kernel_size,
            max_iter=self.max_iter,
            reg_param=self.reg_param,
            cg_tol=self.cg_tol,
            cg_max_iter=self.cg_max_iter,
            convergence_tol=self.convergence_tol,
        )
        self._tv2 = TV2DeconvolutionAlgorithm(
            kernel_size=self.kernel_size,
            max_iter=self.max_iter,
            reg_param=self.reg_param,
            cg_tol=self.cg_tol,
            cg_max_iter=self.cg_max_iter,
            convergence_tol=self.convergence_tol,
            gamma=self.gamma,
            minimize_max_iter=self.minimize_max_iter,
            minimize_ftol=self.minimize_ftol,
        )

    def _sync_algorithms(self):
        shared = {
            "kernel_size": self.kernel_size,
            "max_iter": self.max_iter,
            "reg_param": self.reg_param,
            "cg_tol": self.cg_tol,
            "cg_max_iter": self.cg_max_iter,
            "convergence_tol": self.convergence_tol,
        }
        self._tv1.change_param(shared)
        tv2_updates = dict(shared)
        tv2_updates.update(
            {
                "gamma": self.gamma,
                "minimize_max_iter": self.minimize_max_iter,
                "minimize_ftol": self.minimize_ftol,
            }
        )
        self._tv2.change_param(tv2_updates)

    def change_param(self, param: Any):
        if not isinstance(param, dict):
            return super().change_param(param)

        if "variant" in param and param["variant"] is not None:
            self.variant = str(param["variant"])
        if "kernel_size" in param and param["kernel_size"] is not None:
            self.kernel_size = _normalize_kernel_shape(param["kernel_size"])
        if "max_iter" in param and param["max_iter"] is not None:
            self.max_iter = int(param["max_iter"])
        if "reg_param" in param and param["reg_param"] is not None:
            self.reg_param = float(param["reg_param"])
        if "gamma" in param and param["gamma"] is not None:
            self.gamma = float(param["gamma"])
        if "cg_tol" in param and param["cg_tol"] is not None:
            self.cg_tol = float(param["cg_tol"])
        if "cg_max_iter" in param and param["cg_max_iter"] is not None:
            self.cg_max_iter = int(param["cg_max_iter"])
        if "convergence_tol" in param and param["convergence_tol"] is not None:
            self.convergence_tol = float(param["convergence_tol"])
        if "minimize_max_iter" in param and param["minimize_max_iter"] is not None:
            self.minimize_max_iter = int(param["minimize_max_iter"])
        if "minimize_ftol" in param and param["minimize_ftol"] is not None:
            self.minimize_ftol = float(param["minimize_ftol"])

        self._sync_algorithms()
        return super().change_param(param)

    def process(self, image: np.ndarray):
        variant = (self.variant or "tv2").lower()
        if variant == "tv1":
            return self._tv1.process(image)
        if variant == "tv2":
            return self._tv2.process(image)
        raise ValueError(f"Unsupported PoissonTV variant: {self.variant!r}")

    def get_param(self):
        return [
            ("variant", self.variant),
            ("kernel_size", self.kernel_size),
            ("max_iter", self.max_iter),
            ("reg_param", self.reg_param),
            ("gamma", self.gamma),
            ("cg_tol", self.cg_tol),
            ("cg_max_iter", self.cg_max_iter),
            ("convergence_tol", self.convergence_tol),
            ("minimize_max_iter", self.minimize_max_iter),
            ("minimize_ftol", self.minimize_ftol),
        ]



__all__ = [
    "_2924878374VariationalBayesianBlindDeconvolutionUsingATotalVariationPrior",
    "TV1DeconvolutionAlgorithm",
    "TV2DeconvolutionAlgorithm",
]

__all__ = ["TV1DeconvolutionAlgorithm", "_2924878374VariationalBayesianBlindDeconvolutionUsingATotalVariationPrior"]
