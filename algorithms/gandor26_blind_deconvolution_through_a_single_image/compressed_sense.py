import numpy as np


class CompressedSense:
    """Solve min ||Ax - y||_2^2 + lambda * ||x||_1 via ISTA."""

    def __init__(
        self,
        kernel_shape,
        image_shape,
        lmbda: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-5,
    ) -> None:
        self.kernel_h, self.kernel_w = kernel_shape
        self.image_h, self.image_w = image_shape
        self.kernel_size = self.kernel_h * self.kernel_w
        self.image_size = self.image_h * self.image_w
        self.lambda_ = float(lmbda)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.A = np.zeros((self.image_size, self.kernel_size), dtype=np.float64)
        self.y = np.zeros((self.image_size, 1), dtype=np.float64)
        self.x = np.zeros((self.kernel_size, 1), dtype=np.float64)

    def mat2vector(self, original: np.ndarray, blurred: np.ndarray) -> None:
        if original.shape != (self.image_h, self.image_w):
            raise ValueError("Unexpected original image shape")
        if blurred.shape != (self.image_h, self.image_w):
            raise ValueError("Unexpected blurred image shape")

        rows = []
        for center_y in range(self.image_h):
            for center_x in range(self.image_w):
                patch = np.zeros(self.kernel_size, dtype=np.float64)
                idx = 0
                for ky in range(self.kernel_h):
                    for kx in range(self.kernel_w):
                        yy = center_y + ky - self.kernel_h // 2
                        xx = center_x + kx - self.kernel_w // 2
                        if yy < 0 or yy >= self.image_h or xx < 0 or xx >= self.image_w:
                            val = 0.0
                        else:
                            val = original[yy, xx]
                        patch[idx] = val
                        idx += 1
                rows.append(patch)
        self.A = np.asarray(rows, dtype=np.float64)
        self.y = blurred.reshape(-1, 1).astype(np.float64, copy=False)

    def solve(self) -> None:
        if self.A.size == 0:
            raise RuntimeError("CompressedSense: matrix A is not initialised")

        x = np.zeros_like(self.x)
        ATA = self.A.T @ self.A
        eigvals = np.linalg.eigvalsh(ATA)
        lipschitz = float(eigvals.max()) if eigvals.size else 1.0
        if lipschitz <= 0:
            lipschitz = 1.0
        step = 1.0 / lipschitz
        ATy = self.A.T @ self.y

        for _ in range(self.max_iter):
            grad = ATA @ x - ATy
            x_new = self._soft_threshold(x - step * grad, self.lambda_ * step)
            if np.linalg.norm(x_new - x) < self.tol:
                x = x_new
                break
            x = x_new
        self.x = x

    def vector2mat(self) -> np.ndarray:
        kernel = self.x.reshape((self.kernel_h, self.kernel_w))
        kernel = np.maximum(kernel, 0.0)
        denom = kernel.sum()
        if denom > 0:
            kernel /= denom
        return kernel

    @staticmethod
    def _soft_threshold(x: np.ndarray, thr: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


__all__ = ["CompressedSense"]
