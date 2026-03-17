"""
SDQM — Blind deconvolution via Steepest Descent on a Quotient Manifold.

Based on:
  Wen Huang, "Blind Deconvolution by a Steepest Descent Algorithm
  on a Quotient Manifold", SIIMS 2018.

Ported from ROPTLIB C++ (CFR2BlindDecon2D + CFixedRank2Factors).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import build_B_kernel, build_C_wavelet, build_C_identity, next_power_of_2
from .solvers import CFR2BlindDeconProblem, solve_lrbfgs, solve_rsd

# ── Framework base class import ─────────────────────────────────────────
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path


_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm
# ─────────────────────────────────────────────────────────────────────────


class SDQM(DeconvolutionAlgorithm):
    """Blind deconvolution on the quotient manifold C_*^{K×r} × C_*^{N×r} / GL(r).

    Parameters
    ----------
    kernel_shape : tuple (kh, kw)
        Expected size of the blur kernel.
    max_iter : int
        Maximum solver iterations.
    solver : str
        'lrbfgs' or 'rsd'.
    r : int
        Rank (1 for standard blind deconvolution).
    rho, d, mu : float
        Penalty / incoherence parameters  (set rho=0 to disable).
    use_wavelet : bool
        If True, use Haar-wavelet basis for C; otherwise identity.
    N_ratio : float
        Fraction of L used as wavelet-coefficient count  (N = int(N_ratio * L)).
    tol : float
        Relative gradient tolerance for stopping.
    memory : int
        Number of L-BFGS pairs  (only for 'lrbfgs').
    verbose : bool
        Print solver progress.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int] = (15, 15),
        max_iter: int = 300,
        solver: str = 'lrbfgs',
        r: int = 1,
        rho: float = 0.0,
        d: float = 1.0,
        mu: float = 1.0,
        use_wavelet: bool = True,
        N_ratio: float = 1.0,
        tol: float = 1e-6,
        memory: int = 4,
        verbose: bool = False,
    ):
        super().__init__(name='SDQM')

        self.kernel_shape = kernel_shape
        self.max_iter = max_iter
        self.solver = solver
        self.r = r
        self.rho = rho
        self.d = d
        self.mu = mu
        self.use_wavelet = use_wavelet
        self.N_ratio = N_ratio
        self.tol = tol
        self.memory = memory
        self.verbose = verbose

        self.history: Dict[str, list] = {'f': [], 'grad_norm': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run blind deconvolution.

        Parameters
        ----------
        image : np.ndarray  (H, W), dtype int/float, range [0, 255]
            Blurred grayscale image.

        Returns
        -------
        (restored_image, estimated_kernel)
            restored_image : int16 array [0, 255]
            estimated_kernel : float64 array (kh, kw), sums to 1
        """
        start_time = time.time()

        if image.ndim != 2:
            raise ValueError("Only grayscale images are supported")

        img_h, img_w = image.shape
        image_shape = (img_h, img_w)
        L = img_h * img_w

        # Normalize to [0, 1]
        blurred = image.astype(np.float64) / 255.0

        # ── Build observation y = FFT2(blurred image), flattened ──
        y = np.fft.fft2(blurred).ravel()

        # ── Build B operator (kernel subspace) ──
        kh, kw = self.kernel_shape
        B_op, BH_op, K = build_B_kernel((kh, kw), image_shape)

        # ── Build C operator (image subspace) ──
        if self.use_wavelet:
            N = max(1, int(self.N_ratio * L))
            C_op, CH_op = build_C_wavelet(blurred, N)
        else:
            N = L
            C_op, CH_op = build_C_identity(image_shape)

        # ── Initial point on the manifold ──
        G0, H0 = SDQM._spectral_init(K, N, self.r, L, y, B_op, BH_op, C_op, CH_op, img_h, img_w) #?

        # ── Construct problem ──
        prob = CFR2BlindDeconProblem(
            y=y, B_op=B_op, BH_op=BH_op,
            C_op=C_op, CH_op=CH_op,
            K=K, N=N, L=L, r=self.r,
            rho=self.rho, d=self.d, mu=self.mu,
            image_shape=image_shape,
        )

        # ── Solve ──
        if self.solver == 'lrbfgs':
            G_opt, H_opt, hist = solve_lrbfgs(
                prob, G0, H0,
                max_iter=self.max_iter,
                tol=self.tol,
                memory=self.memory,
                verbose=self.verbose,
            )
        else:
            G_opt, H_opt, hist = solve_rsd(
                prob, G0, H0,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
            )

        self.history = hist

        # ── Extract kernel ──
        kernel = self._extract_kernel(G_opt, H_opt, B_op, image_shape)

        # ── Restore image via Wiener deconvolution with estimated kernel ──
        restored = self._wiener_restore(blurred, kernel)

        elapsed = time.time() - start_time
        self.hyperparams = {
            'time': elapsed,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'final_f': hist['f'][-1] if hist['f'] else None,
            'final_grad_norm': hist['grad_norm'][-1] if hist['grad_norm'] else None,
            'iterations_run': len(hist['f']) - 1,
        }

        x_final = restored * 255.0
        x_final = np.clip(x_final, 0, 255)
        x_final = np.round(x_final).astype(np.int16)
        return x_final, kernel

    # === НОВЫЙ МЕТОД (добавь в класс SDQM) ===
    @staticmethod
    def _spectral_init(K, N, r, L, y, B_op, BH_op, C_op, CH_op, H, W):
        """Algorithm 1 из статьи"""
        A_star_y = np.zeros((K, N), dtype=np.complex128)
        for i in range(K):
            e_i = np.zeros(K, dtype=np.complex128)
            e_i[i] = 1.0
            b_i = np.fft.fft2(B_op(e_i).reshape(H, W)).ravel()          # без /sqrt(L)
            for j in range(N):
                e_j = np.zeros(N, dtype=np.complex128)
                e_j[j] = 1.0
                c_j = (np.fft.ifft2(C_op(e_j).reshape(H, W)) * L).ravel()  # ×L
                A_star_y[i, j] = np.vdot(b_i, y * c_j.conj())

        U, S, Vh = np.linalg.svd(A_star_y, full_matrices=False)
        d = np.abs(S[0])
        scale = np.sqrt(d)

        G0 = U[:, :r] * scale
        H0 = Vh[:r, :].conj().T * scale
        return G0, H0

    # ── Kernel extraction ────────────────────────────────────────────
    def _extract_kernel(self, G, H, B_op, image_shape):
        """Extract real-space kernel from the manifold solution.

        The lifted matrix is  X = G H^*  (K x N with rank r).
        For r = 1, the first column of G (up to scale) gives the kernel
        coefficients in the B-subspace.

        The kernel in spatial domain:  h = B @ g  (zero-padded) →
        the non-zero kh×kw block is the estimated kernel.
        """
        kh, kw = self.kernel_shape

        # g = first column of G (for r=1 this IS G)
        g = G[:, 0]

        # Spatial-domain zero-padded kernel
        h_full = B_op(g).reshape(image_shape)

        # Extract the kh x kw block
        kernel = np.real(h_full[:kh, :kw])

        # Make non-negative and normalize
        kernel = np.abs(kernel)
        s = kernel.sum()
        if s > 1e-15:
            kernel /= s
        return kernel

    # ── Image restoration using estimated kernel ─────────────────────
    @staticmethod
    def _wiener_restore(blurred, kernel, eps=1e-3):
        """Simple Wiener deconvolution to produce the restored image.

        Parameters
        ----------
        blurred : (H, W) float64, range [0, 1]
        kernel  : (kh, kw) float64, sums to ~1
        eps     : regularization

        Returns
        -------
        restored : (H, W) float64, range [0, 1]
        """
        H, W = blurred.shape
        kh, kw = kernel.shape

        # Zero-pad kernel to image size (top-left corner for circular conv)
        kernel_padded = np.zeros((H, W), dtype=np.float64)
        kernel_padded[:kh, :kw] = kernel

        K_fft = np.fft.fft2(kernel_padded)
        B_fft = np.fft.fft2(blurred)

        # Wiener filter:  H^* / (|H|^2 + eps)
        K_conj = np.conj(K_fft)
        wiener = K_conj / (np.abs(K_fft)**2 + eps)
        restored = np.real(np.fft.ifft2(B_fft * wiener))
        return np.clip(restored, 0.0, 1.0)

    # ── Interface methods ────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('max_iter', self.max_iter),
            ('solver', self.solver),
            ('r', self.r),
            ('rho', self.rho),
            ('d', self.d),
            ('mu', self.mu),
            ('use_wavelet', self.use_wavelet),
            ('N_ratio', self.N_ratio),
            ('tol', self.tol),
            ('memory', self.memory),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams


# ── Convenience wrapper for run_algorithm ────────────────────────────

def run_algorithm(image: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Functional wrapper expected by __init__.py.

    Parameters
    ----------
    image : np.ndarray
        Blurred grayscale image (0-255).
    **kwargs
        Forwarded to SDQM constructor.

    Returns
    -------
    (restored_image, estimated_kernel)
    """
    algo = SDQM(**kwargs)
    return algo.process(image)
