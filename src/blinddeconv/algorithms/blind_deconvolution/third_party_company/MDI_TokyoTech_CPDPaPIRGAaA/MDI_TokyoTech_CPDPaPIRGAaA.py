import numpy as np
import scipy.io
import time
import sys
import torch
from typing import Any, Dict, List, Tuple
from pathlib import Path

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            return start.parent.parent
        path = path.parent
    return path

_CURRENT_FILE   = Path(__file__).resolve()
_ALGORITHM_DIR  = _CURRENT_FILE.parent
_PROJECT_ROOT   = _find_project_root(_CURRENT_FILE)
_SRC_DIR        = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

# Ensure local imports work
for _p in [_ALGORITHM_DIR, _ALGORITHMS_DIR, _SRC_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from blinddeconv.algorithms.base import DeconvolutionAlgorithm

try:
    # Importing operators exactly as used in iteration.py
    from pnppds.operators import (
        get_blur_operator,
        get_adj_blur_operator,
        proj_l2_ball,
        proj_C,
        prox_GKL,
    )
    # Importing Denoiser class
    from models.denoiser import Denoiser
except ImportError as e:
    raise ImportError(
        f"Could not import PnP-PDS modules. Ensure 'pnppds' and 'models' folders "
        f"are present in {_ALGORITHM_DIR}. Error: {e}"
    )

# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class MDI_TokyoTech_CPDPaPIRGAaA(DeconvolutionAlgorithm):
    """
    Non-blind image deconvolution via Plug-and-Play Primal-Dual Splitting.
    Wraps the logic from 'iteration.py' in the MDI-TokyoTech repository.
    """

    def __init__(
        self,
        method: str = 'Gaussian-PnPPDS',  # 'Gaussian-PnPPDS' or 'Poisson-PnPPDS'
        max_iter: int = 1200,             # Default from main_gaussian.py (MAX_ITER_BLUR)
        gamma1: float = 0.5,              # Default from main_gaussian.py
        gamma2: float = 0.99,             # Default from main_gaussian.py
        alpha: float = 0.82,              # Default from main_gaussian.py (alpha_n)
        noise_sigma: float = 0.01,        # Gaussian noise level (gaussian_nl)
        my_lambda: float = 0.00125,       # Poisson balancing parameter (lambda_bl from main_poisson.py)
        poisson_eta: float = 100.0,       # Poisson scaling factor
        architecture: str = 'DnCNN_nobn_nch_1_nlev_0.01', # Using 1-channel model for grayscale
        kernel_name: str = 'blur_1',
        verbose: bool = False,
    ) -> None:
        super().__init__(name='PnP-PDS')

        self.method = method
        self.max_iter = max_iter
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.alpha = alpha
        self.noise_sigma = noise_sigma
        self.my_lambda = my_lambda
        self.poisson_eta = poisson_eta
        self.architecture = architecture
        self.kernel_name = kernel_name
        self.verbose = verbose

        self.history: Dict[str, list] = {'convergence': []}
        self.hyperparams: Dict[str, Any] = {}

    @property
    def _path_kernel(self) -> str:
        return str(_ALGORITHM_DIR / 'blur_models' / (self.kernel_name + '.mat'))

    @property
    def _path_prox(self) -> str:
        return str(_ALGORITHM_DIR / 'nn' / (self.architecture + '.pth'))

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # 1. Preparation (Matching test_pnppds.py logic)
        # ------------------------------------------------------------
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0
        
        # Load Kernel
        try:
            kernel_data = scipy.io.loadmat(self._path_kernel)
            # Handle variable names in .mat files
            kernel_key = 'blur' if 'blur' in kernel_data else 'k'
            kernel = np.array(kernel_data[kernel_key])
        except FileNotFoundError:
            raise RuntimeError(f"Kernel file not found: {self._path_kernel}")

        # Define Operators (operators.py)
        # Using lambda wrappers exactly as iteration.py expects callable logic
        phi     = lambda x: get_blur_operator(x, kernel)
        adj_phi = lambda x: get_adj_blur_operator(x, kernel)

        # 2. Instantiate Denoiser (denoiser.py)
        # ------------------------------------------------------------
        # HACK: Fix for PyTorch 2.6+ blocking legacy checkpoints.
        # denoiser.py calls torch.load internally. We patch torch.load temporarily.
        _original_load = torch.load
        def _unsafe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)

        try:
            torch.load = _unsafe_load
            # Initialize Denoiser. 
            # Note: denoiser.py handles GPU/CPU internal logic based on availability.
            # We assume grayscale (ch=1) as per standard deblurring tasks here.
            denoiser_J = Denoiser(file_name=self._path_prox, ch=1)
        finally:
            torch.load = _original_load

        # 3. Initialization (Matching test_pnppds.py)
        # ------------------------------------------------------------
        x_obsrv = y
        
        # In test_pnppds.py:
        # if(poisson_noise): x_0 = x_0 / poisson_eta
        if self.method == 'Poisson-PnPPDS':
            x_n = np.copy(x_obsrv) / self.poisson_eta
        else:
            x_n = np.copy(x_obsrv)

        # Dual variables initialization (iteration.py)
        y_n = np.zeros(x_n.shape)
        y2_n = np.zeros(x_n.shape)
        
        convergence = []
        if self.verbose:
            print(f"[{self.name}] Start {self.method}. MaxIter={self.max_iter}")

        # 4. Main Loop (Matching iteration.py verbatim)
        # ------------------------------------------------------------
        for i in range(self.max_iter):
            x_prev = x_n.copy()

            if self.method == 'Gaussian-PnPPDS':
                # Line-by-line from iteration.py (Gaussian block)
                
                # 1. Primal Update via Denoiser
                # iteration.py: x_n = denoiser_J.denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n))
                # Note: denoiser.py expects numpy, returns numpy.
                x_n = denoiser_J.denoise(x_n - self.gamma1 * (adj_phi(y_n) + y2_n))
                x_n = x_n.squeeze() # Safety for dimension handling

                # 2. Dual Update (y_n)
                # iteration.py: y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
                y_n = y_n + self.gamma2 * phi(2 * x_n - x_prev)

                # 3. Dual Update (y2_n)
                # iteration.py: y2_n = y2_n + gamma2 * (2 * x_n - x_prev)
                y2_n = y2_n + self.gamma2 * (2 * x_n - x_prev)

                # 4. Proximal/Projection for y_n (L2 Ball)
                # iteration.py: y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
                y_n = y_n - self.gamma2 * proj_l2_ball(
                    y_n / self.gamma2, 
                    self.alpha,         # alpha_n
                    self.noise_sigma,   # gaussian_nl
                    0,                  # sp_nl (salt & pepper, usually 0)
                    x_obsrv, 
                    1                   # r (sampling rate, 1 for blur)
                )

                # 5. Proximal/Projection for y2_n (Box [0,1])
                # iteration.py: y2_n = y2_n - gamma2 * op.proj_C(y2_n / gamma2)
                y2_n = y2_n - self.gamma2 * proj_C(y2_n / self.gamma2)

            elif self.method == 'Poisson-PnPPDS':
                # Line-by-line from iteration.py (Poisson block)

                # 1. Primal Update
                # iteration.py: x_n = denoiser_J.denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n))
                x_n = denoiser_J.denoise(x_n - self.gamma1 * (adj_phi(y_n) + y2_n))
                x_n = x_n.squeeze()

                # 2. Dual Update (y_n)
                y_n = y_n + self.gamma2 * phi(2 * x_n - x_prev)

                # 3. Dual Update (y2_n)
                y2_n = y2_n + self.gamma2 * (2 * x_n - x_prev)

                # 4. Proximal for y_n (Generalized KL)
                # iteration.py: y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_eta, x_obsrv)
                y_n = y_n - self.gamma2 * prox_GKL(
                    y_n / self.gamma2, 
                    self.my_lambda / self.gamma2, 
                    self.poisson_eta, 
                    x_obsrv
                )

                # 5. Proximal for y2_n
                y2_n = y2_n - self.gamma2 * proj_C(y2_n / self.gamma2)
            
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Monitoring
            if i % 10 == 0 or i == self.max_iter - 1:
                rel_change = np.linalg.norm((x_n - x_prev).flatten()) / (np.linalg.norm(x_prev.flatten()) + 1e-9)
                convergence.append(rel_change)
                if self.verbose and i % 100 == 0:
                    print(f"Iter {i}/{self.max_iter}, RelChange: {rel_change:.2e}")

        # 5. Finalize
        # ------------------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start_time

        self.history['convergence'] = convergence
        self.hyperparams = {
            'method': self.method,
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'alpha': self.alpha,
            'noise_sigma': self.noise_sigma,
            'elapsed': elapsed
        }

        # Convert back to uint8 image
        restored = np.clip(x_n, 0.0, 1.0) * 255.0
        restored = np.round(restored).astype(np.int16)

        return restored, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('method', self.method),
            ('gamma1', self.gamma1),
            ('noise_sigma', self.noise_sigma),
            ('kernel_name', self.kernel_name),
            ('alpha', self.alpha),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams

def run_algorithm(g, **kwargs):
    algo = MDI_TokyoTech_CPDPaPIRGAaA(**kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.get_hyperparams(), algo.get_history()