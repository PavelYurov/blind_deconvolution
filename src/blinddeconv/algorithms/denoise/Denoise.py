"""
Denoise Wrapper — Image Denoising Algorithm.

Pipeline:
    1. Normalise input to float64 [0, 1].
    2. Trim image to odd dimensions.
    3. Noise estimation via Chen-Pyatykh PCA or Chen ICCV2015 method.
    4. Apply selected denoiser with adaptive parameters from noise estimate.
    5. Return denoised image (int16, [0, 255]) and point-like kernel.

Supported denoisers:
    - bm3d       : Block Matching 3D.
    - guided     : Guided Filter.
    - bilateral  : Bilateral Filter.
    - nlm        : Non-Local Means.
    - tv         : Total Variation (Chambolle).
    - vst+bm3d   : Variance-Stabilized Transform + BM3D (Poisson-Gaussian).
    - act        : Adaptive Curvelet Thresholding.

Noise estimation methods:
    - chen       : Wavelet-based eigenvalue method (ICCV 2015).
    - pca        : PCA + VST + Kurtosis (TIP 2013).
    - none       : No noise estimation; use default parameters.

References:
    Chen G., Zhu F., Heng P.A.:
        "An Efficient Statistical Method for Image Noise Level Estimation",
        ICCV 2015.
    Pyatykh S., Hesser J., Zheng L.:
        "Image Noise Level Estimation by Principal Component Analysis",
        IEEE Trans. Image Process., vol. 22, no. 12, pp. 4874–4883, 2013.
    Dabov K., Foi A., Katkovnik V., Egiazarian K.:
        "Image Denoising by Sparse 3D Transform-Domain Collaborative
         Filtering", IEEE Trans. Image Process., vol. 16, no. 8,
         pp. 2080–2095, Aug. 2007.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

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

def make_size_odd(image):
    """Trim image to odd spatial dimensions."""
    h, w = image.shape[:2]
    h = h if h % 2 == 1 else h - 1
    w = w if w % 2 == 1 else w - 1
    return image[:h, :w]

class DenoiseWrapper(DeconvolutionAlgorithm):
    """
    Pure denoising wrapper algorithm.

    Applies a single denoiser with adaptive parameters estimated from
    the noisy image itself. The output PSF is a point (Dirac delta).

    Parameters
    ----------
    method : str
        Denoiser to use: 'bm3d', 'guided', 'bilateral', 'nlm', 'tv',
        'vst+bm3d', or 'act'.
    noise_estimation : str
        Noise σ estimation method: 'chen', 'pca', or 'none'.
        When 'none', uses default parameters for the denoiser.
    denoiser_params : dict or None
        Denoiser-specific parameters (optional).
        Examples:
            {'weight': 0.1} for TV.
            {'h': 0.1, 'patch_size': 5, 'patch_distance': 7} for NLM.
            {'d': 5, 'sigma_color': 0.1, 'sigma_space': 5.0} for bilateral.
            {'radius': 4, 'eps': 0.001} for guided.
            {'sigma': 0.05} for BM3D.
            {'threshold_setting': 's'} for ACT.
    noise_estimation_params : dict or None
        Parameters for noise estimation method (rarely needed).
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        method: str = 'bm3d',
        noise_estimation: str = 'chen',
        denoiser_params: dict = None,
        noise_estimation_params: dict = None,
        verbose: bool = False,
    ):
        super().__init__(name='DenoiseWrapper')

        valid_methods = {'bm3d', 'guided', 'bilateral', 'nlm', 'tv',
                         'vst+bm3d', 'act', 'median'}
        method_lower = str(method).lower().strip()
        if method_lower not in valid_methods:
            raise ValueError(
                f"Denoiser method='{method}' not supported. "
                f"Choose from: {sorted(valid_methods)}")
        self.method = method_lower

        valid_noise_est = {'chen', 'pca', 'none'}
        noise_est_lower = str(noise_estimation).lower().strip()
        if noise_est_lower not in valid_noise_est:
            raise ValueError(
                f"noise_estimation='{noise_estimation}' not supported. "
                f"Choose from: {sorted(valid_noise_est)}")
        self.noise_estimation = noise_est_lower

        self.denoiser_params = dict(denoiser_params or {})
        self.noise_estimation_params = dict(noise_estimation_params or {})
        self.verbose = bool(verbose)

        # Tracking
        self.history: Dict[str, list] = {}
        self.hyperparams: Dict[str, Any] = {}


    def _estimate_noise(self, image):
        """Estimate noise level σ from image.

        Returns
        -------
        dict with keys:
            'method': str (estimation method used)
            'sigma_norm': float (σ in [0, 1] scale)
            'sigma_pix': float (σ in [0, 255] scale)
            additional keys depend on the method
        """
        if self.noise_estimation == 'none':
            return None

        try:
            if self.noise_estimation == 'chen':
                
                from blinddeconv.algorithms.mod_denoise.chen_noise_estimate import estimate_noise_level

                sigma_norm = estimate_noise_level(
                    image,
                    pch_size=self.noise_estimation_params.get('pch_size', 8),
                )
                return {
                    'method': 'chen',
                    'sigma_norm': float(sigma_norm),
                    'sigma_pix': float(sigma_norm * 255.0),
                }

            elif self.noise_estimation == 'pca':
                
                from blinddeconv.algorithms.mod_denoise.pyatykh_noise_reconstruction import estimate_noise_params

                result = estimate_noise_params(image)
                result['method'] = 'pca'
                if 'sigma_norm' not in result:
                    result['sigma_norm'] = result.get('sigma', 0.0)
                return result

            return None

        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] Warning: noise estimation failed: {e}")
            return None


    def _apply_tv(self, image, noise_info):
        """Total Variation denoising (Chambolle)."""
        from skimage.restoration import denoise_tv_chambolle

        p = dict(self.denoiser_params)
        sigma = (noise_info.get('sigma_norm') if noise_info else None)

        weight = p.get('weight', None)
        if weight is None:
            weight = max(0.01, sigma * 2) if sigma else 0.1

        kwargs = dict(weight=weight, eps=p.get('eps', 0.002))
        try:
            return denoise_tv_chambolle(image, **kwargs,
                                        max_num_iter=p.get('max_num_iter', 200))
        except TypeError:
            return denoise_tv_chambolle(image, **kwargs,
                                        n_iter_max=p.get('max_num_iter', 200))

    def _apply_nlm(self, image, noise_info):
        """Non-Local Means denoising."""
        from skimage.restoration import denoise_nl_means, estimate_sigma

        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        if sigma is None:
            sigma = float(np.mean(estimate_sigma(image)))

        h = p.get('h', 0.8 * sigma)
        patch_size = p.get('patch_size', 5)
        patch_distance = p.get('patch_distance', 7)

        return denoise_nl_means(
            image, h=h, patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=p.get('fast_mode', True)
        )

    def _apply_bilateral(self, image, noise_info):
        """Bilateral filtering."""
        import cv2

        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        d = p.get('d', 5)
        sigma_color = p.get('sigma_color', sigma if sigma else 0.1)
        sigma_space = p.get('sigma_space', 5.0)

        result = cv2.bilateralFilter(
            image.astype(np.float32), d, float(sigma_color),
            float(sigma_space)
        )
        return result.astype(np.float64)

    def _apply_guided(self, image, noise_info):
        """Guided filtering."""
        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        radius = p.get('radius', 4)
        eps = p.get('eps', sigma ** 2 * 4 if sigma else 0.01)

        return self._guided_filter_impl(image, image, radius, eps)

    def _guided_filter_impl(self, I, p, r, eps):
        """Guided filter implementation (from LIP).

        Parameters
        ----------
        I : ndarray (H, W)
            Guidance image.
        p : ndarray (H, W)
            Input image to filter.
        r : int
            Filter radius.
        eps : float
            Regularization parameter.

        Returns
        -------
        q : ndarray (H, W)
            Filtered image.
        """
        from scipy.ndimage import uniform_filter

        I = I.astype(np.float64)
        p = p.astype(np.float64)
        H, W = I.shape

        mean_I = uniform_filter(I, size=2*r+1, mode='reflect')
        mean_p = uniform_filter(p, size=2*r+1, mode='reflect')
        mean_Ip = uniform_filter(I * p, size=2*r+1, mode='reflect')
        mean_II = uniform_filter(I * I, size=2*r+1, mode='reflect')

        var_I = mean_II - mean_I ** 2
        cov_Ip = mean_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = uniform_filter(a, size=2*r+1, mode='reflect')
        mean_b = uniform_filter(b, size=2*r+1, mode='reflect')

        q = mean_a * I + mean_b
        return np.clip(q, 0, 1)

    def _apply_bm3d(self, image, noise_info):
        """BM3D denoising."""
        import bm3d

        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        sigma_psd = p.get('sigma', sigma if sigma else 0.05)

        stage_arg = p.get('stage_arg', bm3d.BM3DStages.ALL_STAGES)
        if isinstance(stage_arg, str):
            _map = {
                'all': bm3d.BM3DStages.ALL_STAGES,
                'ht': bm3d.BM3DStages.HARD_THRESHOLDING,
                'hard': bm3d.BM3DStages.HARD_THRESHOLDING,
                'wiener': bm3d.BM3DStages.WIENER_FILTERING,
            }
            stage_arg = _map.get(stage_arg.lower(), bm3d.BM3DStages.ALL_STAGES)

        return bm3d.bm3d(image, sigma_psd=sigma_psd, stage_arg=stage_arg)

    def _apply_vst_bm3d(self, image, noise_info):
        """VST + BM3D denoising (Poisson-Gaussian noise)."""
        from blinddeconv.algorithms.mod_denoise.vst import vst_bm3d_denoise

        p = dict(self.denoiser_params)

        result, _ = vst_bm3d_denoise(
            image,
            noise_info=noise_info,
            sigma=p.get('sigma', None),
            a=p.get('a', None),
            b=p.get('b', None),
            stage_arg=p.get('stage_arg', None),
            verbose=self.verbose,
        )
        return result

    def _apply_median(self, image, noise_info):
        """Decision-based switching median filter for impulse noise.

        Algorithm:
          1. Compute the median of a (kernel_size x kernel_size) neighbourhood
             for every pixel.
          2. A pixel is flagged as an impulse outlier when
             |pixel - median| > threshold * max_range,
             where max_range is estimated from the local pixel distribution.
          3. Only flagged pixels are replaced with the local median;
             non-impulse pixels are kept unchanged.

        Parameters (via denoiser_params)
        ---------------------------------
        kernel_size : int (default 3)
            Neighbourhood window for the median filter.
        threshold   : float (default 0.3)
            Relative deviation threshold for impulse detection.
            Larger values = less aggressive (fewer pixels replaced).
        """
        from scipy.ndimage import median_filter, uniform_filter

        p = dict(self.denoiser_params)
        ksize     = int(p.get('kernel_size', 3))
        threshold = float(p.get('threshold', 0.3))

        med = median_filter(image, size=ksize, mode='reflect')

        local_mean  = uniform_filter(image, size=ksize, mode='reflect')
        local_sq    = uniform_filter(image ** 2, size=ksize, mode='reflect')
        local_var   = np.clip(local_sq - local_mean ** 2, 0, None)
        local_sigma = np.sqrt(local_var)

        global_sigma = float(np.std(image)) or 1e-6
        scale = np.where(local_sigma > 1e-6, local_sigma, global_sigma)

        mask = np.abs(image - med) > threshold * scale

        result = image.copy()
        result[mask] = med[mask]
        return result

    def _apply_act(self, image, noise_info):
        """Adaptive Curvelet Thresholding."""
        from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise

        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        noise_var = p.get('noise_var', None)
        if noise_var is None and sigma is not None:
            noise_var = sigma ** 2

        result, _ = act_denoise(
            image,
            noise_var=noise_var,
            threshold_setting=p.get('threshold_setting', 's'),
        )
        return result

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Denoise a single image.

        Parameters
        ----------
        image : ndarray
            Input image (H, W) or (H, W, 3), uint8 [0, 255] or float [0, 1].

        Returns
        -------
        x_final : ndarray (H, W), int16, [0, 255]
            Denoised image.
        h : ndarray (1, 1), int16, [0]
            Point-like PSF (Dirac delta) — this is a denoiser, not a
            deconvolver. The "kernel" is always 0 (or 1 at center).
        """
        start_time = time.time()

        f = image.astype(np.float64)
        if f.max() > 1.0:
            f /= 255.0

        if f.ndim == 3 and f.shape[2] == 3:
            f = np.mean(f, axis=2)
        elif f.ndim > 2:
            raise ValueError(f"Expected 2D or 3D RGB image, got shape={f.shape}")

        M_orig, N_orig = f.shape

        f = make_size_odd(f)
        M, N = f.shape

        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(f)
            if self.verbose and noise_info:
                print(
                    f"[{self.name}] Noise estimation ({noise_info['method']}): "
                    f"σ_norm={noise_info.get('sigma_norm', 0):.5f}")

        if self.verbose:
            print(f"[{self.name}] Applying {self.method} denoiser...")

        if self.method == 'tv':
            x_denoised = self._apply_tv(f, noise_info)
        elif self.method == 'nlm':
            x_denoised = self._apply_nlm(f, noise_info)
        elif self.method == 'bilateral':
            x_denoised = self._apply_bilateral(f, noise_info)
        elif self.method == 'guided':
            x_denoised = self._apply_guided(f, noise_info)
        elif self.method == 'bm3d':
            x_denoised = self._apply_bm3d(f, noise_info)
        elif self.method == 'vst+bm3d':
            x_denoised = self._apply_vst_bm3d(f, noise_info)
        elif self.method == 'act':
            x_denoised = self._apply_act(f, noise_info)
        elif self.method == 'median':
            x_denoised = self._apply_median(f, noise_info)
        else:
            raise RuntimeError(f"Unknown denoiser: {self.method}")

        x_denoised = np.clip(x_denoised, 0, 1)

        x_final = x_denoised * 255.0
        x_final = np.round(x_final).astype(np.int16)

        h = np.zeros((1, 1), dtype=np.int16)
        h[0, 0] = 1

        self.hyperparams = {
            'time': time.time() - start_time,
            'method': self.method,
            'noise_estimation': self.noise_estimation,
            'input_shape': (M_orig, N_orig),
            'output_shape': x_final.shape,
        }
        if noise_info is not None:
            self.hyperparams['noise_sigma_norm'] = \
                noise_info.get('sigma_norm', None)
            self.hyperparams['noise_sigma_pix'] = \
                noise_info.get('sigma_pix', None)

        if self.verbose:
            print(f"[{self.name}] Done in {self.hyperparams['time']:.3f}s")

        return x_final, h


    def get_param(self) -> List[Tuple[str, Any]]:
        """Get algorithm parameters."""
        return [
            ('method', self.method),
            ('noise_estimation', self.noise_estimation),
            ('denoiser_params', self.denoiser_params),
            ('noise_estimation_params', self.noise_estimation_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        """Change algorithm parameters."""
        for key, value in params.items():
            if key == 'method':
                valid_methods = {'bm3d', 'guided', 'bilateral', 'nlm', 'tv',
                                 'vst+bm3d', 'act'}
                if str(value).lower() not in valid_methods:
                    raise ValueError(f"Invalid method: {value}")
                self.method = str(value).lower()
            elif key == 'noise_estimation':
                valid_noise = {'chen', 'pca', 'none'}
                if str(value).lower() not in valid_noise:
                    raise ValueError(f"Invalid noise_estimation: {value}")
                self.noise_estimation = str(value).lower()
            elif key == 'denoiser_params':
                self.denoiser_params = dict(value or {})
            elif key == 'noise_estimation_params':
                self.noise_estimation_params = dict(value or {})
            elif key == 'verbose':
                self.verbose = bool(value)

    def get_history(self) -> dict:
        """Get algorithm history."""
        return self.history

    def get_hyperparams(self) -> dict:
        """Get algorithm hyperparameters."""
        return self.hyperparams
