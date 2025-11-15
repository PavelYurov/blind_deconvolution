import numpy as np
from scipy.sparse import diags, eye, issparse
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import convolve2d, fftconvolve
import cv2
import matplotlib.pyplot as plt
import warnings
import gc

# Optional research dependencies (guarded so the framework import doesn't fail)
try:
    from sklearn.linear_model import Lasso
except Exception:  # pragma: no cover
    Lasso = None
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x
try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover
    pywt = None
try:
    from scipy.sparse import random as sparse_random  # type: ignore
except Exception:  # pragma: no cover
    sparse_random = None
try:
    from scipy.optimize import minimize  # type: ignore
except Exception:  # pragma: no cover
    minimize = None

warnings.filterwarnings('ignore')

class VariationalBayesianBID:
    def __init__(self, p=1.2, lambda1=0.01, eta=0.1, max_outer_iter=5, max_inner_iter=2, patch_size=256):
        self.p = p
        self.lambda1 = lambda1
        self.eta = eta
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.patch_size = patch_size
        self.dtype = np.float32
        
    def _compute_derivatives(self, x):
        dx = np.empty_like(x, dtype=self.dtype)
        dy = np.empty_like(x, dtype=self.dtype)
        
        dx[:, :-1] = x[:, 1:] - x[:, :-1]
        dx[:, -1] = 0
        
        dy[:-1, :] = x[1:, :] - x[:-1, :]
        dy[-1, :] = 0
        
        dxx = np.empty_like(x, dtype=self.dtype)
        dxx[:, 1:-1] = x[:, 2:] + x[:, :-2] - 2*x[:, 1:-1]
        dxx[:, 0] = dxx[:, -1] = 0
        
        dyy = np.empty_like(x, dtype=self.dtype)
        dyy[1:-1, :] = x[2:, :] + x[:-2, :] - 2*x[1:-1, :]
        dyy[0, :] = dyy[-1, :] = 0
        
        dxy = np.empty_like(x, dtype=self.dtype)
        dxy[:-1, :-1] = dx[1:, :-1] - dx[:-1, :-1]
        dxy[-1, :] = dxy[:, -1] = 0
        
        return {'h': dx, 'v': dy, 'hh': dxx, 'vv': dyy, 'hv': dxy}
    
    def _compute_laplacian(self, h):
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=self.dtype)
        return convolve2d(h, kernel, mode='same', boundary='symm')
    
    def _compute_alpha(self, x):
        derivs = self._compute_derivatives(x)
        numerator = self.lambda1 * (x.size / max(self.p, 1e-6)) + 1
        denominator = 1e-6
        
        weights = {'h': 0.5, 'v': 0.5, 'hh': 0.25, 'vv': 0.25, 'hv': 0.25}
        for d in derivs:
            denominator += weights[d] * np.sum(np.abs(derivs[d])**self.p)
            
        return min(numerator / denominator, 1e6)
    
    def _compute_gamma(self, h):
        Ch = self._compute_laplacian(h)
        denominator = np.sum(Ch**2) + 1e-6
        return min((h.size + 2) / denominator, 1e6)
    
    def _compute_v(self, x):
        derivs = self._compute_derivatives(x)
        return {d: derivs[d]**2 + 1e-6 for d in derivs}
    
    def _apply_convolution(self, x, h):
        if x.shape[0] > 512 or x.shape[1] > 512:
            result = np.zeros_like(x, dtype=self.dtype)
            pad = h.shape[0] // 2
            x_padded = np.pad(x, ((pad, pad), (pad, pad)), mode='reflect')
            
            for i in range(0, x.shape[0], self.patch_size):
                for j in range(0, x.shape[1], self.patch_size):
                    patch = x_padded[i:i+self.patch_size+2*pad, j:j+self.patch_size+2*pad]
                    conv_patch = fftconvolve(patch, h, mode='valid')
                    result[i:i+self.patch_size, j:j+self.patch_size] = conv_patch
            return result
        else:
            return fftconvolve(x, h, mode='same')
    
    def _update_x_patch(self, a_patch, h, u_patch, alpha, v_patch, W_patch, Phi_patch):
        patch_shape = (self.patch_size, self.patch_size)
        N_patch = self.patch_size * self.patch_size
        
        Wha_patch = np.zeros(N_patch, dtype=self.dtype)
        for i in range(W_patch.shape[0]):
            Wha_patch[i] = np.dot(W_patch[i].toarray().ravel() if issparse(W_patch) else W_patch[i], a_patch)
        
        def conv_operator(x):
            x_img = x.reshape(patch_shape)
            return self._apply_convolution(x_img, h).ravel()
        
        H = LinearOperator((N_patch, N_patch), matvec=conv_operator, rmatvec=conv_operator)
        
        B_diag = v_patch['h'].ravel()**(self.p/2 - 1) + v_patch['v'].ravel()**(self.p/2 - 1)
        
        def A_operator(x):
            Hx = H.matvec(x)
            HT_Hx = H.rmatvec(Hx)
            Bx = B_diag * x
            return self.eta * HT_Hx + alpha * self.p * Bx
        
        A = LinearOperator((N_patch, N_patch), matvec=A_operator)
        b = self.eta * H.rmatvec(Wha_patch + u_patch.ravel())
        
        x, _ = cg(A, b, maxiter=30, tol=1e-2)
        return x.reshape(patch_shape)
    
    def _update_h(self, a, x, u, gamma):
        x_fft = np.fft.fft2(x, s=(x.shape[0]+self.kernel_shape[0]-1, 
                                 x.shape[1]+self.kernel_shape[1]-1))
        
        def objective(h_vec):
            h = h_vec.reshape(self.kernel_shape)
            conv = np.fft.ifft2(x_fft * np.fft.fft2(h, s=x_fft.shape)).real
            conv = conv[:x.shape[0], :x.shape[1]] 
            
            data_term = 0.5 * self.eta * np.sum((conv.ravel() - (self.W @ a + u))**2)
            
            laplacian = convolve2d(h, np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=self.dtype), mode='same')
            reg_term = 0.5 * gamma * np.sum(laplacian**2)
            
            return data_term + reg_term
        
        h0 = np.zeros(self.kernel_shape, dtype=self.dtype)
        center = (self.kernel_shape[0]//2, self.kernel_shape[1]//2)
        h0[center] = 1
        
        res = minimize(objective, h0.ravel(), method='L-BFGS-B', 
                      options={'maxiter': 5, 'maxls': 5, 'maxfun': 10})
        
        return res.x.reshape(self.kernel_shape)
    
    def _process_image_in_patches(self, img, func, **kwargs):
        result = np.zeros_like(img, dtype=self.dtype)
        pad = self.kernel_shape[0] // 2 
        
        for i in range(0, img.shape[0], self.patch_size):
            for j in range(0, img.shape[1], self.patch_size):
                i_start = max(0, i - pad)
                j_start = max(0, j - pad)
                i_end = min(img.shape[0], i + self.patch_size + pad)
                j_end = min(img.shape[1], j + self.patch_size + pad)
                
                patch = img[i_start:i_end, j_start:j_end]

                processed_patch = func(patch, **kwargs)
                
                update_i_start = i if i_start == i else pad
                update_j_start = j if j_start == j else pad
                update_i_end = update_i_start + min(self.patch_size, img.shape[0] - i)
                update_j_end = update_j_start + min(self.patch_size, img.shape[1] - j)
                
                result[i:i+(update_i_end-update_i_start), 
                       j:j+(update_j_end-update_j_start)] = processed_patch[update_i_start:update_i_end, 
                                                                           update_j_start:update_j_end]
        return result
    
    def deconvolve(self, y, Phi, W, img_shape, kernel_shape):
        self.img_shape = img_shape
        self.kernel_shape = kernel_shape
        self.W = W
        self.Phi = Phi
        
        a = np.random.randn(W.shape[1]).astype(self.dtype)
        x = (W @ a).reshape(img_shape).astype(self.dtype)
        h = np.zeros(kernel_shape, dtype=self.dtype)
        center = (kernel_shape[0]//2, kernel_shape[1]//2)
        h[center] = 1  
        u = np.zeros(img_shape[0] * img_shape[1], dtype=self.dtype)
        
        results = {'x': [], 'h': [], 'a': []}
        
        for k in tqdm(range(self.max_outer_iter), desc="Outer iterations"):
            def x_update_func(patch):
                alpha = self._compute_alpha(patch)
                v_patch = self._compute_v(patch)
                
                i, j = 0, 0  
                u_patch = u[i*patch.shape[0]:(i+1)*patch.shape[0], 
                           j*patch.shape[1]:(j+1)*patch.shape[1]].ravel()
                
                return self._update_x_patch(a, h, u_patch, alpha, v_patch, 
                                          W[i*patch.shape[0]:(i+1)*patch.shape[0]], 
                                          Phi[:, i*patch.shape[0]:(i+1)*patch.shape[0]])
            
            x = self._process_image_in_patches(x, x_update_func)
            
            h = self._update_h(a, x, u, self._compute_gamma(h))
            
            conv = self._apply_convolution(x, h)
            residual = y - Phi @ (W @ a)
            beta = (len(y) + 2) / (np.sum(residual**2) + 1e-6)
            tau = (len(a) + 1) / (np.sum(np.abs(a)) + 1e-6)
            
            lasso = Lasso(alpha=tau/(2*beta), max_iter=50, selection='random', 
                         tol=1e-2, warm_start=True)
            lasso.fit(Phi @ W, y)
            a = lasso.coef_.astype(self.dtype)
            
            u = u + (W @ a - conv.ravel())
            
            gc.collect()
            
            if k % 1 == 0:
                results['x'].append(x.copy())
                results['h'].append(h.copy())
                results['a'].append(a.copy())
                
                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(x, cmap='gray')
                plt.title(f'Reconstructed (Iter {k+1})')
                
                plt.subplot(132)
                plt.imshow(h, cmap='gray')
                plt.title('Estimated Kernel')
                
                plt.subplot(133)
                plt.plot(a)
                plt.title('Sparse Coefficients')
                
                plt.tight_layout()
                plt.show()
                plt.close()
        
        return x, h, a, results


# === Integrated, framework-ready implementation ===
# Lightweight VB-style blind deconvolution adapter compatible with the framework.
# Alternates Wiener deconvolution for the latent image with Richardson–Lucy-style
# kernel updates. Works on grayscale or color (single kernel estimated on grayscale).

from .base import DeconvolutionAlgorithm

class VariationalBayesianBIDAlgorithm(DeconvolutionAlgorithm):
    """
    Practical VB-style Blind Deconvolution (framework-ready).

    - Alternates:
      1) Latent image update via Wiener deconvolution
      2) Kernel update via RL-style multiplicative rule

    Notes:
    - Returns uint8 image with original shape.
    - For color images, estimates a single blur kernel on the grayscale proxy
      and deconvolves each channel with that kernel.
    """

    def __init__(self,
                 kernel_size: int = 15,
                 outer_iters: int = 10,
                 x_wiener_reg: float = 1e-2,
                 kernel_update_iters: int = 5,
                 eps: float = 1e-6,
                 multiscale: bool = True,
                 scales: tuple = (0.25, 0.5, 1.0),
                 edge_taper_iters: int = 2,
                 kernel_smooth_sigma: float = 0.0,
                 kernel_l1_thresh: float = 0.0,
                 support: str = 'auto',  # 'auto'|'box'|'disk'|'none'
                 support_radius: int | None = None):
        super().__init__('VariationalBayesianBID')
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.outer_iters = max(1, outer_iters)
        self.x_wiener_reg = float(x_wiener_reg)
        self.kernel_update_iters = max(1, kernel_update_iters)
        self.eps = float(eps)
        self.multiscale = multiscale
        self.scales = tuple(scales)
        self.edge_taper_iters = max(0, edge_taper_iters)
        self.kernel_smooth_sigma = max(0.0, float(kernel_smooth_sigma))
        self.kernel_l1_thresh = max(0.0, float(kernel_l1_thresh))
        self.support = support
        self.support_radius = support_radius

    # ---- helpers ----
    def _normalize_kernel(self, k: np.ndarray) -> np.ndarray:
        k = np.clip(k, 0, None)
        s = k.sum()
        if s <= 0:
            # fallback: delta kernel
            k[:] = 0
            c = k.shape[0]//2, k.shape[1]//2
            k[c] = 1.0
            return k
        return k / s

    def _fft_conv_same(self, img: np.ndarray, k: np.ndarray) -> np.ndarray:
        return fftconvolve(img, k, mode='same')

    def _wiener_deconv(self, y: np.ndarray, k: np.ndarray, reg: float) -> np.ndarray:
        # y, k in float64, y in [0,1]
        H = np.fft.fft2(k, s=y.shape)
        Y = np.fft.fft2(y)
        denom = (np.abs(H) ** 2) + reg
        X = (np.conj(H) * Y) / np.maximum(denom, self.eps)
        x = np.fft.ifft2(X).real
        return np.clip(x, 0.0, 1.0)

    def _flip_kernel(self, k: np.ndarray) -> np.ndarray:
        return np.flip(np.flip(k, axis=0), axis=1)

    def _center_crop(self, arr: np.ndarray, h: int, w: int) -> np.ndarray:
        H, W = arr.shape
        sh = max((H - h) // 2, 0)
        sw = max((W - w) // 2, 0)
        return arr[sh:sh+h, sw:sw+w]

    def _update_kernel_RL(self, y: np.ndarray, x: np.ndarray, k: np.ndarray, iters: int) -> np.ndarray:
        # Multiplicative update: k <- k * (flip(x) * (y / (x*k)))
        k = k.copy()
        for _ in range(iters):
            xk = self._fft_conv_same(x, k)
            ratio = y / np.maximum(xk, self.eps)
            corr_full = self._fft_conv_same(ratio, self._flip_kernel(x))
            # Map adjoint result to kernel support (center crop)
            corr = self._center_crop(corr_full, k.shape[0], k.shape[1])
            k = k * np.maximum(corr, 0)
            # Optional priors
            k = self._apply_kernel_support(k)
            if self.kernel_l1_thresh > 0:
                k = np.where(k >= self.kernel_l1_thresh * k.max(), k, 0)
            if self.kernel_smooth_sigma > 0:
                ks = int(max(3, round(self.kernel_smooth_sigma * 6)) // 2 * 2 + 1)
                k = cv2.GaussianBlur(k, (ks, ks), self.kernel_smooth_sigma)
            k = self._normalize_kernel(k)
        return k

    def _apply_kernel_support(self, k: np.ndarray) -> np.ndarray:
        if self.support == 'none':
            return k
        h, w = k.shape
        cy, cx = h//2, w//2
        if self.support_radius is None:
            r = min(cy, cx)
        else:
            r = min(self.support_radius, min(cy, cx))
        if self.support in ('auto', 'box'):
            # box support within r
            mask = np.zeros_like(k)
            mask[cy-r:cy+r+1, cx-r:cx+r+1] = 1.0
        elif self.support == 'disk':
            yy, xx = np.ogrid[-cy:h-cy, -cx:w-cx]
            mask = ((xx*xx + yy*yy) <= r*r).astype(k.dtype)
        else:
            return k
        return k * mask

    def _estimate_single_kernel(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # y in [0,1], grayscale
        y_prep = self._edge_taper(y)

        def run_single_scale(y_s: np.ndarray, k_init: np.ndarray, k_size: int):
            # Ensure odd size
            if k_size % 2 == 0:
                k_size += 1
            k = np.zeros((k_size, k_size), dtype=np.float64)
            if k_init is not None:
                # center-crop or resize init to current size
                if k_init.shape != (k_size, k_size):
                    k = cv2.resize(k_init, (k_size, k_size), interpolation=cv2.INTER_CUBIC)
                    k = np.maximum(k, 0)
                else:
                    k = k_init.copy()
            else:
                k[k_size//2, k_size//2] = 1.0
            k = self._normalize_kernel(k)
            x = y_s.copy()
            for _ in range(self.outer_iters):
                x = self._wiener_deconv(y_s, k, self.x_wiener_reg)
                k = self._update_kernel_RL(y_s, x, k, self.kernel_update_iters)
            return x, k

        if not self.multiscale or min(y_prep.shape) < 128:
            x, k = run_single_scale(y_prep, None, self.kernel_size)
            return x, k

        # Coarse-to-fine
        k_curr = None
        x_curr = None
        H, W = y_prep.shape
        for s in self.scales:
            h_s, w_s = max(16, int(H * s)), max(16, int(W * s))
            y_s = cv2.resize(y_prep, (w_s, h_s), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
            k_s = max(3, int(round(self.kernel_size * s))) | 1
            x_curr, k_curr = run_single_scale(y_s, k_curr, k_s)
        # Upsample x to original size for return
        x_full = cv2.resize(x_curr, (W, H), interpolation=cv2.INTER_CUBIC)
        k_full = cv2.resize(k_curr, (self.kernel_size, self.kernel_size), interpolation=cv2.INTER_CUBIC)
        k_full = np.maximum(k_full, 0)
        k_full = self._normalize_kernel(k_full)
        return x_full, k_full

    def _edge_taper(self, y: np.ndarray) -> np.ndarray:
        if self.edge_taper_iters <= 0:
            return y
        # Create separable 2D Hann window (NumPy) and blend with blurred image
        h, w = y.shape
        if h < 2 or w < 2:
            return y
        wy = np.hanning(h).reshape(h, 1)
        wx = np.hanning(w).reshape(1, w)
        w2d = (wy * wx)
        # scale to [alpha, 1]
        alpha = 0.3
        w2d = alpha + (1 - alpha) * w2d
        out = y.copy()
        # Use small Gaussian as initial blur for taper blending
        gk = cv2.getGaussianKernel(7, 1.5)
        gk2d = gk @ gk.T
        for _ in range(self.edge_taper_iters):
            by = self._fft_conv_same(out, gk2d)
            out = w2d * out + (1 - w2d) * by
        return np.clip(out, 0.0, 1.0)

    def _deconv_channel(self, ch: np.ndarray, k: np.ndarray) -> np.ndarray:
        ch_f = ch.astype(np.float64) / 255.0
        x = self._wiener_deconv(ch_f, k, self.x_wiener_reg)
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError('Input image is None')

        # Handle grayscale or color
        if image.ndim == 2:
            y_gray = image.astype(np.float64) / 255.0
            x_est, k_est = self._estimate_single_kernel(y_gray)
            out = np.clip(x_est * 255.0, 0, 255).astype(np.uint8)
            return out
        elif image.ndim == 3 and image.shape[2] == 3:
            # Estimate kernel on grayscale proxy
            y_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            _, k_est = self._estimate_single_kernel(y_gray)
            # Deconvolve each channel with the estimated kernel
            b, g, r = cv2.split(image)
            b_r = self._deconv_channel(b, k_est)
            g_r = self._deconv_channel(g, k_est)
            r_r = self._deconv_channel(r, k_est)
            return cv2.merge([b_r, g_r, r_r])
        else:
            raise ValueError('Unsupported image shape for VariationalBayesianBID')


class VBAmizicBIDAlgorithm(DeconvolutionAlgorithm):
    """
    Variational Bayesian Compressive Blind Image Deconvolution (Amizic et al.) —
    practical implementation adapted for standard deblurring (Phi = I) and
    orthonormal synthesis transform W (wavelets). The algorithm follows the
    VB-ADMM updates derived in the paper and solves:

        min_{x,h,a}  (beta/2)||y - W a||^2 + (eta/2)||W a - Hx + u||^2
                      + tau||a||_1 + alpha R_p(x) + (gamma/2)||C h||^2
                      s.t. sum(h)=1, h>=0

    - x update: Conjugate gradient on (eta H^T H + alpha p L_w) x = eta H^T(Wa + u)
      with L_w the weighted Laplacian from p<1 quasi-norm prior.
    - h update: gradient steps on 0.5*eta||x*h - (Wa+u)||^2 + 0.5*gamma||C h||^2
      with non-negativity and normalization constraints.
    - a update: FISTA on 0.5*beta||y - W a||^2 + 0.5*eta||W a - b||^2 + tau||a||_1
      using forward/inverse wavelet operators when available; otherwise W=I.
    - beta, tau updated from residual statistics as in (28)-(29).

    Note: To keep the implementation tractable and stable for typical images,
    we use first-order gradients (h,v) in the x prior (instead of five terms).
    """

    def __init__(self,
                 kernel_size: int = 15,
                 iters: int = 8,
                 x_cg_iters: int = 30,
                 h_grad_iters: int = 8,
                 fista_iters: int = 40,
                 p: float = 0.8,
                 alpha: float = 0.02,
                 gamma: float = 0.1,
                 eta: float = 1.0,
                 beta0: float = 10.0,
                 tau0: float = 0.01,
                 wavelet: str = 'haar',
                 wavelet_level: int = 2,
                 color: bool = False,
                 eps: float = 1e-6,
                 support: str = 'auto',
                 support_radius: int | None = None):
        super().__init__('VB_Amizic_BID')
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.iters = iters
        self.x_cg_iters = x_cg_iters
        self.h_grad_iters = h_grad_iters
        self.fista_iters = fista_iters
        self.p = p
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.beta0 = beta0
        self.tau0 = tau0
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.handle_color = color
        self.eps = eps
        self.support = support
        self.support_radius = support_radius

    # ---------- basic ops ----------
    def _conv_same(self, img: np.ndarray, k: np.ndarray) -> np.ndarray:
        return fftconvolve(img, k, mode='same')

    def _flip(self, k: np.ndarray) -> np.ndarray:
        return np.flip(np.flip(k, 0), 1)

    def _h_transpose(self, r: np.ndarray, x: np.ndarray, k_shape: tuple[int, int]) -> np.ndarray:
        # Adjoint of k -> conv_same(x,k): crop conv(r, flip(x)) to kernel support
        rf = self._conv_same(r, self._flip(x))
        kh, kw = k_shape
        H, W = rf.shape
        sh = max((H - kh)//2, 0)
        sw = max((W - kw)//2, 0)
        return rf[sh:sh+kh, sw:sw+kw]

    def _laplacian(self, a: np.ndarray) -> np.ndarray:
        k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
        return self._conv_same(a, k)

    def _laplacian2(self, a: np.ndarray) -> np.ndarray:
        return self._laplacian(self._laplacian(a))

    def _grad(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        dx[:, :-1] = x[:, 1:] - x[:, :-1]
        dy[:-1, :] = x[1:, :] - x[:-1, :]
        return dx, dy

    def _div(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        out = np.zeros_like(px)
        out[:, :-1] -= px[:, :-1]
        out[:, 1:] += px[:, :-1]
        out[:-1, :] -= py[:-1, :]
        out[1:, :] += py[:-1, :]
        return out

    def _normalize_kernel(self, k: np.ndarray) -> np.ndarray:
        k = np.clip(k, 0, None)
        s = k.sum()
        if s <= 0:
            k[:] = 0
            k[k.shape[0]//2, k.shape[1]//2] = 1.0
            return k
        return k / s

    def _apply_kernel_support(self, k: np.ndarray) -> np.ndarray:
        if self.support == 'none':
            return k
        h, w = k.shape
        cy, cx = h//2, w//2
        r = min(self.support_radius or min(cy, cx), min(cy, cx))
        if self.support in ('auto','box'):
            mask = np.zeros_like(k)
            mask[cy-r:cy+r+1, cx-r:cx+r+1] = 1.0
        elif self.support == 'disk':
            yy, xx = np.ogrid[-cy:h-cy, -cx:w-cx]
            mask = ((xx*xx + yy*yy) <= r*r).astype(k.dtype)
        else:
            return k
        return k*mask

    # ---------- transforms ----------
    def _have_wavelet(self) -> bool:
        return pywt is not None

    def _wt(self, x: np.ndarray):
        if self._have_wavelet():
            coeffs = pywt.wavedec2(x, self.wavelet, level=self.wavelet_level, mode='periodization')
            arr, sl = pywt.coeffs_to_array(coeffs)
            return arr, sl, x.shape
        else:
            # identity
            return x.copy(), None, x.shape

    def _iwt(self, arr: np.ndarray, sl, shape):
        if self._have_wavelet():
            coeffs = pywt.array_to_coeffs(arr, sl, output_format='wavedec2')
            x = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
            return x[:shape[0], :shape[1]]
        else:
            return arr.reshape(shape)

    # ---------- solvers ----------
    def _solve_x(self, yWa_u: np.ndarray, k: np.ndarray, x0: np.ndarray, w_x: np.ndarray, w_y: np.ndarray) -> np.ndarray:
        # CG on A x = b with A = eta H^T H + alpha p L_w
        def H(x):
            return self._conv_same(x, self._flip(k))
        def HT(y):
            return self._conv_same(y, k)
        def A(x):
            # H^T H x
            t = HT(H(x)) * self.eta
            # weighted Laplacian
            dx, dy = self._grad(x)
            t += self.alpha * self.p * self._div(w_x*dx, w_y*dy)
            return t
        b = HT(yWa_u) * self.eta

        x = x0.copy()
        r = b - A(x)
        pdir = r.copy()
        rsold = (r*r).sum()
        for _ in range(self.x_cg_iters):
            Ap = A(pdir)
            denom = (pdir*Ap).sum() + self.eps
            alpha_cg = rsold / denom
            x = x + alpha_cg * pdir
            r = r - alpha_cg * Ap
            rsnew = (r*r).sum()
            if rsnew < 1e-10:
                break
            pdir = r + (rsnew/rsold) * pdir
            rsold = rsnew
        return np.clip(x, 0.0, 1.0)

    def _update_h(self, x: np.ndarray, target: np.ndarray, k: np.ndarray) -> np.ndarray:
        k = k.copy()
        lr = 0.1 / (self.eta * (np.mean(x*x) + self.eps) * (self.kernel_size) + self.gamma + 1e-6)
        for _ in range(self.h_grad_iters):
            xk = self._conv_same(x, k)
            r = xk - target
            grad_data = self._h_transpose(r, x, k.shape) * self.eta
            grad_reg = self._laplacian2(k) * self.gamma
            k = k - lr * (grad_data + grad_reg)
            k = np.maximum(k, 0)
            k = self._apply_kernel_support(k)
            k = self._normalize_kernel(k)
        return k

    def _fista_lasso(self, y: np.ndarray, b: np.ndarray, a0: np.ndarray, L: float, tau: float, sl, shape) -> np.ndarray:
        # solve min_a 0.5 beta||y - Wa||^2 + 0.5 eta||Wa - b||^2 + tau||a||_1
        # gradient wrt a: W_T( (beta+eta)Wa - beta y - eta b )
        t = 1.0
        a = a0.copy()
        z = a.copy()

        def W_of(a):
            return self._iwt(a, sl, shape)
        def WT_of(x):
            arr, _, _ = self._wt(x)
            return arr

        for _ in range(self.fista_iters):
            Wa = W_of(z)
            grad = WT_of((self.beta * (Wa - y) + self.eta * (Wa - b)))
            a_next = z - (1.0/L) * grad
            # soft threshold
            a_next = np.sign(a_next) * np.maximum(np.abs(a_next) - tau/L, 0)
            t_next = 0.5*(1 + np.sqrt(1 + 4*t*t))
            z = a_next + ((t - 1)/t_next) * (a_next - a)
            a, t = a_next, t_next
        return a

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError('Input image is None')
        if image.ndim == 3 and image.shape[2] == 3:
            # Estimate kernel on grayscale and deconvolve channels
            y_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
            result_gray = self._process_gray(y_gray)
            # simple channel-wise deconv using estimated kernel from gray
            # for consistency, re-run x update per channel with same kernel
            k_est = self._last_kernel
            out_channels = []
            for ch in cv2.split(image):
                y = ch.astype(np.float64)/255.0
                # Initialize
                x = self._deconv_init(y, k_est)
                out = np.clip(x*255.0, 0, 255).astype(np.uint8)
                out_channels.append(out)
            return cv2.merge(out_channels)
        elif image.ndim == 2:
            y = image.astype(np.float64)/255.0
            x = self._process_gray(y)
            return np.clip(x*255.0, 0, 255).astype(np.uint8)
        else:
            raise ValueError('Unsupported image shape for VB_Amizic_BID')

    def _deconv_init(self, y: np.ndarray, k: np.ndarray) -> np.ndarray:
        # quick Wiener for channel refinement
        H = np.fft.fft2(k, s=y.shape)
        Y = np.fft.fft2(y)
        reg = 1e-2
        X = np.conj(H) * Y / (np.abs(H)**2 + reg)
        return np.clip(np.fft.ifft2(X).real, 0.0, 1.0)

    def _process_gray(self, y: np.ndarray) -> np.ndarray:
        H, W = y.shape
        # init kernel as delta
        k = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float64)
        k[self.kernel_size//2, self.kernel_size//2] = 1.0
        k = self._apply_kernel_support(k)
        k = self._normalize_kernel(k)

        # init transforms
        a, sl, shp = self._wt(y)
        u = np.zeros_like(y)
        self.beta = float(self.beta0)
        self.tau = float(self.tau0)

        # initialize a with forward transform of y
        a = a.copy()
        x = np.clip(y.copy(), 0.0, 1.0)

        for _ in range(self.iters):
            # weights for x-prior
            dx, dy = self._grad(x)
            w_x = (dx*dx + self.eps)**(self.p/2 - 1)
            w_y = (dy*dy + self.eps)**(self.p/2 - 1)

            # x update (CG)
            Wa = self._iwt(a, sl, shp)
            x = self._solve_x(Wa + u, k, x, w_x, w_y)

            # h update (gradient steps)
            target = Wa + u
            k = self._update_h(x, target, k)

            # hyperparameters beta, tau
            res = y - Wa
            self.beta = (H*W + 2) / (np.sum(res*res) + self.eps)  # (28) with Phi=I
            self.tau = (H*W + 1) / (np.sum(np.abs(a)) + self.eps)  # (29)

            # a update (FISTA)
            b = self._conv_same(x, k) - u
            L = (self.beta + self.eta)
            a = self._fista_lasso(y, b, a, L=L, tau=self.tau, sl=sl, shape=shp)

            # dual update
            Wa = self._iwt(a, sl, shp)
            u = u + (Wa - self._conv_same(x, k))

        self._last_kernel = k.copy()
        return x

def create_measurement_matrix(M, N, dtype=np.float32):
    density = min(0.05, 500000/(M*N))  # Reduced density
    Phi = sparse_random(M, N, density=density, dtype=dtype)
    return Phi

def create_wavelet_matrix(N, dtype=np.float32):
    size = int(np.sqrt(N))
    wavelet = 'haar'
    
    def wavelet_transform(x):
        x_img = x.reshape(size, size)
        coeffs = pywt.wavedec2(x_img, wavelet, level=2)
        arr, _ = pywt.coeffs_to_array(coeffs)
        return arr.ravel()
    
    return LinearOperator((N, N), matvec=wavelet_transform, rmatvec=wavelet_transform, dtype=dtype)

def load_and_prepare_image(image_path, dtype=np.float32):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return (img.astype(dtype) / 255.0)
