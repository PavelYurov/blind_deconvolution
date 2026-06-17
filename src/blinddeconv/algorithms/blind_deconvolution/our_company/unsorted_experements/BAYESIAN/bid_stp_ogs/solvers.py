import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from .utils import (get_gradient_operators, conjugate_gradient, 
                    psf2otf, force_center_mass, keep_largest_component, edgetaper)

class TzikasVB2009Solver:
    def __init__(self, 
                 kernel_shape, 
                 max_iter=50, 
                 cg_iter=20,
                 gamma_ab=(1.0, 100.0), 
                 alpha_ab=(1e-8, 1e-8), 
                 beta_ab=(1e-2, 1e-2),
                 init_beta=100.0,
                 kernel_threshold=0.05, 
                 verbose=False):
        
        self.kernel_shape = kernel_shape
        self.max_iter = max_iter
        self.cg_iter = cg_iter
        
        self.gamma_a, self.gamma_b = gamma_ab
        self.alpha_a, self.alpha_b = alpha_ab
        self.beta_a, self.beta_b = beta_ab
        self.init_beta = init_beta
        self.kernel_threshold = kernel_threshold
        self.verbose = verbose

    def solve(self, g, init_kernel=None):
        rows, cols = g.shape
        N = rows * cols  # <--- Добавлено определение N
        kh, kw = self.kernel_shape
        
        # --- CRITICAL: Apply Edge Tapering ---
        # This prevents boundary discontinuities from polluting the kernel estimate
        g_tapered = edgetaper(g, self.kernel_shape)
        G_hat = fft2(g_tapered)
        
        # Initialization
        if init_kernel is None:
            if self.verbose: print("  [Solver] Init kernel: Gaussian")
            x = np.arange(kw) - kw // 2
            y = np.arange(kh) - kh // 2
            X, Y = np.meshgrid(x, y)
            h_curr = np.exp(-(X**2 + Y**2) / (2 * 2.5**2))
        else:
            if self.verbose: print("  [Solver] Init kernel: Provided")
            h_curr = init_kernel.copy()
            
        h_curr = np.maximum(h_curr, 0)
        h_curr /= (h_curr.sum() + 1e-12)
        
        f_curr = g_tapered.copy()
        
        E_beta = self.init_beta
        E_alpha = np.ones(self.kernel_shape) * 1.0 
        E_gamma_1 = np.ones((rows, cols)) * 50.0
        E_gamma_2 = np.ones((rows, cols)) * 50.0
        
        Q1_hat, Q2_hat = get_gradient_operators((rows, cols))
        Q1_hat_conj = np.conj(Q1_hat)
        Q2_hat_conj = np.conj(Q2_hat)
        
        cy, cx = rows // 2, cols // 2
        dy, dx = kh // 2, kw // 2
        
        inner_loops = 2 
        BETA_CAP = 10000.0 

        for iteration in range(self.max_iter):
            for _ in range(inner_loops):
                # 1. Update Image f (Using tapered G)
                H_hat = psf2otf(h_curr, (rows, cols))
                H_hat_conj = np.conj(H_hat)
                HTH_hat = H_hat_conj * H_hat
                
                rhs_f = E_beta * np.real(ifft2(H_hat_conj * G_hat)).ravel()
                
                def lhs_f(f_flat):
                    f_img = f_flat.reshape((rows, cols))
                    F_hat = fft2(f_img)
                    term1 = E_beta * np.real(ifft2(HTH_hat * F_hat))
                    grad1 = np.real(ifft2(Q1_hat * F_hat))
                    grad2 = np.real(ifft2(Q2_hat * F_hat))
                    term2 = np.real(ifft2(
                        Q1_hat_conj * fft2(E_gamma_1 * grad1) + 
                        Q2_hat_conj * fft2(E_gamma_2 * grad2)
                    ))
                    return (term1 + term2).ravel()
                
                f_curr_flat = conjugate_gradient(lhs_f, rhs_f, f_curr.ravel(), 
                                               max_iter=self.cg_iter, tol=1e-5)
                f_curr = f_curr_flat.reshape((rows, cols))
                f_curr = np.clip(f_curr, 0, 1)
                
                # 2. Update Kernel h (Using tapered G)
                F_hat = fft2(f_curr)
                F_hat_conj = np.conj(F_hat)
                FTF_hat = F_hat_conj * F_hat
                
                correlation_full = E_beta * np.real(ifft2(F_hat_conj * G_hat))
                correlation_shifted = fftshift(correlation_full)
                sl_y = slice(cy - dy, cy - dy + kh)
                sl_x = slice(cx - dx, cx - dx + kw)
                rhs_h = correlation_shifted[sl_y, sl_x].ravel()
                
                E_alpha_flat = E_alpha.ravel()
                
                def lhs_h(h_flat):
                    h_temp = h_flat.reshape((kh, kw))
                    H_temp_hat = psf2otf(h_temp, (rows, cols))
                    res_full = E_beta * np.real(ifft2(FTF_hat * H_temp_hat))
                    res_shifted = fftshift(res_full)
                    res_crop = res_shifted[sl_y, sl_x]
                    return res_crop.ravel() + E_alpha_flat * h_flat
                
                h_curr_flat = conjugate_gradient(lhs_h, rhs_h, h_curr.ravel(), 
                                               max_iter=self.cg_iter + 5, tol=1e-6)
                h_curr = h_curr_flat.reshape((kh, kw))
                
                h_curr = np.maximum(h_curr, 0)
                h_curr /= (h_curr.sum() + 1e-12)
                h_curr = force_center_mass(h_curr, threshold=self.kernel_threshold)

            # 3. Update Hypers
            E_alpha = (self.alpha_a + 0.5) / (self.alpha_b + 0.5 * h_curr**2 + 1e-12)
            
            F_hat_curr = fft2(f_curr)
            grad1 = np.real(ifft2(Q1_hat * F_hat_curr))
            grad2 = np.real(ifft2(Q2_hat * F_hat_curr))
            E_gamma_1 = (self.gamma_a + 0.5) / (self.gamma_b + 0.5 * grad1**2 + 1e-12)
            E_gamma_2 = (self.gamma_a + 0.5) / (self.gamma_b + 0.5 * grad2**2 + 1e-12)
            
            H_hat_curr = psf2otf(h_curr, (rows, cols))
            est_g = np.real(ifft2(H_hat_curr * F_hat_curr))
            resid = g_tapered - est_g # Residual against TAPERED image
            mse = np.mean(resid**2)
            
            E_beta_new = (self.beta_a + N/2.0) / (self.beta_b + 0.5 * np.sum(resid**2))
            E_beta = min(E_beta_new, BETA_CAP)

            if self.verbose and (iteration % 5 == 0):
                print(f"    Iter {iteration}: MSE={mse:.2e}, Beta={E_beta:.1f}")

        # Final Cleanup
        h_curr = keep_largest_component(h_curr, threshold=self.kernel_threshold)
        h_curr /= (h_curr.sum() + 1e-12)
        h_curr = force_center_mass(h_curr, threshold=0.0)

        return f_curr, h_curr