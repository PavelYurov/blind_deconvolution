"""
Реализация VBSolver (Kotera et al. 2017).
Strict implementation of Algorithm 1 with stabilization.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import fftconvolve
from .utils import compute_gradients, compute_divergence

class VBSolver:
    def __init__(self, 
                 blurred_img: np.ndarray, 
                 kernel_size: tuple[int, int], 
                 mask: np.ndarray,
                 b_lambda: float,
                 b_beta: float):
        self.g = blurred_img
        self.H, self.W = self.g.shape
        self.kh, self.kw = kernel_size
        self.mask = mask 
        
        # Hyperparameters
        self.b_lambda = b_lambda
        self.b_beta = b_beta
        # "a" parameters are typically 0
        self.a_lambda = 0.0
        self.a_beta = 0.0
        self.a_alpha = 0.0
        self.b_alpha = 0.0 # Paper says these are negligible
        
        # State
        self.u = self.g.copy()
        self.h = None
        
        # Variational parameters
        self.alpha = 50.0  # Start with moderate confidence
        self.nu = 0.0      # Fixed dof for outliers
        
        self.gamma = np.ones_like(self.g)
        self.lambda_u = np.ones_like(self.g) * 10.0
        self.beta = np.ones(kernel_size) * 100.0
        
        # Covariances (Diagonal approx)
        self.C_h = np.zeros_like(self.g)
        self.C_u = np.zeros(kernel_size)

    def initialize_h(self, h_init: np.ndarray):
        self.h = h_init.copy()
        # Initialize beta based on h (Eq 24)
        # Avoid division by zero
        h2 = self.h**2
        self.beta = (1 + 2*self.a_beta) / (h2 + 2*self.b_beta + 1e-12)

    def run_step(self):
        """One iteration of Algorithm 1."""
        
        # ── 1. Update u ──
        # Eq: (H^T Gamma H + C^h + 1/alpha D^T Lambda D) u = H^T Gamma g
        # Note: We divide prior term by alpha
        
        h_flip = self.h[::-1, ::-1]
        
        # RHS: H^T (Gamma * g)
        weighted_g = self.gamma * self.g
        rhs_u = fftconvolve(weighted_g, h_flip, mode='same')
        
        # Preconditioner for u
        h_norm2 = np.sum(self.h**2)
        # Diag approx: h_norm2 * gamma + C^h + (4 * lambda / alpha)
        diag_A_u = h_norm2 * self.gamma + self.C_h + (4.0 * self.lambda_u / (self.alpha + 1e-6)) + 1e-9
        M_u_inv = 1.0 / diag_A_u
        
        def matvec_u(x_vec):
            x = x_vec.reshape(self.H, self.W)
            # 1. H^T Gamma H x
            Hx = fftconvolve(x, self.h, mode='same')
            term1 = fftconvolve(self.gamma * Hx, h_flip, mode='same')
            # 2. C^h x
            term2 = self.C_h * x
            # 3. 1/alpha D^T Lambda D x
            dy, dx = compute_gradients(x)
            term_prior = compute_divergence(self.lambda_u * dy, self.lambda_u * dx)
            term3 = term_prior / (self.alpha + 1e-6)
            
            return (term1 + term2 + term3).ravel()

        def precond_u(x):
            return x * M_u_inv.ravel()

        A_u_op = LinearOperator((self.g.size, self.g.size), matvec=matvec_u)
        M_u_op = LinearOperator((self.g.size, self.g.size), matvec=precond_u)
        
        u_flat, _ = cg(A_u_op, rhs_u.ravel(), x0=self.u.ravel(), M=M_u_op, atol=1e-5, maxiter=20)
        self.u = u_flat.reshape(self.H, self.W)
        
        # Project u to valid range (Crucial for stability)
        self.u = np.clip(self.u, 0.0, 1.0)

        # ── 2. Update C^u ──
        # Var_u = 1 / diag(A_u). Reusing diag_A_u from preconditioner
        var_u = 1.0 / diag_A_u
        # Clip variance to avoid exploding uncertainty
        var_u = np.clip(var_u, 0, 1.0)
        
        # C^u = conv(gamma, var_u_flip) [crop]
        # In paper: C^u = diag(gamma * (diag(inv_Hess_u)))
        # Interpreted as projecting pixel uncertainty to kernel space
        full_corr = fftconvolve(self.gamma, var_u[::-1, ::-1], mode='same')
        cy, cx = full_corr.shape[0]//2, full_corr.shape[1]//2
        kh, kw = self.kh, self.kw
        self.C_u = full_corr[cy - kh//2 : cy - kh//2 + kh, cx - kw//2 : cx - kw//2 + kw]
        if self.C_u.shape != (kh, kw): self.C_u = np.zeros((kh, kw))

        # ── 3. Update h ──
        # Eq: (U^T Gamma U + C^u + 1/alpha B) h = U^T Gamma g
        
        u_flip = self.u[::-1, ::-1]
        
        # RHS: U^T (Gamma g)
        rhs_h_full = fftconvolve(weighted_g, u_flip, mode='same')
        cy_g, cx_g = rhs_h_full.shape[0]//2, rhs_h_full.shape[1]//2
        rhs_h = rhs_h_full[cy_g - kh//2 : cy_g - kh//2 + kh,
                           cx_g - kw//2 : cx_g - kw//2 + kw].ravel()
        
        # Preconditioner for h
        u2_flip = (self.u**2)[::-1, ::-1]
        diag_UTGU_full = fftconvolve(self.gamma, u2_flip, mode='same')
        diag_UTGU = diag_UTGU_full[cy_g - kh//2 : cy_g - kh//2 + kh, cx_g - kw//2 : cx_g - kw//2 + kw]
        
        # diag_A_h = diag_UTGU + C^u + beta/alpha
        diag_A_h = diag_UTGU + self.C_u + (self.beta / (self.alpha + 1e-6)) + 1e-9
        M_h_inv = 1.0 / diag_A_h
        
        def matvec_h(h_vec):
            h_curr = h_vec.reshape(self.kh, self.kw)
            # 1. U^T Gamma U h
            Uh = fftconvolve(self.u, h_curr, mode='same')
            term1_full = fftconvolve(self.gamma * Uh, u_flip, mode='same')
            term1 = term1_full[cy_g - kh//2 : cy_g - kh//2 + kh,
                               cx_g - kw//2 : cx_g - kw//2 + kw]
            # 2. C^u h
            term2 = self.C_u * h_curr
            # 3. 1/alpha B h
            term3 = (self.beta / (self.alpha + 1e-6)) * h_curr
            return (term1 + term2 + term3).ravel()
            
        def precond_h(x):
            return x * M_h_inv.ravel()

        A_h_op = LinearOperator((self.kh*self.kw, self.kh*self.kw), matvec=matvec_h)
        M_h_op = LinearOperator((self.kh*self.kw, self.kh*self.kw), matvec=precond_h)
        
        h_flat, _ = cg(A_h_op, rhs_h, x0=self.h.ravel(), M=M_h_op, atol=1e-7, maxiter=25)
        self.h = h_flat.reshape(self.kh, self.kw)
        
        # --- Strict Regularization (Enforce Constraints) ---
        # 1. Positivity
        self.h = np.maximum(self.h, 0.0)
        # 2. Normalization (Sum = 1)
        h_sum = self.h.sum()
        if h_sum > 1e-12:
            self.h /= h_sum
        else:
            # Fallback if kernel died
            self.h = np.ones_like(self.h) / self.h.size

        # ── 4. Update C^h ──
        var_h = 1.0 / diag_A_h
        var_h = np.clip(var_h, 0, 1.0)
        self.C_h = fftconvolve(self.gamma, var_h, mode='same')

        # ── 5. Update Hyperparameters ──
        
        # A. Alpha (Noise)
        Hu = fftconvolve(self.u, self.h, mode='same')
        resid_sq = (self.g - Hu)**2
        
        var_Hu = fftconvolve(var_u, self.h**2, mode='same')
        E_error = resid_sq + self.C_h + var_Hu
        
        w_err = np.sum(self.gamma * E_error * self.mask)
        N_eff = np.sum(self.mask)
        
        new_alpha = (N_eff + 2*self.a_alpha) / (w_err + 2*self.b_alpha + 1e-10)
        # Smooth update
        self.alpha = 0.5 * self.alpha + 0.5 * new_alpha
        self.alpha = np.clip(self.alpha, 1.0, 1e6)

        # B. Gamma (Outliers) - nu=0
        self.gamma = 1.0 / (self.alpha * E_error + 1e-9)
        self.gamma = np.clip(self.gamma, 0.0, 10.0)
        self.gamma *= self.mask

        # C. Lambda (Image Prior)
        dy, dx = compute_gradients(self.u)
        E_du2 = (dy**2 + dx**2) + 4.0 * var_u
        self.lambda_u = (1.0 + 2*self.a_lambda) / (E_du2 + 2*self.b_lambda + 1e-12)
        # Limit lambda to prevent total smoothing
        self.lambda_u = np.clip(self.lambda_u, 1e-3, 1e3)

        # D. Beta (Kernel Prior)
        E_h2 = self.h**2 + var_h
        self.beta = (1.0 + 2*self.a_beta) / (E_h2 + 2*self.b_beta + 1e-12)