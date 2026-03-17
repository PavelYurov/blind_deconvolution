"""
Solvers for Bayesian Sparse Blind Deconvolution (NIG-Prior).

Implements the MCMC sampling steps described in Civek & Ertin (2022).
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.special import gammaln
from numpy.fft import fft2, ifft2
from .utils import psf2otf, circshift
from typing import Tuple

# ------------------------------------------------------------------------------
# 1. Hyperparameters Sampling (Section III-B-1)
# ------------------------------------------------------------------------------

def update_hyperparams(
    sigma_sq_x: np.ndarray,
    alpha_cur: float
) -> Tuple[float, float]:
    """
    Sample alpha_x and beta_x.
    
    beta_x | alpha_x, sigma_x^2 ~ Gamma (Eq 16)
    alpha_x | beta_x, sigma_x^2 ~ Non-standard (Eq 15), sampled via Slice Sampling.
    """
    K = sigma_sq_x.size
    
    # 1. Update Beta (Eq 16)
    # The paper denotes G(.; alpha, beta) as Gamma distribution.
    # Posterior params: alpha_new = K * alpha_cur, beta_new = sum(1/sigma_x^2)
    # Note: numpy.random.gamma takes (shape, scale), where scale = 1/rate.
    rate_beta = np.sum(1.0 / (sigma_sq_x + 1e-12))
    shape_beta = K * alpha_cur
    
    # Add small epsilon to scale to prevent zero division
    beta_new = np.random.gamma(shape_beta, 1.0 / (rate_beta + 1e-12))
    
    # 2. Update Alpha (Eq 15) using Slice Sampling
    # log p(alpha) = K*alpha*ln(beta) - K*ln(Gamma(alpha)) - (alpha+1)*sum(ln(sigma)) - ln(alpha)
    # (assuming Jeffreys prior 1/alpha -> -ln(alpha))
    
    sum_ln_sigma = np.sum(np.log(sigma_sq_x + 1e-12))
    
    def log_prob_alpha(a):
        if a <= 1e-6: return -1e10
        t1 = K * a * np.log(beta_new + 1e-12)
        t2 = -K * gammaln(a)
        t3 = -(a + 1) * sum_ln_sigma
        prior = -np.log(a)
        return t1 + t2 + t3 + prior

    # Slice sampling implementation
    w = 0.1 # Window size
    log_y_threshold = log_prob_alpha(alpha_cur) + np.log(np.random.rand() + 1e-12)
    
    # Stepping out
    u = np.random.rand()
    L = alpha_cur - w * u
    R = alpha_cur + w * (1 - u)
    
    # Limit range to avoid instability
    while L > 1e-4 and log_prob_alpha(L) > log_y_threshold: L -= w
    while R < 100.0 and log_prob_alpha(R) > log_y_threshold: R += w
    
    L = max(L, 1e-4)
    
    # Shrinking
    alpha_new = alpha_cur
    for _ in range(20):
        prop = L + np.random.rand() * (R - L)
        if log_prob_alpha(prop) > log_y_threshold:
            alpha_new = prop
            break
        if prop < alpha_cur:
            L = prop
        else:
            R = prop
            
    return alpha_new, beta_new

# ------------------------------------------------------------------------------
# 2. Latent Variances Sampling (Section III-B-2)
# ------------------------------------------------------------------------------

def sample_latent_variances(
    x: np.ndarray, 
    alpha_x: float, 
    beta_x: float
) -> np.ndarray:
    """
    Sample sigma_x^2 from Inverse-Gamma distribution.
    Eq (17): IG(alpha_x + 0.5, beta_x + x_n^2 / 2)
    """
    shape = alpha_x + 0.5
    rate = beta_x + 0.5 * (x**2)
    
    # IG(a, b) ~ 1 / Gamma(a, rate=b)
    # Using numpy: scale = 1/rate
    inv_samples = np.random.gamma(shape, 1.0 / (rate + 1e-12))
    return 1.0 / (inv_samples + 1e-12)

# ------------------------------------------------------------------------------
# 3. Image Sampling (Section III-B-3)
# ------------------------------------------------------------------------------

def sample_image_cg(
    y: np.ndarray, 
    h: np.ndarray, 
    sigma_sq_x: np.ndarray, 
    sigma_sq_v: float
) -> np.ndarray:
    """
    Sample x from N(mu_x, Sigma_x) using Conjugate Gradient with perturbation.
    Eq (18), (19).
    System: (1/sigma_v^2 H^T H + D^-1) x = (1/sigma_v^2) H^T y + noise
    """
    H, W = y.shape
    
    # Prepare OTF
    # psf2otf centers the kernel to prevent image shifting
    otf = psf2otf(h, (H, W))
    otf_conj = np.conj(otf)
    otf_sq = np.abs(otf)**2
    
    inv_sigma_v = 1.0 / sigma_sq_v
    inv_sigma_x = 1.0 / (sigma_sq_x + 1e-12)
    
    # Linear Operator A
    def mv(v_flat):
        v = v_flat.reshape(H, W)
        # Term 1: (1/sigma_v^2) H^T H v
        # Convolution with H, then H^T is equivalent to multiplication by |H(f)|^2 in freq
        term1 = np.real(ifft2(otf_sq * fft2(v))) * inv_sigma_v
        # Term 2: D^-1 v
        term2 = inv_sigma_x * v
        return (term1 + term2).ravel()

    A_op = LinearOperator((H*W, H*W), matvec=mv, dtype=np.float64)
    
    # RHS Construction with Perturbation for sampling
    # b = (1/sigma_v^2) H^T y + (1/sigma_v) H^T n1 + D^(-1/2) n2
    
    # 1. Mean component
    HT_y = np.real(ifft2(otf_conj * fft2(y))) * inv_sigma_v
    
    # 2. Perturbation
    n1 = np.random.randn(H, W) # Noise related to observation
    n2 = np.random.randn(H, W) # Noise related to prior
    
    pert1 = np.real(ifft2(otf_conj * fft2(n1))) * np.sqrt(inv_sigma_v)
    pert2 = np.sqrt(inv_sigma_x) * n2
    
    rhs = (HT_y + pert1 + pert2).ravel()
    
    # CG Solve
    # Use x=0 as start or random
    x_flat, _ = cg(A_op, rhs, atol=1e-6, maxiter=75)
    
    return x_flat.reshape(H, W)

# ------------------------------------------------------------------------------
# 4. Kernel Sampling (Section III-B-4)
# ------------------------------------------------------------------------------

def sample_kernel(
    y: np.ndarray,
    x: np.ndarray,
    sigma_sq_v: float,
    sigma_sq_gamma: float, # Prior variance for kernel
    kernel_shape: Tuple[int, int],
    precomputed_indices: np.ndarray
) -> np.ndarray:
    """
    Sample kernel h from N(mu_gamma, Sigma_gamma).
    Eq (20), (21).
    Sigma_inv = (1/sigma_v^2) X^T X + (1/sigma_gamma^2) I
    mu = Sigma (1/sigma_v^2) X^T y
    """
    H, W = y.shape
    kh, kw = kernel_shape
    K = kh * kw
    
    # Frequency domain calculations
    Xf = fft2(x)
    Yf = fft2(y)
    
    # Autocorrelation of x (top row of Toeplitz matrix X^T X)
    # The result is circularly shifted so lag 0 is at (0,0)
    auto_corr = np.real(ifft2(np.abs(Xf)**2))
    
    # Cross-correlation X^T y
    # Corresponds to B^T y in Eq (21) if A=I
    cross_corr = np.real(ifft2(np.conj(Xf) * Yf))
    
    # Construct dense Sigma inverse matrix (K x K)
    # Using precomputed indices to map lags to matrix positions
    # auto_corr shape is (H, W). precomputed_indices stores (y, x) coords.
    
    # Flatten autocorrelation for indexing
    Sigma_inv = auto_corr[precomputed_indices[:, :, 0], precomputed_indices[:, :, 1]]
    
    # Scale by noise variance
    Sigma_inv /= sigma_sq_v
    
    # Add prior
    Sigma_inv += (1.0 / sigma_sq_gamma) * np.eye(K)
    
    # Add jitter for numerical stability
    Sigma_inv += 1e-8 * np.eye(K)
    
    # Construct RHS vector
    # We need to extract the relevant lags from cross_corr corresponding to the kernel support.
    # Since we use psf2otf which centers the kernel at (0,0) in freq domain logic,
    # we need to be careful.
    # In standard convolution y = h * x, the value y[i] is sum(h[k] * x[i-k]).
    # The derivative wrt h[k] involves correlation at lag k.
    # Since we assume h is defined on grid 0..kh, 0..kw, we just take the top-left crop
    # because standard FFT places lag 0 at (0,0) and positive lags increase indices.
    
    rhs_vec = np.zeros(K)
    coords = [(u, v) for u in range(kh) for v in range(kw)]
    for k, (u, v) in enumerate(coords):
        rhs_vec[k] = cross_corr[u, v]
        
    rhs_vec /= sigma_sq_v
    
    # Sample using Cholesky
    try:
        L = np.linalg.cholesky(Sigma_inv)
        # Solve Sigma * mu = rhs => L * L.T * mu = rhs
        t = np.linalg.solve(L, rhs_vec)
        mu = np.linalg.solve(L.T, t)
        
        # Add noise: h = mu + L^-T * z
        z = np.random.randn(K)
        noise = np.linalg.solve(L.T, z)
        h_flat = mu + noise
    except np.linalg.LinAlgError:
        # Fallback to standard solve if not positive definite (rare with jitter)
        h_flat = np.linalg.solve(Sigma_inv, rhs_vec)
        
    return h_flat.reshape(kh, kw)

# ------------------------------------------------------------------------------
# 5. Noise Variance Sampling via Marginalization (Section III-C-4 / Eq 27)
# ------------------------------------------------------------------------------

def sample_noise_variance_marginalized(
    current_sigma_sq_v: float,
    y: np.ndarray,
    x: np.ndarray,
    sigma_sq_gamma: float,
    kernel_shape: Tuple[int, int],
    precomputed_indices: np.ndarray
) -> float:
    """
    Sample sigma_v^2 from p(sigma_v^2 | y, x).
    Eq (27) marginalizes out gamma (h) to allow better mixing.
    
    p(sigma^2 | ...) \propto (1/sigma^2)^(N/2) * |Sigma_gamma|^0.5 * exp(0.5 * mu^T Sigma^-1 mu - ... )
    
    Since this is a non-standard distribution, we use Slice Sampling (univariate).
    """
    H, W = y.shape
    N = H * W
    K = kernel_shape[0] * kernel_shape[1]
    
    # Precompute terms that don't depend on sigma_v inside the loop (partially)
    Xf = fft2(x)
    Yf = fft2(y)
    
    auto_corr = np.real(ifft2(np.abs(Xf)**2))
    cross_corr = np.real(ifft2(np.conj(Xf) * Yf))
    y_norm_sq = np.sum(y**2)
    
    # Extract needed correlation blocks
    XTX = auto_corr[precomputed_indices[:, :, 0], precomputed_indices[:, :, 1]]
    XTy = np.zeros(K)
    coords = [(u, v) for u in range(kernel_shape[0]) for v in range(kernel_shape[1])]
    for k, (u, v) in enumerate(coords):
        XTy[k] = cross_corr[u, v]
        
    # Log probability function
    def log_prob(sv2):
        if sv2 <= 1e-9: return -1e10
        
        # Calculate Sigma_gamma and mu_gamma for this specific sv2
        # Sigma_gamma^-1 = (1/sv2) XTX + (1/s_gamma) I
        # This is Eq 21
        inv_S = (1.0 / sv2) * XTX + (1.0 / sigma_sq_gamma) * np.eye(K)
        
        try:
            L = np.linalg.cholesky(inv_S)
            # log det(Sigma) = - log det(Sigma^-1) = -2 sum log diag(L)
            log_det_Sigma = -2.0 * np.sum(np.log(np.diag(L)))
            
            # mu = Sigma * (1/sv2) * XTy
            # Solve inv_S * mu = (1/sv2) * XTy
            rhs = (1.0 / sv2) * XTy
            
            # Forward/Back sub
            t = np.linalg.solve(L, rhs) # L t = rhs
            
            # Exponent term: 0.5 * mu^T Sigma^-1 mu
            # mu^T Sigma^-1 mu = (Sigma rhs)^T Sigma^-1 (Sigma rhs) = rhs^T Sigma rhs
            # = rhs^T mu = rhs^T (L^-T t) = (L^-1 rhs)^T t = t^T t
            term_quad = 0.5 * np.dot(t, t)
            
        except np.linalg.LinAlgError:
            return -1e10
            
        # Full Eq 27 log prob
        # (1/sv2)^(N/2) -> -(N/2) log(sv2)
        # prior p(sv2) ~ 1/sv2 -> -log(sv2)
        
        term_likelihood_norm = -(N / 2.0) * np.log(sv2)
        term_residual = - (1.0 / (2.0 * sv2)) * y_norm_sq
        term_determinant = 0.5 * log_det_Sigma
        prior = -np.log(sv2)
        
        return term_likelihood_norm + term_determinant + term_quad + term_residual + prior

    # Slice Sampling
    width = current_sigma_sq_v * 0.1
    log_y = log_prob(current_sigma_sq_v) + np.log(np.random.rand() + 1e-12)
    
    u = np.random.rand()
    L = current_sigma_sq_v - width * u
    R = current_sigma_sq_v + width * (1 - u)
    
    L = max(L, 1e-8)
    
    # Step out restricted
    cnt = 0
    while cnt < 10 and log_prob(L) > log_y: 
        L = max(1e-8, L - width)
        cnt+=1
    cnt = 0
    while cnt < 10 and log_prob(R) > log_y: 
        R += width
        cnt+=1
        
    # Shrink
    new_val = current_sigma_sq_v
    for _ in range(20):
        prop = L + np.random.rand() * (R - L)
        if log_prob(prop) > log_y:
            new_val = prop
            break
        if prop < current_sigma_sq_v: L = prop
        else: R = prop
            
    return new_val

# ------------------------------------------------------------------------------
# 6. Shift Ambiguity Compensation (Section III-D-2)
# ------------------------------------------------------------------------------

def mh_shift_compensation(
    y: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    sigma_sq_x: np.ndarray,
    sigma_sq_v: float,
    sigma_sq_gamma: float,
    precomputed_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Circular Shift Compensation. Eq (29) & (30).
    Proposes a shift (dx, dy) in {-1, 0, 1}.
    Accepts based on ratio of Marginalized Posteriors (integrating out h).
    """
    H, W = y.shape
    K = h.size
    
    # 1. Propose Shift
    shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    idx = np.random.randint(len(shifts))
    shift = shifts[idx] # (dy, dx)
    
    # Proposed State
    # x shifts by +d
    x_prop = circshift(x, shift)
    # sigma_x shifts with x to maintain association
    sigma_x_prop = circshift(sigma_sq_x, shift)
    # h shifts by -d (compensation) - only logically, 
    # but since we integrate out h, we evaluate the score based on x_prop.
    
    # 2. Calculate Log Acceptance Probability
    # The acceptance probability is defined in Eq 30.
    # It requires calculating the Marginal Likelihood score for current and proposed x.
    # Score(x) = 0.5 * log|Sigma_gamma| + 0.5 * mu^T Sigma^-1 mu
    
    def calculate_score(img_x):
        Xf = fft2(img_x)
        Yf = fft2(y)
        auto = np.real(ifft2(np.abs(Xf)**2))
        cross = np.real(ifft2(np.conj(Xf) * Yf))
        
        XTX = auto[precomputed_indices[:, :, 0], precomputed_indices[:, :, 1]]
        XTy = np.zeros(K)
        coords = [(u, v) for u in range(h.shape[0]) for v in range(h.shape[1])]
        for k, (u, v) in enumerate(coords):
            XTy[k] = cross[u, v]
            
        inv_S = (1.0 / sigma_sq_v) * XTX + (1.0 / sigma_sq_gamma) * np.eye(K)
        
        try:
            L = np.linalg.cholesky(inv_S)
            # log |Sigma| = - log |Sigma^-1|
            log_det = -2.0 * np.sum(np.log(np.diag(L)))
            
            # Quadratic term
            rhs = (1.0 / sigma_sq_v) * XTy
            t = np.linalg.solve(L, rhs)
            quad = 0.5 * np.dot(t, t)
            
            return 0.5 * log_det + quad
        except:
            return -np.inf

    score_curr = calculate_score(x)
    score_prop = calculate_score(x_prop)
    
    # Metropolis Step
    # Proposal is symmetric (probability 0.5 for pair)
    # Ratio = exp(score_prop - score_curr)
    
    if np.log(np.random.rand()) < (score_prop - score_curr):
        # Accepted: Return shifted x and shift h in opposite direction
        h_prop = circshift(h, (-shift[0], -shift[1]))
        return x_prop, h_prop, sigma_x_prop
    
    return x, h, sigma_sq_x