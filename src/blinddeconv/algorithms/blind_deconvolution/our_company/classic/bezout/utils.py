import numpy as np
from numpy.linalg import svd
from scipy.fft import fft, ifft

def construct_bezout_matrix(p: np.ndarray, q: np.ndarray, n: int) -> np.ndarray:
    """
    Строит матрицу Безу.
    """
    p_coeffs = np.zeros(n, dtype=complex)
    q_coeffs = np.zeros(n, dtype=complex)
    
    lim_p = min(len(p), n)
    lim_q = min(len(q), n)
    p_coeffs[:lim_p] = p[:lim_p]
    q_coeffs[:lim_q] = q[:lim_q]

    bezout = np.zeros((n, n), dtype=complex)
    
    # Векторизованное построение по Барнетту для стандартного базиса
    # b_ij = sum (p_k * q_{i+j+1-k} - p_{i+j+1-k} * q_k)
    
    for i in range(n):
        for j in range(n):
            limit = min(i, n - 1 - j)
            val = 0j
            for k in range(limit + 1):
                idx2 = i + j + 1 - k
                if idx2 < n:
                    val += p_coeffs[k] * q_coeffs[idx2] - p_coeffs[idx2] * q_coeffs[k]
            bezout[i, j] = val
            
    return bezout

def build_generalized_bezout(p: np.ndarray, q_list: list, n: int) -> np.ndarray:
    blocks = []
    for q in q_list:
        blocks.append(construct_bezout_matrix(p, q, n))
    return np.vstack(blocks)

def estimate_rank(matrix: np.ndarray, epsilon: float) -> int:
    """
    Оценка ранга матрицы.
    """
    # compute_uv=False значительно быстрее и экономит память
    s = svd(matrix, compute_uv=False)
    rank = np.sum(s > epsilon)
    return int(rank)

def poly_div_fft(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    n_fft = len(num)
    num_f = fft(num, n=n_fft)
    den_f = fft(den, n=n_fft)
    
    tol = 1e-12
    den_f[np.abs(den_f) < tol] = tol
    
    quotient_f = num_f / den_f
    quotient = ifft(quotient_f)
    return quotient