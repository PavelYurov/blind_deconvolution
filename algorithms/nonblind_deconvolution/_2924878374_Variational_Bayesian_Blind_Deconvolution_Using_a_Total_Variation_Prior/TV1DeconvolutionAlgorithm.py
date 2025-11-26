from ...base import DeconvolutionAlgorithm
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from scipy.fft import fft2, ifft2
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator, cg

class TV1DeconvolutionAlgorithm(DeconvolutionAlgorithm):
    def create_convolution_matrix(self, psf, shape):
        # Assumes circular boundary conditions
        from scipy.signal import fftconvolve
        kernel = np.pad(psf, [(0, shape[0] - psf.shape[0]), (0, shape[1] - psf.shape[1])], mode='constant')
        kernel = np.roll(kernel, -np.array(psf.shape) // 2, axis=(0, 1))
        return lambda x: fftconvolve(x, kernel, mode='same')

    def apply_psf(self,image, psf):
        return fftconvolve(image, psf, mode='same')

    def precompute_psf_fft(self,psf, image_shape):
        psf_padded = np.pad(psf, [(0, image_shape[0] - psf.shape[0]),
                                (0, image_shape[1] - psf.shape[1])], mode='constant')
        psf_padded = np.roll(psf_padded, -np.array(psf.shape) // 2, axis=(0, 1))
        return fft2(psf_padded)

    def apply_psf_fft_cached(self,image, psf_fft):
        image_fft = fft2(image)
        result_fft = image_fft * psf_fft
        return np.real(ifft2(result_fft))
    
    def compute_weight_matrix(self,x):
        grad_x = np.gradient(x, axis=0)
        grad_y = np.gradient(x, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return 1 / (grad_mag + 1e-3)  # Avoid division by zero
    
    def sparse_laplacian(self,shape):
        n = shape[0] * shape[1]  # Total number of pixels
        diagonals = [-4 * np.ones(n), np.ones(n-1), np.ones(n-1), np.ones(n-shape[1]), np.ones(n-shape[1])]
        offsets = [0, -1, 1, -shape[1], shape[1]]
        L = diags(diagonals, offsets, shape=(n, n), format='csr')
        return L
    
    def solve_regularized(self,A, y, reg_param, laplacian):
        ATA = A.T @ A + reg_param * laplacian
        ATy = A.T @ y
        x = np.linalg.solve(ATA, ATy)
        return x

    def solve_regularized_sparse(self,A_func, y, reg_param, laplacian):
        from scipy.sparse.linalg import LinearOperator

        def matvec(x_vec):
            x_image = x_vec.reshape(y.shape)  # Reshape vector to 2D
            Ax = A_func(x_image)             # Apply convolution in 2D
            Ax_flat = Ax.ravel()             # Flatten Ax to 1D
            return Ax_flat + reg_param * (laplacian @ x_vec)


        shape = y.size
        lin_op = LinearOperator((shape, shape), matvec=matvec)

        # Solve using Conjugate Gradient
        x, _ = cg(lin_op, y.ravel())
        return x.reshape(y.shape)
    
    def process(self, y: np.ndarray):
        # Precompute PSF FFT
        psf_fft = self.precompute_psf_fft(self.psf, y.shape)

        def A_func(img):
            return self.apply_psf_fft_cached(img, psf_fft)

        laplacian = self.sparse_laplacian(y.shape)
        x = y.copy()

        for i in range(self.max_iter):
            print(f"Iteration {i + 1}/{self.max_iter}...")

            def matvec(x_vec):
                x_image = x_vec.reshape(y.shape)
                Ax = A_func(x_image)
                return (Ax.ravel() + self.reg_param * (laplacian @ x_vec))

            lin_op = LinearOperator((y.size, y.size), matvec=matvec)
            x_new, _ = cg(lin_op, y.ravel(), maxiter=200, atol=1e-4)

            if np.linalg.norm(x - x_new.reshape(y.shape)) < 1e-3:
                print(f"Converged at iteration {i + 1}")
                break

            x = x_new.reshape(y.shape)

        return x, self.psf
    
    def __init__(self, psf, max_iter=50, reg_param=1e-2) -> None: 
        """
        Инициализация алгоритма деконволюции.
        
        Аргументы:
            
        """

        super().__init__("TV1DeconvolutionAlgorithm")
        self.timer = -1
        self.psf = psf
        self.max_iter = max_iter
        self.reg_param = reg_param
        pass

    def change_param(self, param):
        self.psf = param['psf'],
        self.max_iter = param['max_iter']
        self.reg_param = param['reg_param']
        pass

    def get_param(self):
        return[
            ('psf', self.psf),
            ('max_iter', self.max_iter),
            ('reg_param', self.reg_param)
        ]
    
