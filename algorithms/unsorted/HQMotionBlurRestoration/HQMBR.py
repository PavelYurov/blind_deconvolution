import time
import numpy as np
from ..base import DeconvolutionAlgorithm

from .convolve import create_line_psf
from .deblur import computeLocalPrior, updatePsi, computeL, updatef, save_mask_as_image
from .helpers import open_image, write_image, kernel_from_image

class HQMBR(DeconvolutionAlgorithm):
    MAX_ITER = 5
    VARS = {
        'gamma': 2, # First iteration, then double
        'lambda1': 0.5, # [0.002, 0.5]
        'k1': 1.1, # [1.1, 1.4]
        'lambda2': 25, # [10, 25]
        'k2': 1.5,
    }
    def __init__(self, param, predict_psf, MAX_ITER = 5, VARS = None):
        super().__init__('HQMBR')
        self.param = param
        if VARS is not None:
            self.VARS = VARS
        self.MAX_ITER = MAX_ITER
        self.predict_psf = predict_psf

    def process(self, image):
        I = np.atleast_3d(np.array(image,dtype=np.float64))
        # Create inicial kernel
        # f = create_line_psf(0, 1, (27, 27)) #-np.pi/4
        f = self.predict_psf

        output_kernel = f

        # Extracted rows, min(width, height)/2 - 12 for example
        n_rows = 260

        # Initialize Latent image with observed image I
        L = I.copy() 
        nL = I.copy() 

        # Compute Omega region with t = 5
        O_THRESHOLD = 5
        s = time.time()
        M = np.zeros_like(I)
        for i in range(I.shape[2]):
            M[:, :, i] = computeLocalPrior(I[:, :, i], f.shape, O_THRESHOLD)
            # save_mask_as_image(M[:, :, i], f"picasso_lp{i}.png")
        # print(f"computeLocalPrior took {time.time() - s}s")

        # Calculate the observed image gradients for each channel
        I_d = [np.gradient(I[:, :, i], axis=(1, 0)) for i in range(I.shape[2])]

        # Initialize Psi
        Psi = [[np.zeros(L.shape[:2]), np.zeros(L.shape[:2])] for _ in range(L.shape[2])]
        nPsi = [[np.zeros(L.shape[:2]), np.zeros(L.shape[:2])] for _ in range(L.shape[2])]

        tempao = 0
        iterations = 0
        while iterations < self.MAX_ITER:
            self.VARS['gamma'] = 2
            delta = 5000
            iters = 0
            while iters < 1:
                s = time.time()
                for i in range(L.shape[2]):
                    L_d = np.gradient(L[:, :, i], axis=(1, 0))
                    nPsi[i] = updatePsi(I_d[i], L_d, M[:, :, i], self.VARS['lambda1'], self.VARS['lambda2'], self.VARS['gamma'])
                    nL[:, :, i] = computeL(L[:, :, i], I[:, :, i], f, nPsi[i], self.VARS['gamma'])
                deltaL = nL - L
                delta = np.linalg.norm(deltaL)
                # print(delta)
                L = nL.copy()
                nPsi = Psi.copy()
                self.VARS['gamma'] *= 2
                iters += 1
            write_image(f'restored\\{iterations}.png', L.copy())
            # write_image(f'picasso_kernel{iterations}.png', f.copy()*(255/np.max(f)))
            f = updatef(L, I, f, n_rows=n_rows, k_cut_ratio=0)

            output_kernel = output_kernel + f

            tempao_atual = time.time() - s
            tempao += tempao_atual
            # print(f'{iterations}: {tempao}s')
            self.VARS['lambda1'] /= self.VARS['k1']
            self.VARS['lambda2'] /= self.VARS['k2']
            iterations += 1
        
        # L = L*255.0
        L = np.round(L).astype(np.int16)
        output_kernel = np.round(output_kernel*255.0).astype(np.int16)
        # print(np.max(output_kernel))
        return L, output_kernel
    
    def change_param(self, param):
        # super().change_param(param)
        self.MAX_ITER = param['max_iter']
        self.VARS['gamma'] = param['gamma']
        self.VARS['lambda1'] = param['lambda1']
        self.VARS['k1'] = param['k1']
        self.VARS['lambda2'] = param['lambda2']
        self.VARS['k2'] = param['k2']

        self.predict_psf = create_line_psf(param['angle'], 1, (param['size'],param['size']))
        # print("complite changes: ", param)

    def get_param(self):
        super().get_param()
        return [('max_iter',self.MAX_ITER),
                ('gamma',self.VARS['gamma']),
                ('lambda1',self.VARS['lambda1']),
                ('k1',self.VARS['k1']),
                ('lambda2',self.VARS['lambda2']),
                ('k2',self.VARS['k2'])]
    

        


            