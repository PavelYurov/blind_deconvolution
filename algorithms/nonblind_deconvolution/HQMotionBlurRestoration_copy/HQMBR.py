import time
import numpy as np
from ...base import DeconvolutionAlgorithm

from .convolve import create_line_psf, gradient
from .deblur import computeLocalPrior, updatePsi, computeL, updatef, save_mask_as_image
from .helpers import open_image, write_image, kernel_from_image
import cv2 as cv

class HQMBR(DeconvolutionAlgorithm):
    MAX_ITER = 5
    VARS = {
        'gamma': 2, # First iteration, then double
        'lambda1': 0.5, # [0.002, 0.5]
        'k1': 1.1, # [1.1, 1.4]
        'lambda2': 25, # [10, 25]
        'k2': 1.5,
    }
    def __init__(self, predict_psf, MAX_ITER = 5, gamma=2,lambda1=0.5,lambda2=25,k1=1.1,k2=1.5,O_THRESHOLD=5, inner_iter = 3, F_THRESHOLD = 0.1, L_THRESHOLD = 0.01,PHI_THRESHOLD=0.01):
        super().__init__('HQMBR_FIX')
        self.param = 1
        self.VARS={
            'gamma': gamma,
            'lambda1': lambda1,
            'k1': k1,
            'lambda2': lambda2,
            'k2': k2,
        }
        self.O_THRESHOLD = O_THRESHOLD
        self.F_THRESHOLD = F_THRESHOLD
        self.L_THRESHOLD = L_THRESHOLD
        self.PHI_THRESHOLD = PHI_THRESHOLD
        self.MAX_ITER = MAX_ITER
        self.predict_psf = predict_psf
        self.inner_iter = inner_iter
        self.gamma = gamma

    def process(self, image):

        I = np.atleast_3d(np.array(image,dtype=np.float64))

        orig_h, orig_w, orig_a = I.shape
        I = cv.copyMakeBorder(I,orig_h//2,orig_h//2,orig_w//2,orig_w//2,borderType=cv.BORDER_REFLECT_101)#cv.BORDER_WRAP

        I = np.atleast_3d(np.array(I,dtype=np.float64))
        # Create inicial kernel
        # f = create_line_psf(0, 1, (27, 27)) #-np.pi/4
        f = self.predict_psf
        nf = f.copy()

        output_kernel = f

        # Extracted rows, min(width, height)/2 - 12 for example
        # n_rows = 260
        h,w,a = I.shape
        n_rows = round(min(w, h)/2) - 12

        # Initialize Latent image with observed image I
        L = I.copy() 
        nL = I.copy() 

        # Compute Omega region with t = 5
        O_THRESHOLD = self.O_THRESHOLD
        s = time.time()
        M = np.zeros_like(I)
        for i in range(I.shape[2]):
            M[:, :, i] = computeLocalPrior(I[:, :, i], f.shape, O_THRESHOLD)
            save_mask_as_image(M[:, :, i], f"restored\\mask.png")
        # print(f"computeLocalPrior took {time.time() - s}s")

        # Calculate the observed image gradients for each channel
        # I_d = [np.gradient(I[:, :, i], axis=(1, 0)) for i in range(I.shape[2])]
        I_d = [gradient(I[:, :, i]) for i in range(I.shape[2])]
        
        # Initialize Psi
        Psi = [[np.zeros(L.shape[:2]), np.zeros(L.shape[:2])] for _ in range(L.shape[2])]
        nPsi = [[np.zeros(L.shape[:2]), np.zeros(L.shape[:2])] for _ in range(L.shape[2])]

        tempao = 0
        iterations = 0
        norm_deltaf = 5000
        while iterations < self.MAX_ITER and (norm_deltaf > self.F_THRESHOLD):
            self.VARS['gamma'] = 2
            norm_delta = 5000
            norm_deltaPsi = 5000
            iters = 0
            while iters < self.inner_iter and (norm_delta > self.L_THRESHOLD or norm_deltaPsi > self.PHI_THRESHOLD):
                s = time.time()
                for i in range(L.shape[2]):
                    # L_d = np.gradient(L[:, :, i], axis=(1, 0))
                    L_d = gradient(L[:, :, i])
                    nPsi[i] = updatePsi(I_d[i], L_d, M[:, :, i], self.VARS['lambda1'], self.VARS['lambda2'], self.VARS['gamma'])
                    nL[:, :, i] = computeL(L[:, :, i], I[:, :, i], f, nPsi[i], self.VARS['gamma'])
                deltaL = nL/255.0 - L/255.0
                norm_delta = np.linalg.norm(deltaL)
                deltaPsi = nPsi[0][0]/255.0 - Psi[0][0]/255.0 + nPsi[0][1]/255.0 - Psi[0][1]/255.0
                norm_deltaPsi = np.linalg.norm(deltaPsi)
                print(f"DL = {norm_delta}  DPsi = {norm_deltaPsi}")
                L = nL.copy()
                Psi = nPsi.copy()
                self.VARS['gamma'] *= 2
                iters += 1
            # nL = nL / np.max(nL)*255
            # L = nL.copy()
            self.VARS['gamma'] = self.gamma

            write_image(f'restored\\{iterations}.png', L.copy())
            write_image(f'restored\\{iterations}_k.png', f.copy()*(255/np.max(f)))
            f = updatef(L, I, f, n_rows=n_rows, k_cut_ratio=0)

            output_kernel =  f.copy()*(255/np.max(f))

            tempao_atual = time.time() - s
            tempao += tempao_atual
            # print(f'{iterations}: {tempao}s')
            self.VARS['lambda1'] /= self.VARS['k1']
            self.VARS['lambda2'] /= self.VARS['k2']
            iterations += 1
            deltaf = f - nf
            nf = f.copy()
            norm_deltaf = np.linalg.norm(deltaf)
            print(f"Df = {norm_deltaf}")
            print(f"iter={iterations}")

        
        # L = L*255.0
        print(np.mean(L))
        L = np.round(L).astype(np.int16)
        output_kernel = np.round(output_kernel).astype(np.int16)
        # print(np.max(output_kernel))
        L = L[orig_h//2:(orig_h+orig_h//2),orig_w//2:(orig_w+orig_w//2),:]
        return L, output_kernel
    
    def change_param(self, param):
        # super().change_param(param)
        self.MAX_ITER = param['max_iter']
        self.VARS['gamma'] = param['gamma']
        self.gamma = param['gamma']
        self.VARS['lambda1'] = param['lambda1']
        self.VARS['k1'] = param['k1']
        self.VARS['lambda2'] = param['lambda2']
        self.VARS['k2'] = param['k2']
        self.O_THRESHOLD = param["O_THRESHOLD"]
        self.F_THRESHOLD = param["F_THRESHOLD"]
        self.L_THRESHOLD = param["L_THRESHOLD"]
        self.PHI_THRESHOLD = param["PHI_THRESHOLD"]
        self.inner_iter = param["inner_iter"]
        self.predict_psf = param['predict_psf']
        # self.predict_psf = create_line_psf(param['angle'], 1, (param['size'],param['size']))
        # print("complite changes: ", param)

    def get_param(self):
        super().get_param()
        return [('max_iter',self.MAX_ITER),
                ('gamma',self.VARS['gamma']),
                ('lambda1',self.VARS['lambda1']),
                ('k1',self.VARS['k1']),
                ('lambda2',self.VARS['lambda2']),
                ('k2',self.VARS['k2']),
                ('O_THRESHOLD', self.O_THRESHOLD),
                ('F_THRESHOLD',self.F_THRESHOLD),
                ('L_THRESHOLD',self.L_THRESHOLD),
                ('PHI_THRESHOLD',self.PHI_THRESHOLD),
                ('inner_iter',self.inner_iter),
                ]
    

        


            