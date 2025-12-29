# https://github.com/mujib2020/Non-blind-and-Blind-Deconvolution-under-Poisson-noise
from __future__ import annotations

import os
from typing import Any, Literal

import cv2
import matlab.engine
import numpy as np

from algorithms.base import DeconvolutionAlgorithm

ALGORITHM_NAME = "mujib2020_Non_blind_and_Blind_Deconvolution_under_Poisson_noise"
SOURCE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "source",
)

class Mujib2020NonBlindAndBlindDeconvolutionUnderPoissonNoise(DeconvolutionAlgorithm):
    def __init__(
        self,
        mode: Literal["EM", "FOTV"] = "FOTV",
        
        kernel_height: int = 11,
        kernel_width: int = 11,
        
        alpha: float = 1.0,
        beta: float = 110.0,
        mu1: float = 1e-1,
        mu2: float = 1.0,
        maxit: int = 150,
    ):
        super().__init__(ALGORITHM_NAME)

        self._eng = matlab.engine.start_matlab()
        self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
        self._eng.cd(SOURCE_PATH, nargout=0)

        self.mode = mode
        self.kernel_height = int(kernel_height)
        self.kernel_width = int(kernel_width)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.maxit = int(maxit)
        
    def change_param(self, param: Any):
        if not isinstance(param, dict):
            return

        if "mode" in param:
            if param["mode"] in ("EM", "FOTV"):
                self.mode = param["mode"]

        if "kernel_height" in param:
            self.kernel_height = int(param["kernel_height"])

        if "kernel_width" in param:
            self.kernel_width = int(param["kernel_width"])

        if "alpha" in param:
            self.alpha = float(param["alpha"])

        if "beta" in param:
            self.beta = float(param["beta"])

        if "mu1" in param:
            self.mu1 = float(param["mu1"])

        if "mu2" in param:
            self.mu2 = float(param["mu2"])

        if "maxit" in param:
            self.maxit = int(param["maxit"])

    def get_param(self) -> list[str, Any]:
        return [
            ("mode", self.mode),
            ("kernel_height", self.kernel_height),
            ("kernel_width", self.kernel_width),
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("mu1", self.mu1),
            ("mu2", self.mu2),
            ("maxit", self.maxit),
        ]

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        image_gray = image_gray.astype(np.float64)

        if image_gray.max() > 0:
            image_gray = image_gray / image_gray.max()

        f_py = image_gray
        f_mat = matlab.double(f_py.tolist())
        self._eng.workspace["f_py"] = f_mat

        self._eng.workspace["Mh_py"] = float(self.kernel_height)
        self._eng.workspace["Nh_py"] = float(self.kernel_width)

        self._eng.workspace["alpha_py"] = float(self.alpha)
        self._eng.workspace["beta_py"] = float(self.beta)
        self._eng.workspace["mu1_py"] = float(self.mu1)
        self._eng.workspace["mu2_py"] = float(self.mu2)
        self._eng.workspace["maxit_py"] = float(self.maxit)

        self._eng.eval(
            "pm = struct("
            "'alpha', alpha_py, "
            "'beta', beta_py, "
            "'mu1', mu1_py, "
            "'mu2', mu2_py, "
            "'maxit', maxit_py "
            ");",
            nargout=0,
        )

        if self.mode == "EM":
            self._eng.eval(
                "[u_py, H_py, output_py] = EM_Blind_Deconv(f_py, Mh_py, Nh_py, pm);",
                nargout=0,
            )
        else:
            self._eng.eval(
                "[u_py, H_py, output_py] = FOTV_deconv_blind(f_py, Mh_py, Nh_py, pm);",
                nargout=0,
            )

        u_mat = self._eng.workspace["u_py"]
        H_mat = self._eng.workspace["H_py"]

        u_np = np.array(u_mat, dtype=np.float64)
        u_np = np.clip(u_np, 0.0, 1.0)

        u_uint8 = (u_np * 255.0).astype(np.uint8)
        if u_uint8.ndim == 2:
            u_bgr = cv2.cvtColor(u_uint8, cv2.COLOR_GRAY2BGR)
        else:
            u_bgr = cv2.cvtColor(u_uint8, cv2.COLOR_RGB2BGR)

        kernel = np.array(H_mat, dtype=np.float64)

        return u_bgr, kernel

    def __del__(self):
        try:
            self._eng.quit()
        except Exception:
            pass
