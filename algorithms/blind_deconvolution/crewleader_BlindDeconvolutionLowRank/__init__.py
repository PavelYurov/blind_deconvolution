# https://github.com/crewleader/BlindDeconvolutionLowRank
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import matlab.engine

from algorithms.base import DeconvolutionAlgorithm

ALGORITHM_NAME = "crewleader_BlindDeconvolutionLowRank"
SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


class CrewleaderBlindDeconvolutionLowRank(DeconvolutionAlgorithm):
    def __init__(
        self,
        psize_y: int = 31,
        psize_x: int = 31,
        gamma: float = 0.6,
        p: float = 0.0,
        beta: float = 1.0 / 320.0,
        thr_e: float = 1.0 / 1500.0,
        num_try: int = 120,
        base_psf: int = 1,
        resize_step: float = 2 ** 0.5,
        alpha_multiplier: float = 2.0,
        min_alpha: float = 0.123,
        num_scale: int | None = None,
        lambda_: float | None = None,
        delta: float | None = None,
        transform_type: str = "projective",
    ):
        super().__init__(ALGORITHM_NAME)
        self._eng = matlab.engine.start_matlab()
        self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
        self._eng.cd(os.path.join(SOURCE_PATH, "code"), nargout=0)

        self.psize_y = int(psize_y)
        self.psize_x = int(psize_x)
        self.gamma = float(gamma)
        self.p = float(p)
        self.beta = float(beta)
        self.thr_e = float(thr_e)
        self.num_try = int(num_try)
        self.base_psf = int(base_psf)
        self.resize_step = float(resize_step)
        self.alpha_multiplier = float(alpha_multiplier)
        self.min_alpha = float(min_alpha)
        self.num_scale = (
            int(num_scale) if num_scale is not None else None
        )
        
        self.lambda_ = float(lambda_) if lambda_ is not None else (0.088 ** 2 * self.gamma)
        self.delta = float(delta) if delta is not None else (0.04 * self.gamma)
        self.transform_type = str(transform_type)

    def change_param(self, param: Any):
        if not isinstance(param, dict):
            return

        if "psize_y" in param:
            self.psize_y = int(param["psize_y"])
        if "psize_x" in param:
            self.psize_x = int(param["psize_x"])
        if "gamma" in param:
            self.gamma = float(param["gamma"])
        if "p" in param:
            self.p = float(param["p"])
        if "beta" in param:
            self.beta = float(param["beta"])
        if "thr_e" in param:
            self.thr_e = float(param["thr_e"])
        if "num_try" in param:
            self.num_try = int(param["num_try"])
        if "base_psf" in param:
            self.base_psf = int(param["base_psf"])
        if "resize_step" in param:
            self.resize_step = float(param["resize_step"])
        if "alpha_multiplier" in param:
            self.alpha_multiplier = float(param["alpha_multiplier"])
        if "min_alpha" in param:
            self.min_alpha = float(param["min_alpha"])
        if "num_scale" in param:
            self.num_scale = int(param["num_scale"])
        if "lambda" in param:
            self.lambda_ = float(param["lambda"])
        if "delta" in param:
            self.delta = float(param["delta"])
        if "transform_type" in param:
            self.transform_type = str(param["transform_type"])

    def get_param(self) -> list[str, Any]:
        return [
            ("psize_y", self.psize_y),
            ("psize_x", self.psize_x),
            ("gamma", self.gamma),
            ("p", self.p),
            ("beta", self.beta),
            ("thr_e", self.thr_e),
            ("num_try", self.num_try),
            ("base_psf", self.base_psf),
            ("resize_step", self.resize_step),
            ("alpha_multiplier", self.alpha_multiplier),
            ("min_alpha", self.min_alpha),
            ("num_scale", self.num_scale),
            ("lambda", self.lambda_),
            ("delta", self.delta),
            ("transform_type", self.transform_type),
        ]

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        image_gray = image_gray.astype(np.float64) / 255.0

        I_mat = matlab.double(image_gray.tolist())
        self._eng.workspace["y_py"] = I_mat

        self._eng.workspace["psize_y"] = float(self.psize_y)
        self._eng.workspace["psize_x"] = float(self.psize_x)
        self._eng.workspace["gamma_param"] = float(self.gamma)
        self._eng.workspace["p_param"] = float(self.p)
        self._eng.workspace["beta_param"] = float(self.beta)
        self._eng.workspace["thr_e_param"] = float(self.thr_e)
        self._eng.workspace["num_try_param"] = float(self.num_try)
        self._eng.workspace["base_psf_param"] = float(self.base_psf)
        self._eng.workspace["resize_step_param"] = float(self.resize_step)
        self._eng.workspace["alpha_mult_param"] = float(self.alpha_multiplier)
        self._eng.workspace["min_alpha_param"] = float(self.min_alpha)
        self._eng.workspace["lambda_param"] = float(self.lambda_)
        self._eng.workspace["delta_param"] = float(self.delta)
        self._eng.workspace["transform_type_param"] = self.transform_type

        if self.num_scale is not None:
            self._eng.workspace["num_scale_param"] = float(self.num_scale)
            self._eng.eval("params.num_scale = num_scale_param;", nargout=0)
        else:
            self._eng.eval(
                "params.psize_y = psize_y; "
                "params.psize_x = psize_x; "
                "params.resize_step = resize_step_param; "
                "params.num_scale = floor(log(min([params.psize_y params.psize_x])/3)/log(params.resize_step));",
                nargout=0,
            )

        self._eng.eval(
            "params.psize_y = psize_y;"
            "params.psize_x = psize_x;"
            "params.gamma = gamma_param;"
            "params.p = p_param;"
            "params.beta = beta_param;"
            "params.thr_e = thr_e_param;"
            "params.num_try = num_try_param;"
            "params.base_psf = base_psf_param;"
            "params.resize_step = resize_step_param;"
            "params.alpha_multiplier = alpha_mult_param;"
            "params.min_alpha = min_alpha_param;"
            "params.display = 0;"
            "params.lambda = lambda_param;"
            "params.delta = delta_param;"
            "params.transform_type = transform_type_param;",
            nargout=0,
        )

        self._eng.eval(
            """
            y = y_py;
            
            psize_y_loc = params.psize_y;
            psize_x_loc = params.psize_x;
            K0 = ones(psize_y_loc, psize_x_loc);
            K0 = K0 / sum(K0(:));

            [vsize_y, vsize_x] = size(y);
            fsize_y = vsize_y + psize_y_loc - 1;
            fsize_x = vsize_x + psize_x_loc - 1;

            varea = true(vsize_y, vsize_x);
            varea = padarray(varea, [psize_y_loc-1, psize_x_loc-1], false, 'post');

            x = padarray(y, [psize_y_loc-1, psize_x_loc-1], 'replicate', 'post');

            lowrank_img = x;
            sparsity_img = false(size(x));

            alpha = params.min_alpha;
            for it = 1:params.num_try
                % шаг по x (solve_image_irls есть в исходном коде)
                x = solve_image_irls(x, K0, y, lowrank_img, sparsity_img, ...
                                     params.gamma, alpha, params.p, ...
                                     200, 1e-6, params.thr_e);

                K0 = solve_psf_constrained(K0, y, x, params.beta, varea);

                alpha = max(params.min_alpha, alpha * exp(-log(2)/params.num_try));
            end

            x_py = x;
            k_py = K0;
            """,
            nargout=0,
        )

        x_mat = self._eng.workspace["x_py"]
        k_mat = self._eng.workspace["k_py"]

        x_np = np.array(x_mat, dtype=np.float64)
        x_np = np.clip(x_np, 0.0, 1.0)
        x_uint8 = (x_np * 255.0).astype(np.uint8)

        if x_uint8.ndim == 2:
            x_bgr = cv2.cvtColor(x_uint8, cv2.COLOR_GRAY2BGR)
        else:
            x_rgb = x_uint8
            x_bgr = cv2.cvtColor(x_rgb, cv2.COLOR_RGB2BGR)

        kernel = np.array(k_mat, dtype=np.float64)

        return x_bgr, kernel

    def __del__(self):
        try:
            self._eng.quit()
        except Exception:
            pass
