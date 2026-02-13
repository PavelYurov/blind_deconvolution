"""
TobiasWolfMathBlindDeconvolutionMHDM - Python Wrapper

Original Implementation:
    TobiasWolf-math (GitHub)
    GitHub Repository: https://github.com/TobiasWolf-math/Blind-Deconvolution-MHDM
    Language/Framework: MATLAB

Reference Paper (if applicable):
    Based on method described in the repository without published paper

Algorithm Description:
    - Iterative alternation between latent image and blur kernel (PSF) updates
    - Uses regularization/prior terms to stabilize deconvolution
    - Outputs a restored image (and kernel estimate when available)

Author: AUTHOR_PROJECT
Wrapper Version: 1.0.0
Original Author: TobiasWolf-math"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import tempfile
import shutil

import sys
import os
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    path = start.resolve()

    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent

    return path

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm
from blinddeconv.system.octave import OctaveEngine

ALGORITHM_NAME = "TobiasWolf_math_Blind_Deconvolution_MHDM"
SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")

KernelSpec = Tuple[int, int]


class TobiasWolfMathBlindDeconvolutionMHDM(DeconvolutionAlgorithm):
	def __init__(
		self,
		lambda0: float = 14e-5,
		mu0: float = 63e4,
		r: float = 1.0,
		s: float = 1e-1,
		tol: float = 1e-10,
		maxits: int = 30,
		stopping: float = 0.0,
		tau: float = 1.001,
		noise_level: Optional[float] = None,
	):
		super().__init__(ALGORITHM_NAME)

		self._eng = OctaveEngine.get_instance()
		if self._eng is None:
			raise RuntimeError("Could not initialize Octave Engine. Check octaveconfig.py and Octave installation.")

		self._eng.addpath(self._eng.genpath(SOURCE_PATH))
		self._eng.cd(SOURCE_PATH)

		try:
			self._eng.eval("pkg load image")
		except Exception as e:
			print(f"Warning: Failed to load Octave 'image' package. Ensure it is installed (pkg install -forge image). Error: {e}")

		# Compatibility shim: sum(..., 'all') is MATLAB R2018b+ syntax,
		# not supported in older Octave versions.
		self._compat_dir = tempfile.mkdtemp(prefix="octave_compat_")
		sum_shim_path = os.path.join(self._compat_dir, "sum.m")
		with open(sum_shim_path, "w") as f:
			f.write(
				"function s = sum(x, varargin)\n"
				"  if nargin >= 2 && ischar(varargin{1}) && strcmp(varargin{1}, 'all')\n"
				"    if nargin == 3\n"
				"      s = builtin('sum', x(:), varargin{2});\n"
				"    else\n"
				"      s = builtin('sum', x(:));\n"
				"    end\n"
				"  else\n"
				"    s = builtin('sum', x, varargin{:});\n"
				"  end\n"
				"end\n"
			)
		self._eng.addpath(self._compat_dir)

		self.lambda0 = float(lambda0)
		self.mu0 = float(mu0)
		self.r = float(r)
		self.s = float(s)
		self.tol = float(tol)
		self.maxits = int(maxits)
		self.stopping = float(stopping)
		self.tau = float(tau)
		self.noise_level = None if noise_level is None else float(noise_level)

	def _compute_stopping(self, image_gray: np.ndarray) -> float:
		if self.stopping > 0:
			return self.stopping

		if self.noise_level is not None and self.noise_level > 0:
			N = image_gray.size
			return self.tau * self.noise_level * np.sqrt(N)

		return 0.0

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "lambda0" in param and param["lambda0"] is not None:
			self.lambda0 = float(param["lambda0"])
		if "mu0" in param and param["mu0"] is not None:
			self.mu0 = float(param["mu0"])
		if "r" in param and param["r"] is not None:
			self.r = float(param["r"])
		if "s" in param and param["s"] is not None:
			self.s = float(param["s"])
		if "tol" in param and param["tol"] is not None:
			self.tol = float(param["tol"])
		if "maxits" in param and param["maxits"] is not None:
			self.maxits = int(param["maxits"])
		if "stopping" in param and param["stopping"] is not None:
			self.stopping = float(param["stopping"])
		if "tau" in param and param["tau"] is not None:
			self.tau = float(param["tau"])
		if "noise_level" in param:
			val = param["noise_level"]
			self.noise_level = None if val is None else float(val)

	def get_param(self) -> list[str, Any]:
		return [
			("lambda0", self.lambda0),
			("mu0", self.mu0),
			("r", self.r),
			("s", self.s),
			("tol", self.tol),
			("maxits", self.maxits),
			("stopping", self.stopping),
			("tau", self.tau),
			("noise_level", self.noise_level),
		]

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if image.ndim == 3 and image.shape[2] == 3:
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			image_gray = image

		image_gray = image_gray.astype(np.float64)
		if image_gray.max() > 1.5:
			image_gray /= 255.0

		self._eng.push('f_py', image_gray)
		self._eng.push('lambda0_py', float(self.lambda0))
		self._eng.push('mu0_py', float(self.mu0))
		self._eng.push('r_py', float(self.r))
		self._eng.push('s_py', float(self.s))
		self._eng.push('tol_py', float(self.tol))
		self._eng.push('maxits_py', float(self.maxits))

		stopping_val = self._compute_stopping(image_gray)
		self._eng.push('stopping_py', float(stopping_val))

		try:
			self._eng.eval(
				"f = f_py;"
				"[m_py,n_py] = size(f);"
				"f_four = fft2(f);"
				"[rr, c] = ismember(f_four, conj(f_four));"
				"c = reshape(c, m_py*n_py, 1);"
				"zero_mask = (c == 0);"
				"c(zero_mask) = find(zero_mask);"
				"M = [c, c(c)];"
				"sortedM = sort(M, 2);"
				"[~, uniqueIdx] = unique(sortedM, 'rows', 'stable');"
				"indices = M(uniqueIdx, :);"
				"[u_end_py, k_end_py, ~, ~, ~] = blind_deconvolution_MHDM( "
				"f, f_four, "
				"lambda0_py, mu0_py, "
				"r_py, s_py, "
				"tol_py, stopping_py, maxits_py, indices);"
			)

			u_end_mat = self._eng.pull('u_end_py')
			k_end_mat = self._eng.pull('k_end_py')
		except Exception as e:
			print(f"Error executing Octave function: {e}")
			if image.ndim == 2:
				return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), np.zeros((3, 3))
			return image, np.zeros((3, 3))

		u_np = np.array(u_end_mat, dtype=np.float64)
		u_np = np.clip(u_np, 0.0, 1.0)
		u_uint8 = (u_np * 255.0).astype(np.uint8)
		u_bgr = cv2.cvtColor(u_uint8, cv2.COLOR_GRAY2BGR)

		kernel = np.array(k_end_mat, dtype=np.float64)

		return u_bgr, kernel

	def __del__(self):
		if hasattr(self, '_compat_dir') and os.path.isdir(self._compat_dir):
			shutil.rmtree(self._compat_dir, ignore_errors=True)


__all__ = ["TobiasWolfMathBlindDeconvolutionMHDM"]
