from __future__ import annotations

from typing import Tuple
import numpy as np

from .pmp import PMP_BD


__all__ = ['PMP_BD_Robust']


class PMP_BD_Robust(PMP_BD):


    _LEGACY_DENOISE_FLAGS_OFF = {
        'impulse_preprocess':   'none',
        'screenot_preprocess':  'none',
        'act_preprocess':       'none',
        'preprocess':           'none',
        'noise_preprocess':     'none',
        'blind_denoise':        'none',
        'pre_nonblind':         'none',
        'histogram_eq':         'none',

        'auto_params':              None,
        'estimate_noise':           'none',
        'estimate_noise_internal':  False,
    }

    _LEGACY_PARAM_DICT_FLAGS = (
        'impulse_params', 'screenot_params', 'act_params',
        'preprocess_params', 'noise_preprocess_params',
        'blind_denoise_params', 'pre_nonblind_params',
        'histogram_eq_params',
    )

    def __init__(self, *args, **kwargs):

        for k, v in self._LEGACY_DENOISE_FLAGS_OFF.items():
            kwargs[k] = v
        for k in self._LEGACY_PARAM_DICT_FLAGS:
            kwargs[k] = None


        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            import inspect
            sig = inspect.signature(PMP_BD.__init__)
            allowed = set(sig.parameters)
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            super().__init__(*args, **kwargs)

        self._last_robust_info: dict | None = None


    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:


        from .noise_orchestrator import robust_denoise

        cleaned, info = robust_denoise(image, verbose=False)
        self._last_robust_info = info


        return super().process(cleaned)
