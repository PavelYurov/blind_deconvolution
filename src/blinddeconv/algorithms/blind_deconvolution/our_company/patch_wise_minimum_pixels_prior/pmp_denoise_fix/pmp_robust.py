"""
pmp_robust.py
=============

PMP blind deconvolution with the mathematically grounded robust noise
pipeline.

Wraps :class:`PMP_BD` from :mod:`pmp` so that the **entire** legacy noise
preprocessing surface (impulse / ScreeNOT / ACT / spatial / PSD / blind-loop /
pre-nonblind / histogram_eq) is disabled, and replaced by a single call to
:func:`noise_orchestrator.robust_denoise` *before* blind deconvolution.

What changes vs ``PMP_BD``
--------------------------
* ``robust_denoise`` is invoked once at the start of ``process(image)``;
  its output (denoised float64 [0,1] grayscale) is fed to the blind
  deconvolution stage.
* All noise-pipeline ``*_preprocess`` / ``blind_denoise`` /
  ``pre_nonblind`` / ``histogram_eq`` / ``auto_params`` /
  ``estimate_noise_internal`` knobs are **forced to their off values**
  on construction; passing them is silently ignored.  The robust pipeline
  is the sole owner of pre-blind denoising.
* Kernel-estimation parameters (``kernel_size``, ``lambda_pmp``,
  ``lambda_grad``, ``xk_iter``, ``patch_r``, ``k_thresh``, ``gamma_correct``,
  ``denoise_eps``, ``denoise_radius``, ``ensemble_denoise``, ...) and the
  non-blind solver (``final_deconv``, …) are passed through unchanged.

After ``.process(image)``, the field ``self._last_robust_info`` holds the
descriptor returned by the orchestrator, useful for logging/analysis.

Usage
-----
    >>> from pmp_denoise_fix.pmp_robust import PMP_BD_Robust
    >>> alg = PMP_BD_Robust(kernel_size=25)        # PMP defaults + robust noise
    >>> restored, kernel = alg.process(noisy_image)
    >>> alg._last_robust_info['log']     # human-readable trace
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from .pmp import PMP_BD


__all__ = ['PMP_BD_Robust']


class PMP_BD_Robust(PMP_BD):
    """:class:`PMP_BD` with all denoising delegated to the robust orchestrator.

    See module docstring.
    """

    # All flags whose presence would cause the legacy preprocess pipeline
    # to fire.  We force these regardless of caller input to guarantee the
    # robust orchestrator is the sole denoising authority.
    _LEGACY_DENOISE_FLAGS_OFF = {
        'impulse_preprocess':   'none',
        'screenot_preprocess':  'none',
        'act_preprocess':       'none',
        'preprocess':           'none',
        'noise_preprocess':     'none',
        'blind_denoise':        'none',
        'pre_nonblind':         'none',
        'histogram_eq':         'none',
        # Disable internal noise estimator + orchestrator
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
        # Hard-override any legacy denoising flags supplied by caller
        for k, v in self._LEGACY_DENOISE_FLAGS_OFF.items():
            kwargs[k] = v
        for k in self._LEGACY_PARAM_DICT_FLAGS:
            kwargs[k] = None

        # Some of these legacy flags may not exist as keyword args on the
        # current PMP_BD signature (the field set has evolved over time);
        # filter out unknown keys to avoid TypeError.
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            import inspect
            sig = inspect.signature(PMP_BD.__init__)
            allowed = set(sig.parameters)
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            super().__init__(*args, **kwargs)

        self._last_robust_info: dict | None = None

    # ────────────────────────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run robust denoising, then PMP blind + non-blind deconvolution.

        Parameters
        ----------
        image : ndarray
            Noisy image, gray or RGB, any dtype/scale supported by
            ``robust_denoise`` (auto-normalised to float64 [0,1] gray).

        Returns
        -------
        restored : ndarray (uint16, [0, 255])
            Restored image as produced by ``PMP_BD.process``.
        kernel   : ndarray (k, k)
            Estimated PSF.
        """
        # Lazy import to avoid pulling BM3D / curvelets at module load.
        from .noise_orchestrator import robust_denoise

        cleaned, info = robust_denoise(image, verbose=False)
        self._last_robust_info = info

        # ``cleaned`` is float64 [0,1] grayscale.  PMP_BD.process accepts
        # any 2D/3D ndarray (handles its own normalisation/grayscale step
        # idempotently — passing already-normalised gray is a no-op).
        return super().process(cleaned)
