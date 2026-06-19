"""
pmp_merged.py
=============

Dispatcher pipeline that combines the two existing PMP variants:

  * ``pmp_denoise.PMP_BD``        (legacy, with ``auto_mode='robust'`` knobs)
  * ``pmp_denoise_fix.PMP_BD_Robust``  (new robust orchestrator wrapper)

Empirical observation (see ``pmp_denoise_fix/test_robust_pmp.py``):

============================  ==========================
Noise type                    Better algorithm
============================  ==========================
Heavy white Gaussian          PMP_BD_Robust
Poisson / Poisson-Gaussian    PMP_BD_Robust
Heavy impulse                 PMP_BD_Robust
Colored Gaussian              PMP_BD (auto_mode='robust')
Periodic                      PMP_BD (auto_mode='robust')
Clean / mild AWGN             PMP_BD (auto_mode='robust')
============================  ==========================

There is no fundamental conflict — the dispatcher classifies the noise
once via the orchestrator's analyser and delegates to the better variant.

Implements the full :class:`DeconvolutionAlgorithm` contract so it plugs
straight into ``Processing.process(alg)``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


from src.blinddeconv.algorithms.base import DeconvolutionAlgorithm

from ..pmp_denoise_fix.noise_orchestrator import (
    analyze_noise,
    _is_truly_correlated,
)


__all__ = ['PMP_BD_Merged']


_LEGACY_DEFAULT_KWARGS: dict = dict(
    denoise_eps=0.006, denoise_radius=2,
    lambda_tv=0.005, lambda_l0=0.002,
    grad_smooth_sigma=0.285,
    pmp_quantile=0.0,
    ensemble_denoise=True,
    impulse_preprocess='auto',
    noise_estimation='pca',
    act_preprocess='auto',
    blind_denoise='guided',
    auto_mode='robust',
)


_HEAVY_SIGMA_DEFAULT = 0.015

_IMPULSE_DENSITY_HEAVY = 0.01

_POISSON_A_NORM = 1e-3


class PMP_BD_Merged(DeconvolutionAlgorithm):


    name = 'PMP_BD_Merged'

    def __init__(
        self,
        kernel_size: int = 51,
        *,
        force: Optional[str] = None,
        heavy_sigma_threshold: float = _HEAVY_SIGMA_DEFAULT,
        impulse_density_heavy: float = _IMPULSE_DENSITY_HEAVY,
        poisson_a_threshold: float = _POISSON_A_NORM,
        robust_kwargs: Optional[dict] = None,
        legacy_kwargs: Optional[dict] = None,
        verbose: bool = True,
        name: str = 'PMP_BD_Merged',
    ):
        if force not in (None, 'robust', 'legacy'):
            raise ValueError(f"force must be None, 'robust' or 'legacy'; "
                             f"got {force!r}")

        super().__init__(name=name)

        self.kernel_size = int(kernel_size)
        self.force = force
        self.heavy_sigma_threshold = float(heavy_sigma_threshold)
        self.impulse_density_heavy = float(impulse_density_heavy)
        self.poisson_a_threshold = float(poisson_a_threshold)
        self.robust_kwargs = dict(robust_kwargs or {})
        self.legacy_kwargs = dict(legacy_kwargs or {})
        self.verbose = bool(verbose)


        self.last_branch: Optional[str] = None
        self.last_descriptor: Optional[dict] = None
        self.last_inner_alg: Any = None


    def change_param(self, param: Dict[str, Any]) -> None:


        own = {'kernel_size', 'force', 'heavy_sigma_threshold',
               'impulse_density_heavy', 'poisson_a_threshold',
               'verbose', 'name'}
        for key, val in param.items():
            if key in own:
                setattr(self, key, val)
            elif key.startswith('robust__'):
                self.robust_kwargs[key[len('robust__'):]] = val
            elif key.startswith('legacy__'):
                self.legacy_kwargs[key[len('legacy__'):]] = val
            else:
                self.robust_kwargs[key] = val
                self.legacy_kwargs[key] = val

    def get_param(self) -> List[Tuple[str, Any]]:

        out: List[Tuple[str, Any]] = [
            ('kernel_size',           self.kernel_size),
            ('force',                 self.force),
            ('heavy_sigma_threshold', self.heavy_sigma_threshold),
            ('impulse_density_heavy', self.impulse_density_heavy),
            ('poisson_a_threshold',   self.poisson_a_threshold),
            ('last_branch',           self.last_branch),
        ]
        for k, v in self.robust_kwargs.items():
            out.append((f'robust__{k}', v))
        for k, v in self.legacy_kwargs.items():
            out.append((f'legacy__{k}', v))
        if self.last_descriptor is not None:
            for k, v in self.last_descriptor.items():
                out.append((f'descriptor__{k}', v))
        return out


    def _classify(self, image: np.ndarray) -> Tuple[str, dict]:

        if self.force is not None:
            return self.force, {'forced': True,
                                'reason': f'force={self.force}'}

        info = analyze_noise(image)
        impulse = info['impulse']
        psd = info['psd']
        pca_norm = info['pca_norm']

        a_norm = float(pca_norm.get('a', 0.0))
        pca_type = str(pca_norm.get('noise_type', 'unknown'))
        density = float(impulse.get('density', 0.0))
        psd_2d = psd.get('psd_2d', None)
        psd_sigma = (float(np.sqrt(max(0.0, psd_2d.mean())))
                     if psd_2d is not None else 0.0)
        truly_correlated = _is_truly_correlated(psd)

        descriptor = {
            'impulse_density':  density,
            'pca_a_norm':       a_norm,
            'pca_noise_type':   pca_type,
            'psd_sigma_norm':   psd_sigma,
            'truly_correlated': truly_correlated,
            'lag1_h':           float(psd.get('lag1_h', 0.0)),
            'lag1_v':           float(psd.get('lag1_v', 0.0)),
        }

        if density >= self.impulse_density_heavy:
            descriptor['reason'] = (
                f"heavy impulse density {density:.4f} >= "
                f"{self.impulse_density_heavy}")
            return 'robust', descriptor

        if (pca_type in ('poisson', 'poisson_gaussian')
                and a_norm > self.poisson_a_threshold):
            descriptor['reason'] = (
                f"Poisson signature a_norm={a_norm:.4g} ({pca_type})")
            return 'robust', descriptor

        if truly_correlated:
            descriptor['reason'] = "truly correlated (lag1>=0.5, radial-CV>=0.3)"
            return 'legacy', descriptor

        if psd_sigma > self.heavy_sigma_threshold:
            descriptor['reason'] = (
                f"heavy white sigma={psd_sigma:.4g} > "
                f"{self.heavy_sigma_threshold}")
            return 'robust', descriptor

        descriptor['reason'] = (
            f"mild noise sigma={psd_sigma:.4g} <= "
            f"{self.heavy_sigma_threshold}, fall through to legacy")
        return 'legacy', descriptor


    def _make_robust(self):
        from ..pmp_denoise_fix.pmp_robust import PMP_BD_Robust
        return PMP_BD_Robust(kernel_size=self.kernel_size,
                             **self.robust_kwargs)

    def _make_legacy(self):
        from ..pmp_denoise.pmp import PMP_BD
        kwargs = dict(_LEGACY_DEFAULT_KWARGS)
        kwargs.update(self.legacy_kwargs)
        return PMP_BD(kernel_size=self.kernel_size, **kwargs)


    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:


        t0 = time.time()
        branch, descriptor = self._classify(image)
        self.last_branch = branch
        self.last_descriptor = descriptor

        if self.verbose:
            print(f"[PMP-Merged] BRANCH = {branch.upper()}")
            for k, v in descriptor.items():
                print(f"   {k:20s} = {v}")

        if branch == 'robust':
            alg = self._make_robust()
        else:
            alg = self._make_legacy()
        self.last_inner_alg = alg


        if self._callback is not None and hasattr(alg, 'set_callback'):
            try:
                alg.set_callback(self._callback)
            except Exception:
                pass

        try:
            restored, kernel = alg.process(image)
        finally:
            self.timer = time.time() - t0
        return restored, kernel
