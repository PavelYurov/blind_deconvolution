from __future__ import annotations

from typing import Tuple

import numpy as np


from ..pmp_denoise.pmp import PMP_BD as _LegacyPMP_BD

__all__ = ['PMP_BD']


_HEAVY_SIGMA_THRESHOLD = 0.015
_IMPULSE_DENSITY_HEAVY = 0.01
_POISSON_A_THRESHOLD = 1e-3


def _classify_branch(image: np.ndarray,
                     heavy_sigma_threshold: float,
                     impulse_density_heavy: float,
                     poisson_a_threshold: float) -> Tuple[str, dict]:


    from ..pmp_denoise_fix.noise_orchestrator import (
        analyze_noise,
        _is_truly_correlated,
    )

    info = analyze_noise(image)
    sigma = float(info.get('sigma', 0.0))
    impulse_density = float(info.get('impulse_density', 0.0))
    poisson_a = float(info.get('poisson_a', 0.0))
    correlated = bool(info.get('correlated', False))


    if correlated and not _is_truly_correlated(info):
        correlated = False
        info['correlated'] = False

    heavy = (sigma >= heavy_sigma_threshold or
             impulse_density >= impulse_density_heavy or
             poisson_a >= poisson_a_threshold or
             correlated)

    return ('robust' if heavy else 'legacy', info)


class PMP_BD(_LegacyPMP_BD):


    _IN_DISPATCH_ATTR = '_pmp_improved_in_dispatch'

    def __init__(self, *args,

                 heavy_sigma_threshold: float = _HEAVY_SIGMA_THRESHOLD,
                 impulse_density_heavy: float = _IMPULSE_DENSITY_HEAVY,
                 poisson_a_threshold: float = _POISSON_A_THRESHOLD,
                 force: str | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.heavy_sigma_threshold = float(heavy_sigma_threshold)
        self.impulse_density_heavy = float(impulse_density_heavy)
        self.poisson_a_threshold = float(poisson_a_threshold)
        self.force = force

        self.last_branch: str | None = None
        self.last_descriptor: dict | None = None


    def process(self, image):


        if getattr(self, self._IN_DISPATCH_ATTR, False):
            return super().process(image)


        if self.auto_mode != 'robust' and self.force is None:
            return super().process(image)

        return self._dispatch(image)


    def _dispatch(self, image):
        if self.force in ('robust', 'legacy'):
            branch = self.force
            descriptor = {'forced': True}
        else:
            branch, descriptor = _classify_branch(
                image,
                heavy_sigma_threshold=self.heavy_sigma_threshold,
                impulse_density_heavy=self.impulse_density_heavy,
                poisson_a_threshold=self.poisson_a_threshold,
            )

        self.last_branch = branch
        self.last_descriptor = descriptor

        if branch == 'robust':
            return self._run_robust(image)
        return self._run_legacy(image)


    def _run_robust(self, image):
        from ..pmp_denoise_fix.pmp_robust import PMP_BD_Robust

        alg = PMP_BD_Robust(kernel_size=self.kernel_size)


        cb = getattr(self, '_callback', None)
        if cb is not None and hasattr(alg, 'set_callback'):
            try:
                alg.set_callback(cb)
            except Exception:
                pass

        result = alg.process(image)


        timer = getattr(alg, 'timer', None)
        if timer is not None:
            self.timer = timer
        return result

    def _run_legacy(self, image):


        setattr(self, self._IN_DISPATCH_ATTR, True)
        try:
            return super().process(image)
        finally:
            setattr(self, self._IN_DISPATCH_ATTR, False)
