"""
pmp_denoise_improved.pmp
========================

Drop-in replacement for ``pmp_denoise.PMP_BD``.

The class :class:`PMP_BD` exposed here keeps the *exact* public name and
constructor signature of the legacy :class:`pmp_denoise.pmp.PMP_BD`.
All behaviour is delegated to the legacy implementation **except** when
the caller asks for ``auto_mode='robust'`` — in that case the merged
per-image dispatcher (used previously in ``pmp_denoise_merge``) decides
whether to run the robust pipeline (``PMP_BD_Robust``) or fall back to
the legacy native ``auto_mode='robust'`` orchestrator.

Decision matrix
---------------
* no kwargs / defaults                    → identical to legacy PMP_BD
* legacy denoise kwargs (impulse_*, ACT,  → identical to legacy PMP_BD
  ScreeNOT, blind_denoise, ...)
* ``auto_mode='robust'``                  → merge dispatcher
* anything else                           → identical to legacy PMP_BD
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Parent class — the **untouched** legacy implementation.
from ..pmp_denoise.pmp import PMP_BD as _LegacyPMP_BD

__all__ = ['PMP_BD']


# ─────────────────────────────────────────────────────────────────────────────
# Routing thresholds (mirror those used in pmp_denoise_merge / pmp_denoise_fix)
# ─────────────────────────────────────────────────────────────────────────────
_HEAVY_SIGMA_THRESHOLD = 0.015      # σ above which we definitely want robust
_IMPULSE_DENSITY_HEAVY = 0.01       # impulse density above which → robust
_POISSON_A_THRESHOLD = 1e-3         # Poisson 'a' coefficient → robust


def _classify_branch(image: np.ndarray,
                     heavy_sigma_threshold: float,
                     impulse_density_heavy: float,
                     poisson_a_threshold: float) -> Tuple[str, dict]:
    """Return ``('robust', info)`` or ``('legacy', info)``.

    Lazy imports keep the legacy default path free of optional deps.
    """
    from ..pmp_denoise_fix.noise_orchestrator import (
        analyze_noise,
        _is_truly_correlated,
    )

    info = analyze_noise(image)
    sigma = float(info.get('sigma', 0.0))
    impulse_density = float(info.get('impulse_density', 0.0))
    poisson_a = float(info.get('poisson_a', 0.0))
    correlated = bool(info.get('correlated', False))

    # Strict gate for *true* correlation (avoid ACT misclassification).
    if correlated and not _is_truly_correlated(info):
        correlated = False
        info['correlated'] = False

    heavy = (sigma >= heavy_sigma_threshold or
             impulse_density >= impulse_density_heavy or
             poisson_a >= poisson_a_threshold or
             correlated)

    return ('robust' if heavy else 'legacy', info)


class PMP_BD(_LegacyPMP_BD):
    """Improved PMP-BD with merge dispatcher.

    Inherits the entire constructor and API from
    :class:`pmp_denoise.pmp.PMP_BD`.  Adds (keyword-only) tuning knobs
    for the merge dispatcher; all carry sensible defaults so existing
    code calling ``PMP_BD(...)`` keeps working unchanged.
    """

    # Re-entrancy guard so the dispatcher's "legacy branch" can invoke
    # the legacy ``auto_mode='robust'`` orchestrator without recursing
    # back through our ``process`` override.
    _IN_DISPATCH_ATTR = '_pmp_improved_in_dispatch'

    def __init__(self, *args,
                 # Extra (keyword-only) merge-dispatcher knobs.
                 heavy_sigma_threshold: float = _HEAVY_SIGMA_THRESHOLD,
                 impulse_density_heavy: float = _IMPULSE_DENSITY_HEAVY,
                 poisson_a_threshold: float = _POISSON_A_THRESHOLD,
                 force: str | None = None,  # None | 'robust' | 'legacy'
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.heavy_sigma_threshold = float(heavy_sigma_threshold)
        self.impulse_density_heavy = float(impulse_density_heavy)
        self.poisson_a_threshold = float(poisson_a_threshold)
        self.force = force
        # Diagnostics populated on every dispatched run.
        self.last_branch: str | None = None
        self.last_descriptor: dict | None = None

    # ── Public API override ────────────────────────────────────────────────
    def process(self, image):
        # Re-entrant call (legacy branch already inside dispatcher) →
        # delegate straight to legacy implementation.
        if getattr(self, self._IN_DISPATCH_ATTR, False):
            return super().process(image)

        # Default / non-robust paths: behave exactly like the parent class.
        if self.auto_mode != 'robust' and self.force is None:
            return super().process(image)

        return self._dispatch(image)

    # ── Merge dispatcher ───────────────────────────────────────────────────
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

    # ── Branch implementations ─────────────────────────────────────────────
    def _run_robust(self, image):
        from ..pmp_denoise_fix.pmp_robust import PMP_BD_Robust

        alg = PMP_BD_Robust(kernel_size=self.kernel_size)

        # Forward callback so the framework UI keeps working.
        cb = getattr(self, '_callback', None)
        if cb is not None and hasattr(alg, 'set_callback'):
            try:
                alg.set_callback(cb)
            except Exception:
                pass

        result = alg.process(image)

        # Bubble up timing info if available.
        timer = getattr(alg, 'timer', None)
        if timer is not None:
            self.timer = timer
        return result

    def _run_legacy(self, image):
        # Run the parent's process() with the re-entrancy guard set so
        # the legacy native ``auto_mode='robust'`` orchestrator (which
        # lives inside the parent class) executes normally without
        # bouncing back through this subclass.
        setattr(self, self._IN_DISPATCH_ATTR, True)
        try:
            return super().process(image)
        finally:
            setattr(self, self._IN_DISPATCH_ATTR, False)
