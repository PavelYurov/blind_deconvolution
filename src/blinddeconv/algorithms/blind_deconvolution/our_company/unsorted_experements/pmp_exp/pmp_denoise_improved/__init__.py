"""pmp_denoise_improved — drop-in replacement for ``pmp_denoise.PMP_BD``.

Same class name and constructor signature as the legacy
:class:`pmp_denoise.pmp.PMP_BD`.  Behaviour:

* defaults / legacy denoise kwargs   → identical to legacy ``PMP_BD``
* ``auto_mode='robust'``             → merged per-image dispatcher
  (chooses between ``PMP_BD_Robust`` and the legacy native
  ``auto_mode='robust'`` orchestrator)
"""

from .pmp import PMP_BD

__all__ = ['PMP_BD']
