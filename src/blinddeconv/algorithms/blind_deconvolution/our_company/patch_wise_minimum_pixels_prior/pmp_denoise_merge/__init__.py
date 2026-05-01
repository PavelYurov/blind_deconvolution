"""pmp_denoise_merge — dispatcher combining pmp_denoise (legacy) and
pmp_denoise_fix (robust orchestrator).

Public class: ``PMP_BD_Merged``.
"""

from .pmp_merged import PMP_BD_Merged  # noqa: F401

__all__ = ['PMP_BD_Merged']
