# from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    summary: str
    details: str | None = None
    recommendation: str | None = None