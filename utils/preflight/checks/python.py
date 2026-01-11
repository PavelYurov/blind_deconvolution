import sys
from packaging.specifiers import SpecifierSet
from ..report import CheckResult


def check_python_version(spec: str) -> CheckResult:
    current = ".".join(map(str, sys.version_info[:3]))

    if current not in SpecifierSet(spec):
        return CheckResult(
            name="python.version",
            ok=False,
            summary="Incompatible Python version",
            details=f"Detected Python {current}, required {spec}",
            recommendation="Install a compatible Python interpreter"
        )

    return CheckResult(
        name="python.version",
        ok=True,
        summary=f"Python {current} is compatible"
    )