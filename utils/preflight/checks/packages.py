from importlib.metadata import version, PackageNotFoundError
from ..report import CheckResult
from packaging.requirements import Requirement


def check_requirement(req_str: str) -> CheckResult:
    req = Requirement(req_str)
    name = req.name

    if req.marker and not req.marker.evaluate():
        return CheckResult(
            name=f"package.{name}",
            ok=True,
            summary="Dependency skipped (environment marker does not apply)"
        )

    try:
        installed = version(name)
    except PackageNotFoundError:
        return CheckResult(
            name=f"package.{name}",
            ok=False,
            summary="Package not installed",
            details="no version constraint",
            recommendation=f"pip install '{req_str}'"
        )

    if req.specifier and not req.specifier.contains(installed, prereleases=True):
        return CheckResult(
            name=f"package.{name}",
            ok=False,
            summary="Installed version does not satisfy constraints",
            details=f"Installed {installed}, required {req.specifier}",
            recommendation=f"pip install '{req_str}'"
        )

    return CheckResult(
        name=f"package.{name}",
        ok=True,
        summary=f"{installed} satisfies {req.specifier or 'any version'}"
    )