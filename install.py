#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass


def _confirm(prompt: str, *, default_no: bool = True) -> bool:
    suffix = " [y/N] " if default_no else " [Y/n] "
    while True:
        answer = input(prompt + suffix).strip().lower()
        if not answer:
            return not default_no
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def _in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _venv_python(venv_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _ensure_venv(venv_dir: str, *, dry_run: bool) -> str:
    py = _venv_python(venv_dir)
    if os.path.exists(py):
        return py

    cmd = [sys.executable, "-m", "venv", venv_dir]
    print("+", " ".join(cmd))
    if not dry_run:
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SystemExit(int(proc.returncode))

    return py


def _pip_with(python_exe: str, *args: str, dry_run: bool) -> int:
    cmd = [python_exe, "-m", "pip", *args]
    print("+", " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def _has_packaging() -> bool:
    try:
        import packaging  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _bootstrap_runtime_dependencies(*, venv_dir: str, dry_run: bool) -> None:
    if _has_packaging():
        return

    print("Missing required runtime dependency: packaging")

    if _in_venv():
        raise SystemExit(_pip_with(sys.executable, "install", "packaging", dry_run=dry_run))

    if not _confirm(f"Create/use venv at '{venv_dir}' and install 'packaging' there?", default_no=True):
        raise SystemExit(2)

    venv_py = _ensure_venv(venv_dir, dry_run=dry_run)
    code = _pip_with(venv_py, "install", "packaging", dry_run=dry_run)
    if code != 0 or dry_run:
        raise SystemExit(code)

    os.execv(venv_py, [venv_py, *sys.argv])


def _ensure_pip_environment(command: str, *, venv_dir: str, dry_run: bool) -> None:
    if command not in {"install", "uninstall"}:
        return
    if _in_venv():
        return

    if not _confirm(f"Use a virtual environment at '{venv_dir}' for pip operations?", default_no=True):
        return

    venv_py = _ensure_venv(venv_dir, dry_run=dry_run)
    if dry_run:
        raise SystemExit(0)

    os.execv(venv_py, [venv_py, *sys.argv])


@dataclass(frozen=True)
class InstallPlan:
    profile: str
    required: list[str]
    applicable: list[str]
    to_install: list[str]
    to_uninstall: list[str]


def _requirement_applies(req_str: str) -> bool:
    from packaging.requirements import Requirement

    req = Requirement(req_str)
    return True if req.marker is None else bool(req.marker.evaluate())


def _pip(*args: str, dry_run: bool) -> int:
    return _pip_with(sys.executable, *args, dry_run=dry_run)


def _print_results(results) -> None:
    for r in results:
        print(f"{r.name}: {'OK' if r.ok else 'FAIL'}")
        if not r.ok:
            print(f"  {r.summary}")
            if r.details:
                print(f"  Details: {r.details}")
            if r.recommendation:
                print(f"  Recommendation: {r.recommendation}")


def build_plan(profile: str) -> InstallPlan:
    from packaging.requirements import Requirement
    from preflight.checks.packages import check_requirement
    from preflight.checks.python import check_python_version
    from preflight.config import load_pyproject, resolve_profile_dependencies

    cfg = load_pyproject()

    py_result = check_python_version(cfg["project"]["requires-python"])
    if not py_result.ok:
        raise RuntimeError(
            f"{py_result.summary}. {py_result.details or ''}".strip()
        )

    required = resolve_profile_dependencies(cfg, profile)
    applicable = [r for r in required if _requirement_applies(r)]

    to_install: list[str] = []
    for req_str in applicable:
        result = check_requirement(req_str)
        if not result.ok:
            to_install.append(req_str)

    to_uninstall: list[str] = []
    for req_str in applicable:
        req = Requirement(req_str)
        to_uninstall.append(req.name)

    return InstallPlan(
        profile=profile,
        required=required,
        applicable=applicable,
        to_install=to_install,
        to_uninstall=to_uninstall,
    )


def cmd_check(args: argparse.Namespace) -> int:
    from preflight.checks.packages import check_requirement
    from preflight.checks.python import check_python_version
    from preflight.config import load_pyproject, resolve_profile_dependencies

    cfg = load_pyproject()
    results = [check_python_version(cfg["project"]["requires-python"])]

    deps = resolve_profile_dependencies(cfg, args.profile)
    for req_str in deps:
        results.append(check_requirement(req_str))

    _print_results(results)
    return 0 if all(r.ok for r in results) else 1


def cmd_install(args: argparse.Namespace) -> int:
    try:
        plan = build_plan(args.profile)
    except RuntimeError as e:
        print(f"Pre-install checks failed: {e}", file=sys.stderr)
        return 2

    if not plan.to_install:
        print("All dependencies are already satisfied.")
        return 0

    print("Missing/unsatisfied dependencies:")
    for req_str in plan.to_install:
        print(f"  - {req_str}")

    if not (args.yes or _confirm("Install these packages?", default_no=True)):
        print("Installation canceled.")
        return 1

    pip_args = []
    if args.upgrade:
        pip_args.append("--upgrade")
    if args.no_cache_dir:
        pip_args.append("--no-cache-dir")

    return _pip("install", *pip_args, *plan.to_install, dry_run=args.dry_run)


def cmd_uninstall(args: argparse.Namespace) -> int:
    try:
        plan = build_plan(args.profile)
    except RuntimeError as e:
        print(f"Pre-uninstall checks failed: {e}", file=sys.stderr)
        return 2

    if not plan.to_uninstall:
        print("No dependencies found for this profile.")
        return 0

    print("Packages to uninstall (by name):")
    for name in plan.to_uninstall:
        print(f"  - {name}")

    if not (args.yes or _confirm("Uninstall these packages?", default_no=True)):
        print("Uninstall canceled.")
        return 1

    uninstall_args = ["-y"] if args.yes else []
    return _pip("uninstall", *uninstall_args, *plan.to_uninstall, dry_run=args.dry_run)


def cmd_list_profiles(_: argparse.Namespace) -> int:
    from preflight.config import load_pyproject

    cfg = load_pyproject()
    tool_cfg = cfg.get("tool", {}).get("preflight", {})
    profiles = tool_cfg.get("profiles", {})

    if not profiles:
        print("No [tool.preflight.profiles] section found.", file=sys.stderr)
        return 1

    for name in sorted(profiles.keys()):
        ref = profiles[name].get("dependencies", "")
        print(f"{name}\t{ref}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive installer for BlindDeconvolution dependencies."
    )
    parser.add_argument(
        "--venv",
        default="venv",
        help="Virtual environment directory to create/use for pip operations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Check Python and package dependencies.")
    p_check.add_argument("profile", nargs="?", default="base")
    p_check.set_defaults(func=cmd_check)

    p_install = sub.add_parser("install", help="Install missing dependencies.")
    p_install.add_argument("profile", nargs="?", default="base")
    p_install.add_argument("-y", "--yes", action="store_true", help="Do not prompt.")
    p_install.add_argument("--dry-run", action="store_true", help="Print pip commands only.")
    p_install.add_argument("--upgrade", action="store_true", help="Pass --upgrade to pip.")
    p_install.add_argument(
        "--no-cache-dir", action="store_true", help="Pass --no-cache-dir to pip."
    )
    p_install.set_defaults(func=cmd_install)

    p_uninstall = sub.add_parser("uninstall", help="Uninstall profile dependencies.")
    p_uninstall.add_argument("profile", nargs="?", default="base")
    p_uninstall.add_argument("-y", "--yes", action="store_true", help="Do not prompt.")
    p_uninstall.add_argument("--dry-run", action="store_true", help="Print pip commands only.")
    p_uninstall.set_defaults(func=cmd_uninstall)

    p_profiles = sub.add_parser("list-profiles", help="List available dependency profiles.")
    p_profiles.set_defaults(func=cmd_list_profiles)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dry_run = bool(getattr(args, "dry_run", False))
    _ensure_pip_environment(args.command, venv_dir=args.venv, dry_run=dry_run)
    _bootstrap_runtime_dependencies(venv_dir=args.venv, dry_run=dry_run)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
