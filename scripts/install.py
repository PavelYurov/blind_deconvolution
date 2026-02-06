#!/usr/bin/env python3
"""
Интерактивный установщик зависимостей.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "scripts" else CURRENT_FILE.parent
STATE_FILE = PROJECT_ROOT / ".dependency_state.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _confirm(prompt: str, *, default_no: bool = True) -> bool:
    suffix = " [y/N] " if default_no else " [Y/n] "
    while True:
        try:
            answer = input(prompt + suffix).strip().lower()
        except KeyboardInterrupt:
            print("\nОтмена.")
            sys.exit(130)
        if not answer:
            return not default_no
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Введите 'y' или 'n'.")

def _in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)

def _get_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"

def _run_cmd(cmd: list[str], dry_run: bool = False, msg: str = None) -> int:
    if msg:
        print(f"[INFO] {msg}", flush=True)
    if dry_run:
        print(f"[DRY-RUN] {' '.join(str(x) for x in cmd)}")
        return 0
    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        return 130

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state_profile(profile: str, packages: list[str]):
    data = load_state()
    current = set(data.get(profile, []))
    
    clean_names = set()
    for pkg in packages:
        name = pkg.split('==')[0].split('>=')[0].split('<')[0].split('~=')[0].strip()
        clean_names.add(name)
        
    current.update(clean_names)
    data[profile] = sorted(list(current))
    
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def ensure_bootstrap_deps(python_exe: str, dry_run: bool):
    deps = ["packaging"]
    if sys.version_info < (3, 11):
        deps.append("tomli")

    cmd = [python_exe, "-m", "pip", "install"] + deps
    
    check_cmd = [python_exe, "-c", "import packaging; import importlib.util; exit(0 if importlib.util.find_spec('packaging') else 1)"]
    if subprocess.run(check_cmd, capture_output=True).returncode == 0:
        return

    print(f"Установка системных зависимостей ({', '.join(deps)})...", flush=True)
    _run_cmd(cmd, dry_run=dry_run)

def prepare_environment(venv_path_str: str, dry_run: bool) -> str:
    venv_path = Path(venv_path_str).resolve()
    
    if _in_venv():
        return sys.executable
    target_python = _get_venv_python(venv_path)
    
    if target_python.exists():
        return str(target_python)

    print(f"Виртуальное окружение не обнаружено.", flush=True)
    if not _confirm(f"Создать venv в '{venv_path}' и использовать его?", default_no=True):
        print("Внимание: Используется системный Python.", flush=True)
        return sys.executable

    print(f"Создание venv в {venv_path}...", flush=True)
    if not dry_run:
        ret = _run_cmd([sys.executable, "-m", "venv", str(venv_path)])
        if ret != 0:
            print("Ошибка создания venv.", file=sys.stderr)
            sys.exit(ret)
    
    return str(target_python)

def get_install_plan(profile: str):
    from utils.preflight.checks.packages import check_requirement
    from utils.preflight.checks.python import check_python_version
    from utils.preflight.config import load_pyproject, resolve_profile_dependencies
    from packaging.requirements import Requirement

    cfg = load_pyproject(PROJECT_ROOT / "pyproject.toml")
    
    if "requires-python" in cfg.get("project", {}):
        res = check_python_version(cfg["project"]["requires-python"])
        if not res.ok:
            print(f"[WARNING] {res.summary}. {res.details}")

    raw_deps = resolve_profile_dependencies(cfg, profile)
    to_install = []
    
    for req_str in raw_deps:
        try:
            req = Requirement(req_str)
            if req.marker and not req.marker.evaluate():
                continue
            res = check_requirement(req_str)
            if not res.ok:
                to_install.append(req_str)
        except Exception as e:
            print(f"Ошибка проверки '{req_str}': {e}")
            
    return to_install

def cmd_install(args):
    try:
        missing = get_install_plan(args.profile)
    except RuntimeError as e:
        print(f"Ошибка конфигурации: {e}", file=sys.stderr)
        return 1

    if not missing:
        print(f"Профиль '{args.profile}': Все зависимости установлены.")
        if not args.dry_run:
            # Для сохранения берем все зависимости профиля
            from utils.preflight.config import load_pyproject, resolve_profile_dependencies
            cfg = load_pyproject(PROJECT_ROOT / "pyproject.toml")
            all_deps = resolve_profile_dependencies(cfg, args.profile)
            save_state_profile(args.profile, all_deps)
        return 0

    print(f"Будет установлено {len(missing)} пакетов для профиля '{args.profile}':")
    for pkg in missing:
        print(f"  - {pkg}")

    if not (args.yes or _confirm("Продолжить?")):
        return 0

    cmd = [sys.executable, "-m", "pip", "install"]
    if args.upgrade: cmd.append("--upgrade")
    if args.no_cache_dir: cmd.append("--no-cache-dir")
    cmd.extend(missing)

    ret = _run_cmd(cmd, dry_run=args.dry_run, msg="Установка пакетов...")
    
    if ret == 0 and not args.dry_run:
        # Получаем полный список для сохранения в историю
        from utils.preflight.config import load_pyproject, resolve_profile_dependencies
        cfg = load_pyproject(PROJECT_ROOT / "pyproject.toml")
        all_deps = resolve_profile_dependencies(cfg, args.profile)
        save_state_profile(args.profile, all_deps)
        print("Готово.")
    
    return ret

def cmd_check(args):
    from utils.preflight.checks.packages import check_requirement
    from utils.preflight.config import load_pyproject, resolve_profile_dependencies

    print(f"Проверка профиля: {args.profile}...", flush=True)
    cfg = load_pyproject(PROJECT_ROOT / "pyproject.toml")
    deps = resolve_profile_dependencies(cfg, args.profile)
    
    ok_count = 0
    for req in deps:
        res = check_requirement(req)
        status = "OK" if res.ok else "MISSING"
        if res.ok: ok_count += 1
        print(f"[{status}] {res.name}")
    
    if ok_count < len(deps):
        print(f"\nСтатус: Не хватает {len(deps) - ok_count} пакетов.")
        return 1
    print("\nСтатус: Все пакеты установлены.")
    return 0

def cmd_list(args):
    from utils.preflight.config import load_pyproject
    cfg = load_pyproject(PROJECT_ROOT / "pyproject.toml")
    profiles = cfg.get("tool", {}).get("preflight", {}).get("profiles", {})
    
    print(f"{'PROFILE':<15} DEPENDENCIES")
    print("-" * 40)
    for name, data in profiles.items():
        print(f"{name:<15} {data.get('dependencies', '')}")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", default=".venv", help="Путь к виртуальному окружению")
    sub = parser.add_subparsers(dest="command", required=True)
    
    p_check = sub.add_parser("check")
    p_check.add_argument("profile", nargs="?", default="base")
    p_check.set_defaults(func=cmd_check)
    
    p_inst = sub.add_parser("install")
    p_inst.add_argument("profile", nargs="?", default="base")
    p_inst.add_argument("-y", "--yes", action="store_true")
    p_inst.add_argument("--dry-run", action="store_true")
    p_inst.add_argument("--upgrade", action="store_true")
    p_inst.add_argument("--no-cache-dir", action="store_true")
    p_inst.set_defaults(func=cmd_install)

    p_list = sub.add_parser("list-profiles")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    
    target_python = prepare_environment(args.venv, getattr(args, "dry_run", False))
    

    if target_python != sys.executable and not getattr(args, "dry_run", False):
        ensure_bootstrap_deps(target_python, dry_run=False)
        
        print(f"--> Передача управления в venv: {target_python}", flush=True)
        child_cmd = [target_python, str(CURRENT_FILE)] + sys.argv[1:]
        
        try:
            ret = subprocess.call(child_cmd)
            sys.exit(ret)
        except Exception as e:
            print(f"Ошибка запуска в venv: {e}")
            sys.exit(1)

    ensure_bootstrap_deps(sys.executable, getattr(args, "dry_run", False))

    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Критическая ошибка: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()