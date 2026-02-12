#!/usr/bin/env python3
"""
Интерактивный деинсталлятор зависимостей.
Работает в паре с install.py.
Удаляет только те пакеты, которые не используются другими активными профилями.
"""
import argparse
import json
import subprocess
import sys
import os
import shutil
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    path = start.resolve()

    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent

    return path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = _find_project_root(CURRENT_FILE)
STATE_FILE = PROJECT_ROOT / ".dependency_state.json"

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(data: dict):
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _confirm(prompt: str, default_yes: bool = False) -> bool:
    suffix = " [Y/n] " if default_yes else " [y/N] "
    while True:
        ans = input(prompt + suffix).strip().lower()
        if not ans: return default_yes
        if ans in ('y', 'yes'): return True
        if ans in ('n', 'no'): return False

def _get_venv_python(venv_dir: str) -> str:
    path = Path(venv_dir)
    if os.name == "nt":
        exe = path / "Scripts" / "python.exe"
    else:
        exe = path / "bin" / "python"
    if not exe.exists():
        return sys.executable
    return str(exe)

def main():
    parser = argparse.ArgumentParser(description="Умное удаление зависимостей профиля.")
    parser.add_argument("--clean-all", action="store_true", help="УДАЛИТЬ виртуальное окружение и файл истории")
    parser.add_argument("profile", nargs="?", help="Имя профиля")
    parser.add_argument("--venv", default=".venv", help="Путь к виртуальному окружению")
    parser.add_argument("-y", "--yes", action="store_true", help="Удалять без подтверждения")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет удалено")
    args = parser.parse_args()

    if args.clean_all:
        venv_path = Path(args.venv).resolve()
        
        print("ВНИМАНИЕ! Вы собираетесь удалить:")
        if venv_path.exists():
            print(f"  - Папку окружения: {venv_path}")
        if STATE_FILE.exists():
            print(f"  - Файл истории:    {STATE_FILE}")

        if not (args.yes or _confirm("\nУдалить всё и сбросить проект?", default_yes=False)):
            sys.exit(0)

        if not args.dry_run:
            if venv_path.exists():
                try:
                    shutil.rmtree(venv_path)
                    print(f"Папка '{venv_path.name}' удалена.")
                except OSError as e:
                    print(f"Ошибка удаления папки: {e}")
            else:
                print("Папка venv не найдена (уже удалена).")

            if STATE_FILE.exists():
                STATE_FILE.unlink()
                print("Файл истории удален.")
        else:
            print("[DRY-RUN] Папки и файлы были бы удалены.")
            
        return

    if not args.profile:
        parser.error("Необходимо указать profile (например 'base') или флаг --clean-all")

    state = load_state()
    if args.profile not in state:
        print(f"Ошибка: Профиль '{args.profile}' не найден в истории установок.")
        print(f"Известные профили: {', '.join(state.keys())}")
        sys.exit(1)

    target_pkgs = set(state[args.profile])
    
    keep_pkgs = set()
    other_profiles = []
    for prof, pkgs in state.items():
        if prof != args.profile:
            keep_pkgs.update(pkgs)
            other_profiles.append(prof)
    
    to_remove = sorted(list(target_pkgs - keep_pkgs))
    shared = sorted(list(target_pkgs & keep_pkgs))

    print(f"Анализ профиля '{args.profile}'...")
    
    if not to_remove and not shared:
        print("Список пакетов пуст. Удаляем запись о профиле.")
        if not args.dry_run:
            del state[args.profile]
            save_state(state)
        return

    if to_remove:
        print(f"\nБудут УДАЛЕНЫ следующие пакеты (эксклюзивные для '{args.profile}'):")
        for pkg in to_remove:
            print(f"  - {pkg}")
    else:
        print(f"\nНет эксклюзивных пакетов для удаления.")

    if shared:
        print(f"\nБудут ОСТАВЛЕНЫ (используются в {', '.join(other_profiles)}):")
        print(f"  {', '.join(shared)}")

    if not to_remove:
        print("\nНикакие файлы не будут удалены с диска, только запись о профиле.")
    
    if not (args.yes or _confirm("\nВыполнить действие?")):
        print("Отмена.")
        sys.exit(0)

    python_exe = _get_venv_python(args.venv)
    
    if python_exe == sys.executable and not args.dry_run:
         print("[WARNING] Виртуальное окружение не найдено. Попытка удаления из текущего python.")

    if to_remove:
        cmd = [python_exe, "-m", "pip", "uninstall", "-y"] + to_remove
        
        if args.dry_run:
            print(f"[DRY-RUN] {' '.join(cmd)}")
        else:
            print(f"Запуск pip uninstall в {python_exe}...")
            ret = subprocess.call(cmd)
            if ret != 0:
                print("Ошибка при удалении пакетов.", file=sys.stderr)
                if not _confirm("Pip вернул ошибку. Всё равно удалить профиль из истории?"):
                    sys.exit(ret)

    if not args.dry_run:
        del state[args.profile]
        save_state(state)
        print(f"Профиль '{args.profile}' удален из {STATE_FILE.name}")

if __name__ == "__main__":
    main()