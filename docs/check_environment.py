import sys
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

MIN_PYTHON = (3, 9)

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE = ROOT / "requirements.txt"


def parse_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None, None
    for sep in ("==", ">=", "<=", ">", "<"):
        if sep in line:
            name, ver = line.split(sep, 1)
            return name.strip(), sep + ver.strip()

    return line, None


def check_python():
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info < MIN_PYTHON:
        print(f"Minimum required: {'.'.join(map(str, MIN_PYTHON))}")
        return False
    return True


def check_requirements():
    if not REQUIREMENTS_FILE.exists():
        print("requirements.txt not found")
        return False
    print("Checking required packages")
    compatible = True
    with open(REQUIREMENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            name, constraint = parse_line(line)
            if not name:
                continue
            try:
                v = version(name)
                print(f"{name}: installed ({v})")
            except PackageNotFoundError:
                print(f"{name}: NOT installed")
                if constraint:
                    print(f"  recommended: pip install {name}{constraint}")
                else:
                    print(f"  recommended: pip install {name}")
                compatible = False

    return compatible


def main():
    print("Environment check")
    print("OS:", platform.system())

    compatible = check_python()
    compatible = check_requirements() and compatible

    if not compatible:
        print("Environment is NOT compatible")
        sys.exit(1)

    print("Environment is compatible")


if __name__ == "__main__":
    main()

