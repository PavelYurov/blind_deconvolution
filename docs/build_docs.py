import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = DOCS / "_build" / "html"


def main():
    print("Building documentation")

    if not DOCS.exists():
        print("ERROR: docs directory not found")
        sys.exit(1)

    conf = DOCS / "conf.py"
    index = DOCS / "index.rst"

    if not conf.exists():
        print("ERROR: conf.py not found in docs")
        sys.exit(1)

    if not index.exists():
        print("ERROR: index.rst not found in docs")
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)

    print("Source:", DOCS)
    print("Output:", OUT)

    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        str(DOCS),
        str(OUT),
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    print("HTML documentation generated")
    print("Open docs/_build/html/index.html")


if __name__ == "__main__":
    main()

