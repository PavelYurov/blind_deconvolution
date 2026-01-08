import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCSRC = ROOT / "docsrc"
DOCS = ROOT / "docs"

def main():
    print("Building documentation")

    if not DOCSRC.exists():
        print("ERROR: docsrc directory not found")
        sys.exit(1)

    conf = DOCSRC / "conf.py"
    index = DOCSRC / "index.rst"

    if not conf.exists():
        print("ERROR: conf.py not found in docsrc")
        sys.exit(1)

    if not index.exists():
        print("ERROR: index.rst not found in docsrc")
        sys.exit(1)

    DOCS.mkdir(exist_ok=True)

    print("Source:", DOCSRC)
    print("Output:", DOCS)

    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        str(DOCSRC),
        str(DOCS)
    ]

    print("Running:", " ".join(cmd))

    subprocess.check_call(cmd)

    print("API documentation generated")
    print("Open docs/index.html")

if __name__ == "__main__":
    main()
