from pathlib import Path
import subprocess

DOCSRC = Path(__file__).resolve().parents[1] / "source"
BUILD = Path(__file__).resolve().parents[1] / "_build" / "html"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def generate_rst():
    print("Generate API")
    for rst in DOCSRC.glob("*.rst"):
        if rst.name not in ("index.rst",):
            rst.unlink()

    cmd = [
        "sphinx-apidoc",
        "-o", str(DOCSRC),
        str(PROJECT_ROOT),
    ]

    subprocess.check_call(cmd)
    print("API generated")


def build_html():
    print("Building documentation")
    BUILD.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sphinx-build", 
        "-b", 
        "html", 
        str(DOCSRC), 
        str(BUILD),
    ]
    subprocess.check_call(cmd)
    print("Documentation build successfully")
    print(f"See documentation in {BUILD}")


def main():
    generate_rst()
    build_html()


if __name__ == "__main__":
    main()


