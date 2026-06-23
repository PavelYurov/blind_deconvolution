import os
import sys
import shutil
import tempfile
import sysconfig
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PYD_DIR = PKG_DIR / "_build_pyd"
C_DIR   = PKG_DIR / "_build_c"

INSTALL_DIR = PKG_DIR.parent / "vdbke_cython_pyd"

PYX_MODULES = sorted(
    p.stem for p in PKG_DIR.glob("*.pyx")
)

print(f"[build] Package dir : {PKG_DIR}")
print(f"[build] .pyd output : {PYD_DIR}")
print(f"[build] .c   output : {C_DIR}")
print(f"[build] Modules     : {PYX_MODULES}")

def cythonize_all():

    from Cython.Compiler.Main import compile as cy_compile, CompilationOptions
    from Cython.Compiler import Options

    directives = {
        'language_level': 3,
        'boundscheck': False,
        'wraparound': True,
        'cdivision': True,
        'initializedcheck': False,
        'nonecheck': False,
        'infer_types': True,
    }
    opts = CompilationOptions(
        language_level=3,
        compiler_directives=directives,
    )
    c_files = {}
    for mod in PYX_MODULES:
        pyx = str(PKG_DIR / f"{mod}.pyx")
        print(f"  cython  {mod}.pyx → {mod}.c")
        result = cy_compile(pyx, opts)
        if result.num_errors:
            raise RuntimeError(f"Cython errors in {mod}.pyx")
        c_files[mod] = str(PKG_DIR / f"{mod}.c")
    return c_files

def build_extensions(c_files):

    import numpy as np
    from setuptools import Extension, Distribution
    from setuptools.command.build_ext import build_ext

    tmp_root = Path(tempfile.gettempdir()) / "cython_dcp"
    tmp_root.mkdir(exist_ok=True)

    if sys.platform.startswith('win'):

        extra_compile_args = ['/O2', '/fp:precise']
        extra_link_args    = []
    else:

        extra_compile_args = ['-O3', '-fno-fast-math', '-fno-finite-math-only']
        extra_link_args    = []

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    built = {}

    for mod, c_path in c_files.items():
        print(f"  cc      {mod}.c → {mod}{ext_suffix}")

        ext = Extension(
            name=mod,
            sources=[c_path],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[

                ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
            ],
        )

        dist = Distribution({"ext_modules": [ext]})
        dist.parse_config_files()

        cmd = build_ext(dist)
        cmd.build_lib = str(tmp_root / "lib")
        cmd.build_temp = str(tmp_root / "tmp")
        cmd.inplace = False
        cmd.ensure_finalized()
        cmd.run()

        for root, _, files in os.walk(str(tmp_root / "lib")):
            for f in files:
                if f.startswith(mod) and f.endswith(ext_suffix):
                    built[mod] = os.path.join(root, f)
                    break

    return built, tmp_root

def install_and_cleanup(built, c_files, tmp_root):

    PYD_DIR.mkdir(exist_ok=True)
    C_DIR.mkdir(exist_ok=True)

    for mod, pyd_path in built.items():
        dest = PYD_DIR / Path(pyd_path).name
        shutil.copy2(pyd_path, dest)
        print(f"  install {dest.relative_to(PKG_DIR)}")
        if INSTALL_DIR.exists():

            for old in INSTALL_DIR.glob(f"{mod}.*.pyd"):
                try:
                    old.unlink()
                except OSError:
                    pass
            dest_install = INSTALL_DIR / Path(pyd_path).name
            shutil.copy2(pyd_path, dest_install)
            print(f"  install ../{dest_install.relative_to(INSTALL_DIR.parent)}")

    for mod, c_path in c_files.items():
        c = Path(c_path)
        if c.exists():
            dest_c = C_DIR / c.name
            shutil.move(str(c), str(dest_c))
            print(f"  archive {dest_c.relative_to(PKG_DIR)}")

    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    for pyd in PKG_DIR.glob("*.pyd"):
        pyd.unlink()

    print("[build] Done — .pyd in _build_pyd/, .c in _build_c/, "
          "source dir clean.")

def main():
    print("\n══ Stage 1: Cython (.pyx → .c) ══")
    c_files = cythonize_all()

    print("\n══ Stage 2: Compile (.c → .pyd) ══")
    built, tmp_root = build_extensions(c_files)

    if len(built) != len(PYX_MODULES):
        missing = set(PYX_MODULES) - set(built.keys())
        print(f"[build] WARNING: failed modules: {missing}")

    print("\n══ Stage 3: Install & Cleanup ══")
    install_and_cleanup(built, c_files, tmp_root)

    print(f"\n[build] Successfully built {len(built)}/{len(PYX_MODULES)} modules")
    for mod in sorted(built.keys()):
        print(f"  ✓ {mod}")

if __name__ == "__main__":
    main()
