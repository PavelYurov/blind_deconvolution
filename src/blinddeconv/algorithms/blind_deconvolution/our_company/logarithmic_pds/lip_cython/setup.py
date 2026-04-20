# setup.py
import sys
import numpy
import glob
from setuptools import setup, Extension
from Cython.Build import cythonize

pyx_files = glob.glob("*.pyx")

if sys.platform.startswith('win'):
    extra_compile_args = ['/O2', '/fp:fast']
else:
    extra_compile_args = ['-O3', '-ffast-math']

ext_modules =[
    Extension(
        name=f.split('.')[0],
        sources=[f],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args
    ) for f in pyx_files
]

setup(
    name="LIP cythonized",
    ext_modules=cythonize(
        ext_modules, 
        language_level="3", 
        annotate=False,
        build_dir="build_c_files"
    )
)