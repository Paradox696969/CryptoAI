from setuptools import setup
from Cython.Build import cythonize
import os
import time
import sys

f = open("code.py", "r")
code = f.read()
f.close()

f = open("main.pyx", "w")
f.write(code)
f.close()


setup(
    name='Main',
    ext_modules=cythonize("main.pyx"),
    zip_safe=False,
)

sys.exit()
