import os
from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

os.environ["CC"] = "gcc"

# extensions = [
#     Extension("primes", ["primes.pyx"],
#         include_dirs=[...],
#         libraries=[...],
#         library_dirs=[...]),
#     # Everything but primes.pyx is included here.
#     Extension("*", ["*.pyx"],
#         include_dirs=[...],
#         libraries=[...],
#         library_dirs=[...]),
# ]

extensions = [
    Extension("compute_cy_memoryviews", ["compute_cy_memoryviews.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="compute_cy_memoryviews",
    ext_modules=cythonize(["compute_cy_memoryviews.pyx"], annotate=True, language_level="3"),
)
