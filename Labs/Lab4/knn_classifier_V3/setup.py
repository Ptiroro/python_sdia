import os
from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

os.environ["CC"] = "gcc"


extensions = [
    Extension("knn_classifier_V3", ["knn_classifier_V3.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="knn_classifier_V3",
    ext_modules=cythonize(["knn_classifier_V3.pyx"], annotate=True, language_level="3"),
)
