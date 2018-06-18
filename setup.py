# -*- coding: UTF-8 -*-
#! /usr/bin/python

import sys
import os
from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.core import Extension
import spl

NAME    = 'spl'
VERSION = spl.__version__
AUTHOR  = 'Ahmed Ratnani, Jalal Lakhlili, Yaman Güçlü'
EMAIL   = 'ratnaniahmed@gmail.com'
URL     = 'http://www.ahmed.ratnani.org'
DESCR   = 'Python package for BSplines/NURBS'
KEYWORDS = ['FEM', 'IGA', 'BSplines']
LICENSE = "LICENSE.txt"

setup_args = dict(
    name             = NAME,
    version          = VERSION,
    description      = DESCR,
    long_description = open('README.rst').read(),
    author           = AUTHOR,
    author_email     = EMAIL,
    license          = LICENSE,
    keywords         = KEYWORDS,
    url              = URL,
#    download_url     = URL+'/get/default.tar.gz',
)

# ...
packages = find_packages()
#packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
# ...

# ...
install_requires = ['numpy', 'scipy']
dependency_links = []
# ...


# ... bspline extension
bsp_ext = Extension(name    = 'spl.core.bsp',
                        sources = ['spl/core/external/bspline.F90',
                                   'spl/core/external/pppack.F90',
                                   'spl/core/bsp_ext.F90',
                                   'spl/core/bsp.F90',
                                   'spl/core/bsp.pyf',
                                   ],
                        f2py_options = ['--quiet'],)


ext_modules  = [bsp_ext]
# ...

# ...
def setup_package():
    setup(packages=packages,
          ext_modules=ext_modules,
          install_requires=install_requires,
          include_package_data=True,
          zip_safe=True,
          dependency_links=dependency_links,
          **setup_args)
# ....
# ..................................................................................
if __name__ == "__main__":
    setup_package()
