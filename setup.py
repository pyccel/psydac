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
AUTHOR  = 'Ahmed Ratnani'
EMAIL   = 'ratnaniahmed@gmail.com'
URL     = 'http://www.ahmed.ratnani.org/'
DESCR   = 'Python package for BSplines/NURBS.'
KEYWORDS = ['FEM', 'IGA', 'bsplines']
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
packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
# ...

# ...
install_requires = ['numpy', 'scipy']
dependency_links = []
# ...

# ...
#INSTALL_DIR = '/home/travis/build/ratnania/spl/usr'
if 'PREFIX' in os.environ:
    INSTALL_DIR = os.environ['PREFIX']
else:
    INSTALL_DIR = os.path.join(os.getcwd(), 'usr')
# ...

# ...
library_dirs = [os.path.join(INSTALL_DIR,"lib")]
libraries = ["spl"]
libraries = libraries[::-1]  # must reverse the order for linking

include_dirs = []
for lib in libraries:
    include_dirs.append(os.path.join(os.path.join(INSTALL_DIR, "include"), lib))
# ...

# ... django extension
django_ext = Extension('spl.core.django',
                       sources=['spl/core/django.pyf',
                                'spl/core/django.F90'],
                       f2py_options=['--quiet'],
#                       define_macros=[
#                                     #('F2PY_REPORT_ATEXIT', 0),
#                                     ('F2PY_REPORT_ON_ARRAY_COPY', 0)],
                       include_dirs=include_dirs,
                       library_dirs=library_dirs,
                       libraries=libraries)

ext_modules  = [django_ext]
# ...

def setup_package():
    setup(packages=packages, \
          ext_modules=ext_modules, \
          install_requires=install_requires, \
          include_package_data=True, \
          zip_safe=True, \
          dependency_links=dependency_links, \
          **setup_args)

if __name__ == "__main__":
    setup_package()
