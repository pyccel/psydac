# -*- coding: UTF-8 -*-
#! /usr/bin/python

from pathlib    import Path
from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

# ...
# Read library version into '__version__' variable
path = Path(__file__).parent / 'psydac' / 'version.py'
exec(path.read_text())
# ...

NAME    = 'psydac'
VERSION = __version__
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
install_requires = [

    # Third-party packages from PyPi
    'numpy>=1.16',
    'scipy>=0.18',
    'sympy>=1.2',
    'matplotlib',
    'pytest',
    'pyyaml',
    'yamlloader',

    # Our packages from PyPi
    'sympde',
    'pyccel',
    'gelato',

    # In addition, we depend on mpi4py and h5py (MPI version).
    # Since h5py must be built from source, we run the commands
    #
    # python3 -m pip install requirements.txt
    # python3 -m pip install .
    'mpi4py',
    'h5py',
]

dependency_links = []
# ...


# ... bspline extension (TODO: remove Fortran files from library)
bsp_ext = Extension(name    = 'psydac.core.bsp',
                        sources = ['psydac/core/external/bspline.F90',
                                   'psydac/core/external/pppack.F90',
                                   'psydac/core/bsp_ext.F90',
                                   'psydac/core/bsp.F90',
                                   'psydac/core/bsp.pyf',
                                   ],
                        f2py_options = ['--quiet'],)


ext_modules  = [bsp_ext]
# ...

# ...
entry_points = {'console_scripts': ['psydac-mesh = psydac.cmd.mesh:main']}
# ...

# ...
def setup_package():
    setup(packages=packages,
          ext_modules=ext_modules,
          install_requires=install_requires,
          include_package_data=True,
          zip_safe=True,
          dependency_links=dependency_links,
          entry_points=entry_points,
          **setup_args)
# ....
# ..................................................................................
if __name__ == "__main__":
    setup_package()
