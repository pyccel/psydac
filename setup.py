# -*- coding: UTF-8 -*-
#! /usr/bin/python

from pathlib    import Path
from importlib  import util
from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

# ...
# Load module 'psydac.version' without running file 'psydac.__init__'
path = Path(__file__).parent / 'psydac' / 'version.py'
spec = util.spec_from_file_location('version', str(path))
mod  = util.module_from_spec(spec)
spec.loader.exec_module(mod)
# ...

NAME    = 'psydac'
VERSION = mod.__version__
AUTHOR  = 'Ahmed Ratnani, Jalal Lakhlili, Yaman Güçlü, Said Hadjout'
EMAIL   = 'ratnaniahmed@gmail.com'
URL     = 'http://www.ahmed.ratnani.org'
DESCR   = 'Python package for BSplines/NURBS'
KEYWORDS = ['FEM', 'IGA', 'BSplines']
LICENSE = "LICENSE.txt"

setup_args = dict(
    name             = NAME,
    version          = VERSION,
    description      = DESCR,
    long_description = open('README.md').read(),
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
    'sympy>=1.5',
    'matplotlib',
    'pytest>=4.5',
    'pyyaml>=5.1',
    'packaging',
    'pyevtk',

    # Our packages from PyPi
    'sympde>=0.13',
    'pyccel>=0.10.1',
    'gelato==0.11',

    # Alternative backend to Pyccel is Numba
    'numba',

    # In addition, we depend on mpi4py and h5py (parallel version)
    'mpi4py',

    # When pyccel is run in parallel with MPI, it uses tblib to pickle
    # tracebacks, which allows mpi4py to broadcast exceptions
    'tblib',

    # Since h5py must be built from source using the MPI compiler and linked
    # to a parallel HDF5 library, the following environment variables must be
    # defined upon calling pip install:
    #
    # CC="mpicc"
    # HDF5_MPI="ON"
    # HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
    #
    # Since we cannot pass the flag `--no-binary h5py' to setuptools, we have
    # to download the binaries from GitHub. Using a separate requirements.txt
    # may cause dependency conflicts which pip is not able to resolve.
    'h5py @ https://github.com/h5py/h5py/archive/refs/heads/master.zip',

    # IGAKIT - not on PyPI
    'igakit @ https://github.com/dalcinl/igakit/archive/refs/heads/master.zip',
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
          python_requires='>=3.7',
          install_requires=install_requires,
          include_package_data=True,
          package_data = {'':['*.txt']},
          zip_safe=True,
          dependency_links=dependency_links,
          entry_points=entry_points,
          **setup_args)
# ....
# ..................................................................................
if __name__ == "__main__":
    setup_package()
