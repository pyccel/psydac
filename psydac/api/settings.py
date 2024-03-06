# coding: utf-8

# ... Determine Pyccel version: compiler names changed with version 1.3.0
import pyccel
pyccel_version = tuple(map(int, pyccel.__version__.split('.')))
pyccel_legacy  = pyccel_version < (1, 3, 0)
# ...


PSYDAC_DEFAULT_FOLDER = {'name':'__psydac__'}

# ... defining PSYDAC backends
PSYDAC_BACKEND_PYTHON = {'name': 'python', 'tag':'python', 'openmp':False}

PSYDAC_BACKEND_GPYCCEL = {'name':     'pyccel',
                      'compiler': 'gfortran' if pyccel_legacy else 'GNU',
                      'flags':    '-O3 -march=native -mtune=native  -mavx -ffast-math -ffree-line-length-none',
                      'folder': '__gpyccel__',
                      'tag':'gpyccel',
                      'openmp':False}

PSYDAC_BACKEND_IPYCCEL = {'name':     'pyccel',
                      'compiler': 'ifort' if pyccel_legacy else 'intel',
                      'flags':    '-O3',
                      'folder': '__ipyccel__',
                      'tag':'ipyccel',
                      'openmp':False}

PSYDAC_BACKEND_PGPYCCEL = {'name':     'pyccel',
                      'compiler': 'pgfortran' if pyccel_legacy else 'PGI',
                      'flags':    '-O3',
                      'folder': '__pgpyccel__',
                       'tag':'pgpyccel',
                       'openmp':False}
# ...

# List of all available backends for accelerating Python code
PSYDAC_BACKENDS = {
    'python'      : PSYDAC_BACKEND_PYTHON,
    'pyccel-gcc'  : PSYDAC_BACKEND_GPYCCEL,
    'pyccel-intel': PSYDAC_BACKEND_IPYCCEL,
    'pyccel-pgi'  : PSYDAC_BACKEND_PGPYCCEL,
}
