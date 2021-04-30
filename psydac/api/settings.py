# coding: utf-8


PSYDAC_DEFAULT_FOLDER = '__psydac__'

# ... defining PSYDAC backends
PSYDAC_BACKEND_PYTHON = {'name': 'python', 'tag':'python'}

PSYDAC_BACKEND_GPYCCEL = {'name':     'pyccel',
                      'compiler': 'gfortran',
                      'flags':    '-O3 -march=native -mtune=native  -mavx -ffast-math',
                      'accelerator': None,
                      'folder': '__gpyccel__',
                      'tag':'gpyccel'}

PSYDAC_BACKEND_IPYCCEL = {'name':     'pyccel',
                      'compiler': 'ifort',
                      'flags':    '-O3',
                      'accelerator': None,
                      'folder': '__ipyccel__',
                      'tag':'ipyccel'}

PSYDAC_BACKEND_PGPYCCEL = {'name':     'pyccel',
                      'compiler': 'pgfortran',
                      'flags':    '-O3',
                      'accelerator': None,
                      'folder': '__pgpyccel__',
                       'tag':'pgpyccel'}
                      
PSYDAC_BACKEND_NUMBA = {'name': 'numba','tag':'numba', 'fastmath':True}

#PSYDAC_BACKEND_PYTHRAN = {'name':'pythran','tag':'pythran'}
# ...

# List of all available backends for accelerating Python code
PSYDAC_BACKENDS = {
    'python'      : PSYDAC_BACKEND_PYTHON,
    'pyccel-gcc'  : PSYDAC_BACKEND_GPYCCEL,
    'pyccel-intel': PSYDAC_BACKEND_IPYCCEL,
    'pyccel-pgi'  : PSYDAC_BACKEND_PGPYCCEL,
    'numba'       : PSYDAC_BACKEND_NUMBA,
#   'pythran'     : PSYDAC_BACKEND_PYTHRAN,
}
