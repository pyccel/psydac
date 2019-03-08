# coding: utf-8


PSYDAC_DEFAULT_FOLDER = '__pycache__/__spl__'


# ... defining PSYDAC backends
PSYDAC_BACKEND_PYTHON = {'name': 'python', 'tag':'python'}

PSYDAC_BACKEND_GPYCCEL = {'name':     'pyccel',
                      'compiler': 'gfortran',
                      'flags':    '-O3',
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
                      
PSYDAC_BACKEND_NUMBA = {'name': 'numba','tag':'numba'}

PSYDAC_BACKEND_PYTHRAN = {'name':'pythran','tag':'pythran'}

PSYDAC_BACKEND = PSYDAC_BACKEND_PYTHON
# ...
