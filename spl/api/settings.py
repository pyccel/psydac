# coding: utf-8


SPL_DEFAULT_FOLDER = '__pycache__/__spl__'


# ... defining SPL backends
SPL_BACKEND_PYTHON = {'name': 'python', 'tag':'python'}

SPL_BACKEND_GPYCCEL = {'name':     'pyccel',
                      'compiler': 'gfortran',
                      'flags':    '-O3',
                      'accelerator': None,
                      'folder': '__gpyccel__',
                      'tag':'gpyccel'}

SPL_BACKEND_IPYCCEL = {'name':     'pyccel',
                      'compiler': 'ifort',
                      'flags':    '-O3',
                      'accelerator': None,
                      'folder': '__ipyccel__',
                      'tag':'ipyccel'}

SPL_BACKEND_PGPYCCEL = {'name':     'pyccel',
                      'compiler': 'pgfortran',
                      'flags':    '-O3',
                      'accelerator': None,
                      'folder': '__pgpyccel__',
                       'tag':'pgpyccel'}
                      
SPL_BACKEND_NUMBA = {'name': 'numba','tag':'numba'}

SPL_BACKEND_PYTHRAN = {'name':'pythran','tag':'pythran'}

SPL_BACKEND = SPL_BACKEND_PYTHON
# ...
