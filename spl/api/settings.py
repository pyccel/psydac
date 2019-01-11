# coding: utf-8


SPL_DEFAULT_FOLDER = '__pycache__/__spl__'


# ... defining SPL backends
SPL_BACKEND_PYTHON = {'name':     'python'}
SPL_BACKEND_PYCCEL = {'name':     'pyccel',
                      'compiler': 'gfortran',
                      'flags':    '-O3',
                      'accelerator': None,
                      'folder': '__pyccel__'}
                      
SPL_BACKEND_NUMBA = {'name': 'numba'}

SPL_BACKEND = SPL_BACKEND_PYTHON
# ...
