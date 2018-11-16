# coding: utf-8


SPL_DEFAULT_FOLDER = '__pycache__/spl'


# ... defining SPL backends
SPL_BACKEND_PYTHON = {'name':     'python'}
SPL_BACKEND_PYCCEL = {'name':     'pyccel',
                      'compiler': 'gfortran',
                      'flags':    '-O3'}

SPL_BACKEND = SPL_BACKEND_PYTHON
# ...
