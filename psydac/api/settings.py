# coding: utf-8
import platform


__all__ = ('PSYDAC_DEFAULT_FOLDER', 'PSYDAC_BACKENDS')

#==============================================================================

PSYDAC_DEFAULT_FOLDER = {'name':'__psydac__'}

# ... defining PSYDAC backends
PSYDAC_BACKEND_PYTHON = {'name': 'python', 'tag':'python', 'openmp':False}

PSYDAC_BACKEND_GPYCCEL  = {'name': 'pyccel',
                       'compiler': 'GNU',
                       'flags'   : '-O3 -march=native -mtune=native -ffast-math',
                       'folder'  : '__gpyccel__',
                       'tag'     : 'gpyccel',
                       'openmp'  : False}

PSYDAC_BACKEND_IPYCCEL  = {'name': 'pyccel',
                       'compiler': 'intel',
                       'flags'   : '-O3',
                       'folder'  : '__ipyccel__',
                       'tag'     :'ipyccel',
                       'openmp'  : False}

PSYDAC_BACKEND_PGPYCCEL = {'name': 'pyccel',
                       'compiler': 'PGI',
                       'flags'   : '-O3 -Munroll',
                       'folder'  : '__pgpyccel__',
                       'tag'     : 'pgpyccel',
                       'openmp'  : False}

PSYDAC_BACKEND_NVPYCCEL = {'name': 'pyccel',
                       'compiler': 'nvidia',
                       'flags'   : '-O3 -Munroll',
                       'folder'  : '__nvpyccel__',
                       'tag'     : 'nvpyccel',
                       'openmp'  : False}
# ...

# Platform-dependent flags
if platform.machine() == 'x86_64':
    PSYDAC_BACKEND_GPYCCEL['flags'] += ' -mavx'
elif platform.machine() == 'arm64' and platform.system() == "Darwin":
    # Check specific CPU type
    import subprocess
    cpu = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string']).decode('utf-8')
    if "Apple M1" in cpu:
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -mcpu=apple-m1 -ffast-math'
    else:
        raise SystemError(f"Unsupported CPU type '{cpu}'")


#==============================================================================

# List of all available backends for accelerating Python code
PSYDAC_BACKENDS = {
    'python'       : PSYDAC_BACKEND_PYTHON,
    'pyccel-gcc'   : PSYDAC_BACKEND_GPYCCEL,
    'pyccel-intel' : PSYDAC_BACKEND_IPYCCEL,
    'pyccel-pgi'   : PSYDAC_BACKEND_PGPYCCEL,
    'pyccel-nvidia': PSYDAC_BACKEND_NVPYCCEL,
}
