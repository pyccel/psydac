#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
The purpose of this module is to pyccelize all PSYDAC kernels, in the case
that these were modified after an editable installation of PSYDAC.
"""
from psydac.cmd.argparse_helpers import add_help_flag, add_version_flag

__all__ = (
    'setup_psydac_compile_parser',
    'psydac_compile',
    'PSYDAC_COMPILE_DESCR',
)

PSYDAC_COMPILE_DESCR = "Accelerate all computational kernels in PSYDAC using Pyccel (editable install only)."

#==============================================================================
def setup_psydac_compile_parser(parser):
    """
    Add the `psydac compile` arguments to the parser.

    Add the `psydac compile` arguments to the parser for command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    parser.add_argument('--language',
        type    = str,
        choices = ('fortran', 'c'),
        default = 'fortran',
        help    = 'Language used to pyccelize all the kernel files (default: fortran).'
    )
    add_help_flag(parser)
    add_version_flag(parser)

#==============================================================================
def psydac_compile(*, language):
    """
    Accelerate all computational kernels in PSYDAC using Pyccel (editable install only).

    Accelerate all computational kernels in PSYDAC using Pyccel (editable install only).

    Parameters
    ----------
    language : str
        Language used to pyccelize the kernels files.
    """
    from pathlib import Path
    import shutil
    import subprocess
    import psydac

    # Absolute path to the psydac directory
    psydac_path = Path(psydac.__file__).parent

    # Glob pattern of all kernel files
    glob = '**/*_kernels.py'

    # Command to be executed
    cmd = (
        shutil.which('pyccel'),
        'make',
        '--language', language,
        '--glob', (psydac_path / glob).as_posix(),
        '--openmp',
    )

    print('Executing command:')
    print(f' {" ".join(cmd)}\n')
    subprocess.run(cmd, shell=False)
