#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
The purpose of this module is to pyccelize all PSYDAC kernels, in the case
that these were modified after an editable installation of PSYDAC.
"""
from psydac.cmd.argparse_helpers import add_help_flag, add_version_flag, exit_with_error_message

__all__ = (
    'setup_psydac_test_parser',
    'psydac_test',
    'PSYDAC_TEST_DESCR',
)

PSYDAC_TEST_DESCR = "Run the PSYDAC test suite."

#==============================================================================
def setup_psydac_test_parser(parser):
    """
    Add the `psydac test` arguments to the parser.

    Add the `psydac test` arguments to the parser for command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """

    group = parser.add_argument_group('Test selection')
    group.add_argument('--mod',
        type    = str,
        help    = 'Only run tests from the specified module (e.g. psydac.linalg).'
    )
    group.add_argument('--mpi',
        action  = 'store_true',
        help    = 'Only run parallel tests with 4 MPI processes (default: run serial tests).',
    )
    group.add_argument('--petsc',
        action  = 'store_true',
        help    = 'Only run tests using PETSc and petsc4py (default: run the other tests).',
    )

    group = parser.add_argument_group('Pytest options')
    group.add_argument('-v', '--verbose',
        action   = 'store_true',
        help     = 'Increase verbosity of Pytest output.'
    )
    group.add_argument('-x', '--exitfirst',
        action = 'store_true',
        help   = 'Exit instantly on first error or failed test.'
    )

    group = parser.add_argument_group('Other options')
    add_help_flag(group)
    add_version_flag(group)

#==============================================================================
def psydac_test(*, mod, mpi, petsc, verbose, exitfirst):
    """
    Run the PSYDAC test suite.
    """
    if mod is None:
        mod = 'psydac'
    else:
        submods = mod.split('.')
        if len(submods) == 0:
            exit_with_error_message("module name cannot be empty")
        elif submods[0] != 'psydac':
            exit_with_error_message("module name must start with 'psydac'")
        try:
            modname = mod.split('::')[0]
            import importlib
            importlib.import_module(modname)
        except ImportError:
            exit_with_error_message(f"module '{modname}' not found")

    # Import modules here to speed up parser
    import os
    import shutil
    import subprocess
    import time

    # Clear Pytest cache from the current working directory
    cache_dir = '.pytest_cache'
    if os.path.isdir(cache_dir):
        print(f'Removing existing Pytest cache directory: {cache_dir}\n', flush=True)
        shutil.rmtree(cache_dir)

    # If no pytest.ini file exists in the current working directory, copy it
    # from the parent directory of this script (which is installed with PSYDAC)
    if not os.path.isfile('pytest.ini'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        pytest_ini = os.path.join(parent_dir, 'pytest.ini')
        if not os.path.isfile(pytest_ini):
            exit_with_error_message(f'could not find pytest.ini file in {parent_dir}')
        else:
            print(f'Copying pytest.ini from: {parent_dir}\n', flush=True)
            shutil.copy(pytest_ini, os.getcwd())

    # Build the list of flags for pytest
    flags = []

    # Set up MPI execution command, if needed
    if mpi:
        mpirun  = shutil.which('mpirun')
        mpi_exe = [mpirun, '-n', '4']
        flags.append('--with-mpi')

        # Determine if we are using OpenMPI, MPICH, or Intel MPI
        result = subprocess.run([mpirun, '--version'], capture_output=True, text=True)
        output = result.stdout.lower() + result.stderr.lower()
        if 'open mpi' in output:
            mpi_implementation = 'OpenMPI'
            oversubscribe_flag = '--oversubscribe'
        elif 'mpich' in output:
            mpi_implementation = 'MPICH'
            oversubscribe_flag = ''
        elif 'intel mpi' in output:
            mpi_implementation = 'Intel MPI'
            oversubscribe_flag = '--oversubscribe' # to be verified
        else:
            exit_with_error_message("cannot determine MPI implementation from output of 'mpirun --version'")

        print(f'MPI implementation detected: {mpi_implementation}')
        if oversubscribe_flag:
            mpi_exe.append(oversubscribe_flag)

    else:
        mpi_exe = []
        flags.extend(['-n', 'auto', '--dist', 'loadgroup'])  # for pytest-xdist

    # If PETSc tests are requested, check that petsc4py is installed
    if petsc:
        try:
            import petsc4py  # noqa: F401
        except ImportError:
            exit_with_error_message("petsc4py is not installed, cannot run PETSc tests")

    # Add appropriate markers for test selection
    flags.extend(['--pyargs', mod])
    mpi_mark = 'mpi' if mpi else 'not mpi'
    petsc_mark = 'petsc' if petsc else 'not petsc'
    flags.extend(['-m', f'({mpi_mark} and {petsc_mark})'])

    # Default flags for pytest
    flags.append('-ra')  # show extra test summary info for skipped, failed, etc.

    # Additional flags
    if verbose:
        flags.append('-v')
    if exitfirst:
        flags.append('-x')

    # Command to be executed
    cmd = [*mpi_exe, shutil.which('pytest'), *flags]

    # Print command
    cmd_to_print = [a.replace('(', '"(',).replace(')', ')"') for a in cmd]
    print('Executing command:')
    print(f' {" ".join(cmd_to_print)}', end='\n\n', flush=True)
    time.sleep(0.1)  # ensure the print is shown before subprocess output

    # Execute the command
    subprocess.run(cmd, shell=False, env=os.environ)
