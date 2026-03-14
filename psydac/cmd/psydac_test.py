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
            import importlib
            importlib.import_module(mod)
        except ImportError:
            exit_with_error_message(f"module '{mod}' not found")

    # Import modules here to speed up parser
    import shutil
    import subprocess

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
        flags.extend(['-n', 'auto'])

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
    flags.extend(['-m', f'{mpi_mark} and {petsc_mark}'])

    # Default flags for pytest
    flags.append('-ra')  # show extra test summary info for skipped, failed, etc.

    # Additional flags
    if verbose:
        flags.append('-v')
    if exitfirst:
        flags.append('-x')

    # Command to be executed
    cmd = [*mpi_exe, shutil.which('pytest'), *flags]

    # Execute the command
    print('Executing command:')
    print(f' {" ".join(cmd)}\n')
    subprocess.run(cmd, shell=False)
