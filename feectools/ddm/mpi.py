"""Detection of whether the process was launched by an MPI launcher.

Importing ``mpi4py.MPI`` calls ``MPI_Init``, and any collective (``bcast``,
``Barrier``, ...) issued afterwards costs something even on a single process.
A plain ``python script.py`` run should therefore never touch MPI at all, even
when mpi4py happens to be installed. This module answers the only question
that decides it: was this process started by ``mpirun``/``mpiexec``/``srun``
(or an equivalent launcher)?

The answer is read from the environment the launcher sets up, so it is
available before mpi4py is imported.
"""

import os
import sys

from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING


# Per-process variables exported by the process managers behind the common
# launchers. Each is set only for processes started *by* the launcher, so the
# presence of any one of them means "this rank belongs to an MPI job".
# SLURM_PROCID is deliberately absent: it is also set for the script of a
# plain `sbatch` job, which is not an MPI launch. `srun` is covered by the
# PMI/PMIX variables its MPI plugin exports.
_LAUNCHER_ENV_VARS = (
    "OMPI_COMM_WORLD_RANK",  # Open MPI (and derivatives: Spectrum, ...)
    "PMI_RANK",  # MPICH, Intel MPI, MS-MPI, Cray, srun (pmi2)
    "PMIX_RANK",  # PMIx, used by srun --mpi=pmix and Open MPI 5
    "MV2_COMM_WORLD_RANK",  # MVAPICH2
    "MPI_LOCALRANKID",  # Hydra (mpiexec.hydra)
    "ALPS_APP_PE",  # Cray ALPS aprun
    "PALS_RANKID",  # Cray PALS palsrun
)

# Escape hatch: force the decision either way without touching code, e.g. for
# a launcher whose variables are not listed above.
_OVERRIDE_ENV_VAR = "STRUPHY_MPI"

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")


def _override() -> bool | None:
    """Value of ``STRUPHY_MPI``, or None if unset/unrecognized."""
    value = os.environ.get(_OVERRIDE_ENV_VAR)
    if value is None:
        return None
    value = value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return None


def launched_under_mpi() -> bool:
    """Whether this process was started by an MPI launcher.

    Returns
    -------
    bool
        True if a launcher's per-rank environment variable is present, or if
        the application itself already initialized MPI (in which case using
        the communicator is free). ``STRUPHY_MPI=0``/``1`` overrides
        the detection.
    """
    override = _override()
    if override is not None:
        return override

    if any(var in os.environ for var in _LAUNCHER_ENV_VARS):
        return True

    # The application may have initialized MPI itself (embedded interpreter,
    # or an explicit `from mpi4py import MPI`). Only inspect mpi4py if it is
    # already imported: importing it here is exactly what must be avoided.
    mpi_module = sys.modules.get("mpi4py.MPI")
    if mpi_module is not None:
        try:
            return bool(mpi_module.Is_initialized())
        except AttributeError:
            return False

    return False


# Might not be needed
class MPICommWrapper:
    def __init__(self, use_mpi=True):
        self.use_mpi = use_mpi
        if use_mpi:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
        else:
            self.comm = MockComm()

    def __getattr__(self, name):
        return getattr(self.comm, name)


class MockComm:
    def __getattr__(self, name):
        # Return a function that does nothing and returns None
        def dummy(*args, **kwargs):
            return None

        return dummy

    # Override some functions
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        return


class MPIwrapper:
    def __init__(
        self,
        use_mpi: bool = False,
        verbose: bool = False,
    ):
        self.use_mpi = use_mpi
        if use_mpi:
            from mpi4py import MPI

            self._MPI = MPI
            if verbose:
                print("MPI is enabled")
        else:
            self._MPI = MockMPI()
            if verbose:
                print("MPI is NOT enabled")

    @property
    def MPI(self):
        return self._MPI


class MockMPI:
    def __getattr__(self, name):
        # Return a function that does nothing and returns None
        def dummy(*args, **kwargs):
            return None

        return dummy

    # Override some functions
    @property
    def COMM_WORLD(self):
        return MockComm()

    # def comm_Get_rank(self):
    #     return 0

    # def comm_Get_size(self):
    #     return 1


def _mpi_disabled():
    """True if the user or the host application opted out of MPI.

    Importing mpi4py initializes MPI, which takes close to a second. Serial runs that
    never use MPI can skip it entirely, in two ways:

    * export ``FEECTOOLS_MPI=0`` before starting Python, or
    * set ``feectools.use_mpi = False`` before this module is first imported
      (in-process, so it is not inherited by subprocesses such as ``mpirun``).

    The MockMPI wrapper below is then used, exactly as if mpi4py were not installed.
    """
    import os

    import feectools

    if getattr(feectools, 'use_mpi', None) is False:
        return True
    return os.environ.get('FEECTOOLS_MPI', '').strip().lower() in ('0', 'false', 'no', 'off')


if launched_under_mpi():
    try:
        # MPI with the CuPy backend needs a CUDA-aware MPI library: device
        # buffers are passed to MPI directly (after synchronize_for_mpi, see
        # the data exchangers). A non-CUDA-aware MPI segfaults on them.
        if _mpi_disabled():
            raise ImportError("MPI disabled (feectools.use_mpi = False or FEECTOOLS_MPI=0)")

        from mpi4py import MPI

        _comm = MPI.COMM_WORLD
        # rank = _comm.Get_rank()
        # size = _comm.Get_size()
        mpi_enabled = True
    except ImportError:
        # mpi4py not installed
        mpi_enabled = False
    except Exception:
        # mpi4py installed but not running under mpirun
        mpi_enabled = False
else:
    mpi_enabled = False

# TODO: add environment variable for mpi use
mpi_wrapper = MPIwrapper(
    use_mpi=mpi_enabled,
    verbose=False,
)

# TYPE_CHECKING is True when type checking (e.g., mypy), but False at runtime.
if TYPE_CHECKING:
    from mpi4py import MPI

    mpi = MPI
else:
    mpi = mpi_wrapper.MPI
