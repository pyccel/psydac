#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import argparse

import pytest

from psydac.utilities.parallel_utils import parallel_run_from_cli


@pytest.fixture
def abort_statuses(monkeypatch) -> list:
    """Record the calls to `mpi4py.run.set_abort_status`, which would
    otherwise terminate the test session at exit."""
    statuses = []
    monkeypatch.setattr('mpi4py.run.set_abort_status', statuses.append)
    # Figures left open by other tests would block `plt.show()` with an interactive backend
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    return statuses

def parser() -> argparse.Namespace:
    return argparse.Namespace(n=3)

#==============================================================================
def test_parallel_run_from_cli(abort_statuses: list) -> None:

    def main(*, n, mpi_comm):
        squares = [i**2 for i in range(n)]
        return locals()

    namespace = parallel_run_from_cli(parser, main)

    assert namespace['squares'] == [0, 1, 4]
    assert abort_statuses == []

def test_parallel_run_from_cli_abort(abort_statuses: list) -> None:

    def main(*, n, mpi_comm):
        raise RuntimeError('failure in main')

    # The exception is re-raised, and all processes will be aborted at exit
    with pytest.raises(RuntimeError, match='failure in main') as excinfo:
        parallel_run_from_cli(parser, main)

    assert abort_statuses == [excinfo.value]
