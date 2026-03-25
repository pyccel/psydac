#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
This module contains only tests which are designed to fail, to check that the
error handling in the `psydac test` command works correctly. This will be used
in the CI to verify that the test suite correctly reports failures, by running
`psydac test` on this file and verifying that the CI fails as expected.
"""
import pytest


msg_tmp = "This {}test is designed to fail to check error handling in the test suite."

def test_failure():
    assert False, msg_tmp.format("")

@pytest.mark.mpi
def test_failure_mpi():
    assert False, msg_tmp.format("MPI ")

@pytest.mark.petsc
def test_failure_petsc():
    assert False, msg_tmp.format("PETSc ")
