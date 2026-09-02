#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Unit test for the OpenMP `ParallelBlock` code-generation branch in the general
AST path (`psydac.api.ast.fem`, guarded by
`add_openmp = is_pyccel and backend['openmp'] and num_threads>1`).

Note: `discretize()` only propagates `num_threads>1` into this branch when the
discrete space is MPI-parallel (`coeff_space.parallel`, i.e. a communicator was
passed in) -- in a plain serial run, `OMP_NUM_THREADS` has *no effect* on code
generation, however high it is set, since `num_threads` silently stays at 1.
This is why no existing test (all of which run without an MPI communicator)
actually exercises `ParallelBlock`, even the ones that use an `openmp=True`
backend.

To exercise it safely without requiring an MPI environment, this test builds
the `AST` object directly (as in `psydac/psydac/api/ast/tests/test_poisson.py`
etc.) with an explicit `num_threads` argument, and inspects the generated
Python source for the `#$ omp parallel` pragma emitted by `ParallelBlock`
(`psydac.api.ast.parser`). No compilation is involved, so no Fortran
toolchain is required.
"""
from sympde.topology import Square, ScalarFunctionSpace
from sympde.topology import element_of
from sympde.expr      import LinearForm, integral
from sympde.expr.evaluation import TerminalExpr

from psydac.api.discretization import discretize
from psydac.api.ast.fem        import AST
from psydac.api.ast.parser     import parse
from psydac.api.printing.pycode import pycode
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_PYTHON

PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP           = PSYDAC_BACKEND_GPYCCEL.copy()
PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP['openmp'] = True

#==============================================================================
def _generate_linear_form_code(backend, num_threads):

    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    v = element_of(V, name='v')
    l = LinearForm(v, integral(domain, v))

    domain_h = discretize(domain, ncells=(6, 6))
    Vh       = discretize(V, domain_h, degree=(2, 2))

    kernel_expr = TerminalExpr(l, domain)[0]
    ast  = AST(l, kernel_expr, Vh, nquads=(3, 3), backend=backend, num_threads=num_threads)
    stmt = parse(ast.expr, settings={'dim': 2, 'nderiv': 0, 'mapping': None, 'target': domain}, backend=backend)

    return pycode(stmt)

#==============================================================================
def test_parallel_block_only_generated_when_openmp_backend_and_multiple_threads():

    # backend['openmp'] = True, but num_threads = 1 => no ParallelBlock.
    code_one_thread = _generate_linear_form_code(PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP, num_threads=1)
    assert '#$ omp parallel' not in code_one_thread

    # backend['openmp'] = True and num_threads > 1 => ParallelBlock generated.
    code_parallel = _generate_linear_form_code(PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP, num_threads=4)
    assert '#$ omp parallel' in code_parallel

    # Even with num_threads > 1, the 'python' backend (openmp=False, not
    # pyccel) must never generate a ParallelBlock.
    code_python_backend = _generate_linear_form_code(PSYDAC_BACKEND_PYTHON, num_threads=4)
    assert '#$ omp parallel' not in code_python_backend
