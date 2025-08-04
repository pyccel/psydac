import string
import random
from typing import Iterable

from sympy import Expr, ImmutableDenseMatrix, Matrix

from sympde.expr.basic           import BasicForm
from sympde.expr.evaluation      import KernelExpression
from sympde.topology.space       import ScalarFunction, VectorFunction, IndexedVectorFunction
from sympde.topology.derivatives import get_atom_logical_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives

from psydac.fem.basic         import FemSpace
from psydac.api.ast.utilities import math_atoms_as_str, get_max_partial_derivatives

# TODO [YG 01.08.2025]: Avoid importing anything from psydac.pyccel
from psydac.pyccel.ast.core import _atomic

#==============================================================================
def generate_random_id(size: int = 8,
                chars: int = string.ascii_lowercase + string.digits) -> str:
    """
    Create a random string of given length to be used in generated file names.

    Parameters
    ----------
    size : int, optional
        Length of the string (default: 8).

    chars : str, optional
        A string with the avalailable characters for random drawing (default:
        ASCII lower case characters + decimal digits)

    Returns
    -------
    str
        A random string of given length, made of the given characters.
    """
    return ''.join(random.choice(chars) for _ in range(size))

#==============================================================================
def compute_max_nderiv(kernel_expr: KernelExpression) -> int:
    """
    Compute the highest derivative order in the given kernel expression.

    Parameters
    ----------
    kernel_expr : KernelExpression (from sympde.expr.evaluation)

    Returns
    -------
    nderiv : int
        The highest order of derivation in `terminal_expr`.

    """
    assert isinstance(kernel_expr, KernelExpression)

    terminal_expr = kernel_expr.expr
    if not isinstance(terminal_expr, (ImmutableDenseMatrix, Matrix)):
        terminal_expr = ImmutableDenseMatrix([[terminal_expr]])
    n_rows, n_cols = terminal_expr.shape

    atoms_types = (ScalarFunction, VectorFunction, IndexedVectorFunction)
    extended_atoms_types = atoms_types + _logical_partial_derivatives

    nderiv = 0
    for i_row in range(n_rows):
        for i_col in range(n_cols):
            texpr = terminal_expr[i_row, i_col]
            atoms = _atomic(texpr, cls=extended_atoms_types)
            Fs = [get_atom_logical_derivatives(a) for a in atoms]
            d = get_max_partial_derivatives(texpr, logical=True, F=Fs)
            nderiv = max(nderiv, max(d.values()))

    return nderiv

#==============================================================================
def compute_imports(expr: Expr,
                    spaces: Iterable[FemSpace],
                    *,
                    openmp: bool
                    ) -> dict[str, list[str]]:
    """
    Compute all the imports to be added to the generated Python code.

    Parameters
    ----------
    expr : sympy.Expr
        The integrand expression of a BilinearForm, LinearForm, or
        Functional. This is a pure SymPy expression where SymPDE partial
        derivatives have been converted to SymPy symbols. See Notes.

    spaces : iterable of psydac.fem.FemSpace
        The discrete spaces which define the finite element representation
        of a BilinearForm, LinearForm, or Functional.

    openmp : bool
        Whether or not OpenMP pragmas and functions are used in the code.

    Returns
    -------
    imports : dict[str, list[str]]
        A dictionary whose keys are the names of the Python modules to be
        imported, and whose values are the names of the corresponding
        objects (variables, functions, classes) to be imported from the
        modules.

    Notes
    -----
    Assume that we start from an object of type BilinearForm, LinearForm,
    or Functional. We take the integrand and expand its vector operations
    with TerminalExpr(), then pull back from physical to logical
    coordinates with LogicalExpr(), and finally convert the symbolic
    partial derivatives with SymbolicExpr(). Where:
        - TerminalExpr is defined in sympde.expr.evaluation
        -  LogicalExpr is defined in sympde.topology.mapping
        - SymbolicExpr is defined in sympde.topology.mapping

    The resulting expression `expr` can be passed to this function.
    """
    assert isinstance(expr, Expr)
    assert all(isinstance(V, FemSpace) for V in spaces)
    assert isinstance(openmp, bool)

    # Determine the type of scalar quantities to be managed in the code
    dtypes = [getattr(V.symbolic_space, 'codomain_type', 'real') for V in spaces]
    assert all(t in ['complex', 'real'] for t in dtypes)
    dtype = 'complex' if 'complex' in dtypes else 'real'

    # TODO uncomment this line when we have a SesquilinearForm defined in SymPDE
    #assert isinstance(expr, SesquilinearForm)

    #... Compute the imports
    math_library  = 'cmath' if dtype=='complex' else 'math' # Function names are the same
    math_imports  = math_atoms_as_str(expr, 'math')
    numpy_imports = ['array', 'zeros', 'zeros_like', 'floor']

    imports = {'numpy': numpy_imports}
    if math_imports:
        imports[math_library] = math_imports
    if openmp:
        imports['pyccel.stdlib.internal.openmp'] = ['omp_get_thread_num']
    #...

    return imports

#==============================================================================
def compute_free_arguments(expr: BasicForm, kernel_expr: KernelExpression) -> tuple[str]:
    """
    The string representation (i.e. the names) of the free arguments in
    the given BilinearForm, LinearForm, or Functional.

    Parameters
    ----------
    expr : BilinearForm | LinearForm | Functional
        The expression of which we want to compute the free arguments.

    kernel_expr : sympde.expr.evaluation.KernelExpression
        The atomic representation of the form, which is obtained after using
        LogicalExpr (if there is a mapping) and TerminalExpr on `expr`.

    Returns
    -------
    tuple[str]
        The string representation (i.e. the names) of the free arguments in
        the given BilinearForm, LinearForm, or Functional.

    """
    assert isinstance(expr, BasicForm)
    assert isinstance(kernel_expr, KernelExpression)

    free_args_dict = expr.get_free_variables()
    free_args_str  = tuple(str(a) for a in free_args_dict)

    return free_args_str
