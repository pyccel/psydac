from sympy.core import Symbol
from sympy import Tuple

from sympde.core.expr import BilinearForm, LinearForm, FunctionForm
from sympde.core.generic import Dot, Inner, Cross
from sympde.core.generic import Grad, Rot, Curl, Div
from sympde.core.geometry import Line, Square, Cube
from sympde.core.derivatives import _partial_derivatives
from sympde.printing.pycode import PythonCodePrinter as SympdePythonCodePrinter


class PythonCodePrinter(SympdePythonCodePrinter):

    def __init__(self, settings=None):
        self._enable_dependencies = settings.pop('enable_dependencies', True)

        SympdePythonCodePrinter.__init__(self, settings=settings)

    def _print_SplBasic(self, expr):
        code = ''
        if self._enable_dependencies and expr.dependencies:
            for dep in expr.dependencies:
                code = '{code}\n{dep}'.format(code=code,
                                              dep=self._print(dep))

        return '{code}\n{func}'.format(code=code, func=self._print(expr.func))

    def _print_Interface(self, expr):
        return self._print(expr.func)


def pycode(expr, **settings):
    """ Converts an expr to a string of Python code
    Parameters
    ==========
    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    enable_dependencies: bool
        Whether or not to print dependencies too (EvalField, Kernel, etc)
    Examples
    ========
    >>> from sympy import tan, Symbol
    >>> from sympy.printing.pycode import pycode
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'
    """
    return PythonCodePrinter(settings).doprint(expr)
