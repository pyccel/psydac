from sympy.core import Symbol

from pyccel.codegen.printing.pycode import PythonCodePrinter as PyccelPythonCodePrinter

from sympde.topology.derivatives import _partial_derivatives
from sympde.topology             import SymbolicExpr

#==============================================================================
class PythonCodePrinter(PyccelPythonCodePrinter):

    def __init__(self, settings=None):
        self._enable_dependencies = settings.pop('enable_dependencies', True)

        PyccelPythonCodePrinter.__init__(self, settings=settings)

    # .........................................................
    #      PSYDAC objects
    # .........................................................
    def _print_SplBasic(self, expr):
        code = ''
        if self._enable_dependencies and expr.dependencies:
            imports = []
            for dep in expr.dependencies:
                imports +=dep.imports
            code = '\n'.join(self._print(i) for i in imports)
            for dep in expr.dependencies:
                code = '{code}\n{dep}'.format(code=code,
                                              dep=self._print(dep))

        return '{code}\n{func}'.format(code=code, func=self._print(expr.func))
        
    def _print_Kernel(self, expr):

        code = ''
        if self._enable_dependencies and expr.dependencies:
            imports = []
            for dep in expr.dependencies:
                imports +=dep.imports
            code = '\n'.join(self._print(i) for i in imports)
            for dep in expr.dependencies:
                code = '{code}\n{dep}'.format(code=code,
                                              dep=self._print(dep))
                                              
        funcs = [func for fs in expr.func for func in fs if func is not None ]

        funcs = '\n'.join(self._print(func) for func in funcs)
        
        return '{code}\n{funcs}'.format(code=code, funcs=funcs)

    def _print_Interface(self, expr):
        code = '\n'.join(self._print(i) for i in expr.imports)

        return code +'\n' + self._print(expr.func)

    def _print_GltInterface(self, expr):
        code = '\n'.join(self._print(i) for i in expr.imports)

        return code +'\n' + self._print(expr.func)

    # .........................................................
    #        SYMPY objects
    # .........................................................
    def _print_AppliedUndef(self, expr):
        if not expr._imp_:
            raise ValueError('_imp_ not impltemented')

        args = ','.join(self._print(i) for i in expr.args)
        fname = self._print(expr.func.__name__)
        return '{fname}({args})'.format(fname=fname, args=args)

#==============================================================================
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
