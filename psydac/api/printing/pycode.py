#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympy.core import Symbol
from sympy.core import S
from sympy.printing.precedence import precedence

from psydac.pyccel.codegen.printing.pycode import PythonCodePrinter as PyccelPythonCodePrinter

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

    def _print_MinusInterfaceOperator(self, expr):
        return self._print(expr.args[0])

    def _print_PlusInterfaceOperator(self, expr):
        return self._print(expr.args[0])

    def _print_FloorDiv(self, expr):
        return "(({})//({}))".format(self._print(expr.arg1), self._print(expr.arg2))

    # .........................................................
    #        SYMPY objects
    # .........................................................
    def _print_AppliedUndef(self, expr):
        args = ','.join(self._print(i) for i in expr.args)
        fname = self._print(expr.func.__name__)
        return '{fname}({args})'.format(fname=fname, args=args)

    def _print_PythonTuple(self, expr):
        args = ', '.join(self._print(i) for i in expr.args)
        return '('+args+')'

    def _hprint_Pow(self, expr, rational=False, sqrt='math.sqrt'):
        """
        Printing helper function for ``Pow``.
        See also: sympy.printing.str.StrPrinter._print_Pow

        Notes
        -----
        This only preprocesses the ``sqrt`` as math formatter

        Examples
        --------
        >>> from sympy.functions import sqrt
        >>> from sympy.printing.pycode import PythonCodePrinter
        >>> from sympy.abc import x
        Python code printer automatically looks up ``math.sqrt``.
        >>> printer = PythonCodePrinter({'standard':'python3'})
        >>> printer._hprint_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._hprint_Pow(sqrt(x), rational=False)
        'math.sqrt(x)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=False)
        '1/math.sqrt(x)'
        Using sqrt from numpy or mpmath
        >>> printer._hprint_Pow(sqrt(x), sqrt='numpy.sqrt')
        'numpy.sqrt(x)'
        >>> printer._hprint_Pow(sqrt(x), sqrt='mpmath.sqrt')
        'mpmath.sqrt(x)'

        """
        PREC = precedence(expr)

        if expr.exp == S.Half and not rational:
            func = self._module_format(sqrt)
            arg = self._print(expr.base)
            return '{func}({arg})'.format(func=func, arg=arg)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                func = self._module_format(sqrt)
                num = self._print(S.One)
                arg = self._print(expr.base)
                return "{num}/{func}({arg})".format(
                    num=num, func=func, arg=arg)

        base_str = self.parenthesize(expr.base, PREC, strict=False)
        exp_str = self.parenthesize(expr.exp, PREC, strict=False)
        return "{}**{}".format(base_str, exp_str)

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='sqrt')

    def _print_Idx(self, expr):
        return self._print(str(expr))

#==============================================================================
def pycode(expr, **settings):
    """ Converts an expr to a string of Python code

    Parameters
    ----------
    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    enable_dependencies: bool
        Whether or not to print dependencies too (EvalField, Kernel, etc)

    Examples
    --------
    >>> from sympy import tan, Symbol
    >>> from sympy.printing.pycode import pycode
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'
    """
    return PythonCodePrinter(settings).doprint(expr)
