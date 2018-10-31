from .ast import SplBasic
from sympy import Basic
from .ast import Kron
from sympde.printing import pycode
from sympde.core import TensorProduct

class DiscreteKron(SplBasic):
    """ 
       Represent the discrete representation of A kron B

    """

    def __new__(cls, expr):
        
        obj  = Basic.__new__(cls, expr)
        func = pycode(Kron(expr)._initialize_dot)
        obj.local_dot = func
        return obj


class DiscreteLinearOperator(SplBasic):
    """ 
       Represent the discrete representation of Sum(Ai kron Bi)

    """

    def __new__(cls, expr):
        for arg in expr.args:
            if not isinstance(arg, TensorProduct):
                raise TypeError('args must be of type DiscreteKron')
        obj  = Basic.__new__(cls, expr)
        func = pycode(Kron(expr)._initialize_dot)
        obj.local_dot = func
        return obj

        
