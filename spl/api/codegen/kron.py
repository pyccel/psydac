from .ast import SplBasic
from .ast import Kron
from sympde.printing import pycode
from sympde.core import TensorProduct

class DiscreteKron(SplBasic):
    """ 
       Represent the discrete representation of A kron B

    """

    def __new__(cls, expr, space):
        
        obj = SplBasic.__new__(cls, expr)

        func = pycode(Kron(expr)._initialize_dot)
        dot = compile(func,'','single')
        dic = {}
        eval(dot, dic)
        func = dic[name]
 
        obj._space    = space
        obj.local_dot = func
        return obj

    @property
    def expr(self):
        return self.args[0]

    @property
    def space(self):
        return self._space

    def dot(self, x)

        space  = x.space
        starts = space.starts
        ends   = space.ends
        pads   = space.pads
        Out    = StencilVector(space)
        X_tmp  = StencilVector(space)
        #self.local_dot(starts, ends, pads, x._data.T, 
        #Out._data.T, X_tmp._data.T, *args1, *args2)

        return Out



class DiscreteLinearOperator(SplBasic):
    """ 
       Represent the discrete representation of Sum(Ai kron Bi)

    """

    def __new__(cls, expr, space):
        for arg in expr.args:
            if not isinstance(arg, TensorProduct):
                raise TypeError('args must be of type DiscreteKron')
        obj  = SplBasic.__new__(cls, expr)
        func = pycode(Kron(expr)._initialize_dot)
        dot = compile(func,'','single')
        dic = {}
        eval(dot, dic)
        func = dic[name]
 
        obj._space    = space
        obj.local_dot = func
        
        return obj

    @property
    def expr(self):
        return self.args[0]

    @property
    def space(self):
        return self._space

    def dot(self, x)

        space  = x.space
        starts = space.starts
        ends   = space.ends
        pads   = space.pads
        Out    = StencilVector(space)
        X_tmp  = StencilVector(space)
        #self.local_dot(starts, ends, pads, x._data.T, 
        #Out._data.T, X_tmp._data.T, *args1, *args2)
      
        return Out


        
