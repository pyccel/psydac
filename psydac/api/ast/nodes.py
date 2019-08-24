
from sympy import symbols, Symbol

from psydac.fem.vector  import BrokenFemSpace


#==============================================================================
class BasicNode(object):

    @property
    def target(self):
        return self._target

    @property
    def args(self):
        return self._args

#==============================================================================
class Grid(BasicNode):

    def __init__(self, target, dim):
        self._target = target
        self._dim    = dim
        self._args   = (Symbol('grid'),)

    @property
    def dim(self):
        return self._dim

#==============================================================================
class GridInterface(BasicNode):

    def __init__(self, target, dim):
        self._target = target
        self._dim    = dim
        self._args   = (Symbol('grid_minus'), Symbol('grid_plus'))

    @property
    def dim(self):
        return self._dim
