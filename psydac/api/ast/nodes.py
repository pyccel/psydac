
from collections import OrderedDict
from sympy import symbols, Symbol

from pyccel.ast.core import IndexedVariable

from psydac.fem.vector  import BrokenFemSpace

from .utilities import variables

#==============================================================================
class BaseNode(object):

    def __init__(self, args, attributs=None):
        # ...
        if attributs is None:
            attributs = OrderedDict()

        elif isinstance(attributs, dict):
            attributs = OrderedDict(attributs)

        elif not isinstance(attributs, OrderedDict):
            raise TypeError('Expecting dict or OrderedDict, or None')
        # ...

        # ...
        self._args   = args
        self._attributs = attributs
        # ...

    @property
    def args(self):
        return self._args

    @property
    def attributs(self):
        return self._attributs

#==============================================================================
class BaseGrid(BaseNode):

    def __init__(self, args=None, target=None, dim=None, attributs=None):
        BaseNode.__init__(self, args, attributs=attributs)

        self._target = target
        self._dim    = dim

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return self._dim

#==============================================================================
class Grid(BaseGrid):

    def __init__(self, target, dim, name=None, label=None):
        # ...
        if name is None:
            name = 'grid'
        # ...

        # ...
        if label is None:
            label = ''

        else:
            label = '_{}'.format(label)
        # ...

        # ...
        name = '{name}{label}'.format(name=name, label=label)
        args = (Symbol(name),)
        # ...

        # ...
        names = ['n_elements{label}_{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        n_elements = variables(names, 'int')

        names = ['element{label}_s{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        element_starts = variables(names, 'int')

        names = ['element{label}_e{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        element_ends   = variables(names, 'int')

        names = ['points{label}_{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        points  = variables(names,  dtype='real', rank=2, cls=IndexedVariable)

        names = ['weights{label}_{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        weights = variables(names, dtype='real', rank=2, cls=IndexedVariable)

        names = ['k{label}_{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        quad_orders = variables(names, dtype='int')

        attributs = {'n_elements':     n_elements,
                     'element_starts': element_starts,
                     'element_ends':   element_ends,
                     'points':         points,
                     'weights':        weights,
                     'quad_orders':    quad_orders}
        # ...

        BaseGrid.__init__(self, args, target, dim,
                          attributs=attributs)

        # ...
        correspondance = {'n_elements':     'n_elements',
                          'element_starts': 'local_element_start',
                          'element_ends':   'local_element_end',
                          'points':         'points',
                          'weights':        'weights',
                          'quad_orders':    'quad_order'}

        self._correspondance = OrderedDict(correspondance)
        # ...

    @property
    def correspondance(self):
        return self._correspondance

    @property
    def n_elements(self):
        return self.attributs['n_elements']

    @property
    def element_starts(self):
        return self.attributs['element_starts']

    @property
    def element_ends(self):
        return self.attributs['element_ends']

    # TODO add quad order?

#==============================================================================
class GridInterface(BaseGrid):

    def __init__(self, target, dim):
        BaseGrid.__init__(self, args=None, target=target, dim=dim)

        self._minus = Grid(target, dim, label='minus')
        self._plus  = Grid(target, dim, label='plus')

    @property
    def grids(self):
        return self._grids

    @property
    def minus(self):
        return self._minus

    @property
    def plus(self):
        return self._plus

    @property
    def args(self):
        return self.minus.args + self.plus.args

    @property
    def n_elements(self):
        return self.minus.n_elements + self.plus.n_elements

    @property
    def element_starts(self):
        return self.minus.element_starts + self.plus.element_starts

    @property
    def element_ends(self):
        return self.minus.element_ends + self.plus.element_ends
