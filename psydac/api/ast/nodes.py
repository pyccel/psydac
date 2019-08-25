
from collections import OrderedDict
from sympy import symbols, Symbol

from pyccel.ast.core import Variable, IndexedVariable

from psydac.fem.vector  import BrokenFemSpace

from .utilities import variables

# TODO improve variables names


#==============================================================================
class BaseNode(object):

    def __init__(self, args=None, attributs=None):
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

    def __init__(self, args=None, target=None, dim=None, attributs=None,
                 element=None, quad=None):
        BaseNode.__init__(self, args, attributs=attributs)

        self._target  = target
        self._dim     = dim
        self._element = element
        self._quad    = quad

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return self._dim

    @property
    def element(self):
        return self._element

    @property
    def quad(self):
        return self._quad

#==============================================================================
class Grid(BaseGrid):

    def __init__(self, target, dim, name=None, label=None, axis_bnd=None):
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

        names = ['k{label}_{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        quad_orders = variables(names, dtype='int')

        attributs = {'n_elements':     n_elements,
                     'element_starts': element_starts,
                     'element_ends':   element_ends,
                     'quad_orders':    quad_orders}
        # ...

        element = Element(dim, axis_bnd=axis_bnd)
        quad    = GlobalQuadrature(element)

        BaseGrid.__init__(self, args, target, dim,
                          attributs=attributs, element=element, quad=quad)

        # ...
        correspondance = {'n_elements':     'n_elements',
                          'element_starts': 'local_element_start',
                          'element_ends':   'local_element_end',
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

    @property
    def quad_orders(self):
        return self.attributs['quad_orders']

#==============================================================================
class GridInterface(BaseGrid):

    def __init__(self, target, dim):
        BaseGrid.__init__(self, args=None, target=target, dim=dim)

        self._minus = Grid(target, dim, label='minus')
        self._plus  = Grid(target, dim, label='plus')

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

    @property
    def quad_orders(self):
        return self.minus.quad_orders + self.plus.quad_orders

    @property
    def points(self):
        return self.minus.points + self.plus.points

    @property
    def weights(self):
        return self.minus.weights + self.plus.weights

#==============================================================================
class BaseElement(BaseNode):
    """Represents an element."""

    def __init__(self, dim, attributs=None):
        BaseNode.__init__(self, attributs=attributs)

        self._dim = dim

    @property
    def dim(self):
        return self._dim

#==============================================================================
class Element(BaseElement):
    """Represents a tensor element."""

    def __init__(self, dim, axis_bnd=None, label=None):

        # ...
        if label is None:
            label = ''

        else:
            label = '_{}'.format(label)
        # ...

        # ...
        names = ['ie{label}_{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        indices_elm   = variables(names, 'int')
        # ...

        # ...
        attributs = {'indices_elm': indices_elm}
        # ...

        BaseElement.__init__(self, dim, attributs=attributs)

        # ...
        if axis_bnd is None:
            axis_bnd = []

        self._axis_bnd = axis_bnd
        # ...

    @property
    def args(self):
        return tuple(self.indices_elm)

    @property
    def indices_elm(self):
        return self.attributs['indices_elm']

    @property
    def axis_bnd(self):
        return self._axis_bnd

#==============================================================================
class ElementInterface(BaseElement):

    def __init__(self, dim, axis_minus=None, axis_plus=None):

        BaseElement.__init__(self, dim)

        self._minus = Element(dim, axis_minus, label='minus')
        self._plus  = Element(dim,  axis_plus,  label='plus')

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
    def indices_elm(self):
        return self.minus.indices_elm + self.plus.indices_elm

#==============================================================================
class BaseGlobalQuadrature(BaseNode):
    """Represents quadrature rule on an element."""

    def __init__(self, element=None, attributs=None):
        BaseNode.__init__(self, attributs=attributs)

        self._element = element
        self._local   = LocalQuadrature(element.dim)

    @property
    def element(self):
        return self._element

    @property
    def local(self):
        return self._local

    @property
    def dim(self):
        return self.element.dim

#==============================================================================
class GlobalQuadrature(BaseGlobalQuadrature):
    """Represents quadrature rule on a tensor element."""

    def __init__(self, element, label=None):
        assert(isinstance(element, Element))

        # ...
        if label is None:
            label = ''

        else:
            label = '_{}'.format(label)
        # ...

        dim = element.dim

        # ...
        names = ['points{label}_{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        points  = variables(names,  dtype='real', rank=2, cls=IndexedVariable)

        names = ['weights{label}_{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        weights = variables(names, dtype='real', rank=2, cls=IndexedVariable)
        # ...

        # ...
        attributs = {'points':  points,
                     'weights': weights}
        # ...

        BaseGlobalQuadrature.__init__(self, element, attributs=attributs)

        # ...
        correspondance = {'points':  'points',
                          'weights': 'weights'}

        self._correspondance = OrderedDict(correspondance)
        # ...

    @property
    def args(self):
        return tuple(self.points + self.weights)

    @property
    def points(self):
        return self.attributs['points']

    @property
    def weights(self):
        return self.attributs['weights']

    @property
    def correspondance(self):
        return self._correspondance

#==============================================================================
class GlobalQuadratureInterface(BaseGlobalQuadrature):

    def __init__(self, element):
        assert(isinstance(element, ElementInterface))

        BaseGlobalQuadrature.__init__(self, element)

        self._minus = GlobalQuadrature(element.minus, label='minus')
        self._plus  = GlobalQuadrature(element.plus,  label='plus')

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
    def points(self):
        return self.minus.points + self.plus.points

    @property
    def weights(self):
        return self.minus.weights + self.plus.weights


#==============================================================================
class BaseLocalQuadrature(BaseNode):
    """Represents quadrature rule on an element."""

    def __init__(self, dim, attributs=None):
        BaseNode.__init__(self, attributs=attributs)

        self._dim = dim

    @property
    def dim(self):
        return self._dim

#==============================================================================
class LocalQuadrature(BaseLocalQuadrature):
    """Represents quadrature rule on a tensor element."""

    def __init__(self, dim, label=None):

        # ...
        if label is None:
            label = ''

        else:
            label = '_{}'.format(label)
        # ...

        # ...
        names = ['quad_u{label}{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        points  = variables(names, dtype='real', rank=1, cls=IndexedVariable)

        names = ['quad_w{label}{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        weights = variables(names, dtype='real', rank=1, cls=IndexedVariable)

        names = ['g{label}{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        indices  = variables(names,  'int')

        names = ['k{label}{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        qds_dim = variables(names,  'int')

        wvol = Variable('real', 'wvol')
        # ...

        # ...
        attributs = {'points':  points,
                     'weights': weights,
                     'indices': indices,
                     'qds_dim': qds_dim,
                     'wvol':    wvol}
        # ...

        BaseLocalQuadrature.__init__(self, dim, attributs=attributs)

    @property
    def args(self):
        return tuple(self.points + self.weights)

    @property
    def points(self):
        return self.attributs['points']

    @property
    def weights(self):
        return self.attributs['weights']

    @property
    def indices(self):
        return self.attributs['indices']

    @property
    def qds_dim(self):
        return self.attributs['qds_dim']

    @property
    def wvol(self):
        return self.attributs['wvol']

#==============================================================================
class BaseGlobalBasis(BaseNode):
    """Represents quadrature rule on an element."""
    _kind = None

    def __init__(self, element, kind=None, attributs=None, ln=1):
        BaseNode.__init__(self, attributs=attributs)

        self._element = element
        self._ln      = ln
        self._kind    = kind

    @property
    def element(self):
        return self._element

    @property
    def ln(self):
        return self._ln

    @property
    def dim(self):
        return self.element.dim

    @property
    def kind(self):
        return self._kind

    @property
    def is_test(self):
        return self.kind == 'test'

    @property
    def is_trial(self):
        return self.kind == 'trial'

#==============================================================================
class GlobalBasis(BaseGlobalBasis):
    """Represents quadrature rule on a tensor element."""

    def __init__(self, element, kind=None, label=None, ln=1):
        assert(isinstance(element, BaseElement))

        # ...
        if label is None:
            label = ''

        else:
            label = '_{}'.format(label)
        # ...

        dim = element.dim

        # ...
        if kind == 'test':
            index = 'i'

        elif kind == 'trial':
            index = 'j'
        # ...

        # ...
        if kind is None:
            kind_str = ''

        else:
            kind_str = '{}_'.format(kind)
        # ...

        # ...
        names = ['is{label}{j}{i}'.format(label=label, j=j, i=i)
                 for j in range(1,dim+1) for i in range(1, ln+1)]
        indices_span = variables(names, 'int')

        names = ['{kind}p{label}{j}{i}'.format(kind=kind_str, label=label, j=j, i=i)
                 for j in range(1,dim+1) for i in range(1, ln+1)]
        pads = variables(names, 'int')

        names = ['{kind}d{label}{j}{i}'.format(kind=kind_str, label=label, j=j, i=i)
                 for j in range(1,dim+1) for i in range(1, ln+1)]
        degrees = variables(names, 'int')

        names = ['{index}l{label}{j}'.format(index=index, label=label, j=j)
                 for j in range(1,dim+1)]
        indices_l = variables(names, 'int')

        names = ['{index}{label}{j}'.format(index=index, label=label, j=j)
                 for j in range(1,dim+1)]
        indices = variables(names,  'int')

        names = ['n{label}{j}'.format(label=label, j=j)
                 for j in range(1,dim+1)]
        npts = variables(names,  'int')

        names = ['{kind}basis{label}_{j}{i}'.format(kind=kind_str, label=label, j=j, i=i)
                 for j in range(1,dim+1) for i in range(1, ln+1)]
        basis = variables(names, dtype='real', rank=4, cls=IndexedVariable)

        names = ['{kind}spans{label}_{j}{i}'.format(kind=kind_str, label=label, j=j, i=i)
                 for j in range(1,dim+1) for i in range(1, ln+1)]
        spans = variables(names, dtype='int', rank=1, cls=IndexedVariable)

        names = ['{kind}bs{label}{j}{i}'.format(kind=kind_str, label=label, j=j, i=i)
                 for j in range(1,dim+1) for i in range(1, ln+1)]
        basis_in_elm = variables(names, dtype='real', rank=3, cls=IndexedVariable)
        # ...

        # ...
        attributs = {'indices_span':  indices_span,
                     'pads':          pads,
                     'degrees':       degrees,
                     'indices_l':     indices_l,
                     'indices':       indices,
                     'npts':          npts,
                     'basis':         basis,
                     'spans':         spans,
                     'basis_in_elm':  basis_in_elm}
        # ...

        BaseGlobalBasis.__init__(self, element, kind=kind, attributs=attributs, ln=ln)

    @property
    def args(self):
        raise NotImplementedError('TODO')

    @property
    def indices_span(self):
        return self.attributs['indices_span']

    @property
    def pads(self):
        return self.attributs['pads']

    @property
    def degrees(self):
        return self.attributs['degrees']

    @property
    def indices_l(self):
        return self.attributs['indices_l']

    @property
    def indices(self):
        return self.attributs['indices']

    @property
    def npts(self):
        return self.attributs['npts']

    @property
    def basis(self):
        return self.attributs['basis']

    @property
    def spans(self):
        return self.attributs['spans']

    @property
    def basis_in_elm(self):
        return self.attributs['basis_in_elm']


#==============================================================================
class BaseLocalBasis(BaseNode):
    """Represents quadrature rule on an element."""
    _kind = None

    def __init__(self, dim, kind=None, attributs=None):
        BaseNode.__init__(self, attributs=attributs)

        self._dim  = dim
        self._kind = kind

    @property
    def dim(self):
        return self._dim

    @property
    def kind(self):
        return self._kind

    @property
    def is_test(self):
        return self.kind == 'test'

    @property
    def is_trial(self):
        return self.kind == 'trial'

#==============================================================================
class LocalBasis(BaseLocalBasis):
    """Represents quadrature rule on a tensor element."""

    def __init__(self, dim, kind=None, label=None):

        # ...
        if label is None:
            label = ''

        else:
            label = '_{}'.format(label)
        # ...

        # ...
        if kind == 'test':
            index = 'i'

        elif kind == 'trial':
            index = 'j'
        # ...

        # ...
        if kind is None:
            kind_str = ''

        else:
            kind_str = '{}_'.format(kind)
        # ...

        # ...
        names = ['{kind}p{label}{j}'.format(kind=kind_str, label=label, j=j)
                 for j in range(1,dim+1)]
        pads = variables(names, 'int')

        names = ['{kind}d{label}{j}'.format(kind=kind_str, label=label, j=j)
                 for j in range(1,dim+1)]
        degrees = variables(names, 'int')

        names = ['{index}l{label}{j}'.format(index=index, label=label, j=j)
                 for j in range(1,dim+1)]
        indices = variables(names, 'int')

        names = ['{kind}bs{label}{j}'.format(kind=kind_str, label=label, j=j)
                 for j in range(1,dim+1)]
        basis = variables(names, dtype='real', rank=3, cls=IndexedVariable)
        # ...

        # ...
        attributs = {'pads':    pads,
                     'degrees': degrees,
                     'indices': indices,
                     'basis':   basis}
        # ...

        BaseLocalBasis.__init__(self, dim, kind=kind, attributs=attributs)

    @property
    def args(self):
        raise NotImplementedError('TODO')

    @property
    def pads(self):
        return self.attributs['pads']

    @property
    def degrees(self):
        return self.attributs['degrees']

    @property
    def indices(self):
        return self.attributs['indices']

    @property
    def basis(self):
        return self.attributs['basis']
