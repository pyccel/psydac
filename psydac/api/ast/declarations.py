
import numpy as np
from sympy import symbols, Symbol

from pyccel.ast.core import IndexedVariable
from pyccel.ast.core import Assign
from pyccel.ast import DottedName
from pyccel.ast.core import Slice
from pyccel.ast.core import Range

from .utilities import variables
from .nodes     import ElementInterface
from .nodes     import Grid

# ...
def init_loop_support(element, n_elements,
                      indices_span, spans, ranges,
                      points_in_elm, points,
                      weights_in_elm, weights,
                      test_basis_in_elm, test_basis,
                      trial_basis_in_elm, trial_basis,
                      is_bilinear, discrete_boundary):
    stmts = []
    if not discrete_boundary:
        return stmts

    # ...
    dim = element.dim
    # ARA we only take the minus one
    if isinstance(element, ElementInterface):
        indices_elm = element.indices[:dim]

    else:
        indices_elm = element.indices
    # ...

    # TODO improve using namedtuple or a specific class ? to avoid the 0 index
    #      => make it easier to understand
    quad_mask = [i[0] for i in discrete_boundary]
    quad_ext  = [i[1] for i in discrete_boundary]

    for i in range(dim-1,-1,-1):
        rx = ranges[i]
        x = indices_elm[i]

        if i in quad_mask:
            i_index = quad_mask.index(i)
            ext = quad_ext[i_index]

            if ext == -1:
                value = rx.start

            elif ext == 1:
                value = rx.stop - 1

            stmts += [Assign(x, value)]

    axis = quad_mask[0]

    # ... assign element index
    ncells = n_elements[axis]
    ie = indices_elm[axis]
    # ...

    # ... assign span index
    i_span = indices_span[axis]
    stmts += [Assign(i_span, spans[axis][ie])]
    # ...

    # ... assign points, weights and basis
    # ie is substitute by 0
    # sympy does not like ':'
    _slice = Slice(None,None)

    stmts += [Assign(points_in_elm[axis], points[axis][0,_slice])]
    stmts += [Assign(weights_in_elm[axis], weights[axis][0,_slice])]
    stmts += [Assign(test_basis_in_elm[axis], test_basis[axis][0,_slice,_slice,_slice])]

    if is_bilinear:
        stmts += [Assign(trial_basis_in_elm[axis], trial_basis[axis][0,_slice,_slice,_slice])]
    # ...

    return stmts
# ...



#==============================================================================
class DeclarationGenerator(object):

    def __init__(self, settings=None):
        self._settings = settings

    @property
    def settings(self):
        return self._settings

    def doit(self, expr):
        return self._visit(expr, **self.settings)

    def _visit(self, expr, **settings):
        classes = type(expr).__mro__
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(expr, **settings)

        # Unknown object, we raise an error.
        raise NotImplementedError('{}'.format(type(expr)))

    # ....................................................
    #           AssemblyNode
    # ....................................................
    def _visit_AssemblyNode(self, expr):
        grid             = expr.grid
        test_basis_node  = expr.test_basis
        trial_basis_node = expr.trial_basis
        discrete_boundary = expr.discrete_boundary # TODO
        is_bilinear       = expr.is_bilinear # TODO
        is_function       = expr.is_function # TODO

        dim            = grid.dim
        element        = grid.element
        quad           = grid.quad
        n_elements     = grid.n_elements
        element_starts = grid.element_starts
        element_ends   = grid.element_ends

        points_in_elm  = quad.local.points
        weights_in_elm = quad.local.weights
        points         = quad.points
        weights        = quad.weights

        indices_span       = test_basis_node.indices_span
        spans              = test_basis_node.spans
        test_basis         = test_basis_node.basis
        test_basis_in_elm  = test_basis_node.basis_in_elm
        trial_basis        = trial_basis_node.basis
        trial_basis_in_elm = trial_basis_node.basis_in_elm
        # ...

        # ...
        if is_function:
            ranges_elm  = [Range(s, e+1) for s,e in zip(element_starts, element_ends)]

        else:
            ranges_elm  = [Range(0, n_elements[i]) for i in range(dim)]
        # ...

        body = init_loop_support( element, n_elements,
                                  indices_span, spans, ranges_elm,
                                  points_in_elm, points,
                                  weights_in_elm, weights,
                                  test_basis_in_elm, test_basis,
                                  trial_basis_in_elm, trial_basis,
                                  is_bilinear, discrete_boundary )
        return body

    # ....................................................
    #           Grid
    # ....................................................
    def _visit_Grid(self, expr):
        grid = expr.args[0]
        element = expr.element
        quad    = expr.quad

        body = []
        for parent in [expr, quad]:
            for k,v in parent.attributs.items():
                body += [Assign(v, DottedName(grid, parent.correspondance[k]))]

        return body

    def _visit_GridInterface(self, expr):
        body  = []
        body += self._visit(expr.minus)
        body += self._visit(expr.plus)

        return body

    # ....................................................
    #           Element
    # ....................................................
    def _visit_Element(self, expr):
        raise NotImplementedError('TODO')

    def _visit_ElementInterface(self, expr):
        raise NotImplementedError('TODO')

    # ....................................................
    #           Quadrature
    # ....................................................
    def _visit_GlobalQuadrature(self, expr):
        # sympy does not like ':'
        _slice = Slice(None,None)

        dim     = expr.dim
        element = expr.element

        # local quadrature attributs
        points_in_elm  = expr.local.points
        weights_in_elm = expr.local.weights

        # global quadrature attributs
        points  = expr.points
        weights = expr.weights

        # element attributs
        indices_elm    = element.indices
        axis_bnd       = element.axis_bnd

        body = []

        body += [Assign(points_in_elm[i], points[i][indices_elm[i],_slice])
                 for i in range(dim) if not(i in axis_bnd) ]

        body += [Assign(weights_in_elm[i], weights[i][indices_elm[i],_slice])
                 for i in range(dim) if not(i in axis_bnd) ]

        return body

    def _visit_GlobalQuadratureInterface(self, expr):
        body  = []
        body += self._visit(expr.minus)
        body += self._visit(expr.plus)

        return body

    # ....................................................
    #           GlobalBasis
    # ....................................................
    def _visit_GlobalBasis(self, expr):
        # sympy does not like ':'
        _slice = Slice(None,None)

        element       = expr.element
        is_test_basis = expr.is_test
        ln            = expr.ln
        dim           = expr.dim

        # basis attributs
        indices_span = expr.indices_span
        spans        = expr.spans
        basis_in_elm = expr.basis_in_elm
        basis        = expr.basis

        # element attributs
        indices_elm  = element.indices
        axis_bnd     = element.axis_bnd

        body = []

        if is_test_basis:
            body += [Assign(indices_span[i*ln+j],
                            spans[i*ln+j][indices_elm[i]])
                     for i,j in np.ndindex(dim, ln) if not(i in axis_bnd)]

        body += [Assign(basis_in_elm[i*ln+j],
                        basis[i*ln+j][indices_elm[i],_slice,_slice,_slice])
                 for i,j in np.ndindex(dim,ln) if not(i in axis_bnd) ]

        return body

#==============================================================================
def generate_declarations(expr, **settings):
    return DeclarationGenerator(settings).doit(expr)
