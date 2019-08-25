
import numpy as np
from sympy import symbols, Symbol

from pyccel.ast.core import IndexedVariable
from pyccel.ast.core import Assign
from pyccel.ast import DottedName
from pyccel.ast.core import Slice

from .utilities import variables
from .nodes import Grid

#==============================================================================
class StatementsGenerator(object):

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
        indices_elm    = element.indices_elm
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
        indices_elm  = element.indices_elm
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
def generate_statements(expr, **settings):
    return StatementsGenerator(settings).doit(expr)
