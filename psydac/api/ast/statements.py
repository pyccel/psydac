
import numpy as np
from sympy import symbols, Symbol

from pyccel.ast.core import IndexedVariable
from pyccel.ast.core import Assign
from pyccel.ast import DottedName
from pyccel.ast.core import Slice

from .utilities import variables
from .declarations import generate_declarations

#==============================================================================
class Block(object):

    def __init__(self, decs, stmts):
        # TODO make attributs as tuple not list
        self._decs  = decs
        self._stmts = stmts

    @property
    def decs(self):
        return self._decs

    @property
    def stmts(self):
        return self._stmts

    def __add__(self, other):
        decs  = list(self.decs)  + list(other.decs)
        stmts = list(self.stmts) + list(other.stmts)
        return Block(decs, stmts)

#==============================================================================
class StatementGenerator(object):

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
        # create declarations associated to the assembly node
        decs  = generate_declarations(expr)
        stmts = []

        return Block(decs, stmts)

    # ....................................................
    #           Grid
    # ....................................................
    def _visit_Grid(self, expr):
        # create declarations associated to a grid
        decs  = generate_declarations(expr)
        stmts = []

        return Block(decs, stmts)

    def _visit_GridInterface(self, expr):
        minus = self._visit(expr.minus)
        plus  = self._visit(expr.plus)

        return minus + plus

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
        # create declarations associated to a quadrature
        decs = generate_declarations(expr)
        stmts = []

        return Block(decs, stmts)

    def _visit_GlobalQuadratureInterface(self, expr):
        minus = self._visit(expr.minus)
        plus  = self._visit(expr.plus)

        return minus + plus

    # ....................................................
    #           GlobalBasis
    # ....................................................
    def _visit_GlobalBasis(self, expr):
        # create declarations associated to a basis
        decs = generate_declarations(expr)
        stmts = []

        return Block(decs, stmts)

#==============================================================================
def generate_statements(expr, **settings):
    return StatementGenerator(settings).doit(expr)
