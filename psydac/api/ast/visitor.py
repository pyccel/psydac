
from sympy import symbols, Symbol

from pyccel.ast.core import IndexedVariable
from pyccel.ast.core import Assign
from pyccel.ast import DottedName

from .utilities import variables
from .nodes import Grid

class PsydacGenerator(object):

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
        raise NotImplementedError('TODO')

    def _visit_Grid(self, expr):
        grid = expr.args[0]
        body = []
        for k,v in expr.attributs.items():
            body += [Assign(v, DottedName(grid, expr.correspondance[k]))]

        return body

    def _visit_GridInterface(self, expr):
        body  = []
        body += self._visit(expr.minus)
        body += self._visit(expr.plus)

        return body


def psydac_visitor(expr, **settings):
    return PsydacGenerator(settings).doit(expr)
