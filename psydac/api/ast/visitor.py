
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
        dim = expr.dim
        grid = expr.args[0]

        body = []

        names = ['n_elements{j}'.format(j=j) for j in range(1,dim+1)]
        n_elements = variables(names, 'int')

        names = ['points{j}'.format(j=j) for j in range(1,dim+1)]
        points  = variables(names,  dtype='real', rank=2, cls=IndexedVariable)

        names = ['weights{j}'.format(j=j) for j in range(1,dim+1)]
        weights = variables(names, dtype='real', rank=2, cls=IndexedVariable)

        names = ['k{j}'.format(j=j) for j in range(1,dim+1)]
        quad_orders = variables(names, dtype='int')

        names = ['element_s{j}'.format(j=j) for j in range(1,dim+1)]
        element_starts = variables(names, 'int')

        names = ['element_e{j}'.format(j=j) for j in range(1,dim+1)]
        element_ends   = variables(names, 'int')

        body += [Assign(n_elements,     DottedName(grid, 'n_elements'))]
        body += [Assign(points,         DottedName(grid, 'points'))]
        body += [Assign(weights,        DottedName(grid, 'weights'))]
        body += [Assign(quad_orders,    DottedName(grid, 'quad_order'))]
        body += [Assign(element_starts, DottedName(grid, 'local_element_start'))]
        body += [Assign(element_ends,   DottedName(grid, 'local_element_end'))]

        return body

    def _visit_GridInterface(self, expr):
        # ...
        def _names(pattern, label, dim):
            return [pattern.format(label=label, j=j) for j in range(1,dim+1)]
        # ...

        dim = expr.dim
        grids = expr.args
        labels = ('minus', 'plus')

        body = []
        for label, grid in zip(labels, grids):
            pattern = 'n_elements{j}_{label}'
            n_elements = variables(_names(pattern, label, dim), 'int')

            pattern = 'points{j}_{label}'
            points  = variables(_names(pattern, label, dim),
                                dtype='real', rank=2, cls=IndexedVariable)

            pattern = 'weights{j}_{label}'
            weights = variables(_names(pattern, label, dim),
                                dtype='real', rank=2, cls=IndexedVariable)

            pattern = 'k{j}_{label}'
            quad_orders = variables(_names(pattern, label, dim), dtype='int')

            pattern = 'element_s{j}_{label}'
            element_starts = variables(_names(pattern, label, dim), 'int')

            pattern = 'element_e{j}_{label}'
            element_ends   = variables(_names(pattern, label, dim), 'int')

            body += [Assign(n_elements,     DottedName(grid, 'n_elements'))]
            body += [Assign(points,         DottedName(grid, 'points'))]
            body += [Assign(weights,        DottedName(grid, 'weights'))]
            body += [Assign(quad_orders,    DottedName(grid, 'quad_order'))]
            body += [Assign(element_starts, DottedName(grid, 'local_element_start'))]
            body += [Assign(element_ends,   DottedName(grid, 'local_element_end'))]

        return body


def psydac_visitor(expr, **settings):
    return PsydacGenerator(settings).doit(expr)
