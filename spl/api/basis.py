# coding: utf-8


from spl.utilities.quadratures import gauss_legendre
from spl.core.bsplines         import ( find_span,
                                        basis_funs,
                                        collocation_matrix,
                                        breakpoints,
                                        greville,
                                        elements_spans,
                                        make_knots,
                                        quadrature_grid,
                                        basis_ders_on_quad_grid )

from spl.fem.splines           import SplineSpace
from spl.fem.tensor            import TensorFemSpace
from spl.fem.vector            import ProductFemSpace
from spl.api.grid              import QuadratureGrid

#==============================================================================
def _eval_basis_SplineSpace( V, points, nderiv ):
    """
    """
    assert( isinstance( V, SplineSpace ) )

    T    = V.knots       # knots sequence
    p    = V.degree      # spline degree

    return basis_ders_on_quad_grid( T, p, points, nderiv )

#==============================================================================
def _eval_basis_TensorFemSpace( V, points, nderiv ):
    """
    """
    if isinstance( nderiv, int ):
        nderiv = [nderiv for d in range(V.ldim)]

    return [_eval_basis_SplineSpace( W, x, n )
            for (W, x, n) in zip(V.spaces, points, nderiv)]

#==============================================================================
# TODO must take the max of degrees if quad_order is not present and
# spaces.degrees are different
def _eval_basis_ProductFemSpace( V, points, nderiv ):
    """
    """
    raise NotImplementedError()

#==============================================================================
def eval_basis( V, points, nderiv ):
    """
    """
    _avail_classes = [SplineSpace, TensorFemSpace, ProductFemSpace]

    classes = type(V).__mro__
    classes = set(classes) & set(_avail_classes)
    classes = list(classes)
    if not classes:
        raise TypeError('> wrong argument type {}'.format(type(V)))

    cls = classes[0]

    pattern = '_eval_basis_{name}'
    func = pattern.format( name = cls.__name__ )

    func = eval(func)
    values = func(V, points, nderiv)
    if isinstance( V, SplineSpace ):
        values = [values]

    return values

#==============================================================================
class BasisValues():
    def __init__( self, V, grid, nderiv ):
        assert( isinstance( grid, QuadratureGrid ) )

        points = grid.points
        if isinstance( V, SplineSpace ):
            points = points[0]

        self._basis = eval_basis( V, points, nderiv )

    @property
    def basis(self):
        return self._basis


################################################
if __name__ == '__main__':
    from numpy import linspace

    # ...
    p=(2,2)
    ne=(2**1,2**1)

    # ... discrete spaces
    # Input data: degree, number of elements
    p1,p2 = p
    ne1,ne2 = ne

    # Create uniform grid
    grid_1 = linspace( 0., 1., num=ne1+1 )
    grid_2 = linspace( 0., 1., num=ne2+1 )

    # Create 1D finite element spaces and precompute quadrature data
    V1 = SplineSpace( p1, grid=grid_1 )
    V2 = SplineSpace( p2, grid=grid_2 )

    # Create 2D tensor product finite element space
    V = TensorFemSpace( V1, V2 )
    # ...

    # ...
    print('=== 1D case ===')
    qd_1d = QuadratureGrid(V1)
    values_1d = BasisValues(V1, qd_1d, nderiv=1)

    print('> basis  = ', values_1d.basis)
    # ...

    # ...
    print('=== 2D case ===')
    qd_2d = QuadratureGrid(V)

    values_2d = BasisValues(V, qd_2d, nderiv=1)

    print('> basis  = ', values_2d.basis)
    # ...
