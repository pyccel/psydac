from mpi4py import MPI
import numpy as np
import pytest

from psydac.fem.splines                import SplineSpace
from psydac.fem.tensor                 import TensorFemSpace
from psydac.mapping.analytical_gallery import Annulus
from psydac.mapping.discrete           import SplineMapping
from psydac.linalg.stencil             import StencilVector, StencilMatrix

from psydac.polar.c1_projections import C1Projector

#==============================================================================
@pytest.mark.parametrize( 'degrees', [(2,2),(2,3),(3,2),(3,3)] )
@pytest.mark.parametrize( 'ncells' , [(4,5),(6,6),(11,14)] )

def test_c1_projections( degrees, ncells, verbose=False ):

    if verbose:
        np.set_printoptions( precision=2, linewidth=200 )

    #--------------------------------------------
    # Setup
    #--------------------------------------------

    # Geometry: logical domain and mapping
    lims_1       = (0, 1)
    lims_2       = (0, 2*np.pi)
    period_1     = False
    period_2     = True
    map_analytic = Annulus( rmin=0, rmax=1 )

    # Discretization: number of elements and polynomial degree
    ne1, ne2 = ncells
    p1 , p2  = degrees

    #--------------------------------------------
    # Spline space and C1 projector
    #--------------------------------------------

    # Uniform grid in logical space
    grid_1 = np.linspace( *lims_1, num=ne1+1 )
    grid_2 = np.linspace( *lims_2, num=ne2+1 )

    # 1D finite element spaces
    V1 = SplineSpace( p1, grid=grid_1, periodic=period_1 )
    V2 = SplineSpace( p2, grid=grid_2, periodic=period_2 )

    # 2D tensor-product space
    V = TensorFemSpace( V1, V2, comm=MPI.COMM_WORLD )

    # Spline mapping
    map_discrete = SplineMapping.from_mapping( V, map_analytic )

    # C1 projector
    proj = C1Projector( map_discrete )

    #--------------------------------------------
    # Linear algebra objects
    #--------------------------------------------

    # Matrix and vector in tensor-product basis
    A = StencilMatrix( V.vector_space, V.vector_space )
    b = StencilVector( V.vector_space )

    # Set values of matrix
    A[:, :, 0, 0] =  4
    A[:, :, 0,-1] = -1
    A[:, :, 0,+1] = -1
    A[:, :,-1, 0] = -2
    A[:, :,+1, 0] = -2

    # Add (symmetric) random perturbation to matrix
    s1, s2 = V.vector_space.starts
    e1, e2 = V.vector_space.ends
    n1, n2 = A.domain.npts
    perturbation = 0.1 * np.random.random( (e1-s1+1, e2-s2+1, p1, p2) )
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            for k1 in range(1,p1):
                for k2 in range(1,p2):
                    j1 = (i1+k1)%n1
                    j2 = (i2+k2)%n2
                    eps = perturbation[i1,i2,k1,k2]
                    A[i1,i2, k1, k2] += eps
                    A[j1,j2,-k1,-k2] += eps

    A.remove_spurious_entries()

    if verbose:
        print( "A:" )
        print( A.toarray() )
        print()

    # Set values of vector
    s1, s2 = b.starts
    e1, e2 = b.ends
    b[s1:e1+1, s2:e2+1] = np.random.random( (e1-s1+1, e2-s2+1) )
    b.update_ghost_regions()

    if verbose:
        print( "b:" )
        print( b.toarray().reshape( b.space.npts ) )
        print()

    #--------------------------------------------
    # Test all methods of C1 projector
    #--------------------------------------------

    # Compute A' = E^T A E
    # Compute b' = E b
    Ap = proj.change_matrix_basis( A )
    bp = proj.change_rhs_basis   ( b )

    # Compute (E^T A E) b' = A' b'
    Ap_bp = Ap.dot( bp )

    # Compute E^T (A (E b')) = A' b'
    E_bp      = proj.convert_to_tensor_basis( bp )
    A_E_bp    = A.dot( E_bp )
    Et_A_E_bp = proj.change_rhs_basis( A_E_bp )

    if verbose:
        print( "(E^T A E) b' :" )
        print( Ap_bp[0].toarray() )
        print( Ap_bp[1].toarray().reshape( Ap_bp[1].space.npts ) )
        print()
        print( "E^T (A (E b')) :" )
        print( Et_A_E_bp[0].toarray() )
        print( Et_A_E_bp[1].toarray().reshape( Et_A_E_bp[1].space.npts ) )
        print()

    # Verity that two results are identical
    kwargs = {'rtol': 1e-10, 'atol': 1e-10}
    assert np.allclose( Ap_bp[0].toarray(), Et_A_E_bp[0].toarray(), **kwargs )
    assert np.allclose( Ap_bp[1].toarray(), Et_A_E_bp[1].toarray(), **kwargs )

    if verbose:
        print( "PASSED" )

    return locals()

#==============================================================================
if __name__ == "__main__":
    namespace = test_c1_projections( degrees=(2,2), ncells=(4,5), verbose=True )
