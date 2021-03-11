
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from sympde.topology import element_of, elements_of
from sympde.calculus import grad, dot, inner, rot, div
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.api.discretization import discretize
from psydac.linalg.basic import LinearOperator
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix
from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.direct_solvers import SparseSolver
from psydac.fem.basic   import FemField
from psydac.fem.vector import ProductFemSpace, VectorFemSpace

from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl, Projector_L2
from psydac.feec.derivatives import Gradient_2D, ScalarCurl_2D

from psydac.feec.derivatives import DiffOperator

#===============================================================================
class FemLinearOperator( LinearOperator ):
    """
    Linear operator, with an additional Fem layer
    """

    def __init__( self, fem_domain=None, fem_codomain=None, matrix=None):
        assert fem_domain
        self._fem_domain   = fem_domain
        if fem_codomain:
            self._fem_codomain = fem_codomain
        else:
            self._fem_codomain = fem_domain
        self._domain   = self._fem_domain.vector_space
        self._codomain = self._fem_codomain.vector_space

        self._matrix = matrix

    @property
    def domain( self ):
        # if self._domain is None:
        #     return self._fem_domain.vector_space
        # else:
        return self._domain

    @property
    def codomain( self ):
        # if self._codomain is None:
        #     return self._fem_codomain.vector_space
        # else:
        return self._codomain

    @property
    def fem_domain( self ):
        return self._fem_domain

    @property
    def fem_codomain( self ):
        return self._fem_codomain

    @property
    def matrix( self ):
        return self._matrix

    @property
    def T(self):
        return self.transpose()

    # ...
    def transpose(self):
        raise NotImplementedError('Class does not provide a transpose() method')

    # ...
    def __call__( self, f ):
        if self._matrix:
            coeffs = self._matrix.dot(f.coeffs)
            return FemField(self.fem_codomain, coeffs=coeffs)
        else:
            raise NotImplementedError('Class does not provide a __call__ method without a matrix')

    # ...
    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        if self._matrix:
            f = FemField(self.fem_domain, coeffs=f_coeffs)
            return self(f).coeffs
        else:
            raise NotImplementedError('Class does not provide a dot method without a matrix')



    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    #
    # @abstractmethod
    # def dot( self, v, out=None ):
    #     pass

#===============================================================================
class ConformingProjection_V0( FemLinearOperator ):
    """
    Conforming projection from global broken space to conforming global space
    Defined by averaging of interface dofs
    """
    def __init__(self, V0h, domain_h):

        FemLinearOperator.__init__(self, fem_domain=V0h)

        V0             = V0h.symbolic_space
        domain         = V0.domain

        u, v = elements_of(V0, names='u, v')
        expr   = u*v  # dot(u,v)

        I      = domain.interfaces  # note: interfaces does not include the boundary
        expr_I = ( plus(u)-minus(u) )*( plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(I, expr_I))

        ah = discretize(a, domain_h, [V0h, V0h])

        self._A = ah.assemble()

        spaces = self._A.domain.spaces
        sp1    = spaces[0]
        sp2    = spaces[1]

        s1 = sp1.starts[I.axis]
        e1 = sp1.ends[I.axis]

        s2 = sp2.starts[I.axis]
        e2 = sp2.ends[I.axis]

        d1     = V0h.spaces[0].degree[I.axis]
        d2     = V0h.spaces[1].degree[I.axis]

        self._A[0,0][:,:,:,:] = 0
        self._A[1,1][:,:,:,:] = 0
        self._A[0,1][:,:,:,:] = 0
        self._A[1,0][:,:,:,:] = 0

        self._A[0,0][:,:,0,0]  = 1
        self._A[1,1][:,:,0,0]  = 1

        self._A[0,0][:,e1,0,0] = 1/2
        self._A[1,1][:,s2,0,0] = 1/2

        self._A[0,1][:,d1,0,-d2] = 1/2
        self._A[1,0][:,s2,0, d1] = 1/2

        self._matrix = self._A

    # # ...
    # def __call__( self, f ):
    #
    #     coeffs = self._A.dot(f.coeffs)
    #     return FemField(self.fem_codomain, coeffs=coeffs)
    #
    # # ...
    # def dot( self, f_coeffs, out=None ):
    #     # coeffs layer
    #     f = FemField(self.fem_domain, coeffs=f_coeffs)
    #     return self(f).coeffs


#===============================================================================
class ConformingProjection_V1( FemLinearOperator ):
    """
    Conforming projection from global broken space to conforming global space

    proj.dot(v) returns the conforming projection of v, computed by solving linear system


    """
    def __init__(self, V1h, domain_h):

        FemLinearOperator.__init__(self, fem_domain=V1h)

        V1             = V1h.symbolic_space
        domain         = V1.domain

        u, v = elements_of(V1, names='u, v')
        expr   = dot(u,v)

        I      = domain.interfaces  # note: interfaces does not include the boundary
        expr_I = dot( plus(u)-minus(u) , plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(I, expr_I))

        ah = discretize(a, domain_h, [V1h, V1h])

        self._A = ah.assemble()

        for b1 in self._A.blocks:
            for b2 in b1:
                if b2 is None:continue
                for b3 in b2.blocks:
                    for A in b3:
                        if A is None:continue
                        A[:,:,:,:] = 0

        self._A[0,0][0,0][:,:,0,0] = 1
        self._A[0,0][1,1][:,:,0,0] = 1

        self._A[1,1][0,0][:,:,0,0] = 1
        self._A[1,1][1,1][:,:,0,0] = 1

        spaces = self._A.domain.spaces
        sp1    = spaces[0]
        sp2    = spaces[1]

        s11 = sp1.spaces[0].starts[I.axis]
        e11 = sp1.spaces[0].ends[I.axis]
        s12 = sp1.spaces[1].starts[I.axis]
        e12 = sp1.spaces[1].ends[I.axis]

        s21 = sp2.spaces[0].starts[I.axis]
        e21 = sp2.spaces[0].ends[I.axis]
        s22 = sp2.spaces[1].starts[I.axis]
        e22 = sp2.spaces[1].ends[I.axis]

        d11     = V1h.spaces[0].spaces[0].degree[I.axis]
        d12     = V1h.spaces[0].spaces[1].degree[I.axis]

        d21     = V1h.spaces[1].spaces[0].degree[I.axis]
        d22     = V1h.spaces[1].spaces[1].degree[I.axis]

        self._A[0,0][0,0][:,e11,0,0] = 1/2
        self._A[0,0][1,1][:,e12,0,0] = 1/2

        self._A[1,1][0,0][:,s21,0,0] = 1/2
        self._A[1,1][1,1][:,s22,0,0] = 1/2


        self._A[0,1][0,0][:,d11,0,-d21] = 1/2
        self._A[0,1][1,1][:,d12,0,-d22] = 1/2

        self._A[1,0][0,0][:,s21,0, d11] = 1/2
        self._A[1,0][1,1][:,s22,0, d12] = 1/2

        self._matrix = self._A

    # # ...
    # def __call__( self, f ):
    #
    #     coeffs = self._A.dot(f.coeffs)
    #     return FemField(self.fem_codomain, coeffs=coeffs)
    #
    # # ...
    # def dot( self, f_coeffs, out=None ):
    #     # coeffs layer
    #     f = FemField(self.fem_domain, coeffs=f_coeffs)
    #     return self(f).coeffs

class DummyConformingProjection_V1( FemLinearOperator ):

    def __init__(self, V1h, domain_h):

        FemLinearOperator.__init__(self, fem_domain=V1h)

        V1             = V1h.symbolic_space
        domain         = V1.domain

        u, v = elements_of(V1, names='u, v')
        expr   = dot(u,v)

        dummy_p = BilinearForm((u,v), integral(domain, expr))

        dummy_ph = discretize(dummy_p, domain_h, [V1h, V1h])

        self._matrix = dummy_ph.assemble()


    # # ...
    # def __call__( self, f ):
    #
    #     coeffs = self._P.dot(f.coeffs)
    #     return FemField(self.fem_codomain, coeffs=coeffs)
    #
    # # ...
    # def dot( self, f_coeffs, out=None ):
    #     # coeffs layer
    #     f = FemField(self.fem_domain, coeffs=f_coeffs)
    #     return self(f).coeffs


#===============================================================================
class BrokenMass( FemLinearOperator ):
    """
    Broken mass matrix for a scalar space (seen as a LinearOperator... to be improved)
    """
    def __init__( self, Vh, domain_h, is_scalar):

        FemLinearOperator.__init__(self, fem_domain=Vh)

        print( "type(Vh) = ", type(Vh) )
        V = Vh.symbolic_space
        domain = V.domain
        # domain_h = V0h.domain  # would be nice
        u, v = elements_of(V, names='u, v')
        if is_scalar:
            expr   = u*v
        else:
            expr   = dot(u,v)
        a = BilinearForm((u,v), integral(domain, expr))
        ah = discretize(a, domain_h, [Vh, Vh])
        self._matrix = ah.assemble() #.toarray()

    # def mat(self):
    #     return self._M
    #
    # def __call__( self, f ):
    #     # Fem layer
    #     Mf_coeffs = self.dot(f.coeffs)
    #     return FemField(self.fem_domain, coeffs=Mf_coeffs)
    #
    # def dot( self, f_coeffs, out=None ):
    #     # coeffs layer
    #     return self._M.dot(f_coeffs)


#==============================================================================
class ComposedLinearOperator( FemLinearOperator ):

    def __init__( self, B, A ):
        assert isinstance(A, FemLinearOperator)
        assert isinstance(B, FemLinearOperator)
        assert B.fem_domain == A.fem_codomain
        FemLinearOperator.__init__(
            self, fem_domain=A.fem_domain, fem_codomain=B.fem_codomain
        )
        if A._matrix and B._matrix:
            print("In ComposedLinearOperator:")
            print( "type(A._matrix) = ", type(A._matrix) )
            print( "type(B._matrix) = ", type(B._matrix) )
            print( A._matrix.shape )
            self._matrix = B._matrix * A._matrix
            print( "type(self._matrix) = ", type(self._matrix) )
            print( self._matrix.shape )

        else:
            self._matrix = None
            self._A = A
            self._B = B

    def __call__( self, f ):
        if self._matrix:
            return FemLinearOperator.__call__(self, f)
        else:
            return self._B(self._A(f))

    def dot( self, f_coeffs, out=None ):
        if self._matrix:
            return FemLinearOperator.dot(self, f_coeffs, out)
        else:
            return self._B.dot(self._A.dot(f_coeffs))

#==============================================================================
class IdLinearOperator( FemLinearOperator ):

    def __init__( self, V ):
        FemLinearOperator.__init__(self, fem_domain=V)

    def __call__( self, f ):
        # fem layer
        return f

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        return f_coeffs

#==============================================================================
class SumLinearOperator( FemLinearOperator ):

    def __init__( self, B, A ):
        assert isinstance(A, FemLinearOperator)
        assert isinstance(B, FemLinearOperator)
        assert B.fem_domain == A.fem_domain
        assert B.fem_codomain == A.fem_codomain
        FemLinearOperator.__init__(
            self, fem_domain=A.fem_domain, fem_codomain=A.fem_codomain
        )
        self._A = A
        self._B = B

    def __call__( self, f ):
        # fem layer
        return  self._B(f) + self._A(f)

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        return  self._B.dot(f_coeffs) + self._A.dot(f_coeffs)

#==============================================================================
class MultLinearOperator( FemLinearOperator ):

    def __init__( self, c, A ):
        assert isinstance(A, FemLinearOperator)
        FemLinearOperator.__init__(
            self, fem_domain=A.fem_domain, fem_codomain=A.fem_codomain
        )
        self._A = A
        self._c = c

    def __call__( self, f ):
        # fem layer
        return self._c * self._A(f)

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        return self._c * self._A.dot(f_coeffs)

#==============================================================================
class BrokenGradient_2D(FemLinearOperator):

    def __init__(self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V0h, fem_codomain=V1h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix for i, D0i in enumerate(D0s)})

    def dot(self, u0_coeffs, out = None):
        return self._matrix.dot(u0_coeffs, out=out)

    def __call__(self, u0):
        return FemField(self.fem_codomain, coeffs = self.dot(u0.coeffs))

    def transpose(self):
        return BrokenTransposedGradient_2D(self.fem_domain, self.fem_codomain)

#==============================================================================
class BrokenTransposedGradient_2D( FemLinearOperator ):

    def __init__( self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V0h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix.T for i, D0i in enumerate(D0s)})

    def dot(self, u0_coeffs, out = None):
        return self._matrix.dot(u0_coeffs, out=out)

    def __call__(self, u0):
        return FemField(self.fem_codomain, coeffs = self.dot(u0.coeffs))

    def transpose(self):
        return BrokenGradient_2D(self.fem_codomain, self.fem_domain)


#==============================================================================
class BrokenScalarCurl_2D(FemLinearOperator):
    def __init__(self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V2h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D1i._matrix for i, D1i in enumerate(D1s)})

    def dot(self, E1_coeffs, out = None):
        return self._matrix.dot(E1_coeffs, out=out)

    def __call__(self, E1):
        return FemField(self.fem_codomain, coeffs = self.dot(E1.coeffs))

    def transpose(self):
        raise NotImplementedError
        # return BrokenTransposedGradient_2D(self.fem_domain, self.fem_codomain)


#==============================================================================
from sympy import Tuple

# def multipatch_Moments_Hcurl(f, V1h, domain_h):
def ortho_proj_Hcurl(EE, V1h, domain_h, M1):
    """
    return orthogonal projection of E on V1h, given M1 the mass matrix
    """
    assert isinstance(EE, Tuple)
    V1 = V1h.symbolic_space
    v = element_of(V1, name='v')
    # x,y = V1.domain.coordinates
    # EE = Tuple(2*x, 2*y)
    # print("in op:", type(EE))
    l = LinearForm(v, integral(V1.domain, dot(v,EE)))
    lh = discretize(l, domain_h, V1h)
    b = lh.assemble()
    sol_coeffs, info = pcg(M1.mat(), b, pc="jacobi", tol=1e-10)

    return FemField(V1h, coeffs=sol_coeffs)

#==============================================================================
class Multipatch_Projector_H1:
    """
    to apply the H1 projection (2D) on every patch
    """
    def __init__(self, V0h):

        self._P0s = [Projector_H1(V) for V in V0h.spaces]
        self._V0h  = V0h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        u0s = [P(fun) for P, fun, in zip(self._P0s, funs_log)]

        u0_coeffs = BlockVector(self._V0h.vector_space, \
                blocks = [u0j.coeffs for u0j in u0s])

        return FemField(self._V0h, coeffs = u0_coeffs)

#==============================================================================
class Multipatch_Projector_Hcurl:

    """
    to apply the Hcurl projection (2D) on every patch
    """
    def __init__(self, V1h, nquads=None):

        self._P1s = [Projector_Hcurl(V, nquads=nquads) for V in V1h.spaces]
        self._V1h  = V1h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        E1s = [P(fun) for P, fun, in zip(self._P1s, funs_log)]

        E1_coeffs = BlockVector(self._V1h.vector_space, \
                blocks = [E1j.coeffs for E1j in E1s])

        return FemField(self._V1h, coeffs = E1_coeffs)

#==============================================================================
class Multipatch_Projector_L2:

    """
    to apply the L2 projection (2D) on every patch
    """
    def __init__(self, V2h, nquads=None):

        self._P2s = [Projector_L2(V, nquads=nquads) for V in V2h.spaces]
        self._V2h  = V2h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        B2s = [P(fun) for P, fun, in zip(self._P2s, funs_log)]

        B2_coeffs = BlockVector(self._V2h.vector_space, \
                blocks = [B2j.coeffs for B2j in B2s])

        return FemField(self._V2h, coeffs = B2_coeffs)


#==============================================================================
# some plotting utilities

def get_scalar_patch_fields(u, V0h):
    # todo: discard now?
    return [FemField(V, coeffs=c) for V, c in zip(V0h.spaces, u.coeffs)]

def get_vector_patch_fields(E, V1h):
    # todo: discard now?
    return [FemField(V, coeffs=c) for V, c in zip(V1h.spaces, E.coeffs)]

from psydac.feec.pull_push     import push_2d_h1, push_2d_hcurl, push_2d_l2

def get_grid_vals_V0(u, V0h, etas, mappings_obj):
    # get the physical field values, given the logical fem field and the logical grid
    # us = get_scalar_patch_fields(u, V0h)   # todo: u.fields should also work
    n_patches = len(mappings_obj)
    # works but less general...
    # u_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(us, etas)]
    u_vals = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        u_vals[k] = np.empty_like(eta_1)
        # todo: don't pass V0h but check type
        if V0h is None:
            # then field is just callable
            uk_field = u[k]
        else:
            # then field is a fem field
            uk_field = u.fields[k]
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                u_vals[k][i, j] = push_2d_h1(uk_field, x1i, x2j)

    u_vals  = np.concatenate(u_vals, axis=1)

    return u_vals


def get_grid_vals_V1(E, V1h, etas, mappings_obj):
    # get the physical field values, given the logical field and logical grid
    # Es = get_vector_patch_fields(E, V1h)  # todo: try with E[k].fields
    n_patches = len(mappings_obj)
    E_x_vals = n_patches*[None]
    E_y_vals = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        E_x_vals[k] = np.empty_like(eta_1)
        E_y_vals[k] = np.empty_like(eta_1)
        # todo: don't pass V1h but check type
        if V1h is None:
            # then E field is just callable
            Ek_field_0 = E[k][0]
            Ek_field_1 = E[k][1]
        else:
            # then E is a fem field
            Ek_field_0 = E[k].fields[0]   # or E.fields[k][0] ?
            Ek_field_1 = E[k].fields[1]
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                E_x_vals[k][i, j], E_y_vals[k][i, j] = \
                    push_2d_hcurl(Ek_field_0, Ek_field_1, x1i, x2j, mappings_obj[k])
    E_x_vals = np.concatenate(E_x_vals, axis=1)
    E_y_vals = np.concatenate(E_y_vals, axis=1)
    return E_x_vals, E_y_vals

def get_grid_vals_V2(B, V2h, etas, mappings_obj):
    # get the physical field values, given the logical fem field and the logical grid
    #Bs = get_scalar_patch_fields(B, V2h)  # todo: B.fields should also work
    n_patches = len(mappings_obj)
    B_vals = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        B_vals[k] = np.empty_like(eta_1)
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                B_vals[k][i, j] = \
                    push_2d_l2(Bs[k], x1i, x2j, mappings_obj[k])

    B_vals  = np.concatenate(B_vals, axis=1)
    return B_vals


def my_small_plot(
        title, vals, titles,
        xx, yy,
        gridlines_x1=None,
        gridlines_x2=None,
):

    n_plots = len(vals)
    assert n_plots == len(titles)
    #fig = plt.figure(figsize=(17., 4.8))
    fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
    fig.suptitle(title, fontsize=14)

    for np in range(n_plots):
        ax = fig.add_subplot(1, n_plots, np+1)

        if gridlines_x1 is not None:
            ax.plot(*gridlines_x1, color='k')
            ax.plot(*gridlines_x2, color='k')

        cp = ax.contourf(xx, yy, vals[np], 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( titles[np] )

    plt.show()
