# coding: utf-8

# Conga operators on piecewise (broken) de Rham sequences

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import eye as sparse_id

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
from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator

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

        if I.axis == 1:
            self._A[0,0][0,0][:,e11,0,0] = 1/2
        else:
            self._A[0,0][1,1][:,e12,0,0] = 1/2

        if I.axis == 1:
            self._A[1,1][0,0][:,s21,0,0] = 1/2
        else:
            self._A[1,1][1,1][:,s22,0,0] = 1/2


        if I.axis == 1:
            self._A[0,1][0,0][:,d11,0,-d21] = 1/2
        else:
            self._A[0,1][1,1][:,d12,0,-d22] = 1/2

        if I.axis == 1:
            self._A[1,0][0,0][:,s21,0, d11] = 1/2
        else:
            self._A[1,0][1,1][:,s22,0, d12] = 1/2

        self._matrix = self._A


#===============================================================================
class BrokenMass( FemLinearOperator ):
    """
    Broken mass matrix for a scalar space (seen as a LinearOperator... to be improved)
    # TODO: (MCP 10.03.2021) define them as Hodge FemLinearOperators

    """
    def __init__( self, Vh, domain_h, is_scalar):

        FemLinearOperator.__init__(self, fem_domain=Vh)

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


#==============================================================================
class BrokenGradient_2D(FemLinearOperator):

    def __init__(self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V0h, fem_codomain=V1h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix for i, D0i in enumerate(D0s)})

    def transpose(self):
        # todo (MCP): define as the dual differential operator
        return BrokenTransposedGradient_2D(self.fem_domain, self.fem_codomain)

#==============================================================================
class BrokenTransposedGradient_2D( FemLinearOperator ):

    def __init__( self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V0h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix.T for i, D0i in enumerate(D0s)})

    def transpose(self):
        # todo (MCP): discard
        return BrokenGradient_2D(self.fem_codomain, self.fem_domain)


#==============================================================================
class BrokenScalarCurl_2D(FemLinearOperator):
    def __init__(self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V2h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D1i._matrix for i, D1i in enumerate(D1s)})

    def transpose(self):
        return BrokenTransposedScalarCurl_2D(V1h=self.fem_domain, V2h=self.fem_codomain)


#==============================================================================
class BrokenTransposedScalarCurl_2D( FemLinearOperator ):

    def __init__( self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V2h, fem_codomain=V1h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D1i._matrix.T for i, D1i in enumerate(D1s)})

    def transpose(self):
        return BrokenScalarCurl_2D(V1h=self.fem_codomain, V2h=self.fem_domain)



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

def get_grid_vals_scalar(u, etas, mappings_obj):
    # get the physical field values, given the logical field and the logical grid
    n_patches = len(mappings_obj)
    # works but less general...
    # u_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(us, etas)]
    u_vals = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        u_vals[k] = np.empty_like(eta_1)
        if isinstance(u,FemField):
            uk_field = u.fields[k]   # todo (MCP): try with u[k].fields?
        else:
            # then field is just callable
            uk_field = u[k]
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                u_vals[k][i, j] = push_2d_h1(uk_field, x1i, x2j)

    u_vals  = np.concatenate(u_vals, axis=1)

    return u_vals


def get_grid_vals_vector(E, etas, mappings_obj):
    # get the physical field values, given the logical field and logical grid
    n_patches = len(mappings_obj)
    E_x_vals = n_patches*[None]
    E_y_vals = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        E_x_vals[k] = np.empty_like(eta_1)
        E_y_vals[k] = np.empty_like(eta_1)
        if isinstance(E,FemField):
            Ek_field_0 = E[k].fields[0]   # or E.fields[k][0] ?
            Ek_field_1 = E[k].fields[1]
        else:
            # then E field is just callable
            Ek_field_0 = E[k][0]
            Ek_field_1 = E[k][1]
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                E_x_vals[k][i, j], E_y_vals[k][i, j] = \
                    push_2d_hcurl(Ek_field_0, Ek_field_1, x1i, x2j, mappings_obj[k])
    E_x_vals = np.concatenate(E_x_vals, axis=1)
    E_y_vals = np.concatenate(E_y_vals, axis=1)
    return E_x_vals, E_y_vals


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
