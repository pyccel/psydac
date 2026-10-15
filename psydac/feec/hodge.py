#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
import numpy as np

from sympde.topology import elements_of
from sympde.topology.space import ScalarFunction
from sympde.calculus import dot
from sympde.expr.expr import BilinearForm
from sympde.expr.expr import integral

from psydac.api.settings import PSYDAC_BACKENDS

# ===============================================================================
class HodgeOperator:
    """
    Change of basis operator: dual basis -> primal basis

        self._linop: matrix (LinearOperator) of the primal Hodge = this is the mass matrix !
        self.dual_linop: this is the INVERSE mass matrix (LinearOperator)

    Parameters
    ----------
    Vh: <FemSpace>
     The discrete space

    domain_h: <Geometry>
     The discrete domain of the projector

    metric : <str>
     the metric of the de Rham complex

    backend_language: <str>
     The backend used to accelerate the code

    Notes
    -----
     We only support the identity metric, this implies that the dual Hodge is the inverse of the primal one.
     # todo: allow for non-identity metrics
    """

    def __init__(self, Vh, domain_h, metric='identity', backend_language='python'):

        self._fem_domain = Vh
        self._fem_codomain = Vh

        # FemLinearOperators
        self._primal_hodge = None
        self._dual_hodge = None

        # LinearOperators
        self._linop = None
        self._dual_linop = None

        self._domain_h = domain_h
        self._backend_language = backend_language

        if not (metric == 'identity'):
            raise NotImplementedError('only the identity metric is available')

        self._metric = metric

    def assemble_matrix(self):
        """
        the Hodge matrix is the patch-wise multi-patch mass matrix
        it is not stored by default but assembled on demand
        """
        from psydac.api.discretization import discretize
        from psydac.fem.basic          import FemLinearOperator

        if self._linop is None:
            Vh = self._fem_domain
            assert Vh == self._fem_codomain

            V = Vh.symbolic_space
            domain = V.domain
            u, v = elements_of(V, names='u, v')

            if isinstance(u, ScalarFunction):
                expr = u * v
            else:
                expr = dot(u, v)

            a = BilinearForm((u, v), integral(domain, expr))
            ah = discretize(a, self._domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[self._backend_language])

            self._linop = ah.assemble()  # Mass matrix in stencil format

            self._primal_hodge = FemLinearOperator(self._fem_domain, self._fem_codomain, linop=self._linop)

    def assemble_dual_matrix(self, solver ='cg', **kwargs):
        """
        the dual Hodge matrix is the patch-wise inverse of the multi-patch mass matrix
        it is not stored by default but computed on demand, by approximate local (patch-wise) inversion of the mass matrix
        """
        from psydac.linalg.solvers import inverse
        from psydac.linalg.block   import BlockLinearOperator
        from psydac.fem.basic      import FemLinearOperator
        
        if self._dual_linop is None:
            if not self._linop:
                self.assemble_matrix()

            M = self._linop  # mass matrix of the (primal) basis

            if self._fem_domain.is_multipatch:

                nrows = M.n_block_rows
                ncols = M.n_block_cols

                inv_M_blocks = [list(b) for b in M.blocks]
                for i in range(nrows):
                    Mii = M[i, i]
                    inv_Mii = inverse(Mii, solver=solver, **kwargs)
                    inv_M_blocks[i][i] = inv_Mii

                self._dual_linop = BlockLinearOperator(M.codomain, M.domain, blocks=inv_M_blocks)
                self._dual_hodge = FemLinearOperator(self._fem_codomain, self._fem_domain, linop=self._dual_linop)
            
            else:
                inv_M = inverse(M, solver=solver, **kwargs)
                self._dual_hodge = FemLinearOperator(self._fem_codomain, self._fem_domain, linop=self._dual_linop)
            
    @property
    def linop(self):
        if self._linop is None:
            self.assemble_matrix()

        return self._linop

    @property
    def dual_linop(self):
        if self._dual_linop is None:
            self.assemble_dual_matrix()

        return self._dual_linop

    @property
    def hodge(self):
        if self._linop is None:
            self.assemble_matrix()

        return self._primal_hodge

    @property
    def dual_hodge(self):
        if self._dual_linop is None:
            self.assemble_dual_matrix()

        return self._dual_hodge
