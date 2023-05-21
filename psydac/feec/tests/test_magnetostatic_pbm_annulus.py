import numpy as np
import pytest
import logging

from psydac.core.bsplines          import make_knots
from psydac.fem.basic              import FemField
from psydac.fem.splines            import SplineSpace
from psydac.fem.tensor             import TensorFemSpace
from psydac.feec.derivatives       import VectorCurl_2D, Divergence_2D
from psydac.feec.global_projectors import Projector_H1, Projector_L2
from psydac.feec.global_projectors import projection_matrix_H1_homogeneous_bc, projection_matrix_Hdiv_homogeneous_bc 
from psydac.ddm.cart               import DomainDecomposition


import numpy as np
import sympy
from typing import Tuple

from sympde.topology  import Derham, Square, IdentityMapping, PolarMapping
from sympde.topology.domain import Domain, Union, Connectivity

from psydac.feec.global_projectors import projection_matrix_Hdiv_homogeneous_bc, projection_matrix_H1_homogeneous_bc

from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.fem  import DiscreteBilinearForm, DiscreteLinearForm
from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.cad.geometry     import Geometry
from psydac.fem.basic import FemField
from psydac.fem.vector import VectorFemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.linalg.block import BlockVector
from psydac.linalg.utilities import array_to_psydac

from scipy.sparse._lil import lil_matrix
from scipy.sparse._coo import coo_matrix

from sympde.calculus      import grad, dot
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr.equation import find, EssentialBC
import sympde.topology as top
from sympde.utilities.utils import plot_domain

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy

from psydac.cad.geometry          import Geometry
from psydac.core.bsplines         import quadrature_grid
from psydac.fem.basic             import FemField
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace
from psydac.linalg.kron           import KroneckerLinearSolver
from psydac.linalg.block          import BlockDiagonalSolver
from psydac.utilities.quadratures import gauss_legendre

from sympde.topology.domain       import Domain

from scipy.sparse import bmat
from scipy.sparse._lil import lil_matrix
from scipy.sparse.linalg import eigs, spsolve
from scipy.sparse.linalg import inv

from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_annulus

def test_magnetostatic_pbm_homogeneous():
    f = sympy.Tuple(1e-10, 1e-10)
    B = solve_magnetostatic_pbm_annulus(f)
    assert isinstance(B, np.ndarray)
    assert np.linalg.norm(B) < 1e-6


if __name__ == '__main__':
    logging.basicConfig(filename='mydebug.log', level=logging.DEBUG, filemode='w')
    test_magnetostatic_pbm_homogeneous()