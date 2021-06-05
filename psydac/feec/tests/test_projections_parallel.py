import numpy as np
import pytest
from mpi4py import MPI

from psydac.linalg.block           import BlockVector
from psydac.linalg.stencil         import StencilVector
from psydac.core.bsplines          import make_knots
from psydac.fem.basic              import FemField
from psydac.fem.splines            import SplineSpace
from psydac.fem.tensor             import TensorFemSpace
from psydac.fem.vector             import ProductFemSpace
from psydac.feec.global_projectors import Projector_H1, Projector_L2, Projector_Hcurl, Projector_Hdiv

def run_projection_comparison(domain, ncells, degree, periodic, funcs, reduce):
    # find the appropriate reduced space (without invoking the code generation machinery)
    if len(domain) == 1:
        if reduce == 0:
            opV = lambda V0: V0
            opP = Projector_H1
        else:
            opV = lambda V0: V0.reduce_degree(axes=[0], basis='M')
            opP = Projector_L2
    elif len(domain) == 2:
        if reduce == 0:
            opV = lambda V0: V0
            opP = Projector_H1
        elif reduce == 1:
            opV = lambda V0: ProductFemSpace(V0.reduce_degree(axes=[0], basis='M'),
                                        V0.reduce_degree(axes=[1], basis='M'))
            opP = Projector_Hcurl
        elif reduce == 2:
            # (note: this would be more instructive, if the index was 1 as well...)
            opV = lambda V0: ProductFemSpace(V0.reduce_degree(axes=[1], basis='M'),
                                        V0.reduce_degree(axes=[0], basis='M'))
            opP = Projector_Hdiv
        else:
            opV = lambda V0: V0.reduce_degree(axes=[0,1], basis='M')
            opP = Projector_L2
    elif len(domain) == 3:
        if reduce == 0:
            opV = lambda V0: V0
            opP = Projector_H1
        elif reduce == 1:
            opV = lambda V0: ProductFemSpace(V0.reduce_degree(axes=[0], basis='M'),
                                        V0.reduce_degree(axes=[1], basis='M'),
                                        V0.reduce_degree(axes=[2], basis='M'))
            opP = Projector_Hcurl
        elif reduce == 2:
            opV = lambda V0: ProductFemSpace(V0.reduce_degree(axes=[1,2], basis='M'),
                                        V0.reduce_degree(axes=[0,2], basis='M'),
                                        V0.reduce_degree(axes=[0,1], basis='M'))
            opP = Projector_Hdiv
        else:
            opV = lambda V0: V0.reduce_degree(axes=[0,1,2], basis='M')
            opP = Projector_L2
    
    # build basic SplineSpaces
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    Ns = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]
    
    # build TensorFemSpaces in serial and parallel
    V0p = TensorFemSpace(*Ns, comm=MPI.COMM_WORLD)
    V0s = TensorFemSpace(*Ns)

    # build reduced spaces in serial and parallel
    Vp = opV(V0p)
    Vs = opV(V0s)

    # build projectors in serial and parallel
    Pp = opP(Vp)
    Ps = opP(Vs)

    # project in serial and parallel
    resp = Pp(funcs).coeffs
    ress = Ps(funcs).coeffs

    # block vector decomposition
    if isinstance(resp, BlockVector):
        blockp = resp.blocks
        blocks = ress.blocks
    elif isinstance(resp, StencilVector):
        blockp = [resp]
        blocks = [ress]
    
    for p, s in zip(blockp, blocks):
        # build data slices in serial and parallel
        slicep = tuple(slice(pad, -pad) for pad in p.space.pads)
        slices = tuple(slice(pad, -pad) for pad in s.space.pads)

        # look for the chunk which is on the local parallel process
        subslice = tuple(slice(s,e+1) for s,e in zip(p.space.starts, p.space.ends))

        # compare
        assert np.allclose(p._data[slicep], s._data[slices][subslice])

#==============================================================================
@pytest.mark.parametrize('domain', [(0, 1)])
@pytest.mark.parametrize('ncells', [13, 37])
@pytest.mark.parametrize('degree', [3])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('funcs', [np.sin, np.exp])
@pytest.mark.parametrize('reduce', [0,1])
@pytest.mark.parallel
def test_projection_parallel_1d(domain, ncells, degree, periodic, funcs, reduce):
    run_projection_comparison([domain], [ncells], [degree], [periodic], [funcs], reduce)

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])              
@pytest.mark.parametrize('ncells', [(27, 15)])              
@pytest.mark.parametrize('degree', [(4, 5)])                 
@pytest.mark.parametrize('periodic', [(True, False), (False, True)])
@pytest.mark.parametrize('funcs', [[np.sin, np.cos], [np.exp, np.exp]])
@pytest.mark.parametrize('reduce', [0,1,2,3])
@pytest.mark.parallel
def test_projection_parallel_2d(domain, ncells, degree, periodic, funcs, reduce):
    run_projection_comparison(domain, ncells, degree, periodic, funcs, reduce) 

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])  
@pytest.mark.parametrize('ncells', [(5, 5, 7)])                       
@pytest.mark.parametrize('degree', [(2, 2, 3)])            
@pytest.mark.parametrize('periodic', [( True, False, False),          
                                      (False,  True, False),
                                      (False, False,  True)])
@pytest.mark.parametrize('funcs', [[np.sin, np.cos, np.sin], [np.exp, np.exp, np.exp]])
@pytest.mark.parametrize('reduce', [0,1,2,3])
@pytest.mark.parallel
def test_projection_parallel_3d(domain, ncells, degree, periodic, funcs, reduce):
    run_projection_comparison(domain, ncells, degree, periodic, funcs, reduce) 
