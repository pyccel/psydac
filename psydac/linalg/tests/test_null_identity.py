import pytest

import numpy as np

from psydac.ddm.cart       import DomainDecomposition, CartDecomposition
from psydac.linalg.stencil import StencilVectorSpace, StencilVector

from psydac.linalg.identity import IdentityLinearOperator, IdentityMatrix, IdentityStencilMatrix
from psydac.linalg.null import NullLinearOperator, NullMatrix, NullStencilMatrix

#===============================================================================
def compute_global_starts_ends(domain_decomposition, npts):
    ndims         = len(npts)
    global_starts = [None]*ndims
    global_ends   = [None]*ndims

    for axis in range(ndims):
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    return tuple(global_starts), tuple(global_ends)

#===============================================================================
@pytest.mark.parametrize( 'seed', [0, 1, 2] )
def test_null_identity(seed):
    np.random.seed(seed)

    # for now, having these fixed and a serial vector space should be enough for these tests
    vnpts = [100, 97]
    vpads = [10, 7]
    vperiodic = [True,False]

    # Create domain decomposition
    D = DomainDecomposition(vnpts, periods=vperiodic)

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, vnpts)

    vcart = CartDecomposition(D, vnpts, global_starts, global_ends, pads=vpads, shifts=[1,1])

    V = StencilVectorSpace( vcart )

    wnpts = [120, 39]
    wpads = [10, 7] # take the same pads for the StencilMatrix constructor to be happy
    wperiodic = [True, False]

    # Create domain decomposition
    D = DomainDecomposition(wnpts, periods=wperiodic)

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, wnpts)

    wcart = CartDecomposition(D, wnpts, global_starts, global_ends, pads=wpads, shifts=[1,1])

    W = StencilVectorSpace( wcart )

    v = V.zeros()
    outv = V.zeros()
    outw = W.zeros()

    vdataslice = tuple(slice(p, -p) for p in V.pads)
    wdataslice = tuple(slice(p, -p) for p in W.pads)
    rawdata = np.random.random(v._data[vdataslice].shape)
    vnulldata = np.zeros(outv._data[vdataslice].shape)
    wnulldata = np.zeros(outw._data[wdataslice].shape)

    def testall(ops, test, run):
        for op in ops:
            v._data[vdataslice] = rawdata
            test(op, run)
            outv._data[vdataslice] = 0.0
            outw._data[wdataslice] = 0.0
    
    run1 = lambda op: op.dot(v)
    run2v = lambda op: op.dot(v, out=outv)
    run2w = lambda op: op.dot(v, out=outw)
    run2i = lambda op: op.dot(v, out=v)

    def testid(op, run):
        np.array_equiv(run(op)._data[vdataslice], rawdata[vdataslice])
    
    def testnull(op, run):
        np.array_equiv(run(op)._data[vdataslice], wnulldata[wdataslice])

    def testnullself(op, run):
        np.array_equiv(run(op)._data[vdataslice], vnulldata[vdataslice])

    idop = [IdentityLinearOperator(V), IdentityMatrix(V), IdentityStencilMatrix(V)]
    nullop = [NullLinearOperator(V, W), NullMatrix(V, W), NullStencilMatrix(V, W)]
    nullselfop = [NullLinearOperator(V, V), NullMatrix(V, V), NullStencilMatrix(V, V)]


    testall(idop, testid, run1)
    testall(idop, testid, run2v)
    testall(idop, testid, run2i)

    testall(nullop, testnull, run1)
    testall(nullop, testnull, run2w)

    testall(nullselfop, testnullself, run1)
    testall(nullselfop, testnullself, run2v)
    testall(nullselfop, testnullself, run2i)
