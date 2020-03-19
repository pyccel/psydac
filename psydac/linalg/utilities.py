# coding: utf-8

from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from psydac.linalg.block   import BlockVector, ProductSpace

import numpy as np

__all__ = ['array_to_stencil']

def array_to_stencil(x, Xh):
    """ converts a numpy array to StencilVector or BlockVector format"""

    if isinstance(Xh, ProductSpace):
        u = BlockVector(Xh)
        starts = [np.array(V.starts) for V in Xh.spaces]
        ends   = [np.array(V.ends)   for V in Xh.spaces]

        for i in range(len(starts)):
            g = tuple(slice(s,e+1) for s,e in zip(starts[i], ends[i]))
            shape = tuple(ends[i]-starts[i]+1)
            u[i][g] = x[:np.product(shape)].reshape(shape)
            x       = x[np.product(shape):]

    elif isinstance(Xh, StencilVectorSpace):

        u =  StencilVector(Xh)
        starts = np.array(Xh.starts)
        ends   = np.array(Xh.ends)
        g = tuple(slice(s, e+1) for s,e in zip(starts, ends))
        shape = tuple(ends-starts+1)
        u[g] = x[:np.product(shape)].reshape(shape)
    else:
        raise ValueError('Xh must be a StencilVectorSpace or a ProductSpace')

    return u
