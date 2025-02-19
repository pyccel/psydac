import numpy as np 
import pytest
from sympde.topology import TorusMapping, TargetMapping, PolarMapping
from psydac.ddm.cart                 import DomainDecomposition
from psydac.mapping.discrete         import NurbsMapping
from psydac.fem.splines              import SplineSpace
from psydac.fem.tensor               import TensorFemSpace

mapping1 = TorusMapping('T_1',R0=10.)
mapping2 = TargetMapping('T_2', c1=1., k=2., D=3., c2=4.)
mapping3 = PolarMapping('P_1', c1=1., c2=2., rmin=3., rmax=4.)

@pytest.mark.parametrize('mapping', [mapping1, mapping2, mapping3])
def test_function_test_evaluate(mapping):
    ldim = mapping.ldim 
    list_int = []
    list_float = []
    list_1d_array = []
    
    for i in range(ldim):
        list_int += [2 + i]
        list_float += [2. + float(i)]
        list_1d_array  += [np.linspace(float(i), float(i+10), 100)]
        
    list_meshgrid = np.meshgrid(*list_1d_array)
    
    print(list_int)
    print(list_float)
    print(len(list_1d_array))
    print(len(list_meshgrid))
    
    out_int = mapping(*list_int) 
    out_float = mapping(*list_float)   
    out_1d_arrays = mapping(*list_1d_array)
    out_meshgrid = mapping(*list_meshgrid)
    
    print(out_int)
    print(out_float)
    print(len(out_1d_arrays))
    print(len(out_meshgrid))
    
    assert len(out_int) == mapping.pdim 
    assert len(out_float) == mapping.pdim 
    assert len(out_1d_arrays) == mapping.pdim 
    assert len(out_meshgrid) == mapping.pdim
    
  
    for arr in out_1d_arrays:
        print(arr.shape)
        assert arr.shape == list_1d_array[0].shape
    
    
    for arr in out_meshgrid:
        print(arr.shape)
        assert arr.shape == list_meshgrid[0].shape 
    
if __name__ == '__main__':
    
    def quart_circle( rmin=0.5, rmax=1.0, center=None ):

        degrees = (2, 1)
        pdim    = 2

        knots = [[0.0 , 0.0 , 0.0 , 1.0 , 1.0 , 1.0],
                    [0.0 , 0.0 , 1.0 , 1.0] ]

        points          = np.zeros((3,2,pdim))
        j = 0
        points[0,j,:]   = [0.0   , -rmin]
        points[1,j,:]   = [-rmin , -rmin]
        points[2,j,:]   = [-rmin , 0.0  ]
        j = 1
        points[0,j,:]   = [0.0   , -rmax]
        points[1,j,:]   = [-rmax , -rmax]
        points[2,j,:]   = [-rmax , 0.0  ]

        if center is not None:
            points[...,0] += center[0]
            points[...,1] += center[1]

        weights         = np.zeros((3,2))
        j = 0
        weights[0,j]   = 1.0
        weights[1,j]   = 0.707106781187
        weights[2,j]   = 1.0
        j = 1
        weights[0,j]   = 1.0
        weights[1,j]   = 0.707106781187
        weights[2,j]   = 1.0

        for i in range(pdim):
            points[...,i] *= weights[...]

        return degrees, knots, points, weights

    degrees, knots, points, weights = quart_circle( rmin=0.5, rmax=1.0, center=None )

    # Create tensor spline space, distributed
    spaces = [SplineSpace( knots=k, degree=p ) for k,p in zip(knots, degrees)]

    ncells   = [len(space.breaks)-1 for space in spaces]
    domain_decomposition = DomainDecomposition(ncells=ncells, periods=[False]*2, comm=None)

    space = TensorFemSpace( domain_decomposition, *spaces )

    mapping = NurbsMapping.from_control_points_weights( space, points, weights )
    
    eta = (0.7, np.pi/3)
    