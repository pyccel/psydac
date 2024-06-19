import numpy as np 
import pytest
import analytical_mappings

mapping1 = analytical_mappings.TorusMapping('T_1',R0=10.)
mapping2 = analytical_mappings.TargetMapping('T_2', c1=1., k=2., D=3., c2=4.)
mapping3 = analytical_mappings.PolarMapping('P_1', c1=1., c2=2., rmin=3., rmax=4.)

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
    test_function_test_evaluate(mapping1)
    