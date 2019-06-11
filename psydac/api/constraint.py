
from sympde.topology import Boundary as sym_Boundary
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.dense   import DenseVectorSpace ,DenseVector,DenseMatrix
from psydac.linalg.block   import BlockVector, BlockMatrix, BlockLinearOperator
from psydac.linalg.block   import ProductSpace
from psydac.linalg.basic   import LinearOperator, Vector, VectorSpace

#===============================================================================
def apply_constraint_1d(test_space, trial_space, index, cs, a):

    if isinstance(a, Vector):
        n   = len(cs)
        V   = DenseVectorSpace(n)
        if isinstance(test_space, ProductSpace):
            spaces = test_space.spaces
            test_space = ProductSpace(*spaces, V)
            blocks     = list(a.blocks) + [DenseVector(V)]
        else:
            test_space = ProductSpace(test_space, V)
            blocks     = [a, DenseVector(V)]

        return BlockVector(test_space, blocks=blocks)
            
    elif isinstance(a, LinearOperator):
        n   = len(cs)
        V   = DenseVectorSpace(n)
        
        if isinstance(test_space, ProductSpace):
            M   = DenseMatrix(V, test_space[index])
            spaces = test_space.spaces
            test_space = ProductSpace(*spaces, V)
        else:
            M   = DenseMatrix(V, test_space)
            test_space = ProductSpace(test_space, V)
            
        if isinstance(trial_space, ProductSpace):
            M_T = DenseMatrix(trial_space[index], V)
            spaces     = trial_space.spaces
            trial_space = ProductSpace(*spaces, V)

        else:
            M_T = DenseMatrix(trial_space, V)
            trial_space = ProductSpace(trial_space, V)
        
        for i  in range(n):
            c = cs[i].assemble()
            M._data[i][:]   = c[:] 
            M_T._data[i][:] = c[:]
        
        if isinstance(a, BlockLinearOperator):
            blocks  = [list(b)+[None] for b in a.blokcs]
            blocks += [None for b in blocks[0]]
        else:
            blocks = [[a, None], [None, None]]
            
        blocks[index][-1] = M
        blocks[-1][index] = M_T

        return BlockMatrix(trial_space, test_space, blocks=blocks)
    else:
        raise ValueError('only LinearOperator and Vectors are available')

#===============================================================================                
def apply_constraint_2d(test_space, trial_space, index, cs, a):

    if isinstance(a, Vector):
        n   = len(cs)
        V   = DenseVectorSpace(n)
        if isinstance(test_space, ProductSpace):
            spaces = test_space.spaces
            test_space = ProductSpace(*spaces, V)
            blocks     = list(a.blocks) + [DenseVector(V)]
            

        else:
            test_space = ProductSpace(test_space, V)
            blocks     = [a, DenseVector(V)]

        return BlockVector(test_space, blocks=blocks)
            
    elif isinstance(a, LinearOperator):
        n   = len(cs)
        V   = DenseVectorSpace(n)
        
        if isinstance(test_space, ProductSpace):
            M   = DenseMatrix(V, test_space[index])
            spaces = test_space.spaces
            test_space = ProductSpace(*spaces, V)
        else:
            M   = DenseMatrix(V, test_space)
            test_space = ProductSpace(test_space, V)
            
        if isinstance(trial_space, ProductSpace):
            M_T = DenseMatrix(trial_space[index], V)
            spaces     = trial_space.spaces
            trial_space = ProductSpace(*spaces, V)

        else:
            M_T = DenseMatrix(trial_space, V)
            trial_space = ProductSpace(trial_space, V)
        
        for i  in range(n):
            c = cs[i].assemble()
            M._data[i][:,:]   = c[:,:] 
            M_T._data[i][:,:] = c[:,:]
        
        if isinstance(a, BlockLinearOperator):
            blocks  = [list(b)+[None] for b in a.blokcs]
            blocks += [None for b in blocks[0]]
        else:
            blocks = [[a, None], [None, None]]
            
        blocks[index][-1] = M
        blocks[-1][index] = M_T

        return BlockMatrix(trial_space, test_space, blocks=blocks)
    else:
        raise ValueError('only LinearOperator and Vectors are available')    

#===============================================================================    
def apply_constraint_3d(test_space, trial_space, index, cs, a):

    if isinstance(a, Vector):
        n   = len(cs)
        V   = DenseVectorSpace(n)
        if isinstance(test_space, ProductSpace):
            spaces = test_space.spaces
            test_space = ProductSpace(*spaces, V)
            blocks     = list(a.blocks) + [DenseVector(V)]
        else:
            test_space = ProductSpace(test_space, V)
            blocks     = [a, DenseVector(V)]

        return BlockVector(test_space, blocks=blocks)
            
    elif isinstance(a, LinearOperator):
        n   = len(cs)
        V   = DenseVectorSpace(n)
        
        if isinstance(test_space, ProductSpace):
            M   = DenseMatrix(V, test_space[index])
            spaces = test_space.spaces
            test_space = ProductSpace(*spaces, V)
        else:
            M   = DenseMatrix(V, test_space)
            test_space = ProductSpace(test_space, V)
            
        if isinstance(trial_space, ProductSpace):
            M_T = DenseMatrix(trial_space[index], V)
            spaces     = trial_space.spaces
            trial_space = ProductSpace(*spaces, V)

        else:
            M_T = DenseMatrix(trial_space, V)
            trial_space = ProductSpace(trial_space, V)
        
        for i  in range(n):
            c = cs[i].assemble()
            M._data[i][:,:,:]   = c[i][:,:,:] 
            M_T._data[i][:,:,:] = c[i][:,:,:]
        
        if isinstance(a, BlockLinearOperator):
            blocks  = [list(b)+[None] for b in a.blokcs]
            blocks += [None for b in blocks[0]]
        else:
            blocks = [[a, None], [None, None]]
            
        blocks[index][-1] = M
        blocks[-1][index] = M_T

        return BlockMatrix(trial_space, test_space, blocks=blocks)
    else:
        raise ValueError('only LinearOperator and Vectors are available')

#==============================================================================
#==============================================================================   
#==============================================================================

def apply_constraint(test_space, trial_space, cs, a, **kwargs):

    _avail_classes = [Vector, LinearOperator]
    
    classes = type(a).__mro__
    classes = set(classes) & set(_avail_classes)
    classes = list(classes)
    if not classes:
        raise TypeError('> wrong argument type {}'.format(type(a)))

    cls = classes[0]


    pattern = 'apply_constraint_{dim}d'
    apply_cs = pattern.format( dim = test_space.ldim, name = cls.__name__)

    apply_cs = eval(apply_cs)

    indices = set(list(zip(*cs))[0])
    cs = [[c[1] for c in cs if c[0]==i] for i in indices]
    
    for index, c in zip(indices, cs):
        a = apply_cs(test_space.vector_space, trial_space.vector_space, index, c, a, **kwargs)

    return a
     
