import numpy as np
from copy import deepcopy
from scipy.sparse import eye as sparse_eye

from sympde.topology import Boundary, Interface, Union
from psydac.core.bsplines import breakpoints
from psydac.utilities.quadratures import gauss_legendre
from psydac.linalg.basic        import IdentityOperator
from psydac.feec.multipatch.operators import get_patch_index_from_face
from psydac.linalg.basic        import IdentityOperator
from psydac.linalg.basic        import LinearOperator, Vector
from psydac.linalg.utilities    import array_to_psydac


####  projection from valentin:



# def get_patch_index_from_face(domain, face):
#     """ Return the patch index of subdomain/boundary

#     Parameters
#     ----------
#     domain : <Sympde.topology.Domain>
#      The Symbolic domain

#     face : <Sympde.topology.BasicDomain>
#      A patch or a boundary of a patch

#     Returns
#     -------
#     i : <int>
#      The index of a subdomain/boundary in the multipatch domain
#     """

#     if domain.mapping:
#         domain = domain.logical_domain
#     if face.mapping:
#         face = face.logical_domain

#     domains = domain.interior.args
#     if isinstance(face, Interface):
#         raise NotImplementedError(
#             "This face is an interface, it has several indices -- I am a machine, I cannot choose. Help.")
#     elif isinstance(face, Boundary):
#         i = domains.index(face.domain)
#     else:
#         i = domains.index(face)
#     return i


## TODO: 
#   decide whether to use this loca2global function, 
#   or the Local2GlobalIndexMap class from psydac/feec/multipatch/bilinear_form_scipy.py
#   (which is probably faster)

def loca2global(multi_index, n_patches, single_patch_shapes):
    """ Convert the local multi index to the global index in the flattened array

    Parameters
    ----------
    multi_index : <tuple|list>
     The multidimentional index is of the form [patch_index, component_index, array_index]
     or [patch_index, array_index] if the number of components is one

    n_patches: int
     The total number of patches

    single_patch_shapes: a list of tuples or a tuple
     It contains the shapes of the multidimentional arrays in a single patch

    Returns
    -------
     I : int
      The global index in the flattened array.

    Examples
    --------
    loca2global([0,0,1],2,[100,100])
    >>> 10000

    loca2global([0,1,0,1],2,[[80,80],[100,100]])
    >>> 6401
    """
    import numpy as np
    if isinstance(single_patch_shapes[0],(int, np.int64)):
        patch_index = multi_index[0]
        ii = multi_index[1:]
        Ip = np.ravel_multi_index(ii, dims=single_patch_shapes, order='C')
        single_patch_size = np.product(single_patch_shapes)
        I = np.ravel_multi_index((patch_index, Ip), dims=(n_patches, single_patch_size), order='C')
    else:
        patch_index = multi_index[0]
        com_index   = multi_index[1]
        ii = multi_index[2:]
        Ipc = np.ravel_multi_index(ii, dims=single_patch_shapes[com_index], order='C')
        sizes = [np.product(s) for s in single_patch_shapes]
        Ip = sum(sizes[:com_index]) + Ipc
        I = np.ravel_multi_index((patch_index, Ip), dims=(n_patches, sum(sizes)), order='C')

    return I


#---------------------------------------------------#

#-#-# CONFORMING PROJECTIONS #-#-#

# TODO: document these conf projections (how many moments are preserved ?)


def smooth_x(V0h,nquads):
    dim_tot = V0h.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    V0      = V0h.symbolic_space
    domain  = V0.domain
    Interfaces  = domain.interfaces
    n_patches = len(V0h.spaces)

    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]


    patch_space = V0h.spaces[0]
    local_shape = [patch_space.spaces[0].nbasis,patch_space.spaces[1].nbasis]
    p      = patch_space.degree                     # spline degrees
    Nel    = patch_space.ncells                     # number of elements
    degree = patch_space.degree
    breakpoints_x = breakpoints(patch_space.knots[0],degree[0])
    breakpoints_y = breakpoints(patch_space.knots[1],degree[1])

    grid = [np.array([deepcopy((0.5*(u[0]+1)*(breakpoints_x[i+1]-breakpoints_x[i])+breakpoints_x[i])) for i in range(Nel[0])]),
            np.array([deepcopy((0.5*(u[1]+1)*(breakpoints_y[i+1]-breakpoints_y[i])+breakpoints_y[i])) for i in range(Nel[1])])]
    _, basis, span, _ = patch_space.preprocess_regular_tensor_grid(grid,der=1)
    
    span = [deepcopy(span[k] + patch_space.vector_space.starts[k] - patch_space.vector_space.shifts[k] * patch_space.vector_space.pads[k]) for k in range(2)]
    px=degree[0]
    enddom = breakpoints_x[-1]
    begdom = breakpoints_x[0]
    denom = enddom-begdom
    #Direction x, interface on the left
    if local_shape[0] < px+3:
        raise ValueError("patch space is too small to preserve enough moments")
        # print(px+3, local_shape[0])
    Mass_mat = np.zeros((px+2,local_shape[0]))
    for poldeg in range(px+2):
        for ie1 in range(Nel[0]):   #loop on cells
            for il1 in range(degree[0]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[0]): #loops on quadrature points
                    v0 = basis[0][ie1,il1,0,q1]
                    x  = grid[0][ie1,q1]
                    val += w[0][q1]*v0*((enddom-x)/denom)**poldeg
                locind=span[0][ie1]-degree[0]+il1
                Mass_mat[poldeg,locind]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:px+3]
    # print(Mass_mat.shape)
    # print(Mat_to_inv.shape)

    Correct_coef = np.linalg.solve(Mat_to_inv,Rhs)
    #Direction x, interface on the right

    for I in Interfaces:
        axis = I.axis
        i_minus = get_patch_index_from_face(domain, I.minus)
        i_plus  = get_patch_index_from_face(domain, I.plus )
        s_minus = V0h.spaces[i_minus]
        s_plus  = V0h.spaces[i_plus]
        n_deg_minus = s_minus.spaces[axis].nbasis
        patch_shape_minus = [s_minus.spaces[0].nbasis,s_minus.spaces[1].nbasis]
        patch_shape_plus  = [s_plus.spaces[0].nbasis,s_plus.spaces[1].nbasis]
        if axis == 0 :
            for i in range(s_plus.spaces[1].nbasis):
                indice_minus = loca2global([i_minus,n_deg_minus-1,i],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,0,i],n_patches,patch_shape_plus)
                Proj[indice_minus,indice_minus]-=1/2
                Proj[indice_plus,indice_plus]-=1/2
                Proj[indice_plus,indice_minus]+=1/2
                Proj[indice_minus,indice_plus]+=1/2
                for p in range(0,px+2):
                    #correction
                    indice_plus_i  = loca2global([i_plus, p+1,                  i], n_patches,patch_shape_plus)
                    indice_minus_i = loca2global([i_minus, n_deg_minus-1-(p+1), i], n_patches,patch_shape_minus)
                    indice_plus    = loca2global([i_plus, 0,                    i], n_patches,patch_shape_plus)
                    indice_minus   = loca2global([i_minus, n_deg_minus-1,       i], n_patches,patch_shape_minus)
                    Proj[indice_plus_i,indice_plus]+=Correct_coef[p]/2
                    Proj[indice_plus_i,indice_minus]-=Correct_coef[p]/2
                    Proj[indice_minus_i,indice_plus]-=Correct_coef[p]/2
                    Proj[indice_minus_i,indice_minus]+=Correct_coef[p]/2
    return Proj

def smooth_y(V0h,nquads):
    dim_tot = V0h.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    V0      = V0h.symbolic_space
    domain  = V0.domain
    Interfaces  = domain.interfaces
    n_patches = len(V0h.spaces)

    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]


    patch_space = V0h.spaces[0]
    local_shape = [patch_space.spaces[0].nbasis,patch_space.spaces[1].nbasis]
    p      = patch_space.degree                     # spline degrees
    Nel    = patch_space.ncells                     # number of elements
    degree = patch_space.degree
    breakpoints_x = breakpoints(patch_space.knots[0],degree[0])
    breakpoints_y = breakpoints(patch_space.knots[1],degree[1])

    grid = [np.array([deepcopy((0.5*(u[0]+1)*(breakpoints_x[i+1]-breakpoints_x[i])+breakpoints_x[i])) for i in range(Nel[0])]),
            np.array([deepcopy((0.5*(u[1]+1)*(breakpoints_y[i+1]-breakpoints_y[i])+breakpoints_y[i])) for i in range(Nel[1])])]
    _, basis, span, _ = patch_space.preprocess_regular_tensor_grid(grid,der=1)
    span = [deepcopy(span[k] + patch_space.vector_space.starts[k] - patch_space.vector_space.shifts[k] * patch_space.vector_space.pads[k]) for k in range(2)]
    py=degree[1]
    enddom = breakpoints_y[-1]
    begdom = breakpoints_y[0]
    denom = enddom-begdom
    if local_shape[1] < py+3:
        raise ValueError("patch space is too small to preserve enough moments")
    Mass_mat = np.zeros((py+2,local_shape[1]))
    for poldeg in range(py+2):
        for ie1 in range(Nel[1]):   #loop on cells
            for il1 in range(py+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis[1][ie1,il1,0,q1]
                    x  = grid[1][ie1,q1]
                    val += w[1][q1]*v0*((enddom-x)/denom)**poldeg
                locind=span[1][ie1]-degree[1]+il1
                Mass_mat[poldeg,locind]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:py+3]
    Correct_coef = np.linalg.solve(Mat_to_inv,Rhs)

    for I in Interfaces:
        axis = I.axis
        i_minus = get_patch_index_from_face(domain, I.minus)
        i_plus  = get_patch_index_from_face(domain, I.plus )
        s_minus = V0h.spaces[i_minus]
        s_plus  = V0h.spaces[i_plus]
        n_deg_minus = s_minus.spaces[axis].nbasis
        patch_shape_minus = [s_minus.spaces[0].nbasis,s_minus.spaces[1].nbasis]
        patch_shape_plus  = [s_plus.spaces[0].nbasis,s_plus.spaces[1].nbasis]

        if axis == 1 :
            for i in range(s_plus.spaces[1].nbasis):
                indice_minus = loca2global([i_minus,i,n_deg_minus-1],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,i,0],n_patches,patch_shape_plus)
                Proj[indice_minus,indice_minus]-=1/2
                Proj[indice_plus,indice_plus]-=1/2
                Proj[indice_plus,indice_minus]+=1/2
                Proj[indice_minus,indice_plus]+=1/2
                for p in range(0,py+2):
                    #correction
                    indice_plus_i  = loca2global([i_plus,  i, p+1                ], n_patches,patch_shape_plus)
                    indice_minus_i = loca2global([i_minus, i, n_deg_minus-1-(p+1)], n_patches,patch_shape_minus)
                    indice_plus    = loca2global([i_plus,  i, 0                  ], n_patches,patch_shape_plus)
                    indice_minus   = loca2global([i_minus, i, n_deg_minus-1      ], n_patches,patch_shape_minus)
                    Proj[indice_plus_i,indice_plus]+=Correct_coef[p]/2
                    Proj[indice_plus_i,indice_minus]-=Correct_coef[p]/2
                    Proj[indice_minus_i,indice_plus]-=Correct_coef[p]/2
                    Proj[indice_minus_i,indice_minus]+=Correct_coef[p]/2
    return Proj

def Conf_proj_0(V0h,nquads):
    n_patches = len(V0h.spaces)
    if n_patches ==1:
        dim_tot = V0h.nbasis
        Proj_op    = IdentityOperator(V0h.vector_space)#sparse_eye(dim_tot,format="lil")
    else : 
        S_x=smooth_x(V0h,nquads)
        S_y=smooth_y(V0h,nquads)
        Proj=S_x@S_y
        Proj_op = Operator_from_scipy(V0h.vector_space,V0h.vector_space,Proj)
    return Proj_op.tosparse()


def Conf_proj_1(V1h,nquads):

    dim_tot = V1h.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    V1      = V1h.symbolic_space
    domain  = V1.domain
    n_patches = len(V1h.spaces)

    if n_patches ==1:
        dim_tot = V1h.nbasis
        Proj    = sparse_eye(dim_tot,format="lil")
        return Proj

    Interfaces  = domain.interfaces

    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]


    patch_space = V1h.spaces[0]
    local_shape = [[patch_space.spaces[0].spaces[0].nbasis,patch_space.spaces[0].spaces[1].nbasis],[patch_space.spaces[1].spaces[0].nbasis,patch_space.spaces[1].spaces[1].nbasis]]
    p      = patch_space.degree                     # spline degrees
    Nel    = patch_space.ncells                     # number of elements
    patch_space_x = patch_space.spaces[0]
    patch_space_y = patch_space.spaces[1]
    degree_x = patch_space_x.degree
    degree_y = patch_space_y.degree
    breakpoints_x_x = breakpoints(patch_space_x.knots[0],degree_x[0])
    breakpoints_x_y = breakpoints(patch_space_x.knots[1],degree_x[1])

    grid = [np.array([deepcopy((0.5*(u[0]+1)*(breakpoints_x_x[i+1]-breakpoints_x_x[i])+breakpoints_x_x[i])) for i in range(Nel[0])]),
            np.array([deepcopy((0.5*(u[1]+1)*(breakpoints_x_y[i+1]-breakpoints_x_y[i])+breakpoints_x_y[i])) for i in range(Nel[1])])]
    _, basis_x, span_x, _ = patch_space_x.preprocess_regular_tensor_grid(grid,der=1)
    _, basis_y, span_y, _ = patch_space_y.preprocess_regular_tensor_grid(grid,der=1)
    span_x = [deepcopy(span_x[k] + patch_space_x.vector_space.starts[k] - patch_space_x.vector_space.shifts[k] * patch_space_x.vector_space.pads[k]) for k in range(2)]
    span_y = [deepcopy(span_y[k] + patch_space_y.vector_space.starts[k] - patch_space_y.vector_space.shifts[k] * patch_space_y.vector_space.pads[k]) for k in range(2)]
    px=degree_x[0]
    enddom = breakpoints_x_x[-1]
    begdom = breakpoints_x_x[0]
    denom = enddom-begdom
    #Direction x
    Mass_mat = np.zeros((px+1,local_shape[0][0]))
    for poldeg in range(px+1):
        for ie1 in range(Nel[0]):   #loop on cells
            for il1 in range(degree_x[0]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[0]): #loops on quadrature points
                    v0 = basis_x[0][ie1,il1,0,q1]
                    x  = grid[0][ie1,q1]
                    val += w[0][q1]*v0*((enddom-x)/denom)**poldeg
                locindx=span_x[0][ie1]-degree_x[0]+il1
                Mass_mat[poldeg,locindx]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:px+2]
    Correct_coef_x = np.linalg.solve(Mat_to_inv,Rhs)

    py=degree_y[1]
    enddom = breakpoints_x_y[-1]
    begdom = breakpoints_x_y[0]
    denom = enddom-begdom
    #Direction y
    Mass_mat = np.zeros((py+1,local_shape[1][1]))
    for poldeg in range(py+1):
        for ie1 in range(Nel[1]):   #loop on cells
            for il1 in range(degree_y[1]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis_y[1][ie1,il1,0,q1]
                    y  = grid[1][ie1,q1]
                    val += w[1][q1]*v0*((enddom-y)/denom)**poldeg
                locindy=span_y[1][ie1]-degree_y[1]+il1
                Mass_mat[poldeg,locindy]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:py+2]
    Correct_coef_y = np.linalg.solve(Mat_to_inv,Rhs)
    #Direction x, interface on the right
    """Mass_mat = np.zeros((px+1,local_shape[0][0]))
    for poldeg in range(px+1):
        for ie1 in range(Nel[0]):   #loop on cells
            for il1 in range(degree_x[0]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[0]): #loops on quadrature points
                    v0 = basis_x[0][ie1,il1,0,q1]
                    x  = grid_x[0][ie1,q1]
                    val += v0*((x-begdom)/denom)**poldeg
                locindx=span_x[0][ie1]-degree_x[0]+il1
                Mass_mat[poldeg,locindx]+=deepcopy(val)

    Rhs = Mass_mat[:,-1]
    Mat_to_inv = Mass_mat[:,-px-2:-1]
    print(Rhs)
    print(Mat_to_inv)

    Correct_coef_minus = np.linalg.solve(Mat_to_inv,Rhs)
    print(Correct_coef_minus)"""
    if V1.name=="Hdiv":
        for I in Interfaces:
            axis = I.axis
            i_minus = get_patch_index_from_face(domain, I.minus)
            i_plus  = get_patch_index_from_face(domain, I.plus )
            s_minus = V1h.spaces[i_minus]
            s_plus  = V1h.spaces[i_plus]
            n_deg_minus = s_minus.spaces[axis].spaces[axis].nbasis
            patch_shape_minus = [[s_minus.spaces[0].spaces[0].nbasis,s_minus.spaces[0].spaces[1].nbasis],
                                [s_minus.spaces[1].spaces[0].nbasis,s_minus.spaces[1].spaces[1].nbasis]]
            patch_shape_plus  = [[s_plus.spaces[0].spaces[0].nbasis,s_plus.spaces[0].spaces[1].nbasis],
                                [s_plus.spaces[1].spaces[0].nbasis,s_plus.spaces[1].spaces[1].nbasis]]
            if axis == 0 :
                for i in range(s_plus.spaces[0].spaces[1].nbasis):
                    indice_minus = loca2global([i_minus,0,n_deg_minus-1,i],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,0,0,i],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
                    for p in range(0,px+1):
                        #correction
                        indice_plus_i  = loca2global([i_plus,  0, p+1,                 i], n_patches,patch_shape_plus)
                        indice_minus_i = loca2global([i_minus, 0, n_deg_minus-1-(p+1), i], n_patches,patch_shape_minus)
                        indice_plus    = loca2global([i_plus,  0, 0,                   i], n_patches,patch_shape_plus)
                        indice_minus   = loca2global([i_minus, 0, n_deg_minus-1,       i], n_patches,patch_shape_minus)
                        Proj[indice_plus_i,indice_plus]+=Correct_coef_x[p]/2
                        Proj[indice_plus_i,indice_minus]-=Correct_coef_x[p]/2
                        Proj[indice_minus_i,indice_plus]-=Correct_coef_x[p]/2
                        Proj[indice_minus_i,indice_minus]+=Correct_coef_x[p]/2


            elif axis == 1 :
                for i in range(s_plus.spaces[1].spaces[0].nbasis):
                    indice_minus = loca2global([i_minus,1,i,n_deg_minus-1],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,1,i,0],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
                    for p in range(0,py+1):
                        #correction
                        indice_plus_i  = loca2global([i_plus,  1, i, p+1                ], n_patches,patch_shape_plus)
                        indice_minus_i = loca2global([i_minus, 1, i, n_deg_minus-1-(p+1)], n_patches,patch_shape_minus)
                        indice_plus    = loca2global([i_plus,  1, i, 0                  ], n_patches,patch_shape_plus)
                        indice_minus   = loca2global([i_minus, 1, i, n_deg_minus-1      ], n_patches,patch_shape_minus)
                        Proj[indice_plus_i,indice_plus]+=Correct_coef_y[p]/2
                        Proj[indice_plus_i,indice_minus]-=Correct_coef_y[p]/2
                        Proj[indice_minus_i,indice_plus]-=Correct_coef_y[p]/2
                        Proj[indice_minus_i,indice_minus]+=Correct_coef_y[p]/2
    elif V1.name=="Hcurl":
        for I in Interfaces:
            axis = I.axis
            i_minus = get_patch_index_from_face(domain, I.minus)
            i_plus  = get_patch_index_from_face(domain, I.plus )
            s_minus = V1h.spaces[i_minus]
            s_plus  = V1h.spaces[i_plus]
            naxis = (axis+1)%2
            n_deg_minus = s_minus.spaces[naxis].spaces[axis].nbasis
            patch_shape_minus = [[s_minus.spaces[0].spaces[0].nbasis,s_minus.spaces[0].spaces[1].nbasis],
                                [s_minus.spaces[1].spaces[0].nbasis,s_minus.spaces[1].spaces[1].nbasis]]
            patch_shape_plus  = [[s_plus.spaces[0].spaces[0].nbasis,s_plus.spaces[0].spaces[1].nbasis],
                                [s_plus.spaces[1].spaces[0].nbasis,s_plus.spaces[1].spaces[1].nbasis]]
            if axis == 0 :
                for i in range(s_plus.spaces[1].spaces[1].nbasis):
                    indice_minus = loca2global([i_minus,1,n_deg_minus-1,i],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,1,0,i],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
            elif axis == 1 :
                for i in range(s_plus.spaces[0].spaces[0].nbasis):
                    indice_minus = loca2global([i_minus,0,i,n_deg_minus-1],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,0,i,0],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
    else :
        print("Error in Conf_proj_1 : wrong kind of space")
    return Proj

class Operator_from_scipy(LinearOperator):

    def __init__(self, domain, codomain, A):
        self._matrix = A
        self._domain = domain
        self._codomain = codomain

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return None

    def toarray(self):
        raise NotImplementedError('toarray() is not defined for Operator_from_scipy.')

    def tosparse(self):
        return self._matrix

    def transpose(self, conjugate=False):
        return Operator_from_scipy(domain=self._codomain, codomain=self._domain, A=self._matrix.T)

    def dot(self, v, out=None):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            vh = v.toarray()
            outh = self._matrix.dot(vh)
            out =array_to_psydac(outh, self._codomain)
        else:
            vh = v.toarray()
            outh = self._matrix.dot(vh)
            out =array_to_psydac(outh, self._codomain)
        return out






def smooth_x_c1(V0h,nquads,hom_bc):
    dim_tot = V0h.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    V0      = V0h.symbolic_space
    domain  = V0.domain
    Interfaces  = domain.interfaces
    boundary = domain.boundary
    n_patches = len(V0h.spaces)

    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]


    patch_space = V0h.spaces[0]
    local_shape = [patch_space.spaces[0].nbasis,patch_space.spaces[1].nbasis]
    p      = patch_space.degree                     # spline degrees
    Nel    = patch_space.ncells                     # number of elements
    degree = patch_space.degree
    breakpoints_x = breakpoints(patch_space.knots[0],degree[0])
    breakpoints_y = breakpoints(patch_space.knots[1],degree[1])

    grid = [np.array([deepcopy((0.5*(u[0]+1)*(breakpoints_x[i+1]-breakpoints_x[i])+breakpoints_x[i])) for i in range(Nel[0])]),
            np.array([deepcopy((0.5*(u[1]+1)*(breakpoints_y[i+1]-breakpoints_y[i])+breakpoints_y[i])) for i in range(Nel[1])])]
    _, basis, span, _ = patch_space.preprocess_regular_tensor_grid(grid,der=1)

    span = [deepcopy(span[k] + patch_space.vector_space.starts[k] - patch_space.vector_space.shifts[k] * patch_space.vector_space.pads[k]) for k in range(2)]
    px=degree[0]
    enddom = breakpoints_x[-1]
    begdom = breakpoints_x[0]
    denom = enddom-begdom
    #Direction x, interface on the left
    if local_shape[0] < px+3:
        raise ValueError("patch space is too small to preserve enough moments")
        # print(px+3, local_shape[0])
    Mass_mat = np.zeros((px+2,local_shape[0]))
    for poldeg in range(px+2):
        for ie1 in range(Nel[0]):   #loop on cells
            for il1 in range(degree[0]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[0]): #loops on quadrature points
                    v0 = basis[0][ie1,il1,0,q1]
                    x  = grid[0][ie1,q1]
                    val += w[0][q1]*v0*((enddom-x)/denom)**poldeg
                locind=span[0][ie1]-degree[0]+il1
                Mass_mat[poldeg,locind]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:px+3]
    # print(Mass_mat.shape)
    # print(Mat_to_inv.shape)

    Correct_coef = np.linalg.solve(Mat_to_inv,Rhs)
    #Direction x, interface on the right

    for I in Interfaces:
        axis = I.axis
        i_minus = get_patch_index_from_face(domain, I.minus)
        i_plus  = get_patch_index_from_face(domain, I.plus )
        s_minus = V0h.spaces[i_minus]
        s_plus  = V0h.spaces[i_plus]
        n_deg_minus = s_minus.spaces[axis].nbasis
        patch_shape_minus = [s_minus.spaces[0].nbasis,s_minus.spaces[1].nbasis]
        patch_shape_plus  = [s_plus.spaces[0].nbasis,s_plus.spaces[1].nbasis]
        if axis == 0 :
            for i in range(s_plus.spaces[1].nbasis):
                indice_minus = loca2global([i_minus,n_deg_minus-1,i],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,0,i],n_patches,patch_shape_plus)
                indice_minus_1 = loca2global([i_minus,n_deg_minus-2,i],n_patches,patch_shape_minus)
                indice_plus_1  = loca2global([i_plus,1,i],n_patches,patch_shape_plus)
                Proj[indice_minus,indice_minus]-=1/2
                Proj[indice_plus,indice_plus]-=1/2
                Proj[indice_plus,indice_minus]+=1/2
                Proj[indice_minus,indice_plus]+=1/2
                Proj[indice_minus_1,indice_minus_1]-=1/2
                Proj[indice_plus_1,indice_plus_1]-=1/2
                Proj[indice_minus_1,indice_plus]+=1
                Proj[indice_plus_1,indice_minus]+=1
                Proj[indice_minus_1,indice_plus_1]-=1/2
                Proj[indice_plus_1,indice_minus_1]-=1/2
                #TODO : moment preserving correction
                #for p in range(0,px+2):
                #    #correction
                #    indice_plus_i  = loca2global([i_plus, p+1,                  i], n_patches,patch_shape_plus)
                #    indice_minus_i = loca2global([i_minus, n_deg_minus-1-(p+1), i], n_patches,patch_shape_minus)
                #    indice_plus    = loca2global([i_plus, 0,                    i], n_patches,patch_shape_plus)
                #    indice_minus   = loca2global([i_minus, n_deg_minus-1,       i], n_patches,patch_shape_minus)
                #    Proj[indice_plus_i,indice_plus]+=Correct_coef[p]/2
                #    Proj[indice_plus_i,indice_minus]-=Correct_coef[p]/2
                #    Proj[indice_minus_i,indice_plus]-=Correct_coef[p]/2
                #    Proj[indice_minus_i,indice_minus]+=Correct_coef[p]/2
    if hom_bc:
        for b in boundary : 
            axis = b.axis
            ext = b.ext
            i_patch = get_patch_index_from_face(domain,b)
            space = V0h.spaces[i_patch]
            n_deg = space.spaces[axis].nbasis
            patch_shape = [space.spaces[0].nbasis,space.spaces[1].nbasis]
            if axis == 0 :
                for i in range(space.spaces[1].nbasis):
                    if ext ==+1 :
                        indice = loca2global([i_patch,n_deg-1,i],n_patches,patch_shape)
                        indice_1 = loca2global([i_minus,n_deg-2,i],n_patches,patch_shape)
                    elif ext == -1 : 
                        indice = loca2global([i_patch,0,i],n_patches,patch_shape)
                        indice_1 = loca2global([i_minus,1,i],n_patches,patch_shape)
                    else :
                        ValueError("wrong value for ext")
                    Proj[indice,indice]-=1
                    #Proj[indice_1,indice_1]-=1/2
    
    return Proj

def smooth_y_c1(V0h,nquads,hom_bc):
    dim_tot = V0h.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    V0      = V0h.symbolic_space
    domain  = V0.domain
    Interfaces  = domain.interfaces
    boundary = domain.boundary
    n_patches = len(V0h.spaces)

    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]


    patch_space = V0h.spaces[0]
    local_shape = [patch_space.spaces[0].nbasis,patch_space.spaces[1].nbasis]
    p      = patch_space.degree                     # spline degrees
    Nel    = patch_space.ncells                     # number of elements
    degree = patch_space.degree
    breakpoints_x = breakpoints(patch_space.knots[0],degree[0])
    breakpoints_y = breakpoints(patch_space.knots[1],degree[1])

    grid = [np.array([deepcopy((0.5*(u[0]+1)*(breakpoints_x[i+1]-breakpoints_x[i])+breakpoints_x[i])) for i in range(Nel[0])]),
            np.array([deepcopy((0.5*(u[1]+1)*(breakpoints_y[i+1]-breakpoints_y[i])+breakpoints_y[i])) for i in range(Nel[1])])]
    _, basis, span, _ = patch_space.preprocess_regular_tensor_grid(grid,der=1)
    span = [deepcopy(span[k] + patch_space.vector_space.starts[k] - patch_space.vector_space.shifts[k] * patch_space.vector_space.pads[k]) for k in range(2)]
    py=degree[1]
    enddom = breakpoints_y[-1]
    begdom = breakpoints_y[0]
    denom = enddom-begdom
    if local_shape[1] < py+3:
        raise ValueError("patch space is too small to preserve enough moments")
    Mass_mat = np.zeros((py+2,local_shape[1]))
    for poldeg in range(py+2):
        for ie1 in range(Nel[1]):   #loop on cells
            for il1 in range(py+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis[1][ie1,il1,0,q1]
                    x  = grid[1][ie1,q1]
                    val += w[1][q1]*v0*((enddom-x)/denom)**poldeg
                locind=span[1][ie1]-degree[1]+il1
                Mass_mat[poldeg,locind]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:py+3]
    Correct_coef = np.linalg.solve(Mat_to_inv,Rhs)

    for I in Interfaces:
        axis = I.axis
        i_minus = get_patch_index_from_face(domain, I.minus)
        i_plus  = get_patch_index_from_face(domain, I.plus )
        s_minus = V0h.spaces[i_minus]
        s_plus  = V0h.spaces[i_plus]
        n_deg_minus = s_minus.spaces[axis].nbasis
        patch_shape_minus = [s_minus.spaces[0].nbasis,s_minus.spaces[1].nbasis]
        patch_shape_plus  = [s_plus.spaces[0].nbasis,s_plus.spaces[1].nbasis]

        if axis == 1 :
            for i in range(s_plus.spaces[1].nbasis):
                indice_minus = loca2global([i_minus,i,n_deg_minus-1],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,i,0],n_patches,patch_shape_plus)
                indice_minus_1 = loca2global([i_minus,i,n_deg_minus-2],n_patches,patch_shape_minus)
                indice_plus_1  = loca2global([i_plus,i,1],n_patches,patch_shape_plus)
                Proj[indice_minus,indice_minus]-=1/2
                Proj[indice_plus,indice_plus]-=1/2
                Proj[indice_plus,indice_minus]+=1/2
                Proj[indice_minus,indice_plus]+=1/2
                Proj[indice_minus_1,indice_minus_1]-=1/2
                Proj[indice_plus_1,indice_plus_1]-=1/2
                Proj[indice_minus_1,indice_plus]+=1
                Proj[indice_plus_1,indice_minus]+=1
                Proj[indice_minus_1,indice_plus_1]-=1/2
                Proj[indice_plus_1,indice_minus_1]-=1/2
                #for p in range(0,py+2):
                #    #TODO correction
                #    indice_plus_i  = loca2global([i_plus,  i, p+1                ], n_patches,patch_shape_plus)
                #    indice_minus_i = loca2global([i_minus, i, n_deg_minus-1-(p+1)], n_patches,patch_shape_minus)
                #    indice_plus    = loca2global([i_plus,  i, 0                  ], n_patches,patch_shape_plus)
                #    indice_minus   = loca2global([i_minus, i, n_deg_minus-1      ], n_patches,patch_shape_minus)
                #    Proj[indice_plus_i,indice_plus]+=Correct_coef[p]/2
                #    Proj[indice_plus_i,indice_minus]-=Correct_coef[p]/2
                #    Proj[indice_minus_i,indice_plus]-=Correct_coef[p]/2
                #    Proj[indice_minus_i,indice_minus]+=Correct_coef[p]/2

    if hom_bc:
        for b in boundary : 
            axis = b.axis
            ext = b.ext
            i_patch = get_patch_index_from_face(domain,b)
            space = V0h.spaces[i_patch]
            n_deg = space.spaces[axis].nbasis
            patch_shape = [space.spaces[0].nbasis,space.spaces[1].nbasis]
            if axis == 1 :
                for i in range(space.spaces[0].nbasis):
                    if ext ==+1 :
                        indice = loca2global([i_patch,i,n_deg-1],n_patches,patch_shape)
                        indice_1 = loca2global([i_patch,i,n_deg-2],n_patches,patch_shape)
                    elif ext == -1 : 
                        indice = loca2global([i_patch,i,0],n_patches,patch_shape)
                        indice_1 = loca2global([i_patch,i,1],n_patches,patch_shape)
                    else :
                        ValueError("wrong value for ext")
                    Proj[indice,indice]-=1
                    #Proj[indice_1,indice_1]-=1/2

    return Proj

def Conf_proj_0_c1(V0h,nquads,hom_bc=False):
    n_patches = len(V0h.spaces)
    if n_patches ==1:
        dim_tot = V0h.nbasis
        Proj_op    = IdentityOperator(V0h.vector_space)#sparse_eye(dim_tot,format="lil")
    else : 
        S_x=smooth_x_c1(V0h,nquads,hom_bc)
        S_y=smooth_y_c1(V0h,nquads,hom_bc)
        Proj=S_x@S_y
        Proj_op = Operator_from_scipy(V0h.vector_space,V0h.vector_space,Proj)
    return Proj_op.tosparse()


def Conf_proj_1_c1(V1h,nquads, hom_bc):

    dim_tot = V1h.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    V1      = V1h.symbolic_space
    domain  = V1.domain
    n_patches = len(V1h.spaces)

    if n_patches ==1:
        dim_tot = V1h.nbasis
        Proj    = sparse_eye(dim_tot,format="lil")
        return Proj

    boundary = domain.boundary
    Interfaces  = domain.interfaces

    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]


    patch_space = V1h.spaces[0]
    local_shape = [[patch_space.spaces[0].spaces[0].nbasis,patch_space.spaces[0].spaces[1].nbasis],[patch_space.spaces[1].spaces[0].nbasis,patch_space.spaces[1].spaces[1].nbasis]]
    p      = patch_space.degree                     # spline degrees
    Nel    = patch_space.ncells                     # number of elements
    patch_space_x = patch_space.spaces[0]
    patch_space_y = patch_space.spaces[1]
    degree_x = patch_space_x.degree
    degree_y = patch_space_y.degree
    breakpoints_x_x = breakpoints(patch_space_x.knots[0],degree_x[0])
    breakpoints_x_y = breakpoints(patch_space_x.knots[1],degree_x[1])

    grid = [np.array([deepcopy((0.5*(u[0]+1)*(breakpoints_x_x[i+1]-breakpoints_x_x[i])+breakpoints_x_x[i])) for i in range(Nel[0])]),
            np.array([deepcopy((0.5*(u[1]+1)*(breakpoints_x_y[i+1]-breakpoints_x_y[i])+breakpoints_x_y[i])) for i in range(Nel[1])])]
    _, basis_x, span_x, _ = patch_space_x.preprocess_regular_tensor_grid(grid,der=1)
    _, basis_y, span_y, _ = patch_space_y.preprocess_regular_tensor_grid(grid,der=1)
    span_x = [deepcopy(span_x[k] + patch_space_x.vector_space.starts[k] - patch_space_x.vector_space.shifts[k] * patch_space_x.vector_space.pads[k]) for k in range(2)]
    span_y = [deepcopy(span_y[k] + patch_space_y.vector_space.starts[k] - patch_space_y.vector_space.shifts[k] * patch_space_y.vector_space.pads[k]) for k in range(2)]
    px=degree_x[0]
    enddom = breakpoints_x_x[-1]
    begdom = breakpoints_x_x[0]
    denom = enddom-begdom
    #Direction x
    Mass_mat = np.zeros((px+1,local_shape[0][0]))
    for poldeg in range(px+1):
        for ie1 in range(Nel[0]):   #loop on cells
            for il1 in range(degree_x[0]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[0]): #loops on quadrature points
                    v0 = basis_x[0][ie1,il1,0,q1]
                    x  = grid[0][ie1,q1]
                    val += w[0][q1]*v0*((enddom-x)/denom)**poldeg
                locindx=span_x[0][ie1]-degree_x[0]+il1
                Mass_mat[poldeg,locindx]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:px+2]
    Correct_coef_x = np.linalg.solve(Mat_to_inv,Rhs)

    py=degree_y[1]
    enddom = breakpoints_x_y[-1]
    begdom = breakpoints_x_y[0]
    denom = enddom-begdom
    #Direction y
    Mass_mat = np.zeros((py+1,local_shape[1][1]))
    for poldeg in range(py+1):
        for ie1 in range(Nel[1]):   #loop on cells
            for il1 in range(degree_y[1]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis_y[1][ie1,il1,0,q1]
                    y  = grid[1][ie1,q1]
                    val += w[1][q1]*v0*((enddom-y)/denom)**poldeg
                locindy=span_y[1][ie1]-degree_y[1]+il1
                Mass_mat[poldeg,locindy]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:py+2]
    Correct_coef_y = np.linalg.solve(Mat_to_inv,Rhs)
    #Direction x, interface on the right
    """Mass_mat = np.zeros((px+1,local_shape[0][0]))
    for poldeg in range(px+1):
        for ie1 in range(Nel[0]):   #loop on cells
            for il1 in range(degree_x[0]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[0]): #loops on quadrature points
                    v0 = basis_x[0][ie1,il1,0,q1]
                    x  = grid_x[0][ie1,q1]
                    val += v0*((x-begdom)/denom)**poldeg
                locindx=span_x[0][ie1]-degree_x[0]+il1
                Mass_mat[poldeg,locindx]+=deepcopy(val)

    Rhs = Mass_mat[:,-1]
    Mat_to_inv = Mass_mat[:,-px-2:-1]
    print(Rhs)
    print(Mat_to_inv)

    Correct_coef_minus = np.linalg.solve(Mat_to_inv,Rhs)
    print(Correct_coef_minus)"""
    if V1.name=="Hdiv":
        for I in Interfaces:
            axis = I.axis
            i_minus = get_patch_index_from_face(domain, I.minus)
            i_plus  = get_patch_index_from_face(domain, I.plus )
            s_minus = V1h.spaces[i_minus]
            s_plus  = V1h.spaces[i_plus]
            n_deg_minus = s_minus.spaces[axis].spaces[axis].nbasis
            patch_shape_minus = [[s_minus.spaces[0].spaces[0].nbasis,s_minus.spaces[0].spaces[1].nbasis],
                                [s_minus.spaces[1].spaces[0].nbasis,s_minus.spaces[1].spaces[1].nbasis]]
            patch_shape_plus  = [[s_plus.spaces[0].spaces[0].nbasis,s_plus.spaces[0].spaces[1].nbasis],
                                [s_plus.spaces[1].spaces[0].nbasis,s_plus.spaces[1].spaces[1].nbasis]]
            if axis == 0 :
                for i in range(s_plus.spaces[0].spaces[1].nbasis):
                    indice_minus = loca2global([i_minus,0,n_deg_minus-1,i],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,0,0,i],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
                    for p in range(0,px+1):
                        #correction
                        indice_plus_i  = loca2global([i_plus,  0, p+1,                 i], n_patches,patch_shape_plus)
                        indice_minus_i = loca2global([i_minus, 0, n_deg_minus-1-(p+1), i], n_patches,patch_shape_minus)
                        indice_plus    = loca2global([i_plus,  0, 0,                   i], n_patches,patch_shape_plus)
                        indice_minus   = loca2global([i_minus, 0, n_deg_minus-1,       i], n_patches,patch_shape_minus)
                        Proj[indice_plus_i,indice_plus]+=Correct_coef_x[p]/2
                        Proj[indice_plus_i,indice_minus]-=Correct_coef_x[p]/2
                        Proj[indice_minus_i,indice_plus]-=Correct_coef_x[p]/2
                        Proj[indice_minus_i,indice_minus]+=Correct_coef_x[p]/2


            elif axis == 1 :
                for i in range(s_plus.spaces[1].spaces[0].nbasis):
                    indice_minus = loca2global([i_minus,1,i,n_deg_minus-1],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,1,i,0],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
                    for p in range(0,py+1):
                        #correction
                        indice_plus_i  = loca2global([i_plus,  1, i, p+1                ], n_patches,patch_shape_plus)
                        indice_minus_i = loca2global([i_minus, 1, i, n_deg_minus-1-(p+1)], n_patches,patch_shape_minus)
                        indice_plus    = loca2global([i_plus,  1, i, 0                  ], n_patches,patch_shape_plus)
                        indice_minus   = loca2global([i_minus, 1, i, n_deg_minus-1      ], n_patches,patch_shape_minus)
                        Proj[indice_plus_i,indice_plus]+=Correct_coef_y[p]/2
                        Proj[indice_plus_i,indice_minus]-=Correct_coef_y[p]/2
                        Proj[indice_minus_i,indice_plus]-=Correct_coef_y[p]/2
                        Proj[indice_minus_i,indice_minus]+=Correct_coef_y[p]/2
    elif V1.name=="Hcurl":
        for I in Interfaces:
            axis = I.axis
            i_minus = get_patch_index_from_face(domain, I.minus)
            i_plus  = get_patch_index_from_face(domain, I.plus )
            s_minus = V1h.spaces[i_minus]
            s_plus  = V1h.spaces[i_plus]
            naxis = (axis+1)%2
            n_deg_minus = s_minus.spaces[naxis].spaces[axis].nbasis
            patch_shape_minus = [[s_minus.spaces[0].spaces[0].nbasis,s_minus.spaces[0].spaces[1].nbasis],
                                [s_minus.spaces[1].spaces[0].nbasis,s_minus.spaces[1].spaces[1].nbasis]]
            patch_shape_plus  = [[s_plus.spaces[0].spaces[0].nbasis,s_plus.spaces[0].spaces[1].nbasis],
                                [s_plus.spaces[1].spaces[0].nbasis,s_plus.spaces[1].spaces[1].nbasis]]
            if axis == 0 :
                for i in range(s_plus.spaces[1].spaces[1].nbasis):
                    indice_minus = loca2global([i_minus,1,n_deg_minus-1,i],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,1,0,i],n_patches,patch_shape_plus)
                    indice_minus_1 = loca2global([i_minus,1,n_deg_minus-2,i],n_patches,patch_shape_minus)
                    indice_plus_1  = loca2global([i_plus,1,1,i],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
                    Proj[indice_minus_1,indice_minus_1]-=1/2
                    Proj[indice_plus_1,indice_plus_1]-=1/2
                    Proj[indice_minus_1,indice_plus]+=1
                    Proj[indice_plus_1,indice_minus]+=1
                    Proj[indice_minus_1,indice_plus_1]-=1/2
                    Proj[indice_plus_1,indice_minus_1]-=1/2
            elif axis == 1 :
                for i in range(s_plus.spaces[0].spaces[0].nbasis):
                    indice_minus = loca2global([i_minus,0,i,n_deg_minus-1],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,0,i,0],n_patches,patch_shape_plus)
                    indice_minus_1 = loca2global([i_minus,0,i,n_deg_minus-2],n_patches,patch_shape_minus)
                    indice_plus_1  = loca2global([i_plus,0,i,1],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
                    Proj[indice_minus_1,indice_minus_1]-=1/2
                    Proj[indice_plus_1,indice_plus_1]-=1/2
                    Proj[indice_minus_1,indice_plus]+=1
                    Proj[indice_plus_1,indice_minus]+=1
                    Proj[indice_minus_1,indice_plus_1]-=1/2
                    Proj[indice_plus_1,indice_minus_1]-=1/2

        if hom_bc:
            for b in boundary : 
                axis = b.axis
                ext = b.ext
                i_patch = get_patch_index_from_face(domain,b)
                space = V1h.spaces[i_patch]
                naxis = (axis+1)%2
                n_deg = space.spaces[naxis].spaces[axis].nbasis
                patch_shape  = [[space.spaces[0].spaces[0].nbasis,space.spaces[0].spaces[1].nbasis],
                                [space.spaces[1].spaces[0].nbasis,space.spaces[1].spaces[1].nbasis]]
                if axis == 0 :
                    for i in range(space.spaces[1].spaces[1].nbasis):
                        if ext ==+1 :
                            indice = loca2global([i_patch,1,n_deg-1,i],n_patches,patch_shape)
                            indice_1 = loca2global([i_patch,1,n_deg-2,i],n_patches,patch_shape)
                        elif ext == -1 : 
                            indice = loca2global([i_patch,1,0,i],n_patches,patch_shape)
                            indice_1 = loca2global([i_patch,1,1,i],n_patches,patch_shape)
                        else :
                            ValueError("wrong value for ext")
                        Proj[indice,indice]-=1
                        #Proj[indice_1,indice_1]-=1/2
                elif axis == 1 :
                    for i in range(space.spaces[0].spaces[0].nbasis):
                        if ext ==+1 :
                            indice = loca2global([i_patch,0,i,n_deg-1],n_patches,patch_shape)
                            indice_1 = loca2global([i_patch,0,i,n_deg-2],n_patches,patch_shape)
                        elif ext == -1 : 
                            indice = loca2global([i_patch,0,i,0],n_patches,patch_shape)
                            indice_1 = loca2global([i_patch,0,i,1],n_patches,patch_shape)
                        else :
                            ValueError("wrong value for ext")
                        Proj[indice,indice]-=1
                        #Proj[indice_1,indice_1]-=1/2
    else :
        print("Error in Conf_proj_1 : wrong kind of space")
    return Proj