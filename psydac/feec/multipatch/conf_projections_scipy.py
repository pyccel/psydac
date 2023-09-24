import numpy as np
from copy import deepcopy
from scipy.sparse import eye as sparse_eye

from sympde.topology import Boundary, Interface, Union
from psydac.core.bsplines import breakpoints
from psydac.utilities.quadratures import gauss_legendre, gauss_lobatto
from psydac.linalg.basic        import IdentityOperator
from psydac.feec.multipatch.operators import get_patch_index_from_face
from psydac.linalg.basic        import IdentityOperator
from psydac.linalg.basic        import LinearOperator, Vector
from psydac.linalg.utilities    import array_to_psydac


####  projection from valentin:

mom_pres = True
proj_op = 0

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
        # print(f'ii = {ii}')
        # print(f'single_patch_shapes[com_index] = {single_patch_shapes[com_index]}')
        # print(single_patch_shapes[com_index])
        Ipc = np.ravel_multi_index(ii, dims=single_patch_shapes[com_index], order='C')
        sizes = [np.product(s) for s in single_patch_shapes]
        Ip = sum(sizes[:com_index]) + Ipc
        I = np.ravel_multi_index((patch_index, Ip), dims=(n_patches, sum(sizes)), order='C')

    return I


def glob_ind_interface_scalfield(i, j, side, k_patch, axis, patch_shape, n_patches, nb_dofs_across):
    # returns global index of dof close to interface, for a scalar field
    #
    # i: dof index along interface
    # j: relative dof index (distance) from interface
    # side: -1 or +1 for minus or plus patch of interface
    # k_patch: index of patch
    # nb_dofs_across: nb dofs in direction perpendicular to interface
    if side == 1:
        absolute_j = j 
    else:
        absolute_j = nb_dofs_across-1-j
    
    if axis == 0:
        return loca2global([k_patch, absolute_j, i], n_patches, patch_shape)
    else:
        return loca2global([k_patch, i, absolute_j], n_patches, patch_shape)
    
def glob_ind_interface_vecfield(i, j, side, k_patch, axis, patch_shape, n_patches, nb_dofs_across, comp):
    # returns global index of dof close to interface, for a scalar field
    #
    # i: dof index along interface
    # j: relative dof index (distance) from interface
    # side: -1 or +1 for minus or plus patch of interface
    # k_patch: index of patch
    # nb_dofs_across: nb dofs in direction perpendicular to interface
    # comp: component
    if side == 1:
        absolute_j = j 
    else:
        absolute_j = nb_dofs_across-1-j
    
    if axis == 0:
        # print(f'local-index = {absolute_j}, {i}   //  (nb_dofs_across = {nb_dofs_across})')
        return loca2global([k_patch, comp, absolute_j, i], n_patches, patch_shape)
    else:
        return loca2global([k_patch, comp, i, absolute_j], n_patches, patch_shape)
    

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
                if mom_pres:
                    for p in range(0,px+2):
                        #correction
                        indice_plus_j  = loca2global([i_plus, p+1,                  i], n_patches,patch_shape_plus)
                        indice_minus_j = loca2global([i_minus, n_deg_minus-1-(p+1), i], n_patches,patch_shape_minus)
                        indice_plus    = loca2global([i_plus, 0,                    i], n_patches,patch_shape_plus)
                        indice_minus   = loca2global([i_minus, n_deg_minus-1,       i], n_patches,patch_shape_minus)
                        Proj[indice_plus_j,indice_plus]+=Correct_coef[p]/2
                        Proj[indice_plus_j,indice_minus]-=Correct_coef[p]/2
                        Proj[indice_minus_j,indice_plus]-=Correct_coef[p]/2
                        Proj[indice_minus_j,indice_minus]+=Correct_coef[p]/2
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
                if mom_pres:
                    for p in range(0,py+2):
                        #correction
                        indice_plus_j  = loca2global([i_plus,  i, p+1                ], n_patches,patch_shape_plus)
                        indice_minus_j = loca2global([i_minus, i, n_deg_minus-1-(p+1)], n_patches,patch_shape_minus)
                        indice_plus    = loca2global([i_plus,  i, 0                  ], n_patches,patch_shape_plus)
                        indice_minus   = loca2global([i_minus, i, n_deg_minus-1      ], n_patches,patch_shape_minus)
                        Proj[indice_plus_j,indice_plus]+=Correct_coef[p]/2
                        Proj[indice_plus_j,indice_minus]-=Correct_coef[p]/2
                        Proj[indice_minus_j,indice_plus]-=Correct_coef[p]/2
                        Proj[indice_minus_j,indice_minus]+=Correct_coef[p]/2
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
                    if mom_pres:                    
                        for p in range(0,px+1):
                            #correction
                            indice_plus_j  = loca2global([i_plus,  0, p+1,                 i], n_patches,patch_shape_plus)
                            indice_minus_j = loca2global([i_minus, 0, n_deg_minus-1-(p+1), i], n_patches,patch_shape_minus)
                            indice_plus    = loca2global([i_plus,  0, 0,                   i], n_patches,patch_shape_plus)
                            indice_minus   = loca2global([i_minus, 0, n_deg_minus-1,       i], n_patches,patch_shape_minus)
                            Proj[indice_plus_j,indice_plus]+=Correct_coef_x[p]/2
                            Proj[indice_plus_j,indice_minus]-=Correct_coef_x[p]/2
                            Proj[indice_minus_j,indice_plus]-=Correct_coef_x[p]/2
                            Proj[indice_minus_j,indice_minus]+=Correct_coef_x[p]/2


            elif axis == 1 :
                for i in range(s_plus.spaces[1].spaces[0].nbasis):
                    indice_minus = loca2global([i_minus,1,i,n_deg_minus-1],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,1,i,0],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
                    if mom_pres:
                        for p in range(0,py+1):
                            #correction
                            indice_plus_j  = loca2global([i_plus,  1, i, p+1                ], n_patches,patch_shape_plus)
                            indice_minus_j = loca2global([i_minus, 1, i, n_deg_minus-1-(p+1)], n_patches,patch_shape_minus)
                            indice_plus    = loca2global([i_plus,  1, i, 0                  ], n_patches,patch_shape_plus)
                            indice_minus   = loca2global([i_minus, 1, i, n_deg_minus-1      ], n_patches,patch_shape_minus)
                            Proj[indice_plus_j,indice_plus]+=Correct_coef_y[p]/2
                            Proj[indice_plus_j,indice_minus]-=Correct_coef_y[p]/2
                            Proj[indice_minus_j,indice_plus]-=Correct_coef_y[p]/2
                            Proj[indice_minus_j,indice_minus]+=Correct_coef_y[p]/2
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
    """Create the matrix enforcing the C1 continuity with moment preservation in the x direction for the V0 space
    
    Parameters
    ----------
    V0h : The discrete broken V0 space in which we will compute the projection

    nquads : number of integration points to compute the moment preserving weights

    hom_bc : Wether or not enforce boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
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
    if local_shape[0] < px+2:
        raise ValueError("patch space is too small to preserve enough moments")
    Mass_mat = np.zeros((px+1,local_shape[0]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(px+1):
        for ie1 in range(Nel[0]):   #loop on cells
            for il1 in range(px+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[0]): #loops on quadrature points
                    v0 = basis[0][ie1,il1,0,q1]
                    x  = grid[0][ie1,q1]
                    val += w[0][q1]*v0*((enddom-x)/denom)**poldeg
                locind=span[0][ie1]-degree[0]+il1
                Mass_mat[poldeg,locind]+=val
    Rhs_0 = Mass_mat[:,0]
    Rhs_1 = Mass_mat[:,1]
    Mat_to_inv = Mass_mat[:,2:px+3]
    Mat_to_inv_bnd = Mass_mat[:,1:px+2]

    Correct_coef_0 = np.linalg.solve(Mat_to_inv,Rhs_0)
    Correct_coef_1 = np.linalg.solve(Mat_to_inv,Rhs_1)
    Correct_coef_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)

    # using formulas similar to Conf_proj_1_c1, but here along axis = 0 only
    proj_degs = [px,0] # this differs from Conf_proj_1_c1
    a_same = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # a_same[axis]: coefs of P B0 on same patch
    a_ngbr = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # a_ngbr[axis]: coefs of P B0 on neighbor patch
    b_same = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # b_same[axis]: coefs of P B1 on same patch
    b_ngbr = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # b_ngbr[axis]: coefs of P B1 on neighbor patch
    corr_coef_0 = [Correct_coef_0, None]  # this differs from Conf_proj_1_c1
    corr_coef_1 = [Correct_coef_1, None]  # this differs from Conf_proj_1_c1

    for axis in [0] :
        p_ax = proj_degs[axis]
        a_sm = a_same[axis]
        a_nb = a_ngbr[axis]
        b_sm = b_same[axis]
        b_nb = b_ngbr[axis]
        # projection coefs:
        a_sm[0] = 1/2
        if proj_op == 0:
            # new slope is average of old ones
            a_sm[1] = 0  
        elif proj_op == 1:
            # new slope is average of old ones after averaging of interface coef
            a_sm[1] = 1/2
        elif proj_op == 2:
            # new slope is average of reconstructed ones using local values and slopes
            a_sm[1] = 1/(2*p_ax)
        else:
            # just to try something else
            a_sm[1] = proj_op/2
            
        b_sm[0] = 0
        b_sm[1] = 1/2
        # C1 conformity + proj constraints:
        a_nb[0] = a_sm[0]
        a_nb[1] = 2*a_sm[0] - a_sm[1]
        b_nb[0] = b_sm[0]
        b_nb[1] = 2*b_sm[0] - b_sm[1]
        if mom_pres:
            cc_0_ax = corr_coef_0[axis]
            cc_1_ax = corr_coef_1[axis]
            for p in range(0,p_ax+1):
                # correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                # modified by the C1 enforcement
                a_sm[p+2] = (1-a_sm[0]) * cc_0_ax[p]     -a_sm[1]  * cc_1_ax[p]
                b_sm[p+2] =   -b_sm[0]  * cc_0_ax[p] + (1-b_sm[1]) * cc_1_ax[p]
                # proj constraints:
                b_nb[p+2] = b_sm[p+2]
                a_nb[p+2] = -(a_sm[p+2] + 2*b_sm[p+2])

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
            a_sm = a_same[axis]
            a_nb = a_ngbr[axis]
            b_sm = b_same[axis]
            b_nb = b_ngbr[axis]

            for i in range(s_plus.spaces[1].nbasis):
                indice_minus = loca2global([i_minus,n_deg_minus-1,i],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,0,i],n_patches,patch_shape_plus)
                indice_minus_1 = loca2global([i_minus,n_deg_minus-2,i],n_patches,patch_shape_minus)
                indice_plus_1  = loca2global([i_plus,1,i],n_patches,patch_shape_plus)
                
                #changing this coefficients ensure C1 continuity at the interface
                # Proj[indice_minus,indice_minus]-=1/2 
                # Proj[indice_plus,indice_plus]-=1/2   
                # Proj[indice_plus,indice_minus]+=1/2  
                # Proj[indice_minus,indice_plus]+=1/2  
                # Proj[indice_minus_1,indice_minus_1]-=1/2 
                # Proj[indice_plus_1,indice_plus_1]-=1/2 
                # Proj[indice_minus_1,indice_plus]+=1 
                # Proj[indice_plus_1,indice_minus]+=1 
                # Proj[indice_minus_1,indice_plus_1]-=1/2
                # Proj[indice_plus_1,indice_minus_1]-=1/2

                Proj[indice_minus,indice_minus] += (a_sm[0]-1)
                Proj[indice_plus, indice_plus ] += (a_sm[0]-1)
                Proj[indice_plus, indice_minus] += a_nb[0]
                Proj[indice_minus,indice_plus ] += a_nb[0]

                # note: b_sm[0] = b_nb[0] is hard coded here
                Proj[indice_minus_1,indice_minus_1] += (b_sm[1]-1)
                Proj[indice_minus_1,indice_minus  ] += a_sm[1]    
                Proj[indice_minus_1,indice_plus   ] += a_nb[1]
                Proj[indice_minus_1,indice_plus_1 ] += b_nb[1]
                
                Proj[indice_plus_1,indice_minus_1]  += b_nb[1]      
                Proj[indice_plus_1,indice_minus  ]  += a_nb[1]      
                Proj[indice_plus_1,indice_plus   ]  += a_sm[1]      
                Proj[indice_plus_1,indice_plus_1 ]  += (b_sm[1]-1) 

                # if mom_pres:
                for p in range(0,px+1):
                    #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                    #modified by the C1 enforcement

                    # glob_ind_minus(i,j) = loca2global([i_minus,1,n_deg_minus-1-j,i], n_patches, patch_shape_minus)
                    # indice_minus_i = glob_ind_minus (i,p+2)
                    indice_minus_j = loca2global([i_minus, n_deg_minus-1-(p+2), i], n_patches,patch_shape_minus)
                    indice_plus_j  = loca2global([i_plus, p+2,                  i], n_patches,patch_shape_plus)                    
                    # indice_plus    = loca2global([i_plus, 0,                    i], n_patches,patch_shape_plus)
                    # indice_minus   = loca2global([i_minus, n_deg_minus-1,       i], n_patches,patch_shape_minus)
                    # indice_plus    = loca2global([i_plus, 1,                    i], n_patches,patch_shape_plus)
                    # indice_minus   = loca2global([i_minus, n_deg_minus-2,       i], n_patches,patch_shape_minus)
                    # Proj[indice_plus_i,indice_plus]+=Correct_coef_0[p]/2
                    # Proj[indice_plus_i,indice_minus]-=Correct_coef_0[p]/2
                    # Proj[indice_minus_i,indice_plus]-=Correct_coef_0[p]/2
                    # Proj[indice_minus_i,indice_minus]+=Correct_coef_0[p]/2
                    # Proj[indice_minus_i, indice_minus_1]+=Correct_coef_1[p]/2
                    # Proj[indice_plus_i, indice_plus_1]+=Correct_coef_1[p]/2
                    # Proj[indice_minus_i, indice_plus]-=Correct_coef_1[p]
                    # Proj[indice_plus_i, indice_minus]-=Correct_coef_1[p]
                    # Proj[indice_minus_i, indice_plus_1]+=Correct_coef_1[p]/2
                    # Proj[indice_plus_i, indice_minus_1]+=Correct_coef_1[p]/2

                    #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                    #modified by the C1 enforcement
                    # indice_minus_i = glob_ind_plus (i,p+2)
                    # indice_plus_i  = glob_ind_plus (i,p+2)
                    
                    Proj[indice_minus_j,indice_minus]   += a_sm[p+2]
                    Proj[indice_plus_j, indice_plus ]   += a_sm[p+2]
                    Proj[indice_plus_j, indice_minus]   += a_nb[p+2]
                    Proj[indice_minus_j,indice_plus ]   += a_nb[p+2]

                    Proj[indice_minus_j,indice_minus_1] += b_sm[p+2]
                    Proj[indice_plus_j,indice_plus_1]   += b_sm[p+2]
                    Proj[indice_minus_j,indice_plus_1]  += b_nb[p+2]
                    Proj[indice_plus_j,indice_minus_1]  += b_nb[p+2]

    if hom_bc:
        for b in boundary : 
            axis = b.axis
            ext = b.ext
            i_patch = get_patch_index_from_face(domain,b)
            space = V0h.spaces[i_patch]
            n_deg = space.spaces[axis].nbasis
            patch_shape = [space.spaces[0].nbasis,space.spaces[1].nbasis]
            if axis == 0 :
                a_sm = a_same[axis]
                a_nb = a_ngbr[axis]
                b_sm = b_same[axis]
                b_nb = b_ngbr[axis]

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

                    if( abs(Proj[indice,indice]) > 1e-10 ):
                        print(f'STRANGE (x): Proj[indice,indice] = {Proj[indice,indice]} ... ?')

                    if mom_pres:
                        for p in range(0,px+1):
                            #correction
                            if ext ==+1 :
                                indice_j = loca2global([i_patch,n_deg-(p+2),i],n_patches,patch_shape)
                            elif ext == -1 : 
                                indice_j = loca2global([i_patch,p+2,i],n_patches,patch_shape)
                            else :
                                ValueError("wrong value for ext")
                            Proj[indice_j,indice]+=Correct_coef_bnd[p]    

    return Proj

def smooth_y_c1(V0h,nquads,hom_bc):
    """Create the matrix enforcing the C1 continuity with moment preservation in the y direction for the V0 space
    
    Parameters
    ----------
    V0h : The discrete broken V0 space in which we will compute the projection

    nquads : number of integration points to compute the moment preserving weights

    hom_bc : Wether or not enforce boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
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
    Mass_mat = np.zeros((py+1,local_shape[1]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(py+1):
        for ie1 in range(Nel[1]):   #loop on cells
            for il1 in range(py+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis[1][ie1,il1,0,q1]
                    x  = grid[1][ie1,q1]
                    val += w[1][q1]*v0*((enddom-x)/denom)**poldeg
                locind=span[1][ie1]-degree[1]+il1
                Mass_mat[poldeg,locind]+=val
    Rhs_0 = Mass_mat[:,0]
    Rhs_1 = Mass_mat[:,1]
    Mat_to_inv = Mass_mat[:,2:py+3]
    Mat_to_inv_bnd = Mass_mat[:,1:py+2]

    Correct_coef_0 = np.linalg.solve(Mat_to_inv,Rhs_0)
    Correct_coef_1 = np.linalg.solve(Mat_to_inv,Rhs_1)
    Correct_coef_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)

    # using formulas similar to Conf_proj_1_c1, but here along axis = 1 only
    proj_degs = [0,py] # this differs from Conf_proj_1_c1
    a_same = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # a_same[axis]: coefs of P B0 on same patch
    a_ngbr = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # a_ngbr[axis]: coefs of P B0 on neighbor patch
    b_same = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # b_same[axis]: coefs of P B1 on same patch
    b_ngbr = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # b_ngbr[axis]: coefs of P B1 on neighbor patch
    corr_coef_0 = [None, Correct_coef_0]  # this differs from Conf_proj_1_c1
    corr_coef_1 = [None, Correct_coef_1]  # this differs from Conf_proj_1_c1

    for axis in [1] :
        p_ax = proj_degs[axis]
        a_sm = a_same[axis]
        a_nb = a_ngbr[axis]
        b_sm = b_same[axis]
        b_nb = b_ngbr[axis]
        # projection coefs:
        a_sm[0] = 1/2
        if proj_op == 0:
            # new slope is average of old ones
            a_sm[1] = 0  
        elif proj_op == 1:
            # new slope is average of old ones after averaging of interface coef
            a_sm[1] = 1/2
        elif proj_op == 2:
            # new slope is average of reconstructed ones using local values and slopes
            a_sm[1] = 1/(2*p_ax)
        else:
            # just to try something else
            a_sm[1] = proj_op/2
            
        b_sm[0] = 0
        b_sm[1] = 1/2
        # C1 conformity + proj constraints:
        a_nb[0] = a_sm[0]
        a_nb[1] = 2*a_sm[0] - a_sm[1]
        b_nb[0] = b_sm[0]
        b_nb[1] = 2*b_sm[0] - b_sm[1]
        if mom_pres:
            cc_0_ax = corr_coef_0[axis]
            cc_1_ax = corr_coef_1[axis]
            for p in range(0,p_ax+1):
                # correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                # modified by the C1 enforcement
                a_sm[p+2] = (1-a_sm[0]) * cc_0_ax[p]     -a_sm[1]  * cc_1_ax[p]
                b_sm[p+2] =   -b_sm[0]  * cc_0_ax[p] + (1-b_sm[1]) * cc_1_ax[p]
                # proj constraints:
                b_nb[p+2] = b_sm[p+2]
                a_nb[p+2] = -(a_sm[p+2] + 2*b_sm[p+2])

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
            a_sm = a_same[axis]
            a_nb = a_ngbr[axis]
            b_sm = b_same[axis]
            b_nb = b_ngbr[axis]

            for i in range(s_plus.spaces[1].nbasis):
                indice_minus = loca2global([i_minus,i,n_deg_minus-1],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,i,0],n_patches,patch_shape_plus)
                indice_minus_1 = loca2global([i_minus,i,n_deg_minus-2],n_patches,patch_shape_minus)
                indice_plus_1  = loca2global([i_plus,i,1],n_patches,patch_shape_plus)
                #changing this coefficients ensure C1 continuity at the interface
                # Proj[indice_minus,indice_minus]-=1/2
                # Proj[indice_plus,indice_plus]-=1/2
                # Proj[indice_plus,indice_minus]+=1/2
                # Proj[indice_minus,indice_plus]+=1/2
                # Proj[indice_minus_1,indice_minus_1]-=1/2
                # Proj[indice_plus_1,indice_plus_1]-=1/2
                # Proj[indice_minus_1,indice_plus]+=1
                # Proj[indice_plus_1,indice_minus]+=1
                # Proj[indice_minus_1,indice_plus_1]-=1/2
                # Proj[indice_plus_1,indice_minus_1]-=1/2

                # note: b_sm[0] = b_nb[0] is hard coded here
                Proj[indice_minus,indice_minus] += (a_sm[0]-1)
                Proj[indice_plus, indice_plus ] += (a_sm[0]-1)
                Proj[indice_plus, indice_minus] += a_nb[0]
                Proj[indice_minus,indice_plus ] += a_nb[0]

                Proj[indice_minus_1,indice_minus_1] += (b_sm[1]-1)
                Proj[indice_minus_1,indice_minus  ] += a_sm[1]    
                Proj[indice_minus_1,indice_plus   ] += a_nb[1]
                Proj[indice_minus_1,indice_plus_1 ] += b_nb[1]
                
                Proj[indice_plus_1,indice_minus_1]  += b_nb[1]      
                Proj[indice_plus_1,indice_minus  ]  += a_nb[1]      
                Proj[indice_plus_1,indice_plus   ]  += a_sm[1]      
                Proj[indice_plus_1,indice_plus_1 ]  += (b_sm[1]-1) 

                # if mom_pres:
                for p in range(0,py+1):
                    #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                    #modified by the C1 enforcement
                    indice_plus_j  = loca2global([i_plus,  i, p+2                ], n_patches,patch_shape_plus)
                    indice_minus_j = loca2global([i_minus, i, n_deg_minus-1-(p+2)], n_patches,patch_shape_minus)
                    # indice_plus    = loca2global([i_plus,  i, 0                  ], n_patches,patch_shape_plus)
                    # indice_minus   = loca2global([i_minus, i, n_deg_minus-1      ], n_patches,patch_shape_minus)
                    # indice_minus_1 = loca2global([i_minus,i,n_deg_minus-2],n_patches,patch_shape_minus)
                    # indice_plus_1  = loca2global([i_plus,i,1],n_patches,patch_shape_plus)
                    # Proj[indice_plus_j,indice_plus]+=Correct_coef_0[p]/2
                    # Proj[indice_plus_j,indice_minus]-=Correct_coef_0[p]/2
                    # Proj[indice_minus_j,indice_plus]-=Correct_coef_0[p]/2
                    # Proj[indice_minus_j,indice_minus]+=Correct_coef_0[p]/2
                    # Proj[indice_minus_j,indice_minus_1]+=Correct_coef_1[p]/2
                    # Proj[indice_plus_j,indice_plus_1]+=Correct_coef_1[p]/2
                    # Proj[indice_minus_j,indice_plus]-=Correct_coef_1[p]
                    # Proj[indice_plus_j,indice_minus]-=Correct_coef_1[p]
                    # Proj[indice_minus_j,indice_plus_1]+=Correct_coef_1[p]/2
                    # Proj[indice_plus_j,indice_minus_1]+=Correct_coef_1[p]/2
                    Proj[indice_minus_j,indice_minus]   += a_sm[p+2]
                    Proj[indice_plus_j, indice_plus ]   += a_sm[p+2]
                    Proj[indice_plus_j, indice_minus]   += a_nb[p+2]
                    Proj[indice_minus_j,indice_plus ]   += a_nb[p+2]

                    Proj[indice_minus_j,indice_minus_1] += b_sm[p+2]
                    Proj[indice_plus_j,indice_plus_1]   += b_sm[p+2]
                    Proj[indice_minus_j,indice_plus_1]  += b_nb[p+2]
                    Proj[indice_plus_j,indice_minus_1]  += b_nb[p+2]

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

                    if( abs(Proj[indice,indice]) > 1e-10 ):
                        print(f'STRANGE (y): Proj[indice,indice] = {Proj[indice,indice]} ... ?')

                    if mom_pres:
                        for p in range(0,py+1):
                            #correction
                            if ext ==+1 :
                                indice_i = loca2global([i_patch,i,n_deg-(p+2)],n_patches,patch_shape)
                            elif ext == -1 : 
                                indice_i = loca2global([i_patch,i,p+2],n_patches,patch_shape)
                            else :
                                ValueError("wrong value for ext")
                            Proj[indice_i,indice]+=Correct_coef_bnd[p]    
    return Proj

def Conf_proj_0_c1(V0h,nquads,hom_bc=False):
    """Create the matrix enforcing the C1 continuity with moment preservation for the V0 space
    
    Parameters
    ----------
    V0h : The discrete broken V0 space in which we will compute the projection

    nquads : number of integration points to compute the moment preserving weights

    hom_bc : Wether or not enforce boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
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
    """Create the matrix enforcing the C1 continuity with moment preservation in the tangential direction for the V1 space
    
    Parameters
    ----------
    V0h : The discrete broken V0 space in which we will compute the projection

    nquads : number of integration points to compute the moment preserving weights

    hom_bc : Wether or not enforce boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """

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
    Nel    = patch_space.ncells                     # number of elements
    patch_space_x = patch_space.spaces[0]
    patch_space_y = patch_space.spaces[1]
    degree_x = patch_space_x.degree
    degree_y = patch_space_y.degree
    breakpoints_x_x = breakpoints(patch_space_x.knots[0],degree_x[0])
    breakpoints_x_y = breakpoints(patch_space_x.knots[1],degree_x[1])

    #compute grid for the correction coefficients ensuring moment preservation
    grid = [np.array([deepcopy((0.5*(u[0]+1)*(breakpoints_x_x[i+1]-breakpoints_x_x[i])+breakpoints_x_x[i])) for i in range(Nel[0])]),
            np.array([deepcopy((0.5*(u[1]+1)*(breakpoints_x_y[i+1]-breakpoints_x_y[i])+breakpoints_x_y[i])) for i in range(Nel[1])])]
    _, basis_x, span_x, _ = patch_space_x.preprocess_regular_tensor_grid(grid,der=0)
    _, basis_y, span_y, _ = patch_space_y.preprocess_regular_tensor_grid(grid,der=0)
    span_x = [deepcopy(span_x[k] + patch_space_x.vector_space.starts[k] - patch_space_x.vector_space.shifts[k] * patch_space_x.vector_space.pads[k]) for k in range(2)]
    span_y = [deepcopy(span_y[k] + patch_space_y.vector_space.starts[k] - patch_space_y.vector_space.shifts[k] * patch_space_y.vector_space.pads[k]) for k in range(2)]
    px=degree_x[1]
    enddom = breakpoints_x_x[-1]
    begdom = breakpoints_x_x[0]
    denom = enddom-begdom
    #Direction y (average x component)
    Mass_mat = np.zeros((px+1,local_shape[0][1]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(px+1):
        for ie1 in range(Nel[1]):   #loop on cells
            cell_size = breakpoints_x_y[ie1+1]-breakpoints_x_y[ie1]
            for il1 in range(px+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis_x[1][ie1,il1,0,q1]
                    x  = grid[1][ie1,q1]
                    val += cell_size*0.5*w[1][q1]*v0*((x-begdom)/denom)**poldeg
                locindx=span_x[1][ie1]-degree_x[1]+il1
                Mass_mat[poldeg,locindx]+=val
    Rhs_0 = Mass_mat[:,0]
    Rhs_1 = Mass_mat[:,1]
    Mat_to_inv = Mass_mat[:,2:px+3]
    Mat_to_inv_bnd = Mass_mat[:,1:px+2]

    Correct_coef_x_0 = np.linalg.solve(Mat_to_inv,Rhs_0) #coefficient to ensure moment preservation due to modification of first coeff
    Correct_coef_x_1 = np.linalg.solve(Mat_to_inv,Rhs_1) # same second coeff

    Correct_coef_x_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)

    py=degree_y[0]
    enddom = breakpoints_x_y[-1]
    begdom = breakpoints_x_y[0]
    denom = enddom-begdom
    #Direction x (average y component)
    Mass_mat = np.zeros((py+1,local_shape[1][0]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(py+1):
        for ie1 in range(Nel[1]):   #loop on cells
            cell_size = breakpoints_x_y[ie1+1]-breakpoints_x_y[ie1]
            for il1 in range(py+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis_y[0][ie1,il1,0,q1]
                    y  = grid[1][ie1,q1]
                    val += cell_size/2*w[1][q1]*v0*((y-begdom)/denom)**poldeg 
                locindy=span_y[0][ie1]-degree_y[0]+il1
                Mass_mat[poldeg,locindy]+=val
                
    Rhs_0 = Mass_mat[:,0]
    Rhs_1 = Mass_mat[:,1]
    Mat_to_inv = Mass_mat[:,2:py+3]
    Mat_to_inv_bnd = Mass_mat[:,1:py+2]

    Correct_coef_y_1 = np.linalg.solve(Mat_to_inv,Rhs_1)
    Correct_coef_y_0 = np.linalg.solve(Mat_to_inv,Rhs_0)

    Correct_coef_y_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)
    #Direction y, interface on the right
    if V1.name=="Hdiv":
        NotImplementedError("only the Hcurl is really implemented")
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
                    if mom_pres:
                        for p in range(0,px+1):
                            #correction
                            indice_plus_j  = loca2global([i_plus,  0, p+2,                 i], n_patches,patch_shape_plus)
                            indice_minus_j = loca2global([i_minus, 0, n_deg_minus-1-(p+2), i], n_patches,patch_shape_minus)
                            indice_plus    = loca2global([i_plus,  0, 0,                   i], n_patches,patch_shape_plus)
                            indice_minus   = loca2global([i_minus, 0, n_deg_minus-1,       i], n_patches,patch_shape_minus)
                            Proj[indice_plus_j,indice_plus]+=Correct_coef_x_0[p]/2
                            Proj[indice_plus_j,indice_minus]-=Correct_coef_x_0[p]/2
                            Proj[indice_minus_j,indice_plus]-=Correct_coef_x_0[p]/2
                            Proj[indice_minus_j,indice_minus]+=Correct_coef_x_0[p]/2


            elif axis == 1 :
                for i in range(s_plus.spaces[1].spaces[0].nbasis):
                    indice_minus = loca2global([i_minus,1,i,n_deg_minus-1],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,1,i,0],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
                    if mom_pres:
                        for p in range(0,py+1):
                            #correction
                            indice_plus_j  = loca2global([i_plus,  1, i, p+2                ], n_patches,patch_shape_plus)
                            indice_minus_j = loca2global([i_minus, 1, i, n_deg_minus-1-(p+2)], n_patches,patch_shape_minus)
                            indice_plus    = loca2global([i_plus,  1, i, 0                  ], n_patches,patch_shape_plus)
                            indice_minus   = loca2global([i_minus, 1, i, n_deg_minus-1      ], n_patches,patch_shape_minus)
                            Proj[indice_plus_j,indice_plus]+=Correct_coef_y_0[p]/2
                            Proj[indice_plus_j,indice_minus]-=Correct_coef_y_0[p]/2
                            Proj[indice_minus_j,indice_plus]-=Correct_coef_y_0[p]/2
                            Proj[indice_minus_j,indice_minus]+=Correct_coef_y_0[p]/2

    elif V1.name=="Hcurl":

        # on each d=axis we use the 1D projection on B(x_d) splines along x_{d+1} direction 
        # spline degrees: 
        # py = patch_space.spaces[1].degree[0] for axis = 0
        # px = patch_space.spaces[0].degree[1] for axis = 1
        proj_degs = [py,px]
        a_same = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # a_same[axis]: coefs of P B0 on same patch
        a_ngbr = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # a_ngbr[axis]: coefs of P B0 on neighbor patch
        b_same = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # b_same[axis]: coefs of P B1 on same patch
        b_ngbr = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # b_ngbr[axis]: coefs of P B1 on neighbor patch
        corr_coef_0 = [Correct_coef_y_0,Correct_coef_x_0]
        corr_coef_1 = [Correct_coef_y_1,Correct_coef_x_1]

        for axis in [0,1] :
            p_ax = proj_degs[axis]
            a_sm = a_same[axis]
            a_nb = a_ngbr[axis]
            b_sm = b_same[axis]
            b_nb = b_ngbr[axis]
            # projection coefs:
            a_sm[0] = 1/2
            if proj_op == 0:
                # new slope is average of old ones
                a_sm[1] = 0  
            elif proj_op == 1:
                # new slope is average of old ones after averaging of interface coef
                a_sm[1] = 1/2
            elif proj_op == 2:
                # new slope is average of reconstructed ones using local values and slopes
                a_sm[1] = 1/(2*p_ax)
            else:
                # just to try something else
                a_sm[1] = proj_op/2
                
            b_sm[0] = 0
            b_sm[1] = 1/2
            # C1 conformity + proj constraints:
            a_nb[0] = a_sm[0]
            a_nb[1] = 2*a_sm[0] - a_sm[1]
            b_nb[0] = b_sm[0]
            b_nb[1] = 2*b_sm[0] - b_sm[1]
            if mom_pres:
                cc_0_ax = corr_coef_0[axis]
                cc_1_ax = corr_coef_1[axis]
                for p in range(0,p_ax+1):
                    # correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                    # modified by the C1 enforcement
                    a_sm[p+2] = (1-a_sm[0]) * cc_0_ax[p]     -a_sm[1]  * cc_1_ax[p]
                    b_sm[p+2] =   -b_sm[0]  * cc_0_ax[p] + (1-b_sm[1]) * cc_1_ax[p]
                    # proj constraints:
                    b_nb[p+2] = b_sm[p+2]
                    a_nb[p+2] = -(a_sm[p+2] + 2*b_sm[p+2])

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

            # defining functions glob_ind_minus(i,j) and glob_ind_plus(i,j):
            # to return global index of local (relative) multi-index on minus/plus patch
            #   i: index along interface
            #   j: index across interface (relative: 0 is at interface)
            
            if axis == 0:
                def glob_ind_minus(i,j):
                    return loca2global([i_minus,1,n_deg_minus-1-j,i], n_patches, patch_shape_minus)
                def glob_ind_plus(i,j):
                    return loca2global([i_plus, 1,              j,i], n_patches, patch_shape_plus)

            elif axis == 1:
                def glob_ind_minus(i,j):
                    return loca2global([i_minus,0,i,n_deg_minus-1-j], n_patches, patch_shape_minus)

                def glob_ind_plus(i,j):
                    return loca2global([i_plus, 0,i,              j], n_patches, patch_shape_plus)
                
            else:
                raise ValueError(axis)
                        
            a_sm = a_same[axis]
            a_nb = a_ngbr[axis]
            b_sm = b_same[axis]
            b_nb = b_ngbr[axis]

            for i in range(s_plus.spaces[naxis].spaces[naxis].nbasis):

                indice_minus_1 = glob_ind_minus(i,1)
                indice_minus   = glob_ind_minus(i,0)
                indice_plus    = glob_ind_plus (i,0)
                indice_plus_1  = glob_ind_plus (i,1)
                
                #changing these coefficients ensure C1 continuity at the interface                                    
                Proj[indice_minus,indice_minus] += (a_sm[0]-1)
                Proj[indice_plus, indice_plus ] += (a_sm[0]-1)
                Proj[indice_plus, indice_minus] += a_nb[0]
                Proj[indice_minus,indice_plus ] += a_nb[0]

                # note: b_sm[0] = b_nb[0] is hard coded here
                Proj[indice_minus_1,indice_minus_1] += (b_sm[1]-1)
                Proj[indice_minus_1,indice_minus  ] += a_sm[1]    
                Proj[indice_minus_1,indice_plus   ] += a_nb[1]
                Proj[indice_minus_1,indice_plus_1 ] += b_nb[1]
                
                Proj[indice_plus_1,indice_minus_1]  += b_nb[1]      
                Proj[indice_plus_1,indice_minus  ]  += a_nb[1]      
                Proj[indice_plus_1,indice_plus   ]  += a_sm[1]      
                Proj[indice_plus_1,indice_plus_1 ]  += (b_sm[1]-1) 
                
                # print(' NEW -- axis = ', axis, 'patches: ', i_minus, i_plus)
                # for i1 in [indice_minus_1, indice_minus, indice_plus, indice_plus_1]:
                #     for i2 in [indice_minus_1, indice_minus, indice_plus, indice_plus_1]:
                #         print(i1, i2, ' : ', Proj[i1,i2])
                
                for p in range(0,py+1):
                    #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                    #modified by the C1 enforcement
                    indice_minus_j = glob_ind_minus (i,p+2)
                    indice_plus_j  = glob_ind_plus (i,p+2)
                    
                    Proj[indice_minus_j,indice_minus]   += a_sm[p+2]
                    Proj[indice_plus_j, indice_plus ]   += a_sm[p+2]
                    Proj[indice_plus_j, indice_minus]   += a_nb[p+2]
                    Proj[indice_minus_j,indice_plus ]   += a_nb[p+2]

                    Proj[indice_minus_j,indice_minus_1] += b_sm[p+2]
                    Proj[indice_plus_j,indice_plus_1]   += b_sm[p+2]
                    Proj[indice_minus_j,indice_plus_1]  += b_nb[p+2]
                    Proj[indice_plus_j,indice_minus_1]  += b_nb[p+2]

                    # print(' - MP - ')
                    # for i1 in [indice_minus_j, indice_plus_j]:
                    #     for i2 in [indice_minus_1, indice_minus, indice_plus, indice_plus_1]:
                    #         print(i1, i2, ' : ', Proj[i1,i2])

                # if axis == 1: exit()

            # elif axis == 1 :
            #     # gamma = 1  # smoother than 0 ?
            #     for i in range(s_plus.spaces[0].spaces[0].nbasis):
            #         indice_minus = loca2global([i_minus,0,i,n_deg_minus-1],n_patches,patch_shape_minus)
            #         indice_plus  = loca2global([i_plus,0,i,0],n_patches,patch_shape_plus)
            #         indice_minus_1 = loca2global([i_minus,0,i,n_deg_minus-2],n_patches,patch_shape_minus)
            #         indice_plus_1  = loca2global([i_plus,0,i,1],n_patches,patch_shape_plus)
            #         #changing this coefficients ensure C1 continuity at the interface
            #         Proj[indice_minus,indice_minus]-=1/2
            #         Proj[indice_plus,indice_plus]-=1/2
            #         Proj[indice_plus,indice_minus]+=1/2
            #         Proj[indice_minus,indice_plus]+=1/2
            #         # Proj[indice_minus,indice_minus] =0
            #         # Proj[indice_plus,indice_plus]=0
            #         # Proj[indice_plus,indice_minus]=0
            #         # Proj[indice_minus,indice_plus]=0
            #         Proj[indice_minus_1,indice_minus_1] -= 1/2
            #         Proj[indice_minus_1,indice_minus  ]  =  gamma/2
            #         Proj[indice_minus_1,indice_plus   ]  = (2 - gamma)/2
            #         Proj[indice_minus_1,indice_plus_1 ]  = -1/2
                    
            #         Proj[indice_plus_1,indice_minus_1]  = -1/2
            #         Proj[indice_plus_1,indice_minus  ]  = (2 + gamma)/2
            #         Proj[indice_plus_1,indice_plus   ]  = -gamma/2
            #         Proj[indice_plus_1,indice_plus_1 ] -= 1/2

            #         # Proj[indice_minus_1,indice_minus_1] = 0
            #         # Proj[indice_plus_1,indice_plus_1]-=1/2
            #         # Proj[indice_plus_1,indice_plus_1] = 0
            #         # Proj[indice_minus_1,indice_plus] = 0 
            #         # Proj[indice_plus_1,indice_minus] =0 #+=1
            #         # Proj[indice_plus_1,indice_minus]=0
            #         # Proj[indice_minus_1,indice_plus_1] = 0
            #         # Proj[indice_plus_1,indice_minus_1]-=1/2
            #         # Proj[indice_plus_1,indice_minus_1]=0
            #         if mom_pres:
            #             for p in range(0,px+1):
            #                 #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
            #                 #modified by the C1 enforcement
            #                 indice_plus_i  = loca2global([i_plus,  0, i, p+2                ], n_patches,patch_shape_plus)
            #                 indice_minus_i = loca2global([i_minus, 0, i, n_deg_minus-1-(p+2)], n_patches,patch_shape_minus)
            #                 Proj[indice_plus_i,indice_plus]+=Correct_coef_x_0[p]/2
            #                 Proj[indice_plus_i,indice_minus]-=Correct_coef_x_0[p]/2
            #                 Proj[indice_minus_i,indice_plus]-=Correct_coef_x_0[p]/2
            #                 Proj[indice_minus_i,indice_minus]+=Correct_coef_x_0[p]/2
            #                 Proj[indice_minus_i,indice_minus_1]+=Correct_coef_x_1[p]/2
            #                 Proj[indice_plus_i,indice_plus_1]+=Correct_coef_x_1[p]/2
            #                 Proj[indice_minus_i,indice_plus]-=Correct_coef_x_1[p]
            #                 Proj[indice_plus_i,indice_minus]-=Correct_coef_x_1[p]
            #                 Proj[indice_minus_i,indice_plus_1]+=Correct_coef_x_1[p]/2
            #                 Proj[indice_plus_i,indice_minus_1]+=Correct_coef_x_1[p]/2

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
                        #set bnd dof to zero
                        Proj[indice,indice]-=1
                        if mom_pres:
                            for p in range(0,py+1):
                            #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                            #modified by the C1 enforcement
                                if ext ==+1 :
                                    indice_i = loca2global([i_patch,1,n_deg-1-(p+1),i],n_patches,patch_shape)
                                elif ext == -1 : 
                                    indice_i = loca2global([i_patch,1,p+1,i],n_patches,patch_shape)
                                else :
                                    ValueError("wrong value for ext")
                                Proj[indice_i,indice]+=Correct_coef_y_bnd[p]
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
                        if mom_pres:
                            for p in range(0,px+1):
                            #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                            #modified by the C1 enforcement
                                if ext ==+1 :
                                    indice_i = loca2global([i_patch,0,i,n_deg-1-(p+1)],n_patches,patch_shape)
                                elif ext == -1 : 
                                    indice_i = loca2global([i_patch,0,i,p+1],n_patches,patch_shape)
                                else :
                                    ValueError("wrong value for ext")
                                Proj[indice_i,indice]+=Correct_coef_x_bnd[p]
                            #Proj[indice_1,indice_1]-=1/2
    else :
        print("Error in Conf_proj_1 : wrong kind of space")

    Proj_par = sparse_eye(dim_tot,format="lil")

    enddom = breakpoints_x_x[-1]
    begdom = breakpoints_x_x[0]
    denom = enddom-begdom
    #Direction x (average x component)
    Mass_mat = np.zeros((px+1,local_shape[0][0]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(px+1):
        for ie1 in range(Nel[0]):   #loop on cells
            cell_size = breakpoints_x_x[ie1+1]-breakpoints_x_x[ie1]
            for il1 in range(degree_x[0]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[0]): #loops on quadrature points
                    v0 = basis_x[0][ie1,il1,0,q1]
                    x  = grid[0][ie1,q1]
                    val += cell_size*0.5*w[0][q1]*v0*((x-begdom)/denom)**poldeg
                locindx=span_x[0][ie1]-degree_x[0]+il1
                Mass_mat[poldeg,locindx]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:px+2]
    Mat_to_inv_bnd = Mass_mat[:,1:px+2]

    Correct_coef_x = np.linalg.solve(Mat_to_inv,Rhs) #coefficient to ensure moment preservation due to modification of first coeff

    Correct_coef_x_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs)

    enddom = breakpoints_x_y[-1]
    begdom = breakpoints_x_y[0]
    denom = enddom-begdom
    #Direction x (average y component)
    Mass_mat = np.zeros((py+1,local_shape[1][1]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(py+1):
        for ie1 in range(Nel[1]):   #loop on cells
            cell_size = breakpoints_x_y[ie1+1]-breakpoints_x_y[ie1]
            for il1 in range(degree_y[1]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis_y[1][ie1,il1,0,q1]
                    y  = grid[1][ie1,q1]
                    val += cell_size/2*w[1][q1]*v0*((y-begdom)/denom)**poldeg 
                locindy=span_y[1][ie1]-degree_y[1]+il1
                Mass_mat[poldeg,locindy]+=val
                
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:py+2]
    Mat_to_inv_bnd = Mass_mat[:,1:py+2]

    Correct_coef_y = np.linalg.solve(Mat_to_inv,Rhs)

    Correct_coef_y_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs)

    for I in Interfaces:
        axis = I.axis
        i_minus = get_patch_index_from_face(domain, I.minus)
        i_plus  = get_patch_index_from_face(domain, I.plus )
        s_minus = V1h.spaces[i_minus]
        s_plus  = V1h.spaces[i_plus]
        naxis = (axis+1)%2
        patch_shape_minus = [[s_minus.spaces[0].spaces[0].nbasis,s_minus.spaces[0].spaces[1].nbasis],
                            [s_minus.spaces[1].spaces[0].nbasis,s_minus.spaces[1].spaces[1].nbasis]]
        patch_shape_plus  = [[s_plus.spaces[0].spaces[0].nbasis,s_plus.spaces[0].spaces[1].nbasis],
                            [s_plus.spaces[1].spaces[0].nbasis,s_plus.spaces[1].spaces[1].nbasis]]
        n_deg_minus = s_minus.spaces[axis].spaces[axis].nbasis
        if axis == 0 :
            for i in range(s_plus.spaces[0].spaces[1].nbasis):
                indice_minus = loca2global([i_minus,  0,n_deg_minus-1,i],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,   0,   0,i],n_patches,patch_shape_plus)
                #changing this coefficients ensure C0 continuity at the interface
                Proj_par[indice_minus,indice_minus]-=1/2
                Proj_par[indice_plus,indice_plus]-=1/2
                Proj_par[indice_plus,indice_minus]+=1/2
                Proj_par[indice_minus,indice_plus]+=1/2
                if mom_pres:
                    for p in range(0,px+1):
                        #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                        #modified by the C0 enforcement
                        indice_plus_j  = loca2global([i_plus,  0, p+1,                 i], n_patches,patch_shape_plus)
                        indice_minus_j = loca2global([i_minus, 0, n_deg_minus-1-(p+1), i], n_patches,patch_shape_minus)

                        Proj_par[indice_plus_j,indice_plus]+=Correct_coef_x[p]/2
                        Proj_par[indice_plus_j,indice_minus]-=Correct_coef_x[p]/2
                        Proj_par[indice_minus_j,indice_plus]-=Correct_coef_x[p]/2
                        Proj_par[indice_minus_j,indice_minus]+=Correct_coef_x[p]/2
        elif axis == 1 :
            for i in range(s_plus.spaces[1].spaces[0].nbasis):
                indice_minus = loca2global([i_minus,  1,i,n_deg_minus-1],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,   1, i,  0,],n_patches,patch_shape_plus)
                #changing this coefficients ensure C0 continuity at the interface
                Proj_par[indice_minus,indice_minus]-=1/2
                Proj_par[indice_plus,indice_plus]-=1/2
                Proj_par[indice_plus,indice_minus]+=1/2
                Proj_par[indice_minus,indice_plus]+=1/2
                if mom_pres:
                    for p in range(0,py+1):
                        #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                        #modified by the C0 enforcement
                        indice_plus_j  = loca2global([i_plus,  1, i, p+1                ], n_patches,patch_shape_plus)
                        indice_minus_j = loca2global([i_minus, 1, i, n_deg_minus-1-(p+1)], n_patches,patch_shape_minus)

                        Proj_par[indice_plus_j,indice_plus]+=Correct_coef_y[p]/2
                        Proj_par[indice_plus_j,indice_minus]-=Correct_coef_y[p]/2
                        Proj_par[indice_minus_j,indice_plus]-=Correct_coef_y[p]/2
                        Proj_par[indice_minus_j,indice_minus]+=Correct_coef_y[p]/2

    print((Proj@Proj_par-Proj_par@Proj))

    assert(np.allclose((Proj@Proj_par).todense(),(Proj_par@Proj).todense()))

    return Proj@Proj_par


def smooth_1d_V0_Cr(V0h, conf_axis, reg=0, p_moments=-1, nquads=None, hom_bc=False):
    """Create the matrix enforcing Cr continuity with moment preservation in one direction for the V0 space
    
    Parameters
    ----------
    V0h : The discrete broken V0 (or V2) space in which we will compute the projection

    conf_axis : axis along which conformity is imposed

    reg : order of imposed continuity (-1: no continuity, 0 or 1) 

    p_moments : degree of polynomial moments to preserve (-1 for none)

    nquads : number of integration points to compute the moment preserving weights

    hom_bc : Wether or not enforce homogeneous boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
    dim_tot = V0h.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    if reg < 0:
        return Proj
    
    assert reg in [0,1]

    V0      = V0h.symbolic_space
    domain  = V0.domain
    Interfaces  = domain.interfaces
    boundary = domain.boundary
    n_patches = len(V0h.spaces)

    patch_space = V0h.spaces[0]
    local_shape = [patch_space.spaces[0].nbasis,patch_space.spaces[1].nbasis]
    # p      = patch_space.degree                     # spline degrees
    Nel    = patch_space.ncells                     # number of elements
    degree = patch_space.degree
    breakpoints_xy = [breakpoints(patch_space.knots[axis],degree[axis]) for axis in range(2)]
    # breakpoints_x = breakpoints(patch_space.knots[0],degree[0])
    # breakpoints_y = breakpoints(patch_space.knots[1],degree[1])

    if nquads is None:
        # default: Gauss-Legendre quadratures should be exact for polynomials of deg ≤ 2*degree
        nquads = [ degree[axis]+1 for axis in range(2)]

    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]

    grid = [np.array([deepcopy((0.5*(u[axis]+1)*(breakpoints_xy[axis][i+1]-breakpoints_xy[axis][i])+breakpoints_xy[axis][i])) 
                      for i in range(Nel[axis])])
            for axis in range(2)]
    _, basis, span, _ = patch_space.preprocess_regular_tensor_grid(grid,der=1)  # todo: why not der=0 ?

    span = [deepcopy(span[k] + patch_space.vector_space.starts[k] - patch_space.vector_space.shifts[k] * patch_space.vector_space.pads[k]) for k in range(2)]
    p_axis = degree[conf_axis]
    enddom = breakpoints_xy[conf_axis][-1]
    begdom = breakpoints_xy[conf_axis][0]
    denom = enddom-begdom
    
    a_sm = np.zeros(p_moments+2+reg)   # coefs of P B0 on same patch
    a_nb = np.zeros(p_moments+2+reg)   # coefs of P B0 on neighbor patch
    
    # projection coefs:
    a_sm[0] = 1/2
    a_nb[0] = a_sm[0]

    if reg > 0:
        b_sm = np.zeros(p_moments+3)   # coefs of P B1 on same patch
        b_nb = np.zeros(p_moments+3)   # coefs of P B1 on neighbor patch
        if proj_op == 0:
            # new slope is average of old ones
            a_sm[1] = 0  
        elif proj_op == 1:
            # new slope is average of old ones after averaging of interface coef
            a_sm[1] = 1/2
        elif proj_op == 2:
            # new slope is average of reconstructed ones using local values and slopes
            a_sm[1] = 1/(2*p_axis)
        else:
            # just to try something else
            a_sm[1] = proj_op/2

        a_nb[1] = 2*a_sm[0] - a_sm[1]
        b_sm[0] = 0
        b_sm[1] = 1/2
        b_nb[0] = b_sm[0]
        b_nb[1] = 2*b_sm[0] - b_sm[1]
    
    if p_moments >= 0:
        # to preserve moments of degree p we need 1+p conforming basis functions in the patch (the "interior" ones)
        # and for the given regularity constraint, there are local_shape[conf_axis]-2*(1+reg) such conforming functions 
        if local_shape[conf_axis]-2*(1+reg) < 1+p_moments:
            raise ValueError(f"patch space is too small to preserve moments of degree {p_moments}")
        # computing the contribution to every moment of the differents basis function
        # for simplicity we assemble the full matrix with all basis functions (ok if patches not too large)
        Mass_mat = np.zeros((p_moments+1,local_shape[conf_axis]))    
        for poldeg in range(p_moments+1):
            for ie1 in range(Nel[conf_axis]):   #loop on cells
                for il1 in range(p_axis+1): #loops on basis function in each cell
                    val=0.
                    for q1 in range(nquads[conf_axis]): #loops on quadrature points
                        v0 = basis[conf_axis][ie1,il1,0,q1]
                        x  = grid[conf_axis][ie1,q1]
                        val += w[conf_axis][q1]*v0*((enddom-x)/denom)**poldeg
                    locind=span[conf_axis][ie1]-p_axis+il1
                    Mass_mat[poldeg,locind]+=val
        Rhs_0 = Mass_mat[:,0]
        if reg == 0:
            Mat_to_inv = Mass_mat[:,1:p_moments+2]
        else:
            Mat_to_inv = Mass_mat[:,2:p_moments+3]
        Correct_coef_0 = np.linalg.solve(Mat_to_inv,Rhs_0)    
        cc_0_ax = Correct_coef_0
        
        if reg > 0:
            Rhs_1 = Mass_mat[:,1]
            Correct_coef_1 = np.linalg.solve(Mat_to_inv,Rhs_1)    
            cc_1_ax = Correct_coef_1

        if hom_bc:
            # homogeneous bc is on the point value: no constraint on the derivatives
            # so only the projection of B0 (to 0) has to be corrected
            Mat_to_inv_bnd = Mass_mat[:,1:p_moments+2]
            Correct_coef_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)

        for p in range(0,p_moments+1):
            # correction for moment preserving : 
            # we use the first p_moments+1 conforming ("interior") functions to preserve the p+1 moments
            # modified by the C0 or C1 enforcement
            if reg == 0:
                a_sm[p+1] = (1-a_sm[0]) * cc_0_ax[p]
                # proj constraint:
                a_nb[p+1] = -a_sm[p+1]
            
            else:
                a_sm[p+2] = (1-a_sm[0]) * cc_0_ax[p]     -a_sm[1]  * cc_1_ax[p]
                b_sm[p+2] =   -b_sm[0]  * cc_0_ax[p] + (1-b_sm[1]) * cc_1_ax[p]
                
                # proj constraint:
                b_nb[p+2] = b_sm[p+2]
                a_nb[p+2] = -(a_sm[p+2] + 2*b_sm[p+2])

    for I in Interfaces:
        axis = I.axis
        i_minus = get_patch_index_from_face(domain, I.minus)
        i_plus  = get_patch_index_from_face(domain, I.plus )
        s_minus = V0h.spaces[i_minus]
        s_plus  = V0h.spaces[i_plus]
        n_deg_minus = s_minus.spaces[axis].nbasis
        n_deg_plus = s_plus.spaces[axis].nbasis
        patch_shape_minus = [s_minus.spaces[0].nbasis,s_minus.spaces[1].nbasis]
        patch_shape_plus  = [s_plus.spaces[0].nbasis,s_plus.spaces[1].nbasis]
        
        if axis == conf_axis :

            # loop over dofs on the interface (assuming same grid on plus and minus patches)
            for i in range(s_plus.spaces[1-conf_axis].nbasis):

                indice_minus = glob_ind_interface_scalfield(
                    i,0,
                    side=-1,k_patch=i_minus,
                    axis=conf_axis,
                    patch_shape=patch_shape_minus,
                    n_patches=n_patches,
                    nb_dofs_across=n_deg_minus)
                
                indice_plus = glob_ind_interface_scalfield(
                    i,0,
                    side=+1,k_patch=i_plus,
                    axis=conf_axis,
                    patch_shape=patch_shape_plus,
                    n_patches=n_patches,
                    nb_dofs_across=n_deg_plus)

                Proj[indice_minus,indice_minus] += (a_sm[0]-1)
                Proj[indice_plus, indice_plus ] += (a_sm[0]-1)
                Proj[indice_plus, indice_minus] += a_nb[0]
                Proj[indice_minus,indice_plus ] += a_nb[0]

                if reg > 0:

                    indice_minus_1 = glob_ind_interface_scalfield(
                        i,1,
                        side=-1,k_patch=i_minus,
                        axis=conf_axis,
                        patch_shape=patch_shape_minus,
                        n_patches=n_patches,
                        nb_dofs_across=n_deg_minus)
                    
                    indice_plus_1 = glob_ind_interface_scalfield(
                        i,1,
                        side=+1,k_patch=i_plus,
                        axis=conf_axis,
                        patch_shape=patch_shape_plus,
                        n_patches=n_patches,
                        nb_dofs_across=n_deg_plus)

                    # note: b_sm[0] = b_nb[0] is hard coded
                    Proj[indice_minus_1,indice_minus_1] += (b_sm[1]-1)   # todo: replace these '+=' with '=' (except for diag terms)
                    Proj[indice_minus_1,indice_minus  ] += a_sm[1]    
                    Proj[indice_minus_1,indice_plus   ] += a_nb[1]
                    Proj[indice_minus_1,indice_plus_1 ] += b_nb[1]
                    
                    Proj[indice_plus_1,indice_minus_1]  += b_nb[1]      
                    Proj[indice_plus_1,indice_minus  ]  += a_nb[1]      
                    Proj[indice_plus_1,indice_plus   ]  += a_sm[1]      
                    Proj[indice_plus_1,indice_plus_1 ]  += (b_sm[1]-1) 

                # if mom_pres:
                for p in range(0,p_moments+1):
                    # correction for moment preservation: modify the projection of the interface functions with the interior ones
                    j = p+reg+1 # index of interior function (relative to interface)
                    indice_minus_j = glob_ind_interface_scalfield(
                        i,j,
                        side=-1,k_patch=i_minus,
                        axis=conf_axis,
                        patch_shape=patch_shape_minus,
                        n_patches=n_patches,
                        nb_dofs_across=n_deg_minus)
                    
                    indice_plus_j = glob_ind_interface_scalfield(
                        i,j,
                        side=+1,k_patch=i_plus,
                        axis=conf_axis,
                        patch_shape=patch_shape_plus,
                        n_patches=n_patches,
                        nb_dofs_across=n_deg_plus)

                    # if conf_axis == 0:
                    #     indice_minus_j = loca2global([i_minus, n_deg_minus-1-(p+reg+1), i], n_patches,patch_shape_minus)
                    #     indice_plus_j  = loca2global([i_plus,  p+reg+1,                 i], n_patches,patch_shape_plus)                    
                    # else:
                    #     indice_minus_j = loca2global([i_minus, i, n_deg_minus-1-(p+reg+1)], n_patches,patch_shape_minus)
                    #     indice_plus_j  = loca2global([i_plus,  i, p+reg+1                ], n_patches,patch_shape_plus)                    
                    
                    Proj[indice_minus_j,indice_minus]   += a_sm[j]
                    Proj[indice_plus_j, indice_plus ]   += a_sm[j]
                    Proj[indice_plus_j, indice_minus]   += a_nb[j]
                    Proj[indice_minus_j,indice_plus ]   += a_nb[j]

                    if reg > 0:
                        Proj[indice_minus_j,indice_minus_1] += b_sm[j]
                        Proj[indice_plus_j,indice_plus_1]   += b_sm[j]
                        Proj[indice_minus_j,indice_plus_1]  += b_nb[j]
                        Proj[indice_plus_j,indice_minus_1]  += b_nb[j]

    if hom_bc:
        for b in boundary : 
            axis = b.axis
            ext = b.ext
            i_patch = get_patch_index_from_face(domain,b)
            space = V0h.spaces[i_patch]
            n_deg = space.spaces[1-axis].nbasis
            patch_shape = [space.spaces[0].nbasis,space.spaces[1].nbasis]
            if axis == conf_axis:
                # a_sm = a_same[axis]
                # a_nb = a_ngbr[axis]
                # b_sm = b_same[axis]
                # b_nb = b_ngbr[axis]

                for i in range(space.spaces[1-axis].nbasis):
                    indice = glob_ind_interface_scalfield(i,0,side=-ext,k_patch=i_patch,
                                                          axis=axis,
                                                          patch_shape=patch_shape,
                                                          n_patches=n_patches,
                                                          nb_dofs_across=n_deg)
                    # Proj[indice,indice] -= 1
                    Proj[indice,indice] = 0
                    # # print(f'({axis},{ext})-boundary-diag: Proj[{indice},{indice}] = {Proj[indice,indice]}')
                    # if( abs(Proj[indice,indice]) > 1e-10 ):
                    #     print(f'STRANGE (x): Proj[indice,indice] = {Proj[indice,indice]} ... ?')

                    for p in range(0,p_moments+1):
                        # correction (with only 1 interface function on boundary)
                        j = p+1
                        indice_j = glob_ind_interface_scalfield(
                            i,j,
                            side=-ext,k_patch=i_patch,
                            axis=axis,
                            patch_shape=patch_shape,
                            n_patches=n_patches,
                            nb_dofs_across=n_deg)

                        Proj[indice_j,indice] += Correct_coef_bnd[p]
                        
                        # print(f'({axis},{ext})-boundary-d+{p}: Proj[{indice_j},{indice}] = {Proj[indice_j,indice]}')

    return Proj


def smooth_1d_V1_Cr(V1h, conf_axis, conf_comp, reg=0, p_moments=-1, nquads=None, hom_bc=False):
    """Create the matrix enforcing Cr continuity with moment preservation in one direction for a component of the V1 space
    
    Parameters
    ----------
    V1h : The discrete broken space (vector-valued) in which we will compute the projection

    conf_axis : axis along which conformity is imposed

    conf_comp : component for which conformity is imposed

    reg : order of imposed continuity (-1: no continuity, 0 or 1) 

    p_moments : degree of polynomial moments to preserve (-1 for none)

    nquads : number of integration points to compute the moment preserving weights

    hom_bc : Wether or not enforce homogeneous boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
    dim_tot = V1h.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    if reg < 0:
        return Proj
    
    assert reg in [0,1]

    V1      = V1h.symbolic_space
    domain  = V1.domain
    Interfaces  = domain.interfaces
    boundary = domain.boundary
    n_patches = len(V1h.spaces)

    patch_space = V1h.spaces[0]
    # local_shape = [patch_space.spaces[axis].nbasis,patch_space.spaces[axis].nbasis]
    local_shape = [[patch_space.spaces[comp].spaces[axis].nbasis 
                    for axis in range(2)] for comp in range(2)]
    # p      = patch_space.degree                     # spline degrees
    Nel    = patch_space.ncells                     # number of elements
    patch_space_x, patch_space_y = [patch_space.spaces[comp] for comp in range(2)]
    #  = patch_space.spaces[1]
    # degree_x = patch_space_x.degree
    # degree_y = patch_space_y.degree
    p_comp_axis = patch_space.spaces[conf_comp].degree[conf_axis]
    degree = patch_space.degree  # ? which degree is this ?? see below
    print(f'degree[conf_comp][conf_axis] = {degree[conf_comp][conf_axis]}')
    print(f'p_comp_axis = patch_space.spaces[conf_comp].degree[conf_axis] = {patch_space.spaces[conf_comp].degree[conf_axis]}')
    assert p_comp_axis == degree[conf_comp][conf_axis]

    breaks_comp_axis = [[breakpoints(patch_space.spaces[comp].knots[axis],degree[comp][axis])
                              for axis in range(2)] for comp in range(2)]
    # breakpoints_x_y = breakpoints(patch_space_x.knots[1],degree_x[1])
    # breakpoints_xy = [breakpoints(patch_space.knots[axis],degree[axis]) for axis in range(2)]
    # breakpoints_x = breakpoints(patch_space.knots[0],degree[0])
    # breakpoints_y = breakpoints(patch_space.knots[1],degree[1])
    
    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]
    
    # grid = [np.array([deepcopy((0.5*(u[0]+1)*(breakpoints_x_x[i+1]-breakpoints_x_x[i])+breakpoints_x_x[i])) for i in range(Nel[0])]),
    #         np.array([deepcopy((0.5*(u[1]+1)*(breakpoints_x_y[i+1]-breakpoints_x_y[i])+breakpoints_x_y[i])) for i in range(Nel[1])])]
    grid = [np.array([deepcopy((0.5*(u[axis]+1)*(breaks_comp_axis[0][axis][i+1]-breaks_comp_axis[0][axis][i])+breaks_comp_axis[0][axis][i])) 
                      for i in range(Nel[axis])])
            for axis in range(2)]

    _, basis_x, span_x, _ = patch_space_x.preprocess_regular_tensor_grid(grid,der=0)
    _, basis_y, span_y, _ = patch_space_y.preprocess_regular_tensor_grid(grid,der=0)
    span_x = [deepcopy(span_x[k] + patch_space_x.vector_space.starts[k] - patch_space_x.vector_space.shifts[k] * patch_space_x.vector_space.pads[k]) for k in range(2)]
    span_y = [deepcopy(span_y[k] + patch_space_y.vector_space.starts[k] - patch_space_y.vector_space.shifts[k] * patch_space_y.vector_space.pads[k]) for k in range(2)]
    # px=degree_x[1]
    basis = [basis_x, basis_y]
    span = [span_x, span_y]
    enddom = breaks_comp_axis[0][0][-1]
    begdom = breaks_comp_axis[0][0][0]
    denom = enddom-begdom

    # grid = [np.array([deepcopy((0.5*(u[axis]+1)*(breakpoints_xy[axis][i+1]-breakpoints_xy[axis][i])+breakpoints_xy[axis][i])) 
    #                   for i in range(Nel[axis])])
    #         for axis in range(2)]
    # _, basis, span, _ = patch_space.preprocess_regular_tensor_grid(grid,der=1)  # todo: why not der=0 ?

    if nquads is None:
        # default: Gauss-Legendre quadratures should be exact for polynomials of deg ≤ 2*degree
        nquads = [ degree[axis]+1 for axis in range(2)]

    # p_axis = degree[conf_axis]

    # projection coefficients
    a_sm = np.zeros(p_moments+2+reg)   # coefs of P B0 on same patch
    a_nb = np.zeros(p_moments+2+reg)   # coefs of P B0 on neighbor patch
    
    a_sm[0] = 1/2
    a_nb[0] = a_sm[0]

    if reg > 0:
        b_sm = np.zeros(p_moments+3)   # coefs of P B1 on same patch
        b_nb = np.zeros(p_moments+3)   # coefs of P B1 on neighbor patch
        if proj_op == 0:
            # new slope is average of old ones
            a_sm[1] = 0  
        elif proj_op == 1:
            # new slope is average of old ones after averaging of interface coef
            a_sm[1] = 1/2
        elif proj_op == 2:
            # new slope is average of reconstructed ones using local values and slopes
            a_sm[1] = 1/(2*p_comp_axis)
        else:
            # just to try something else
            a_sm[1] = proj_op/2

        a_nb[1] = 2*a_sm[0] - a_sm[1]
        b_sm[0] = 0
        b_sm[1] = 1/2
        b_nb[0] = b_sm[0]
        b_nb[1] = 2*b_sm[0] - b_sm[1]
    
    if p_moments >= 0:
        # to preserve moments of degree p we need 1+p conforming basis functions in the patch (the "interior" ones)
        # and for the given regularity constraint, there are local_shape[conf_comp][conf_axis]-2*(1+reg) such conforming functions 
        if local_shape[conf_comp][conf_axis]-2*(1+reg) < 1+p_moments:
            raise ValueError(f"patch space is too small to preserve moments of degree {p_moments}")
        # computing the contribution to every moment of the differents basis function
        # for simplicity we assemble the full matrix with all basis functions (ok if patches not too large)
        Mass_mat = np.zeros((p_moments+1,local_shape[conf_comp][conf_axis]))    
        for poldeg in range(p_moments+1):
            for ie1 in range(Nel[conf_axis]):   #loop on cells
                # cell_size = breaks_comp_axis[conf_comp][conf_axis][ie1+1]-breakpoints_x_y[ie1]  # todo: try without (probably not needed
                for il1 in range(p_comp_axis+1): #loops on basis function in each cell
                    val=0.
                    for q1 in range(nquads[conf_axis]): #loops on quadrature points
                        v0 = basis[conf_comp][conf_axis][ie1,il1,0,q1]
                        xd = grid[conf_axis][ie1,q1]
                        val += w[conf_axis][q1]*v0*((enddom-xd)/denom)**poldeg
                    locind=span[conf_comp][conf_axis][ie1]-p_comp_axis+il1
                    Mass_mat[poldeg,locind]+=val
        Rhs_0 = Mass_mat[:,0]
        if reg == 0:
            Mat_to_inv = Mass_mat[:,1:p_moments+2]
        else:
            Mat_to_inv = Mass_mat[:,2:p_moments+3]
        Correct_coef_0 = np.linalg.solve(Mat_to_inv,Rhs_0)    
        cc_0_ax = Correct_coef_0
        
        if reg > 0:
            Rhs_1 = Mass_mat[:,1]
            Correct_coef_1 = np.linalg.solve(Mat_to_inv,Rhs_1)    
            cc_1_ax = Correct_coef_1

        if hom_bc:
            # homogeneous bc is on the point value: no constraint on the derivatives
            # so only the projection of B0 (to 0) has to be corrected
            Mat_to_inv_bnd = Mass_mat[:,1:p_moments+2]
            Correct_coef_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)

        for p in range(0,p_moments+1):
            # correction for moment preserving : 
            # we use the first p_moments+1 conforming ("interior") functions to preserve the p+1 moments
            # modified by the C0 or C1 enforcement
            if reg == 0:
                a_sm[p+1] = (1-a_sm[0]) * cc_0_ax[p]
                # proj constraint:
                a_nb[p+1] = -a_sm[p+1]
            
            else:
                a_sm[p+2] = (1-a_sm[0]) * cc_0_ax[p]     -a_sm[1]  * cc_1_ax[p]
                b_sm[p+2] =   -b_sm[0]  * cc_0_ax[p] + (1-b_sm[1]) * cc_1_ax[p]
                
                # proj constraint:
                b_nb[p+2] = b_sm[p+2]
                a_nb[p+2] = -(a_sm[p+2] + 2*b_sm[p+2])

    for I in Interfaces:
        axis = I.axis
        if axis == conf_axis :

            i_minus = get_patch_index_from_face(domain, I.minus)
            i_plus  = get_patch_index_from_face(domain, I.plus )
            s_minus = V1h.spaces[i_minus]
            s_plus  = V1h.spaces[i_plus]
            # todo: rename n_deg_** -> nb_dofs_comp_axis ?
            nb_dofs_minus = s_minus.spaces[conf_comp].spaces[conf_axis].nbasis
            nb_dofs_plus  = s_plus.spaces[conf_comp].spaces[conf_axis].nbasis
            nb_dofs_face  = s_minus.spaces[conf_comp].spaces[1-conf_axis].nbasis
            # here we assume the same grid along the interface on plus and minus patches:
            assert nb_dofs_face == s_plus.spaces[conf_comp].spaces[1-conf_axis].nbasis

            # n_deg_minus = s_minus.spaces[axis].nbasis
            # n_deg_plus = s_plus.spaces[axis].nbasis
            # patch_shape_minus = [s_minus.spaces[0].nbasis,s_minus.spaces[1].nbasis]
            # patch_shape_plus  = [s_plus.spaces[0].nbasis,s_plus.spaces[1].nbasis]
            patch_shape_minus = [[s_minus.spaces[cp].spaces[ax].nbasis 
                                for ax in range(2)] for cp in range(2)]
            patch_shape_plus = [[s_minus.spaces[cp].spaces[ax].nbasis 
                                for ax in range(2)] for cp in range(2)]

            # loop over dofs on the interface
            for i in range(nb_dofs_face):
                # print(f'loop on face i = {i} for axis = {conf_axis}')
                # print(f'patch_shape_minus[conf_comp] = {patch_shape_minus[conf_comp]}')
                indice_minus = glob_ind_interface_vecfield(
                    i,0,
                    side=-1,k_patch=i_minus,
                    axis=conf_axis,
                    patch_shape=patch_shape_minus,
                    n_patches=n_patches,
                    nb_dofs_across=nb_dofs_minus,
                    comp=conf_comp)
                
                indice_plus = glob_ind_interface_vecfield(
                    i,0,
                    side=+1,k_patch=i_plus,
                    axis=conf_axis,
                    patch_shape=patch_shape_plus,
                    n_patches=n_patches,
                    nb_dofs_across=nb_dofs_plus,
                    comp=conf_comp)

                Proj[indice_minus,indice_minus] += (a_sm[0]-1)
                Proj[indice_plus, indice_plus ] += (a_sm[0]-1)
                Proj[indice_plus, indice_minus] += a_nb[0]
                Proj[indice_minus,indice_plus ] += a_nb[0]

                if reg > 0:

                    indice_minus_1 = glob_ind_interface_vecfield(
                        i,1,
                        side=-1,k_patch=i_minus,
                        axis=conf_axis,
                        patch_shape=patch_shape_minus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_minus,
                        comp=conf_comp)
                    
                    indice_plus_1 = glob_ind_interface_vecfield(
                        i,1,
                        side=+1,k_patch=i_plus,
                        axis=conf_axis,
                        patch_shape=patch_shape_plus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_plus,
                        comp=conf_comp)

                    # note: b_sm[0] = b_nb[0] is hard coded
                    Proj[indice_minus_1,indice_minus_1] += (b_sm[1]-1)   # todo: replace these '+=' with '=' (except for diag terms)
                    Proj[indice_minus_1,indice_minus  ] += a_sm[1]    
                    Proj[indice_minus_1,indice_plus   ] += a_nb[1]
                    Proj[indice_minus_1,indice_plus_1 ] += b_nb[1]
                    
                    Proj[indice_plus_1,indice_minus_1]  += b_nb[1]      
                    Proj[indice_plus_1,indice_minus  ]  += a_nb[1]      
                    Proj[indice_plus_1,indice_plus   ]  += a_sm[1]      
                    Proj[indice_plus_1,indice_plus_1 ]  += (b_sm[1]-1) 

                # if mom_pres:
                for p in range(0,p_moments+1):
                    # correction for moment preservation: modify the projection of the interface functions with the interior ones
                    j = p+reg+1 # index of interior function (relative to interface)
                    indice_minus_j = glob_ind_interface_vecfield(
                        i,j,
                        side=-1,k_patch=i_minus,
                        axis=conf_axis,
                        patch_shape=patch_shape_minus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_minus,
                        comp=conf_comp)
                    
                    indice_plus_j = glob_ind_interface_vecfield(
                        i,j,
                        side=+1,k_patch=i_plus,
                        axis=conf_axis,
                        patch_shape=patch_shape_plus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_plus,
                        comp=conf_comp)
                    
                    Proj[indice_minus_j,indice_minus]   += a_sm[j]
                    Proj[indice_plus_j, indice_plus ]   += a_sm[j]
                    Proj[indice_plus_j, indice_minus]   += a_nb[j]
                    Proj[indice_minus_j,indice_plus ]   += a_nb[j]

                    if reg > 0:
                        Proj[indice_minus_j,indice_minus_1] += b_sm[j]
                        Proj[indice_plus_j,indice_plus_1]   += b_sm[j]
                        Proj[indice_minus_j,indice_plus_1]  += b_nb[j]
                        Proj[indice_plus_j,indice_minus_1]  += b_nb[j]

    if hom_bc:
        for b in boundary : 
            axis = b.axis
            if axis == conf_axis:
                ext = b.ext
                i_patch = get_patch_index_from_face(domain,b)
                patch_space = V1h.spaces[i_patch]
                nb_dofs_axis = s_minus.spaces[conf_comp].spaces[conf_axis].nbasis
                nb_dofs_face = s_minus.spaces[conf_comp].spaces[1-conf_axis].nbasis
                patch_shape = [[patch_space.spaces[cp].spaces[ax].nbasis 
                                    for ax in range(2)] for cp in range(2)]

                # n_deg = space.spaces[1-axis].nbasis
                # patch_shape = [space.spaces[0].nbasis,space.spaces[1].nbasis]
            
                # a_sm = a_same[axis]
                # a_nb = a_ngbr[axis]
                # b_sm = b_same[axis]
                # b_nb = b_ngbr[axis]

                for i in range(nb_dofs_face):
                    indice = glob_ind_interface_vecfield(
                        i,0,
                        side=-ext,k_patch=i_patch,
                        axis=conf_axis,
                        patch_shape=patch_shape,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_axis,
                        comp=conf_comp)

                    Proj[indice,indice] = 0

                    for p in range(0,p_moments+1):
                        # correction (with only 1 interface function on boundary)
                        j = p+1
                        indice_j = glob_ind_interface_vecfield(
                            i,j,
                            side=-ext,k_patch=i_patch,
                            axis=conf_axis,
                            patch_shape=patch_shape,
                            n_patches=n_patches,
                            nb_dofs_across=nb_dofs_axis,
                            comp=conf_comp)

                        Proj[indice_j,indice] += Correct_coef_bnd[p]
                        
    return Proj

# todo: write a helper function that returns all conforming projections for a given sequence

def Conf_proj_0_c01(V0h, reg=0, p_moments=-1, nquads=None, hom_bc=False):
    """
    Create the matrix enforcing the C0 or C1 continuity with moment preservation for the V0 (or V2) space
    
    Parameters
    ----------
    V0h : The discrete broken V0 (or V2) space in which we will compute the projection

    reg : order of imposed continuity for both axes (-1: no continuity, 0 or 1) 

    p_moments : degree of polynomial moments to preserve (-1 for none)

    nquads : number of integration points to compute the moment preserving weights

    hom_bc : Wether or not enforce homogeneous boundary conditions
        
    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
    n_patches = len(V0h.spaces)
    if n_patches ==1:
        dim_tot = V0h.nbasis
        Proj_op    = IdentityOperator(V0h.vector_space)#sparse_eye(dim_tot,format="lil")
    else : 
        S_x=smooth_1d_V0_Cr(V0h, conf_axis=0, reg=reg, p_moments=p_moments, nquads=nquads, hom_bc=hom_bc)
        S_y=smooth_1d_V0_Cr(V0h, conf_axis=1, reg=reg, p_moments=p_moments, nquads=nquads, hom_bc=hom_bc)
        Proj=S_x@S_y
        Proj_op = Operator_from_scipy(V0h.vector_space,V0h.vector_space,Proj)
    return Proj_op.tosparse()



def Conf_proj_1_c01(V1h, reg=0, p_moments=-1, nquads=None, hom_bc=False):
    """
    Create the matrix enforcing the C0 or C1 continuity with moment preservation for the V1 space
    
    Parameters
    ----------
    V1h : The discrete broken V1 space in which we will compute the projection

    reg : order of imposed continuity (for the corresponding V0 space) (-1: no continuity, 0 or 1) 

    p_moments : degree of polynomial moments to preserve (-1 for none)

    nquads : number of integration points to compute the moment preserving weights

    hom_bc : Wether or not enforce homogeneous boundary conditions (in Hdiv or Hcurl) 

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
    V1      = V1h.symbolic_space
    dim_tot = V1h.nbasis
    # domain  = V1.domain
    n_patches = len(V1h.spaces)

    Proj = sparse_eye(dim_tot,format="lil")
    if n_patches ==1:
        if hom_bc:
            raise NotImplementedError("conf Proj on homogeneous bc not implemented for single patch case")
        else:
            pass 
    
    else : 
        if V1.name=="Hdiv":
            for axis in range(2):
                # in normal component: enforce C_reg smoothness and prescribed bc
                P_high = smooth_1d_V1_Cr(V1h, conf_axis=axis, conf_comp=axis, reg=reg, p_moments=p_moments, nquads=nquads, hom_bc=hom_bc)
                Proj = Proj @ P_high
                if reg > 0:
                    # in tangential component: enforce C_{reg-1} smoothness and no bc
                    P_low = smooth_1d_V1_Cr(V1h, conf_axis=axis, conf_comp=1-axis, reg=reg-1, p_moments=p_moments, nquads=nquads, hom_bc=False)
                    Proj = Proj @ P_low
        
        elif V1.name=="Hcurl":
            for axis in range(2):
                # in tangential component: enforce C_reg smoothness and prescribed bc
                P_high = smooth_1d_V1_Cr(V1h, conf_axis=axis, conf_comp=1-axis, reg=reg, p_moments=p_moments, nquads=nquads, hom_bc=hom_bc)
                Proj = Proj @ P_high
                if reg > 0:
                    # in normal component: enforce C_{reg-1} smoothness and no bc
                    P_low = smooth_1d_V1_Cr(V1h, conf_axis=axis, conf_comp=axis, reg=reg-1, p_moments=p_moments, nquads=nquads, hom_bc=False)
                    Proj = Proj @ P_low

        else:
            raise NotImplementedError(f"No conformity rule for V1.name = {V1.name}")                

    return Proj


# ===========================================================================================================
#
# temp version for a common C0 / C1 conf proj operator for V1h

def Conf_proj_1_c01_backup(V1h, reg, nquads, hom_bc):
    """
    Create the matrix enforcing C0 or C1 continuity between patches in the tangential (Hcurl) or normal (Hdiv) direction for the V1 space,
    with moment preservation 
    
    Parameters
    ----------
    V1h : The discrete broken V1 space in which we will compute the projection

    reg: the desired order of continuity at patch interfaces (0 or 1)

    nquads : number of integration points to compute the moment preserving weights

    hom_bc : Wether or not enforce boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """

    dim_tot = V1h.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    V1      = V1h.symbolic_space
    domain  = V1.domain
    n_patches = len(V1h.spaces)

    if n_patches ==1:
        dim_tot = V1h.nbasis
        Proj    = sparse_eye(dim_tot,format="lil")
        return Proj

    if reg > 1:
        print("[Conf_proj_1_c01] WARNING: high order regularity not implemented yet -- sticking to C1 conformity")
        reg = 1

    boundary = domain.boundary
    Interfaces  = domain.interfaces

    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]


    patch_space = V1h.spaces[0]
    local_shape = [[patch_space.spaces[0].spaces[0].nbasis,patch_space.spaces[0].spaces[1].nbasis],[patch_space.spaces[1].spaces[0].nbasis,patch_space.spaces[1].spaces[1].nbasis]]
    Nel    = patch_space.ncells                     # number of elements
    patch_space_x = patch_space.spaces[0]
    patch_space_y = patch_space.spaces[1]
    degree_x = patch_space_x.degree
    degree_y = patch_space_y.degree
    breakpoints_x_x = breakpoints(patch_space_x.knots[0],degree_x[0])
    breakpoints_x_y = breakpoints(patch_space_x.knots[1],degree_x[1])

    #compute grid for the correction coefficients ensuring moment preservation
    grid = [np.array([deepcopy((0.5*(u[0]+1)*(breakpoints_x_x[i+1]-breakpoints_x_x[i])+breakpoints_x_x[i])) for i in range(Nel[0])]),
            np.array([deepcopy((0.5*(u[1]+1)*(breakpoints_x_y[i+1]-breakpoints_x_y[i])+breakpoints_x_y[i])) for i in range(Nel[1])])]
    _, basis_x, span_x, _ = patch_space_x.preprocess_regular_tensor_grid(grid,der=0)
    _, basis_y, span_y, _ = patch_space_y.preprocess_regular_tensor_grid(grid,der=0)
    span_x = [deepcopy(span_x[k] + patch_space_x.vector_space.starts[k] - patch_space_x.vector_space.shifts[k] * patch_space_x.vector_space.pads[k]) for k in range(2)]
    span_y = [deepcopy(span_y[k] + patch_space_y.vector_space.starts[k] - patch_space_y.vector_space.shifts[k] * patch_space_y.vector_space.pads[k]) for k in range(2)]
    px=degree_x[1]
    enddom = breakpoints_x_x[-1]
    begdom = breakpoints_x_x[0]
    denom = enddom-begdom
    #Direction y (average x component)
    Mass_mat = np.zeros((px+1,local_shape[0][1]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(px+1):
        for ie1 in range(Nel[1]):   #loop on cells
            cell_size = breakpoints_x_y[ie1+1]-breakpoints_x_y[ie1]
            for il1 in range(px+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis_x[1][ie1,il1,0,q1]
                    x  = grid[1][ie1,q1]
                    val += cell_size*0.5*w[1][q1]*v0*((x-begdom)/denom)**poldeg
                locindx=span_x[1][ie1]-degree_x[1]+il1
                Mass_mat[poldeg,locindx]+=val
    Rhs_0 = Mass_mat[:,0]
    Rhs_1 = Mass_mat[:,1]
    Mat_to_inv = Mass_mat[:,2:px+3]
    Mat_to_inv_bnd = Mass_mat[:,1:px+2]

    Correct_coef_x_0 = np.linalg.solve(Mat_to_inv,Rhs_0) #coefficient to ensure moment preservation due to modification of first coeff
    Correct_coef_x_1 = np.linalg.solve(Mat_to_inv,Rhs_1) # same second coeff

    Correct_coef_x_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)

    py=degree_y[0]
    enddom = breakpoints_x_y[-1]
    begdom = breakpoints_x_y[0]
    denom = enddom-begdom
    #Direction x (average y component)
    Mass_mat = np.zeros((py+1,local_shape[1][0]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(py+1):
        for ie1 in range(Nel[1]):   #loop on cells
            cell_size = breakpoints_x_y[ie1+1]-breakpoints_x_y[ie1]
            for il1 in range(py+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis_y[0][ie1,il1,0,q1]
                    y  = grid[1][ie1,q1]
                    val += cell_size/2*w[1][q1]*v0*((y-begdom)/denom)**poldeg 
                locindy=span_y[0][ie1]-degree_y[0]+il1
                Mass_mat[poldeg,locindy]+=val
                
    Rhs_0 = Mass_mat[:,0]
    Rhs_1 = Mass_mat[:,1]
    Mat_to_inv = Mass_mat[:,2:py+3]
    Mat_to_inv_bnd = Mass_mat[:,1:py+2]

    Correct_coef_y_1 = np.linalg.solve(Mat_to_inv,Rhs_1)
    Correct_coef_y_0 = np.linalg.solve(Mat_to_inv,Rhs_0)

    Correct_coef_y_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)
    #Direction y, interface on the right
    if V1.name=="Hdiv":
        NotImplementedError("only the Hcurl is really implemented")
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
                    if mom_pres:
                        for p in range(0,px+1):
                            #correction
                            indice_plus_j  = loca2global([i_plus,  0, p+2,                 i], n_patches,patch_shape_plus)
                            indice_minus_j = loca2global([i_minus, 0, n_deg_minus-1-(p+2), i], n_patches,patch_shape_minus)
                            indice_plus    = loca2global([i_plus,  0, 0,                   i], n_patches,patch_shape_plus)
                            indice_minus   = loca2global([i_minus, 0, n_deg_minus-1,       i], n_patches,patch_shape_minus)
                            Proj[indice_plus_j,indice_plus]+=Correct_coef_x_0[p]/2
                            Proj[indice_plus_j,indice_minus]-=Correct_coef_x_0[p]/2
                            Proj[indice_minus_j,indice_plus]-=Correct_coef_x_0[p]/2
                            Proj[indice_minus_j,indice_minus]+=Correct_coef_x_0[p]/2


            elif axis == 1 :
                for i in range(s_plus.spaces[1].spaces[0].nbasis):
                    indice_minus = loca2global([i_minus,1,i,n_deg_minus-1],n_patches,patch_shape_minus)
                    indice_plus  = loca2global([i_plus,1,i,0],n_patches,patch_shape_plus)
                    Proj[indice_minus,indice_minus]-=1/2
                    Proj[indice_plus,indice_plus]-=1/2
                    Proj[indice_plus,indice_minus]+=1/2
                    Proj[indice_minus,indice_plus]+=1/2
                    if mom_pres:
                        for p in range(0,py+1):
                            #correction
                            indice_plus_j  = loca2global([i_plus,  1, i, p+2                ], n_patches,patch_shape_plus)
                            indice_minus_j = loca2global([i_minus, 1, i, n_deg_minus-1-(p+2)], n_patches,patch_shape_minus)
                            indice_plus    = loca2global([i_plus,  1, i, 0                  ], n_patches,patch_shape_plus)
                            indice_minus   = loca2global([i_minus, 1, i, n_deg_minus-1      ], n_patches,patch_shape_minus)
                            Proj[indice_plus_j,indice_plus]+=Correct_coef_y_0[p]/2
                            Proj[indice_plus_j,indice_minus]-=Correct_coef_y_0[p]/2
                            Proj[indice_minus_j,indice_plus]-=Correct_coef_y_0[p]/2
                            Proj[indice_minus_j,indice_minus]+=Correct_coef_y_0[p]/2

    elif V1.name=="Hcurl":

        # on each d=axis we use the 1D projection on B(x_d) splines along x_{d+1} direction 
        # spline degrees: 
        # py = patch_space.spaces[1].degree[0] for axis = 0
        # px = patch_space.spaces[0].degree[1] for axis = 1
        proj_degs = [py,px]
        a_same = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # a_same[axis]: coefs of P B0 on same patch
        a_ngbr = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # a_ngbr[axis]: coefs of P B0 on neighbor patch
        b_same = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # b_same[axis]: coefs of P B1 on same patch
        b_ngbr = [np.zeros(proj_degs[axis]+3) for axis in [0,1]] # b_ngbr[axis]: coefs of P B1 on neighbor patch
        corr_coef_0 = [Correct_coef_y_0,Correct_coef_x_0]
        corr_coef_1 = [Correct_coef_y_1,Correct_coef_x_1]

        for axis in [0,1] :
            p_ax = proj_degs[axis]
            a_sm = a_same[axis]
            a_nb = a_ngbr[axis]
            b_sm = b_same[axis]
            b_nb = b_ngbr[axis]
            # projection coefs:
            a_sm[0] = 1/2
            a_nb[0] = a_sm[0]
            if reg == 0:
                b_sm[1] = 1
            else:
                if proj_op == 0:
                    # new slope is average of old ones
                    a_sm[1] = 0  
                elif proj_op == 1:
                    # new slope is average of old ones after averaging of interface coef
                    a_sm[1] = 1/2
                elif proj_op == 2:
                    # new slope is average of reconstructed ones using local values and slopes
                    a_sm[1] = 1/(2*p_ax)
                else:
                    # just to try something else
                    a_sm[1] = proj_op/2
                    
                a_nb[1] = 2*a_sm[0] - a_sm[1]
                b_sm[0] = 0
                b_sm[1] = 1/2
                b_nb[0] = b_sm[0]
                b_nb[1] = 2*b_sm[0] - b_sm[1]
            
            if mom_pres:
                cc_0_ax = corr_coef_0[axis]
                if reg > 0:
                    cc_1_ax = corr_coef_1[axis]
                for p in range(0,p_ax+1):
                    # correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                    # modified by the C0 or C1 enforcement
                    if reg == 0:
                        a_sm[p+2] = (1-a_sm[0]) * cc_0_ax[p]
                        a_nb[p+2] = -a_sm[p+2]
                    
                    else:
                        a_sm[p+2] = (1-a_sm[0]) * cc_0_ax[p]     -a_sm[1]  * cc_1_ax[p]
                        b_sm[p+2] =   -b_sm[0]  * cc_0_ax[p] + (1-b_sm[1]) * cc_1_ax[p]
                        # proj constraints:
                        b_nb[p+2] = b_sm[p+2]
                        a_nb[p+2] = -(a_sm[p+2] + 2*b_sm[p+2])

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

            # defining functions glob_ind_minus(i,j) and glob_ind_plus(i,j):
            # to return global index of local (relative) multi-index on minus/plus patch
            #   i: index along interface
            #   j: index across interface (relative: 0 is at interface)
            
            if axis == 0:
                def glob_ind_minus(i,j):
                    return loca2global([i_minus,1,n_deg_minus-1-j,i], n_patches, patch_shape_minus)
                def glob_ind_plus(i,j):
                    return loca2global([i_plus, 1,              j,i], n_patches, patch_shape_plus)

            elif axis == 1:
                def glob_ind_minus(i,j):
                    return loca2global([i_minus,0,i,n_deg_minus-1-j], n_patches, patch_shape_minus)

                def glob_ind_plus(i,j):
                    return loca2global([i_plus, 0,i,              j], n_patches, patch_shape_plus)
                
            else:
                raise ValueError(axis)
                        
            a_sm = a_same[axis]
            a_nb = a_ngbr[axis]
            b_sm = b_same[axis]
            b_nb = b_ngbr[axis]

            for i in range(s_plus.spaces[naxis].spaces[naxis].nbasis):

                indice_minus_1 = glob_ind_minus(i,1)
                indice_minus   = glob_ind_minus(i,0)
                indice_plus    = glob_ind_plus (i,0)
                indice_plus_1  = glob_ind_plus (i,1)
                
                #changing these coefficients ensure C1 continuity at the interface                    
                Proj[indice_minus,indice_minus] += (a_sm[0]-1)
                Proj[indice_plus, indice_plus ] += (a_sm[0]-1)
                Proj[indice_plus, indice_minus] += a_nb[0]
                Proj[indice_minus,indice_plus ] += a_nb[0]

                # note: b_sm[0] = b_nb[0] is hard coded here
                Proj[indice_minus_1,indice_minus_1] += (b_sm[1]-1)
                Proj[indice_minus_1,indice_minus  ] += a_sm[1]    
                Proj[indice_minus_1,indice_plus   ] += a_nb[1]
                Proj[indice_minus_1,indice_plus_1 ] += b_nb[1]
                
                Proj[indice_plus_1,indice_minus_1]  += b_nb[1]      
                Proj[indice_plus_1,indice_minus  ]  += a_nb[1]      
                Proj[indice_plus_1,indice_plus   ]  += a_sm[1]      
                Proj[indice_plus_1,indice_plus_1 ]  += (b_sm[1]-1) 
                
                # print(' NEW -- axis = ', axis, 'patches: ', i_minus, i_plus)
                # for i1 in [indice_minus_1, indice_minus, indice_plus, indice_plus_1]:
                #     for i2 in [indice_minus_1, indice_minus, indice_plus, indice_plus_1]:
                #         print(i1, i2, ' : ', Proj[i1,i2])
                p_ax = proj_degs[axis]
                for p in range(0,p_ax+1):
                    #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                    #modified by the C1 enforcement
                    indice_minus_j = glob_ind_minus (i,p+2)
                    indice_plus_j  = glob_ind_plus (i,p+2)
                    
                    Proj[indice_minus_j,indice_minus]   += a_sm[p+2]
                    Proj[indice_plus_j, indice_plus ]   += a_sm[p+2]
                    Proj[indice_plus_j, indice_minus]   += a_nb[p+2]
                    Proj[indice_minus_j,indice_plus ]   += a_nb[p+2]

                    Proj[indice_minus_j,indice_minus_1] += b_sm[p+2]
                    Proj[indice_plus_j,indice_plus_1]   += b_sm[p+2]
                    Proj[indice_minus_j,indice_plus_1]  += b_nb[p+2]
                    Proj[indice_plus_j,indice_minus_1]  += b_nb[p+2]

                    # print(' - MP - ')
                    # for i1 in [indice_minus_j, indice_plus_j]:
                    #     for i2 in [indice_minus_1, indice_minus, indice_plus, indice_plus_1]:
                    #         print(i1, i2, ' : ', Proj[i1,i2])

                # if axis == 1: exit()

            # elif axis == 1 :
            #     # gamma = 1  # smoother than 0 ?
            #     for i in range(s_plus.spaces[0].spaces[0].nbasis):
            #         indice_minus = loca2global([i_minus,0,i,n_deg_minus-1],n_patches,patch_shape_minus)
            #         indice_plus  = loca2global([i_plus,0,i,0],n_patches,patch_shape_plus)
            #         indice_minus_1 = loca2global([i_minus,0,i,n_deg_minus-2],n_patches,patch_shape_minus)
            #         indice_plus_1  = loca2global([i_plus,0,i,1],n_patches,patch_shape_plus)
            #         #changing this coefficients ensure C1 continuity at the interface
            #         Proj[indice_minus,indice_minus]-=1/2
            #         Proj[indice_plus,indice_plus]-=1/2
            #         Proj[indice_plus,indice_minus]+=1/2
            #         Proj[indice_minus,indice_plus]+=1/2
            #         # Proj[indice_minus,indice_minus] =0
            #         # Proj[indice_plus,indice_plus]=0
            #         # Proj[indice_plus,indice_minus]=0
            #         # Proj[indice_minus,indice_plus]=0
            #         Proj[indice_minus_1,indice_minus_1] -= 1/2
            #         Proj[indice_minus_1,indice_minus  ]  =  gamma/2
            #         Proj[indice_minus_1,indice_plus   ]  = (2 - gamma)/2
            #         Proj[indice_minus_1,indice_plus_1 ]  = -1/2
                    
            #         Proj[indice_plus_1,indice_minus_1]  = -1/2
            #         Proj[indice_plus_1,indice_minus  ]  = (2 + gamma)/2
            #         Proj[indice_plus_1,indice_plus   ]  = -gamma/2
            #         Proj[indice_plus_1,indice_plus_1 ] -= 1/2

            #         # Proj[indice_minus_1,indice_minus_1] = 0
            #         # Proj[indice_plus_1,indice_plus_1]-=1/2
            #         # Proj[indice_plus_1,indice_plus_1] = 0
            #         # Proj[indice_minus_1,indice_plus] = 0 
            #         # Proj[indice_plus_1,indice_minus] =0 #+=1
            #         # Proj[indice_plus_1,indice_minus]=0
            #         # Proj[indice_minus_1,indice_plus_1] = 0
            #         # Proj[indice_plus_1,indice_minus_1]-=1/2
            #         # Proj[indice_plus_1,indice_minus_1]=0
            #         if mom_pres:
            #             for p in range(0,px+1):
            #                 #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
            #                 #modified by the C1 enforcement
            #                 indice_plus_i  = loca2global([i_plus,  0, i, p+2                ], n_patches,patch_shape_plus)
            #                 indice_minus_i = loca2global([i_minus, 0, i, n_deg_minus-1-(p+2)], n_patches,patch_shape_minus)
            #                 Proj[indice_plus_i,indice_plus]+=Correct_coef_x_0[p]/2
            #                 Proj[indice_plus_i,indice_minus]-=Correct_coef_x_0[p]/2
            #                 Proj[indice_minus_i,indice_plus]-=Correct_coef_x_0[p]/2
            #                 Proj[indice_minus_i,indice_minus]+=Correct_coef_x_0[p]/2
            #                 Proj[indice_minus_i,indice_minus_1]+=Correct_coef_x_1[p]/2
            #                 Proj[indice_plus_i,indice_plus_1]+=Correct_coef_x_1[p]/2
            #                 Proj[indice_minus_i,indice_plus]-=Correct_coef_x_1[p]
            #                 Proj[indice_plus_i,indice_minus]-=Correct_coef_x_1[p]
            #                 Proj[indice_minus_i,indice_plus_1]+=Correct_coef_x_1[p]/2
            #                 Proj[indice_plus_i,indice_minus_1]+=Correct_coef_x_1[p]/2

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
                
                # TODO: use a single function with axis as argument
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
                        #set bnd dof to zero
                        Proj[indice,indice]-=1
                        if mom_pres:
                            for p in range(0,py+1):
                            #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                            #modified by the C1 enforcement
                                if ext ==+1 :
                                    indice_i = loca2global([i_patch,1,n_deg-1-(p+2),i],n_patches,patch_shape)
                                elif ext == -1 : 
                                    indice_i = loca2global([i_patch,1,p+2,i],n_patches,patch_shape)
                                else :
                                    ValueError("wrong value for ext")
                                Proj[indice_i,indice]+=Correct_coef_y_bnd[p]
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
                        if mom_pres:
                            for p in range(0,px+1):
                            #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                            #modified by the C1 enforcement
                                if ext ==+1 :
                                    indice_i = loca2global([i_patch,0,i,n_deg-1-(p+2)],n_patches,patch_shape)
                                elif ext == -1 : 
                                    indice_i = loca2global([i_patch,0,i,p+2],n_patches,patch_shape)
                                else :
                                    ValueError("wrong value for ext")
                                Proj[indice_i,indice]+=Correct_coef_x_bnd[p]
                            #Proj[indice_1,indice_1]-=1/2
    else :
        print("Error in Conf_proj_1 : wrong kind of space")

    if reg == 0:

        # C0 smoothness is imposed for the proper component between interfaces, we may return the matrix
        return Proj
    
    # otherwise we need to treat the other component 
    # TODO: we should have a single function that treats one component with C0 or C1 smoothness -- and bc's

    Proj_par = sparse_eye(dim_tot,format="lil")

    enddom = breakpoints_x_x[-1]
    begdom = breakpoints_x_x[0]
    denom = enddom-begdom
    #Direction x (average x component)
    Mass_mat = np.zeros((px+1,local_shape[0][0]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(px+1):
        for ie1 in range(Nel[0]):   #loop on cells
            cell_size = breakpoints_x_x[ie1+1]-breakpoints_x_x[ie1]
            for il1 in range(degree_x[0]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[0]): #loops on quadrature points
                    v0 = basis_x[0][ie1,il1,0,q1]
                    x  = grid[0][ie1,q1]
                    val += cell_size*0.5*w[0][q1]*v0*((x-begdom)/denom)**poldeg
                locindx=span_x[0][ie1]-degree_x[0]+il1
                Mass_mat[poldeg,locindx]+=val
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:px+2]
    Mat_to_inv_bnd = Mass_mat[:,1:px+2]

    Correct_coef_x = np.linalg.solve(Mat_to_inv,Rhs) #coefficient to ensure moment preservation due to modification of first coeff

    Correct_coef_x_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs)

    enddom = breakpoints_x_y[-1]
    begdom = breakpoints_x_y[0]
    denom = enddom-begdom
    #Direction x (average y component)
    Mass_mat = np.zeros((py+1,local_shape[1][1]))
    #computing the contribution to every moment of the differents basis function
    for poldeg in range(py+1):
        for ie1 in range(Nel[1]):   #loop on cells
            cell_size = breakpoints_x_y[ie1+1]-breakpoints_x_y[ie1]
            for il1 in range(degree_y[1]+1): #loops on basis function in each cell
                val=0.
                for q1 in range(nquads[1]): #loops on quadrature points
                    v0 = basis_y[1][ie1,il1,0,q1]
                    y  = grid[1][ie1,q1]
                    val += cell_size/2*w[1][q1]*v0*((y-begdom)/denom)**poldeg 
                locindy=span_y[1][ie1]-degree_y[1]+il1
                Mass_mat[poldeg,locindy]+=val
                
    Rhs = Mass_mat[:,0]
    Mat_to_inv = Mass_mat[:,1:py+2]
    Mat_to_inv_bnd = Mass_mat[:,1:py+2]

    Correct_coef_y = np.linalg.solve(Mat_to_inv,Rhs)

    Correct_coef_y_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs)

    for I in Interfaces:
        axis = I.axis
        i_minus = get_patch_index_from_face(domain, I.minus)
        i_plus  = get_patch_index_from_face(domain, I.plus )
        s_minus = V1h.spaces[i_minus]
        s_plus  = V1h.spaces[i_plus]
        naxis = (axis+1)%2
        patch_shape_minus = [[s_minus.spaces[0].spaces[0].nbasis,s_minus.spaces[0].spaces[1].nbasis],
                            [s_minus.spaces[1].spaces[0].nbasis,s_minus.spaces[1].spaces[1].nbasis]]
        patch_shape_plus  = [[s_plus.spaces[0].spaces[0].nbasis,s_plus.spaces[0].spaces[1].nbasis],
                            [s_plus.spaces[1].spaces[0].nbasis,s_plus.spaces[1].spaces[1].nbasis]]
        n_deg_minus = s_minus.spaces[axis].spaces[axis].nbasis
        if axis == 0 :
            for i in range(s_plus.spaces[0].spaces[1].nbasis):
                indice_minus = loca2global([i_minus,  0,n_deg_minus-1,i],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,   0,   0,i],n_patches,patch_shape_plus)
                #changing this coefficients ensure C0 continuity at the interface
                Proj_par[indice_minus,indice_minus]-=1/2
                Proj_par[indice_plus,indice_plus]-=1/2
                Proj_par[indice_plus,indice_minus]+=1/2
                Proj_par[indice_minus,indice_plus]+=1/2
                if mom_pres:
                    for p in range(0,px+1):
                        #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                        #modified by the C0 enforcement
                        indice_plus_j  = loca2global([i_plus,  0, p+1,                 i], n_patches,patch_shape_plus)
                        indice_minus_j = loca2global([i_minus, 0, n_deg_minus-1-(p+1), i], n_patches,patch_shape_minus)

                        Proj_par[indice_plus_j,indice_plus]+=Correct_coef_x[p]/2
                        Proj_par[indice_plus_j,indice_minus]-=Correct_coef_x[p]/2
                        Proj_par[indice_minus_j,indice_plus]-=Correct_coef_x[p]/2
                        Proj_par[indice_minus_j,indice_minus]+=Correct_coef_x[p]/2
        elif axis == 1 :
            for i in range(s_plus.spaces[1].spaces[0].nbasis):
                indice_minus = loca2global([i_minus,  1,i,n_deg_minus-1],n_patches,patch_shape_minus)
                indice_plus  = loca2global([i_plus,   1, i,  0,],n_patches,patch_shape_plus)
                #changing this coefficients ensure C0 continuity at the interface
                Proj_par[indice_minus,indice_minus]-=1/2
                Proj_par[indice_plus,indice_plus]-=1/2
                Proj_par[indice_plus,indice_minus]+=1/2
                Proj_par[indice_minus,indice_plus]+=1/2
                if mom_pres:
                    for p in range(0,py+1):
                        #correction for moment preserving : modify p+1 other basis function to preserve the p+1 moments
                        #modified by the C0 enforcement
                        indice_plus_j  = loca2global([i_plus,  1, i, p+1                ], n_patches,patch_shape_plus)
                        indice_minus_j = loca2global([i_minus, 1, i, n_deg_minus-1-(p+1)], n_patches,patch_shape_minus)

                        Proj_par[indice_plus_j,indice_plus]+=Correct_coef_y[p]/2
                        Proj_par[indice_plus_j,indice_minus]-=Correct_coef_y[p]/2
                        Proj_par[indice_minus_j,indice_plus]-=Correct_coef_y[p]/2
                        Proj_par[indice_minus_j,indice_minus]+=Correct_coef_y[p]/2

    print((Proj@Proj_par-Proj_par@Proj))

    assert(np.allclose((Proj@Proj_par).todense(),(Proj_par@Proj).todense()))

    return Proj@Proj_par
