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


def glob_ind_interface_scalfield(i, j, side, k_patch, axis, patch_shape, n_patches, nb_dofs_across):
    '''
    returns global index of dof close to interface, for a scalar field
    
    i: dof index along interface
    j: relative dof index (distance) from interface
    side: -1 or +1 for minus or plus patch of interface
    k_patch: index of patch
    nb_dofs_across: nb dofs in direction perpendicular to interface
    '''
    if side == 1:
        absolute_j = j 
    else:
        absolute_j = nb_dofs_across-1-j
    
    if axis == 0:
        return loca2global([k_patch, absolute_j, i], n_patches, patch_shape)
    else:
        return loca2global([k_patch, i, absolute_j], n_patches, patch_shape)
    
def glob_ind_interface_vecfield(i, j, side, k_patch, axis, patch_shape, n_patches, nb_dofs_across, comp):
    '''    
    returns global index of dof close to interface, for a scalar field
        
    i: dof index along interface
    j: relative dof index (distance) from interface
    side: -1 or +1 for minus or plus patch of interface
    k_patch: index of patch
    nb_dofs_across: nb dofs in direction perpendicular to interface
    comp: component
    '''
    if side == 1:
        absolute_j = j 
    else:
        absolute_j = nb_dofs_across-1-j
    
    if axis == 0:
        return loca2global([k_patch, comp, absolute_j, i], n_patches, patch_shape)
    else:
        return loca2global([k_patch, comp, i, absolute_j], n_patches, patch_shape)
    

#---------------------------------------------------#

#-#-# UNIVARIATE CONFORMING PROJECTIONS #-#-#

def univariate_conf_proj_scalar_space(Vh, conf_axis, reg=0, p_moments=-1, nquads=None, C1_proj_opt=None, hom_bc=False):
    """
    Create the matrix enforcing Cr continuity (r = 0 or 1) with moment preservation along one axis for a scalar space
    
    Parameters
    ----------
    Vh : The discrete broken scalar space in which we will compute the projection

    conf_axis : axis along which conformity is imposed

    reg : order of imposed continuity (-1: no continuity, 0 or 1) 

    p_moments : degree of polynomial moments to preserve (-1 for none)

    nquads : number of integration points to compute the moment preserving weights

    C1_proj_opt : option for the projection of derivatives at interface
    
    hom_bc : Wether or not enforce homogeneous boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
    dim_tot = Vh.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    if reg < 0:
        return Proj
    
    assert reg in [0,1]

    V0      = Vh.symbolic_space
    domain  = V0.domain
    Interfaces  = domain.interfaces
    boundary = domain.boundary
    n_patches = len(Vh.spaces)

    patch_space = Vh.spaces[0]
    local_shape = [patch_space.spaces[0].nbasis,patch_space.spaces[1].nbasis]
    Nel    = patch_space.ncells                     # number of elements
    degree = patch_space.degree
    breakpoints_xy = [breakpoints(patch_space.knots[axis],degree[axis]) for axis in range(2)]

    #Creating vector of weights for moments preserving
    if nquads is None:
        # default: Gauss-Legendre quadratures should be exact for polynomials of deg ≤ 2*degree
        nquads = [ degree[axis]+1 for axis in range(2)]
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

    if reg == 1:
        if C1_proj_opt is None:
            # default option:
            C1_proj_opt = 1
        b_sm = np.zeros(p_moments+3)   # coefs of P B1 on same patch
        b_nb = np.zeros(p_moments+3)   # coefs of P B1 on neighbor patch
        if C1_proj_opt == 0:
            # new slope is average of old ones
            a_sm[1] = 0  
        elif C1_proj_opt == 1:
            # new slope is average of old ones after averaging of interface coef
            a_sm[1] = 1/2
        elif C1_proj_opt == 2:
            # new slope is average of reconstructed ones using local values and slopes
            a_sm[1] = 1/(2*p_axis)
        else:
            # just to try something else
            a_sm[1] = C1_proj_opt/2

        a_nb[1] = 2*a_sm[0] - a_sm[1]
        b_sm[0] = 0
        b_sm[1] = 1/2
        b_nb[0] = b_sm[0]
        b_nb[1] = 2*b_sm[0] - b_sm[1]
    
    if p_moments >= 0:
        # to preserve moments of degree p we need 1+p conforming basis functions in the patch (the "interior" ones)
        # and for the given regularity constraint, there are local_shape[conf_axis]-2*(1+reg) such conforming functions 
        p_max = local_shape[conf_axis]-2*(1+reg) - 1
        if p_max < p_moments:
            print( " ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            print( " **         WARNING -- WARNING -- WARNING ")
            print(f" ** conf. projection imposing C{reg} smoothness on scalar space along axis {conf_axis}:")            
            print(f" ** there are not enough dofs in a patch to preserve moments of degree {p_moments} !")
            print(f" ** Only able to preserve up to degree --> {p_max} <-- ")
            print( " ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            p_moments = p_max

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
        
        if reg == 1:
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

            k_minus = get_patch_index_from_face(domain, I.minus)
            k_plus  = get_patch_index_from_face(domain, I.plus )
            s_minus = Vh.spaces[k_minus]
            s_plus  = Vh.spaces[k_plus]
            nb_dofs_minus = s_minus.spaces[axis].nbasis
            nb_dofs_plus = s_plus.spaces[axis].nbasis
            nb_dofs_face = s_plus.spaces[1-conf_axis].nbasis
            assert nb_dofs_face == s_minus.spaces[1-conf_axis].nbasis
            patch_shape_minus = [s_minus.spaces[0].nbasis,s_minus.spaces[1].nbasis]
            patch_shape_plus  = [s_plus.spaces[0].nbasis,s_plus.spaces[1].nbasis]
        
            # loop over dofs on the interface (assuming same grid on plus and minus patches)
            for i in range(nb_dofs_face):

                index_minus = glob_ind_interface_scalfield(
                    i,0,
                    side=-1,k_patch=k_minus,
                    axis=conf_axis,
                    patch_shape=patch_shape_minus,
                    n_patches=n_patches,
                    nb_dofs_across=nb_dofs_minus)
                
                index_plus = glob_ind_interface_scalfield(
                    i,0,
                    side=+1,k_patch=k_plus,
                    axis=conf_axis,
                    patch_shape=patch_shape_plus,
                    n_patches=n_patches,
                    nb_dofs_across=nb_dofs_plus)

                Proj[index_minus,index_minus] = a_sm[0]
                Proj[index_plus, index_plus ] = a_sm[0]
                Proj[index_plus, index_minus] = a_nb[0]
                Proj[index_minus,index_plus ] = a_nb[0]

                if reg == 1:

                    index_minus_1 = glob_ind_interface_scalfield(
                        i,1,
                        side=-1,k_patch=k_minus,
                        axis=conf_axis,
                        patch_shape=patch_shape_minus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_minus)
                    
                    index_plus_1 = glob_ind_interface_scalfield(
                        i,1,
                        side=+1,k_patch=k_plus,
                        axis=conf_axis,
                        patch_shape=patch_shape_plus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_plus)

                    # note: b_sm[0] = b_nb[0] is hard coded
                    Proj[index_minus_1,index_minus_1] = b_sm[1]
                    Proj[index_minus_1,index_minus  ] = a_sm[1]    
                    Proj[index_minus_1,index_plus   ] = a_nb[1]
                    Proj[index_minus_1,index_plus_1 ] = b_nb[1]
                    
                    Proj[index_plus_1,index_minus_1]  = b_nb[1]      
                    Proj[index_plus_1,index_minus  ]  = a_nb[1]      
                    Proj[index_plus_1,index_plus   ]  = a_sm[1]      
                    Proj[index_plus_1,index_plus_1 ]  = b_sm[1]

                for p in range(0,p_moments+1):
                    # correction for moment preservation: modify the projection of the interface functions with the interior ones
                    j = p+reg+1   # index of interior function (relative to interface)
                    index_minus_j = glob_ind_interface_scalfield(
                        i,j,
                        side=-1,k_patch=k_minus,
                        axis=conf_axis,
                        patch_shape=patch_shape_minus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_minus)
                    
                    index_plus_j = glob_ind_interface_scalfield(
                        i,j,
                        side=+1,k_patch=k_plus,
                        axis=conf_axis,
                        patch_shape=patch_shape_plus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_plus)

                    Proj[index_minus_j,index_minus]   = a_sm[j]
                    Proj[index_plus_j, index_plus ]   = a_sm[j]
                    Proj[index_plus_j, index_minus]   = a_nb[j]
                    Proj[index_minus_j,index_plus ]   = a_nb[j]

                    if reg == 1:
                        Proj[index_minus_j,index_minus_1] = b_sm[j]
                        Proj[index_plus_j,index_plus_1]   = b_sm[j]
                        Proj[index_minus_j,index_plus_1]  = b_nb[j]
                        Proj[index_plus_j,index_minus_1]  = b_nb[j]
    
    if hom_bc:
        for b in boundary : 
            axis = b.axis
            if axis == conf_axis:

                ext = b.ext
                k_patch = get_patch_index_from_face(domain,b)
                patch_space = Vh.spaces[k_patch]
                patch_shape = [patch_space.spaces[ax].nbasis for ax in range(2)]
                nb_dofs_axis = patch_shape[conf_axis] # patch_space.spaces[conf_axis].nbasis
                nb_dofs_face = patch_shape[1-conf_axis] # patch_space.spaces[1-conf_axis].nbasis

                for i in range(nb_dofs_face):
                    index = glob_ind_interface_scalfield(i,0,side=-ext,k_patch=k_patch,
                                                          axis=axis,
                                                          patch_shape=patch_shape,
                                                          n_patches=n_patches,
                                                          nb_dofs_across=nb_dofs_axis)
                    Proj[index,index] = 0

                    for p in range(0,p_moments+1):
                        # correction (with only 1 interface function on boundary)
                        j = p+1
                        index_j = glob_ind_interface_scalfield(
                            i,j,
                            side=-ext,k_patch=k_patch,
                            axis=axis,
                            patch_shape=patch_shape,
                            n_patches=n_patches,
                            nb_dofs_across=nb_dofs_axis)

                        Proj[index_j,index] = Correct_coef_bnd[p]
                        
    return Proj


def univariate_conf_proj_vector_space(Vh, conf_axis, conf_comp, reg=0, p_moments=-1, nquads=None, C1_proj_opt=None, hom_bc=False):
    """
    Create the matrix enforcing Cr continuity (r = 0 or 1) with moment preservation along one axis for a component of a vector-valued space
    
    Parameters
    ----------
    Vh : The discrete broken space (vector-valued) in which we will compute the projection

    conf_axis : axis along which conformity is imposed

    conf_comp : component for which conformity is imposed

    reg : order of imposed continuity (-1: no continuity, 0 or 1) 

    p_moments : degree of polynomial moments to preserve (-1 for none)

    nquads : number of integration points to compute the moment preserving weights

    C1_proj_opt : option for the projection of derivatives at interface

    hom_bc : Wether or not enforce homogeneous boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
    dim_tot = Vh.nbasis
    Proj    = sparse_eye(dim_tot,format="lil")
    if reg < 0:
        return Proj
    
    assert reg in [0,1]

    V      = Vh.symbolic_space
    domain  = V.domain
    Interfaces  = domain.interfaces
    boundary = domain.boundary
    n_patches = len(Vh.spaces)

    patch_space = Vh.spaces[0]
    local_shape = [[patch_space.spaces[comp].spaces[axis].nbasis 
                    for axis in range(2)] for comp in range(2)]
    Nel    = patch_space.ncells                     # number of elements
    patch_space_x, patch_space_y = [patch_space.spaces[comp] for comp in range(2)]
    degree = patch_space.degree 
    p_comp_axis = degree[conf_comp][conf_axis]

    breaks_comp_axis = [[breakpoints(patch_space.spaces[comp].knots[axis],degree[comp][axis])
                              for axis in range(2)] for comp in range(2)]
    
    #Creating vector of weights for moments preserving
    if nquads is None:
        # default: Gauss-Legendre quadratures should be exact for polynomials of deg ≤ 2*degree
        nquads = [ degree[conf_comp][axis]+1 for axis in range(2)]
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]
    
    grid = [np.array([deepcopy((0.5*(u[axis]+1)*(breaks_comp_axis[0][axis][i+1]-breaks_comp_axis[0][axis][i])+breaks_comp_axis[0][axis][i])) 
                      for i in range(Nel[axis])])
            for axis in range(2)]

    _, basis_x, span_x, _ = patch_space_x.preprocess_regular_tensor_grid(grid,der=0)
    _, basis_y, span_y, _ = patch_space_y.preprocess_regular_tensor_grid(grid,der=0)
    span_x = [deepcopy(span_x[k] + patch_space_x.vector_space.starts[k] - patch_space_x.vector_space.shifts[k] * patch_space_x.vector_space.pads[k]) for k in range(2)]
    span_y = [deepcopy(span_y[k] + patch_space_y.vector_space.starts[k] - patch_space_y.vector_space.shifts[k] * patch_space_y.vector_space.pads[k]) for k in range(2)]
    basis = [basis_x, basis_y]
    span = [span_x, span_y]
    enddom = breaks_comp_axis[0][0][-1]
    begdom = breaks_comp_axis[0][0][0]
    denom = enddom-begdom

    if nquads is None:
        # default: Gauss-Legendre quadratures should be exact for polynomials of deg ≤ 2*degree
        nquads = [ degree[axis]+1 for axis in range(2)]

    # projection coefficients
    a_sm = np.zeros(p_moments+2+reg)   # coefs of P B0 on same patch
    a_nb = np.zeros(p_moments+2+reg)   # coefs of P B0 on neighbor patch
    
    a_sm[0] = 1/2
    a_nb[0] = a_sm[0]

    if reg == 1:
        if C1_proj_opt is None:
            # default option:
            C1_proj_opt = 1

        b_sm = np.zeros(p_moments+3)   # coefs of P B1 on same patch
        b_nb = np.zeros(p_moments+3)   # coefs of P B1 on neighbor patch
        if C1_proj_opt == 0:
            # new slope is average of old ones
            a_sm[1] = 0  
        elif C1_proj_opt == 1:
            # new slope is average of old ones after averaging of interface coef
            a_sm[1] = 1/2
        elif C1_proj_opt == 2:
            # new slope is average of reconstructed ones using local values and slopes
            a_sm[1] = 1/(2*p_comp_axis)
        else:
            # just to try something else
            a_sm[1] = C1_proj_opt/2

        a_nb[1] = 2*a_sm[0] - a_sm[1]
        b_sm[0] = 0
        b_sm[1] = 1/2
        b_nb[0] = b_sm[0]
        b_nb[1] = 2*b_sm[0] - b_sm[1]
    
    if p_moments >= 0:
        # to preserve moments of degree p we need 1+p conforming basis functions in the patch (the "interior" ones)
        # and for the given regularity constraint, there are local_shape[conf_comp][conf_axis]-2*(1+reg) such conforming functions 
        p_max = local_shape[conf_comp][conf_axis]-2*(1+reg) - 1
        if p_max < p_moments:
            print( " ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            print( " **         WARNING -- WARNING -- WARNING ")
            print(f" ** conf. projection imposing C{reg} smoothness on component {conf_comp} along axis {conf_axis}:")            
            print(f" ** there are not enough dofs in a patch to preserve moments of degree {p_moments} !")
            print(f" ** Only able to preserve up to degree --> {p_max} <-- ")
            print( " ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            p_moments = p_max

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
        
        if reg == 1:
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

            k_minus = get_patch_index_from_face(domain, I.minus)
            k_plus  = get_patch_index_from_face(domain, I.plus )
            s_minus = Vh.spaces[k_minus]
            s_plus  = Vh.spaces[k_plus]
            nb_dofs_minus = s_minus.spaces[conf_comp].spaces[conf_axis].nbasis
            nb_dofs_plus  = s_plus.spaces[conf_comp].spaces[conf_axis].nbasis
            nb_dofs_face  = s_minus.spaces[conf_comp].spaces[1-conf_axis].nbasis
            # here we assume the same grid along the interface on plus and minus patches:
            assert nb_dofs_face == s_plus.spaces[conf_comp].spaces[1-conf_axis].nbasis

            patch_shape_minus = [[s_minus.spaces[cp].spaces[ax].nbasis 
                                for ax in range(2)] for cp in range(2)]
            patch_shape_plus = [[s_minus.spaces[cp].spaces[ax].nbasis 
                                for ax in range(2)] for cp in range(2)]

            # loop over dofs on the interface
            for i in range(nb_dofs_face):
                index_minus = glob_ind_interface_vecfield(
                    i,0,
                    side=-1,k_patch=k_minus,
                    axis=conf_axis,
                    patch_shape=patch_shape_minus,
                    n_patches=n_patches,
                    nb_dofs_across=nb_dofs_minus,
                    comp=conf_comp)
                
                index_plus = glob_ind_interface_vecfield(
                    i,0,
                    side=+1,k_patch=k_plus,
                    axis=conf_axis,
                    patch_shape=patch_shape_plus,
                    n_patches=n_patches,
                    nb_dofs_across=nb_dofs_plus,
                    comp=conf_comp)

                Proj[index_minus,index_minus] = a_sm[0]
                Proj[index_plus, index_plus ] = a_sm[0]
                Proj[index_plus, index_minus] = a_nb[0]
                Proj[index_minus,index_plus ] = a_nb[0]

                if reg == 1:

                    index_minus_1 = glob_ind_interface_vecfield(
                        i,1,
                        side=-1,k_patch=k_minus,
                        axis=conf_axis,
                        patch_shape=patch_shape_minus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_minus,
                        comp=conf_comp)
                    
                    index_plus_1 = glob_ind_interface_vecfield(
                        i,1,
                        side=+1,k_patch=k_plus,
                        axis=conf_axis,
                        patch_shape=patch_shape_plus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_plus,
                        comp=conf_comp)

                    # note: b_sm[0] = b_nb[0] is hard coded
                    Proj[index_minus_1,index_minus_1] = b_sm[1]
                    Proj[index_minus_1,index_minus  ] = a_sm[1]    
                    Proj[index_minus_1,index_plus   ] = a_nb[1]
                    Proj[index_minus_1,index_plus_1 ] = b_nb[1]
                    
                    Proj[index_plus_1,index_minus_1]  = b_nb[1]      
                    Proj[index_plus_1,index_minus  ]  = a_nb[1]      
                    Proj[index_plus_1,index_plus   ]  = a_sm[1]      
                    Proj[index_plus_1,index_plus_1 ]  = b_sm[1]

                for p in range(0,p_moments+1):
                    # correction for moment preservation: modify the projection of the interface functions with the interior ones
                    j = p+reg+1 # index of interior function (relative to interface)
                    index_minus_j = glob_ind_interface_vecfield(
                        i,j,
                        side=-1,k_patch=k_minus,
                        axis=conf_axis,
                        patch_shape=patch_shape_minus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_minus,
                        comp=conf_comp)
                    
                    index_plus_j = glob_ind_interface_vecfield(
                        i,j,
                        side=+1,k_patch=k_plus,
                        axis=conf_axis,
                        patch_shape=patch_shape_plus,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_plus,
                        comp=conf_comp)
                    
                    Proj[index_minus_j,index_minus]   = a_sm[j]
                    Proj[index_plus_j, index_plus ]   = a_sm[j]
                    Proj[index_plus_j, index_minus]   = a_nb[j]
                    Proj[index_minus_j,index_plus ]   = a_nb[j]

                    if reg == 1:
                        Proj[index_minus_j,index_minus_1] = b_sm[j]
                        Proj[index_plus_j,index_plus_1]   = b_sm[j]
                        Proj[index_minus_j,index_plus_1]  = b_nb[j]
                        Proj[index_plus_j,index_minus_1]  = b_nb[j]

    if hom_bc:
        for b in boundary : 
            axis = b.axis
            if axis == conf_axis:
                ext = b.ext
                k_patch = get_patch_index_from_face(domain,b)
                patch_space = Vh.spaces[k_patch]
                patch_shape = [[patch_space.spaces[cp].spaces[ax].nbasis 
                                    for ax in range(2)] for cp in range(2)]

                nb_dofs_axis = patch_shape[conf_comp][conf_axis]
                nb_dofs_face = patch_shape[conf_comp][1-conf_axis]

                for i in range(nb_dofs_face):
                    index = glob_ind_interface_vecfield(
                        i,0,
                        side=-ext,k_patch=k_patch,
                        axis=conf_axis,
                        patch_shape=patch_shape,
                        n_patches=n_patches,
                        nb_dofs_across=nb_dofs_axis,
                        comp=conf_comp)

                    Proj[index,index] = 0

                    for p in range(0,p_moments+1):
                        # correction (with only 1 interface function on boundary)
                        j = p+1
                        index_j = glob_ind_interface_vecfield(
                            i,j,
                            side=-ext,k_patch=k_patch,
                            axis=conf_axis,
                            patch_shape=patch_shape,
                            n_patches=n_patches,
                            nb_dofs_across=nb_dofs_axis,
                            comp=conf_comp)

                        Proj[index_j,index] = Correct_coef_bnd[p]
                        
    return Proj

def conf_proj_scalar_space(Vh, reg_orders=[0,0], deg_moments=[-1,-1], nquads=None, C1_proj_opt=None, hom_bc_list=[False,False]):
    """
    Create the matrix enforcing the C0 or C1 continuity with moment preservation for a scalar space over 2D domain
    
    Parameters
    ----------
    Vh : The scalar-valued broken space in which we will compute the projection

    reg_orders : orders of imposed continuity (per axis) (-1: no continuity, 0 or 1) 

    deg_moments : degrees (per axis) of polynomial moments to preserve (-1 for none)

    nquads : number of integration points to compute the moment preserving weights

    C1_proj_opt : option for the projection of derivatives at interface

    hom_bc_list : wether or not enforce homogeneous boundary conditions (per axis)
        
    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """

    # n_patches = len(Vh.spaces)
    dim_tot = Vh.nbasis
    Proj = sparse_eye(dim_tot,format="lil")
    
    # if n_patches ==1:
    #     if hom_bc:
    #         raise NotImplementedError("conf Proj on homogeneous bc not implemented for single patch case")
    #     else:
    #         pass 

    # else : 
    for axis in range(2):
        P_ax = univariate_conf_proj_scalar_space(
            Vh, conf_axis=axis, 
            reg=reg_orders[axis], p_moments=deg_moments[axis], 
            nquads=nquads, C1_proj_opt=C1_proj_opt, hom_bc=hom_bc_list[axis]
            )
        Proj = Proj @ P_ax
        
    return Proj




def conf_proj_vector_space(Vh, reg_orders=[[0,0],[0,0]], deg_moments=[[-1,-1],[-1,-1]], nquads=None, C1_proj_opt=None, hom_bc_list=[[False,False],[False,False]]):
    """
    Create the matrix enforcing C0 or C1 continuity with moment preservation for a vector-valued (2D) space over 2D domain
    
    Parameters
    ----------
    Vh : The vector-valued broken space in which we will compute the projection

    reg_orders : orders of imposed continuity (per component and axis) (-1: no continuity, 0 or 1) 

    deg_moments : degrees (per component and axis) of polynomial moments to preserve (-1 for none)

    nquads : number of integration points to compute the moment preserving weights

    C1_proj_opt : option for the projection of derivatives at interface

    hom_bc_list : wether or not enforce homogeneous boundary conditions (per component and axis)

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """
    # n_patches = len(Vh.spaces)
    dim_tot = Vh.nbasis
    Proj = sparse_eye(dim_tot,format="lil")

    for comp in range(2):
        for axis in range(2):
            hom_bc_cp_ax = hom_bc_list[comp][axis]
            reg = reg_orders[comp][axis]
            if hom_bc_cp_ax or reg >= 0:
                P_cp_ax = univariate_conf_proj_vector_space(
                    Vh, conf_comp=comp, conf_axis=axis, 
                    reg=reg, p_moments=deg_moments[comp][axis], nquads=nquads, C1_proj_opt=C1_proj_opt, hom_bc=hom_bc_cp_ax
                    )
                Proj = Proj @ P_cp_ax
            else:
                # no constraints: do nothing
                pass

    return Proj

def conf_projectors_scipy(derham_h, single_space=None, reg=0, mom_pres=False, nquads=None, C1_proj_opt=None, hom_bc=False):
    """
    Return all conforming projections for a given sequence
    
    Parameters
    ----------
    derham_h : the discrete broken sequence (2D)

    single_space: ('V0', 'V1', 'V2' or None) if specified, a single projector is computed

    reg : order of imposed continuity for the V0 space (0 or 1 for now) 

    mom_pres : (bool flag) whether to preserve polynomial moments (default = same as degree, in each space)

    nquads : number of integration points to compute the moment preserving weights

    C1_proj_opt : option for the projection of derivatives at interface (see details in code)

    hom_bc : (bool flag) wether or not to enforce homogeneous boundary conditions

    Returns
    -------
    Proj : scipy sparse matrix of the projection

    """    
    n_patches = len(derham_h.V0.spaces)
    if n_patches == 1:
        raise NotImplementedError("[conf_projectors_scipy]: this function should only be used on multipatch domains")
    
    if not reg in [0,1]:
        raise NotImplementedError("[conf_projectors_scipy]: regularity constraint should be 0 or 1 (for now)")
    
    assert single_space in ['V0', 'V1', 'V2', None]

    if single_space in ['V0', None]:
        V0h = derham_h.V0
        reg_orders = [reg,reg]
        hom_bc_list = [hom_bc,hom_bc]
        if mom_pres:
            deg_moments = V0h.spaces[0].degree
        else:
            deg_moments = [-1,-1]
        cP0 = conf_proj_scalar_space(
            V0h, reg_orders=reg_orders, deg_moments=deg_moments, nquads=nquads, C1_proj_opt=C1_proj_opt, hom_bc_list=hom_bc_list
            )
        if single_space == 'V0':
            return cP0 
    else:
        cP0 = None

    if single_space in ['V1', None]:

        V1h = derham_h.V1
        V1 = V1h.symbolic_space
        if mom_pres:
            deg_moments = V1h.spaces[0].degree
        else:
            deg_moments = [[-1,-1],[-1,-1]]
        if V1.name=="Hdiv":
            reg_orders  = [[reg,    reg-1], [reg-1, reg   ]]
            hom_bc_list = [[hom_bc, False], [False, hom_bc]]
        elif V1.name=="Hcurl":
            reg_orders  = [[reg-1, reg   ], [reg,    reg-1]]
            hom_bc_list = [[False, hom_bc], [hom_bc, False]]
        else:
            raise NotImplementedError(f"[conf_projectors_scipy]: no conformity rule for V1.name = {V1.name}")                

        cP1 = conf_proj_vector_space(
            V1h, reg_orders=reg_orders, deg_moments=deg_moments, nquads=nquads, C1_proj_opt=C1_proj_opt, hom_bc_list=hom_bc_list
            )
        if single_space == 'V1':
            return cP1   
    else:
        cP1 = None
           
    if single_space in ['V2', None]:
        V2h = derham_h.V2
        reg_orders  = [reg-1,reg-1]
        hom_bc_list = [False,False]
        if mom_pres:
            deg_moments = V2h.spaces[0].degree
        else:
            deg_moments = [-1,-1]
        cP2 = conf_proj_scalar_space(
            V2h, reg_orders=reg_orders, deg_moments=deg_moments, nquads=nquads, C1_proj_opt=C1_proj_opt, hom_bc_list=hom_bc_list
            )
        if single_space == 'V2':
            return cP2
    else:
        cP2 = None           

    return cP0, cP1, cP2

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
