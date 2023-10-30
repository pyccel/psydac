def assemble_dofs_for_weighted_basisfuns_1d(mat : 'float[:,:]', starts_in : 'int[:]', ends_in : 'int[:]', pads_in : 'int[:]', starts_out : 'int[:]', ends_out : 'int[:]', pads_out : 'int[:]', fun_q : 'float[:]', wts1 : 'float[:,:]', span1 : 'int[:,:]', basis1 : 'float[:,:,:]', dim1_in : int, p1_out : int):
    '''Kernel for assembling the matrix

    A_(i,j) = DOFS_i(fun*Lambda^in_j) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_i are the degrees-of-freedom of the output space (codomain, must not be a product space), 
    Lambda^in_j are the basis functions of the input space (domain, must not be a product space), and fun is an arbitrary function.

    Parameters
    ----------
        mat : 2d float array
            _data attribute of StencilMatrix.

        starts_in : int
            Starting index of the input space (domain) of a distributed StencilMatrix.

        ends_in : int
            Ending index of the input space (domain) of a distributed StencilMatrix.

        pads_in : int
            Paddings of the input space (domain) of a distributed StencilMatrix.

        starts_out : int
            Starting indices of the output space (codomain) of a distributed StencilMatrix.

        ends_out : int
            Ending indices of the output space (codomain) of a distributed StencilMatrix.

        pads_out : int
            Paddings of the output space (codomain) of a distributed StencilMatrix.

        fun_q : 1d float array
            The function evaluated at the points (nq*ii + iq), where iq a local quadrature point of interval ii.
            
        wts1 : 2d float array
            Quadrature weights in format (ii, iq).

        span1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq).

        basis1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (ii, iq, basis function).

        dim1_in : int
            Dimension of the first direction of the input space

        p1_out : int
            Spline degree of the first direction of the output space
    '''
    
    # Start/end indices and paddings for distributed stencil matrix of input space
    # si1 = starts_in[0}
    # ei1 = ends_in[0]
    pi1 = pads_in[0]

    # Start/end indices for distributed stencil matrix of output space
    so1 = starts_out[0]
    # eo1 = ends_out[0]
    po1 = pads_out[0]

    # Spline degrees of input space
    p1 = basis1.shape[2] - 1
    
    # number of quadrature points
    nq1 = span1.shape[1]

    # Set output to zero
    mat[:] = 0.

    # Dimensions of output space
    dim1_out = span1.shape[0]
    # Interval (either element or sub-interval thereof)
    # -------------------------------------------------
    cumsub_i = 0  # Cumulative sub-interval index
    for ii in range(span1.shape[0]):
        i = ii - cumsub_i  # local DOF index

        # Quadrature point index in interval
        # ----------------------------------
        for iq in range(nq1):

            funval = fun_q[nq1*ii + iq] * wts1[ii, iq]

            # Basis function of input space:
            # ------------------------------
            for b1 in range(p1 + 1):
                m = (span1[ii, iq] - p1 + b1)  # global index
                # basis value
                value = funval * basis1[ii, iq, b1]

                # Find column index for _data:
                if dim1_out <= dim1_in:
                    cut1 = p1
                else:
                    cut1 = p1_out

                # Diff of global indices, needs to be adjusted for boundary conditions --> col1
                col1_tmp = m - (i + so1)
                if col1_tmp > cut1:
                    m = m - dim1_in
                elif col1_tmp < -cut1:
                    m = m + dim1_in
                # add padding
                col1 = pi1 + m - (i + so1)

                # Row index: padding + local index.
                mat[po1 + i, col1] += value


def assemble_dofs_for_weighted_basisfuns_2d(mat : 'float[:,:,:,:]', starts_in : 'int[:]', ends_in : 'int[:]', pads_in : 'int[:]', starts_out : 'int[:]', ends_out : 'int[:]', pads_out : 'int[:]', fun_q : 'float[:,:]', wts1 : 'float[:,:]', wts2 : 'float[:,:]', span1 : 'int[:,:]', span2 : 'int[:,:]', basis1 : 'float[:,:,:]', basis2 : 'float[:,:,:]', dim1_in : int, dim2_in : int, p1_out : int, p2_out : int):
    '''Kernel for assembling the matrix

    A_(ij,kl) = DOFS_ij(fun*Lambda^in_kl) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_ij are the degrees-of-freedom of the output space (codomain, must not be a product space), 
    Lambda^in_kl are the basis functions of the input space (domain, must not be a product space), and fun is an arbitrary function.

    Parameters
    ----------
        mat : 4d float array
            _data attribute of StencilMatrix.

        starts_in : 1d int array
            Starting indices of the input space (domain) of a distributed StencilMatrix.

        ends_in : 1d int array
            Ending indices of the input space (domain) of a distributed StencilMatrix.

        pads_in : 1d int array
            Paddings of the input space (domain) of a distributed StencilMatrix.

        starts_out : 1d int array
            Starting indices of the output space (codomain) of a distributed StencilMatrix.

        ends_out : 1d int array
            Ending indices of the output space (codomain) of a distributed StencilMatrix.

        pads_out : 1d int array
            Paddings of the output space (codomain) of a distributed StencilMatrix.

        fun_q : 2d float array
            The function evaluated at the points (nq_i*ii + iq, nq_j*jj + jq), where iq a local quadrature point of interval ii.
            
        wts1 : 2d float array
            Quadrature weights in direction eta1 in format (ii, iq).
            
        wts2 : 2d float array
            Quadrature weights in direction eta2 in format (jj, jq).

        span1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq).

        span2 : 2d int array
            Knot span indices in direction eta2 in format (jj, jq).

        basis1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (ii, iq, basis function).

        basis2 : 3d float array
            Values of p2 + 1 non-zero eta-2 basis functions at quadrature points in format (jj, jq, basis function).

        dim1_in : int
            Dimension of the first direction of the input space

        dim2_in : int
            Dimension of the second direction of the input space

        p1_out : int
            Spline degree of the first direction of the output space

        p2_out : int
            Spline degree of the second direction of the output space
    '''

    # Start/end indices and paddings for distributed stencil matrix of input space
    # si1 = starts_in[0]
    # si2 = starts_in[1]
    # ei1 = ends_in[0]
    # ei2 = ends_in[1]
    pi1 = pads_in[0]
    pi2 = pads_in[1]

    # Start/end indices for distributed stencil matrix of output space
    so1 = starts_out[0]
    so2 = starts_out[1]
    # eo1 = ends_out[0]
    # eo2 = ends_out[1]
    po1 = pads_out[0]
    po2 = pads_out[1]

    # Spline degrees of input space
    p1 = basis1.shape[2] - 1
    p2 = basis2.shape[2] - 1
    
    # number of quadrature points
    nq1 = span1.shape[1]
    nq2 = span2.shape[1]

    # Set output to zero
    mat[:] = 0.

    # Dimensions of output space
    dim1_out = span1.shape[0]
    dim2_out = span2.shape[0]

    # Interval (either element or sub-interval thereof)
    # -------------------------------------------------
    cumsub_i = 0  # Cumulative sub-interval index
    for ii in range(span1.shape[0]):
        i = ii - cumsub_i  # local DOF index

        cumsub_j = 0  # Cumulative sub-interval index
        for jj in range(span2.shape[0]):
            j = jj - cumsub_j  # local DOF index

            # Quadrature point index in interval
            # ----------------------------------
            for iq in range(nq1):
                for jq in range(nq2):

                    funval = fun_q[nq1*ii + iq, nq2*jj + jq] * wts1[ii, iq] * wts2[jj, jq]

                    # Basis function of input space:
                    # ------------------------------
                    for b1 in range(p1 + 1):
                        m = (span1[ii, iq] - p1 + b1)  # global index
                        # basis value
                        val1 = funval * basis1[ii, iq, b1]

                        # Find column index for _data:
                        if dim1_out <= dim1_in:
                            cut1 = p1
                        else:
                            cut1 = p1_out

                        # Diff of global indices, needs to be adjusted for boundary conditions --> col1
                        col1_tmp = m - (i + so1)
                        if col1_tmp > cut1:
                            m = m - dim1_in
                        elif col1_tmp < -cut1:
                            m = m + dim1_in
                        # add padding
                        col1 = pi1 + m - (i + so1)

                        for b2 in range(p2 + 1):
                            # global index
                            n = (span2[jj, jq] - p2 + b2)
                            value = val1 * basis2[jj, jq, b2]

                            # Find column index for _data:
                            if dim2_out <= dim2_in:
                                cut2 = p2
                            else:
                                cut2 = p2_out

                            # Diff of global indices, needs to be adjusted for boundary conditions --> col2
                            col2_tmp = n - (j + so2)
                            if col2_tmp > cut2:
                                n = n - dim2_in
                            elif col2_tmp < -cut2:
                                n = n + dim2_in
                            # add padding
                            col2 = pi2 + n - (j + so2)

                            # Row index: padding + local index.
                            mat[po1 + i, po2 + j, col1, col2] += value



def assemble_dofs_for_weighted_basisfuns_3d(mat : 'float[:,:,:,:,:,:]', starts_in : 'int[:]', ends_in : 'int[:]', pads_in : 'int[:]', starts_out : 'int[:]', ends_out : 'int[:]', pads_out : 'int[:]', fun_q : 'float[:,:,:]', wts1 : 'float[:,:]', wts2 : 'float[:,:]', wts3 : 'float[:,:]', span1 : 'int[:,:]', span2 : 'int[:,:]', span3 : 'int[:,:]', basis1 : 'float[:,:,:]', basis2 : 'float[:,:,:]', basis3 : 'float[:,:,:]', dim1_in : int, dim2_in : int, dim3_in : int, p1_out : int, p2_out : int, p3_out : int):
    '''Kernel for assembling the matrix

    A_(ijk,mno) = DOFS_ijk(fun*Lambda^in_mno) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_ijk are the degrees-of-freedom of the output space (codomain, must not be a product space), 
    Lambda^in_mno are the basis functions of the input space (domain, must not be a product space), and fun is an arbitrary function.

    Parameters
    ----------
        mat : 6d float array
            _data attribute of StencilMatrix.

        starts_in : 1d int array
            Starting indices of the input space (domain) of a distributed StencilMatrix.

        ends_in : 1d int array
            Ending indices of the input space (domain) of a distributed StencilMatrix.

        pads_in : 1d int array
            Paddings of the input space (domain) of a distributed StencilMatrix.

        starts_out : 1d int array
            Starting indices of the output space (codomain) of a distributed StencilMatrix.

        ends_out : 1d int array
            Ending indices of the output space (codomain) of a distributed StencilMatrix.

        pads_out : 1d int array
            Paddings of the output space (codomain) of a distributed StencilMatrix.

        fun_q : 3d float array
            The function evaluated at the points (nq_i*ii + iq, nq_j*jj + jq, nq_k*kk + kq), where iq a local quadrature point of interval ii.
            
        wts1 : 2d float array
            Quadrature weights in direction eta1 in format (ii, iq).
            
        wts2 : 2d float array
            Quadrature weights in direction eta2 in format (jj, jq).
            
        wts3 : 2d float array
            Quadrature weights in direction eta3 in format (kk, kq).

        span1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq).

        span2 : 2d int array
            Knot span indices in direction eta2 in format (jj, jq).

        span3 : 2d int array
            Knot span indices in direction eta3 in format (kk, kq).

        basis1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (ii, iq, basis function).

        basis2 : 3d float array
            Values of p2 + 1 non-zero eta-2 basis functions at quadrature points in format (jj, jq, basis function).

        basis3 : 3d float array
            Values of p3 + 1 non-zero eta-3 basis functions at quadrature points in format (kk, kq, basis function).

        dim1_in : int
            Dimension of the first direction of the input space

        dim2_in : int
            Dimension of the second direction of the input space

        dim3_in : int
            Dimension of the third direction of the input space

        p1_out : int
            Spline degree of the first direction of the output space

        p2_out : int
            Spline degree of the second direction of the output space

        p3_out : int
            Spline degree of the third direction of the output space
    '''

    # Start/end indices and paddings for distributed stencil matrix of input space
    # si1 = starts_in[0]
    # si2 = starts_in[1]
    # si3 = starts_in[2]
    # ei1 = ends_in[0]
    # ei2 = ends_in[1]
    # ei3 = ends_in[2]
    pi1 = pads_in[0]
    pi2 = pads_in[1]
    pi3 = pads_in[2]

    # Start/end indices for distributed stencil matrix of output space
    so1 = starts_out[0]
    so2 = starts_out[1]
    so3 = starts_out[2]
    # eo1 = ends_out[0]
    # eo2 = ends_out[1]
    # eo3 = ends_out[2]
    po1 = pads_out[0]
    po2 = pads_out[1]
    po3 = pads_out[2]

    # Spline degrees of input space
    p1 = basis1.shape[2] - 1
    p2 = basis2.shape[2] - 1
    p3 = basis3.shape[2] - 1
    
    # number of quadrature points
    nq1 = span1.shape[1]
    nq2 = span2.shape[1]
    nq3 = span3.shape[1]

    # Set output to zero
    mat[:] = 0.

    # Dimensions of output space
    dim1_out = span1.shape[0]
    dim2_out = span2.shape[0]
    dim3_out = span3.shape[0]

    # Interval (either element or sub-interval thereof)
    # -------------------------------------------------
    cumsub_i = 0  # Cumulative sub-interval index
    for ii in range(span1.shape[0]):
        i = ii - cumsub_i  # local DOF index

        cumsub_j = 0  # Cumulative sub-interval index
        for jj in range(span2.shape[0]):
            j = jj - cumsub_j  # local DOF index

            cumsub_k = 0  # Cumulative sub-interval index
            for kk in range(span3.shape[0]):
                k = kk - cumsub_k  # local DOF index

                # Quadrature point index in interval
                # ----------------------------------
                for iq in range(nq1):
                    for jq in range(nq2):
                        for kq in range(nq3):

                            funval = fun_q[nq1*ii + iq, nq2*jj + jq, nq3*kk + kq] * wts1[ii, iq] * wts2[jj, jq] * wts3[kk, kq] 

                            # Basis function of input space:
                            # ------------------------------
                            for b1 in range(p1 + 1):
                                m = (span1[ii, iq] - p1 + b1)  # global index
                                # basis value
                                val1 = funval * basis1[ii, iq, b1]

                                # Find column index for _data:
                                if dim1_out <= dim1_in:
                                    cut1 = p1
                                else:
                                    cut1 = p1_out

                                # Diff of global indices, needs to be adjusted for boundary conditions --> col1
                                col1_tmp = m - (i + so1)
                                if col1_tmp > cut1:
                                    m = m - dim1_in
                                elif col1_tmp < -cut1:
                                    m = m + dim1_in
                                # add padding
                                col1 = pi1 + m - (i + so1)

                                for b2 in range(p2 + 1):
                                    # global index
                                    n = (span2[jj, jq] - p2 + b2)
                                    val2 = val1 * basis2[jj, jq, b2]

                                    # Find column index for _data:
                                    if dim2_out <= dim2_in:
                                        cut2 = p2
                                    else:
                                        cut2 = p2_out

                                    # Diff of global indices, needs to be adjusted for boundary conditions --> col2
                                    col2_tmp = n - (j + so2)
                                    if col2_tmp > cut2:
                                        n = n - dim2_in
                                    elif col2_tmp < -cut2:
                                        n = n + dim2_in
                                    # add padding
                                    col2 = pi2 + n - (j + so2)

                                    for b3 in range(p3 + 1):
                                        # global index
                                        o = (span3[kk, kq] - p3 + b3)
                                        value = val2 * basis3[kk, kq, b3]

                                        # Find column index for _data:
                                        if dim3_out <= dim3_in:
                                            cut3 = p3
                                        else:
                                            cut3 = p3_out

                                        # Diff of global indices, needs to be adjusted for boundary conditions --> col3
                                        col3_tmp = o - (k + so3)
                                        if col3_tmp > cut3:
                                            o = o - dim3_in
                                        elif col3_tmp < -cut3:
                                            o = o + dim3_in
                                        # add padding
                                        col3 = pi3 + o - (k + so3)

                                        # Row index: padding + local index.
                                        mat[po1 + i, po2 + j, po3 + k, col1, col2, col3] += value


def assemble_dofs_for_weighted_basisfuns_1d_ff(mat : 'float[:,:]', starts_in : 'int[:]', ends_in : 'int[:]', pads_in : 'int[:]', starts_out : 'int[:]', ends_out : 'int[:]', pads_out : 'int[:]', starts_c : 'int[:]', ends_c : 'int[:]', pads_c : 'int[:]', wts1 : 'float[:,:]', span1 : 'int[:,:]', basis1 : 'float[:,:,:]', coeffs_f : 'float[:]', span_c1 : 'int[:,:]', basis_c1 : 'float[:,:,:]', dim1_in : int, dim1_c : int, p1_out : int):
    '''Kernel for assembling the matrix

    A_(i,j) = DOFS_i(fun*Lambda^in_j) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_i are the degrees-of-freedom of the output space (codomain, must not be a product space), 
    Lambda^in_j are the basis functions of the input space (domain, must not be a product space), and fun is an arbitrary function.

    Parameters
    ----------
        mat : 2d float array
            _data attribute of StencilMatrix.

        starts_in : int
            Starting index of the input space (domain) of a distributed StencilMatrix.

        ends_in : int
            Ending index of the input space (domain) of a distributed StencilMatrix.

        pads_in : int
            Paddings of the input space (domain) of a distributed StencilMatrix.

        starts_out : int
            Starting indices of the output space (codomain) of a distributed StencilMatrix.

        ends_out : int
            Ending indices of the output space (codomain) of a distributed StencilMatrix.

        pads_out : int
            Paddings of the output space (codomain) of a distributed StencilMatrix.
        
        starts_c : 1d int array
            Starting indices of the coefficient (femfield f) space.

        ends_c : 1d int array
            Ending indices of the coefficient (femfield f) space.

        pads_c : 1d int array
            Paddings of the coefficient (femfield f) space.

        wts1 : 2d float array
            Quadrature weights in format (ii, iq).

        span1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq).

        basis1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (ii, iq, basis function).

        coeffs_f : 3d float array
            Coefficient of the femfield f

        span_c1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq) for the coefficient FemField f.

        basis_c1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions of the space of f at quadrature points in format (ii, iq, basis function).

        dim1_in : int
            Dimension of the first direction of the input space
            
        dim1_c : int
            Dimension of the first direction of the coefficient space

        p1_out : int
            Spline degree of the first direction of the output space
    '''
    
    # Start/end indices and paddings for distributed stencil matrix of input space
    # si1 = starts_in[0}
    # ei1 = ends_in[0]
    pi1 = pads_in[0]

    # Start/end indices for distributed stencil matrix of output space
    so1 = starts_out[0]
    # eo1 = ends_out[0]
    po1 = pads_out[0]

    sc1 = starts_c[0]
    ec1 = ends_c[0]
    pc1 = pads_c[0]

    # Spline degrees of input space
    p1 = basis1.shape[2] - 1

    p1_c = basis_c1.shape[2] - 1

    # number of quadrature points
    nq1 = span1.shape[1]

    # Set output to zero
    mat[:] = 0.

    # Dimensions of output space
    dim1_out = span1.shape[0]

    #local dof index
    for i in range(span1.shape[0]):
        # Quadrature point index in interval
        # ----------------------------------
        for iq in range(nq1):

            f_val = 0.

            for b1 in range(p1_c + 1):
                # global index
                m = (span_c1[i, iq] - p1_c + b1)
                #local index
                m_loc = m-sc1+pc1
                if m_loc>ec1+2*pc1:
                    m_loc-= dim1_c
                f_val += basis_c1[i, iq, b1] * coeffs_f[m_loc] 

            funval = wts1[i, iq] * f_val

            # Basis function of input space:
            # ------------------------------
            for b1 in range(p1 + 1):
                m = (span1[i, iq] - p1 + b1)  # global index
                # basis value
                value = funval * basis1[i, iq, b1]

                # Find column index for _data:
                if dim1_out <= dim1_in:
                    cut1 = p1
                else:
                    cut1 = p1_out

                # Diff of global indices, needs to be adjusted for boundary conditions --> col1
                col1_tmp = m - (i + so1)
                if col1_tmp > cut1:
                    m = m - dim1_in
                elif col1_tmp < -cut1:
                    m = m + dim1_in
                # add padding
                col1 = pi1 + m - (i + so1)

                # Row index: padding + local index.
                mat[po1 + i, col1] += value


def assemble_dofs_for_weighted_basisfuns_2d_ff(mat : 'float[:,:,:,:]', starts_in : 'int[:]', ends_in : 'int[:]', pads_in : 'int[:]', starts_out : 'int[:]', ends_out : 'int[:]', pads_out : 'int[:]', starts_c : 'int[:]', ends_c : 'int[:]', pads_c : 'int[:]', wts1 : 'float[:,:]', wts2 : 'float[:,:]', span1 : 'int[:,:]', span2 : 'int[:,:]', basis1 : 'float[:,:,:]', basis2 : 'float[:,:,:]', coeffs_f : 'float[:,:]', span_c1 : 'int[:,:]', span_c2 : 'int[:,:]', basis_c1 : 'float[:,:,:]', basis_c2 : 'float[:,:,:]', dim1_in : int, dim2_in : int, dim1_c : int, dim2_c : int, p1_out : int, p2_out : int):
    '''Kernel for assembling the matrix

    A_(ij,kl) = DOFS_ij(fun*Lambda^in_kl) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_ij are the degrees-of-freedom of the output space (codomain, must not be a product space), 
    Lambda^in_kl are the basis functions of the input space (domain, must not be a product space), and is a FemField object.

    Parameters
    ----------
        mat : 4d float array
            _data attribute of StencilMatrix.

        starts_in : 1d int array
            Starting indices of the input space (domain) of a distributed StencilMatrix.

        ends_in : 1d int array
            Ending indices of the input space (domain) of a distributed StencilMatrix.

        pads_in : 1d int array
            Paddings of the input space (domain) of a distributed StencilMatrix.

        starts_out : 1d int array
            Starting indices of the output space (codomain) of a distributed StencilMatrix.

        ends_out : 1d int array
            Ending indices of the output space (codomain) of a distributed StencilMatrix.

        pads_out : 1d int array
            Paddings of the output space (codomain) of a distributed StencilMatrix.

        starts_c : 1d int array
            Starting indices of the coefficient (femfield f) space.

        ends_c : 1d int array
            Ending indices of the coefficient (femfield f) space.

        pads_c : 1d int array
            Paddings of the coefficient (femfield f) space.

        wts1 : 2d float array
            Quadrature weights in direction eta1 in format (ii, iq).
            
        wts2 : 2d float array
            Quadrature weights in direction eta2 in format (jj, jq).

        span1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq).

        span2 : 2d int array
            Knot span indices in direction eta2 in format (jj, jq).

        basis1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (ii, iq, basis function).

        basis2 : 3d float array
            Values of p2 + 1 non-zero eta-2 basis functions at quadrature points in format (jj, jq, basis function).

        coeffs_f : 2d float array
            Coefficient of the femfield f

        span_c1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq) for the coefficient FemField f.

        span_c2 : 2d int array
            Knot span indices in direction eta2 in format (jj, jq) for the coefficient FemField f.

        basis_c1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions of the space of f at quadrature points in format (ii, iq, basis function).

        basis_c2 : 3d float array
            Values of p2 + 1 non-zero eta-2 basis functions of the space of f at quadrature points in format (jj, jq, basis function).

        dim1_in : int
            Dimension of the first direction of the input space

        dim2_in : int
            Dimension of the second direction of the input space

        dim1_c : int
            Dimension of the first direction of the coefficient space

        dim2_c : int
            Dimension of the second direction of the coefficient space

        p1_out : int
            Spline degree of the first direction of the output space

        p2_out : int
            Spline degree of the second direction of the output space
    '''

    # Start/end indices and paddings for distributed stencil matrix of input space
    # si1 = starts_in[0]
    # si2 = starts_in[1]
    # ei1 = ends_in[0]
    # ei2 = ends_in[1]
    pi1 = pads_in[0]
    pi2 = pads_in[1]

    # Start/end indices for distributed stencil matrix of output space
    so1 = starts_out[0]
    so2 = starts_out[1]
    # eo1 = ends_out[0]
    # eo2 = ends_out[1]
    po1 = pads_out[0]
    po2 = pads_out[1]

    sc1 = starts_c[0]
    sc2 = starts_c[1]
    ec1 = ends_c[0]
    ec2 = ends_c[1]
    pc1 = pads_c[0]
    pc2 = pads_c[1]

    # Spline degrees of input space
    p1 = basis1.shape[2] - 1
    p2 = basis2.shape[2] - 1

    p1_c = basis_c1.shape[2] - 1
    p2_c = basis_c2.shape[2] - 1
    
    # number of quadrature points
    nq1 = span1.shape[1]
    nq2 = span2.shape[1]

    # Set output to zero
    mat[:] = 0.

    # Dimensions of output space
    dim1_out = span1.shape[0]
    dim2_out = span2.shape[0]


    # Interval (either element or sub-interval thereof)
    # -------------------------------------------------
    # local DOF index
    for i in range(span1.shape[0]):

        for j in range(span2.shape[0]):

            # Quadrature point index in interval
            # ----------------------------------
            for iq in range(nq1):
                for jq in range(nq2):
                    
                    f_val = 0.

                    for b1 in range(p1_c + 1):
                        # global index
                        m = (span_c1[i, iq] - p1_c + b1)
                        #local index
                        m_loc = m-sc1+pc1
                        if m_loc>ec1+2*pc1:
                            m_loc-= dim1_c
                            
                        for b2 in range(p2_c + 1):
                            # global index
                            n = (span_c2[j, jq] - p2_c + b2)
                            #local index
                            n_loc = n-sc2+pc2
                            if n_loc>ec2+2*pc2:
                                n_loc-= dim2_c
                            f_val += basis_c1[i, iq, b1] * basis_c2[j, jq, b2] * coeffs_f[m_loc,n_loc] 


                    funval = wts1[i, iq] * wts2[j, jq] * f_val

                    # Basis function of input space:
                    # ------------------------------
                    for b1 in range(p1 + 1):
                        m = (span1[i, iq] - p1 + b1)  # global index
                        # basis value
                        val1 = funval * basis1[i, iq, b1]

                        # Find column index for _data:
                        if dim1_out <= dim1_in:
                            cut1 = p1
                        else:
                            cut1 = p1_out

                        # Diff of global indices, needs to be adjusted for boundary conditions --> col1
                        col1_tmp = m - (i + so1)
                        if col1_tmp > cut1:
                            m = m - dim1_in
                        elif col1_tmp < -cut1:
                            m = m + dim1_in
                        # add padding
                        col1 = pi1 + m - (i + so1)

                        for b2 in range(p2 + 1):
                            # global index
                            n = (span2[j, jq] - p2 + b2)
                            value = val1 * basis2[j, jq, b2]

                            # Find column index for _data:
                            if dim2_out <= dim2_in:
                                cut2 = p2
                            else:
                                cut2 = p2_out

                            # Diff of global indices, needs to be adjusted for boundary conditions --> col2
                            col2_tmp = n - (j + so2)
                            if col2_tmp > cut2:
                                n = n - dim2_in
                            elif col2_tmp < -cut2:
                                n = n + dim2_in
                            # add padding
                            col2 = pi2 + n - (j + so2)

                            # Row index: padding + local index.
                            mat[po1 + i, po2 + j, col1, col2] += value



def assemble_dofs_for_weighted_basisfuns_3d_ff(mat : 'float[:,:,:,:,:,:]', starts_in : 'int[:]', ends_in : 'int[:]', pads_in : 'int[:]', starts_out : 'int[:]', ends_out : 'int[:]', pads_out : 'int[:]', starts_c : 'int[:]', ends_c : 'int[:]', pads_c : 'int[:]', wts1 : 'float[:,:]', wts2 : 'float[:,:]', wts3 : 'float[:,:]', span1 : 'int[:,:]', span2 : 'int[:,:]', span3 : 'int[:,:]', basis1 : 'float[:,:,:]', basis2 : 'float[:,:,:]', basis3 : 'float[:,:,:]', coeffs_f : 'float[:,:,:]', span_c1 : 'int[:,:]', span_c2 : 'int[:,:]', span_c3 : 'int[:,:]', basis_c1 : 'float[:,:,:]', basis_c2 : 'float[:,:,:]', basis_c3 : 'float[:,:,:]', dim1_in : int, dim2_in : int, dim3_in : int, dim1_c : int, dim2_c : int, dim3_c : int, p1_out : int, p2_out : int, p3_out : int):
    '''Kernel for assembling the matrix

    A_(ijk,mno) = DOFS_ijk(fun*Lambda^in_mno) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_ijk are the degrees-of-freedom of the output space (codomain, must not be a product space), 
    Lambda^in_mno are the basis functions of the input space (domain, must not be a product space), and fun is an arbitrary function.

    Parameters
    ----------
        mat : 6d float array
            _data attribute of StencilMatrix.

        starts_in : 1d int array
            Starting indices of the input space (domain) of a distributed StencilMatrix.

        ends_in : 1d int array
            Ending indices of the input space (domain) of a distributed StencilMatrix.

        pads_in : 1d int array
            Paddings of the input space (domain) of a distributed StencilMatrix.

        starts_out : 1d int array
            Starting indices of the output space (codomain) of a distributed StencilMatrix.

        ends_out : 1d int array
            Ending indices of the output space (codomain) of a distributed StencilMatrix.

        pads_out : 1d int array
            Paddings of the output space (codomain) of a distributed StencilMatrix.

        starts_c : 1d int array
            Starting indices of the coefficient (femfield f) space.

        ends_c : 1d int array
            Ending indices of the coefficient (femfield f) space.

        pads_c : 1d int array
            Paddings of the coefficient (femfield f) space.
 
        wts1 : 2d float array
            Quadrature weights in direction eta1 in format (ii, iq).
            
        wts2 : 2d float array
            Quadrature weights in direction eta2 in format (jj, jq).
            
        wts3 : 2d float array
            Quadrature weights in direction eta3 in format (kk, kq).

        span1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq).

        span2 : 2d int array
            Knot span indices in direction eta2 in format (jj, jq).

        span3 : 2d int array
            Knot span indices in direction eta3 in format (kk, kq).

        basis1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (ii, iq, basis function).

        basis2 : 3d float array
            Values of p2 + 1 non-zero eta-2 basis functions at quadrature points in format (jj, jq, basis function).

        basis3 : 3d float array
            Values of p3 + 1 non-zero eta-3 basis functions at quadrature points in format (kk, kq, basis function).

        coeffs_f : 3d float array
            Coefficient of the femfield f

        span_c1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq) for the coefficient FemField f.

        span_c2 : 2d int array
            Knot span indices in direction eta2 in format (jj, jq) for the coefficient FemField f.

        span_c3 : 2d int array
            Knot span indices in direction eta3 in format (jj, jq) for the coefficient FemField f.

        basis_c1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions of the space of f at quadrature points in format (ii, iq, basis function).

        basis_c2 : 3d float array
            Values of p2 + 1 non-zero eta-2 basis functions of the space of f at quadrature points in format (jj, jq, basis function).
        
        basis_c3 : 3d float array
            Values of p3 + 1 non-zero eta-3 basis functions of the space of f at quadrature points in format (kk, kq, basis function).

        dim1_in : int
            Dimension of the first direction of the input space

        dim2_in : int
            Dimension of the second direction of the input space

        dim3_in : int
            Dimension of the third direction of the input space
        
        dim1_c : int
            Dimension of the first direction of the coefficient space

        dim2_c : int
            Dimension of the second direction of the coefficient space

        dim3_c : int
            Dimension of the third direction of the coefficient space

        p1_out : int
            Spline degree of the first direction of the output space

        p2_out : int
            Spline degree of the second direction of the output space

        p3_out : int
            Spline degree of the third direction of the output space
    '''

    # Start/end indices and paddings for distributed stencil matrix of input space
    # si1 = starts_in[0]
    # si2 = starts_in[1]
    # si3 = starts_in[2]
    # ei1 = ends_in[0]
    # ei2 = ends_in[1]
    # ei3 = ends_in[2]
    pi1 = pads_in[0]
    pi2 = pads_in[1]
    pi3 = pads_in[2]

    # Start/end indices for distributed stencil matrix of output space
    so1 = starts_out[0]
    so2 = starts_out[1]
    so3 = starts_out[2]
    # eo1 = ends_out[0]
    # eo2 = ends_out[1]
    # eo3 = ends_out[2]
    po1 = pads_out[0]
    po2 = pads_out[1]
    po3 = pads_out[2]

    sc1 = starts_c[0]
    sc2 = starts_c[1]
    sc3 = starts_c[2]
    ec1 = ends_c[0]
    ec2 = ends_c[1]
    ec3 = ends_c[2]
    pc1 = pads_c[0]
    pc2 = pads_c[1]
    pc3 = pads_c[2]

    # Spline degrees of input space
    p1 = basis1.shape[2] - 1
    p2 = basis2.shape[2] - 1
    p3 = basis3.shape[2] - 1

    p1_c = basis_c1.shape[2] - 1
    p2_c = basis_c2.shape[2] - 1
    p3_c = basis_c3.shape[2] - 1
    
    # number of quadrature points
    nq1 = span1.shape[1]
    nq2 = span2.shape[1]
    nq3 = span3.shape[1]

    # Set output to zero
    mat[:] = 0.

    # Dimensions of output space
    dim1_out = span1.shape[0]
    dim2_out = span2.shape[0]
    dim3_out = span3.shape[0]

    # Interval (either element or sub-interval thereof)
    # -------------------------------------------------
    # local DOF index
    for i in range(span1.shape[0]):

        for j in range(span2.shape[0]):

            for k in range(span3.shape[0]):

                # Quadrature point index in interval
                # ----------------------------------
                for iq in range(nq1):
                    for jq in range(nq2):
                        for kq in range(nq3):

                            f_val = 0.

                            for b1 in range(p1_c + 1):
                                # global index
                                m = (span_c1[i, iq] - p1_c + b1)
                                #local index
                                m_loc = m-sc1+pc1
                                if m_loc>ec1+2*pc1:
                                    m_loc-= dim1_c
                                for b2 in range(p2_c + 1):
                                    # global index
                                    n = (span_c2[j, jq] - p2_c + b2)
                                    #local index
                                    n_loc = n-sc2+pc2
                                    if n_loc>ec2+2*pc2:
                                        n_loc-= dim2_c
                                    for b3 in range(p3_c + 1):
                                        # global index
                                        o = (span_c3[k, kq] - p3_c + b3)
                                        #local index
                                        o_loc = o-sc3+pc3
                                        if o_loc>ec3+2*pc3:
                                            o_loc-= dim3_c
                                        f_val += basis_c1[i, iq, b1] * basis_c2[j, jq, b2] * basis_c3[k, kq, b3] * coeffs_f[m_loc,n_loc,o_loc] 

                            funval = wts1[i, iq] * wts2[j, jq] * wts3[k, kq] * f_val 

                            # Basis function of input space:
                            # ------------------------------
                            for b1 in range(p1 + 1):
                                m = (span1[i, iq] - p1 + b1)  # global index
                                # basis value
                                val1 = funval * basis1[i, iq, b1]

                                # Find column index for _data:
                                if dim1_out <= dim1_in:
                                    cut1 = p1
                                else:
                                    cut1 = p1_out

                                # Diff of global indices, needs to be adjusted for boundary conditions --> col1
                                col1_tmp = m - (i + so1)
                                if col1_tmp > cut1:
                                    m = m - dim1_in
                                elif col1_tmp < -cut1:
                                    m = m + dim1_in
                                # add padding
                                col1 = pi1 + m - (i + so1)

                                for b2 in range(p2 + 1):
                                    # global index
                                    n = (span2[j, jq] - p2 + b2)
                                    val2 = val1 * basis2[j, jq, b2]

                                    # Find column index for _data:
                                    if dim2_out <= dim2_in:
                                        cut2 = p2
                                    else:
                                        cut2 = p2_out

                                    # Diff of global indices, needs to be adjusted for boundary conditions --> col2
                                    col2_tmp = n - (j + so2)
                                    if col2_tmp > cut2:
                                        n = n - dim2_in
                                    elif col2_tmp < -cut2:
                                        n = n + dim2_in
                                    # add padding
                                    col2 = pi2 + n - (j + so2)

                                    for b3 in range(p3 + 1):
                                        # global index
                                        o = (span3[k, kq] - p3 + b3)
                                        value = val2 * basis3[k, kq, b3]

                                        # Find column index for _data:
                                        if dim3_out <= dim3_in:
                                            cut3 = p3
                                        else:
                                            cut3 = p3_out

                                        # Diff of global indices, needs to be adjusted for boundary conditions --> col3
                                        col3_tmp = o - (k + so3)
                                        if col3_tmp > cut3:
                                            o = o - dim3_in
                                        elif col3_tmp < -cut3:
                                            o = o + dim3_in
                                        # add padding
                                        col3 = pi3 + o - (k + so3)

                                        # Row index: padding + local index.
                                        mat[po1 + i, po2 + j, po3 + k, col1, col2, col3] += value

