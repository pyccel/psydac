def evaluate_dofs_1d_1form_array(
        quad_x1:'float[:,:]', # quadrature points
        quad_w1:'float[:,:]', # quadrature weights
        F:'float[:]',       # array of degrees of freedom (intent out)
        f:'float[:,:]'        # input scalar function (callable)
        ):
    k1 = quad_x1.shape[1]

    n1, = F.shape
    for i1 in range(n1):
        F[i1] = 0.0
        for g1 in range(k1):
            F[i1] += quad_w1[i1, g1] * f[i1, g1]
