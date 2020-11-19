from sympy import lambdify
# =======================================================================
def pull_v(funcs_ini, mapping):

    coords   = mapping.logical_coordinates
    exprs    = mapping.expressions
    f1,f2,f3 = lambdify(coords, exprs[0]), lambdify(coords, exprs[1]), lambdify(coords, exprs[2])
    J_inv    = lambdify(coords, mapping.jacobian_inv_expr)

    def fun(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_inv_value = J_inv(xi1, xi2, xi3)
        value_1 = J_inv_value[0,0]*a1_phys + J_inv_value[0,1]*a2_phys + J_inv_value[0,2]*a3_phys
        value_2 = J_inv_value[1,0]*a1_phys + J_inv_value[1,1]*a2_phys + J_inv_value[1,2]*a3_phys
        value_3 = J_inv_value[2,0]*a1_phys + J_inv_value[2,1]*a2_phys + J_inv_value[2,2]*a3_phys
        return value_1, value_2, value_3

    return fun

#==============================================================================
def pull_0(func_ini, mapping):

    coords   = mapping.logical_coordinates
    exprs    = mapping.expressions
    f1,f2,f3 = lambdify(coords, exprs[0]), lambdify(coords, exprs[1]), lambdify(coords, exprs[2])

    def fun(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        value = func_ini(x, y, z)
        return value

    return fun

#==============================================================================
def pull_1(funcs_ini, mapping):

    coords   = mapping.logical_coordinates
    exprs    = mapping.expressions
    f1,f2,f3 = lambdify(coords, exprs[0]), lambdify(coords, exprs[1]), lambdify(coords, exprs[2])
    J_T      = lambdify(coords, mapping.jacobian_expr.T)


    def fun1(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_T_value = J_T(xi1, xi2, xi3)
        value_1 = J_T_value[0,0]*a1_phys + J_T_value[0,1]*a2_phys + J_T_value[0,2]*a3_phys
        return value_1

    def fun2(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_T_value = J_T(xi1, xi2, xi3)

        value_2 = J_T_value[1,0]*a1_phys + J_T_value[1,1]*a2_phys + J_T_value[1,2]*a3_phys

        return value_2

    def fun3(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_T_value = J_T(xi1, xi2, xi3)

        value_3 = J_T_value[2,0]*a1_phys + J_T_value[2,1]*a2_phys + J_T_value[2,2]*a3_phys
        return value_3

    return fun1, fun2, fun3

#==============================================================================
def pull_2(funcs_ini, mapping):

    coords   = mapping.logical_coordinates
    exprs    = mapping.expressions
    f1,f2,f3 = lambdify(coords, exprs[0]), lambdify(coords, exprs[1]), lambdify(coords, exprs[2])
    J_inv    = lambdify(coords, mapping.jacobian_inv_expr)
    det      = lambdify(coords, mapping.metric_det_expr**0.5)

    def fun1(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_inv_value = J_inv(xi1, xi2, xi3)
        det_value   = det(xi1, xi2, xi3)

        value_1 = J_inv_value[0,0]*a1_phys + J_inv_value[0,1]*a2_phys + J_inv_value[0,2]*a3_phys

        return det_value*value_1

    def fun2(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_inv_value = J_inv(xi1, xi2, xi3)
        det_value   = det(xi1, xi2, xi3)

        value_2 = J_inv_value[1,0]*a1_phys + J_inv_value[1,1]*a2_phys + J_inv_value[1,2]*a3_phys

        return det_value*value_2

    def fun3(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_inv_value = J_inv(xi1, xi2, xi3)
        det_value   = det(xi1, xi2, xi3)

        value_3 = J_inv_value[2,0]*a1_phys + J_inv_value[2,1]*a2_phys + J_inv_value[2,2]*a3_phys

        return det_value*value_3
 
    return fun1, fun2, fun3

#==============================================================================
def pull_3(func_ini, mapping):

    coords   = mapping.logical_coordinates
    exprs    = mapping.expressions
    f1,f2,f3 = lambdify(coords, exprs[0]), lambdify(coords, exprs[1]), lambdify(coords, exprs[2])
    det      = lambdify(coords, mapping.metric_det_expr**0.5)

    def fun(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        det_value = det(xi1, xi2, xi3)
        value     = func_ini(x, y, z)
        return det_value*value

    return fun

#==============================================================================

def push_1(a1, a2, a3, xi1, xi2, xi3, mapping):
    

    coords      = mapping.logical_coordinates
    exprs       = mapping.expressions
    f1,f2,f3    = lambdify(coords, exprs[0]), lambdify(coords, exprs[1]), lambdify(coords, exprs[2])
    J_inv       = lambdify(coords, mapping.jacobian_inv_expr)
    J_inv_value = J_inv(xi1, xi2, xi3)

    value1 = (J_inv_value[0,0]*a1(xi1, xi2, xi3) +
              J_inv_value[1,0]*a2(xi1, xi2, xi3) +
              J_inv_value[2,0]*a3(xi1, xi2, xi3) )

    value2 = (J_inv_value[0,1]*a1(xi1, xi2, xi3) +
              J_inv_value[1,1]*a2(xi1, xi2, xi3) +
              J_inv_value[2,1]*a3(xi1, xi2, xi3) )

    value3 = (J_inv_value[0,2]*a1(xi1, xi2, xi3) +
              J_inv_value[1,2]*a2(xi1, xi2, xi3) +
              J_inv_value[2,2]*a3(xi1, xi2, xi3) )

    return value1, value2, value3


#==============================================================================
def push_2(a1, a2, a3, xi1, xi2, xi3, mapping):

    coords    = mapping.logical_coordinates
    exprs     = mapping.expressions
    f1,f2,f3  = lambdify(coords, exprs[0]), lambdify(coords, exprs[1]), lambdify(coords, exprs[2])
    J         = lambdify(coords, mapping.jacobian_expr)
    J_value   = J(xi1, xi2, xi3)
    det       = lambdify(coords, mapping.metric_det_expr**0.5)
    det_value = det(xi1, xi2, xi3)

    value1 = ( J_value[0,0]*a1(xi1, xi2, xi3) +
               J_value[0,1]*a2(xi1, xi2, xi3) +
               J_value[0,2]*a3(xi1, xi2, xi3) ) / det_value

    value2 = ( J_value[1,0]*a1(xi1, xi2, xi3) +
               J_value[1,1]*a2(xi1, xi2, xi3) +
               J_value[1,2]*a3(xi1, xi2, xi3) ) / det_value

    value3 = ( J_value[2,0]*a1(xi1, xi2, xi3) +
               J_value[2,1]*a2(xi1, xi2, xi3) +
               J_value[2,2]*a3(xi1, xi2, xi3) ) / det_value

    return value1, value2, value3
