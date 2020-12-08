from sympy import lambdify
# =======================================================================
def pull_v(funcs_ini, mapping):

    mapping  = mapping.get_callable_mapping()
    f1,f2,f3 = mapping._func_eval
    J_inv    = mapping._jacobian_inv

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

    mapping  = mapping.get_callable_mapping()
    f1,f2,f3 = mapping._func_eval

    def fun(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        value = func_ini(x, y, z)
        return value

    return fun

#==============================================================================
def pull_1(funcs_ini, mapping):

    mapping  = mapping.get_callable_mapping()
    f1,f2,f3 = mapping._func_eval
    jacobian = mapping._jacobian

    def fun1(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_T_value = jacobian(xi1, xi2, xi3).T
        value_1 = J_T_value[0,0]*a1_phys + J_T_value[0,1]*a2_phys + J_T_value[0,2]*a3_phys
        return value_1

    def fun2(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_T_value = jacobian(xi1, xi2, xi3).T

        value_2 = J_T_value[1,0]*a1_phys + J_T_value[1,1]*a2_phys + J_T_value[1,2]*a3_phys

        return value_2

    def fun3(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_T_value = jacobian(xi1, xi2, xi3).T

        value_3 = J_T_value[2,0]*a1_phys + J_T_value[2,1]*a2_phys + J_T_value[2,2]*a3_phys
        return value_3

    return fun1, fun2, fun3

#==============================================================================
def pull_2(funcs_ini, mapping):

    mapping    = mapping.get_callable_mapping()
    f1,f2,f3   = mapping._func_eval
    J_inv      = mapping._jacobian_inv
    metric_det = mapping._metric_det

    def fun1(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        a1_phys = funcs_ini[0](x, y, z)
        a2_phys = funcs_ini[1](x, y, z)
        a3_phys = funcs_ini[2](x, y, z)

        J_inv_value = J_inv(xi1, xi2, xi3)
        det_value   = metric_det(xi1, xi2, xi3)**0.5

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
        det_value   = metric_det(xi1, xi2, xi3)**0.5

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
        det_value   = metric_det(xi1, xi2, xi3)**0.5

        value_3 = J_inv_value[2,0]*a1_phys + J_inv_value[2,1]*a2_phys + J_inv_value[2,2]*a3_phys

        return det_value*value_3
 
    return fun1, fun2, fun3

#==============================================================================
def pull_3(func_ini, mapping):

    mapping    = mapping.get_callable_mapping()
    f1,f2,f3   = mapping._func_eval
    metric_det = mapping._metric_det

    def fun(xi1, xi2, xi3):
        x = f1(xi1, xi2, xi3)
        y = f2(xi1, xi2, xi3)
        z = f3(xi1, xi2, xi3)

        det_value = metric_det(xi1, xi2, xi3)**0.5
        value     = func_ini(x, y, z)
        return det_value*value

    return fun

#==============================================================================

def push_1(a1, a2, a3, xi1, xi2, xi3, mapping):

    mapping    = mapping.get_callable_mapping()
    J_inv      = mapping._jacobian_inv

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

    mapping    = mapping.get_callable_mapping()
    J          = mapping._jacobian
    metric_det = mapping._metric_det

    J_value    = J(xi1, xi2, xi3)
    det_value  = metric_det(xi1, xi2, xi3)**0.5

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
