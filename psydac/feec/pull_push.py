# coding: utf-8

from sympde.topology.callable_mapping import BasicCallableMapping

__all__ = (
    #
    # Pull-back operators
    # -------------------
    'pull_1d_h1',
    'pull_1d_l2',
    'pull_2d_h1',
    'pull_2d_hcurl',
    'pull_2d_hdiv',
    'pull_2d_l2',
    'pull_3d_v',  # NOTE: what is this used for?
    'pull_3d_h1',
    'pull_3d_hcurl',
    'pull_3d_hdiv',
    'pull_3d_l2',
    #
    # Push-forward operators
    # ----------------------
    'push_1d_h1',
    'push_1d_l2',
    'push_2d_h1',
    'push_2d_hcurl',
    'push_2d_hdiv',
    'push_2d_l2',
    'push_3d_h1',
    'push_3d_hcurl',
    'push_3d_hdiv',
    'push_3d_l2',
)

#==============================================================================
# 1D PULL-BACKS
#==============================================================================
def pull_1d_h1(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 1

    def f_logical(eta1):
        x, = F(eta1)
        return f(x)

    return f_logical

#==============================================================================
def pull_1d_l2(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 1

    def f_logical(eta1):
        x, = F(eta1)

        det_value = F.metric_det(eta1)**0.5
        value     = f(x)
        return det_value*value

    return f_logical

#==============================================================================
# 2D PULL-BACKS
#==============================================================================
def pull_2d_h1(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 2

    def f_logical(eta1, eta2):
        x, y = F(eta1, eta2)
        return f(x, y)

    return f_logical

#==============================================================================
def pull_2d_hcurl(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 2

    # Assume that f is a list/tuple of callable functions
    f1, f2 = f

    def f1_logical(eta1, eta2):
        x, y = F(eta1, eta2)

        a1_phys = f1(x, y)
        a2_phys = f2(x, y)

        J_T_value = F.jacobian(eta1, eta2).T
        value_1   = J_T_value[0, 0] * a1_phys + J_T_value[0, 1] * a2_phys
        return value_1

    def f2_logical(eta1, eta2):
        x, y = F(eta1, eta2)

        a1_phys = f1(x, y)
        a2_phys = f2(x, y)

        J_T_value = F.jacobian(eta1, eta2).T
        value_2   = J_T_value[1, 0] * a1_phys + J_T_value[1, 1] * a2_phys

        return value_2

    return f1_logical, f2_logical

#==============================================================================
def pull_2d_hdiv(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 2

    # Assume that f is a list/tuple of callable functions
    f1, f2 = f

    def f1_logical(eta1, eta2):
        x, y = F(eta1, eta2)

        a1_phys = f1(x, y)
        a2_phys = f2(x, y)

        J_inv_value = F.jacobian_inv(eta1, eta2)
        det_value   = F.metric_det(eta1, eta2)**0.5
        value_1     = J_inv_value[0, 0] * a1_phys + J_inv_value[0, 1] * a2_phys

        return det_value * value_1

    def f2_logical(eta1, eta2):
        x, y = F(eta1, eta2)

        a1_phys = f1(x, y)
        a2_phys = f2(x, y)

        J_inv_value = F.jacobian_inv(eta1, eta2)
        det_value   = F.metric_det(eta1, eta2)**0.5
        value_2     = J_inv_value[1, 0] * a1_phys + J_inv_value[1, 1] * a2_phys

        return det_value * value_2

    return f1_logical, f2_logical

#==============================================================================
def pull_2d_l2(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 2

    def f_logical(eta1, eta2):
        x, y = F(eta1, eta2)

        det_value = F.metric_det(eta1, eta2)**0.5
        value     = f(x, y)
        return det_value * value

    return f_logical

#==============================================================================
# 3D PULL-BACKS
#==============================================================================

# TODO [YG 05.10.2022]:
# Remove? But it makes sense to return a vector-valued function...

def pull_3d_v(funcs_ini, mapping):

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
def pull_3d_h1(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 3

    def f_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)
        return f(x, y, z)

    return f_logical

#==============================================================================
def pull_3d_hcurl(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 3

    # Assume that f is a list/tuple of callable functions
    f1, f2, f3 = f

    def f1_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)
        
        a1_phys = f1(x, y, z)
        a2_phys = f2(x, y, z)
        a3_phys = f3(x, y, z)

        J_T_value = F.jacobian(eta1, eta2, eta3).T
        value_1   = J_T_value[0, 0] * a1_phys + J_T_value[0, 1] * a2_phys + J_T_value[0, 2] * a3_phys
        return value_1

    def f2_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)
        
        a1_phys = f1(x, y, z)
        a2_phys = f2(x, y, z)
        a3_phys = f3(x, y, z)

        J_T_value = F.jacobian(eta1, eta2, eta3).T
        value_2   = J_T_value[1, 0] * a1_phys + J_T_value[1, 1] * a2_phys + J_T_value[1, 2] * a3_phys
        return value_2

    def f3_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)
        
        a1_phys = f1(x, y, z)
        a2_phys = f2(x, y, z)
        a3_phys = f3(x, y, z)

        J_T_value = F.jacobian(eta1, eta2, eta3).T
        value_3   = J_T_value[2, 0] * a1_phys + J_T_value[2, 1] * a2_phys + J_T_value[2, 2] * a3_phys
        return value_3

    return f1_logical, f2_logical, f3_logical

#==============================================================================
def pull_3d_hdiv(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 3

    # Assume that f is a list/tuple of callable functions
    f1, f2, f3 = f

    def f1_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)
        
        a1_phys = f1(x, y, z)
        a2_phys = f2(x, y, z)
        a3_phys = f3(x, y, z)

        J_inv_value = F.jacobian_inv(eta1, eta2, eta3)
        det_value   = F.metric_det(eta1, eta2, eta3)**0.5
        value_1     = J_inv_value[0, 0] * a1_phys + J_inv_value[0, 1] * a2_phys + J_inv_value[0, 2] * a3_phys

        return det_value * value_1

    def f2_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)
        
        a1_phys = f1(x, y, z)
        a2_phys = f2(x, y, z)
        a3_phys = f3(x, y, z)

        J_inv_value = F.jacobian_inv(eta1, eta2, eta3)
        det_value   = F.metric_det(eta1, eta2, eta3)**0.5
        value_2     = J_inv_value[1, 0] * a1_phys + J_inv_value[1, 1] * a2_phys + J_inv_value[1, 2] * a3_phys

        return det_value * value_2

    def f3_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)
        
        a1_phys = f1(x, y, z)
        a2_phys = f2(x, y, z)
        a3_phys = f3(x, y, z)

        J_inv_value = F.jacobian_inv(eta1, eta2, eta3)
        det_value   = F.metric_det(eta1, eta2, eta3)**0.5
        value_3     = J_inv_value[2, 0] * a1_phys + J_inv_value[2, 1] * a2_phys + J_inv_value[2, 2] * a3_phys

        return det_value * value_3

    return f1_logical, f2_logical, f3_logical

#==============================================================================
def pull_3d_l2(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 3

    def f_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)

        det_value = F.metric_det(eta1, eta2, eta3)**0.5
        value     = f(x, y, z)
        return det_value*value

    return f_logical

#==============================================================================
# PUSH-FORWARD operators:
#   These push-forward operators take logical coordinates,
#   so they just transform the values of the field.
#   For H1 push-forward there is no transform, so no mapping is involved.
#==============================================================================

#==============================================================================
# 1D PUSH-FORWARD
#==============================================================================
def push_1d_h1(func, xi1):
    return func(xi1)

def push_1d_l2(func, xi1, mapping):

    mapping    = mapping.get_callable_mapping()
    metric_det = mapping._metric_det

    det_value = metric_det(xi1)**0.5
    value     = func(xi1)
    return value/det_value

#==============================================================================
# 2D PUSH-FORWARDS
#==============================================================================
def push_2d_h1(func, xi1, xi2):
    return func(xi1, xi2)

def push_2d_hcurl(a1, a2, xi1, xi2, mapping):

    F = mapping.get_callable_mapping()
    J_inv_value = F.jacobian_inv(xi1, xi2)

    a1_value = a1(xi1, xi2)
    a2_value = a2(xi1, xi2)

    value1 = J_inv_value[0, 0] * a1_value + J_inv_value[1, 0] * a2_value
    value2 = J_inv_value[0, 1] * a1_value + J_inv_value[1, 1] * a2_value

    return value1, value2

#==============================================================================
def push_2d_hdiv(a1, a2, xi1, xi2, mapping):

    mapping    = mapping.get_callable_mapping()
    J          = mapping._jacobian
    metric_det = mapping._metric_det

    J_value    = J(xi1, xi2)
    det_value  = metric_det(xi1, xi2)**0.5

    value1 = ( J_value[0,0]*a1(xi1, xi2) +
               J_value[0,1]*a2(xi1, xi2)) / det_value

    value2 = ( J_value[1,0]*a1(xi1, xi2) +
               J_value[1,1]*a2(xi1, xi2)) / det_value

    return value1, value2

#==============================================================================
def push_2d_l2(func, xi1, xi2, mapping):

    F = mapping.get_callable_mapping()

    #    det_value = F.metric_det(xi1, xi2)**0.5
    # MCP correction: use the determinant of the mapping Jacobian
    J         = F._jacobian
    J_value   = J(xi1, xi2)
    det_value = J_value[0,0]*J_value[1,1]-J_value[1,0]*J_value[0,1]
    value     = func(xi1, xi2)

    return value / det_value

#==============================================================================
# 3D PUSH-FORWARDS
#==============================================================================
def push_3d_h1(func, xi1, xi2, xi3):
    return func(xi1, xi2, xi3)

def push_3d_hcurl(a1, a2, a3, xi1, xi2, xi3, mapping):

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
def push_3d_hdiv(a1, a2, a3, xi1, xi2, xi3, mapping):

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

#==============================================================================
def push_3d_l2(func, xi1, xi2, xi3, mapping):

    mapping    = mapping.get_callable_mapping()
    metric_det = mapping._metric_det

    det_value = metric_det(xi1, xi2, xi3)**0.5
    value     = func(xi1, xi2, xi3)
    return value/det_value
