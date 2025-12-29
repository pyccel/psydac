#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympde.topology.callable_mapping import BasicCallableMapping


__all__ = (
    #
    # Pull-back operators
    # -------------------
    'pull_1d_h1',
    'pull_1d_l2',
    'pull_2d_h1vec',
    'pull_2d_h1',
    'pull_2d_hcurl',
    'pull_2d_hdiv',
    'pull_2d_l2',
    'pull_3d_h1vec',  # NOTE: what is this used for?
    'pull_3d_h1',
    'pull_3d_hcurl',
    'pull_3d_hdiv',
    'pull_3d_l2',
    #
    # Push-forward operators
    # ----------------------
    'push_1d_h1',
    'push_1d_l2',
    'push_2d_h1_vec',
    'push_2d_h1',
    'push_2d_hcurl',
    'push_2d_hdiv',
    'push_2d_l2',
    'push_3d_h1',
    'push_3d_hcurl',
    'push_3d_hdiv',
    'push_3d_l2'
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
        return det_value * value

    return f_logical

#==============================================================================
# 2D PULL-BACKS
#==============================================================================
def pull_2d_h1vec(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 2    

    f1, f2 = f

    def f1_logical(eta1, eta2):
        x, y = F(eta1, eta2)

        a1_phys = f1(x, y)
        a2_phys = f2(x, y)

        J_inv_value = F.jacobian_inv(eta1, eta2)
        value_1 = J_inv_value[0, 0] * a1_phys + J_inv_value[0, 1] * a2_phys
        return value_1

    def f2_logical(eta1, eta2):
        x, y = F(eta1, eta2)

        a1_phys = f1(x, y)
        a2_phys = f2(x, y)

        J_inv_value = F.jacobian_inv(eta1, eta2)
        value_2 = J_inv_value[1, 0] * a1_phys + J_inv_value[1, 1] * a2_phys
        return value_2

    return f1_logical, f2_logical

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

def pull_3d_h1vec(f, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 3

    f1, f2, f3 = f

    def f1_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)

        a1_phys = f1(x, y, z)
        a2_phys = f2(x, y, z)
        a3_phys = f3(x, y, z)

        J_inv_value = F.jacobian_inv(eta1, eta2, eta3)
        value_1 = J_inv_value[0, 0] * a1_phys + J_inv_value[0, 1] * a2_phys + J_inv_value[0, 2] * a3_phys
        return value_1

    def f2_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)

        a1_phys = f1(x, y, z)
        a2_phys = f2(x, y, z)
        a3_phys = f3(x, y, z)

        J_inv_value = F.jacobian_inv(eta1, eta2, eta3)
        value_2 = J_inv_value[1, 0] * a1_phys + J_inv_value[1, 1] * a2_phys + J_inv_value[1, 2] * a3_phys
        return value_2

    def f3_logical(eta1, eta2, eta3):
        x, y, z = F(eta1, eta2, eta3)

        a1_phys = f1(x, y, z)
        a2_phys = f2(x, y, z)
        a3_phys = f3(x, y, z)

        J_inv_value = F.jacobian_inv(eta1, eta2, eta3)
        value_2 = J_inv_value[2, 0] * a1_phys + J_inv_value[2, 1] * a2_phys + J_inv_value[2, 2] * a3_phys
        return value_2

    return f1_logical, f2_logical, f3_logical

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
        return det_value * value

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
def push_1d_h1(f, eta):
    return f(eta)

def push_1d_l2(f, eta, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 1

    return f(eta) / F.metric_det(eta)**0.5

#==============================================================================
# 2D PUSH-FORWARDS
#==============================================================================
def push_2d_h1_vec(f1, f2, eta1, eta2):
    eta = eta1, eta2
    return f1(*eta), f2(*eta)

def push_2d_h1(f, eta1, eta2):
    eta = eta1, eta2
    return f(*eta)

#def push_2d_hcurl(f, eta, F):
def push_2d_hcurl(f1, f2, eta1, eta2, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 2

#    # Assume that f is a list/tuple of callable functions
#    f1, f2 = f
    eta = eta1, eta2

    J_inv_value = F.jacobian_inv(*eta)

    f1_value = f1(*eta)
    f2_value = f2(*eta)

    value1 = J_inv_value[0, 0] * f1_value + J_inv_value[1, 0] * f2_value
    value2 = J_inv_value[0, 1] * f1_value + J_inv_value[1, 1] * f2_value

    return value1, value2

#==============================================================================
#def push_2d_hdiv(f, eta, F):
def push_2d_hdiv(f1, f2, eta1, eta2, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 2

#    # Assume that f is a list/tuple of callable functions
#    f1, f2 = f
    eta = eta1, eta2

    J_value   = F.jacobian(*eta)
    det_value = F.metric_det(*eta)**0.5

    f1_value = f1(*eta)
    f2_value = f2(*eta)

    value1 = (J_value[0, 0] * f1_value + J_value[0, 1] * f2_value) / det_value
    value2 = (J_value[1, 0] * f1_value + J_value[1, 1] * f2_value) / det_value

    return value1, value2

#==============================================================================
#def push_2d_l2(f, eta, F):
def push_2d_l2(f, eta1, eta2, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 2

    eta = eta1, eta2

    #    det_value = F.metric_det(eta1, eta2)**0.5
    # MCP correction: use the determinant of the mapping Jacobian
    J_value   = F.jacobian(*eta)
    det_value = J_value[0, 0] * J_value[1, 1] - J_value[1, 0] * J_value[0, 1]

    return f(*eta) / det_value

#==============================================================================
# 3D PUSH-FORWARDS
#==============================================================================
#def push_3d_h1(f, eta):
def push_3d_h1(f, eta1, eta2, eta3):
    eta = eta1, eta2, eta3
    return f(*eta)

#def push_3d_hcurl(f, eta, F):
def push_3d_hcurl(f1, f2, f3, eta1, eta2, eta3, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 3

#    # Assume that f is a list/tuple of callable functions
#    f1, f2, f3 = f
    eta = eta1, eta2, eta3

    f1_value = f1(*eta)
    f2_value = f2(*eta)
    f3_value = f3(*eta)

    J_inv_value = F.jacobian_inv(*eta)

    value1 = (J_inv_value[0, 0] * f1_value +
              J_inv_value[1, 0] * f2_value +
              J_inv_value[2, 0] * f3_value )

    value2 = (J_inv_value[0, 1] * f1_value +
              J_inv_value[1, 1] * f2_value +
              J_inv_value[2, 1] * f3_value )

    value3 = (J_inv_value[0, 2] * f1_value +
              J_inv_value[1, 2] * f2_value +
              J_inv_value[2, 2] * f3_value )

    return value1, value2, value3

#==============================================================================
#def push_3d_hdiv(f, eta, F):
def push_3d_hdiv(f1, f2, f3, eta1, eta2, eta3, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 3

#    # Assume that f is a list/tuple of callable functions
#    f1, f2, f3 = f
    eta = eta1, eta2, eta3

    f1_value  = f1(*eta)
    f2_value  = f2(*eta)
    f3_value  = f3(*eta)
    J_value   = F.jacobian(*eta)
    det_value = F.metric_det(*eta)**0.5

    value1 = ( J_value[0, 0] * f1_value +
               J_value[0, 1] * f2_value +
               J_value[0, 2] * f3_value ) / det_value

    value2 = ( J_value[1, 0] * f1_value +
               J_value[1, 1] * f2_value +
               J_value[1, 2] * f3_value ) / det_value

    value3 = ( J_value[2, 0] * f1_value +
               J_value[2, 1] * f2_value +
               J_value[2, 2] * f3_value ) / det_value

    return value1, value2, value3

#==============================================================================
#def push_3d_l2(f, eta, F):
def push_3d_l2(f, eta1, eta2, eta3, F):

    assert isinstance(F, BasicCallableMapping)
    assert F.ldim == 3

    eta = eta1, eta2, eta3

    det_value = F.metric_det(*eta)**0.5

    return f(*eta) / det_value
