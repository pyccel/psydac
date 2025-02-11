from pyccel.decorators import template
from collections.abc import Callable

#==============================================================================
# 1D DEGREES OF FREEDOM
#==============================================================================

@template(name='T', types=[float, complex])
def evaluate_dofs_1d_0form(
        intp_x1:'T[:]',                 # interpolation points
              F:'T[:]',                 # array of degrees of freedom (intent out)
              f:Callable[['T[:]'], 'T'] # input scalar function (callable)
        ):

    (n1,) = F.shape

    for i1 in range(n1):
        F[i1] = f(intp_x1[i1])

#------------------------------------------------------------------------------
@template(name='T', types=[float, complex])
def evaluate_dofs_1d_1form(
        quad_x1:'T[:,:]',                  # quadrature points
        quad_w1:'T[:,:]',                  # quadrature weights
              F:'T[:]',                    # array of degrees of freedom (intent out)
              f:Callable[['T[:,:]'], 'T']  # input scalar function (callable)
        ):

    k1 = quad_x1.shape[1]

    n1, = F.shape
    for i1 in range(n1):
        F[i1] = 0.0
        for g1 in range(k1):
            F[i1] += quad_w1[i1, g1] * f(quad_x1[i1, g1])

#==============================================================================
# 2D DEGREES OF FREEDOM
#==============================================================================

@template(name='T', types=[float, complex])
def evaluate_dofs_2d_0form(
        intp_x1:'T[:]', intp_x2:'T[:]',         # interpolation points
              F:'T[:,:]',                       # array of degrees of freedom (intent out)
              f:Callable[['T[:]', 'T[:]'], 'T'] # input scalar function (callable)
        ):

    n1, n2 = F.shape

    for i1 in range(n1):
        for i2 in range(n2):
            F[i1, i2] = f(intp_x1[i1], intp_x2[i2])

#------------------------------------------------------------------------------
@template(name='T', types=[float, complex])
def evaluate_dofs_2d_1form_hcurl(
        intp_x1:'T[:]'  , intp_x2:'T[:]'  ,           # interpolation points
        quad_x1:'T[:,:]', quad_x2:'T[:,:]',           # quadrature points
        quad_w1:'T[:,:]', quad_w2:'T[:,:]',           # quadrature weights
             F1:'T[:,:]',      F2:'T[:,:]',           # arrays of degrees of freedom (intent out)
             f1:Callable[['T[:,:]', 'T[:]'  ], 'T'],  # input scalar functions (callable)
             f2:Callable[['T[:]'  , 'T[:,:]'], 'T'],  #
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F1[i1, i2] = 0.0
            for g1 in range(k1):
                F1[i1, i2] += quad_w1[i1, g1] * f1(quad_x1[i1, g1], intp_x2[i2])

    n1, n2 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F2[i1, i2] = 0.0
            for g2 in range(k2):
                F2[i1, i2] += quad_w2[i2, g2] * f2(intp_x1[i1], quad_x2[i2, g2])

#------------------------------------------------------------------------------
@template(name='T', types=[float, complex])
def evaluate_dofs_2d_1form_hdiv(
        intp_x1:'T[:]',   intp_x2:'T[:]'  ,           # interpolation points
        quad_x1:'T[:,:]', quad_x2:'T[:,:]',           # quadrature points
        quad_w1:'T[:,:]', quad_w2:'T[:,:]',           # quadrature weights
             F1:'T[:,:]',      F2:'T[:,:]',           # arrays of degrees of freedom (intent out)
             f1:Callable[['T[:]'  , 'T[:,:]'], 'T'],  # input scalar functions (callable)
             f2:Callable[['T[:,:]', 'T[:]'  ], 'T'],  #
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F1[i1, i2] = 0.0
            for g2 in range(k2):
                F1[i1, i2] += quad_w2[i2, g2] * f1(intp_x1[i1], quad_x2[i2, g2])

    n1, n2 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F2[i1, i2] = 0.0
            for g1 in range(k1):
                F2[i1, i2] += quad_w1[i1, g1] * f2(quad_x1[i1, g1], intp_x2[i2])

#------------------------------------------------------------------------------
@template(name='T', types=[float, complex])
def evaluate_dofs_2d_2form(
        quad_x1:'T[:,:]', quad_x2:'T[:,:]',          # quadrature points
        quad_w1:'T[:,:]', quad_w2:'T[:,:]',          # quadrature weights
              F:'T[:,:]',                            # array of degrees of freedom (intent out)
              f:Callable[['T[:,:]', 'T[:,:]'], 'T'], # input scalar function (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F[i1, i2] = 0.0
            for g1 in range(k1):
                for g2 in range(k2):
                    F[i1, i2] += quad_w1[i1, g1] * quad_w2[i2, g2] * \
                            f(quad_x1[i1, g1], quad_x2[i2, g2])

#------------------------------------------------------------------------------
@template(name='T', types=[float, complex])
def evaluate_dofs_2d_vec(
        intp_x1:'T[:]',   intp_x2:'T[:]'  ,      # interpolation points
             F1:'T[:,:]',      F2:'T[:,:]',      # array of degrees of freedom (intent out)
             f1:Callable[['T[:]', 'T[:]'], 'T'], # input scalar function (callable)
             f2:Callable[['T[:]', 'T[:]'], 'T'], #
        ):

    n1, n2 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F1[i1, i2] = f1(intp_x1[i1], intp_x2[i2])

    n1, n2 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F2[i1, i2] = f2(intp_x1[i1], intp_x2[i2])


#==============================================================================
# 3D DEGREES OF FREEDOM
#==============================================================================
@template(name='T', types=[float, complex])
def evaluate_dofs_3d_0form(
        intp_x1:'T[:]', intp_x2:'T[:]', intp_x3:'T[:]', # interpolation points
              F:'T[:,:,:]',                             # array of degrees of freedom (intent out)
              f:Callable[['T[:]','T[:]','T[:]'], 'T']   # input scalar function (callable)
        ):

    n1, n2, n3 = F.shape

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F[i1, i2, i3] = f(intp_x1[i1], intp_x2[i2], intp_x3[i3])

#------------------------------------------------------------------------------
@template(name='T', types=[float, complex])
def evaluate_dofs_3d_1form(
        intp_x1:'T[:]',   intp_x2:'T[:]',   intp_x3:'T[:]',     # interpolation points
        quad_x1:'T[:,:]', quad_x2:'T[:,:]', quad_x3:'T[:,:]',   # quadrature points
        quad_w1:'T[:,:]', quad_w2:'T[:,:]', quad_w3:'T[:,:]',   # quadrature weights
             F1:'T[:,:,:]',    F2:'T[:,:,:]',    F3:'T[:,:,:]', # arrays of degrees of freedom (intent out)
             f1:Callable[['T[:,:]','T[:]'  ,'T[:]'  ], 'T'],    # input scalar functions (callable)
             f2:Callable[['T[:]'  ,'T[:,:]','T[:]'  ], 'T'],
             f3:Callable[['T[:]'  ,'T[:]'  ,'T[:,:]'], 'T']
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]
    k3 = quad_x3.shape[1]

    n1, n2, n3 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F1[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    F1[i1, i2, i3] += quad_w1[i1, g1] * \
                            f1(quad_x1[i1, g1], intp_x2[i2], intp_x3[i3])

    n1, n2, n3 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F2[i1, i2, i3] = 0.0
                for g2 in range(k2):
                    F2[i1, i2, i3] += quad_w2[i2, g2] * \
                            f2(intp_x1[i1], quad_x2[i2, g2], intp_x3[i3])

    n1, n2, n3 = F3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F3[i1, i2, i3] = 0.0
                for g3 in range(k3):
                    F3[i1, i2, i3] += quad_w3[i3, g3] * \
                            f3(intp_x1[i1], intp_x2[i2], quad_x3[i3, g3])

#------------------------------------------------------------------------------
@template(name='T', types=[float, complex])
def evaluate_dofs_3d_2form(
        intp_x1:'T[:]',   intp_x2:'T[:]',   intp_x3:'T[:]'    , # interpolation points
        quad_x1:'T[:,:]', quad_x2:'T[:,:]', quad_x3:'T[:,:]'  , # quadrature points
        quad_w1:'T[:,:]', quad_w2:'T[:,:]', quad_w3:'T[:,:]'  , # quadrature weights
             F1:'T[:,:,:]',    F2:'T[:,:,:]',    F3:'T[:,:,:]', # arrays of degrees of freedom (intent out)
             f1:Callable[['T[:]'  , 'T[:,:]', 'T[:,:]'], 'T'] , # input scalar functions (callable)
             f2:Callable[['T[:,:]',   'T[:]', 'T[:,:]'], 'T'] , #
             f3:Callable[['T[:,:]', 'T[:,:]', 'T[:]'  ], 'T']   #
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]
    k3 = quad_x3.shape[1]

    n1, n2, n3 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F1[i1, i2, i3] = 0.0
                for g2 in range(k2):
                    for g3 in range(k3):
                        F1[i1, i2, i3] += quad_w2[i2, g2] * quad_w3[i3, g3] * \
                            f1(intp_x1[i1], quad_x2[i2, g2], quad_x3[i3, g3])

    n1, n2, n3 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F2[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g3 in range(k3):
                        F2[i1, i2, i3] += quad_w1[i1, g1] * quad_w3[i3, g3] * \
                            f2(quad_x1[i1, g1], intp_x2[i2], quad_x3[i3, g3])

    n1, n2, n3 = F3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F3[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g2 in range(k2):
                        F3[i1, i2, i3] += quad_w1[i1, g1] * quad_w2[i2, g2] * \
                            f3(quad_x1[i1, g1], quad_x2[i2, g2], intp_x3[i3])

#------------------------------------------------------------------------------
@template(name='T', types=[float, complex])
def evaluate_dofs_3d_3form(
        quad_x1:'T[:,:]', quad_x2:'T[:,:]', quad_x3:'T[:,:]', # quadrature points
        quad_w1:'T[:,:]', quad_w2:'T[:,:]', quad_w3:'T[:,:]', # quadrature weights
              F:'T[:,:,:]',                                   # array of degrees of freedom (intent out)
              f:Callable[['T[:,:]','T[:,:]','T[:,:]'], 'T'],  # input scalar function (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]
    k3 = quad_x3.shape[1]

    n1, n2, n3 = F.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g2 in range(k2):
                        for g3 in range(k3):
                            F[i1, i2, i3] += \
                                    quad_w1[i1, g1] * quad_w2[i2, g2] * quad_w3[i3, g3] * \
                                    f(quad_x1[i1, g1], quad_x2[i2, g2], quad_x3[i3, g3])

#------------------------------------------------------------------------------
@template(name='T', types=[float, complex])
def evaluate_dofs_3d_vec(
        intp_x1:'T[:]',   intp_x2:'T[:]',   intp_x3:'T[:]',     # interpolation points
             F1:'T[:,:,:]',    F2:'T[:,:,:]',    F3:'T[:,:,:]', # arrays of degrees of freedom (intent out)
             f1:Callable[['T[:]', 'T[:]', 'T[:]'], 'T'] ,       # input scalar functions (callable)
             f2:Callable[['T[:]', 'T[:]', 'T[:]'], 'T'] ,       #
             f3:Callable[['T[:]', 'T[:]', 'T[:]'], 'T']         #
        ):

    # evaluate input functions at interpolation points (make sure that points are in [0, 1])
    n1, n2, n3 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F1[i1, i2, i3] = f1(intp_x1[i1], intp_x2[i2], intp_x3[i3])

    n1, n2, n3 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F2[i1, i2, i3] = f2(intp_x1[i1], intp_x2[i2], intp_x3[i3])

    n1, n2, n3 = F3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F3[i1, i2, i3] = f3(intp_x1[i1], intp_x2[i2], intp_x3[i3])