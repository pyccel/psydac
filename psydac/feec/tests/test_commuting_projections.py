# -*- coding: UTF-8 -*-

from psydac.feec.global_projectors import Projector_H1, Projector_L2, Projector_Hcurl, Projector_Hdiv
from psydac.feec.derivatives import Gradient_3D, Curl_3D, Divergence_3D
from psydac.fem.tensor import TensorFemSpace, SplineSpace
from psydac.fem.vector import ProductFemSpace
from psydac.linalg.block import BlockVector
from psydac.core.bsplines import make_knots

from mpi4py import MPI
import numpy as np

#==============================================================================
## ----------------------
## function to be derived
## ----------------------
#fun  = lambda xi1 : np.sin( xi1 )
#Dfun = lambda xi1 : np.cos( xi1 )

##-----------------
## Create the grid:
##-----------------
## side lengths of logical cube
#L = 2*np.pi 

## spline degrees
#p = 3   

## periodic boundary conditions (use 'False' if clamped)
#bc = False 

## loop over different number of elements (convergence test)
#Nel_cases = [16]

## loop over different number of quadrature points per element
#Nq_cases = [1, 2, 3, 4, 5, 6, 7, 8]

#for Nel in Nel_cases:
#    
#    print('Nel=', Nel)
#    
#    # element boundaries
#    el_b = np.linspace(0., L, Nel + 1) 

#    # knot sequences
#    T = bsp.make_knots(el_b, p, bc)

#    H1 = TensorFemSpace(SplineSpace(p, knots=T, periodic=False, basis='B'))
#    L2 = H1.reduce_degree(axes=[0], basis='M')

#    # create an instance of the H1 projector class
#    P0 = H1_Projector(H1)

#    # Build gradient
#    grad = Grad(H1, L2)

#    for Nq in Nq_cases:

#        # create an instance of the L2 projector class
#        P1 = L2_Projector(L2, quads=[Nq])

#        # compute coefficients
#        coeffs_0 = P0(fun)
#        coeffs_1 = P1(Dfun)

#        # Compute discrete derivative
#        Dfun_h = grad(coeffs_0)

#        # Test commuting property
#        print( 'Nq=', Nq, np.max( np.abs(coeffs_1.toarray()-Dfun_h.toarray()) ) )
#        
#    print('')

#print(grad)
#print(grad._matrix.tosparse())

#==============================================================================
# Analytical functions
#==============================================================================

fun1    = lambda xi1, xi2, xi3 : np.sin(xi1)*np.sin(xi2)*np.sin(xi3)
D1fun1  = lambda xi1, xi2, xi3 : np.cos(xi1)*np.sin(xi2)*np.sin(xi3)
D2fun1  = lambda xi1, xi2, xi3 : np.sin(xi1)*np.cos(xi2)*np.sin(xi3)
D3fun1  = lambda xi1, xi2, xi3 : np.sin(xi1)*np.sin(xi2)*np.cos(xi3)

fun2    = lambda xi1, xi2, xi3 :   np.sin(2*xi1)*np.sin(2*xi2)*np.sin(2*xi3)
D1fun2  = lambda xi1, xi2, xi3 : 2*np.cos(2*xi1)*np.sin(2*xi2)*np.sin(2*xi3)
D2fun2  = lambda xi1, xi2, xi3 : 2*np.sin(2*xi1)*np.cos(2*xi2)*np.sin(2*xi3)
D3fun2  = lambda xi1, xi2, xi3 : 2*np.sin(2*xi1)*np.sin(2*xi2)*np.cos(2*xi3)

fun3    = lambda xi1, xi2, xi3 :   np.sin(3*xi1)*np.sin(3*xi2)*np.sin(3*xi3)
D1fun3  = lambda xi1, xi2, xi3 : 3*np.cos(3*xi1)*np.sin(3*xi2)*np.sin(3*xi3)
D2fun3  = lambda xi1, xi2, xi3 : 3*np.sin(3*xi1)*np.cos(3*xi2)*np.sin(3*xi3)
D3fun3  = lambda xi1, xi2, xi3 : 3*np.sin(3*xi1)*np.sin(3*xi2)*np.cos(3*xi3)

cf1 = lambda xi1, xi2, xi3 : D2fun3(xi1, xi2, xi3) - D3fun2(xi1, xi2, xi3)
cf2 = lambda xi1, xi2, xi3 : D3fun1(xi1, xi2, xi3) - D1fun3(xi1, xi2, xi3)
cf3 = lambda xi1, xi2, xi3 : D1fun2(xi1, xi2, xi3) - D2fun1(xi1, xi2, xi3)

difun = lambda xi1, xi2, xi3 : D1fun1(xi1, xi2, xi3)+ D2fun2(xi1, xi2, xi3) + D3fun3(xi1, xi2, xi3)

#==============================================================================
# Test parameters
#==============================================================================

# Side lengths of logical cube [0, L]^3
L = [2*np.pi, 2*np.pi , 2*np.pi] 

# Spline degrees
p = [2, 3, 3]

# Periodic boundary conditions (use 'False' if clamped)
bc = [True, False, True]

# FOR TESTING: loop over different number of elements (convergence test)
Nel_cases = [10]

# FOR TESTING: loop over different number of quadrature points per element
Nq_cases = [1, 2, 3, 4, 5]

# FOR TESTING: choose to test 'grad', 'curl', or 'div'
diff = 'curl'

#==============================================================================
# Run test
#==============================================================================

print('-' * 35)
print('Testing commuting property for ' + diff)
print('-' * 35)

for Nel in Nel_cases:
    
    print('Nel = {}'.format(Nel))
    
    # number of elements
    Nel = [Nel, Nel, Nel]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)] 

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i) for el_b_i, p_i, bc_i in zip(el_b, p, bc)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]
    H1     = TensorFemSpace(*Vs, comm=MPI.COMM_WORLD)
    spaces = [H1.reduce_degree(axes=[0], basis='M'),
              H1.reduce_degree(axes=[1], basis='M'),
              H1.reduce_degree(axes=[2], basis='M')]

    Hcurl  = ProductFemSpace(*spaces)
    
    spaces = [H1.reduce_degree(axes=[1,2], basis='M'),
              H1.reduce_degree(axes=[0,2], basis='M'),
              H1.reduce_degree(axes=[0,1], basis='M')]

    Hdiv  = ProductFemSpace(*spaces)
    
    L2  = H1.reduce_degree(axes=[0,1,2], basis='M')

    # create an instance of the H1 projector class
    P0 = Projector_H1(H1)

    # Build linear operators on stencil arrays
    grad = Gradient_3D(H1, Hcurl)
    curl = Curl_3D(Hcurl, Hdiv)
    div  = Divergence_3D(Hdiv, L2)

    for Nq in Nq_cases:
        
        # number of elements
        Nq = [Nq, Nq, Nq]

        # create an instance of the projector class
        P1 = Projector_Hcurl(Hcurl, Nq)
        P2 = Projector_Hdiv(Hdiv, Nq)
        P3 = Projector_L2(L2, Nq)

        #-------------------------------------
        # Projections and discrete derivatives
        #-------------------------------------

        if diff == 'grad':
            u0 = P0(fun1)
            u1 = P1((D1fun1, D2fun1, D3fun1))
            Dfun_h = grad(u0)
            Dfun_proj = u1

        elif diff == 'curl':
            u1 = P1((fun1, fun2, fun3))
            u2 = P2((cf1, cf2, cf3))
            Dfun_h = curl(u1)
            Dfun_proj = u2

        elif diff == 'div':
            u2 = P2((fun1, fun2, fun3))
            u3 = P3(difun)
            Dfun_h = div(u2)
            Dfun_proj = u3

        else:
            raise ValueError("Unrecognized option for 'diff': {}".format(diff))

        # Test commuting property
        print( 'Nq = {};  Error = {:.3e}'.format(Nq, abs((Dfun_proj.coeffs-Dfun_h.coeffs).toarray()).max()) )
        
    print('')
