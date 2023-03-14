from sympde.topology              import Square
from psydac.feec.multipatch.api   import discretize
from sympde.topology              import Derham
from psydac.feec.basis_projectors import BasisProjectionOperator
import numpy as np

def test_basis_projector():
    ### INITIALISATION ###
    domain = Square()
    ncells = (4,4)
    degree = (2,2)
    nquads = [4*(d + 1) for d in degree]
    perio = [True,True]
    domain_h = discretize(domain, ncells=ncells, periodic=perio)

    derham  = Derham(domain, ["H1", "Hdiv", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree, get_vec = True)
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    Xh  = derham_h.Vvec

    P0, P1, P2, PX = derham_h.projectors(nquads=nquads)
    f_x_plus_y = lambda x, y : x+y
    f_cos_x_sin_y = lambda x,y : np.cos(x)*np.sin(y)
    f_exp_x_exp_2y = lambda x,y : np.exp(x)*np.exp(2*y)
    f_x2_plus_y = lambda x,y : x**2+y


    ### TEST V0->V0 ###
    fun = [[f_x_plus_y]]
    P0_0fv = BasisProjectionOperator(P0, V0h, fun)

    const_1 = P0(lambda x,y :1)

    sol_with_op = P0_0fv.dot(const_1.coeffs)
    sol_no_op  = P0(f_x_plus_y) 
    assert(np.allclose(sol_with_op.toarray(),sol_no_op.coeffs.toarray(),1e-12))

    f_test  = P0(f_cos_x_sin_y)
    sol_with_op = P0_0fv.dot(f_test.coeffs)
    sol_no_op  = P0(lambda x, y : f_x_plus_y(x,y)*f_test(x,y))
    assert(np.allclose(sol_with_op.toarray(),sol_no_op.coeffs.toarray(),1e-12))


    ### TEST V1 -> V0 ###
    fun = [[f_x_plus_y,f_x2_plus_y]]
    f_test  = P1([f_cos_x_sin_y, f_exp_x_exp_2y])
    P1_0fv = BasisProjectionOperator(P0, V1h, fun)  
    sol_with_op = P1_0fv.dot(f_test.coeffs)
    sol_no_op  = P0(lambda x, y : f_x_plus_y(x,y)*f_test[0](x,y)+f_x2_plus_y(x,y)*f_test[1](x,y))
    assert(np.allclose(sol_with_op.toarray(),sol_no_op.coeffs.toarray(),1e-12))

if __name__ == '__main__':
    test_basis_projector()