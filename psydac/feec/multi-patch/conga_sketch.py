
# def run_poisson_2d(solution, f, domain, ncells, degree):
#
#     #+++++++++++++++++++++++++++++++
#     # 1. Abstract model
#     #+++++++++++++++++++++++++++++++
#
#     V   = ScalarFunctionSpace('V', domain, kind=None)
#
#     u, v = elements_of(V, names='u, v')
#     nn   = NormalVector('nn')
#
#
#     # MCP: should be imposed by penalization, I am not sure that 'essential' means that
#     bc   = EssentialBC(u, 0, domain.boundary)
#
#     error  = u - solution
#
#     I = domain.interfaces
#
#
#     # expr_I =- 0.5*dot(grad(plus(u)),nn)*minus(v)  + 0.5*dot(grad(minus(v)),nn)*plus(u)  - kappa*plus(u)*minus(v)\
#     #         + 0.5*dot(grad(minus(u)),nn)*plus(v)  - 0.5*dot(grad(plus(v)),nn)*minus(u)  - kappa*plus(v)*minus(u)\
#     #         - 0.5*dot(grad(minus(v)),nn)*minus(u) - 0.5*dot(grad(minus(u)),nn)*minus(v) + kappa*minus(u)*minus(v)\
#     #         - 0.5*dot(grad(plus(v)),nn)*plus(u)   - 0.5*dot(grad(plus(u)),nn)*plus(v)   + kappa*plus(u)*plus(v)
#
#     # global conforming projection
#
#
#
#
#
#     conf_proj = LinearOperator()
#
#
#
#     l2norm = Norm(error, domain, kind='l2')
#     h1norm = Norm(error, domain, kind='h1')
#
#     #+++++++++++++++++++++++++++++++
#     # 2. Discretization
#     #+++++++++++++++++++++++++++++++
#
#     domain_h = discretize(domain, ncells=ncells)
#     Vh       = discretize(V, domain_h, degree=degree)
#
#     equation_h = discretize(equation, domain_h, [Vh, Vh])
#
#     l2norm_h = discretize(l2norm, domain_h, Vh)
#     h1norm_h = discretize(h1norm, domain_h, Vh)
#
#     x  = equation_h.solve()
#
#     uh = VectorFemField( Vh )
#
#     for i in range(len(uh.coeffs[:])):
#         uh.coeffs[i][:,:] = x[i][:,:]
#
#     l2_error = l2norm_h.assemble(u=uh)
#     h1_error = h1norm_h.assemble(u=uh)
#
#     return l2_error, h1_error
#
#





#===============================================================================
class ConformingProjection( LinearOperator ):
    """
    Conforming projection from global broken space to conforming global space

    proj.dot(v) returns the conforming projection of v, computed by solving linear system

    Parameters
    ----------
    todo

    """
    def __init__( self, V0h ):

        # assert isinstance( domain_h , discretized domain )  ## ?
        # assert isinstance( Vh , discretized space )  ## ?

        V0 = V0h.symbolic_space  # (check)
        domain = V0.domain
        domain_h = V0h.domain  # (check)
        
        self._domain   = V0h
        self._codomain = V0h

        u, v = elements_of(V0, names='u, v')
        expr   = dot(u,v)

        kappa  = 10**20
        I = domain.interfaces  # note: interfaces does not include the boundary 
        expr_I = kappa*( plus(u)-minus(u) )*( plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(I, expr_I))


        ah = discretize(a, domain_h, [V0h, V0h])

        self._A = ah.assemble()  #.toarray()

        f = element_of(V0, name='f')
        l = LinearForm(v, f*v)
        self._lh = discretize(l, domain_h, V0h)


    def __call__( self, f ):

        
        b = self._lh.assemble(f=f)
        
        sol_coeffs, info = cg( self._A, b, tol=1e-13, verbose=True )
        
        return  FemField(self.codomain, coeffs=sol_coeffs)
           
        
    def dot( self, f_coeffs, out=None ):

        # todo: build field from coeffs and use __call__
      
        return something




def test_conga_2d():
    """
    assembles a conforming projection on the conforming subspace,
    and the corresponding gradient operator on the broken multipatch space

    performs two tests:
      - commuting diagram
      -

    """


    #+++++++++++++++++++++++++++++++
    # . Domain
    #+++++++++++++++++++++++++++++++

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)
    
    derham_1  = Derham(D1)
    derham_2  = Derham(D2)

    --> here build the local de rham operators
    
    
    
    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    derham  = Derham(domain)  # try


    #+++++++++++++++++++++++++++++++
    # . Discrete space
    #+++++++++++++++++++++++++++++++

    ncells=[2**2, 2**2]
    degree=[2,2]

    domain_h = discretize(domain, ncells=ncells)
    
    ## derham for later
    # derham_h = discretize(derham, domain_h, degree=degree)      # build them by hand if this doesn't work
    # V0h       = derham_h.V0
    # V1h       = derham_h.V1
    
    # broken multipatch spaces
    V0h = discretize(derham.V0, domain_h, degree=degree)
    V1h = discretize(derham.V1, domain_h, degree=degree)
    
    
    #+++++++++++++++++++++++++++++++
    # . Some matrices
    #+++++++++++++++++++++++++++++++

    # identity operator on V0h
    I0 = IdentityMatrix(V0h)

    # mass matrix of V1   (mostly taken from psydac/api/tests/test_api_feec_3d.py)
    u1, v1 = elements_of(derham.V1, names='u1, v1')
    a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    a1_h = discretize(a1, domain_h, (V1h, V1h), backend=PSYDAC_BACKEND_GPYCCEL)
    M1 = a1_h.assemble()  #.tosparse().tocsc()

    #+++++++++++++++++++++++++++++++
    # . Differential operators
    #   on conforming and broken spaces
    #+++++++++++++++++++++++++++++++

    # "broken grad" operator, coincides with the grad on the conforming subspace of V0h
    # later: broken_D0 = Gradient_2D(V0h, V1h)   # on multi-patch domains we should maybe provide the "BrokenGradient"
    # or broken_D0 = derham_h.D0 ?
        
    V0h_0 = V0h.spaces[0]  # V0h on domain 1
    V0h_1 = V0h.spaces[1]  # V0h on domain 2
    V1h_0 = ProductSpace(V1h.spaces[0], V1h.spaces[1])  # V1h on domain 1
    V1h_1 = ProductSpace(V1h.spaces[2], V1h.spaces[3])  # V1h on domain 2
    
    D0_0 = Gradient_2D(V0h_0, V1h_0)
    D0_1 = Gradient_2D(V0h_1, V1h_1)
    
    broken_D0 = BlockMatrix(V0h, V1h, blocks=[[D0_0, None],[None, D0_1]])
    
    # projection from broken multipatch space to conforming subspace
    Pconf_0 = ConformingProjection(V0h)

    # Conga grad operator (on the broken V0h)
    D0 = broken_D0.matmat(Pconf_0)

    # Transpose of the Conga grad operator (using the symmetry of Pconf_0)
    D0_transp = Pconf_0.matmat(broken_D0.T)


    #+++++++++++++++++++++++++++++++
    # . Commuting projectors
    #+++++++++++++++++++++++++++++++

    # create an instance of the H1 projector class
    # P0 = Projector_H1(V0h)
    
    
    P0_1 = ... Projector_H1(V0h_1)
    P0_2 = ... Projector_H1(V0h_1)   # find proper command
    
    -> assemble P0 of u as a BlockVector.
    
    
    
    

    # create an instance of the projector class
    P1 = Projector_Hcurl(V1h)


    #+++++++++++++++++++++++++++++++
    # . test commuting diagram
    #+++++++++++++++++++++++++++++++

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)
    D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)

    u0        = P0(fun1)
    u0_conf   = Pconf_0(u0)
    u1        = P1((D1fun1, D2fun1))
    Dfun_h    = D0(u0)
    Dfun_proj = u1

    # todo: plot the different fields for visual check

    # P0 should map into a conforming function, so we should have u0_conf = u0
    error = (u0.coeffs-u0_conf.coeffs).toarray().max()
    assert abs(error)<1e-9
    print(error)

    # test commuting diagram on the multipatch spaces
    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray().max()
    assert abs(error)<1e-9
    print(error)

    #+++++++++++++++++++++++++++++++
    # . test Poisson solver
    #+++++++++++++++++++++++++++++++

    x,y = domain.coordinates
    solution = x**2 + y**2
    f        = -4

    v = element_of(V0, 'v')
    l = LinearForm(v, f*v)
    b = discretize(l, domain_h, V0h)

    A = D0_transp.matmat(M1.matmat(D0)) + (I0 - Pconf_0)

    solution, info = cg( A, b, tol=1e-13, verbose=True )

    l2_error, h1_error = run_poisson_2d(solution, f, domain, )

    # todo: plot the solution for visual check

    print(l2_error)



