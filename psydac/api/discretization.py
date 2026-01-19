#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add nquads too?

import numpy as np
from sympy import Expr as sym_Expr

from sympde.expr     import BasicForm as sym_BasicForm
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm
from sympde.expr     import Functional as sym_Functional
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Norm as sym_Norm, SemiNorm as sym_SemiNorm
from sympde.expr     import TerminalExpr

from sympde.topology import BasicFunctionSpace
from sympde.topology import VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import Domain
from sympde.topology import Derham
from sympde.topology import LogicalExpr
from sympde.topology import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType

from gelato.expr import GltExpr as sym_GltExpr

from psydac.api.fem_bilinear_form import DiscreteBilinearForm as DiscreteBilinearForm_SF
from psydac.api.fem_sum_form import DiscreteSumForm
from psydac.api.fem          import DiscreteBilinearForm
from psydac.api.fem          import DiscreteLinearForm
from psydac.api.fem          import DiscreteFunctional
from psydac.api.feec         import DiscreteDeRham, MultipatchDiscreteDeRham
from psydac.api.glt          import DiscreteGltExpr
from psydac.api.expr         import DiscreteExpr
from psydac.api.equation     import DiscreteEquation
from psydac.api.utilities    import flatten
from psydac.fem.basic        import FemSpace
from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.fem.partitioning import create_cart, construct_connectivity, construct_interface_spaces, construct_reduced_interface_spaces
from psydac.fem.vector       import MultipatchFemSpace, VectorFemSpace
from psydac.cad.geometry     import Geometry
from psydac.linalg.stencil   import StencilVectorSpace
from psydac.linalg.block     import BlockVectorSpace

__all__ = (
    'discretize',
    'discretize_derham',
    'discretize_derham_multipatch',
    'reduce_space_degrees',
    'discretize_space',
    'discretize_domain'
)

#==============================================================================
def change_dtype(V, dtype):
    """
    Given a FemSpace V, change its underlying coeff_space (i.e. the space of
    its coefficients) so that it matches the required data type.

    Parameters
    ----------
    Vh : FemSpace
        The FEM space, which is modified in place.

    dtype : float or complex
        Datatype of the new coeff_space.

    Returns
    -------
    FemSpace
        The same FEM space passed as input, which was modified in place.
    """
    if not V.coeff_space.dtype == dtype:
        if isinstance(V.coeff_space, BlockVectorSpace):
            # Recreate the BlockVectorSpace
            new_spaces = []
            for v in V.spaces:
                change_dtype(v, dtype)
                new_spaces.append(v.coeff_space)
            V._coeff_space = BlockVectorSpace(*new_spaces, connectivity=V.coeff_space.connectivity)

        # If the coeff_space is a StencilVectorSpace
        else:
            # Recreate the StencilVectorSpace
            interfaces = V.coeff_space.interfaces
            V._coeff_space = StencilVectorSpace(V.coeff_space.cart, dtype=dtype)

            # Recreate the interface in the StencilVectorSpace
            for (axis, ext), interface_space in interfaces.items():
                V.coeff_space.set_interface(axis, ext, interface_space.cart)

    return V

#==============================================================================
def get_max_degree_of_one_space(Vh):
    """
    Get the maximum polynomial degree of a finite element space, along each
    logical (parametric) coordinate.

    Parameters
    ----------
    Vh : FemSpace
        The finite element space under investigation.

    Returns
    -------
    list[int]
        The maximum polynomial degre of Vh with respect to each coordinate.

    """

    if isinstance(Vh, TensorFemSpace):
        return Vh.degree

    elif isinstance(Vh, VectorFemSpace):
        return [max(p) for p in zip(*Vh.degree)]

    elif isinstance(Vh, MultipatchFemSpace):
        degree = [get_max_degree_of_one_space(Vh_i) for Vh_i in Vh.spaces]
        return [max(p) for p in zip(*degree)]

    else:
        raise TypeError(f'Type({V}) not understood')


def get_max_degree(*spaces):
    """
    Get the maximum polynomial degree across several finite element spaces,
    along each logical (parametric) coordinate.

    Parameters
    ----------
    *spaces : tuple[FemSpace]
        The finite element spaces under investigation.

    Returns
    -------
    list[int]
        The maximum polynomial degree across all spaces, with respect to each
        coordinate.

    """
    degree = [get_max_degree_of_one_space(Vh) for Vh in spaces]
    return [max(p) for p in zip(*degree)]

#==============================================================================           
def discretize_derham(derham, domain_h, *, get_H1vec_space=False, **kwargs):
    """
    Create a discrete de Rham sequence from a symbolic one.
    
    This function creates the discrete spaces from the symbolic ones, and then
    creates a DiscreteDeRham object from them.

    Parameters
    ----------
    derham : sympde.topology.space.Derham
        The symbolic Derham sequence.

    domain_h : Geometry
        Discrete domain where the spaces will be discretized.

    get_H1vec_space : bool, default=False
        True to also get the "Hvec" space discretizing (H1)^n vector fields.

    **kwargs : dict
        Optional parameters for the space discretization.

    Returns
    -------
    DiscreteDeRham
      The discrete de Rham sequence containing the discrete spaces, 
      differential operators and projectors.

    See Also
    --------
    discretize_space
    """

    ldim    = derham.shape
    bases   = ['B'] + ldim * ['M']
    spaces  = [discretize_space(V, domain_h, basis=basis, **kwargs)
               for V, basis in zip(derham.spaces, bases)]

    if get_H1vec_space:
        X = VectorFunctionSpace('X', domain_h.domain, kind='h1')
        V0h = spaces[0]
        Xh  = VectorFemSpace(*([V0h]*ldim))
        Xh.symbolic_space = X
        #We still need to specify the symbolic space because of "_recursive_element_of" not implemented in sympde
        spaces.append(Xh)

    return DiscreteDeRham(domain_h, *spaces)

#==============================================================================
def discretize_derham_multipatch(derham, domain_h, **kwargs):
    """
    Create a discrete multipatch de Rham sequence from a symbolic one.
    
    This function creates the broken discrete spaces from the symbolic ones, and then
    creates a MultipatchDiscreteDeRham object from them.

    Parameters
    ----------
    derham : sympde.topology.space.Derham
        The symbolic Derham sequence.

    domain_h : Geometry
        Discrete domain where the spaces will be discretized.

    **kwargs : dict
        Optional parameters for the space discretization.

    Returns
    -------
    MultipatchDiscreteDeRham
      The discrete multipatch de Rham sequence containing the discrete spaces, 
      differential operators and projectors.

    See Also
    --------
    discretize_derham
    discretize_space
    """

    ldim   = derham.shape
    bases  = ['B'] + ldim * ['M']
    spaces = [discretize_space(V, domain_h, basis=basis, **kwargs) \
            for V, basis in zip(derham.spaces, bases)]

    return MultipatchDiscreteDeRham(
        domain_h = domain_h,
        spaces   = spaces
    )

#==============================================================================
def reduce_space_degrees(V, Vh, *, basis='B', sequence='DR'):
    """
    This function takes a tensor FEM space Vh and reduces some degrees in order
    to obtain a tensor FEM space Wh that matches the symbolic space V in a
    certain sequence of spaces. Where the degree is reduced, Wh employs either
    a B-spline or an M-spline basis.

    For example let [p1, p2, p3] indicate the degrees and [r1, r2, r3] indicate
    the interior multiplicites in each direction of the space Vh before
    reduction. The degrees and multiplicities of the reduced spaces are
    specified as follows:
    
    With the 'DR' sequence in 3D, all multiplicies are [r1, r2, r3] and we have
     'H1'   : degree = [p1, p2, p3]
     'Hcurl': degree = [[p1-1, p2, p3], [p1, p2-1, p3], [p1, p2, p3-1]]
     'Hdiv' : degree = [[p1, p2-1, p3-1], [p1-1, p2, p3-1], [p1-1, p2-1, p3]]
     'L2'   : degree = [p1-1, p2-1, p3-1]

    With the 'TH' sequence in 2D we have:
     'H1' : degree = [[p1, p2], [p1, p2]], multiplicity = [[r1, r2], [r1, r2]]
     'L2' : degree = [p1-1, p2-1], multiplicity = [r1-1, r2-1]

    With the 'RT' sequence in 2D we have:
    'H1' : degree = [[p1, p2-1], [p1-1, p2]], multiplicity = [[r1,r2], [r1,r2]]
    'L2' : degree = [p1-1, p2-1], multiplicity = [r1, r2]

    With the 'N' sequence in 2D we have:
    'H1' : degree = [[p1, p2], [p1, p2]], multiplicity = [[r1,r2+1], [r1+1,r2]]
    'L2' : degree = [p1-1, p2-1], multiplicity = [r1, r2]

    For more details see:

      [1] : A. Buffa, J. Rivas, G. Sangalli, and R.G. Vazquez. Isogeometric
      Discrete Differential Forms in Three Dimensions. SIAM J. Numer. Anal.,
      49:818-844, 2011. DOI:10.1137/100786708. (Section 4.1)

      [2] : A. Buffa, C. de Falco, and G. Sangalli. IsoGeometric Analysis:
      Stable elements for the 2D Stokes equation. Int. J. Numer. Meth. Fluids,
      65:1407-1422, 2011. DOI:10.1002/fld.2337. (Section 3)

      [3] : A. Bressan, and G. Sangalli. Isogeometric discretizations of the
      Stokes problem: stability analysis by the macroelement technique. IMA J.
      Numer. Anal., 33(2):629-651, 2013. DOI:10.1093/imanum/drr056.

    Parameters
    ----------
    V : FunctionSpace
        The symbolic space.

    Vh : TensorFemSpace
        The tensor product FEM space.

    basis: str
        The basis function of the reduced spaces, it can be either 'B' for
        B-spline basis or 'M' for M-spline basis.

    sequence: str
        The sequence used to reduce the space. The available choices are:
          'DR': for the de Rham sequence, as described in [1],
          'TH': for Taylor-Hood elements, as described in [2].
        Not implemented yet:
          'N' : for Nedelec elements, as described in [2],
          'RT': for Raviart-Thomas elements, as described in [2].

    Returns
    -------
    Wh : TensorFemSpace, VectorFemSpace
      The reduced space.

    """
    multiplicity = Vh.multiplicity
    if isinstance(V.kind, HcurlSpaceType):
        if sequence == 'DR':
            if V.ldim == 2:
                spaces = [Vh.reduce_degree(axes=[0], multiplicity=multiplicity[0:1], basis=basis),
                          Vh.reduce_degree(axes=[1], multiplicity=multiplicity[1:] , basis=basis)]
            elif V.ldim == 3:
                spaces = [Vh.reduce_degree(axes=[0], multiplicity=multiplicity[0:1], basis=basis),
                          Vh.reduce_degree(axes=[1], multiplicity=multiplicity[1:2], basis=basis),
                          Vh.reduce_degree(axes=[2], multiplicity=multiplicity[2:] , basis=basis)]
            else:
                raise NotImplementedError('TODO')
        else:
            raise NotImplementedError('The sequence {} is not currently available for the space kind {}'.format(sequence, V.kind))
        Wh = VectorFemSpace(*spaces)

    elif isinstance(V.kind, HdivSpaceType):
        if sequence == 'DR':
            if V.ldim == 2:
                spaces = [Vh.reduce_degree(axes=[1], multiplicity=multiplicity[:1], basis=basis),
                          Vh.reduce_degree(axes=[0], multiplicity=multiplicity[1:], basis=basis)]
            elif V.ldim == 3:
                spaces = [Vh.reduce_degree(axes=[1,2], multiplicity=multiplicity[1:], basis=basis),
                          Vh.reduce_degree(axes=[0,2], multiplicity=[multiplicity[0], multiplicity[2]], basis=basis),
                          Vh.reduce_degree(axes=[0,1], multiplicity=multiplicity[:2], basis=basis)]
            else:
                raise NotImplementedError('TODO')
        else:
            raise NotImplementedError('The sequence {} is not currently available for the space kind {}'.format(sequence, V.kind))
        Wh = VectorFemSpace(*spaces)

    elif isinstance(V.kind, L2SpaceType):
        if sequence == 'DR':
            if V.ldim == 1:
                Wh = Vh.reduce_degree(axes=[0], multiplicity=multiplicity, basis=basis)
            elif V.ldim == 2:
                Wh = Vh.reduce_degree(axes=[0,1], multiplicity=multiplicity, basis=basis)
            elif V.ldim == 3:
                Wh = Vh.reduce_degree(axes=[0,1,2], multiplicity=multiplicity, basis=basis)
        elif sequence == 'TH':
            multiplicity = [max(1,m-1) for m in multiplicity]
            if V.ldim == 1:
                Wh = Vh.reduce_degree(axes=[0], multiplicity=multiplicity, basis=basis)
            elif V.ldim == 2:
                Wh = Vh.reduce_degree(axes=[0,1], multiplicity=multiplicity, basis=basis)
            elif V.ldim == 3:
                Wh = Vh.reduce_degree(axes=[0,1,2], multiplicity=multiplicity, basis=basis)
        else:
            raise NotImplementedError('The sequence {} is not currently available for the space kind {}'.format(sequence, V.kind))

    elif isinstance(V.kind, (H1SpaceType, UndefinedSpaceType)):
        Wh = Vh  # Do not reduce space

    else:
        raise NotImplementedError('Cannot create FEM space with kind = {}'.format(V.kind))

    if isinstance(V, VectorFunctionSpace):
        if isinstance(V.kind, (H1SpaceType, L2SpaceType, UndefinedSpaceType)):
            Wh = VectorFemSpace(*[Wh]*V.ldim)

    return Wh

#==============================================================================
# TODO knots
def discretize_space(V, domain_h, *, degree=None, multiplicity=None, knots=None, basis='B', sequence='DR'):
    """
    This function creates the discretized space starting from the symbolic space.

    Parameters
    ----------
    V : <FunctionSpace>
        The symbolic space.

    domain_h   : <Geometry>
        The discretized domain.

    degree : list | dict
        The degree of the h1 space in each direction.

    multiplicity: list | dict
        The multiplicity of knots for the h1 space in each direction.

    knots: list | dict
        The knots sequence of the h1 space in each direction.

    basis: str
        The type of basis function can be 'B' for B-splines or 'M' for M-splines.

    sequence: str
        The sequence used to reduce the space. The available choices are:
          'DR': for the de Rham sequence, as described in [1],
          'TH': for Taylor-Hood elements, as described in [2].
        Not implemented yet:
          'N' : for Nedelec elements, as described in [2],
          'RT': for Raviart-Thomas elements, as described in [2].

    For more details see:

      [1] : A. Buffa, J. Rivas, G. Sangalli, and R.G. Vazquez. Isogeometric
      Discrete Differential Forms in Three Dimensions. SIAM J. Numer. Anal.,
      49:818-844, 2011. DOI:10.1137/100786708. (Section 4.1)

      [2] : A. Buffa, C. de Falco, and G. Sangalli. IsoGeometric Analysis:
      Stable elements for the 2D Stokes equation. Int. J. Numer. Meth. Fluids,
      65:1407-1422, 2011. DOI:10.1002/fld.2337. (Section 3)

      [3] : A. Bressan, and G. Sangalli. Isogeometric discretizations of the
      Stokes problem: stability analysis by the macroelement technique. IMA J.
      Numer. Anal., 33(2):629-651, 2013. DOI:10.1093/imanum/drr056.

    Returns
    -------
    Vh : <FemSpace>
        The discrete FEM space.
    """

#    we have two cases, the case where we have a geometry file,
#    and the case where we have either an analytical mapping or without a mapping.
#    We build the dictionary g_spaces for each interior domain, where it conatians the interiors as keys and the spaces as values,
#    we then create the compatible spaces if needed with the suitable basis functions.

    comm                = domain_h.comm
    ldim                = V.ldim
    is_rational_mapping = False

    assert sequence in ['DR', 'TH', 'N', 'RT']
    if sequence in ['TH', 'N', 'RT']:
        assert isinstance(V, ProductSpace) and len(V.spaces) == 2

    # Define data type of our TensorFemSpace
    dtype = float
    # TODO remove when codomain_type is implemented in SymPDE
    if hasattr(V, 'codomain_type'):
        if V.codomain_type == 'complex':
            dtype = complex

    g_spaces   = {}
    domain     = domain_h.domain

    if len(domain)==1:
        interiors  = [domain.interior]
    else:
        interiors  = list(domain.interior.args)

    connectivity = construct_connectivity(domain)
    if isinstance(domain_h, Geometry) and all(domain_h.mappings.values()):
        # from a discrete geoemtry
        if interiors[0].name in domain_h.mappings:
            mappings  = [domain_h.mappings[inter.name] for inter in interiors]
        else:
            mappings  = [domain_h.mappings[inter.logical_domain.name] for inter in interiors]

        # Get all the FEM spaces from the mapping and convert their coeff_space at the dtype needed
        spaces    = [change_dtype(m.space, dtype) for m in mappings]
        g_spaces  = dict(zip(interiors, spaces))
        spaces    = [S.spaces for S in spaces]

        if not( comm is None ) and ldim == 1:
            raise NotImplementedError('must create a TensorFemSpace in 1d')

    else:

        if isinstance( degree, (list, tuple) ):
            degree = {I.name:degree for I in interiors}
        else:
            assert isinstance(degree, dict)

        if isinstance( multiplicity, (list, tuple) ):
            multiplicity = {I.name:multiplicity for I in interiors}
        elif multiplicity is None:
            multiplicity = {I.name:(1,)*len(degree[I.name]) for I in interiors}
        else:
            assert isinstance(multiplicity, dict)

        if isinstance(knots, (list, tuple)):
            assert len(interiors) == 1
            knots = {interiors[0].name:knots}

        if len(interiors) == 1:
            ddms = [domain_h.ddm]
        else:
            ddms = domain_h.ddm.domains

        spaces = [None]*len(interiors)
        for i,interior in enumerate(interiors):
            ncells     = domain_h.ncells[interior.name]
            periodic   = domain_h.periodic[interior.name]
            degree_i   = degree[interior.name]
            multiplicity_i = multiplicity[interior.name]
            min_coords = interior.min_coords
            max_coords = interior.max_coords

            assert len(ncells) == len(periodic) == len(degree_i)  == len(multiplicity_i) == len(min_coords) == len(max_coords)

            if knots is None:
                # Create uniform grid
                grids = [np.linspace(xmin, xmax, num=ne + 1)
                         for xmin, xmax, ne in zip(min_coords, max_coords, ncells)]

                # Create 1D finite element spaces and precompute quadrature data
                spaces[i] = [SplineSpace( p, multiplicity=m, grid=grid , periodic=P) for p,m,grid,P in zip(degree_i, multiplicity_i,grids, periodic)]
            else:
                 # Create 1D finite element spaces and precompute quadrature data
                spaces[i] = [SplineSpace( p, knots=T , periodic=P) for p,T, P in zip(degree_i, knots[interior.name], periodic)]


        carts    = create_cart(ddms, spaces)
        g_spaces = {inter : TensorFemSpace(ddms[i], *spaces[i], cart=carts[i], dtype=dtype) for i, inter in enumerate(interiors)}


        for i,j in connectivity:
            ((axis_i, ext_i), (axis_j , ext_j)) = connectivity[i, j]
            minus = interiors[i]
            plus  = interiors[j]
            max_ncells = [max(ni,nj) for ni,nj in zip(domain_h.ncells[minus.name],domain_h.ncells[plus.name])]
            g_spaces[minus].add_refined_space(ncells=max_ncells)
            g_spaces[plus].add_refined_space(ncells=max_ncells)

        # ... construct interface spaces
        construct_interface_spaces(domain_h.ddm, g_spaces, carts, interiors, connectivity)

    new_g_spaces = {}
    for inter in g_spaces:
        Vh = g_spaces[inter]
        if isinstance(V, ProductSpace):
            spaces = [reduce_space_degrees(Vi, Vh, basis=basis, sequence=sequence) for Vi in V.spaces]
            spaces = [Vh.spaces if isinstance(Vh, VectorFemSpace) else Vh for Vh in spaces]
            spaces = flatten(spaces)
            Vh     = VectorFemSpace(*spaces)
        else:
            Vh = reduce_space_degrees(V, Vh, basis=basis, sequence=sequence)

        Vh.symbolic_space = V
        for key in Vh._refined_space:
            Vh.get_refined_space(key).symbolic_space = V

        new_g_spaces[inter]   = Vh

    construct_reduced_interface_spaces(g_spaces, new_g_spaces, interiors, connectivity)
    spaces = list(new_g_spaces.values())

    if len(interiors) > 1:
        assert all((isinstance(Wh, FemSpace) and not Wh.is_multipatch) for Wh in spaces)
        Vh = MultipatchFemSpace(*spaces, connectivity=connectivity)
    else:
        assert all(isinstance(Wh, FemSpace) for Wh in spaces)
        if  len(spaces) == 1:
            Vh = spaces[0]
        else:
            Vh = VectorFemSpace(*spaces)
    
    Vh.symbolic_space = V

    return Vh


#==============================================================================
def discretize_domain(domain, *, filename=None, ncells=None, periodic=None, comm=None, mpi_dims_mask=None):

    if comm is not None:
        # Create a copy of the communicator
        comm = comm.Dup()

    if not (filename or ncells):
        raise ValueError("Must provide either 'filename' or 'ncells'")

    elif filename and ncells:
        raise ValueError("Cannot provide both 'filename' and 'ncells'")

    elif filename:
        return Geometry(filename=filename, comm=comm, mpi_dims_mask=mpi_dims_mask)

    elif ncells:
        return Geometry.from_topological_domain(domain, ncells, periodic=periodic, comm=comm, mpi_dims_mask=mpi_dims_mask)

#==============================================================================
def discretize(a, *args, **kwargs):

    if isinstance(a, (sym_BasicForm, sym_GltExpr, sym_Expr)):
        domain_h = args[0]
        assert isinstance(domain_h, Geometry)
        domain  = domain_h.domain
        dim     = domain.dim

        # The current implementation of the sum factorization algorithm does not support OpenMP parallelization
        # It is not properly tested, whether this way of excluding OpenMP parallelized code from using the sum factorization algorithm works.
        backend = kwargs.get('backend')# or None
        assembly_backend = kwargs.get('assembly_backend')# or None
        assembly_backend = backend or assembly_backend
        openmp = False if assembly_backend is None else assembly_backend.get('openmp')

        # Since all keyword arguments are hidden in kwargs, we cannot set the
        # defaults in the function signature. Here we set the defaults for
        # sum_factorization using dict.setdefault: we default to True for
        # bilinear forms in 3D, and to False otherwise. If the user has chosen
        # differently, we let the code run and see what happens!
        #
        # We also default to False if the code is to be parallelized w/ OpenMP.
        # TODO [YG 21.07.2025]: Drop this restriction
        if isinstance(a, sym_BilinearForm):
            default = (dim == 3 and not openmp)
            kwargs.setdefault('sum_factorization', default)

        mapping = domain_h.domain.mapping
        kwargs['symbolic_mapping'] = mapping

    #...
    # In the case of Equation, BilinearForm, LinearForm, or Functional, we
    # need the number of quadrature points along each direction.
    #
    # If not given, we set `nquads[i] = max_p[i] + 1`, where `max_p[i]` is the
    # maximum polynomial degree of the spaces along direction i.
    #
    # If a scalar integer is passed, we use the same number of quadrature
    # points in all directions.
    if isinstance(a, (sym_BasicForm, sym_Equation)):
        nquads = kwargs.get('nquads', None)
        if nquads is None:
            spaces = args[1]
            if not hasattr(spaces, '__iter__'):
                spaces = [spaces]
            nquads = [p + 1 for p in get_max_degree(*spaces)]
        elif not hasattr(nquads, '__iter__'):
            assert isinstance(nquads, int)
            domain_h = args[0]
            nquads = [nquads] * domain_h.ldim
        kwargs['nquads'] = nquads
    #...

    if isinstance(a, sym_BasicForm):
        if isinstance(a, (sym_Norm, sym_SemiNorm)):
            kernel_expr = TerminalExpr(a, domain)
            if not mapping is None:
                kernel_expr = tuple(LogicalExpr(i, domain) for i in kernel_expr)
        else:
            if not mapping is None:
                a       = LogicalExpr (a, domain)
                domain  = domain.logical_domain

            kernel_expr = TerminalExpr(a, domain)

        if len(kernel_expr) > 1:
            return DiscreteSumForm(a, kernel_expr, *args, **kwargs)

    # TODO uncomment when the SesquilinearForm subclass of bilinearForm is create in SymPDE
    # if isinstance(a, sym_SesquilinearForm):
    #     return DiscreteSesquilinearForm(a, kernel_expr, *args, **kwargs)

    if isinstance(a, sym_BilinearForm):
        # Currently sum factorization can only be used for interior domains
        from sympde.expr.evaluation import DomainExpression
        is_interior_expr = isinstance(kernel_expr, DomainExpression)

        if kwargs.pop('sum_factorization') and is_interior_expr:
            return DiscreteBilinearForm_SF(a, kernel_expr, *args, **kwargs)
        else:
            return DiscreteBilinearForm(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_LinearForm):
        return DiscreteLinearForm(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_Functional):
        return DiscreteFunctional(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_Equation):
        return DiscreteEquation(a, *args, **kwargs)

    elif isinstance(a, BasicFunctionSpace):
        return discretize_space(a, *args, **kwargs)
        
    elif isinstance(a, Derham) and not a.V0.is_broken:
        return discretize_derham(a, *args, **kwargs)

    elif isinstance(a, Derham) and a.V0.is_broken:
        return discretize_derham_multipatch(a, *args, **kwargs)

    elif isinstance(a, Domain):
        return discretize_domain(a, *args, **kwargs)

    elif isinstance(a, sym_GltExpr):
        return DiscreteGltExpr(a, *args, **kwargs)
    elif isinstance(a, sym_Expr):
        return DiscreteExpr(a, *args, **kwargs)

    else:
        raise NotImplementedError('given {}'.format(type(a)))
