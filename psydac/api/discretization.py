# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?
import os
from sympy import Expr as sym_Expr
import numpy as np

from sympde.expr     import BasicForm as sym_BasicForm
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm
from sympde.expr     import Functional as sym_Functional
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Norm as sym_Norm
from sympde.expr     import TerminalExpr
from sympde.topology import Domain, Interface
from sympde.topology import Line, Square, Cube
from sympde.topology import BasicFunctionSpace
from sympde.topology import VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import Derham
from sympde.topology import LogicalExpr
from sympde.topology import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType
from sympde.topology.basic import Union

from gelato.expr import GltExpr as sym_GltExpr

from psydac.api.fem          import DiscreteBilinearForm
from psydac.api.fem          import DiscreteLinearForm
from psydac.api.fem          import DiscreteFunctional
from psydac.api.fem          import DiscreteSumForm
from psydac.api.feec         import DiscreteDerham
from psydac.api.glt          import DiscreteGltExpr
from psydac.api.expr         import DiscreteExpr
from psydac.api.equation     import DiscreteEquation
from psydac.api.utilities    import flatten
from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.fem.vector       import ProductFemSpace
from psydac.cad.geometry     import Geometry
from psydac.mapping.discrete import NurbsMapping

__all__ = ('discretize',)

#==============================================================================           
def discretize_derham(derham, domain_h, *args, **kwargs):

    ldim     = derham.shape
    mapping  = derham.spaces[0].domain.mapping

    bases  = ['B'] + ldim * ['M']
    spaces = [discretize_space(V, domain_h, *args, basis=basis, **kwargs) \
            for V, basis in zip(derham.spaces, bases)]

    return DiscreteDerham(mapping, *spaces)

#==============================================================================
def reduce_space_degrees(V, Vh, basis='B', sequence='DR'):
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
        The tensor product fem space.

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

    Results
    -------
    Wh : TensorFemSpace, ProductFemSpace
      The reduced space

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
        Wh = ProductFemSpace(*spaces)

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
        Wh = ProductFemSpace(*spaces)

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
            Wh = ProductFemSpace(*[Wh]*V.ldim)

    return Wh

def create_cart(spaces, interiors, comm, interfaces=None, nprocs=None, reverse_axis=None):

    from psydac.ddm.cart import CartDecomposition, MultiCartDecomposition, InterfacesCartDecomposition
    num_threads     = int(os.environ.get('OMP_NUM_THREADS',1))
    interfaces_cart = None

    if len(interiors) == 1:
        spaces = spaces[0]
        npts         = [V.nbasis   for V in spaces]
        pads         = [V._pads    for V in spaces]
        degree       = [V.degree   for V in spaces]
        multiplicity = [V.multiplicity for V in spaces]
        periods      = [V.periodic for V in spaces]

        cart = CartDecomposition(
            npts         = npts,
            pads         = pads,
            periods      = periods,
            reorder      = True,
            comm         = comm,
            shifts       = multiplicity,
            nprocs       = nprocs,
            reverse_axis = reverse_axis,
            num_threads  = num_threads
        )
    else:
        npts         = [[V.nbasis   for V in space_i] for space_i in spaces]
        pads         = [[V._pads    for V in space_i] for space_i in spaces]
        degree       = [[V.degree   for V in space_i] for space_i in spaces]
        multiplicity = [[V.multiplicity for V in space_i] for space_i in spaces]
        periods      = [[V.periodic for V in space_i] for space_i in spaces]

        cart = MultiCartDecomposition(
            npts         = npts,
            pads         = pads,
            periods      = periods,
            reorder      = True,
            comm         = comm,
            shifts       = multiplicity,
            num_threads  = num_threads)

        if interfaces:
            interfaces_info = {}
            for e in interfaces:
                i = interiors.index(e.minus.domain)
                j = interiors.index(e.plus.domain)
                interfaces_info[i, j] = ((e.minus.axis, e.plus.axis),(e.minus.ext, e.plus.ext))

            interfaces_cart = InterfacesCartDecomposition(cart, interfaces_info)

    return cart, interfaces_cart


#==============================================================================
# TODO knots
def discretize_space(V, domain_h, *args, **kwargs):
    """
    This function creates the discretized space starting from the symbolic space.

    Parameters
    ----------

    V : <FunctionSpace>
        the symbolic space

    domain_h   : <Geometry>
        the discretized domain

    Returns
    -------
    Vh : <FemSpace>
        represents the discrete fem space

    """

#    we have two cases, the case where we have a geometry file,
#    and the case where we have either an analytical mapping or without a mapping.
#    We build the dictionary g_spaces for each interior domain, where it conatians the interiors as keys and the spaces as values,
#    we then create the compatible spaces if needed with the suitable basis functions.

    degree              = kwargs.pop('degree', None)
    comm                = domain_h.comm
    ldim                = V.ldim
    periodic            = kwargs.pop('periodic', [False]*ldim)
    basis               = kwargs.pop('basis', 'B')
    knots               = kwargs.pop('knots', None)
    quad_order          = kwargs.pop('quad_order', None)
    sequence            = kwargs.pop('sequence', 'DR')
    is_rational_mapping = False

    assert sequence in ['DR', 'TH', 'N', 'RT']
    if sequence in ['TH', 'N', 'RT']:
        assert isinstance(V, ProductSpace) and len(V.spaces) == 2

    g_spaces = {}
    if isinstance(domain_h, Geometry) and all(domain_h.mappings.values()):
        # from a discrete geoemtry
        if len(domain_h.mappings.values()) > 1:
            raise NotImplementedError('Multipatch not yet available')

        interiors = [domain_h.domain.interior]
        mappings  = [domain_h.mappings[inter.logical_domain.name] for inter in interiors]
        spaces    = [m.space for m in mappings]
        g_spaces  = dict(zip(interiors, spaces))

        if not( comm is None ) and ldim == 1:
            raise NotImplementedError('must create a TensorFemSpace in 1d')

    elif not( degree is None ):

        assert(isinstance( degree, (list, tuple) ))
        assert( len(degree) == ldim )
        assert(hasattr(domain_h, 'ncells'))
        interiors = domain_h.domain.interior
        if isinstance(interiors, Union):
            interiors = interiors.args
            interfaces = domain_h.domain.interfaces

            if isinstance(interfaces, Interface):
                interfaces = [interfaces]
            elif isinstance(interfaces, Union):
                interfaces = interfaces.args
            else:
                interfaces = []
        else:
            interiors = [interiors]

        if isinstance(knots, (list, tuple)):
            assert len(interiors) == 1
            knots = {interior.name:knots}

        spaces = [None]*len(interiors)
        for i,interior in enumerate(interiors):
            ncells     = domain_h.ncells
            min_coords = interior.min_coords
            max_coords = interior.max_coords

            if knots is None:
                # Create uniform grid
                grids = [np.linspace(xmin, xmax, num=ne + 1)
                         for xmin, xmax, ne in zip(min_coords, max_coords, ncells)]

                # Create 1D finite element spaces and precompute quadrature data
                spaces[i] = [SplineSpace( p, grid=grid , periodic=P) for p,grid, P in zip(degree, grids, periodic)]
            else:
                 # Create 1D finite element spaces and precompute quadrature data
                spaces[i] = [SplineSpace( p, knots=T , periodic=P) for p,T, P in zip(degree, knots[interior.name], periodic)]

        if comm is not None:
            cart, interfaces_cart = create_cart(spaces, interiors, comm, interfaces=interfaces, nprocs=None, reverse_axis=None)

            if interfaces_cart:
                interfaces_cart = interfaces_cart.carts

            if len(interiors) == 1:
                carts = [cart]
            else:
                carts = cart.carts

        for i,interior in enumerate(interiors):
            if comm is not None:
                Vh = TensorFemSpace( *spaces[i], cart=carts[i], quad_order=quad_order)
            else:
                Vh = TensorFemSpace( *spaces[i], quad_order=quad_order)

            g_spaces[interior] = Vh

        for e in interfaces:
            i = interiors.index(e.minus.domain)
            j = interiors.index(e.plus.domain)
            print('start', comm.rank)
            if comm is not None and not interfaces_cart[i,j].is_comm_null:
                if not carts[i].is_comm_null:
                    assert carts[j].is_comm_null
                    g_spaces[e.plus.domain] = TensorFemSpace( *spaces[j], cart=interfaces_cart[i,j], quad_order=quad_order)
                elif not carts[j].is_comm_null:
                    assert carts[i].is_comm_null
                    g_spaces[e.minus.domain] = TensorFemSpace( *spaces[i], cart=interfaces_cart[i,j], quad_order=quad_order)

            print('end', comm.rank)
    for inter in g_spaces:
        Vh = g_spaces[inter]
        if isinstance(V, ProductSpace):
            spaces = [reduce_space_degrees(Vi, Vh, basis=basis, sequence=sequence) for Vi in V.spaces]
            spaces = [Vh.spaces if isinstance(Vh, ProductFemSpace) else Vh for Vh in spaces]
            spaces = flatten(spaces)
            Vh     = ProductFemSpace(*spaces)
        else:
            Vh = reduce_space_degrees(V, Vh, basis=basis, sequence=sequence)

        Vh.symbolic_space = V
        g_spaces[inter]    = Vh

    Vh = ProductFemSpace(*g_spaces.values())

    Vh.symbolic_space      = V

    return Vh

#==============================================================================
def discretize_domain(domain, *, filename=None, ncells=None, comm=None):

#    if comm is not None:
#        # Create a copy of the communicator
#        comm = comm.Dup()

    if not (filename or ncells):
        raise ValueError("Must provide either 'filename' or 'ncells'")

    elif filename and ncells:
        raise ValueError("Cannot provide both 'filename' and 'ncells'")

    elif filename:
        return Geometry(filename=filename, comm=comm)

    elif ncells:
        return Geometry.from_topological_domain(domain, ncells, comm)

#==============================================================================
def discretize(a, *args, **kwargs):

    if isinstance(a, (sym_BasicForm, sym_GltExpr, sym_Expr)):
        domain_h = args[0]
        assert( isinstance(domain_h, Geometry) )
        domain  = domain_h.domain
        mapping = domain_h.domain.mapping
        kwargs['mapping'] = mapping

    if isinstance(a, sym_BasicForm):
        if isinstance(a, sym_Norm):
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

    if isinstance(a, sym_BilinearForm):
        return DiscreteBilinearForm(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_LinearForm):
        return DiscreteLinearForm(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_Functional):
        return DiscreteFunctional(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_Equation):
        return DiscreteEquation(a, *args, **kwargs)

    elif isinstance(a, BasicFunctionSpace):
        return discretize_space(a, *args, **kwargs)
        
    elif isinstance(a, Derham):
        return discretize_derham(a, *args, **kwargs)

    elif isinstance(a, Domain):
        return discretize_domain(a, *args, **kwargs)

    elif isinstance(a, sym_GltExpr):
        return DiscreteGltExpr(a, *args, **kwargs)
        
    elif isinstance(a, sym_Expr):
        return DiscreteExpr(a, *args, **kwargs)

    else:
        raise NotImplementedError('given {}'.format(type(a)))
