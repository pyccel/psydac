# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call


from sympde.expr     import BasicForm as sym_BasicForm
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm
from sympde.expr     import Functional as sym_Functional
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Boundary as sym_Boundary, Interface as sym_Interface
from sympde.expr     import Norm as sym_Norm
from sympde.topology import Domain, Boundary
from sympde.topology import Line, Square, Cube
from sympde.topology import BasicFunctionSpace
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import Mapping
from sympde.topology import H1SpaceType, L2SpaceType, UndefinedSpaceType

from sympde.calculus.core  import PlusInterfaceOperator, MinusInterfaceOperator

from psydac.api.basic           import BasicDiscrete
from psydac.api.basic           import random_string
from psydac.api.grid            import QuadratureGrid, BasisValues
from psydac.api.ast.glt         import GltKernel
from psydac.api.ast.glt         import GltInterface
from psydac.api.glt             import DiscreteGltExpr
from psydac.api.utilities       import flatten

from psydac.linalg.stencil      import StencilVector, StencilMatrix, StencilInterfaceMatrix

from psydac.linalg.block        import BlockVector, BlockMatrix
from psydac.cad.geometry        import Geometry
from psydac.mapping.discrete    import SplineMapping, NurbsMapping
from psydac.fem.vector          import ProductFemSpace
from psydac.fem.basic           import FemField
from psydac.fem.vector          import VectorFemField

from collections import OrderedDict
from sympy import ImmutableDenseMatrix, Matrix
import inspect
import sys
import numpy as np

def get_quad_order(Vh):
    if isinstance(Vh, ProductFemSpace):
        Vh = Vh.spaces[0]
    return tuple([g.weights.shape[1] for g in Vh.quad_grids])

def collect_spaces(space, *args):
    """
    This function collect the arguments used in the assembly function

    Parameters
    ----------
    space: <FunctionSpace>
        the symbolic space

    args : <list>
        list of discrete space components like basis values, spans, ...

    Returns
    -------
    args : <list>
        list of discrete space components elements used in the asembly

    """

    if isinstance(space, ProductSpace):
        spaces = space.spaces
        indices = []
        i = 0
        for space in spaces:
            if isinstance(space, VectorFunctionSpace):
                if isinstance(space.kind, (H1SpaceType, L2SpaceType, UndefinedSpaceType)):
                    indices.append(i)
                else:
                    indices += [i+j for j in range(space.ldim)]
                i = i + space.ldim
            else:
                indices.append(i)
                i = i + 1
        args = [[e[i] for i in indices] for e in args]

    elif isinstance(space, VectorFunctionSpace):
        if isinstance(space.kind, (H1SpaceType, L2SpaceType, UndefinedSpaceType)):
            args = [[e[0]] for e in args]

    return args

def construct_test_space_arguments(basis_values):
    space          = basis_values.space
    test_basis     = basis_values.basis
    spans          = basis_values.spans
    test_degrees   = space.degree

    test_basis, test_degrees, spans = collect_spaces(space.symbolic_space, test_basis, test_degrees, spans)

    test_basis    = flatten(test_basis)
    test_degrees  = flatten(test_degrees)
    spans         = flatten(spans)
    return test_basis, test_degrees, spans

def construct_trial_space_arguments(basis_values):
    space          = basis_values.space
    trial_basis    = basis_values.basis
    trial_degrees  = space.degree

    trial_basis, trial_degrees = collect_spaces(space.symbolic_space, trial_basis, trial_degrees)

    trial_basis    = flatten(trial_basis)
    trial_degrees = flatten(trial_degrees)
    return trial_basis, trial_degrees

def construct_quad_grids_arguments(grid):
    points         = grid.points
    weights        = grid.weights
    quads          = flatten(list(zip(points, weights)))

    quads_degree   = flatten(grid.quad_order)
    n_elements     = grid.n_elements
    return n_elements, quads, quads_degree

def reset_arrays(*args):
    for a in args: a[:] = 0.

def do_nothing(*args):
    pass

#==============================================================================
class DiscreteBilinearForm(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_BilinearForm):
            raise TypeError('> Expecting a symbolic BilinearForm')

        if not args:
            raise ValueError('> fem spaces must be given as a list/tuple')

        assert( len(args) == 2 )

        # ...
        domain_h = args[0]
        assert( isinstance(domain_h, Geometry) )

        mapping = list(domain_h.mappings.values())[0]
        self._mapping = mapping

        is_rational_mapping = False
        if not( mapping is None ):
            is_rational_mapping = isinstance( mapping, NurbsMapping )

        self._is_rational_mapping = is_rational_mapping
        # ...
        self._spaces = args[1]
        trial_space  = self.spaces[0]
        test_space   = self.spaces[1]

        kwargs['discrete_space']      = self.spaces
        kwargs['mapping']             = self.spaces[0].symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm
        quad_order                    = kwargs.pop('quad_order', get_quad_order(test_space))

        BasicDiscrete.__init__(self, expr, kernel_expr, quad_order=quad_order, **kwargs)

        domain       = self.kernel_expr.target
        self._matrix = kwargs.pop('matrix', None)

        if self._matrix is None:
            if isinstance(test_space, ProductFemSpace):
                self._matrix = BlockMatrix(trial_space.vector_space, test_space.vector_space)
            elif isinstance(trial_space, ProductFemSpace):
                self._matrix = BlockMatrix(trial_space.vector_space, test_space.vector_space)

        test_sym_space   = test_space.symbolic_space
        trial_sym_space  = trial_space.symbolic_space
        if test_sym_space.is_broken:
            domains = test_sym_space.domain.interior.args
            if isinstance(domain, sym_Interface):
                ij = [domains.index(domain.minus.domain), domains.index(domain.plus.domain)]
                if isinstance(self.kernel_expr.test, PlusInterfaceOperator):
                    ij.reverse()
                i,j = ij
                test_space  = test_space.spaces[i]
                trial_space = trial_space.spaces[j]
            else:
                if isinstance(domain, sym_Boundary):
                    i = domains.index(domain.domain)
                else:
                    i = domains.index(domain)
                test_space  = test_space.spaces[i]
                trial_space = trial_space.spaces[i]
            self._spaces    = (trial_space, test_space)
            self._spaces[0].symbolic_space = trial_sym_space
            self._spaces[1].symbolic_space = test_sym_space

        if isinstance(domain, sym_Boundary):
            axis      = domain.axis
            test_ext  = domain.ext
            trial_ext = domain.ext
        elif isinstance(domain, sym_Interface):
            axis       = domain.axis
            test_ext   = -1 if isinstance(self.kernel_expr.test,  PlusInterfaceOperator) else 1
            trial_ext  = -1 if isinstance(self.kernel_expr.trial, PlusInterfaceOperator) else 1
        else:
            axis      = None
            test_ext  = None
            trial_ext = None

        if isinstance(domain, (sym_Boundary, sym_Interface)):
            if test_ext == -1:
                start = test_space.vector_space.starts[axis]
                if start != 0 :
                    self._func = do_nothing
            elif test_ext == 1:
                end = test_space.vector_space.ends[axis]
                nb  = test_space.spaces[axis].nbasis
                if end+1 != nb:
                    self._func = do_nothing

        grid              = QuadratureGrid( test_space, axis, test_ext )
        self._grid        = grid
        self._test_basis  = BasisValues( test_space,  nderiv = self.max_nderiv , trial=False, grid=grid, ext=test_ext)
        self._trial_basis = BasisValues( trial_space, nderiv = self.max_nderiv , trial=True, grid=grid, ext=trial_ext)

        self._args                 = self.construct_arguments()

    @property
    def spaces(self):
        return self._spaces

    @property
    def grid(self):
        return self._grid

    @property
    def test_basis(self):
        return self._test_basis

    @property
    def trial_basis(self):
        return self._trial_basis

    @property
    def global_matrices(self):
        return self._global_matrices

    @property
    def args(self):
        return self._args

    def assemble(self, *, reset=True, **kwargs):
        if self._free_args:
            args = self._args
            basis   = ()
            spans   = ()
            degrees = ()
            coeffs  = ()
            consts  = ()
            for key in self._free_args:
                v = kwargs[key]
                if isinstance(v, FemField):
                    space   = v.space
                    basis_v = BasisValues( v.space, nderiv = self.max_nderiv)
                    bs, d,s = construct_test_space_arguments(basis_v)
                    basis   += tuple(bs)
                    spans   += tuple(s)
                    degrees += tuple(d)
                    coeffs  += (v._coeffs._data,)
                elif isinstance(v, VectorFemField):
                    space     = v.space
                    basis_v   = BasisValues( v.space, nderiv = self.max_nderiv)
                    bs, d,s   = construct_test_space_arguments(basis_v)
                    basis   += tuple(bs)
                    spans   += tuple(s)
                    degrees += tuple(d)
                    coeffs  += tuple(e._data for e in v.coeffs[:])
                else:
                    consts  += (v,)
            args = self.args + basis + spans + degrees + coeffs + consts

        else:
            args = self._args

        if reset:
            reset_arrays(*self.global_matrices)

        self._func(*args)
        return self._matrix

    def construct_arguments(self):

        test_basis, test_degrees, spans = construct_test_space_arguments(self.test_basis)
        trial_basis, trial_degrees        = construct_trial_space_arguments(self.trial_basis)
        n_elements, quads, quad_degrees   = construct_quad_grids_arguments(self.grid)

        pads                    = self.spaces[0].vector_space.pads
        element_mats, global_mats = self.allocate_matrices()
        self._global_matrices   = [M._data for M in global_mats]

        if self.mapping:
            mapping = [e._coeffs._data for e in self.mapping._fields]
            if self.is_rational_mapping:
                mapping = [*mapping, self.mapping._weights_field._coeffs._data]
        else:
            mapping = []
        args = (*test_basis, *trial_basis, *spans, *quads, *test_degrees, *trial_degrees, *n_elements, *quad_degrees, *pads, *element_mats, *self._global_matrices, *mapping)
        return args

    def allocate_matrices(self):

        spaces          = self.spaces
        expr            = self.kernel_expr.expr
        target          = self.kernel_expr.target
        global_mats     = OrderedDict()
        element_mats      = OrderedDict()
        test_space      = spaces[1].vector_space
        trial_space     = spaces[0].vector_space
        test_degree     = np.array(spaces[1].degree)
        trial_degree    = np.array(spaces[0].degree)
        test_sym_space  = spaces[1].symbolic_space
        trial_sym_space = spaces[0].symbolic_space
        is_broken       = test_sym_space.is_broken
        domain          = test_sym_space.domain.interior.args if is_broken else test_sym_space.domain.interior

        if isinstance(expr, (ImmutableDenseMatrix, Matrix)):
            pads         = np.empty((len(test_degree),len(trial_degree),len(test_degree[0])), dtype=int)
            for i in range(len(test_degree)):
                for j in range(len(trial_degree)):
                    td  = test_degree[i]
                    trd = trial_degree[j]
                    pads[i,j][:] = np.array([td, trd]).max(axis=0)
        else:
            pads = test_degree

        if isinstance(expr, (ImmutableDenseMatrix, Matrix)):

            shape = expr.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if expr[i,j].is_zero:
                        continue
                    elif is_broken:
                        ii = shape[0]*domain.index(target) + i
                        jj = shape[1]*domain.index(target) + j
                        if self._matrix and self._matrix[ii,jj]:
                            global_mats[ii,jj] = self._matrix[ii,jj]
                        else:
                            global_mats[ii,jj] = StencilMatrix(trial_space.spaces[j], test_space.spaces[i], pads = tuple(pads[i,j]))
                        element_mats[ii,jj]  = np.empty((*(test_degree[i]+1),*(2*pads[i,j]+1)))

                    else:
                        if self._matrix and self._matrix[i,j]:
                            global_mats[i,j] = self._matrix[i,j]
                        else:
                            global_mats[i,j] = StencilMatrix(trial_space.spaces[j], test_space.spaces[i], pads = tuple(pads[i,j]))
                        element_mats[i,j]  = np.empty((*(test_degree[i]+1),*(2*pads[i,j]+1)))

            self._matrix = BlockMatrix(trial_space, test_space, global_mats)

        elif is_broken:
            if isinstance(target, sym_Interface):
                axis        = target.axis
                test_spans  = self.test_basis.spans
                trial_spans = self.trial_basis.spans
                ij = [domain.index(target.minus.domain),domain.index(target.plus.domain)]
                if isinstance(self.kernel_expr.test, PlusInterfaceOperator):
                    ij.reverse()
                ii, jj = ij
                if self._matrix[ii,jj]:
                    global_mats[ii,jj] = self._matrix[ii,jj]
                elif self._func != do_nothing:
                    global_mats[ii,jj] = StencilInterfaceMatrix(trial_space, test_space, trial_spans[0][axis][0], test_spans[0][axis][0], axis)
                element_mats[ii,jj]  = np.empty((*(test_degree+1),*(2*trial_degree+1)))
            else:
                if isinstance(target, Boundary):
                    i = domain.index(target.domain)
                else:
                    i = domain.index(target)
                j = i
                if self._matrix[i,j]:
                    global_mats[i,j] = self._matrix[i,j]
                else:
                    global_mats[i,j] = StencilMatrix(trial_space, test_space)

                element_mats[i,j]  = np.empty((*(test_degree+1),*(2*trial_degree+1)))
            for ij in global_mats:
                self._matrix[ij]  = global_mats[ij]
        else:
            if self._matrix:
                global_mats[0,0] = self._matrix
            else:
                global_mats[0,0] = StencilMatrix(trial_space, test_space, pads=tuple(pads))

            element_mats[0,0]  = np.empty((*(test_degree+1),*(2*pads+1)))
            self._matrix     = global_mats[0,0]
        return element_mats.values(), global_mats.values()


#==============================================================================
class DiscreteLinearForm(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_LinearForm):
            raise TypeError('> Expecting a symbolic LinearForm')

        assert( len(args) == 2 )

        domain_h = args[0]
        assert( isinstance(domain_h, Geometry) )

        mapping = list(domain_h.mappings.values())[0]
        self._mapping = mapping

        is_rational_mapping = False
        if not( mapping is None ):
            is_rational_mapping = isinstance( mapping, NurbsMapping )

        self._is_rational_mapping = is_rational_mapping

        self._space = args[1]

        kwargs['discrete_space']      = self.space
        kwargs['mapping']             = self.space.symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm
        quad_order                    = kwargs.pop('quad_order', get_quad_order(self.space))

        BasicDiscrete.__init__(self, expr, kernel_expr, quad_order=quad_order, **kwargs)

        # ...
        domain     = self.kernel_expr.target

        self._vector = kwargs.pop('vector', None)
        if self._vector is None:
            if isinstance(self._space, ProductFemSpace):
                self._vector = BlockVector(self._space.vector_space)

        test_sym_space   = self._space.symbolic_space
        if test_sym_space.is_broken:
            domains = test_sym_space.domain.interior.args

            if isinstance(domain, sym_Boundary):
                i = domains.index(domain.domain)
            else:
                i = domains.index(domain)
            self._space  = self._space.spaces[i]

        self._space.symbolic_space  = test_sym_space

        if not isinstance(domain, sym_Boundary):
            ext  = None
            axis = None
        else:
            ext  = domain.ext
            axis = domain.axis
            if ext == -1:
                start = self.space.vector_space.starts[domain.axis]
                if start != 0 :
                    self._func = do_nothing
            elif ext == 1:
                end = self.space.vector_space.ends[domain.axis]
                nb  = self.space.spaces[domain.axis].nbasis
                if end+1 != nb:
                    self._func = do_nothing

        grid             = QuadratureGrid( self.space, axis=axis, ext=ext )
        self._grid       = grid
        self._test_basis = BasisValues( self.space, nderiv = self.max_nderiv, grid=grid, ext=ext)

        self._args = self.construct_arguments()

    @property
    def space(self):
        return self._space

    @property
    def grid(self):
        return self._grid

    @property
    def test_basis(self):
        return self._test_basis

    @property
    def global_matrices(self):
        return self._global_matrices

    @property
    def args(self):
        return self._args

    def assemble(self, *, reset=True, **kwargs):
        if self._free_args:
            args = self._args
            basis   = ()
            spans   = ()
            degrees = ()
            coeffs  = ()
            consts  = ()
            for key in self._free_args:
                v = kwargs[key]
                if isinstance(v, FemField):
                    space   = v.space
                    basis_v = BasisValues( v.space, nderiv = self.max_nderiv)
                    bs, d,s = construct_test_space_arguments(basis_v)
                    basis   += tuple(bs)
                    spans   += tuple(s)
                    degrees += tuple(d)
                    coeffs  += (v._coeffs._data,)
                elif isinstance(v, VectorFemField):
                    space   = v.space
                    basis_v   = BasisValues( v.space, nderiv = self.max_nderiv)
                    bs, d,s = construct_test_space_arguments(basis_v)
                    basis   += tuple(bs)
                    spans   += tuple(s)
                    degrees += tuple(d)
                    coeffs  += tuple(e._data for e in v.coeffs[:])
                else:
                    consts  += (v,)
            args = self.args + basis + spans + degrees + coeffs + consts

        else:
            args = self._args

        if reset:
            reset_arrays(*self.global_matrices)

        self._func(*args)
        return self._vector

    def construct_arguments(self):

        tests_basis, tests_degrees, spans = construct_test_space_arguments(self.test_basis)
        n_elements, quads, quads_degree   = construct_quad_grids_arguments(self.grid)

        global_pads   = self.space.vector_space.pads

        element_mats, global_mats = self.allocate_matrices()
        self._global_matrices   = [M._data for M in global_mats]

        if self.mapping:
            mapping   = [e._coeffs._data for e in self.mapping._fields]
            if self.is_rational_mapping:
                mapping = [*mapping, self.mapping._weights_field._coeffs._data]
        else:
            mapping   = []
        args = (*tests_basis, *spans, *quads, *tests_degrees, *n_elements, *quads_degree, *global_pads, *element_mats, *self._global_matrices, *mapping)
        return args

    def allocate_matrices(self):
        space       = self.space
        expr        = self.kernel_expr.expr
        global_mats = OrderedDict()
        element_mats  = OrderedDict()
        test_space  = space.vector_space
        test_degree = np.array(space.degree)

        target      = self.kernel_expr.target
        sym_space   = space.symbolic_space
        is_broken   = sym_space.is_broken
        domain      = sym_space.domain.interior.args if is_broken else sym_space.domain.interior
        if isinstance(expr, (ImmutableDenseMatrix, Matrix)):
            expr = expr[:]
            for i in range(len(expr)):
                if expr[i].is_zero:
                    continue
                else:
                    if  self._vector[i]:
                        global_mats[i] = self._vector[i]
                    else:
                        global_mats[i] = StencilVector(test_space.spaces[i])
                    element_mats[i] = np.empty([*(test_degree[i]+1)])

            for i in global_mats:
                self._vector[i] = global_mats[i]
        elif is_broken:
            if isinstance(target, Boundary):
                i = domain.index(target.domain)
            else:
                i = domain.index(target)

            if self._vector[i]:
                global_mats[i] = self._vector[i]
            else:
                global_mats[i] = StencilVector(test_space)

            element_mats[i] = np.empty([*(test_degree+1)])

            for i in global_mats:
                self._vector[i] = global_mats[i]
        else:
            if self._vector:
                global_mats[0] = self._vector
            else:
                global_mats[0] = StencilVector(test_space)
                self._vector   = global_mats[0]
            element_mats[0]  = np.empty([*(test_degree+1)])
        self._global_mats = list(global_mats.values())
        return element_mats.values(), global_mats.values()


#==============================================================================
class DiscreteFunctional(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_Functional):
            raise TypeError('> Expecting a symbolic Functional')

        assert( len(args) == 2 )

        # ...
        domain_h = args[0]
        assert( isinstance(domain_h, Geometry) )

        mapping = list(domain_h.mappings.values())[0]
        self._mapping = mapping

        is_rational_mapping = False
        if not( mapping is None ):
            is_rational_mapping = isinstance( mapping, NurbsMapping )

        self._is_rational_mapping = is_rational_mapping

        self._space = args[1]

        kwargs['discrete_space']      = self.space
        kwargs['mapping']             = self.space.symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm
        quad_order                    = kwargs.pop('quad_order', get_quad_order(self.space))

        BasicDiscrete.__init__(self, expr, kernel_expr, quad_order=quad_order, **kwargs)

        # ...
        self._vector = kwargs.pop('vector', None)
        domain     = self.kernel_expr.target
        # ...


        test_sym_space   = self._space.symbolic_space
        if test_sym_space.is_broken:
            domains = test_sym_space.domain.interior.args

            if isinstance(domain, sym_Boundary):
                i = domains.index(domain.domain)
            else:
                i = domains.index(domain)
            self._space  = self._space.spaces[i]

        self._symbolic_space  = test_sym_space
        self._domain          = domain

        if isinstance(domain, sym_Boundary):
            ext        = domain.ext
            axis       = domain.axis
        else:
            ext        = None
            axis       = None

        # ...
        grid             = QuadratureGrid( self.space,  axis=axis, ext=ext)
        self._grid       = grid
        self._test_basis = BasisValues( self.space, nderiv = self.max_nderiv, grid=grid, ext=ext)

        self._args = self.construct_arguments()

    @property
    def space(self):
        return self._space

    @property
    def grid(self):
        return self._grid

    @property
    def test_basis(self):
        return self._test_basis

    def construct_arguments(self):
        sk          = self.grid.local_element_start
        ek          = self.grid.local_element_end
        points      = [p[s:e+1] for s,e,p in zip(sk,ek,self.grid.points)]
        weights     = [w[s:e+1] for s,e,w in zip(sk,ek,self.grid.weights)]
        n_elements  = [e-s+1 for s,e in zip(sk,ek)]
        tests_basis = [[bs[s:e+1] for s,e,bs in zip(sk,ek,basis)] for basis in self.test_basis.basis]
        spans       = [[sp[s:e+1] for s,e,sp in zip(sk,ek,spans)] for spans in self.test_basis.spans]

        space         = self.space
        tests_degrees = self.space.degree

        tests_basis, tests_degrees, spans = collect_spaces(space.symbolic_space, tests_basis, tests_degrees, spans)

        tests_basis   = flatten(tests_basis)
        tests_degrees = flatten(tests_degrees)
        spans         = flatten(spans)
        quads         = flatten(list(zip(points, weights)))
        quads_degree  = flatten(self.grid.quad_order)
        global_pads   = self.space.vector_space.pads

        element_mats, vector = np.empty((1,)), np.empty((1,))

        if self._vector is None:
            self._vector = vector

        if self.mapping:
            mapping = [e._coeffs._data for e in self.mapping._fields]
            if self.is_rational_mapping:
                mapping = [*mapping, self.mapping._weights_field._coeffs._data]
        else:
            mapping = []

        args = (*tests_basis, *spans, *quads, *tests_degrees, *n_elements, *quads_degree, *global_pads, element_mats, self._vector, *mapping)

        return args

    def assemble(self, **kwargs):

        if self._free_args:
            args = self._args
            free_args = self._free_args
            for key in free_args:
                if isinstance(kwargs[key], FemField):
                    
                    args += (kwargs[key]._coeffs._data,)
                elif isinstance(kwargs[key], VectorFemField) and not self._symbolic_space.is_broken:
                    args += tuple(e._data for e in kwargs[key].coeffs[:])
                elif isinstance(kwargs[key], VectorFemField) and self._symbolic_space.is_broken:
                    index = self._symbolic_space.domain.interior.args.index(self._domain)
                    args += (kwargs[key].coeffs[index]._data, )

                else:
                    args += (kwargs[key], )
        else:
            args = self._args

        self._vector[:] = 0
        self._func(*args)

        v = self._vector[0]

        if isinstance(self.expr, sym_Norm):
            if not( self.comm is None ):
                v = self.comm.allreduce(sendobj=v)

            if self.expr.exponent == 2:
                # add abs because of 0 machine
                v = np.sqrt(np.abs(v))
            else:
                raise NotImplementedError('TODO')
        return v

#==============================================================================
class DiscreteSumForm(BasicDiscrete):

    def __init__(self, a, kernel_expr, *args, **kwargs):
        if not isinstance(a, (sym_BilinearForm, sym_LinearForm, sym_Functional)):
            raise TypeError('> Expecting a symbolic BilinearForm, LinearFormn Functional')

        self._expr = a

        backend = kwargs.get('backend', None)
        self._backend = backend

        folder = kwargs.get('folder', None)
        self._folder = self._initialize_folder(folder)

        # create a module name if not given
        tag = random_string( 8 )

        # ...
        forms = []
        free_args = []
        self._kernel_expr = kernel_expr
        for e in kernel_expr:
            kwargs['target'] = e.target
            if isinstance(a, sym_BilinearForm):
                ah = DiscreteBilinearForm(a, e, *args, **kwargs)
                kwargs['matrix'] = ah._matrix

            elif isinstance(a, sym_LinearForm):
                ah = DiscreteLinearForm(a, e, *args, **kwargs)
                kwargs['vector'] = ah._vector
            elif isinstance(a, sym_Functional):
                ah = DiscreteFunctional(a, e, *args, **kwargs)
                kwargs['vector'] = ah._vector
            forms.append(ah)
            free_args.append(ah.free_args)
            kwargs['boundary'] = None

        self._forms     = forms
        self._free_args = tuple(set(free_args))
        # ...

    @property
    def forms(self):
        return self._forms

    @property
    def free_args(self):
        return self._free_args

    def assemble(self, **kwargs):
        M = self.forms[0].assemble(**kwargs)
        for form in self.forms[1:]:
            M = form.assemble(reset=False, **kwargs)
        return M
