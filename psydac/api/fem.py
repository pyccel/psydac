# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call


from sympde.expr     import BasicForm as sym_BasicForm
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm
from sympde.expr     import Functional as sym_Functional
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Boundary as sym_Boundary
from sympde.expr     import Norm as sym_Norm
from sympde.topology import Domain, Boundary
from sympde.topology import Line, Square, Cube
from sympde.topology import BasicFunctionSpace
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import Mapping
from sympde.topology import H1SpaceType, L2SpaceType, UndefinedSpaceType

from psydac.api.basic           import BasicDiscrete
from psydac.api.basic           import random_string
from psydac.api.grid            import QuadratureGrid, BoundaryQuadratureGrid
from psydac.api.grid            import BasisValues
from psydac.api.ast.glt         import GltKernel
from psydac.api.ast.glt         import GltInterface
from psydac.api.glt             import DiscreteGltExpr
from psydac.api.utilities       import flatten

from psydac.linalg.stencil      import StencilVector, StencilMatrix
from psydac.linalg.block        import BlockVector, BlockMatrix
from psydac.cad.geometry        import Geometry
from psydac.mapping.discrete    import SplineMapping, NurbsMapping
from psydac.fem.vector          import ProductFemSpace
from psydac.fem.basic           import FemField
from psydac.fem.vector          import VectorFemField

from collections import OrderedDict
from sympy import Matrix
import inspect
import sys
import numpy as np

def collect_spaces(space, *ls):

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
        new_ls = []
        for e in ls:
            new_ls   += [[e[i] for i in indices]]
        ls = new_ls
    elif isinstance(space, VectorFunctionSpace):
        if isinstance(space.kind, (H1SpaceType, L2SpaceType, UndefinedSpaceType)):
            return [[e[0]] for e in ls]

    return ls

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
        # ...
        kwargs['discrete_space']      = self.spaces
        kwargs['mapping']             = self.spaces[0].symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm

        boundary = kwargs.pop('boundary', [])
        if boundary and isinstance(boundary, list):
            kwargs['boundary'] = boundary[0]
        elif boundary:
            kwargs['boundary'] = boundary

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...
        trial_space = self.spaces[0]
        test_space  = self.spaces[1]
        # ...

        # ...
        quad_order = kwargs.pop('quad_order', None)
        domain   = self.ast.domain
        # ...

        # ...
        # TODO must check that spaces lead to the same QuadratureGrid
        if not isinstance(domain, sym_Boundary):
            self._grid = QuadratureGrid( test_space, quad_order = quad_order )

        else:   
            self._grid = BoundaryQuadratureGrid( test_space,
                                                 domain.axis,
                                                 domain.ext,
                                                 quad_order = quad_order )
        # ...
        self._test_basis = BasisValues( test_space, self.grid,
                                        nderiv = self.max_nderiv )
        self._trial_basis = BasisValues( trial_space, self.grid,
                                         nderiv = self.max_nderiv )

        self._matrix = kwargs.pop('matrix', None)
        self._args  = self.construct_arguments()

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
    def args(self):
        return self._args

    def assemble(self, **kwargs):
        if self._free_args:
            args = self._args
            for key in self._free_args:
                if isinstance(kwargs[key], FemField):
                    args += (kwargs[key]._coeffs._data,)
                elif isinstance(kwargs[key], VectorFemField):
                    args += tuple(e._data for e in kwargs[key].coeffs[:])
                else:
                    args += (kwargs[key], )
        else:
            args = self._args
        self._func(*args)
        return self._matrix

    def construct_arguments(self):

        tests_basis = self.test_basis.basis
        trial_basis = self.trial_basis.basis
        tests_degrees = self.spaces[1].degree
        trials_degrees = self.spaces[0].degree
        spans = self.test_basis.spans
        tests_basis, tests_degrees, spans = collect_spaces(self.spaces[1].symbolic_space, tests_basis, tests_degrees, spans)
        trial_basis, trials_degrees       = collect_spaces(self.spaces[0].symbolic_space, trial_basis, trials_degrees)
        tests_basis = flatten(tests_basis)
        trial_basis = flatten(trial_basis)
        tests_degrees = flatten(tests_degrees)
        trials_degrees = flatten(trials_degrees)
        spans = flatten(spans)
        points = self.grid.points
        weights = self.grid.weights
        quads   = flatten(list(zip(points, weights)))

        quads_degree = flatten(self.grid.quad_order)
        n_elements   = self.grid.n_elements
        global_pads = self.spaces[0].vector_space.pads
        local_mats, global_mats = self.allocate_matrices()
        global_mats = [M._data for M in global_mats]
        if self.mapping:
            mapping = [e._coeffs._data for e in self.mapping._fields]
        else:
            mapping = []
        args = (*tests_basis, *trial_basis, *spans, *quads, *tests_degrees, *trials_degrees, *n_elements, *quads_degree, *global_pads, *local_mats, *global_mats, *mapping)
        return args

    def allocate_matrices(self):
        spaces       = self.spaces
        expr         = self.kernel_expr.expr
        global_mats  = OrderedDict()
        local_mats   = OrderedDict()
        test_space   = spaces[1].vector_space
        trial_space  = spaces[0].vector_space
        test_degree  = np.array(spaces[1].degree)
        trial_degree = np.array(spaces[0].degree)
        if isinstance(expr, Matrix):
            pads         = np.zeros((len(test_degree),len(trial_degree),len(test_degree[0])), dtype=int)
            for i in range(len(test_degree)):
                for j in range(len(trial_degree)):
                    td  = test_degree[i]
                    trd = trial_degree[j]
                    pads[i,j][:] = np.array([td, trd]).max(axis=0)
        else:
            pads = test_degree

        if isinstance(expr, Matrix):
            for i in range(expr.shape[0]):
                for j in range(expr.shape[1]):
                    if expr[i,j].is_zero:
                        continue
                    else:
                        if self._matrix and self._matrix[i,j]:
                            global_mats[i,j] = self._matrix[i,j]
                        else:
                            global_mats[i,j] = StencilMatrix(trial_space.spaces[j], test_space.spaces[i], pads = tuple(pads[i,j]))
                        local_mats[i,j]  = np.zeros((*(test_degree[i]+1),*(2*pads[i,j]+1)))

            self._matrix = BlockMatrix(trial_space, test_space, global_mats)
        else:
            if self._matrix:
                global_mats[0,0] = self._matrix
            else:
                global_mats[0,0] = StencilMatrix(trial_space, test_space)
            local_mats[0,0]  = np.zeros((*(test_degree+1),*(2*pads+1)))
            self._matrix     = global_mats[0,0]
        return local_mats.values(), global_mats.values()


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

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...
        quad_order = kwargs.pop('quad_order', None)
        domain     = self.ast.domain
        # ...

        if not isinstance(domain, sym_Boundary):
            self._grid = QuadratureGrid( self.space, quad_order = quad_order )

        else:
            self._grid = BoundaryQuadratureGrid( self.space,
                                                 domain.axis,
                                                 domain.ext,
                                                 quad_order = quad_order )

        self._test_basis = BasisValues( self.space, self.grid,
                                        nderiv = self.max_nderiv )

        self._vector = kwargs.pop('vector', None)
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
    def args(self):
        return self._args

    def assemble(self, **kwargs):
        if self._free_args:
            args = self._args
            for key in self._free_args:
                if isinstance(kwargs[key], FemField):
                    args += (kwargs[key]._coeffs._data,)
                elif isinstance(kwargs[key], VectorFemField):
                    args += tuple(e._data for e in kwargs[key].coeffs[:])
                else:
                    args += (kwargs[key], )
        else:
            args = self._args
        self._func(*args)
        return self._vector

    def construct_arguments(self):

        tests_basis = self.test_basis.basis
        tests_degrees = self.space.degree
        spans = self.test_basis.spans

        tests_basis, tests_degrees, spans = collect_spaces(self.space.symbolic_space, tests_basis, tests_degrees, spans)

        tests_basis = flatten(tests_basis)
        tests_degrees = flatten(tests_degrees)
        spans = flatten(spans)

        points        = self.grid.points
        weights       = self.grid.weights
        quads         = flatten(list(zip(points, weights)))
        quads_degree  = flatten(self.grid.quad_order)
        n_elements    = self.grid.n_elements
        global_pads   = self.space.vector_space.pads
        local_mats, global_mats = self.allocate_matrices()
        global_mats   = [M._data for M in global_mats]
        if self.mapping:
            mapping   = [e._coeffs._data for e in self.mapping._fields]
        else:
            mapping   = []
        args = (*tests_basis, *spans, *quads, *tests_degrees, *n_elements, *quads_degree, *global_pads, *local_mats, *global_mats, *mapping)
        return args

    def allocate_matrices(self):
        space       = self.space
        expr        = self.kernel_expr.expr
        global_mats = OrderedDict()
        local_mats  = OrderedDict()
        test_space  = space.vector_space
        test_degree = np.array(space.degree)
        if isinstance(expr, Matrix):
            expr = expr[:]
            for i in range(len(expr)):
                    if expr[i].is_zero:
                        continue
                    else:
                        if self._vector and self.vector[i]:
                            global_mats[i] = self._vector[i]
                        else:
                            global_mats[i] = StencilVector(test_space.spaces[i])

                        local_mats[i] = np.zeros([*(test_degree[i]+1)])
            self._vector = BlockVector(test_space)
            for i in global_mats:
                self._vector[i] = global_mats[i]
        else:
            if self._vector:
                global_mats[0] = self._vector
            else:
                global_mats[0] = StencilVector(test_space)
                self._vector   = global_mats[0]
            local_mats[0]  = np.zeros([*(test_degree+1)])
        self._global_mats = list(global_mats.values())
        return local_mats.values(), global_mats.values()


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

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...
        quad_order = kwargs.pop('quad_order', None)
        boundary   = kwargs.pop('boundary',   None)
        # ...

        if boundary is None:
            self._grid = QuadratureGrid( self.space, quad_order = quad_order )

        else:
            self._grid = BoundaryQuadratureGrid( self.space,
                                                 boundary.axis,
                                                 boundary.ext,
                                                 quad_order = quad_order )

        # ...
        self._test_basis = BasisValues( self.space, self.grid,
                                        nderiv = self.max_nderiv )
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

        tests_basis = self.test_basis.basis
        tests_degrees = self.space.degree
        spans = self.test_basis.spans
        tests_basis, tests_degrees, spans = collect_spaces(self.space.symbolic_space, tests_basis, tests_degrees, spans)
        tests_basis   = flatten(tests_basis)
        tests_degrees = flatten(tests_degrees)
        spans         = flatten(spans)
        points        = self.grid.points
        weights       = self.grid.weights
        quads         = flatten(list(zip(points, weights)))
        quads_degree  = flatten(self.grid.quad_order)
        n_elements    = self.grid.n_elements
        global_pads   = self.space.vector_space.pads
        local_mats, global_mats = np.zeros((1,)), np.zeros((1,))
        if self.mapping:
            mapping = [e._coeffs._data for e in self.mapping._fields]
        else:
            mapping = []

        args = (*tests_basis, *spans, *quads, *tests_degrees, *n_elements, *quads_degree, *global_pads, local_mats, global_mats, *mapping)
        self._global_mats = global_mats
        return args

    def assemble(self, **kwargs):

        if self._free_args:
            args = self._args
            free_args = self._free_args
            for key in free_args:
                if isinstance(kwargs[key], FemField):
                    args += (kwargs[key]._coeffs._data,)
                elif isinstance(kwargs[key], VectorFemField):
                    args += tuple(e._data for e in kwargs[key].coeffs[:])
                else:
                    args += (kwargs[key], )
        else:
            args = self._args

        self._func(*args)
        v = self._global_mats[0]
        
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
        boundaries = kwargs.pop('boundary', [])
        for e in kernel_expr:
            kwargs['target'] = e.target
            if isinstance(e.target, sym_Boundary):
                boundary = [i for i in boundaries if i is e.target]
                if boundary: kwargs['boundary'] = boundary[0]

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

            kwargs['boundary'] = None

        self._forms = forms
        # ...

    @property
    def forms(self):
        return self._forms

    def assemble(self, **kwargs):
        for form in self.forms:
            M = form.assemble(**kwargs)
        return M
