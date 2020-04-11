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
from psydac.api.grid            import QuadratureGrid, BoundaryQuadratureGrid
from psydac.api.grid            import BasisValues
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

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...
        trial_space = self.spaces[0]
        test_space  = self.spaces[1]

        # ...
        quad_order   = kwargs.pop('quad_order', None)
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
            self._spaces = (trial_space, test_space)
        self._test_symbolic_space  = test_sym_space
        self._trial_symbolic_space = trial_sym_space

        # TODO must check that spaces lead to the same QuadratureGrid

        if isinstance(domain, sym_Boundary):
            self._grid = BoundaryQuadratureGrid( test_space,
                                                 domain.axis,
                                                 domain.ext,
                                                 quad_order = quad_order )
            test_ext  = domain.ext
            trial_ext = domain.ext
        elif isinstance(domain, sym_Interface):
            test_ext  = -1 if isinstance(self.kernel_expr.test,  PlusInterfaceOperator) else 1
            trial_ext = -1 if isinstance(self.kernel_expr.trial, PlusInterfaceOperator) else 1
            self._grid = BoundaryQuadratureGrid( test_space,
                         domain.axis,
                         test_ext,
                         quad_order = quad_order )
        else:
            test_ext  = None
            trial_ext = None
            self._grid = QuadratureGrid( test_space, quad_order = quad_order )
        # ...
        self._test_basis = BasisValues( test_space, self.grid,
                                        nderiv = self.max_nderiv , ext=test_ext)
        self._trial_basis = BasisValues( trial_space, self.grid,
                                         nderiv = self.max_nderiv , ext=trial_ext)

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
        target      = self.kernel_expr.target
        tests_degrees = self.spaces[1].degree
        trials_degrees = self.spaces[0].degree
        spans = self.test_basis.spans
        tests_basis, tests_degrees, spans = collect_spaces(self._test_symbolic_space, tests_basis, tests_degrees, spans)
        trial_basis, trials_degrees       = collect_spaces(self._trial_symbolic_space, trial_basis, trials_degrees)
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
        spaces          = self.spaces
        expr            = self.kernel_expr.expr
        target          = self.kernel_expr.target
        global_mats     = OrderedDict()
        local_mats      = OrderedDict()
        test_space      = spaces[1].vector_space
        trial_space     = spaces[0].vector_space
        test_degree     = np.array(spaces[1].degree)
        trial_degree    = np.array(spaces[0].degree)
        test_sym_space  = self._test_symbolic_space
        trial_sym_space = self._trial_symbolic_space
        is_broken       = test_sym_space.is_broken
        domain          = test_sym_space.domain.interior.args if is_broken else test_sym_space.domain.interior
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
                        local_mats[ii,jj]  = np.zeros((*(test_degree[i]+1),*(2*pads[i,j]+1)))
                    else:
                        if self._matrix and self._matrix[i,j]:
                            global_mats[i,j] = self._matrix[i,j]
                        else:
                            global_mats[i,j] = StencilMatrix(trial_space.spaces[j], test_space.spaces[i], pads = tuple(pads[i,j]))
                        local_mats[i,j]  = np.zeros((*(test_degree[i]+1),*(2*pads[i,j]+1)))

            self._matrix = BlockMatrix(trial_space, test_space, global_mats)

        elif is_broken:
            if isinstance(target, sym_Interface):
                axis = target.axis
                test_spans  = self.test_basis.spans
                trial_spans = self.trial_basis.spans
                ij = [domain.index(target.minus.domain),domain.index(target.plus.domain)]
                if isinstance(self.kernel_expr.test, PlusInterfaceOperator):
                    ij.reverse()
                ii, jj = ij
                if self._matrix[ii,jj]:
                    global_mats[ii,jj] = self._matrix[ii,jj]
                else:
                    global_mats[ii,jj] = StencilInterfaceMatrix(trial_space, test_space, trial_spans[0][axis][0], test_spans[0][axis][0], axis)
                local_mats[ii,jj]  = np.zeros((*(test_degree+1),*(2*trial_degree+1)))
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

                local_mats[i,j]  = np.zeros((*(test_degree+1),*(2*trial_degree+1)))
            for ij in global_mats:
                self._matrix[ij]  = global_mats[ij]
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

        self._symbolic_space  = test_sym_space

        if not isinstance(domain, sym_Boundary):
            self._grid = QuadratureGrid( self.space, quad_order = quad_order )
            ext        = None

        else:
            self._grid = BoundaryQuadratureGrid( self.space,
                                                 domain.axis,
                                                 domain.ext,
                                                 quad_order = quad_order )
            ext = domain.ext

        self._test_basis = BasisValues( self.space, self.grid,
                                        nderiv = self.max_nderiv, ext=ext)


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

        tests_basis, tests_degrees, spans = collect_spaces(self._symbolic_space, tests_basis, tests_degrees, spans)

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
        target      = self.kernel_expr.target
        sym_space   = self._symbolic_space
        is_broken   = sym_space.is_broken
        domain      = sym_space.domain.interior.args if is_broken else sym_space.domain.interior
        if isinstance(expr, Matrix):
            expr = expr[:]
            for i in range(len(expr)):
                if expr[i].is_zero:
                    continue
                else:
                    if  self._vector[i]:
                        global_mats[i] = self._vector[i]
                    else:
                        global_mats[i] = StencilVector(test_space.spaces[i])
                    local_mats[i] = np.zeros([*(test_degree[i]+1)])

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

            local_mats[i] = np.zeros([*(test_degree+1)])
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
            self._grid = BoundaryQuadratureGrid( self.space,
                                         boundary.axis,
                                         boundary.ext,
                                         quad_order = quad_order )
            ext        = domain.ext
        else:
            self._grid = QuadratureGrid( self.space, quad_order = quad_order )
            ext        = None


        # ...
        self._test_basis = BasisValues( self.space, self.grid,
                                        nderiv = self.max_nderiv, ext=ext)
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

        tests_degrees = self.space.degree
        tests_basis, tests_degrees, spans = collect_spaces(self._symbolic_space, tests_basis, tests_degrees, spans)
        tests_basis   = flatten(tests_basis)
        tests_degrees = flatten(tests_degrees)
        spans         = flatten(spans)
        quads         = flatten(list(zip(points, weights)))
        quads_degree  = flatten(self.grid.quad_order)
        global_pads   = self.space.vector_space.pads
        local_mats, vector = np.zeros((1,)), np.zeros((1,))

        if self._vector is None:
            self._vector = vector

        if self.mapping:
            mapping = [e._coeffs._data for e in self.mapping._fields]
        else:
            mapping = []

        args = (*tests_basis, *spans, *quads, *tests_degrees, *n_elements, *quads_degree, *global_pads, local_mats, self._vector, *mapping)

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

        self._forms = forms
        # ...

    @property
    def forms(self):
        return self._forms

    def assemble(self, **kwargs):
        for form in self.forms:
            M = form.assemble(**kwargs)
        return M
