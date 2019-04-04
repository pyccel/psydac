# coding: utf-8

# TODO for the moment we assume Product of same space

import numpy as np
from itertools import product

from psydac.api.basic         import BasicCodeGen
from psydac.api.settings      import PSYDAC_BACKEND_PYTHON, PSYDAC_DEFAULT_FOLDER
from psydac.api.grid          import CollocationBasisValues

from psydac.api.ast.expr      import ExprKernel, ExprInterface

from psydac.cad.geometry      import Geometry
from psydac.mapping.discrete  import SplineMapping, NurbsMapping

from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace
from sympy import Expr

#==============================================================================
class DiscreteExpr(BasicCodeGen):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, Expr):
            raise TypeError('> Expecting a symbolic expression')

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

        # ...
        self._space = args[1]
        # ...

        # ...
        kwargs['mapping'] = self.space.symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping

        BasicCodeGen.__init__(self, kernel_expr, **kwargs)
        # ...

#        print('====================')
#        print(self.dependencies_code)
#        print('====================')
#        print(self.interface_code)
#        print('====================')
#        import sys; sys.exit(0)

    @property
    def mapping(self):
        return self._mapping

    @property
    def space(self):
        return self._space

    # TODO add comm and treate parallel case
    def _create_ast(self, expr, tag, **kwargs):

        mapping             = kwargs.pop('mapping', None)
        backend             = kwargs.pop('backend', PSYDAC_BACKEND_PYTHON)
        is_rational_mapping = kwargs.pop('is_rational_mapping', None)

        # ...

        kernel = ExprKernel( expr, self.space,
                            name = 'kernel_{}'.format(tag),
                            mapping = mapping,
                            is_rational_mapping = is_rational_mapping,
                            backend = backend )
                            
        from psydac.api.printing.pycode import pycode
        print(pycode(kernel.func))
        
        interface = ExprInterface( kernel,
                                  name = 'interface_{}'.format(tag),
                                  mapping = mapping,
                                  is_rational_mapping = is_rational_mapping,
                                  backend = backend )
        # ...
        print(pycode(interface.func))

        ast = {'kernel': kernel, 'interface': interface}
        return ast


    def _check_arguments(self, **kwargs):

        # TODO do we need a method from Interface to map the dictionary of arguments
        # that are passed for the call (in the same spirit of build_arguments)
        # the idea is to be sure of their order, since they can be passed to
        # Fortran

        _kwargs = {}

        # ... mandatory arguments
        sym_args = self.interface.in_arguments
        keys = [str(a) for a in sym_args]
        for key in keys:
            try:
                # we use x1 for the call rather than arr_x1, to keep x1 inside
                # the loop
                if key == 'x1':
                    _kwargs['arr_x1'] = kwargs[key]

                elif key == 'x2':
                    _kwargs['arr_x2'] = kwargs[key]

                elif key == 'x3':
                    _kwargs['arr_x3'] = kwargs[key]

                else:
                    _kwargs[key] = kwargs[key]
            except:
                raise KeyError('Unconsistent argument with interface')
        # ...

        # ... optional (inout) arguments
        sym_args = self.interface.inout_arguments
        keys = [str(a) for a in sym_args]
        for key in keys:
            try:
                _kwargs[key] = kwargs[key]
            except:
                pass
        # ...

        return _kwargs

    def __call__(self, *args, **kwargs):

        kwargs = self._check_arguments(**kwargs)

        Vh = self.space
        is_block = False
        if isinstance(Vh, ProductFemSpace):
            Vh = Vh.spaces[0]
            is_block = True

        if not isinstance(Vh, TensorFemSpace):
            raise NotImplementedError('Only TensorFemSpace is available for the moment')

        args = (Vh,) + args 

        dim = Vh.ldim

        # ...
        if self.kernel.fields:
            nderiv = self.interface.max_nderiv
            xis = [kwargs['arr_x{}'.format(i)] for i in range(1,dim+1)]
            grid = tuple(xis)
            # TODO assert that xis are inside the space domain
            basis_values = CollocationBasisValues(grid, Vh, nderiv=nderiv)
            args = args + (basis_values,)

        values = self.func(*args, **kwargs)

        return values


