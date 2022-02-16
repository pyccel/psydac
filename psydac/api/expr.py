# coding: utf-8

# TODO for the moment we assume Product of same space
# TODO properly treat expression with mapping

from itertools import product
from sympy import Expr
import numpy as np

from sympde.expr import TerminalExpr

from psydac.api.glt           import GltBasicCodeGen as BasicCodeGen
from psydac.api.settings      import PSYDAC_BACKEND_PYTHON, PSYDAC_DEFAULT_FOLDER
from psydac.api.grid          import CollocationBasisValues
from psydac.api.ast.expr      import ExprKernel, ExprInterface
from psydac.cad.geometry      import Geometry
from psydac.mapping.discrete  import SplineMapping, NurbsMapping
from psydac.fem.splines       import SplineSpace
from psydac.fem.tensor        import TensorFemSpace
from psydac.fem.vector        import ProductFemSpace

#==============================================================================
class DiscreteExpr(BasicCodeGen):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, Expr):
            raise TypeError('> Expecting a symbolic expression')

        if not args:
            raise ValueError('> fem spaces must be given as a list/tuple')

        assert( len(args) == 2 )

        # ...
        domain_h = args[0]
        domain   = domain_h.domain
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
        kernel_expr = TerminalExpr(expr, domain)
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
        
        interface = ExprInterface( kernel,
                                  name = 'interface_{}'.format(tag),
                                  mapping = mapping,
                                  is_rational_mapping = is_rational_mapping,
                                  backend = backend )
        # ...

        ast = {'kernel': kernel, 'interface': interface}
        return ast


    def __call__(self, *args, **kwargs):
        
        Vh = self.space
        dim = Vh.ldim
        assert len(args) == dim
        
        is_block = False
        fields = self.interface.kernel.fields + self.interface.kernel.vector_fields
        
        if fields:
            nderiv = self.interface.max_nderiv
            fields = [kwargs[F.name] for F in fields]
            grid  = args
            
            # TODO assert that xis are inside the space domain
            # TODO generalize to use multiple fields
            coeffs = ()
            for F in fields:
                if isinstance(Vh, ProductFemSpace):
                    basis_values = [CollocationBasisValues(grid, V, nderiv=nderiv) for V in Vh.spaces]
                    basis = [bs.basis for bs in basis_values]
                    spans = [bs.spans for bs in basis_values]
                    # transpose the basis and spans
                    degrees = list(np.array(Vh.degree).T.flatten())
                    basis   = list(map(list, zip(*basis)))
                    spans   = list(map(list, zip(*spans)))
                    basis   = [b for bs in basis for b in bs]
                    spans   = [s for sp in spans for s in sp]
                    coeffs  = coeffs + tuple(F.coeffs[i] for i in range(len(Vh.spaces)))
                else:
                    
                    basis_values = CollocationBasisValues(grid, Vh, nderiv=nderiv)
                    basis   = basis_values.basis
                    spans   = basis_values.spans
                    degrees = Vh.degree
                    coeffs  = coeffs + (F.coeffs,)

            args = grid + coeffs + (*degrees, *basis, *spans)
            
        args = (Vh,) + args 
        values = self.func(*args)

        return values


