# coding: utf-8

# TODO for the moment we assume Product of same space

import numpy as np
from itertools import product
from scipy.linalg import eig as eig_solver

from gelato.expr     import GltExpr as sym_GltExpr

from psydac.api.basic         import BasicCodeGen
from psydac.api.ast.glt       import GltKernel
from psydac.api.ast.glt       import GltInterface
from psydac.api.settings      import PSYDAC_BACKEND_PYTHON, PSYDAC_DEFAULT_FOLDER
from psydac.api.grid          import CollocationBasisValues

from psydac.cad.geometry      import Geometry
from psydac.mapping.discrete  import SplineMapping, NurbsMapping

from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace


#==============================================================================
class DiscreteGltExpr(BasicCodeGen):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_GltExpr):
            raise TypeError('> Expecting a symbolic Glt expression')

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
        self._spaces = args[1]
        # ...

        # ...
        kwargs['mapping'] = self.spaces[0].symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping

        BasicCodeGen.__init__(self, expr, **kwargs)
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
    def spaces(self):
        return self._spaces

    # TODO add comm and treate parallel case
    def _create_ast(self, expr, tag, **kwargs):

        mapping             = kwargs.pop('mapping', None)
        backend             = kwargs.pop('backend', PSYDAC_BACKEND_PYTHON)
        is_rational_mapping = kwargs.pop('is_rational_mapping', None)

        # ...
        kernel = GltKernel( expr, self.spaces,
                            name = 'kernel_{}'.format(tag),
                            mapping = mapping,
                            is_rational_mapping = is_rational_mapping,
                            backend = backend )

        interface = GltInterface( kernel,
                                  name = 'interface_{}'.format(tag),
                                  mapping = mapping,
                                  is_rational_mapping = is_rational_mapping,
                                  backend = backend )
        # ...

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

    def evaluate(self, *args, **kwargs):

        kwargs = self._check_arguments(**kwargs)

        Vh = self.spaces[0]
        is_block = False
        if isinstance(Vh, ProductFemSpace):
            Vh = Vh.spaces[0]
            is_block = True

        if not isinstance(Vh, TensorFemSpace):
            raise NotImplementedError('Only TensorFemSpace is available for the moment')

        args = args + (Vh,)

        dim = Vh.ldim

        # ...
        if self.expr.form.fields or self.mapping:
            nderiv = self.interface.max_nderiv
            xis = [kwargs['arr_x{}'.format(i)] for i in range(1,dim+1)]
            grid = tuple(xis)
            # TODO assert that xis are inside the space domain
            basis_values = CollocationBasisValues(grid, Vh, nderiv=nderiv)
            args = args + (basis_values,)
        # ...

        if self.mapping:
            args = args + (self.mapping,)

        values = self.func(*args, **kwargs)

        if is_block:
            # n_rows = n_cols here
            n_rows = self.interface.n_rows
            n_cols = self.interface.n_cols
            nbasis = [V.nbasis for V in Vh.spaces]

            d = {}
            i = 0
            for i_row in range(0, n_rows):
                for i_col in range(0, n_cols):
                    d[i_row, i_col] = values[i]
                    i += 1

            eig_mat = np.zeros((n_rows,*nbasis))

            # ... compute dtype of the matrix
            dtype = 'float'
            are_complex = [i == 'complex' for i in self.interface.global_mats_types]
            if any(are_complex):
                dtype = 'complex'
            # ...

            mat = np.zeros((n_rows,n_cols), dtype=dtype)

            if dim == 2:
                for i1 in range(0, nbasis[0]):
                    for i2 in range(0, nbasis[1]):
                        mat[...] = 0.
                        for i_row in range(0,n_rows):
                            for i_col in range(0,n_cols):
                                mat[i_row,i_col] = d[i_row,i_col][i1, i2]
                        w,v = eig_solver(mat)
                        wr = w.real
                        eig_mat[:,i1,i2] = wr[:]

            elif dim == 3:
                for i1 in range(0, nbasis[0]):
                    for i2 in range(0, nbasis[1]):
                        for i3 in range(0, nbasis[2]):
                            mat[...] = 0.
                            for i_row in range(0,n_rows):
                                for i_col in range(0,n_cols):
                                    mat[i_row,i_col] = d[i_row,i_col][i1, i2, i3]
                            w,v = eig_solver(mat)
                            wr = w.real
                            eig_mat[:,i1,i2,i3] = wr[:]

            else:
                raise NotImplementedError('')

            values = eig_mat

        return values

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def eig(self, **kwargs):
        """
        Approximates the eigenvalues of the matrix associated to the given
        bilinear form.
        the current algorithm is based on a uniform sampling of the glt symbol.
        """
        Vh = self.spaces[0]
        if isinstance(Vh, ProductFemSpace):
            Vh = Vh.spaces[0]

        if not isinstance(Vh, TensorFemSpace):
            raise NotImplementedError('Only TensorFemSpace is available for the moment')

        nbasis = [V.nbasis for V in Vh.spaces]
        bounds = [V.domain for V in Vh.spaces]
        dim    = Vh.ldim

        # ... fourier variables (as arguments)
        ts = [np.linspace(-np.pi, np.pi, n) for n in nbasis]
        args = tuple(ts)
        # ...

        # ... space variables (as key words)
        if self.interface.with_coordinates:
            xs = [np.linspace(bound[0], bound[1], n) for n, bound in zip(nbasis, bounds)]
            for n,x in zip(['x1', 'x2', 'x3'][:dim], xs):
                kwargs[n] = x
        # ...

        values = self(*args, **kwargs)

        return values
