#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from sympde.expr.expr import (
    BilinearForm as sym_BilinearForm,
    LinearForm as sym_LinearForm,
    Functional as sym_Functional
)

from .basic             import BasicDiscrete
from .fem               import DiscreteFunctional
from .fem               import DiscreteLinearForm
from .fem               import DiscreteBilinearForm
from .fem_bilinear_form import DiscreteBilinearForm as DiscreteBilinearForm_SF
from .fem_common        import reset_arrays
from .utilities         import random_string

__all__ = ('DiscreteSumForm',)

#==============================================================================
class DiscreteSumForm(BasicDiscrete):

    def __init__(self, a, kernel_expr, *args, **kwargs):

        # Sum factorization is only implemented for bilinear forms in 3D, in
        # which case we use it by default. A 2D implementation should be the
        # next step, hence we allow the user to pass `sum_factorization=True`
        # even if not supported yet. In the case of linear forms or functionals
        # this option is irrelevant for now, so we ignore it.
        #
        # In every case we remove the `sum_factorization` key from the dict
        # in order to avoid errors, because none of the class constructors
        # accept this argument.
        sum_factorization = kwargs.pop('sum_factorization', a.ldim == 3)

        # TODO Uncomment when the SesquilinearForm exist in SymPDE
        #if not isinstance(a, (sym_BilinearForm, sym_SesquilinearForm, sym_LinearForm, sym_Functional)):
            # raise TypeError('> Expecting a symbolic BilinearForm, SesquilinearForm, LinearForm, Functional')
        if not isinstance(a, (sym_BilinearForm, sym_LinearForm, sym_Functional)):
            raise TypeError('> Expecting a symbolic BilinearForm, LinearForm, Functional')

        self._expr = a
        backend = kwargs.pop('backend', None)
        self._backend = backend

        folder = kwargs.get('folder', None)
        self._folder = self._initialize_folder(folder)

        # create a module name if not given
        tag = random_string(8)

        # ...
        forms = []
        free_args = []
        self._kernel_expr = kernel_expr
        operator = None
        for e in kernel_expr:

            # Currently sum factorization can only be used for interior domains
            from sympde.expr.evaluation import DomainExpression
            is_interior_expr = isinstance(e, DomainExpression)

            if isinstance(a, sym_LinearForm):
                kwargs['update_ghost_regions'] = False
                ah = DiscreteLinearForm(a, e, *args, backend=backend, **kwargs)
                kwargs['vector'] = ah._vector
                operator = ah._vector

            # TODO Uncomment when the SesquilinearForm exist in SymPDE
            # elif isinstance(a, sym_SesquilinearForm):
            #     kwargs['update_ghost_regions'] = False
            #     ah = DiscreteSesquilinearForm(a, e, *args, assembly_backend=backend, **kwargs)
            #     kwargs['matrix'] = ah._matrix
            #     operator = ah._matrix

            elif isinstance(a, sym_BilinearForm):
                kwargs['update_ghost_regions'] = False
                if sum_factorization and is_interior_expr:
                    ah = DiscreteBilinearForm_SF(a, e, *args, assembly_backend=backend, **kwargs)
                else:
                    ah = DiscreteBilinearForm(a, e, *args, assembly_backend=backend, **kwargs)
                kwargs['matrix'] = ah._matrix
                operator = ah._matrix

            elif isinstance(a, sym_Functional):
                ah = DiscreteFunctional(a, e, *args, backend=backend, **kwargs)

            forms.append(ah)
            free_args.extend(ah.free_args)

        if isinstance(a, sym_BilinearForm):
            is_broken   = len(args[0].domain)>1
            if self._backend is not None and is_broken:
                for mat in kwargs['matrix']._blocks.values():
                    mat.set_backend(backend)
            elif self._backend is not None:
                kwargs['matrix'].set_backend(backend)

        self._forms         = forms
        self._operator      = operator
        self._free_args     = tuple(set(free_args))
        self._is_functional = isinstance(a, sym_Functional)
        # ...

    @property
    def forms(self):
        return self._forms

    @property
    def free_args(self):
        return self._free_args

    @property
    def is_functional(self):
        return self._is_functional

    def assemble(self, *, reset=True, **kwargs):
        if not self.is_functional:
            if reset :
                reset_arrays(*[i for M in self.forms for i in M.global_matrices])

            for form in self.forms:
                form.assemble(reset=False, **kwargs)
            self._operator.exchange_assembly_data()
            return self._operator
        else:
            M = [form.assemble(**kwargs) for form in self.forms]
            M = np.sum(M)
            return M
