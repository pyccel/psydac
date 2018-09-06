from sympy import Basic
from sympy import symbols, Symbol, IndexedBase, Matrix
from sympy import Mul, Add

from pyccel.ast.core import For
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Slice
from pyccel.ast.core import Range
from pyccel.ast.core import FunctionDef
from pyccel.ast import Zeros
from pyccel.ast import Import
from pyccel.parser.parser import _atomic

from sympde.core import Constant
from sympde.core import Field
from sympde.core import atomize
from sympde.core import BilinearForm, LinearForm, FunctionForm, BasicForm
from sympde.core.derivatives import _partial_derivatives
from sympde.core.space import TestFunction
from sympde.core.space import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, FunctionForm
from sympde.printing.pycode import pycode

FunctionalForms = (BilinearForm, LinearForm, FunctionForm)

def compute_atoms_expr(atom,indexes_qds,indexes_test,
                      indexes_trial, basis_trial,
                      basis_test,cords,test_function):

    cls = (_partial_derivatives,
           VectorTestFunction,
           TestFunction)

    if not isinstance(atom, cls):
        raise TypeError('atom must be of type {}'.format(str(cls)))

    if isinstance(atom, _partial_derivatives):
        direction = atom.grad_index + 1
        test      = test_function in atom.atoms(TestFunction)
    else:
        direction = 0
        test      = atom == test_function

    if test:
        basis  = basis_test
        idxs   = indexes_test
    else:
        basis  = basis_trial
        idxs   = indexes_trial

    args = []
    dim  = len(indexes_test)
    for i in range(dim):
        if direction == i+1:
            args.append(basis[i][idxs[i],1,indexes_qds[i]])
        else:
            args.append(basis[i][idxs[i],0,indexes_qds[i]])
    return Assign(atom, Mul(*args))

def compute_atoms_expr_field(atom, indexes_qds,
                            idxs, basis,
                            test_function):

    if not is_field(atom):
        raise TypeError('atom must be a field expr')

    field = list(atom.atoms(Field))[0]
    field_name = 'coeff_'+str(field.name)
    if isinstance(atom, _partial_derivatives):
        direction = atom.grad_index + 1

    else:
        direction = 0

    args = []
    dim  = len(idxs)
    for i in range(dim):
        if direction == i+1:
            args.append(basis[i][idxs[i],1,indexes_qds[i]])
        else:
            args.append(basis[i][idxs[i],0,indexes_qds[i]])

    body = [Assign(atom, Mul(*args))]
    args = [IndexedBase(field_name)[idxs],atom]
    val_name = pycode(atom) + '_values'
    val  = IndexedBase(val_name)[indexes_qds]
    body.append(AugAssign(val,'+',Mul(*args)))
    return body

def is_field(expr):

    if isinstance(expr, _partial_derivatives):
        return is_field(expr.args[0])

    elif isinstance(expr, Field):
        return True

    return False

class Kernel(Basic):

    def __new__(cls, weak_form, name=None):
        if not isinstance(weak_form, FunctionalForms):
            raise TypeError(' instance not a weak formulation')

        if name is None:
            form = weak_form.__class__.__name__
            ID = abs(hash(weak_form))
            name = '{form}_{ID}'.format(form=form, ID=ID)

        obj = Basic.__new__(cls, weak_form)
        obj._name = name
        return obj

    @property
    def weak_form(self):
        return self._args[0].expr

    @property
    def test_function(self):
        return self._args[0].test_functions[0]

    @property
    def trial_function(self):
        return self._args[0].trial_functions[0]

    @property
    def name(self):
        return self._name

    @property
    def fields(self):
        return self._args[0].fields

    @property
    def dim(self):
        return self._args[0].ldim

    @property
    def mapping(self):
        return self._args[0].mapping

    @property
    def coordinates(self):
        cor = self._args[0].coordinates
        if self.dim==1:
            cor = [cor]
        return cor

    @property
    def mapping(self):
        return self._args[0].mapping

    @property
    def is_lineair(self):
        return isinstance(self._args[0], LinearForm)

    @property
    def is_bilinear(self):
        return isinstance(self._args[0], BilinearForm)

    @property
    def is_function(self):
        return isinstance(self._args[0], FunctionForm)

    @property
    def expr(self):
        cls = (_partial_derivatives,
              VectorTestFunction,
              TestFunction)

        if self.is_bilinear:
            dim_trial = self.dim
        else:
            dim_trial = 0

        weak_form = self.weak_form

        expr = atomize(weak_form)
        dim_test = self.dim
        dim      = self.dim
        coordinates = self.coordinates
        conts  = tuple(expr.atoms(Constant))

        atoms  = _atomic(expr, cls=cls)

        atomic_expr_field = [atom for atom in atoms if is_field(atom)]
        atomic_expr       = [atom for atom in atoms if atom not in atomic_expr_field ]

        fields_str    = tuple(map(pycode, atomic_expr_field))
        field_atoms   = tuple(expr.atoms(Field))

        test_function = self.test_function

        # creation of symbolic vars
        if isinstance(expr, Matrix):
            sh   = expr.shape
            mats = symbols('mat0:{}(0:{})'.format(sh[0], sh[1]),cls=IndexedBase)
            v    = symbols('v0:{}(0:{})'.format(sh[0], sh[1]),cls=IndexedBase)
            expr = expr[:]
            ln   = len(expr)

        else:
            mats = (IndexedBase('mat'),)
            v    = [symbols('v')]
            expr = [expr]
            ln   = 1

        wvol          = symbols('wvol')
        basis_trial   = symbols('trial_bs0:%d'%dim_trial, cls=IndexedBase)
        basis_test    = symbols('test_bs0:%d'%dim_test, cls=IndexedBase)
        weighted_vols = symbols('w0:%d'%dim, cls=IndexedBase)
        positions     = symbols('u0:%d'%dim, cls=IndexedBase)
        test_pads     = symbols('test_p1:%d'%(dim_test+1))
        trial_pads    = symbols('trial_p1:%d'%(dim_test+1))
        indexes_qds   = symbols('g1:%d'%(dim+1))
        qds_dim       = symbols('k1:%d'%(dim+1))
        indexes_test  = symbols('il1:%d'%(dim_test+1))
        indexes_trial = symbols('jl1:%d'%(dim_trial+1))
        fields        = symbols(fields_str)
        fields_val    = symbols(tuple(f+'_values' for f in fields_str),cls=IndexedBase)
        fields_coeff   = symbols(tuple('coeff_'+str(f) for f in field_atoms),cls=IndexedBase)

        # ranges
        ranges_qdr   = [Range(qds_dim[i]) for i in range(dim)]
        ranges_test  = [Range(test_pads[i]) for i in range(dim_test)]
        ranges_trial = [Range(trial_pads[i]) for i in range(dim_trial)]
        # ...

        # body of kernel
        body   = [Assign(coordinates[i],positions[i][indexes_qds[i]])\
                 for i in range(dim)]
        body  += [compute_atoms_expr(atom,indexes_qds,
                                      indexes_test,
                                      indexes_trial,
                                      basis_trial,
                                      basis_test,
                                      coordinates,
                                      test_function) for atom in atomic_expr]

        weighted_vol = [weighted_vols[i][indexes_qds[i]] for i in range(dim)]
        weighted_vol = Mul(*weighted_vol)
        # ...
        # add fields
        for i in range(len(fields_val)):
            body.append(Assign(fields[i],fields_val[i][indexes_qds]))

        body.append(Assign(wvol,weighted_vol))

        for i in range(ln):
            body.append(AugAssign(v[i],'+',Mul(expr[i],wvol)))
        # ...

        # ...
        # put the body in for loops of quadrature points
        for i in range(dim-1,-1,-1):
            body = [For(indexes_qds[i],ranges_qdr[i],body)]

        # initialization of intermediate vars
        inits = [Assign(v[i],0.0) for i in range(ln)]
        body = inits + body
        # ...

        if dim_trial:
            trial_idxs = tuple([indexes_trial[i]+trial_pads[i]-indexes_test[i] for i in range(dim)])
            idxs = indexes_test + trial_idxs
        else:
            idxs = indexes_test

        for i in range(ln):
            body.append(Assign(mats[i][idxs],v[i]))

        # ...
        # put the body in tests and trials for loops
        for i in range(dim_trial-1,-1,-1):
            body = [For(indexes_trial[i],ranges_trial[i],body)]

        for i in range(dim_test-1,-1,-1):
            body = [For(indexes_test[i],ranges_test[i],body)]
        # ...

        # ...
        # initialization of the matrix
        inits = [mats[i][[Slice(None,None)]*(dim_test+dim_trial)] for i in range(ln)]
        inits = [Assign(e, 0.0) for e in inits]
        body =  inits + body

        # calculate field values
        allocate = [Assign(f, Zeros(qds_dim)) for f in fields_val]
        f_body   = [e  for atom in atomic_expr_field
                       for e in compute_atoms_expr_field(atom,
                       indexes_qds, indexes_test,basis_test,
                       test_function)]
#        f_body   = [e  for atom in field_atoms
#                       for e in compute_atoms_expr_field(atom,
#                       indexes_qds, indexes_test,basis_test,
#                       test_function)]

        if f_body:
            # put the body in for loops of quadrature points
            for i in range(dim-1,-1,-1):
                f_body = [For(indexes_qds[i],ranges_qdr[i],f_body)]

            # put the body in tests for loops
            for i in range(dim-1,-1,-1):
                f_body = [For(indexes_test[i],ranges_test[i],f_body)]

        body = allocate + f_body + body

        # function args
        func_args = (test_pads + trial_pads + qds_dim + basis_test + basis_trial
                     + positions + weighted_vols + mats + conts + fields_coeff)

        return FunctionDef(self.name, list(func_args), [], body)
