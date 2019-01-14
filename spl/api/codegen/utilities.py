from pyccel.ast import Variable, IndexedVariable
import re
import string
from sympy.utilities.iterables import cartes
_range = re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')


def variables(names, dtype, **args):

    def contruct_variable(cls, name, dtype, rank, **args):
        if cls == Variable:
            return Variable(dtype,  name, rank=rank, **args)
        elif cls == IndexedVariable:

            return IndexedVariable(name, dtype=dtype, rank=rank, **args)
        else:
            raise TypeError('only Variables and IndexedVariables are supported')
            
    result = []
    cls = args.pop('cls', Variable)
    
    rank = args.pop('rank', 0)
    
    if isinstance(names, str):
        marker = 0
        literals = [r'\,', r'\:', r'\ ']
        for i in range(len(literals)):
            lit = literals.pop(0)
            if lit in names:
                while chr(marker) in names:
                    marker += 1
                lit_char = chr(marker)
                marker += 1
                names = names.replace(lit, lit_char)
                literals.append((lit_char, lit[1:]))
        def literal(s):
            if literals:
                for c, l in literals:
                    s = s.replace(c, l)
            return s

        names = names.strip()
        as_seq = names.endswith(',')
        if as_seq:
            names = names[:-1].rstrip()
        if not names:
            raise ValueError('no symbols given')

        # split on commas
        names = [n.strip() for n in names.split(',')]
        if not all(n for n in names):
            raise ValueError('missing symbol between commas')
        # split on spaces
        for i in range(len(names) - 1, -1, -1):
            names[i: i + 1] = names[i].split()
    
        seq = args.pop('seq', as_seq)  

        for name in names:
            if not name:
                raise ValueError('missing variable')

            if ':' not in name:
                var = contruct_variable(cls, literal(name), dtype, rank, **args)
                result.append(var)
                continue

            split = _range.split(name)
            # remove 1 layer of bounding parentheses around ranges
            for i in range(len(split) - 1):
                if i and ':' in split[i] and split[i] != ':' and \
                        split[i - 1].endswith('(') and \
                        split[i + 1].startswith(')'):
                    split[i - 1] = split[i - 1][:-1]
                    split[i + 1] = split[i + 1][1:]
            for i, s in enumerate(split):
                if ':' in s:
                    if s[-1].endswith(':'):
                        raise ValueError('missing end range')
                    a, b = s.split(':')
                    if b[-1] in string.digits:
                        a = 0 if not a else int(a)
                        b = int(b)
                        split[i] = [str(c) for c in range(a, b)]
                    else:
                        a = a or 'a'
                        split[i] = [string.ascii_letters[c] for c in range(
                            string.ascii_letters.index(a),
                            string.ascii_letters.index(b) + 1)]  # inclusive
                    if not split[i]:
                        break
                else:
                    split[i] = [s]
            else:
                seq = True
                if len(split) == 1:
                    names = split[0]
                else:
                    names = [''.join(s) for s in cartes(*split)]
                if literals:
                    result.extend([contruct_variable(cls, literal(s), dtype, rank, **args) for s in names])
                else:
                    result.extend([contruct_variable(cls, s, dtype, rank, **args) for s in names])

        if not seq and len(result) <= 1:
            if not result:
                return ()
            return result[0]

        return tuple(result)
    elif isinstance(names,(tuple,list)):
        return tuple(variables(i, dtype, cls=cls,rank=rank,**args) for i in names)
    else:
        raise TypeError('Expecting a string')




    
    
def build_pythran_types_header(name, args, order=None):
    """
    builds a types decorator from a list of arguments (of FunctionDef)
    """
    types = []
    for a in args:
        if isinstance(a, Variable):
            dtype = pythran_dtypes[a.dtype.name.lower()]

        elif isinstance(a, IndexedVariable):
            dtype = pythran_dtypes[a.dtype.name.lower()]

        else:
            raise TypeError('unepected type for {}'.format(a))

        if a.rank > 0:
            shape = ['[]' for i in range(0, a.rank)]
            shape = ''.join(i for i in shape)
            dtype = '{dtype}{shape}'.format(dtype=dtype, shape=shape)
            if order and a.rank > 1:
                dtype = "{dtype}".format(dtype=dtype, ordering=order)

        types.append(dtype)
    types = ', '.join(_type for _type in types)
    header = '#pythran export {name}({types})'.format(name=name, types=types)
    return header
    
pythran_dtypes = {'real':'float','int':'int'}
