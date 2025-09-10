# coding: utf-8

import os
import string
import random

from sympy.core.containers import Tuple
from sympy import Matrix, ImmutableDenseMatrix, MutableDenseNDimArray

__all__ = (
    'flatten',
    'mkdir_p',
    'touch_init_file',
    'random_string',
    'write_code'
)
#==============================================================================
def flatten(args):

    types_to_flatten = (
        list,
        tuple,
        Tuple,
        Matrix,
        ImmutableDenseMatrix,
        MutableDenseNDimArray,
    )

    ls = []
    def rec_flatten(args, ls):
        if isinstance(args, types_to_flatten):
            for i in tuple(args):
                rec_flatten(i, ls)
        else:
            ls.append(args)
    rec_flatten(args, ls)

    if isinstance(args, tuple):
        return tuple(ls)
    elif isinstance(args, Tuple):
        return Tuple(*ls)
    else:
        return ls

#==============================================================================
def mkdir_p(folder):
    if os.path.isdir(folder):
        return
    os.makedirs(folder, exist_ok=True)

#==============================================================================
def touch_init_file(path):
    mkdir_p(path)
    path = os.path.join(path, '__init__.py')
    with open(path, 'a'):
        os.utime(path, None)

#==============================================================================
def random_string(size: int = 8,
                chars: int = string.ascii_lowercase + string.digits) -> str:
    """
    Create a random string of given length to be used in generated file names.

    Parameters
    ----------
    size : int, optional
        Length of the string (default: 8).

    chars : str, optional
        A string with the avalailable characters for random drawing (default:
        ASCII lower case characters + decimal digits)

    Returns
    -------
    str
        A random string of given length, made of the given characters.
    """
    return ''.join(random.choice(chars) for _ in range(size))

#==============================================================================
def write_code(filename, code, folder=None):
    if not folder:
        folder = os.getcwd()

    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise ValueError('{} folder does not exist'.format(folder))

    filename = os.path.basename( filename )
    filename = os.path.join(folder, filename)

    # TODO check if init exists
    # add __init__.py for imports
    touch_init_file(folder)

    f = open(filename, 'w')
    for line in code:
        f.write(line)
    f.close()

    return filename
