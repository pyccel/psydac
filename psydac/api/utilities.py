# coding: utf-8

from sympy.core.containers import Tuple
from sympy import Matrix, ImmutableDenseMatrix, MutableDenseNDimArray

import inspect
import sys
import os
import importlib
import string
import random
import numpy as np

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
def random_string( n ):
    # we remove uppercase letters because of f2py
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

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

