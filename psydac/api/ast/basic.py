from collections import OrderedDict
from itertools import groupby
import numpy as np

from sympy import Basic

#==============================================================================
class SplBasic(Basic):

    def __new__(cls, tag, name=None, prefix=None, debug=False, detailed=False,
                mapping=None, domain=None, is_rational_mapping=None):

        if name is None:
            if prefix is None:
                raise ValueError('prefix must be given')

            name = '{prefix}_{tag}'.format(tag=tag, prefix=prefix)

        obj = Basic.__new__(cls)

        obj._name                = name
        obj._tag                 = tag
        obj._dependencies        = []
        obj._debug               = debug
        obj._detailed            = detailed
        obj._mapping             = mapping
        obj._domain              = domain
        obj._is_rational_mapping = is_rational_mapping
        obj._imports = []

        return obj

    @property
    def name(self):
        return self._name

    @property
    def tag(self):
        return self._tag

    @property
    def func(self):
        return self._func

    @property
    def basic_args(self):
        return self._basic_args

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def debug(self):
        return self._debug

    @property
    def detailed(self):
        return self._detailed

    @property
    def domain(self):
        return self._domain

    @property
    def mapping(self):
        return self._mapping

    @property
    def is_rational_mapping(self):
        return self._is_rational_mapping

    @property
    def boundary(self):
        return self._boundary

    @property
    def imports(self):
        return self._imports
