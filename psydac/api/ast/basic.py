#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympy import Basic

#==============================================================================
class SplBasic(Basic):

    def __new__(cls, tag, name=None, prefix=None, debug=False, detailed=False,
                mapping=None, domain=None, is_rational_mapping=None, comm=None):

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
        obj._comm                = comm
        obj._imports = []

        return obj

    @property
    def name(self):
        return self._name

    @property
    def tag(self):
        return self._tag

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
    def mapping(self):
        return self._mapping

    @property
    def domain(self):
        return self._domain

    @property
    def is_rational_mapping(self):
        return self._is_rational_mapping

    @property
    def comm(self):
        return self._comm

    @property
    def imports(self):
        return self._imports

    #--------------------------------------------------------------------------
    # WARNING: PROPERTIES ACCESSING ATTRIBUTES THAT ARE NOT IN BASE CLASS
    #--------------------------------------------------------------------------
    @property
    def func(self):
        return self._func

    @property
    def basic_args(self):
        return self._basic_args

    @property
    def boundary(self):
        return self._boundary
