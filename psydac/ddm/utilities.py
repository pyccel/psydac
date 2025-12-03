#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from .cart                       import CartDecomposition, InterfaceCartDecomposition
from .blocking_data_exchanger    import BlockingCartDataExchanger
from .nonblocking_data_exchanger import NonBlockingCartDataExchanger
from .interface_data_exchanger   import InterfaceCartDataExchanger


__all__ = ('get_data_exchanger',)

def get_data_exchanger(cart, dtype, *, coeff_shape=(),  assembly=False, axis=None, shape=None, blocking=True):

    if isinstance(cart, InterfaceCartDecomposition):
        return InterfaceCartDataExchanger(cart, dtype, coeff_shape=coeff_shape)

    elif isinstance(cart, CartDecomposition):
        if blocking:
            return BlockingCartDataExchanger(cart, dtype, coeff_shape=coeff_shape, assembly=assembly, axis=axis, shape=shape)

        else:
            return NonBlockingCartDataExchanger(cart, dtype, coeff_shape=coeff_shape, assembly=assembly, axis=axis, shape=shape)
    else:
        raise TypeError('cart can only be of type CartDecomposition or InterfaceCartDecomposition')
        

