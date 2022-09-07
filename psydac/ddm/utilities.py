# coding: utf-8

from .cart                       import CartDecomposition, InterfaceCartDecomposition
from .blocking_data_exchanger    import BlockingCartDataExachanger
from .nonblocking_data_exchanger import NonBlockingCartDataExachanger
from .interface_data_exchanger   import InterfaceCartDataExchanger

__all__ = ['get_data_exchanger']

def get_data_exchanger(cart, dtype, *, coeff_shape=(),  assembly=False, axis=None, shape=None, blocking=False):

    if isinstance(cart, InterfaceCartDecomposition):
        return InterfaceCartDataExchanger(cart, dtype, coeff_shape=coeff_shape)

    elif isinstance(cart, CartDecomposition):
        if blocking:
            return BlockingCartDataExachanger(cart, dtype, coeff_shape=coeff_shape, assembly=assembly, axis=axis, shape=shape)

        else:
            return NonBlockingCartDataExachanger(cart, dtype, coeff_shape=coeff_shape, assembly=assembly, axis=axis, shape=shape)
    else:
        raise TypeError('cart can only be of type CartDecomposition or InterfaceCartDecomposition')
        

