from feectools.core import bsplines

__all__ = ['bsplines']

try:
    from feectools.core import bsp
    from feectools.core import interface
except ImportError:
    pass
else:
    __all__.extend( ['bsp','interface'] )
