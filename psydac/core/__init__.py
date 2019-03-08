from psydac.core import bsplines

__all__ = ['bsplines']

try:
    from psydac.core import bsp
    from psydac.core import interface
except ImportError:
    pass
else:
    __all__.extend( ['bsp','interface'] )
