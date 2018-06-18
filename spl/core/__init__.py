from spl.core import bsplines

__all__ = ['bsplines']

try:
    from spl.core import bsp
    from spl.core import interface
except ImportError:
    pass
else:
    __all__.extend( ['bsp','interface'] )
