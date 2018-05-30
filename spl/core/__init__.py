from spl.core import bsplines_non_uniform

__all__ = ['bsplines_non_uniform']

try:
    from spl.core import bsp
    from spl.core import interface
except ImportError:
    pass
else:
    __all__.extend( ['bsp','interface'] )
