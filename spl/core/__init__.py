try:
    from spl.core import bsp
    from spl.core import interface
except ImportError:
    pass
else:
    __all__ = ['bsp','interface']
