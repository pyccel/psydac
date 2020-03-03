from sympy.core.containers import Tuple

def flatten(args):
    ls = []
    def rec_flatten(args, ls):
        if isinstance(args, (list, tuple, Tuple)):
            for i in args:
                rec_flatten(i,ls)
        else:
            ls.append(args)
    rec_flatten(args, ls)

    if isinstance(args, tuple):
        return tuple(ls)
    elif isinstance(args, Tuple):
        return Tuple(*ls)
    else:
        return ls
