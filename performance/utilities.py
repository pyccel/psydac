from tabulate import tabulate

__all__ = ('print_timings_table',)

#==============================================================================
def print_timings_table(python, other):
    """
    Print a table with the timings of pure Python and Pyccel, and the speedups.

    The two input dictionaries should be compatible, in the sense that `other`
    should have all the keys of `python`. Each key refers to a different
    operation that was timed in the code.

    Parameters
    ----------
    python : dict[str, float]
        Timing using the Python backend.

    other : dict[str, float]
        Timing using some Pyccel backend.

    Returns
    -------
    str
        The LaTeX table with timings and speedups.
    """
    table   = []
    headers = ['Assembly time', 'Python', 'Pyccel', 'Speedup']

    for kind, time_python in python.items():
        time_other = other[kind]
        speedup = time_python / time_other
        line = [kind, time_python, time_other, speedup]
        table.append(line)

    print(tabulate(table, headers=headers, tablefmt='latex'))
