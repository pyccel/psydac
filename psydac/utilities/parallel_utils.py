# ------------------------------------------------------------------------- #
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
# ------------------------------------------------------------------------- #
"""
Common general utilities for MPI-parallel simulations.
"""

__all__ = ("gather_vlen_array", "gather_vlen_arrays", "parallel_run_from_cli")


def parallel_run_from_cli(parser, main):
    """Parse CLI arguments and run main, in parallel.

    This function makes sure that the standard output remains clean when the
    size of the MPI communicator is > 1, in the case of argparse's help or
    error messages. In addition, it prints the MPI communicator size and the
    CLI arguments. When not run (serially) from IPython, it also calls
    `plt.show()` to keep any Matplotlib windows open after `main` is run.

    Parameters
    ----------
    parser : callable
        Function which parses the CLI arguments using the argparse library.
        It has no input arguments, and it returns an argparse Namespace.
        This function is only run by the root process, which has MPI rank 0.
        The output Namespace is then broadcasted to all the other processes.

    main : callable
        Main function to be run in parallel. Its arguments are all keyword-
        only, and most of them are extracted from the parser output using
        vars(). An additional important argument is `mpi_comm`, which is an
        instance of `mpi4py.MPI.Comm`. This function returns a dict with all
        its local variables, e.g. by using `return locals()`, which is useful
        for interactive usage. If matplotlib figures are generated, these
        are made visible with `fig.show()`, but without calling `plt.show()`.

    Returns
    -------
    dict
        Dictionary with all the local variables. Useful for interactive usage.
    """
    from mpi4py import MPI

    # Select MPI communicator
    mpi_comm = MPI.COMM_WORLD

    # Let root process (with rank=0) parse the CLI arguments and broadcast them
    # to all the other processes in the communicator. This keeps the standard
    # output clean in the case of help and error messages.
    args = None
    try:
        if mpi_comm.rank == 0:
            args = parser()
    finally:
        args = mpi_comm.bcast(args, root=0)
        if args is None:
            import sys

            sys.exit(0)

    # Convert argparse Namespace to dictionary and add MPI communicator to it
    args_dict = vars(args)
    args_dict["mpi_comm"] = mpi_comm

    # Print information to standard output
    if mpi_comm.rank == 0:
        import pprint

        main_name = main.__name__
        if mpi_comm.size == 1:
            msg = f"Running function '{main_name}' in serial with arguments:"
        else:
            msg = (
                f"Running function '{main_name}' in parallel with "
                f"{mpi_comm.size} MPI processes, and arguments:"
            )
        print(msg)
        pprint.pp(args_dict)
        print(flush=True)

    # Store local variables in a dictionary
    namespace = locals()

    # Run simulation in parallel, and update local variables
    namespace.update(main(**args_dict))

    # Keep matplotlib windows open (not necessary in IPython sessions)
    try:
        __IPYTHON__
    except NameError:
        import matplotlib.pyplot as plt

        namespace["plt"] = plt
        plt.show()

    # Return all local variables
    return namespace


def gather_vlen_arrays(arrays, mpi_comm, mpi_root=0):
    """
    Gather several 1D NumPy arrays of the same local length onto root process.
    Return None on the other processes. Length may vary between processes.

    Parameters
    ----------
    arrays : iterable of 1D arrays
        All local arrays must have the same length and dtype
        All ranks must pass the same number of arrays
    mpi_comm : mpi4py.MPI.Comm
        MPI communicator.
    mpi_root : int, default=0
        Rank of root process in MPI communicator.

    Returns
    -------
    tuple[int, ...] or None
        On root, tuple of gathered 1D arrays. None on the other processes.
    """

    import numpy as np
    from mpi4py.util.dtlib import from_numpy_dtype

    rank = mpi_comm.rank
    arrays = tuple(np.asarray(a) for a in arrays)
    dtype = arrays[0].dtype
    arr_len = arrays[0].size

    for a in arrays:
        if a.dtype != dtype:
            raise TypeError("all arrays must have same dtype")
        if a.size != arr_len:
            raise ValueError("all arrays must have same size")

    # reshape as (arr_len, num of arrays) and flatten
    arrays_flat = np.ascontiguousarray(np.column_stack(arrays)).ravel(order="C")

    local_len = np.array(arrays_flat.size, dtype=np.int64)

    # Gather local flattened array sizes on root process
    recvcounts = np.empty(mpi_comm.size, dtype=np.int64) if rank == mpi_root else None
    mpi_comm.Gather(local_len, recvcounts, root=mpi_root)

    if rank == mpi_root:
        displs = np.empty(mpi_comm.size, dtype=np.int64)
        displs[0] = 0
        displs[1:] = np.cumsum(recvcounts[:-1])

        total_len = int(np.sum(recvcounts))
        global_array = np.empty(total_len, dtype=dtype)

    else:
        displs = None
        global_array = None
    mpi_type = from_numpy_dtype(dtype)

    mpi_comm.Gatherv(
        arrays_flat, [global_array, recvcounts, displs, mpi_type], root=mpi_root
    )

    if rank != mpi_root:
        return None

    num_arr = len(arrays)
    gathered = global_array.reshape(-1, num_arr, order="C")
    return tuple(gathered[:, j].copy() for j in range(num_arr))


def gather_vlen_array(v, mpi_comm, mpi_root=0):
    """Gather 1D arrays of possibly different lengths onto root process
    Other processes return None
    Local arrays must be of the same type
    """

    result = gather_vlen_arrays((v,), mpi_comm, mpi_root=mpi_root)
    if mpi_comm.rank == mpi_root:
        return result[0]
    return None
