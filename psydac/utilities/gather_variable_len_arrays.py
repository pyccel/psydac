import numpy as np
from mpi4py.util.dtlib import from_numpy_dtype


def gather_vlen_arrays(arrays, mpi_comm, mpi_root=0):
    """
    Gather several 1D NumPy arrays of the same local length onto root process. Length may vary between processes.

    Parameters
    ----------
    arrays : iterable of 1D arrays
        All local arrays must have the same length and dtype
        All ranks must pass the same number of arrays
    mpi_comm : MPI communicator
    mpi_root : int, default=0

    Returns
    -------
    On root:
        tuple of gathered 1D arrays
    On other ranks:
        None
    """

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


# ===============================================================================
def gather_vlen_array(v, mpi_comm, mpi_root=0):
    """Gather 1D arrays of possibly different lengths onto root process
    Other processes return None
    Local arrays must be of the same type
    """

    result = gather_vlen_arrays((v,), mpi_comm, mpi_root=mpi_root)
    if mpi_comm.rank == mpi_root:
        return result[0]
    return None
