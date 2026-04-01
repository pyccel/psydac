import numpy as np
from mpi4py.util.dtlib import from_numpy_dtype

def gather_vlen_array(v, mpi_comm):

    """ Gather 1D arrays of possibly different lengths onto root process
        Other processes return None
        Local arrays must be of the same type
    """
    dtype = v.dtype

    v = np.ascontiguousarray(v, dtype=dtype)
    local_len = np.array(v.size, dtype=np.int64)

    # Gather local array sizes on root process
    recvcounts = np.empty(mpi_comm.size, dtype=np.int64) if mpi_comm.rank == 0 else None
    mpi_comm.Gather(local_len, recvcounts, root=0)

    if mpi_comm.rank == 0:
        displs = np.empty(mpi_comm.size, dtype=np.int64)
        displs[0] = 0
        displs[1:] = np.cumsum(recvcounts[:-1])

        total_len = int(np.sum(recvcounts))
        global_array = np.empty(total_len, dtype=dtype)

    else:
        displs = None
        global_array = None
    mpi_type = from_numpy_dtype(dtype)

    mpi_comm.Gatherv(v, [global_array, recvcounts, displs, mpi_type], root=0)

    return global_array