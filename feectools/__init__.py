# MPI switch, read once when feectools.ddm.mpi is first imported:
#   None  -> use mpi4py if it is available (default)
#   False -> do not import mpi4py, use MockMPI (serial runs; saves ~1 s of import time)
use_mpi = None
