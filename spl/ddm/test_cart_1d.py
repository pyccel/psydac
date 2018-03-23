import numpy as np
from mpi4py import MPI

from cart import Cart

#===============================================================================
# INPUT PARAMETERS
#===============================================================================

# Number of elements
n1 = 135

# Padding ('thickness' of ghost region)
p1 = 3

# Periodicity
period1 = True

#===============================================================================
# DOMAIN DECOMPOSITION
#===============================================================================

# Parallel info
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Decomposition of Cartesian domain
cart = Cart( npts=[n1+1], pads=[p1], periods=[period1], reorder=False, comm=comm )

# Local 1D array (extended domain)
u = np.zeros( cart.shape, dtype=int ) # NOTE: 64-bit INTEGER!

# Global indices of first and last elements of array
s1, = cart.starts
e1, = cart.ends

# Create MPI subarray datatypes for accessing non-contiguous data
send_types = {}
recv_types = {}
for disp in [-1,1]:
    info = cart.get_shift_info( 0, disp )

    send_types[disp] = MPI.INT64_T.Create_subarray(
        sizes    = u.shape,
        subsizes = info[ 'buf_shape' ],
        starts   = info['send_starts'],
    ).Commit()

    recv_types[disp] = MPI.INT64_T.Create_subarray(
        sizes    = u.shape,
        subsizes = info[ 'buf_shape' ],
        starts   = info['recv_starts'],
    ).Commit()

# Print some info
if rank == 0:
    print( "" )

for k in range(size):
    if k == rank:
        print( "RANK = {}".format( rank ) )
        print( "---------" )
        print( ". s1:e1 = {:2d}:{:2d}".format( s1,e1 ) )
        print( "", flush=True )
    comm.Barrier()

#===============================================================================
# TEST
#===============================================================================

# Fill in true domain with u[i]=i
u[p1:-p1] = [i1 for i1 in range(s1,e1+1)]

# Choose non-negative invertible function tag(disp) >= 0
# NOTE: different values of disp must return different tags!
tag = lambda disp: 42+disp

# Requests' handles
requests = []

# Start receiving data (MPI_IRECV)
for disp in [-1,+1]:
    info     = cart.get_shift_info( 0, disp )
    recv_buf = (u, 1, recv_types[disp])
    recv_req = cart.comm_cart.Irecv( recv_buf, info['rank_source'], tag(disp) )
    requests.append( recv_req )

# Start sending data (MPI_ISEND)
for disp in [-1,+1]:
    info     = cart.get_shift_info( 0, disp )
    send_buf = (u, 1, send_types[disp])
    send_req = cart.comm_cart.Isend( send_buf, info['rank_dest'], tag(disp) )
    requests.append( send_req )

# Wait for end of data exchange (MPI_WAITALL)
MPI.Request.Waitall( requests )

#===============================================================================
# CHECK RESULTS
#===============================================================================

# Verify that ghost cells contain correct data (note periodic domain!)
success = all( u[:] == [i1%(n1+1) for i1 in range(s1-p1,e1+p1+1)] )

# Print error messages (if any) in orderly fashion
for k in range(size):
    if k == rank and not success:
        print( "Rank {}: wrong ghost cell data!".format( rank ), flush=True )
    comm.Barrier()

success_global = comm.reduce( success, op=MPI.LAND, root=0 )

if comm.Get_rank() == 0:
    if success_global:
        print( "PASSED", end='\n\n', flush=True )
    else:
        print( "FAILED", end='\n\n', flush=True )
