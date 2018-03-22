import numpy as np
from itertools import product
from mpi4py    import MPI

from cart import Cart

#===============================================================================
# INPUT PARAMETERS
#===============================================================================

# Number of elements
n1 = 135
n2 = 77

# Padding ('thickness' of ghost region)
p1 = 3
p2 = 2

# Periodicity
period1 = True
period2 = False

#===============================================================================
# DOMAIN DECOMPOSITION
#===============================================================================

# Parallel info
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Decomposition of Cartesian domain
cart = Cart(
    npts    = [n1+1,n2+1],
    pads    = [p1,p2],
    periods = [period1, period2],
    reorder = False,
    comm    = comm,
)

# Local 2D array with 2D vector data (extended domain)
shape = list( cart.shape ) + [2]
u = np.zeros( shape, dtype=int ) # NOTE: 64-bit INTEGER!

# Global indices of first and last elements of array
s1,s2 = cart.starts
e1,e2 = cart.ends

# Create MPI subarray datatypes for accessing non-contiguous data
send_types = {}
recv_types = {}
for shift in product( [-1,0,1], repeat=2 ):
    if shift == (0,0):
        continue
    info = cart.get_sendrecv_info( shift )

    buf_shape   = list( info[ 'buf_shape' ] ) + [2]
    send_starts = list( info['send_starts'] ) + [0]
    recv_starts = list( info['recv_starts'] ) + [0]

    send_types[shift] = MPI.INT64_T.Create_subarray(
        sizes    = u.shape,
        subsizes = buf_shape,
        starts   = send_starts,
    ).Commit()

    recv_types[shift] = MPI.INT64_T.Create_subarray(
        sizes    = u.shape,
        subsizes = buf_shape,
        starts   = recv_starts,
    ).Commit()

# Print some info
if rank == 0:
    print( "" )

for k in range(size):
    if k == rank:
        print( "Proc. # {}".format( rank ) )
        print( "---------" )
        print( ". s1:e1 = {:2d}:{:2d}".format( s1,e1 ) )
        print( ". s2:e2 = {:2d}:{:2d}".format( s2,e2 ) )
        print( "", flush=True )
    comm.Barrier()

#===============================================================================
# TEST
#===============================================================================

# Fill in true domain with u[i1_loc,i2_loc,:]=[i1_glob,i2_glob]
u[p1:-p1,p2:-p2,:] = [[(i1,i2) for i2 in range(s2,e2+1)] for i1 in range(s1,e1+1)]

status = MPI.Status()

# Exchange ghost cell information
for shift in product( [-1,0,1], repeat=2 ):
    if shift == (0,0):
        continue

    # Get communication info for given shift
    info = cart.get_sendrecv_info( shift )

    # Send and receive data
    cart.comm_cart.Sendrecv(
        sendbuf = [u, 1, send_types[shift]],
        dest    = info['rank_dest'],
        recvbuf = [u, 1, recv_types[shift]],
        source  = info['rank_source'],
        status  = status,
    )

#===============================================================================
# CHECK RESULTS
#===============================================================================

# Verify that ghost cells contain correct data (note periodic domain!)
val = lambda i1,i2: (i1%(n1+1),i2) if 0<=i2<=n2 else (0,0)
uex = [[val(i1,i2) for i2 in range(s2-p2,e2+p2+1)] for i1 in range(s1-p1,e1+p1+1)]

success = (u == uex).all()

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
