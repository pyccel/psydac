#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

#===============================================================================
# LOAD DATA
#===============================================================================

# Load all files starting with 'solution' and ending with '.h5'
path  = '.'
files = [f for f in os.listdir(path) \
         if os.path.isfile( os.path.join(path,f) ) \
         and f.startswith("solution_") \
         and f.endswith  (".h5")]

print( *files, sep='\n' )

# Find limits of domain
shape = [0,0]
for f in files:
    h5 = h5py.File( f, mode='r' )
    shape[0] = max( shape[0], h5['ends'][0]+1 )
    shape[1] = max( shape[1], h5['ends'][1]+1 )
    h5.close()

print( shape )

# Create global arrays
u    = np.zeros( shape )
u_ex = np.zeros( shape )

# Copy local arrays to correct global blocks
for f in files:
    h5 = h5py.File( f, mode='r' )
    index = tuple( slice( s,e+1 ) for s,e in zip( h5['starts'], h5['ends'] ) )
    u   [index] = h5['u']
    u_ex[index] = h5['u_ex']

#===============================================================================
# PLOT DATA
#===============================================================================

def colorbar( im, ax ):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = ax.figure
    divider = make_axes_locatable( ax )
    cax = divider.append_axes( "right", size="5%", pad="5%" )
    return fig.colorbar( im, cax=cax )


fig,axes = plt.subplots( 1,2, figsize=[10,4] )

im0 = axes[0].contourf( u )
cb0 = colorbar( im0, ax=axes[0] )
axes[0].set_title ( "Solution u(x1,x2)" )
axes[0].set_xlabel( "i1" )
axes[0].set_ylabel( "i2", rotation='horizontal' )

im1 = axes[1].contourf( u-u_ex )
cb1 = colorbar( im1, ax=axes[1] )
axes[1].set_title( "Error E(x1,x2)" )
axes[1].set_xlabel( "i1" )
axes[1].set_ylabel( "i2", rotation='horizontal' )

for ax in axes:
    ax.set_aspect( 'equal' )

fig.tight_layout()
plt.show()


