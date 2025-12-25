#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Post-processing script for example `poisson_2d_mapping.py`.

Plot the mass and stiffness matrices built in the tensor-product spline space V
and in the C1 polar spline space V'.

USAGE
=====

$ ipython

In [1]: run poisson_2d_mapping.py -t target -n 10 20 -s -c

In [2]: run -i visualize_matrices.py

"""

from matplotlib import colors

globals().update(namespace)

#===============================================================================

#----------------
# Plot M
mat = M.toarray()
#----------------
fig,ax = plt.subplots(1,1)
ax.set_title("Tensor-product mass matrix M")
im = ax.matshow(mat, norm=colors.LogNorm(), cmap='hot_r')
cb = fig.colorbar(im, ax=ax)
fig.show()

#----------------
# Plot M'
mat = Mp.toarray()
#----------------
fig,ax = plt.subplots(1,1)
ax.set_title("C^1 mass matrix M' (projection of M)")
im = ax.matshow(mat, norm=colors.LogNorm(), cmap='hot_r')
cb = fig.colorbar(im, ax=ax)
fig.show()

#----------------
# Plot S
mat = S.toarray()
#----------------
fig,ax = plt.subplots(1,1)
ax.set_title("Tensor-product stiffness matrix S")
norm = colors.SymLogNorm(linthresh=1.0e-8, vmin=mat.min(), vmax=mat.max())
im = ax.matshow(mat, norm=norm, cmap='seismic')
cb = fig.colorbar(im, ax=ax)
fig.show()

#----------------
# Plot S'
mat = Sp.toarray()
#----------------
fig,ax = plt.subplots(1,1)
ax.set_title("C^1 stiffness matrix S' (projection of S)")
norm = colors.SymLogNorm(linthresh=1.0e-6, vmin=-mat.max(), vmax=mat.max())
im = ax.matshow(mat, norm=norm, cmap='seismic')
cb = fig.colorbar(im, ax=ax)
fig.show()
