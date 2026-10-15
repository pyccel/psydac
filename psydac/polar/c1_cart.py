#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from psydac.ddm.cart import CartDecomposition

__all__ = ('C1_Cart',)

#==============================================================================
class C1_Cart(CartDecomposition):
    """
    A CartDecomposition subclass whose objects are generated from an existing
    CartDecomposition object by removing all degrees of freedom corresponding
    to i_d = 0, 1 with d being the radial dimension.

    This is used to apply a C1-correction to a tensor-product finite element
    space built on a mapping with a polar singularity.

    Parameters
    ----------
    cart : psydac.ddm.CartDecomposition
        MPI decomposition of a logical grid with Cartesian topology.

    radial_dim : int
        'Radial' dimension where degrees of freedom should be removed.

    """
    def __init__(self, cart, radial_dim=0):

        assert isinstance(cart, CartDecomposition)
        assert isinstance(radial_dim, int)
        assert 0 <= radial_dim < cart.ndim
        assert cart.ndim >= 2
        assert cart.periods[radial_dim] == False

        # Initialize in standard way
        super().__init__(cart.domain_decomposition, cart.npts, cart.global_starts, cart.global_ends, cart.pads, cart.shifts)

        # Reduce start/end index (and number of points) in radial dimension by 2
        self._starts = tuple((max(0, s-2) if d==radial_dim else s) for (d, s) in enumerate(self.starts))
        self._ends   = tuple((       e-2  if d==radial_dim else e) for (d, e) in enumerate(self.ends  ))
        self._npts   = tuple((       n-2  if d==radial_dim else n) for (d, n) in enumerate(self.npts  ))

        self._global_starts = list(self._global_starts)
        self._global_ends   = list(self._global_ends)

        self._global_starts[radial_dim] -= 2
        self._global_ends  [radial_dim] -= 2

        # Make sure that we start counting from index 0
        self._global_starts[radial_dim][0] = 0

        # Recompute shape of local arrays in topology (with ghost regions)
        self._shape = tuple(e-s+1 + 2*p for s, e, p in zip(self._starts, self._ends, self._pads))
 
        # Store "parent" cart object for later reference
        self._parent_cart = cart

        # Stop here in the serial case
        if self._comm is None:
            return

        # Recompute information for communicating with neighbors
        self._shift_info = {}
        for dimension in range(self._ndims):
            for disp in [-1, 1]:
                self._shift_info[dimension, disp] = \
                        self._compute_shift_info(dimension, disp)

    # ...
    @property
    def parent_cart(self):
        return self._parent_cart
