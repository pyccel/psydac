# coding: utf-8

import numpy as np
from itertools import product

from .cart import CartDecomposition

#===============================================================================
class PetscCart:

    def __init__(self, cart):
        assert isinstance(cart, CartDecomposition)

        try:
            from petsc4py import PETSc
        except ImportError:
            raise 

        self._cart  = cart
        self._petsc = PETSc

        self._indices          = self.create_indices()
        self._extended_indices = self.create_extended_indices()
        self._ao               = self.create_Ao()
        self._l2g_mapping      = self.create_LGMap()

        # Compute local shape of local arrays in topology (without ghost regions)
        self._local_shape = tuple( e-s+1 for s,e in zip( cart._starts, cart._ends ) )

        # Compute local size of local arrays in topology (without ghost regions)
        self._local_size  = np.product(self._local_shape)
 

    @property
    def cart( self ):
        return self._cart

    @property
    def petsc( self ):
        return self._petsc

    @property
    def indices( self ):
        return self._indices

    @property
    def extended_indices( self ):
        return self._extended_indices

    @property
    def ao( self ):
        return self._ao

    @property
    def local_size( self ):
        return self._local_size

    @property
    def local_shape( self ):
        return self._local_shape

    @property
    def l2g_mapping( self ):
        return self._l2g_mapping
        
    def create_indices( self ):
        cart    = self.cart
        indices = product(*cart._grids)
        npts    = cart.npts
        array   = [np.ravel_multi_index(i, npts) for i in indices]
        return array

    def create_extended_indices( self ):
        cart    = self.cart
        indices = product(*cart._extended_grids)
        npts    = cart.npts
        mode    = tuple('wrap' if P else 'clip' for P in cart.periods)
        array   = [np.ravel_multi_index(i, npts, mode=mode) for i in indices]
        return array

    def create_Ao( self ):
        cart    = self.cart
        indices = self.indices
        return self.petsc.AO().createBasic(indices, comm=cart.comm)

    def create_LGMap( self ):
        cart    = self.cart
        indices = self.extended_indices
        ao      = self.ao
        return self.petsc.LGMap().create(ao.app2petsc(indices), comm=cart.comm)

    def create_g2n(self, gvec, natural):
        cart     = self.cart
        indices  = self.indices
        size     = self.local_size
        start,_  = natural.getOwnershipRange()

        from_is = self.petsc.IS().createStride(size, start, 1, comm=cart.comm)
        to_is   = self.petsc.IS().createGeneral(indices, comm=cart.comm)

        return self.petsc.Scatter().create(gvec, from_is, natural, to_is)

