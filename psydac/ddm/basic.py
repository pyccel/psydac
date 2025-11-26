#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from abc import ABC, abstractmethod


__all__ = ('CartDataExchanger',)
#===============================================================================
class CartDataExchanger(ABC):
    """
    Type that takes care of updating the ghost regions (padding) of a
    multi-dimensional array distributed according to the given Cartesian
    decomposition of a tensor-product grid of coefficients.

    Each coefficient in the decomposed grid may have multiple components,
    contiguous in memory.

    Parameters
    ----------
    cart : psydac.ddm.CartDecomposition
        Object that contains all information about the Cartesian decomposition
        of a tensor-product grid of coefficients.

    dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
        Datatype of single coefficient (if scalar) or of each of its
        components (if vector).

    coeff_shape : [tuple(int) | list(int)]
        Shape of a single coefficient, if this is multi-dimensional
        (optional: by default, we assume scalar coefficients).

    """

    #---------------------------------------------------------------------------
    # Public interface
    #---------------------------------------------------------------------------

    @abstractmethod
    def prepare_communications(self, u):
        pass

    @abstractmethod
    def start_update_ghost_regions( self, array, requests ):
        """
        Update ghost regions in a numpy array with dimensions compatible with
        CartDecomposition (and coeff_shape) provided at initialization.

        Parameters
        ----------
        array : numpy.ndarray
            Multidimensional array corresponding to local subdomain in
            decomposed tensor grid, including padding.

        requests : tuple|None
            The requests of the communications.

        """

    @abstractmethod
    def end_update_ghost_regions( self, array, requests ):
        pass

    @abstractmethod
    def start_exchange_assembly_data( self, array ):
        """
        Update ghost regions after the assembly algorithm in a numpy array
        with dimensions compatible with CartDecomposition (and coeff_shape)
        provided at initialization.

        Parameters
        ----------
        array : numpy.ndarray
            Multidimensional array corresponding to local subdomain in
            decomposed tensor grid, including padding.
        """

    @abstractmethod
    def end_exchange_assembly_data( self, array ):
        pass

