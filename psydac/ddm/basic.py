#===============================================================================
class CartDataExchanger:
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
    def get_assembly_send_type( self, *args ):
        pass

    # ...
    @abstractmethod
    def get_assembly_recv_type( self, *args ):
        pass

    @abstractmethod
    def prepare_communications(self, u):
        pass

    @abstractmethod
    def start_update_ghost_regions(self, **kwargs):
        pass

    @abstractmethod
    def end_update_ghost_regions(self, **kwargs):
        pass

    @abstractmethod
    def start_exchange_assembly_data( self, array ):
        pass

    @abstractmethod
    def end_exchange_assembly_data( self, array, requests ):
        pass

    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------
    @staticmethod
    @abstractmethods
    def _create_buffer_types( *args, **kwargs ):
        pass

    # ...
    @staticmethod
    @abstractmethod
    def _create_assembly_buffer_types( *args, **kwargs )
        pass
