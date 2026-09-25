#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

class PolarModel2D:
    r"""
    Base class for analytical models on mapped 2D polar domains.

    Parameters
    ----------
    domain_log : sympde.topology.Domain
        Logical domain on which the analytical mapping is defined.

    analytical_mapping : sympde.topology.mapping.Mapping
        Analytical mapping from the logical domain to the physical domain.

    Attributes
    ----------
    logical_bounds : tuple[tuple[float, ...], ...]
        Bounds of the logical domain.

    mapping : sympde.topology.mapping.Mapping
        Mapping used by the solver. It is either the analytical mapping or
        its spline approximation. Initialized by calling `build_geometry`.

    analytical_mapping : sympde.topology.mapping.Mapping
        Original analytical mapping from the logical to the physical domain.

    domain : sympde.topology.Domain
        Physical domain associated with ``mapping``. Initialized by
        calling ``build_geometry``.

    domain_log : sympde.topology.Domain
        Logical domain.

    geometry_export_time : float
        Time spent exporting the discrete geometry, in seconds.
        It is zero when the analytical mapping is used directly.

    """

    def __init__(self, domain_log, analytical_mapping):
        from sympde.topology.mapping import Mapping

        assert isinstance(analytical_mapping, Mapping)

        self._domain_log = domain_log
        self._analytical_mapping = analytical_mapping

        self._domain = None
        self._mapping = None
        self._geometry_export_time = 0.0

    def build_geometry(
        self,
        ncells,
        degree,
        periodic,
        mpi_comm,
        use_spline_mapping,
        filename="geo.h5",
        verbose=False,
    ):
        """
        Build the physical domain and the mapping used by the solver.

        Parameters
        ----------
        ncells : sequence of int
            Number of cells in each logical coordinate direction.

        degree : sequence of int
            Polynomial degree of the spline space in each logical coordinate
            direction.

        periodic : sequence of bool
            Periodicity of the spline space in each logical coordinate direction.

        mpi_comm : mpi4py.MPI.Comm
            MPI communicator used to construct and export the discrete geometry.

        use_spline_mapping : bool
            If ``True``, approximate the analytical mapping by a spline mapping.
            If ``False``, use the analytical mapping directly.

        filename : str, default="geo.h5"
            Name of the HDF5 file used to export the spline geometry.

        verbose : bool, default=False
            If ``True``, print additional information when checking the regularity
            of the spline mapping in serial.

        """

        if not use_spline_mapping:
            # Only symbolic mapping is necessary
            self._mapping = self._analytical_mapping
            self._domain = self._analytical_mapping(self._domain_log)
            return

        from sympde.topology.domain import Domain
        from sympde.topology.mapping import Mapping

        from psydac.cad.geometry import Geometry
        from psydac.feec.polar.examples.utils_congapol import (
            create_tensor_spline_space,
        )
        from psydac.mapping.discrete import SplineMapping

        from time import time

        V = create_tensor_spline_space(
            ncells,
            degree,
            periodic,
            self.logical_bounds,
            mpi_comm,
        )

        map_analytic = self._analytical_mapping.get_callable_mapping()
        map_discrete = SplineMapping.from_mapping(V, map_analytic)

        # Create symbolic mapping with callable mapping as spline
        mapping = Mapping("M", dim=2)
        mapping.set_callable_mapping(map_discrete)
        self._mapping = mapping

        t0 = time()

        # In order to create a sympde.Domain object from this mapping we have
        # to create first a HDF5 file and then load as sympde.Domain.fromfile
        geometry = Geometry.from_discrete_mapping(
            map_discrete,
            comm=mpi_comm,
        )
        geometry.export(filename)

        self._geometry_export_time = time() - t0
        self._domain = Domain.from_file(filename)

        if mpi_comm.size == 1:
            from psydac.feec.polar.examples.utils_congapol import check_regular_ring_map
            check_regular_ring_map(map_discrete, verbose=verbose)

    @property
    def logical_bounds(self):
        return self._domain_log.bounds1, self._domain_log.bounds2

    @property
    def mapping(self):
        return self._mapping

    @property
    def analytical_mapping(self):
        return self._analytical_mapping

    @property
    def domain(self):
        return self._domain

    @property
    def domain_log(self):
        return self._domain_log

    @property
    def geometry_export_time(self):
        return self._geometry_export_time
