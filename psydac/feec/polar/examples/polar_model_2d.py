#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

class PolarModel2D:
    """Base class for analytical models on mapped 2D polar domains."""

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
        """Build the physical domain and the mapping used by the solver."""

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
