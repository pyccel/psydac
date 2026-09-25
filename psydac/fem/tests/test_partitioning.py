#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os

from sympde.topology import Domain, Interface, Square, IdentityMapping, PolarMapping

from psydac.fem.partitioning import (construct_connectivity,
                                      construct_join_connectivity,
                                      connectivity_to_join_tuples)

base_dir  = os.path.dirname(os.path.realpath(__file__))
mesh_dir  = os.path.join(base_dir, '..', '..', 'cad', 'mesh', 'multipatch')


def _legacy_construct_connectivity(domain):
    # Verbatim pre-D9 body of `construct_connectivity`
    # (psydac/fem/partitioning.py:96-110 at f6a1912c). Used as the A/B
    # reference for test_construct_connectivity_matches_legacy_implementation.
    interfaces = domain.interfaces if domain.interfaces else []
    if len(domain)==1:
        interiors  = [domain.interior]
    else:
        interiors  = list(domain.interior.args)
        if interfaces:
            interfaces = [interfaces] if isinstance(interfaces, Interface) else list(interfaces.args)

    connectivity = {}
    for e in interfaces:
        i = interiors.index(e.minus.domain)
        j = interiors.index(e.plus.domain)
        connectivity[i, j] = ((e.minus.axis, e.minus.ext),(e.plus.axis, e.plus.ext))

    return connectivity


def _two_patch_domain(ornt, *, return_patches=False):
    A = Square('A', bounds1=(0., 1.), bounds2=(0., 1.))
    B = Square('B', bounds1=(0., 1.), bounds2=(0., 1.))
    D1 = IdentityMapping('M1', 2)(A)
    D2 = IdentityMapping('M2', 2)(B)
    connectivity = [((0, 1, 1), (1, 1, -1), ornt)]
    domain = Domain.join([D1, D2], connectivity, 'two_patch')
    if return_patches:
        return domain, [D1, D2]
    return domain


def _four_patch_domain():
    # Mirrors psydac/api/tests/test_2d_multipatch_mapping_poisson.py::
    # test_poisson_2d_4_patch_dirichlet_0 (itself in the spirit of
    # psydac/api/tests/build_domain.py), stripped down to just the domain.
    import numpy as np
    A = Square('A', bounds1=(0.2, 0.6), bounds2=(0, np.pi))
    B = Square('B', bounds1=(0.2, 0.6), bounds2=(np.pi, 2*np.pi))
    C = Square('C', bounds1=(0.6, 1.), bounds2=(0, np.pi))
    D = Square('D', bounds1=(0.6, 1.), bounds2=(np.pi, 2*np.pi))

    D1 = PolarMapping('M1', 2, c1=0., c2=0., rmin=0., rmax=1.)(A)
    D2 = PolarMapping('M2', 2, c1=0., c2=0., rmin=0., rmax=1.)(B)
    D3 = PolarMapping('M3', 2, c1=0., c2=0., rmin=0., rmax=1.)(C)
    D4 = PolarMapping('M4', 2, c1=0., c2=0., rmin=0., rmax=1.)(D)

    patches = [D1, D2, D3, D4]
    connectivity = [((0, 1, 1), (1, 1,-1), 1),
                    ((2, 1, 1), (3, 1,-1), 1),
                    ((0, 0, 1), (2, 0,-1), 1),
                    ((1, 0, 1), (3, 0,-1), 1)]
    return Domain.join(patches, connectivity, 'four_patch')


#==============================================================================
def test_construct_connectivity_matches_legacy_implementation():
    # R1: pin construct_connectivity's dict *and* key order against a
    # verbatim copy of the pre-D9 implementation -- `==` on dicts ignores
    # order, so both are checked explicitly.
    domains = [
        Domain.from_file(os.path.join(mesh_dir, 'square.h5')),
        Domain.from_file(os.path.join(mesh_dir, 'magnet.h5')),
        _four_patch_domain(),
    ]
    for domain in domains:
        expected = _legacy_construct_connectivity(domain)
        actual   = construct_connectivity(domain)
        assert actual == expected
        assert list(actual.keys()) == list(expected.keys())

    # Cross-check against the doc's measured baseline values.
    square = Domain.from_file(os.path.join(mesh_dir, 'square.h5'))
    assert construct_connectivity(square) == {(0, 1): ((0, 1), (0, -1))}

    magnet = Domain.from_file(os.path.join(mesh_dir, 'magnet.h5'))
    assert construct_connectivity(magnet) == {(0, 1): ((1, 1), (1, -1)),
                                               (1, 2): ((1, 1), (1, -1)),
                                               (2, 3): ((1, 1), (1, 1))}


#==============================================================================
def test_join_connectivity_keeps_orientation():
    domain = _two_patch_domain(ornt=-1)

    join_connectivity = construct_join_connectivity(domain)
    assert len(join_connectivity) == 1
    assert join_connectivity[0][2] == -1

    # construct_connectivity is the lossy projection: no orientation anywhere
    # in its output.
    connectivity = construct_connectivity(domain)
    for value in connectivity.values():
        assert len(value) == 2
        for leg in value:
            assert len(leg) == 2  # (axis, ext), no third (orientation) slot


#==============================================================================
def test_join_connectivity_round_trips_through_domain_join():
    for ornt in (1, -1):
        domain, patches = _two_patch_domain(ornt=ornt, return_patches=True)

        rebuilt = Domain.join(patches, construct_join_connectivity(domain), domain.name)

        interfaces_before = domain.interfaces
        interfaces_after  = rebuilt.interfaces
        if isinstance(interfaces_before, Interface):
            interfaces_before = [interfaces_before]
        if isinstance(interfaces_after, Interface):
            interfaces_after = [interfaces_after]

        before = [(e.minus.axis, e.minus.ext, e.plus.axis, e.plus.ext, e.ornt) for e in interfaces_before]
        after  = [(e.minus.axis, e.minus.ext, e.plus.axis, e.plus.ext, e.ornt) for e in interfaces_after]
        assert after == before


#==============================================================================
def test_connectivity_to_join_tuples_defaults_to_ornt_1():
    connectivity = {(0, 1): ((0, 1), (0, -1))}

    default = connectivity_to_join_tuples(connectivity)
    assert default == [((0, 0, 1), (1, 0, -1), 1)]

    negative = connectivity_to_join_tuples(connectivity, ornt=-1)
    assert negative == [((0, 0, 1), (1, 0, -1), -1)]


#==============================================================================
def test_construct_connectivity_single_patch_is_empty():
    domain = Square('Omega', bounds1=(0., 1.), bounds2=(0., 1.))
    assert construct_connectivity(domain) == {}
    assert construct_join_connectivity(domain) == []
