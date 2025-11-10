import numpy as np


def print_map_polar_coeffs(map_discrete):
    # Print spline mapping
    print('Spline mapping:')
    # print(map_discrete)
    print('vars(map_discrete):')
    print(vars(map_discrete))
    print()
    # print(vars(map_discrete._control_points))
    # print(vars(map_discrete._control_points._mapping))
    # print(vars(map_discrete._control_points._mapping._control_points))
    # print(map_discrete._control_points._mapping._fields)
    print('vars(map_discrete._fields[0]) :')
    print(vars(map_discrete._fields[0]))

    print()
    print('vars(map_discrete._fields[0]._space._spaces[0]) :')
    # print(vars(map_discrete._fields[0]._space))
    print(vars(map_discrete._fields[0]._space._spaces[0]))
    print('vars(map_discrete._fields[0]._space._spaces[1]) :')
    print(vars(map_discrete._fields[0]._space._spaces[1]))
    n_s = map_discrete._fields[0]._space._spaces[0]._nbasis
    n_theta = map_discrete._fields[0]._space._spaces[1]._nbasis
    assert n_s == map_discrete._fields[1]._space._spaces[0]._nbasis
    assert n_theta == map_discrete._fields[1]._space._spaces[1]._nbasis
    # deg_s = degree[0]
    # print('ncells_s = ', n_s-deg_s, ' = ', ncells[0])
    print('n_s = ', n_s)
    print('n_theta = ', n_theta)
    print()

    # print('map_discrete._fields[0]._coeffs :')
    # print(map_discrete._fields[0]._coeffs)
    # print()

    map_0_c = map_discrete._fields[0]._coeffs.toarray()
    map_1_c = map_discrete._fields[1]._coeffs.toarray()

    print('pole: ', map_0_c[0], map_1_c[0])
    map_0_c -= map_0_c[0]
    map_1_c -= map_1_c[0]

    # print(map_0_c**2 + map_1_c**2)
    radius_pole = np.sqrt(map_0_c[:n_theta] ** 2 + map_1_c[:n_theta] ** 2)
    radius_first_ring = np.sqrt(
        map_0_c[n_theta:2 * n_theta] ** 2 + map_1_c[n_theta:2 * n_theta] ** 2
    )
    rho_1 = radius_first_ring[0]

    cs = map_0_c[n_theta:2 * n_theta] / rho_1
    sn = map_1_c[n_theta:2 * n_theta] / rho_1
    theta = np.arctan2(sn, cs)
    # print('cos theta_j ?', map_0_c[4:8]/rho_1)
    # print('sin theta_j ?', map_1_c[4:8]/rho_1)

    # print('theta_j:', theta)
    print('radius_pole:', radius_pole)
    print('radius first ring:', radius_first_ring)
    print('D theta_j:', np.mod(theta[1:] - theta[:-1], 2 * np.pi))
    # print('theta_j ?', np.arcsin(map_1_c[4:8]/rho_1))
    # exit()

    # angle2 =
    # print(vars(map_discrete._control_points._mapping._fields[0]._coeffs))
    # exit()


def check_regular_ring_map(map_discrete, verbose=False):
    n_s = map_discrete._fields[0]._space._spaces[0]._nbasis
    n_theta = map_discrete._fields[0]._space._spaces[1]._nbasis
    assert n_s == map_discrete._fields[1]._space._spaces[0]._nbasis
    assert n_theta == map_discrete._fields[1]._space._spaces[1]._nbasis
    # deg_s = degree[0]
    # print('ncells_s = ', n_s-deg_s, ' = ', ncells[0])
    if verbose:
        print('n_s = ', n_s)
        print('n_theta = ', n_theta)
        print()

    map_0_c = map_discrete._fields[0]._coeffs.toarray()
    map_1_c = map_discrete._fields[1]._coeffs.toarray()

    # print('pole: ', map_0_c[0], map_1_c[0])
    map_0_c -= map_0_c[0]
    map_1_c -= map_1_c[0]

    # print(map_0_c**2 + map_1_c**2)
    radius_pole = np.sqrt(map_0_c[:n_theta] ** 2 + map_1_c[:n_theta] ** 2)
    radius_first_ring = np.sqrt(
        map_0_c[n_theta:2 * n_theta] ** 2 + map_1_c[n_theta:2 * n_theta] ** 2
    )
    rho_1 = radius_first_ring[0]

    cs = map_0_c[n_theta:2 * n_theta] / rho_1
    sn = map_1_c[n_theta:2 * n_theta] / rho_1
    theta = np.arctan2(sn, cs)
    # print('cos theta_j ?', map_0_c[4:8]/rho_1)
    # print('sin theta_j ?', map_1_c[4:8]/rho_1)

    delta_theta = np.mod(theta[1:] - theta[:-1], 2 * np.pi)
    if verbose:
        print('radius_pole:', radius_pole)
        print('radius first ring:', radius_first_ring)
        print('D theta_j:', delta_theta)
        print()

    circle_error = np.linalg.norm(radius_pole[1:] - radius_pole[:-1])
    regularity_error = np.linalg.norm(delta_theta[1:] - delta_theta[:-1])
    regularity_check = abs(circle_error) + abs(regularity_error) < 1e-12

    print('CHECK: spline mapping is regular: ', regularity_check)
    if not regularity_check:
        print(' - circle error on 1st ring:', circle_error)
        print(' - regularity error on 1st ring:', regularity_error)
    print()
