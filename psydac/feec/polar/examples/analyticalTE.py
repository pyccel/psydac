from sympde.topology import Square
from sympde.topology.analytical_mapping import PolarMapping
from psydac.feec.pull_push import push_2d_hcurl, push_2d_l2
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt


def add_colorbar(im, ax, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.3)
    cbar = ax.get_figure().colorbar(im, cax=cax, **kwargs)
    return cbar


class CircularCavitySolution:
    """
    Time-harmonic solution of Maxwell's equations in a disk-like domain with
    perfectly conducting walls. This is a "transverse electric" solution, with
    E = (Ex, Ey) and B = Bz. Domain is [0, R] x [0, 2pi].

    Parameters
    ----------
    R : float
        domain radius

    c: float
        Speed of light in arbitrary units

    (m, n): int
        Mode number. Warning: m > 0, n >= 0

    D: float
        shift of logical center (in "Target" mapping with c0=D*R2, c1=0, k=0, D=D)

    scale: float
        Rescaling the values by a real factor. Default = 1

    """

    def __init__(self, R, c, m, n, D=0, scale=1, variables='log'):
        from numpy import pi
        from scipy.special import jnp_zeros
        pnm = jnp_zeros(n, m)[-1]
        kc = pnm / R
        omega = c * kc

        phase = pi / 4

        self.c = c
        self.scale = scale
        self.n = n
        self.kc = kc
        self.omega = omega
        self.phase = phase
        self._R = R
        assert 0 <= D < .5
        self._D = D
        # assert variables in ['log', 'phys']
        # self._logical = (variables == 'log')

    # Exact solutions for electric and magnetic field with polar parametrization of disk domain
    def Es_ex(self, t, s, theta):
        from numpy import sin  # , cos, sqrt, arctan2
        from scipy.special import jv

        scale = self.scale
        n = self.n
        kc = self.kc
        omega = self.omega
        phase = self.phase
        c = self.c

        return - scale * c * n / (s * kc + 1e-10) * sin(n * theta) * jv(n, kc * s) * sin(omega * t + phase)

    def Et_ex(self, t, s, theta, s_factor=True):
        """
        if s_factor: multiply by s (as in logical field)
        """
        from numpy import sin, cos
        from scipy.special import jvp

        scale = self.scale
        n = self.n
        kc = self.kc
        omega = self.omega
        phase = self.phase
        c = self.c

        val = - scale * c * cos(n * theta) * jvp(n, kc * s) * sin(omega * t + phase)
        if s_factor:
            val *= s
        return val

    def B_ex(self, t, s, theta, s_factor=True):
        """
        if s_factor: multiply by s (as in logical field)
        """
        from numpy import cos
        from scipy.special import jv

        scale = self.scale
        n = self.n
        kc = self.kc
        omega = self.omega
        phase = self.phase

        val = scale * cos(n * theta) * jv(n, kc * s) * cos(omega * t + phase)
        if s_factor:
            val *= s
        return val
        # The magnitude of B is approximately equal to scale / 3

    def Bt_ex(self, t, s, theta):
        '''
        = dB/dt
        todo: change the name
        '''
        from numpy import cos, sin
        from scipy.special import jv

        scale = self.scale
        n = self.n
        kc = self.kc
        omega = self.omega
        phase = self.phase

        return scale * omega * s * cos(n * theta) * jv(n, kc * s) * sin(omega * t + phase)

    # physical field

    def get_radius_angle(self, x, y):
        from numpy import sqrt, arctan2  # ,  sin, cos
        r = sqrt(x * x + y * y)
        alpha = arctan2(y, x)
        # print(f'r*np.cos(alpha) - x = {r*np.cos(alpha) - x}')
        # print(f'r*np.sin(alpha) - y = {r*np.sin(alpha) - y}')
        return r, alpha

    def Ex_ex(self, t, x, y):
        from numpy import cos, sin

        r, alpha = self.get_radius_angle(x, y)
        return cos(alpha) * self.Es_ex(t, r, alpha) - sin(alpha) * self.Et_ex(t, r, alpha, s_factor=False)

    def Ey_ex(self, t, x, y):
        from numpy import cos, sin

        r, alpha = self.get_radius_angle(x, y)
        return sin(alpha) * self.Es_ex(t, r, alpha) + cos(alpha) * self.Et_ex(t, r, alpha, s_factor=False)

    def Bz_ex(self, t, x, y):

        r, alpha = self.get_radius_angle(x, y)
        return self.B_ex(t, r, alpha, s_factor=False)


def main():
    # TODO: update with D_shift

    # Physical domain is rectangle [0, R] x [0, 2pi]
    R = 2.0

    # Speed of light equal c and scaling of the fields by a scale factor
    c = 1
    scale = 1

    # Mode number
    # (m, n) = (1, 0)
    # (m, n) = (2, 1)
    (m, n) = (2, 3)

    # Exact solution
    exact_solution = CircularCavitySolution(R=R, c=c, m=m, n=n, scale=scale)

    # Exact fields, as callable functions of (t, s, theta)
    Es_ex = exact_solution.Es_ex
    Et_ex = exact_solution.Et_ex
    B_ex = exact_solution.B_ex
    Bt_ex = exact_solution.Bt_ex

    # Logical domain: [0, R] x [0, 2pi]
    logical_domain = Square('Omega', bounds1=[0, R], bounds2=[0, 2 * pi])

    # Physical domain: disk of radius R obtained as image of the logical_domain
    # with the analytical mapping of a circle
    mapping = PolarMapping('F', c1=0, c2=0, rmin=0, rmax=1)
    domain = mapping(logical_domain)
    F = mapping.get_callable_mapping()

    # Set time
    t = 0

    Es = lambda x, y: Es_ex(t, x, y)
    Et = lambda x, y: Et_ex(t, x, y)
    B = lambda x, y: B_ex(t, x, y)
    Bt = lambda x, y: Bt_ex(t, x, y)

    # Plot of fields
    N = 100

    rho = np.linspace(1e-20, R, N)
    theta = np.linspace(0, 2 * pi, N)
    rho, theta = np.meshgrid(rho, theta, indexing='ij')
    x, y = F(rho, theta)

    Ex_values = np.empty_like(rho)
    Ey_values = np.empty_like(rho)
    B_values = np.empty_like(rho)

    valerr = 0
    for i, x1i in enumerate(rho[:, 0]):
        for j, x2j in enumerate(theta[0, :]):
            Ex_values[i, j], Ey_values[i, j] = \
                push_2d_hcurl(Es, Et, x1i, x2j, F)

            B_values[i, j] = push_2d_l2(B, x1i, x2j, F)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    im0 = axs[0, 0].contourf(x, y, Ex_values, 50)
    im1 = axs[0, 1].contourf(x, y, Ey_values, 50)
    im2 = axs[1, 0].contourf(x, y, np.sqrt(Ex_values ** 2 + Ey_values ** 2), 50)
    im3 = axs[1, 1].contourf(x, y, B_values, 50)
    axs[0, 0].set_title(r'$E_x$')
    axs[0, 1].set_title(r'$E_y$')
    axs[1, 0].set_title(r'$||\mathbf{E}||$')
    axs[1, 1].set_title('$B$')
    add_colorbar(im0, axs[0, 0])
    add_colorbar(im1, axs[0, 1])
    add_colorbar(im2, axs[1, 0])
    add_colorbar(im3, axs[1, 1])

    # Test: curl E = - d_t B_z with finite Differences
    # on a tensor grid in the square inscribed the disk
    N = 160
    l = R / np.sqrt(2)  # edge length

    x, dx = np.linspace(-l, l, N, retstep=True)
    y, dy = np.linspace(-l, l, N, retstep=True)
    x, y = np.meshgrid(x, y, indexing='ij')
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x) % (2 * pi)

    Ex_values = np.empty_like(rho)
    Ey_values = np.empty_like(rho)
    Bt_values = np.empty_like(rho)

    ni, nj = rho.shape
    for i in range(ni):
        for j in range(nj):
            x1_ij = rho[i, j]
            x2_ij = theta[i, j]
            Ex_values[i, j], Ey_values[i, j] = \
                push_2d_hcurl(Es, Et, x1_ij, x2_ij, F)

            Bt_values[i, j] = push_2d_l2(Bt, x1_ij, x2_ij, F)

    fig, axs = plt.subplots(2, 3, figsize=(15, 15))
    im1 = axs[0, 0].contourf(x, y, np.sqrt(Ex_values ** 2 + Ey_values ** 2))
    im2 = axs[0, 1].contourf(x, y, -Bt_values)
    axs[0, 0].set_title(r'$||\mathbf{E}||$')
    axs[0, 1].set_title(r'$-\partial_t B$')
    for axi in axs.flat:
        axi.set_aspect('equal')
    add_colorbar(im1, axs[0, 0])
    add_colorbar(im2, axs[0, 1])

    curlE_values = np.zeros_like(rho)
    valerr = 0
    for i in range(1, ni - 1):
        for j in range(1, nj - 1):
            curlE_values[i, j] = (Ey_values[i + 1, j] - Ey_values[i - 1, j]) / (2 * dx) \
                                 - (Ex_values[i, j + 1] - Ex_values[i, j - 1]) / (2 * dy)

            d = np.abs(Bt_values[i, j] + curlE_values[i, j])

            if valerr < d:
                valerr = d

    skip = (slice(None, None, int(N / 20)), slice(None, None, int(N / 20)))
    im3 = axs[1, 0].contourf(x, y, curlE_values)
    axs[1, 0].quiver(x[skip], y[skip], Ex_values[skip], Ey_values[skip])
    add_colorbar(im3, axs[1, 0])
    axs[1, 0].set_title(r'curl $\mathbf{E}$')
    im4 = axs[1, 1].contourf(x, y, -Bt_values)
    axs[1, 1].quiver(x[skip], y[skip], Ex_values[skip], Ey_values[skip])
    add_colorbar(im4, axs[1, 1])
    im5 = axs[0, 2].contourf(x, y, Ex_values)
    im6 = axs[1, 2].contourf(x, y, Ey_values)
    add_colorbar(im5, axs[0, 2])
    add_colorbar(im6, axs[1, 2])
    print('|curlE + d_t B| <= ', valerr)


if __name__ == "__main__":
    main()
    plt.show()


