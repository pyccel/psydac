# ------------------------------------------------------------------------- #
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
# ------------------------------------------------------------------------- #

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from sympde.topology import Square
from sympde.topology.analytical_mapping import PolarMapping

from psydac.api.tests.test_api_feec_2d import add_colorbar
from psydac.feec.pull_push import push_2d_hcurl, push_2d_l2


class TESolution(ABC):
    """
    Base class for analytical/initial solutions of the 2D TE Maxwell problem.

    The physical fields are

        E = (Ex, Ey),
        B = Bz.

    Subclasses must provide the physical field components.
    """

    @abstractmethod
    def Ex_ex(self, t, x, y):
        pass

    @abstractmethod
    def Ey_ex(self, t, x, y):
        pass

    @abstractmethod
    def Bz_ex(self, t, x, y):
        pass


class CircularCavitySolution(TESolution):
    """
    Time-harmonic solution of Maxwell's equations in a disk-like domain with
    perfectly conducting walls. This is a "transverse electric" solution, with
    E = (Ex, Ey) and B = Bz. The logical domain is [0, R] x [0, 2pi].

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

    def __init__(self, R, c, m, n, D=0, scale=1):
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
        assert 0 <= D < 0.5
        self._D = D

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

        return (
            -scale
            * c
            * n
            / (s * kc + 1e-10)
            * sin(n * theta)
            * jv(n, kc * s)
            * sin(omega * t + phase)
        )

    def Et_ex(self, t, s, theta, s_factor=True):
        """
        if s_factor: multiply by s (as in logical field)
        """
        from numpy import cos, sin
        from scipy.special import jvp

        scale = self.scale
        n = self.n
        kc = self.kc
        omega = self.omega
        phase = self.phase
        c = self.c

        val = -scale * c * cos(n * theta) * jvp(n, kc * s) * sin(omega * t + phase)
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

    def dB_dt_ex(self, t, s, theta):
        """
        = dB/dt
        """
        from numpy import cos, sin
        from scipy.special import jv

        scale = self.scale
        n = self.n
        kc = self.kc
        omega = self.omega
        phase = self.phase

        return (
            -scale * omega * s * cos(n * theta) * jv(n, kc * s) * sin(omega * t + phase)
        )

    # physical field

    def get_radius_angle(self, x, y):
        from numpy import arctan2, sqrt  # ,  sin, cos

        r = sqrt(x * x + y * y)
        alpha = arctan2(y, x)
        return r, alpha

    def Ex_ex(self, t, x, y):
        from numpy import cos, sin

        r, alpha = self.get_radius_angle(x, y)
        return cos(alpha) * self.Es_ex(t, r, alpha) - sin(alpha) * self.Et_ex(
            t, r, alpha, s_factor=False
        )

    def Ey_ex(self, t, x, y):
        from numpy import cos, sin

        r, alpha = self.get_radius_angle(x, y)
        return sin(alpha) * self.Es_ex(t, r, alpha) + cos(alpha) * self.Et_ex(
            t, r, alpha, s_factor=False
        )

    def Bz_ex(self, t, x, y):

        r, alpha = self.get_radius_angle(x, y)
        return self.B_ex(t, r, alpha, s_factor=False)


class GaussianInitialCondition(TESolution):
    """
    Initial Gaussian circular wave for the TE Maxwell test.

    This class defines the initial condition used for the Gaussian wave
    propagation experiment. It is not an exact time-dependent Maxwell solution.
    The electric field is initialized as a localized rotational Gaussian pulse,

        E0(x, y) = scale * (y - y0, -(x - x0))
                  * exp(-((x - x0)^2 + (y - y0)^2) / (2 sigma^2)),

    and the magnetic field is initialized as

        B0 = curl E0 = d_x Ey - d_y Ex.

    Parameters
    ----------
    sigma : float
        Width of the Gaussian pulse.

    x0, y0 : float
        Center of the Gaussian pulse in physical coordinates.

    scale : float, optional
        Amplitude scaling factor for the initial fields.
    """

    def __init__(self, sigma, x0, y0, scale=1):

        self.x0 = x0
        self.y0 = y0
        self.sigma = sigma
        self.scale = scale

    def _gaussian(self, x, y):
        from numpy import exp

        X = x - self.x0
        Y = y - self.y0
        sig2 = self.sigma**2
        return self.scale * exp(-(X * X + Y * Y) / (2 * sig2))

    def Ex_ex(self, t, x, y):
        Y = y - self.y0
        return Y * self._gaussian(x, y)

    def Ey_ex(self, t, x, y):
        X = x - self.x0
        return -X * self._gaussian(x, y)

    def Bz_ex(self, t, x, y):
        """
        Bz = curl E = d_x Ey - d_y Ex
        """
        X = x - self.x0
        Y = y - self.y0
        sig2 = self.sigma**2
        r2 = X * X + Y * Y
        return (r2 / sig2 - 2.0) * self._gaussian(x, y)

    def dBz_dt_ex(self, t, x, y):
        """
        ∂/∂t(Bz) = -curl(E)
        """
        X = x - self.x0
        Y = y - self.y0
        sig2 = self.sigma**2
        r2 = X * X + Y * Y
        return -(r2 / sig2 - 2.0) * self._gaussian(x, y)


# =============================================================================
# SCRIPT FUNCTIONALITY
# =============================================================================
def main():
    """
    This function is not currently used. It is kept for possible future development.
    """

    # Set time
    t = 0

    # Logical domain is rectangle [0, R] x [0, 2pi]
    R = 2.0

    # Speed of light equal c and scaling of the fields by a scale factor
    c = 1
    scale = 1

    # Mode number
    # (m, n) = (1, 0)
    # (m, n) = (2, 1)
    m, n = (2, 3)

    # Exact solution
    # TODO: allow switching between solutions using CLI arguments
#    exact_solution = CircularCavitySolution(R=R, c=c, m=m, n=n, scale=scale)
    exact_solution = GaussianInitialCondition(sigma=0.3, x0=0.2, y0=0.2, scale=scale)

    # Logical domain: [0, R] x [0, 2pi]
    logical_domain = Square("Omega", bounds1=[0, R], bounds2=[0, 2 * pi])

    # Physical domain: disk of radius R obtained as image of the logical_domain
    # with the analytical mapping of a circle
    mapping = PolarMapping("F", c1=0, c2=0, rmin=0, rmax=1)
    domain = mapping(logical_domain)
    F = mapping.get_callable_mapping()

    # Is the solution available in logical coordinates?
    log_field_names = ("Es_ex", "Et_ex", "B_ex", "dB_dt_ex")
    is_log_solution = all(hasattr(exact_solution, f) for f in log_field_names)

    # Exact logical fields at given time t
    if is_log_solution:
        Es = lambda x, y: exact_solution.Es_ex(t, x, y)
        Et = lambda x, y: exact_solution.Et_ex(t, x, y)
        B = lambda x, y: exact_solution.B_ex(t, x, y)
        dB_dt = lambda x, y: exact_solution.dB_dt_ex(t, x, y)

    # Plot of fields
    N = 50

    # 2D grids, logical (rho, theta) and physical (x, y)
    rho = np.linspace(1e-20, R, N + 1)
    theta = np.linspace(0, 2 * pi, N * 2)
    rho, theta = np.meshgrid(rho, theta, indexing="ij")
    x, y = F(rho, theta)

    # If exact solution is given in logical coordinates, use push-forward
    if is_log_solution:
        Ex_values = np.empty_like(rho)
        Ey_values = np.empty_like(rho)
        Bz_values = np.empty_like(rho)
        for i, x1i in enumerate(rho[:, 0]):
            for j, x2j in enumerate(theta[0, :]):
                Ex_values[i, j], Ey_values[i, j] = push_2d_hcurl(Es, Et, x1i, x2j, F)
                Bz_values[i, j] = push_2d_l2(B, x1i, x2j, F)
    # Otherwise, access exact solution in physical coordinates
    else:
        Ex_values = exact_solution.Ex_ex(t, x, y)
        Ey_values = exact_solution.Ey_ex(t, x, y)
        Bz_values = exact_solution.Bz_ex(t, x, y)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Analytical solution at t = {t}")
    im0 = axs[0, 0].contourf(x, y, Ex_values, 50)
    im1 = axs[0, 1].contourf(x, y, Ey_values, 50)
    im2 = axs[1, 0].contourf(x, y, np.sqrt(Ex_values**2 + Ey_values**2), 50)
    im3 = axs[1, 1].contourf(x, y, Bz_values, 50)
    axs[0, 0].set_title(r"$E_x$")
    axs[0, 1].set_title(r"$E_y$")
    axs[1, 0].set_title(r"$||\mathbf{E}||$")
    axs[1, 1].set_title("$B_z$")
    add_colorbar(im0, axs[0, 0])
    add_colorbar(im1, axs[0, 1])
    add_colorbar(im2, axs[1, 0])
    add_colorbar(im3, axs[1, 1])
    for ax in axs.flat:
        lines_const_rho = x[::5, :].T, y[::5, :].T
        lines_const_theta = x[:, ::5], y[:, ::5]
        kwargs = dict(linewidth=0.5, color="k", zorder=100)
        ax.plot(*lines_const_rho, **kwargs)
        ax.plot(*lines_const_theta, **kwargs)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x", rotation="horizontal")
        ax.set_ylabel("y", rotation="horizontal")
    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Test: curl E = - d_t B_z with finite Differences
    # on a tensor grid in the square inscribed the disk
    N = 160
    l = R / np.sqrt(2)  # edge length

    # 2D grids, logical (rho, theta) and physical (x, y)
    x, dx = np.linspace(-l, l, N, retstep=True)
    y, dy = np.linspace(-l, l, N, retstep=True)
    x, y = np.meshgrid(x, y, indexing="ij")
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) % (2 * pi)

    # If exact solution is given in logical coordinates, use push-forward
    if is_log_solution:
        Ex_values = np.empty_like(rho)
        Ey_values = np.empty_like(rho)
        Bz_values = np.empty_like(rho)
        Bt_values = np.empty_like(rho)
        ni, nj = rho.shape
        for i in range(ni):
            for j in range(nj):
                x1_ij = rho[i, j]
                x2_ij = theta[i, j]
                Ex_values[i, j], Ey_values[i, j] = push_2d_hcurl(
                    Es, Et, x1_ij, x2_ij, F
                )
                Bz_values[i, j] = push_2d_l2(B, x1_ij, x2_ij, F)
                Bt_values[i, j] = push_2d_l2(dB_dt, x1_ij, x2_ij, F)
    # Otherwise, access exact solution in physical coordinates
    else:
        Ex_values = exact_solution.Ex_ex(t, x, y)
        Ey_values = exact_solution.Ey_ex(t, x, y)
        Bz_values = exact_solution.Bz_ex(t, x, y)
        Bt_values = exact_solution.dBz_dt_ex(t, x, y)

    # Compute curl(E)
    curlE_values = np.zeros_like(rho)
    curlE_values[1:-1, 1:-1] = (Ey_values[2:, 1:-1] - Ey_values[0:-2, 1:-1]) / (
        2 * dx
    ) - (Ex_values[1:-1, 2:] - Ex_values[1:-1, 0:-2]) / (2 * dy)

    # Maximum consistency error on grid
    valerr = abs(Bt_values + curlE_values).max()
    print(f"|curl E + d_t B| <= {valerr}")

    # Data slicing for quiver plots
    skip = (slice(None, None, int(N / 20)), slice(None, None, int(N / 20)))

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Analytical solution at t = {t}: consistency checks in inscribed square"
    )

    ax = axs[0, 0]
    ax.set_title("$E_x$")
    im = ax.contourf(x, y, Ex_values)
    add_colorbar(im, ax)

    ax = axs[0, 1]
    ax.set_title("$E_y$")
    im = ax.contourf(x, y, Ey_values)
    add_colorbar(im, ax)

    ax = axs[1, 0]
    ax.set_title(r"$||\mathbf{E}||$")
    im = ax.contourf(x, y, np.sqrt(Ex_values**2 + Ey_values**2))
    add_colorbar(im, ax)

    ax = axs[1, 1]
    ax.set_title(r"$B_z$")
    im = ax.contourf(x, y, Bz_values)
    add_colorbar(im, ax)

    ax = axs[0, 2]
    ax.set_title(r"curl $\mathbf{E}$")
    im = ax.contourf(x, y, curlE_values)
    add_colorbar(im, ax)
    ax.quiver(x[skip], y[skip], Ex_values[skip], Ey_values[skip])

    ax = axs[1, 2]
    ax.set_title(r"$-\partial_t B_z$")
    im = ax.contourf(x, y, -Bt_values)
    add_colorbar(im, ax)
    ax.quiver(x[skip], y[skip], Ex_values[skip], Ey_values[skip])

    for ax in axs.flat:
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x", rotation="horizontal")
        ax.set_ylabel("y", rotation="horizontal")

    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    main()
    plt.show()
