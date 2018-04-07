# -*- coding: UTF-8 -*-

import numpy as np

from spl.core import make_open_knots
from spl.conga import SimpleCongaDDM

# import os.path
# import spl.core as core
# print ('writing path : ')
# print (os.path.abspath(core.__file__))
# exit()

import matplotlib
matplotlib.use('Agg')   # backend configuration: high quality PNG images using the Anti-Grain Geometry engine
import matplotlib.pyplot as plt


def test_conga_ddm_1d(verbose=False):
    # ...
    N_cells_sub = 48         # nb cells on each subdomain
    p = 3                    # spline degree
    m = 2                    # degree of the preserved moments
    # ...

    if verbose:
        print("building the conga-ddm object")
    cddm = SimpleCongaDDM(p, m, N_cells_sub, watch_your_steps=False, n_checks=7, use_macro_elem_duals=True)

    # check: plot a reg spline and dual:
    visual_check = False
    if visual_check:
        filename = 'dual.png'
        if verbose:
            print("check: plot a reg spline and dual in file "+filename)
        N_points = 200
        vis_grid, vis_h = np.linspace(-2, p+1+2, N_points, endpoint=True, retstep=True)
        vals = [cddm._ref_conservative_dual_function(xi) for xi in vis_grid]
        fig = plt.figure()
        plt.clf()
        plt.plot(vis_grid, vals, '-', color='b', label="dual")
        vals = [cddm._ref_b_spline(xi) for xi in vis_grid]
        plt.plot(vis_grid, vals, '-', color='k', label="b_spline")
        plt.title('spline and dual')
        plt.legend(loc='lower left')
        fig.savefig(filename)
        plt.clf()
        if verbose:
            print("done")

    f = lambda u: u*(1.-u)
    if verbose:
        print("approx some function")
    L2_proj = True
    if L2_proj:
        cddm.l2_project_on_sub_domain(f, sub='left')
        cddm.l2_project_on_sub_domain(f, sub='right')
    else:
        cddm.histopolation_on_sub_domain(f, sub='left')
        # cddm.histopolation_on_sub_domain(f, sub='right')
    if verbose:
        print("smooth projection")
    cddm.smooth_projection()
    if verbose:
        print("plot the resulting splines")
    cddm.plot_spline(filename="disco.png", spline_type="discontinuous")
    cddm.plot_spline(filename="conti.png", spline_type="continuous")
    print("done")


####################################################################################
if __name__ == '__main__':

    test_conga_ddm_1d(verbose=True)
