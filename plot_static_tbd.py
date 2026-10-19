# Test of the static plots (plot_2d) with one snapshot of psydac/fem/generate_plot_data_13Okt/field_info.h5.
# Saves one PNG per case in zz_static_plots/ (nothing is shown). Run from anywhere: python plot_static_tbd.py
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# This file sits next to the psydac/ source tree. Drop that directory so
# `import psydac` is the package installed in the environment.
sys.path[:] = [
    p for p in sys.path
    if (Path.cwd() if p == "" else Path(p)).resolve() != ROOT
]

import h5py
from sympde.topology import Square
from psydac.api.postprocessing import PostProcessManager


def _load_local(name):
    path = ROOT / "psydac" / "fem" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pu = _load_local("plotting_utilities2")
get_mapping = _load_local("anisotropic_maxwell_to_be_deleted").get_mapping

domain = get_mapping("WigglyHalfAnnulus", R0=2, R1=4)(Square("S", bounds1=(0, 1), bounds2=(0, 1)))

data = ROOT / "psydac" / "fem" / "generate_plot_data_13Okt" / "field_info.h5"
out  = ROOT / "zz_static_plots"
out.mkdir(exist_ok=True)
snapshot = 50

# --- load one snapshot into FemFields E (vector-valued, hcurl) and B (scalar, l2)
with h5py.File(data, "r") as f:
    ppm = PostProcessManager(domain=domain, space_file=data.parent / f.attrs["spaces"], fields_file=str(data))
    ppm.load_snapshot(snapshot, "E", "B")
    E, B = ppm.fields["E"], ppm.fields["B"]
    t = f[f"snapshot_{snapshot:04d}"].attrs["t"]
    ppm.close()

# --- grid values (instead of FemFields), to test that path as well
mappings = E.space.symbolic_space.domain.mappings
etas, xx, yy = pu.get_plotting_grid(mappings, N=(50, 200))
E_vals = pu.get_grid_vals(E, etas, list(mappings.values()))     # [x-component, y-component], one entry per patch
B_vals = pu.get_grid_vals(B, etas, list(mappings.values()))





#* auto-titles
# this should automatically complete the titles
# it should also never plot the absolute value of anything
# it should also not plot B twice (components and magnitude)
pu.plot_2d((E, B), titles=("E", "B"), components=True, magnitude=True, show_plot=True)

# this should plot the normal basic names, not magnitudified names
pu.plot_2d((E, B), titles=("E", "B"), magnitude=True, show_plot=True)

# this should just not fail
pu.plot_2d((E, B), titles=("E1", "E2", "E3", "B"), components=True, magnitude=True, show_plot=True)
