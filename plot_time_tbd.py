# Save the snapshots in psydac/fem/generate_plot_data_13Okt/field_info.h5 as a GIF.
# Run from anywhere: python plot_time_tbd.py
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


def _load_local(name):
    path = ROOT / "psydac" / "fem" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_2d_time = _load_local("plotting_utilities2").plot_2d_time
get_mapping = _load_local("anisotropic_maxwell_to_be_deleted").get_mapping







domain = get_mapping("WigglyHalfAnnulus", R0=2, R1=4)(Square("S", bounds1=(0, 1), bounds2=(0, 1)))

filename = "zzanimation.gif"
data = ROOT / "psydac" / "fem" / "generate_plot_data_13Okt" / "field_info.h5"

if input("normal: ") == "y":
    with h5py.File(data, "r") as f:
        saved = plot_2d_time(
            f,
            filename,
            domain=domain,
            fields=({"fem_field": "E", "magnitude": True, "components": True}),
            titles=(r"$E$", r"$B$"),
            fps=50,
            N_vis=(100, 100)
        )
    print(f"Saved {saved}")



# The same three cases as in plot_static_tbd.py (default N_vis=(20, 20), so they are quick)
#* auto-titles
if input("check auto titles: ") == "y":
    with h5py.File(data, "r") as f:
        # this should automatically complete the titles
        # it should also never plot the absolute value of anything
        # it should also not plot B twice (components and magnitude)
        print("Saved", plot_2d_time(f, "zz_time_1.gif", domain=domain, fields=("E", "B"),
                                    titles=("E", "B"), components=True, magnitude=True, fps=50))

        # this should plot the normal basic names, not magnitudified names
        print("Saved", plot_2d_time(f, "zz_time_2.gif", domain=domain, fields=("E", "B"),
                                    titles=("E", "B"), components=False, magnitude=True, fps=50))

        # this should just not fail
        print("Saved", plot_2d_time(f, "zz_time_3.gif", domain=domain, fields=("E", "B"),
                                    titles=("E1", "E2", "E3", "B"), components=True, magnitude=True, fps=50))



#* take string path instead of opening file
#* save to other format: pngs [check if gif still works]
if input("check png format and string as input: ") == "y":
    #* keeps the .gif ending even if format is png
    plot_2d_time(data, "zz_as_pngs.gif", format="png", domain=domain, fields=("E", "B"),
                                    titles=("E", "B"), components=True, magnitude=True, fps=50)
    #* automatically adds .gif to the name if missing for gifs
    plot_2d_time(data, "zz_no_ending_gif", format="gif", domain=domain, fields=("E", "B"),
                                    titles=("E", "B"), components=True, magnitude=True, fps=50)




#! symlog
#! take coefficient-file + femspace or coefficient-file + domain-yml + mapping-file
    # coeefs.h5 + FemSpace  //  coeefs.h5 + geometry.h5 + mapping.yml
#! same_scale_for_all inside the dicts, to have same for components, or for all all
#!??? SCHAUEN, was als erstes kommt: magnitude oder components, um die order der plots <-> title zu bestimmen


#* {"magnitude": True, "components": True} => show both magn and comp plots
#* auto names
