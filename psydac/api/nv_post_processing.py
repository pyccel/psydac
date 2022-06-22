import os
import pytest
import numpy as np

from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import Domain

from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil
from psydac.fem                import FemField

from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_grid_vals
from psydac.feec.multipatch.plotting_utilities import get_patch_knots_gridlines, my_small_plot

import matplotlib.pyplot as plt
from matplotlib import animation


domain = Domain.from_file(filename)
V1 = VectorFunctionSpace('V1', domain, kind='H1')
V2 = ScalarFunctionSpace('V2', domain, kind='L2')
X  = ProductSpace(V1, V2)

domain_h = discretize(domain, filename=filename)

# ... discrete spaces
V1h = discretize(V1, domain_h)
V2h = discretize(V2, domain_h)
Xh  = discretize(X, domain_h)

fields_list = [np.load(fields_folder+"/u_p_{}".format(k)) for k in range(150)]
fields_list = [array_to_stencil(f, V1h.vector_space) for f in fields_list]
fields_list = [FemField(V1h, coeffs=f) for f in fields_list]
    
