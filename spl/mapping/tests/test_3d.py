# -*- coding: UTF-8 -*-
import numpy as np
import os
from spl.mapping import Mapping

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')

def test_cube():
    filename = os.path.join(data_dir, 'cube.nml')
    mapping = Mapping(filename=filename, p_dim=3)

    u = v = w = np.linspace(0., 1., 5)
    y = mapping.evaluate(u, v, w)
    print(y)

test_cube()
