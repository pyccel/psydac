# -*- coding: UTF-8 -*-
import numpy as np
import os
from spl.mapping import Mapping

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')

def test_line():
    filename = os.path.join(data_dir, 'line.nml')
    mapping = Mapping(filename=filename, p_dim=1)

    u = np.linspace(0., 1., 5)
    y = mapping.evaluate(u)
    print(y)

test_line()
