# -*- coding: UTF-8 -*-
import numpy as np
import os
from spl.mapping import Mapping

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

def test_circle():
    from caid.cad_geometry import circle
    geometry = circle()
    mapping = Mapping(geometry=geometry)
    mapping.export(os.path.join(data_dir,"circle.nml"))

def test_square():
    from caid.cad_geometry import square
    geometry = square()
    mapping = Mapping(geometry=geometry)
    mapping.export(os.path.join(data_dir,"square.nml"))

test_square()
test_circle()
