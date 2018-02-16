# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import f90nml
from clapp.spl.mapping import Mapping

def test1():
    # TODO we can avoid giving p_dim, if we read first the nml file from python,
    # set internaly p_dim and then use it to branch with the fortran core
    mapping = Mapping(filename="mapping.nml", p_dim=3)
    print(("mapping.id : ", mapping.id))
    print (mapping)
    mapping.export("mapping_out.nml")

    print("test1: passed")

def test2():
    from caid.cad_geometry import cube
    geometry = cube()
    mapping = Mapping(geometry=geometry)
    print(("mapping.id : ", mapping.id))
    print (mapping)
    mapping.export("mapping_out.nml")

    print("test2: passed")

def test3():
    from caid.cad_geometry import cube
    geometry = cube()
    mapping = Mapping(geometry=geometry)

    u = v = w = np.linspace(0., 1., 5)
    y = mapping.evaluate(u, v, w)
    print(y)

    print("test3: passed")

def test4():
    from caid.cad_geometry import cube
    geometry = cube()
    mapping = Mapping(geometry=geometry)

    geo = mapping.to_cad_geometry()
    geo.plotMesh()
    plt.show()

    print("test4: passed")

#test1()
#test2()
#test3()
test4()
