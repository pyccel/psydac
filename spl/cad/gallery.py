# coding: utf-8
#
# TODO export not working yet

from spl.cad.basic    import BasicDiscreteDomain, Topology
from spl.cad.basic    import DiscreteBoundary
from spl.cad.basic    import ProductDiscreteDomain
from spl.cad.geometry import Geometry

class Interval(BasicDiscreteDomain):
    def __init__(self, bounds):
        self._bounds = bounds
        self._ldim = 1
        self._pdim = 1

    @property
    def bounds(self):
        return self._bounds



class Line(Geometry):
    def __init__(self, bounds):
        interval = Interval(bounds)

        Gamma_1 = DiscreteBoundary(interval, axis=0, ext=-1)
        Gamma_2 = DiscreteBoundary(interval, axis=0, ext=1)

        topology = Topology(boundaries=[Gamma_1, Gamma_2])
        return Geometry.__init__( self, patches=[interval],
                                 topology=topology )


class Square(Geometry):
    def __init__(self, *bounds):
        assert(isinstance(bounds, (tuple, list)))
        assert(all([isinstance(i, (tuple, list)) for i in bounds]))

        Ix1 = Interval(bounds[0])
        Ix2 = Interval(bounds[1])

        domain = ProductDiscreteDomain(Ix1, Ix2)

        dim = 2
        boundaries = []
        i = 0
        for axis in range(dim):
            for ext in [-1, 1]:
                Gamma = DiscreteBoundary(domain, axis=axis, ext=ext)
                boundaries += [Gamma]

                i += 1

        topology = Topology(boundaries=boundaries)
        return Geometry.__init__( self, patches=[domain],
                                 topology=topology )

class Cube(Geometry):
    def __init__(self, *bounds):
        assert(isinstance(bounds, (tuple, list)))
        assert(all([isinstance(i, (tuple, list)) for i in bounds]))

        Ix1 = Interval(bounds[0])
        Ix2 = Interval(bounds[1])
        Ix3 = Interval(bounds[2])

        domain = ProductDiscreteDomain(Ix1, Ix2, Ix3)

        dim = 3
        boundaries = []
        i = 0
        for axis in range(dim):
            for ext in [-1, 1]:
                Gamma = DiscreteBoundary(domain, axis=axis, ext=ext)
                boundaries += [Gamma]

                i += 1

        topology = Topology(boundaries=boundaries)
        return Geometry.__init__( self, patches=[domain],
                                 topology=topology )



#############################
if __name__ == '__main__':

    def test_line():
        domain = Line([0,1])

    def test_square():
        domain = Square([0,1], [0,1])

    def test_cube():
        domain = Cube([0,1], [0,1], [0,1])

    test_line()
    test_square()
    test_cube()

