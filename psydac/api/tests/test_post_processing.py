import pytest

from sympde.topology import Square, ScalarFunctionSpace, VectorFunctionSpace
from psydac.api.discretization import discretize
from psydac.fem.basic import FemField
from psydac.api.postprocessing import OutputManager

def test_OuputManager():

    domain = Square('D')
    A = ScalarFunctionSpace('A', domain, kind='H1')
    B = VectorFunctionSpace('B', domain, kind=None)

    domain_h = discretize(domain, ncells=[5,5])

    Ah = discretize(A, domain_h, degree = [3,3])
    Bh = discretize(B, domain_h, degree = [2,2])

    uh = FemField(Ah)
    vh = FemField(Bh)

    Om = OutputManager('file.yml','file.h5')
    Om.add_spaces(Ah, Bh)

    Om.set_static().export_fields(uh=uh, vh=vh)

    Om.add_snapshot(t=0., ts=0).export_fields(uh=uh, vh=vh)
    Om.add_snapshot(t=1., ts=1).export_fields(uh=uh, vh=vh)



    expected_spaces_info = {'ndim': 2,
                            'fields': 'file.h5',
                            'patches': [{'name': 'D',
                                         'scalar_spaces': [{'name': 'A',
                                                            'ldim': 2,
                                                            'kind': 'h1',
                                                            'dtype': "<class 'float'>",
                                                            'rational': False,
                                                            'periodic': [False, False],
                                                            'degree': [3, 3],
                                                            'basis': ['B', 'B'],
                                                            'knots': [
                                                                [0.0, 0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0, 1.0],
                                                                [0.0, 0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0, 1.0]
                                                            ]
                                                            },
                                                           {'name': 'B[0]',
                                                            'ldim': 2,
                                                            'kind': 'undefined',
                                                            'dtype': "<class 'float'>",
                                                            'rational': False,
                                                            'periodic': [False, False],
                                                            'degree': [2, 2],
                                                            'basis': ['B', 'B'],
                                                            'knots': [
                                                                [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0],
                                                                [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0]
                                                            ]
                                                            },
                                                           {'name': 'B[1]',
                                                            'ldim': 2,
                                                            'kind': 'undefined',
                                                            'dtype': "<class 'float'>",
                                                            'rational': False,
                                                            'periodic': [False, False],
                                                            'degree': [2, 2],
                                                            'basis': ['B', 'B'],
                                                            'knots': [
                                                                [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0],
                                                                [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0]
                                                            ]
                                                            }
                                                           ],
                                         'vector_spaces': [{'name': 'B',
                                                            'kind': 'undefined',
                                                            'components': [
                                                                {'name': 'B[0]',
                                                                 'ldim': 2,
                                                                 'kind': 'undefined',
                                                                 'dtype': "<class 'float'>",
                                                                 'rational': False,
                                                                 'periodic': [False, False],
                                                                 'degree': [2, 2],
                                                                 'basis': ['B', 'B'],
                                                                 'knots': [
                                                                     [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                      0.6000000000000001, 0.8, 1.0, 1.0, 1.0],
                                                                     [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                      0.6000000000000001, 0.8, 1.0, 1.0, 1.0]
                                                                 ]
                                                                 },
                                                                {'name': 'B[1]',
                                                                 'ldim': 2,
                                                                 'kind': 'undefined',
                                                                 'dtype': "<class 'float'>",
                                                                 'rational': False,
                                                                 'periodic': [False, False],
                                                                 'degree': [2, 2],
                                                                 'basis': ['B', 'B'],
                                                                 'knots': [
                                                                     [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                      0.6000000000000001, 0.8, 1.0, 1.0, 1.0],
                                                                     [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                      0.6000000000000001, 0.8, 1.0, 1.0, 1.0]
                                                                 ]
                                                                 }
                                                            ]
                                                            }]
                                         }]
                            }

    assert(Om._spaces_info == expected_spaces_info)

if __name__ == '__main__':
    test_OuputManager()