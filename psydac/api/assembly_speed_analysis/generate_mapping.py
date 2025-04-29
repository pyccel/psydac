import  os
import  numpy as np

from    sympde.topology import Mapping, ScalarFunctionSpace, Cube

from    psydac.api.discretization  import discretize
from    psydac.cad.geometry        import Geometry
from    psydac.mapping.discrete    import SplineMapping

def make_quarter_torus_geometry(ncells, degree, comm=None):
    
    class QuarterTorus(Mapping):
        _expressions = {'x': 'x1 * cos(x2)',
                        'y': 'x1 * sin(x2)',
                        'z': 'x3'}

        _ldim        = 3
        _pdim        = 3

    if (ncells[0] == ncells[1]) and (ncells[0] == ncells[2]):
        nc = f'{ncells[0]}'
    else:
        nc = f'{ncells[0]}_{ncells[1]}_{ncells[2]}'
    
    if (degree[0] == degree[1]) and (degree[0] == degree[2]):
        de = f'{degree[0]}'
    else:
        de = f'{degree[0]}_{degree[1]}_{degree[2]}'

    name = f'quarter_torus_n{nc}_d{de}'

    domain = Cube('C', bounds1=(0.5, 1), bounds2=(0, np.pi/2), bounds3=(0, 1))
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    V = ScalarFunctionSpace('V', domain)
    V_h = discretize(V, domain_h, degree=degree)

    M = QuarterTorus('QT')
    map_discrete = SplineMapping.from_mapping(V_h, M.get_callable_mapping())

    geometry = Geometry.from_discrete_mapping(map_discrete, comm=comm)

    if ((comm is not None and comm.rank==0) or (comm is None)):
        if not os.path.isdir('geometry'):
            os.makedirs('geometry')

    path = f'geometry/{name}.h5'

    geometry.export(path)

    return path
