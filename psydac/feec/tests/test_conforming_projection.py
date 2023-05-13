import numpy as np

from sympde.topology  import Derham, Square, IdentityMapping, PolarMapping
from sympde.topology.domain import Domain, Union, Connectivity

from psydac.feec.global_projectors import projection_matrix_Hdiv_homogeneous_bc

from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.cad.geometry     import Geometry
from psydac.fem.basic import FemField
from psydac.fem.vector import VectorFemSpace
from psydac.linalg.utilities import array_to_psydac



def plot_Hdiv_homogeneous_bc_square():
    """
    Plot a H(div)-conforming FE field on square 
    before and after applying projection
    """
    # Initialize square domain with identity mapping
    A  = Square('A',bounds1=(0, 1), bounds2=(0, 1))
    M1 = IdentityMapping('M1', dim=2)
    domain = M1(A)
    assert isinstance(A, Domain)
    derham  = Derham(domain, ["H1", "Hdiv", "L2"])

    # Discretize the domain, vector space
    ncells = [3, 5]
    domain_h : Geometry = discretize(domain, ncells=ncells)   # Vh space
    derham_h= discretize(derham, domain_h, degree=[2,2])
    assert isinstance(derham_h, DiscreteDerham)
    V1h : VectorFemSpace = derham_h.V1

    # Compute the projecton matrix and define vector field with coefficients set
    # to one
    proj_V1 = projection_matrix_Hdiv_homogeneous_bc(V1h)
    G1h_coeffs_arr = np.concatenate( (np.ones(V1h.spaces[0].nbasis),
                                      np.ones(V1h.spaces[1].nbasis)))
    G1h_coeffs = array_to_psydac(G1h_coeffs_arr, V1h.vector_space)
    G1h = FemField(V1h, G1h_coeffs)

    # Export the data before projection
    output_manager = OutputManager('spaces_projection.yml', 
                                   'fields_projection.h5')
    output_manager.add_spaces(V1=V1h)
    output_manager.export_space_info()
    output_manager.set_static()
    output_manager.export_fields(G1h=G1h)

    # Project the field and export it together with field before projection
    G1h_proj_coeffs_arr = proj_V1 @ G1h_coeffs_arr
    G1h_proj_coeffs = array_to_psydac(G1h_proj_coeffs_arr, V1h.vector_space)
    G1h_proj = FemField(V1h, G1h_proj_coeffs)
    output_manager.export_fields(G1h_proj=G1h_proj)
    post_processor = PostProcessManager(domain=domain, 
                                        space_file='spaces_projection.yml',
                                        fields_file='fields_projection.h5')
    post_processor.export_to_vtk('field_projection_vtk', npts_per_cell=3,
                                    fields=("G1h", "G1h_proj"))


def plot_Hdiv_homogeneous_bc_annulus():
    """
    Plot a H(div)-conforming FE field on annulus 
    before and after applying projection
    """
    # Define symbolic domain and spaces
    rmin = 1.0
    rmax = 2.0
    logical_domain = Square("logical_domain", (0,1), (0,2*np.pi))
    boundary = Union(logical_domain.get_boundary(axis=0, ext=-1),
                     logical_domain.get_boundary(axis=0, ext=1))
    ###DEBUG###
    print("\nboundary:", type(boundary), boundary)
    print("logical_domain.interior:", logical_domain.interior)
    ###########
    logical_domain = Domain(name="logical_domain", interiors=logical_domain.interior,
                            boundaries=boundary, dim=2)
    polar_mapping = PolarMapping("polar_mapping", dim=2, c1=0., 
                                 c2=0., rmin=rmin, rmax=rmax)
    annulus = polar_mapping(logical_domain)
    ###DEBUG###
    print("\nannulus:", type(annulus), annulus)
    print("\tannulus.mapping:", type(annulus.mapping), annulus.mapping)
    print("\tannulus.connectivity:", type(annulus.connectivity), annulus.connectivity)
    print("\tannulus.dtype:", type(annulus.dtype), annulus.dtype)
    ###########
    derham  = Derham(annulus, ["H1", "Hdiv", "L2"])

    # Discretize the domain, vector space
    ncells = [3, 5]
    domain_h : Geometry = discretize(annulus, ncells=ncells, periodic=[False, True])
    derham_h= discretize(derham, domain_h, degree=[2,2])
    assert isinstance(derham_h, DiscreteDerham)
    V1h : VectorFemSpace = derham_h.V1

    # Compute the projecton matrix and define vector field 
    proj_V1 = projection_matrix_Hdiv_homogeneous_bc(V1h)
    ###DEBUG###
    print("type(proj_V1):", type(proj_V1))
    ###########
    # One component is zero the other has coefficients equal to one
    G1h_coeffs_arr = np.concatenate( (np.ones(V1h.spaces[0].nbasis),
                                    np.zeros(V1h.spaces[1].nbasis)))
    G1h_coeffs = array_to_psydac(G1h_coeffs_arr, V1h.vector_space)
    G1h = FemField(V1h, G1h_coeffs)

    # Export fields before and after projection
    output_manager = OutputManager('spaces_projection.yml', 
                                    'fields_projection.h5')
    output_manager.add_spaces(V1=V1h)
    output_manager.export_space_info()
    output_manager.set_static()
    output_manager.export_fields(G1h=G1h)
    G1h_proj_coeffs_arr = proj_V1 @ G1h_coeffs_arr
    G1h_proj_coeffs = array_to_psydac(G1h_proj_coeffs_arr, V1h.vector_space)
    G1h_proj = FemField(V1h, G1h_proj_coeffs)

    output_manager.export_fields(G1h_proj=G1h_proj)
    post_processor = PostProcessManager(domain=annulus, 
                                        space_file='spaces_projection.yml',
                                        fields_file='fields_projection.h5')
    post_processor.export_to_vtk('field_projection_vtk', npts_per_cell=3,
                                    fields=("G1h", "G1h_proj"))



if __name__ == "__main__":

    test_domain = "annulus"

    if test_domain == "annulus":
        plot_Hdiv_homogeneous_bc_annulus()
    elif test_domain == "square":
        plot_Hdiv_homogeneous_bc_square()