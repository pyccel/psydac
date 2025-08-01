# -*- coding: utf-8 -*-
"""
Test module for discretization functions with custom grid parameter.

This module tests the integration between the grid parameter in Geometry
and its usage in the discretize_space function.
"""

import pytest
import numpy as np

from sympde.topology import Square, Line, Cube
from sympde.topology import ScalarFunctionSpace

from psydac.cad.geometry import Geometry
from psydac.api.discretization import discretize_space


def test_discretize_space_custom_grid_1d():
    """Test discretize_space with custom grid in 1D."""
    
    # Create a Line domain
    domain = Line('Omega', bounds=(0.0, 1.0))
    
    # Define custom grid with non-uniform spacing
    custom_grid = [np.array([0.0, 0.1, 0.4, 0.7, 0.9, 1.0])]  # 5 cells
    
    # Create geometry with custom grid
    ncells = {domain.name: [5]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use our custom grid
    degree = {domain.name: [2]}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree)
    
    # Verify that the breaks match our custom grid
    breaks = discrete_space.spaces[0].breaks
    assert np.allclose(breaks, custom_grid[0]), f"Expected {custom_grid[0]}, got {breaks}"
    
    # Verify that the number of cells is correct
    assert discrete_space.spaces[0].ncells == 5


def test_discretize_space_custom_grid_2d():
    """Test discretize_space with custom grid in 2D."""
    
    # Create a Square domain
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    
    # Define custom grid with non-uniform spacing
    custom_grid = [
        np.array([0.0, 0.2, 0.6, 1.0]),      # 3 cells in x-direction
        np.array([0.0, 0.3, 0.5, 0.8, 1.0])  # 4 cells in y-direction
    ]
    
    # Create geometry with custom grid
    ncells = {domain.name: [3, 4]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use our custom grid
    degree = {domain.name: [2, 2]}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree)
    
    # Verify that the breaks match our custom grid for both directions
    for i, (space, expected_grid) in enumerate(zip(discrete_space.spaces, custom_grid)):
        breaks = space.breaks
        assert np.allclose(breaks, expected_grid), f"Direction {i}: Expected {expected_grid}, got {breaks}"
        
        # Verify number of cells
        expected_ncells = len(expected_grid) - 1
        assert space.ncells == expected_ncells


def test_discretize_space_fallback_uniform_grid():
    """Test that when grid=None, uniform grid is used."""
    
    # Create a Square domain
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    
    # Create geometry without custom grid (grid=None)
    ncells = {domain.name: [4, 4]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=None)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use uniform grid
    degree = {domain.name: [2, 2]}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree)
    
    # Verify that uniform grids are used
    for i, space in enumerate(discrete_space.spaces):
        breaks = space.breaks
        
        # For uniform grid, breaks should be evenly spaced
        expected_breaks = np.linspace(0.0, 1.0, ncells[domain.name][i] + 1)
        assert np.allclose(breaks, expected_breaks), f"Expected uniform grid {expected_breaks}, got {breaks}"


@pytest.mark.parametrize("degree", [[1], [2], [3]])
def test_discretize_space_custom_grid_1d_various_degrees(degree):
    """Test discretize_space with custom grid for various degrees in 1D."""
    
    # Create a Line domain
    domain = Line('Omega', bounds=(0.0, 1.0))
    
    # Define custom grid
    custom_grid = [np.array([0.0, 0.2, 0.5, 0.8, 1.0])]  # 4 cells
    
    # Create geometry with custom grid
    ncells = {domain.name: [4]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree_dict = {domain.name: degree}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree_dict)
    
    # Verify that the breaks match our custom grid
    breaks = discrete_space.spaces[0].breaks
    assert np.allclose(breaks, custom_grid[0])
    
    # Verify degree
    assert discrete_space.spaces[0].degree == degree[0]


@pytest.mark.parametrize("ncells,expected_cells", [
    ([3], 3),
    ([5], 5), 
    ([8], 8)
])
def test_discretize_space_custom_grid_1d_various_ncells(ncells, expected_cells):
    """Test discretize_space with custom grid for various numbers of cells in 1D."""
    
    # Create a Line domain
    domain = Line('Omega', bounds=(0.0, 1.0))
    
    # Define custom grid with specified number of cells
    custom_grid = [np.linspace(0.0, 1.0, expected_cells + 1)]
    
    # Create geometry with custom grid
    ncells_dict = {domain.name: ncells}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells_dict, mappings=mappings, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree = {domain.name: [2]}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree)
    
    # Verify that the breaks match our custom grid
    breaks = discrete_space.spaces[0].breaks
    assert np.allclose(breaks, custom_grid[0])
    
    # Verify number of cells
    assert discrete_space.spaces[0].ncells == expected_cells


def test_discretize_space_custom_grid_2d_non_uniform():
    """Test discretize_space with highly non-uniform custom grid in 2D."""
    
    # Create a Square domain
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    
    # Define highly non-uniform custom grid
    custom_grid = [
        np.array([0.0, 0.01, 0.02, 0.5, 0.98, 0.99, 1.0]),  # 6 cells, clustered at boundaries
        np.array([0.0, 0.1, 0.9, 1.0])  # 3 cells, clustered at boundaries
    ]
    
    # Create geometry with custom grid
    ncells = {domain.name: [6, 3]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree = {domain.name: [3, 2]}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree)
    
    # Verify that the breaks match our custom grid for both directions
    for i, (space, expected_grid) in enumerate(zip(discrete_space.spaces, custom_grid)):
        breaks = space.breaks
        assert np.allclose(breaks, expected_grid), f"Direction {i}: Expected {expected_grid}, got {breaks}"
        
        # Verify that spacing is indeed non-uniform
        spacings = np.diff(breaks)
        assert not np.allclose(spacings, spacings[0]), f"Grid should be non-uniform in direction {i}"


def test_discretize_space_custom_grid_3d():
    """Test discretize_space with custom grid in 3D."""
    
    # Create a Cube domain
    domain = Cube('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0), bounds3=(0.0, 1.0))
    
    # Define custom grid with non-uniform spacing in 3D
    custom_grid = [
        np.array([0.0, 0.3, 0.7, 1.0]),          # 3 cells in x-direction
        np.array([0.0, 0.2, 0.5, 0.8, 1.0]),     # 4 cells in y-direction
        np.array([0.0, 0.4, 1.0])                # 2 cells in z-direction
    ]
    
    # Create geometry with custom grid
    ncells = {domain.name: [3, 4, 2]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use our custom grid
    degree = {domain.name: [2, 2, 2]}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree)
    
    # Verify that the breaks match our custom grid for all directions
    for i, (space, expected_grid) in enumerate(zip(discrete_space.spaces, custom_grid)):
        breaks = space.breaks
        assert np.allclose(breaks, expected_grid), f"Direction {i}: Expected {expected_grid}, got {breaks}"
        
        # Verify number of cells
        expected_ncells = len(expected_grid) - 1
        assert space.ncells == expected_ncells


def test_discretize_space_custom_grid_3d_non_uniform():
    """Test discretize_space with highly non-uniform custom grid in 3D."""
    
    # Create a Cube domain
    domain = Cube('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0), bounds3=(0.0, 1.0))
    
    # Define highly non-uniform custom grid - clustered at boundaries
    custom_grid = [
        np.array([0.0, 0.01, 0.5, 0.99, 1.0]),   # 4 cells in x, clustered at boundaries
        np.array([0.0, 0.1, 0.9, 1.0]),          # 3 cells in y, clustered at boundaries  
        np.array([0.0, 0.02, 0.03, 0.97, 0.98, 1.0])  # 5 cells in z, very clustered
    ]
    
    # Create geometry with custom grid
    ncells = {domain.name: [4, 3, 5]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree = {domain.name: [2, 3, 2]}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree)
    
    # Verify that the breaks match our custom grid for all directions
    for i, (space, expected_grid) in enumerate(zip(discrete_space.spaces, custom_grid)):
        breaks = space.breaks
        assert np.allclose(breaks, expected_grid), f"Direction {i}: Expected {expected_grid}, got {breaks}"
        
        # Verify that spacing is indeed non-uniform
        spacings = np.diff(breaks)
        assert not np.allclose(spacings, spacings[0]), f"Grid should be non-uniform in direction {i}"


@pytest.mark.parametrize("degree", [[1, 1, 1], [2, 2, 2], [3, 2, 1], [1, 3, 2]])
def test_discretize_space_custom_grid_3d_various_degrees(degree):
    """Test discretize_space with custom grid for various degrees in 3D."""
    
    # Create a Cube domain
    domain = Cube('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0), bounds3=(0.0, 1.0))
    
    # Define custom grid
    custom_grid = [
        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),   # 4 cells in x
        np.array([0.0, 0.33, 0.67, 1.0]),        # 3 cells in y
        np.array([0.0, 0.5, 1.0])                # 2 cells in z
    ]
    
    # Create geometry with custom grid
    ncells = {domain.name: [4, 3, 2]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree_dict = {domain.name: degree}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree_dict)
    
    # Verify that the breaks match our custom grid for all directions
    for i, (space, expected_grid) in enumerate(zip(discrete_space.spaces, custom_grid)):
        breaks = space.breaks
        assert np.allclose(breaks, expected_grid)
        
        # Verify degree
        assert space.degree == degree[i]


def test_discretize_space_custom_grid_3d_preserves_boundary_values():
    """Test that custom grid preserves boundary values in 3D."""
    
    # Create a Cube domain with specific bounds
    domain = Cube('Omega', bounds1=(-1.0, 2.0), bounds2=(0.5, 3.5), bounds3=(-0.5, 1.5))
    
    # Define custom grid that respects the domain bounds
    custom_grid = [
        np.array([-1.0, 0.0, 1.0, 2.0]),         # x: 3 cells from -1 to 2
        np.array([0.5, 1.5, 2.5, 3.5]),          # y: 3 cells from 0.5 to 3.5
        np.array([-0.5, 0.0, 0.5, 1.0, 1.5])     # z: 4 cells from -0.5 to 1.5
    ]
    
    # Create geometry with custom grid
    ncells = {domain.name: [3, 3, 4]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree = {domain.name: [2, 2, 2]}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree)
    
    # Verify that the breaks match our custom grid for all directions
    for i, (space, expected_grid) in enumerate(zip(discrete_space.spaces, custom_grid)):
        breaks = space.breaks
        assert np.allclose(breaks, expected_grid)
        
        # Verify boundary values are preserved
        assert breaks[0] == expected_grid[0]  # First boundary
        assert breaks[-1] == expected_grid[-1]  # Last boundary
    
    # Verify specific boundary values
    assert discrete_space.spaces[0].breaks[0] == -1.0  # x min
    assert discrete_space.spaces[0].breaks[-1] == 2.0  # x max
    assert discrete_space.spaces[1].breaks[0] == 0.5   # y min
    assert discrete_space.spaces[1].breaks[-1] == 3.5  # y max
    assert discrete_space.spaces[2].breaks[0] == -0.5  # z min
    assert discrete_space.spaces[2].breaks[-1] == 1.5  # z max


def test_discretize_space_fallback_uniform_grid_3d():
    """Test that when grid=None, uniform grid is used in 3D."""
    
    # Create a Cube domain
    domain = Cube('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0), bounds3=(0.0, 1.0))
    
    # Create geometry without custom grid (grid=None)
    ncells = {domain.name: [3, 4, 2]}
    mappings = {domain.name: None}
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=None)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use uniform grid
    degree = {domain.name: [2, 2, 2]}
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree=degree)
    
    # Verify that uniform grids are used in all directions
    for i, space in enumerate(discrete_space.spaces):
        breaks = space.breaks
        
        # For uniform grid, breaks should be evenly spaced
        expected_breaks = np.linspace(0.0, 1.0, ncells[domain.name][i] + 1)
        assert np.allclose(breaks, expected_breaks), f"Expected uniform grid {expected_breaks}, got {breaks}"


# ==============================================================================
# GRID SECURITY TESTS
# ==============================================================================

def test_grid_security_type_validation():
    """Test that grid type validation works correctly."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test invalid grid type
    ncells = {domain.name: [4]}
    mappings = {domain.name: None}
    
    with pytest.raises(TypeError, match="Grid must be a list or tuple"):
        geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid="invalid")
        discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})


def test_grid_security_dimension_mismatch():
    """Test that grid dimension validation works correctly."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test 2D grid for 1D domain
    ncells = {domain.name: [4]}
    mappings = {domain.name: None}
    
    with pytest.raises(ValueError, match="Grid dimensions .* must match domain dimensions"):
        invalid_grid = [[0, 0.25, 0.5, 0.75, 1], [0, 0.5, 1]]  # 2D grid for 1D domain
        geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=invalid_grid)
        discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})


def test_grid_security_length_consistency():
    """Test that grid length vs ncells consistency is enforced."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test grid length inconsistent with ncells
    ncells = {domain.name: [4]}
    mappings = {domain.name: None}
    
    with pytest.raises(ValueError, match="grid length .* must be ncells\\+1"):
        invalid_grid = [[0, 0.33, 0.66, 1]]  # 3 cells but ncells=[4]
        geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=invalid_grid)
        discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})


def test_grid_security_monotonic_order():
    """Test that grid points must be in strictly increasing order."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test non-monotonic grid
    ncells = {domain.name: [4]}
    mappings = {domain.name: None}
    
    with pytest.raises(ValueError, match="grid points must be strictly increasing"):
        invalid_grid = [[0, 0.5, 0.25, 0.75, 1]]  # Not monotonic
        geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=invalid_grid)
        discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})


def test_grid_security_boundary_mismatch():
    """Test that grid boundaries must match domain boundaries."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test start boundary mismatch
    ncells = {domain.name: [4]}
    mappings = {domain.name: None}
    
    with pytest.raises(ValueError, match="grid start .* must match domain minimum"):
        invalid_grid = [[0.1, 0.3, 0.6, 0.8, 1]]  # Start doesn't match domain [0,1]
        geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=invalid_grid)
        discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})
    
    # Test end boundary mismatch
    with pytest.raises(ValueError, match="grid end .* must match domain maximum"):
        invalid_grid = [[0, 0.2, 0.4, 0.6, 0.9]]  # End doesn't match domain [0,1]
        geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=invalid_grid)
        discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})


def test_grid_security_2d_validation():
    """Test grid security checks work correctly in 2D."""
    
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test inconsistent 2D grid dimensions
    ncells = {domain.name: [3, 2]}
    mappings = {domain.name: None}
    
    with pytest.raises(ValueError, match="Dimension 1: grid length .* must be ncells\\+1"):
        invalid_grid = [
            [0, 0.33, 0.66, 1],      # 3 cells ✓
            [0, 0.5, 1, 1.5]         # 4 points but should be 3 ✗
        ]
        geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=invalid_grid)
        discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2, 2]})


def test_grid_security_boundary_tolerance():
    """Test that small numerical differences in boundaries are tolerated."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test grid within tolerance (should work)
    ncells = {domain.name: [2]}
    mappings = {domain.name: None}
    
    # Small boundary differences (within tolerance)
    tolerance_grid = [[1e-15, 0.5, 1.0 - 1e-15]]
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=tolerance_grid)
    
    # This should work without raising an exception
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})
    assert discrete_space.spaces[0].ncells == 2


def test_grid_security_boundary_tolerance_exceeded():
    """Test that boundary differences outside tolerance are rejected."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test boundaries outside tolerance
    ncells = {domain.name: [2]}
    mappings = {domain.name: None}
    
    with pytest.raises(ValueError, match="grid start .* must match domain minimum"):
        # Boundaries outside tolerance
        invalid_grid = [[1e-10, 0.5, 1.0 - 1e-10]]
        geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=invalid_grid)
        discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})


def test_grid_security_3d_validation():
    """Test grid security checks work correctly in 3D."""
    
    domain = Cube('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0), bounds3=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test valid 3D grid (should work)
    ncells = {domain.name: [2, 3, 2]}
    mappings = {domain.name: None}
    
    valid_grid = [
        [0, 0.5, 1.0],           # 2 cells in x
        [0, 0.33, 0.67, 1.0],    # 3 cells in y  
        [0, 0.4, 1.0]            # 2 cells in z
    ]
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=valid_grid)
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2, 2, 2]})
    
    # Verify all dimensions work correctly
    for i, expected_ncells in enumerate([2, 3, 2]):
        assert discrete_space.spaces[i].ncells == expected_ncells
    
    # Test invalid 3D grid (dimension 2 has wrong length)
    with pytest.raises(ValueError, match="Dimension 2: grid length .* must be ncells\\+1"):
        invalid_grid = [
            [0, 0.5, 1.0],              # 2 cells ✓
            [0, 0.33, 0.67, 1.0],       # 3 cells ✓
            [0, 0.25, 0.5, 0.75, 1.0]   # 4 cells instead of 2 ✗
        ]
        geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=invalid_grid)
        discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2, 2, 2]})


@pytest.mark.parametrize("error_type,grid_config,error_pattern", [
    ("type", "invalid_string", "Grid must be a list or tuple"),
    ("length", [[0, 0.5, 1]], "grid length .* must be ncells\\+1"),
    ("monotonic", [[0, 0.8, 0.5, 0.9, 1]], "grid points must be strictly increasing"),
    ("boundary_start", [[0.1, 0.3, 0.6, 0.8, 1]], "grid start .* must match domain minimum"),
    ("boundary_end", [[0, 0.2, 0.4, 0.6, 0.9]], "grid end .* must match domain maximum"),
])
def test_grid_security_parametrized_errors(error_type, grid_config, error_pattern):
    """Parametrized test for various grid security error conditions."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    ncells = {domain.name: [4]}
    mappings = {domain.name: None}
    
    if error_type == "type":
        with pytest.raises(TypeError, match=error_pattern):
            geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=grid_config)
            discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})
    else:
        with pytest.raises(ValueError, match=error_pattern):
            geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=grid_config)
            discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [2]})


def test_grid_security_preserves_functionality():
    """Test that security checks don't interfere with normal functionality."""
    
    # Test that valid grids still work perfectly
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Define valid but complex custom grid
    ncells = {domain.name: [4, 3]}
    mappings = {domain.name: None}
    
    valid_grid = [
        [0, 0.1, 0.3, 0.7, 1.0],    # 4 cells, non-uniform
        [0, 0.2, 0.8, 1.0]          # 3 cells, non-uniform
    ]
    
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings, grid=valid_grid)
    discrete_space = discretize_space(symbolic_space, domain_h=geo, degree={domain.name: [3, 2]})
    
    # Verify that everything works as expected
    for i, (space, expected_grid) in enumerate(zip(discrete_space.spaces, valid_grid)):
        assert np.allclose(space.breaks, expected_grid)
        assert space.ncells == len(expected_grid) - 1
        assert space.degree == [3, 2][i]
    
    # Verify that the grid is indeed non-uniform
    for i, space in enumerate(discrete_space.spaces):
        spacings = np.diff(space.breaks)
        assert not np.allclose(spacings, spacings[0]), f"Grid should be non-uniform in direction {i}"


# CLEAN UP SYMPY NAMESPACE
def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
