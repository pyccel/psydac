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

from psydac.api.discretization import discretize


def test_discretize_space_custom_grid_1d():
    """Test discretize_space with custom grid in 1D."""
    
    # Create a Line domain
    domain = Line('Omega', bounds=(0.0, 1.0))
    
    # Define custom grid with non-uniform spacing
    custom_grid = [np.array([0.0, 0.1, 0.4, 0.7, 0.9, 1.0])]  # 5 cells
    
    # Create geometry with custom grid
    ncells = [5]
    domain_h = discretize(domain, ncells=ncells, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use our custom grid
    degree = [2]
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)

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
    ncells = [3, 4]
    domain_h = discretize(domain, ncells=ncells, grid=custom_grid)
    
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use our custom grid
    degree = [2, 2]
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)

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
    ncells = [4, 4]
    domain_h = discretize(domain, ncells=ncells)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use uniform grid
    degree = [2, 2]
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)
    
    # Verify that uniform grids are used
    for i, space in enumerate(discrete_space.spaces):
        breaks = space.breaks
        
        # For uniform grid, breaks should be evenly spaced
        expected_breaks = np.linspace(0.0, 1.0, ncells[i] + 1)
        assert np.allclose(breaks, expected_breaks), f"Expected uniform grid {expected_breaks}, got {breaks}"


@pytest.mark.parametrize("degree", [[1], [2], [3]])
def test_discretize_space_custom_grid_1d_various_degrees(degree):
    """Test discretize_space with custom grid for various degrees in 1D."""
    
    # Create a Line domain
    domain = Line('Omega', bounds=(0.0, 1.0))
    
    # Define custom grid
    custom_grid = [np.array([0.0, 0.2, 0.5, 0.8, 1.0])]  # 4 cells
    
    # Create geometry with custom grid
    ncells = [4]
    
    domain_h = discretize(domain, ncells=ncells, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)

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
    domain_h = discretize(domain, ncells=ncells, grid=custom_grid)

    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree = [2]
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)

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
    ncells = [6, 3]
    
    domain_h = discretize(domain, ncells=ncells, grid=custom_grid)
    
    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree = [3, 2]
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)

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
    ncells = [3, 4, 2]

    domain_h = discretize(domain, ncells=ncells, grid=custom_grid)

    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use our custom grid
    degree = [2, 2, 2]
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)

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
    ncells = [4, 3, 5]

    domain_h = discretize(domain, ncells=ncells, grid=custom_grid)

    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree = [2, 3, 2]
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)
    
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
    ncells = [4, 3, 2]

    domain_h = discretize(domain, ncells=ncells, grid=custom_grid)

    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)

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
    ncells = [3, 3, 4]
    
    domain_h = discretize(domain, ncells=ncells, grid=custom_grid)

    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space
    degree = [2, 2, 2]
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)

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
    ncells = [3, 4, 2]
    
    domain_h = discretize(domain, ncells=ncells, grid=None)

    # Create symbolic space
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Discretize the space - this should use uniform grid
    degree = [2, 2, 2]
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=degree)

    # Verify that uniform grids are used in all directions
    for i, space in enumerate(discrete_space.spaces):
        breaks = space.breaks
        
        # For uniform grid, breaks should be evenly spaced
        expected_breaks = np.linspace(0.0, 1.0, ncells[i] + 1)
        assert np.allclose(breaks, expected_breaks), f"Expected uniform grid {expected_breaks}, got {breaks}"


# ==============================================================================
# GRID SECURITY TESTS
# ==============================================================================

def test_grid_security_type_validation():
    """Test that grid type validation works correctly."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test invalid grid type
    ncells = [4]
    
    
    with pytest.raises(TypeError, match="Grid must be a list, tuple, or dict"):
        domain_h = discretize(domain, ncells=ncells, grid="invalid")
        discretize(symbolic_space, domain_h=domain_h, degree=[2])


def test_grid_security_dimension_mismatch():
    """Test that grid dimension validation works correctly."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test 2D grid for 1D domain
    ncells = [4]
    
    
    with pytest.raises(ValueError, match="Grid dimensions .* must match domain dimensions"):
        invalid_grid = [[0, 0.25, 0.5, 0.75, 1], [0, 0.5, 1]]  # 2D grid for 1D domain
        domain_h = discretize(domain, ncells=ncells, grid=invalid_grid)
        discretize(symbolic_space, domain_h=domain_h, degree=[2])


def test_grid_security_length_consistency():
    """Test that grid length vs ncells consistency is enforced."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test grid length inconsistent with ncells
    ncells = [4]
    
    
    with pytest.raises(ValueError, match="grid length .* must be ncells\\+1"):
        invalid_grid = [[0, 0.33, 0.66, 1]]  # 3 cells but ncells=[4]
        domain_h = discretize(domain, ncells=ncells, grid=invalid_grid)
        discretize(symbolic_space, domain_h=domain_h, degree=[2])


def test_grid_security_monotonic_order():
    """Test that grid points must be in strictly increasing order."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test non-monotonic grid
    ncells = [4]
    
    
    with pytest.raises(ValueError, match="grid points must be strictly increasing"):
        invalid_grid = [[0, 0.5, 0.25, 0.75, 1]]  # Not monotonic
        domain_h = discretize(domain, ncells=ncells, grid=invalid_grid)
        discretize(symbolic_space, domain_h=domain_h, degree=[2])


def test_grid_security_boundary_mismatch():
    """Test that grid boundaries must match domain boundaries."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test start boundary mismatch
    ncells = [4]
    
    
    with pytest.raises(ValueError, match="grid start .* must match domain minimum"):
        invalid_grid = [[0.1, 0.3, 0.6, 0.8, 1]]  # Start doesn't match domain [0,1]
        domain_h = discretize(domain, ncells=ncells, grid=invalid_grid)
        discretize(symbolic_space, domain_h=domain_h, degree=[2])

    # Test end boundary mismatch
    with pytest.raises(ValueError, match="grid end .* must match domain maximum"):
        invalid_grid = [[0, 0.2, 0.4, 0.6, 0.9]]  # End doesn't match domain [0,1]
        domain_h = discretize(domain, ncells=ncells, grid=invalid_grid)
        discretize(symbolic_space, domain_h=domain_h, degree=[2])


def test_grid_security_2d_validation():
    """Test grid security checks work correctly in 2D."""
    
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test inconsistent 2D grid dimensions
    ncells = [3, 2]
    
    
    with pytest.raises(ValueError, match="Dimension 1: grid length .* must be ncells\\+1"):
        invalid_grid = [
            [0, 0.33, 0.66, 1],      # 3 cells ✓
            [0, 0.5, 1, 1.5]         # 4 points but should be 3 ✗
        ]
        domain_h = discretize(domain, ncells=ncells, grid=invalid_grid)
        discretize(symbolic_space, domain_h=domain_h, degree=[2, 2])


def test_grid_security_boundary_tolerance():
    """Test that small numerical differences in boundaries are tolerated."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test grid within tolerance (should work)
    ncells = [2]
    
    
    # Small boundary differences (within tolerance)
    tolerance_grid = [[1e-15, 0.5, 1.0 - 1e-15]]
    domain_h = discretize(domain, ncells=ncells, grid=tolerance_grid)

    # This should work without raising an exception
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=[2])
    assert discrete_space.spaces[0].ncells == 2


def test_grid_security_boundary_tolerance_exceeded():
    """Test that boundary differences outside tolerance are rejected."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test boundaries outside tolerance
    ncells = [2]
    
    
    with pytest.raises(ValueError, match="grid start .* must match domain minimum"):
        # Boundaries outside tolerance
        invalid_grid = [[1e-10, 0.5, 1.0 - 1e-10]]
        domain_h = discretize(domain, ncells=ncells, grid=invalid_grid)
        discretize(symbolic_space, domain_h=domain_h, degree=[2])


def test_grid_security_3d_validation():
    """Test grid security checks work correctly in 3D."""
    
    domain = Cube('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0), bounds3=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Test valid 3D grid (should work)
    ncells = [2, 3, 2]
    
    
    valid_grid = [
        [0, 0.5, 1.0],           # 2 cells in x
        [0, 0.33, 0.67, 1.0],    # 3 cells in y  
        [0, 0.4, 1.0]            # 2 cells in z
    ]
    domain_h = discretize(domain, ncells=ncells, grid=valid_grid)
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=[2, 2, 2])

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
        domain_h = discretize(domain, ncells=ncells, grid=invalid_grid)
        discretize(symbolic_space, domain_h=domain_h, degree=[2, 2, 2])


@pytest.mark.parametrize("error_type,grid_config,error_pattern", [
    ("type", "invalid_string", "Grid must be a list, tuple, or dict"),
    ("length", [[0, 0.5, 1]], "grid length .* must be ncells\\+1"),
    ("monotonic", [[0, 0.8, 0.5, 0.9, 1]], "grid points must be strictly increasing"),
    ("boundary_start", [[0.1, 0.3, 0.6, 0.8, 1]], "grid start .* must match domain minimum"),
    ("boundary_end", [[0, 0.2, 0.4, 0.6, 0.9]], "grid end .* must match domain maximum"),
])
def test_grid_security_parametrized_errors(error_type, grid_config, error_pattern):
    """Parametrized test for various grid security error conditions."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    ncells = [4]
    
    
    if error_type == "type":
        with pytest.raises(TypeError, match=error_pattern):
            domain_h = discretize(domain, ncells=ncells, grid=grid_config)
            discretize(symbolic_space, domain_h=domain_h, degree=[2])
    else:
        with pytest.raises(ValueError, match=error_pattern):
            geo = discretize(domain, ncells=ncells, grid=grid_config)
            discretize(symbolic_space, domain_h=geo, degree=[2])


def test_grid_security_preserves_functionality():
    """Test that security checks don't interfere with normal functionality."""
    
    # Test that valid grids still work perfectly
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    # Define valid but complex custom grid
    ncells = [4, 3]
    
    
    valid_grid = [
        [0, 0.1, 0.3, 0.7, 1.0],    # 4 cells, non-uniform
        [0, 0.2, 0.8, 1.0]          # 3 cells, non-uniform
    ]

    domain_h = discretize(domain, ncells=ncells, grid=valid_grid)
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=[3, 2])

    # Verify that everything works as expected
    for i, (space, expected_grid) in enumerate(zip(discrete_space.spaces, valid_grid)):
        assert np.allclose(space.breaks, expected_grid)
        assert space.ncells == len(expected_grid) - 1
        assert space.degree == [3, 2][i]
    
    # Verify that the grid is indeed non-uniform
    for i, space in enumerate(discrete_space.spaces):
        spacings = np.diff(space.breaks)
        assert not np.allclose(spacings, spacings[0]), f"Grid should be non-uniform in direction {i}"

def test_grid_with_none_values_2d():
    """Test grid with None values in 2D - mixed None and custom grids."""
    
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    ncells = [3, 4]
    
    # None for dimension 0, custom for dimension 1
    grid_mixed = [
        None,                           # Uniform grid (will be auto-generated)
        [0.0, 0.2, 0.6, 0.8, 1.0]     # Custom grid
    ]
    
    domain_h = discretize(domain, ncells=ncells, grid=grid_mixed)
    
    # Verify the grid was processed correctly
    actual_grid = domain_h.grid
    assert 'Omega' in actual_grid
    
    # Check dimension 0 (should be uniform)
    expected_dim0 = np.linspace(0.0, 1.0, 4)  # 3 cells + 1 = 4 points
    assert np.allclose(actual_grid['Omega'][0], expected_dim0)
    
    # Check dimension 1 (should be preserved custom)
    expected_dim1 = [0.0, 0.2, 0.6, 0.8, 1.0]
    assert np.allclose(actual_grid['Omega'][1], expected_dim1)

def test_grid_with_all_none_values_2d():
    """Test grid with all None values in 2D - should generate uniform grids."""
    
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    ncells = [2, 3]
    
    # All None values
    grid_all_none = [None, None]
    
    domain_h = discretize(domain, ncells=ncells, grid=grid_all_none)
    
    # Verify uniform grids were generated
    actual_grid = domain_h.grid
    assert 'Omega' in actual_grid
    
    # Both dimensions should be uniform
    expected_dim0 = np.linspace(0.0, 1.0, 3)  # 2 cells + 1 = 3 points
    expected_dim1 = np.linspace(0.0, 1.0, 4)  # 3 cells + 1 = 4 points
    
    assert np.allclose(actual_grid['Omega'][0], expected_dim0)
    assert np.allclose(actual_grid['Omega'][1], expected_dim1)

def test_grid_with_none_values_3d():
    """Test grid with None values in 3D - mixed None and custom grids."""
    
    domain = Cube('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0), bounds3=(0.0, 1.0))
    ncells = [2, 3, 2]
    
    # Mixed None and custom grids
    grid_mixed_3d = [
        None,                                   # Uniform for dim 0
        [0.0, 0.33, 0.67, 1.0],                # Custom for dim 1  
        None                                    # Uniform for dim 2
    ]
    
    domain_h = discretize(domain, ncells=ncells, grid=grid_mixed_3d)
    
    # Verify the grid was processed correctly
    actual_grid = domain_h.grid
    assert 'Omega' in actual_grid
    
    # Check dimension 0 (should be uniform)
    expected_dim0 = np.linspace(0.0, 1.0, 3)  # 2 cells + 1 = 3 points
    assert np.allclose(actual_grid['Omega'][0], expected_dim0)
    
    # Check dimension 1 (should be preserved custom)
    expected_dim1 = [0.0, 0.33, 0.67, 1.0]
    assert np.allclose(actual_grid['Omega'][1], expected_dim1)
    
    # Check dimension 2 (should be uniform)
    expected_dim2 = np.linspace(0.0, 1.0, 3)  # 2 cells + 1 = 3 points
    assert np.allclose(actual_grid['Omega'][2], expected_dim2)

def test_grid_with_none_values_1d():
    """Test grid with None values in 1D."""
    
    domain = Line('Omega', bounds=(0.0, 1.0))
    ncells = [4]
    
    # None for 1D
    grid_none_1d = [None]
    
    domain_h = discretize(domain, ncells=ncells, grid=grid_none_1d)
    
    # Verify uniform grid was generated
    actual_grid = domain_h.grid
    assert 'Omega' in actual_grid
    
    expected_grid = np.linspace(0.0, 1.0, 5)  # 4 cells + 1 = 5 points
    assert np.allclose(actual_grid['Omega'], expected_grid)

def test_grid_none_values_respects_domain_bounds():
    """Test that None values generate uniform grids respecting domain bounds."""
    
    # Test with non-standard domain bounds
    domain = Square('Omega', bounds1=(2.0, 5.0), bounds2=(-1.0, 3.0))
    ncells = [3, 2]
    
    grid_all_none = [None, None]
    
    domain_h = discretize(domain, ncells=ncells, grid=grid_all_none)
    
    actual_grid = domain_h.grid
    assert 'Omega' in actual_grid
    
    # Check that bounds are respected
    expected_dim0 = np.linspace(2.0, 5.0, 4)   # bounds1 with 3 cells + 1 = 4 points
    expected_dim1 = np.linspace(-1.0, 3.0, 3)  # bounds2 with 2 cells + 1 = 3 points
    
    assert np.allclose(actual_grid['Omega'][0], expected_dim0)
    assert np.allclose(actual_grid['Omega'][1], expected_dim1)

def test_grid_none_mixed_with_discretize_space():
    """Test that grid with None values works with discretize_space."""
    
    domain = Square('Omega', bounds1=(0.0, 1.0), bounds2=(0.0, 1.0))
    symbolic_space = ScalarFunctionSpace('V', domain)
    
    ncells = [2, 3]
    grid_mixed = [
        [0.0, 0.3, 1.0],                # Custom for dim 0 (2 cells)
        None                            # Uniform for dim 1
    ]
    
    domain_h = discretize(domain, ncells=ncells, grid=grid_mixed)
    discrete_space = discretize(symbolic_space, domain_h=domain_h, degree=[2, 2])
    
    # Verify it works
    assert discrete_space.spaces[0].ncells == 2
    assert discrete_space.spaces[1].ncells == 3
    
    # Verify the grid
    actual_grid = domain_h.grid
    expected_dim0 = [0.0, 0.3, 1.0]
    expected_dim1 = np.linspace(0.0, 1.0, 4)  # 3 cells + 1 = 4 points
    
    assert np.allclose(actual_grid['Omega'][0], expected_dim0)
    assert np.allclose(actual_grid['Omega'][1], expected_dim1)


# CLEAN UP SYMPY NAMESPACE
def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
