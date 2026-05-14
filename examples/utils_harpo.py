"""
Utilities for HARPO examples.
"""

import  numpy   as np

from scipy.sparse               import bmat, csc_matrix
from scipy.sparse.linalg        import inv
from scipy.sparse.linalg        import spsolve, eigsh

from    sympde.topology import Cube, Mapping, Derham, Domain, BasicCallableMapping

class HollowTorus(Mapping):
    _expressions = {'x': '( r + (x1 - r) * cos(x3) ) * cos(x2)',
                    'y': '( r + (x1 - r) * cos(x3) ) * sin(x2)',
                    'z': '(x1 - r) * sin(x3)'}
    _ldim        = 3
    _pdim        = 3

class HollowTorusUnit(Mapping):
    """
    The hollow torus mapping corresponding to the Struphy HollowTorus domain.
    It should be defined on the unit cube (0,1)^3, with logical coordinates 
    (x1, x2, x3) corresponding to normalized radial, poloidal, and toroidal coordinates respectively.
    """
    _expressions = {'x': '( (a1 + (a2 - a1)*x1) * cos(x2*2*pi) + R0 ) * cos(x3*2*pi)', 
                    'y': '( (a1 + (a2 - a1)*x1) * cos(x2*2*pi) + R0 ) * sin(x3*2*pi)',
                    'z':   '(a1 + (a2 - a1)*x1) * sin(x2*2*pi)'}
    _ldim        = 3
    _pdim        = 3



def get_eigenvalues(nb_eigs, sigma, A_m, M_m):
    """
    Compute the eigenvalues of the matrix A close to sigma and right-hand-side M
    Function seen and adapted from >>> psydac_dev/psydac/feec/multipatch/examples/hcurl_eigen_pbms_conga_2d.py <<< 
    (Commit a748a4d8c1569a8765f6688d228f65ea6073c252)

    Parameters
    ----------
    nb_eigs : int
        Number of eigenvalues to compute
    sigma : float
        Value close to which the eigenvalues are computed
    A_m : sparse matrix
        Matrix A
    M_m : sparse matrix
        Matrix M
    """

    #print()
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- ')
    print(
        'computing {0} eigenvalues (and eigenvectors) close to sigma={1} with scipy.sparse.eigsh...'.format(nb_eigs, sigma))
    mode = 'normal'
    which = 'LM'
    ncv = 4 * nb_eigs
    max_shape_splu = 24000
    if A_m.shape[0] >= max_shape_splu:
        raise ValueError(f'Matrix too large.')
        
    eigenvalues, eigenvectors = eigsh(
        A_m, k=nb_eigs, M=M_m, sigma=sigma, mode=mode, which=which, ncv=ncv)

    print("done: eigenvalues found: " + repr(eigenvalues))
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----')
    print()

    return eigenvalues, eigenvectors


class StruphyCallableMapping(BasicCallableMapping):
    """
    Concrete implementation of BasicCallableMapping wrapping a Struphy domain.
    
    Parameters
    ----------
    struphy_domain : object
        A Struphy domain object with __call__, jacobian, jacobian_inv, and metric methods
        (e.g., str_domains.HollowCylinder())
    logical_ldim : int
        Number of logical dimensions (default: 3)
    physical_pdim : int
        Number of physical dimensions (default: 3)
    normalize_factors : tuple, optional
        Tuple of (scale_x1, scale_x2, scale_x3) to normalize input coordinates.
        If provided, input is scaled as (x1/scale_x1, x2/scale_x2, x3/scale_x3)
    """
    
    def __init__(self, struphy_domain, logical_ldim=3, physical_pdim=3, normalize_factors=None):
        self._struphy_domain = struphy_domain
        self._ldim = logical_ldim
        self._pdim = physical_pdim
        self._normalize_factors = normalize_factors
    
    def __call__(self, *eta):
        """Evaluate mapping at location eta."""
        if len(eta) != self._ldim:
            raise ValueError(f"Expected {self._ldim} coordinates, got {len(eta)}")
        
        # Normalize coordinates if factors provided
        if self._normalize_factors is not None:
            eta_normalized = tuple(e / f for e, f in zip(eta, self._normalize_factors))
        else:
            eta_normalized = eta
        
        # Call the Struphy domain
        result = self._struphy_domain(*eta_normalized, squeeze_out=True)
        
        # Convert to numpy array if needed
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        
        return tuple(result)
    
    def jacobian(self, *eta):
        """Compute Jacobian matrix at location eta using Struphy's jacobian method."""
        if len(eta) != self._ldim:
            raise ValueError(f"Expected {self._ldim} coordinates, got {len(eta)}")
        
        # Normalize coordinates if factors provided
        if self._normalize_factors is not None:
            eta_normalized = tuple(e / f for e, f in zip(eta, self._normalize_factors))
        else:
            eta_normalized = eta
        
        # Use Struphy's jacobian method
        jac = self._struphy_domain.jacobian(*eta_normalized, squeeze_out=True)
        
        # Convert to numpy array if needed
        if not isinstance(jac, np.ndarray):
            jac = np.array(jac)
        
        return jac
    
    def jacobian_inv(self, *eta):
        """Compute inverse Jacobian matrix at location eta using Struphy's jacobian_inv method."""
        if self._ldim != self._pdim:
            raise ValueError("Jacobian inverse only defined for square mappings")
        
        if len(eta) != self._ldim:
            raise ValueError(f"Expected {self._ldim} coordinates, got {len(eta)}")
        
        # Normalize coordinates if factors provided
        if self._normalize_factors is not None:
            eta_normalized = tuple(e / f for e, f in zip(eta, self._normalize_factors))
        else:
            eta_normalized = eta
        
        # Use Struphy's jacobian_inv method
        jac_inv = self._struphy_domain.jacobian_inv(*eta_normalized, squeeze_out=True)
        
        # Convert to numpy array if needed
        if not isinstance(jac_inv, np.ndarray):
            jac_inv = np.array(jac_inv)
        
        return jac_inv
    
    def metric(self, *eta):
        """Compute metric tensor (G = J^T @ J) at location eta using Struphy's metric method."""
        if len(eta) != self._ldim:
            raise ValueError(f"Expected {self._ldim} coordinates, got {len(eta)}")
        
        # Normalize coordinates if factors provided
        if self._normalize_factors is not None:
            eta_normalized = tuple(e / f for e, f in zip(eta, self._normalize_factors))
        else:
            eta_normalized = eta
        
        # Use Struphy's metric method
        metric = self._struphy_domain.metric(*eta_normalized, squeeze_out=True)
        
        # Convert to numpy array if needed
        if not isinstance(metric, np.ndarray):
            metric = np.array(metric)
        
        return metric
    
    def metric_det(self, *eta):
        """Compute determinant of metric tensor at location eta."""
        metric = self.metric(*eta)
        return np.linalg.det(metric)
    
    @property
    def ldim(self):
        """Number of logical/parametric dimensions."""
        return self._ldim
    
    @property
    def pdim(self):
        """Number of physical dimensions."""
        return self._pdim
