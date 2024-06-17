from abc import ABC, ABCMeta, abstractmethod
from sympy import IndexedBase

__all__ = (
    'MappingMeta',
    'AbstractMapping',
)

class MappingMeta(ABCMeta,type(IndexedBase)):
    pass

#==============================================================================
class AbstractMapping(ABC,metaclass=MappingMeta):
    """
    Transformation of coordinates, which can be evaluated.

    F: R^l -> R^p
    F(eta) = x

    with l <= p
    """
    @abstractmethod
    def __call__(self, *args):
        """ Evaluate mapping at either a single point or the full domain. """

    @abstractmethod
    def jacobian_eval(self, *eta):
        """ Compute Jacobian matrix at location eta. """

    @abstractmethod
    def jacobian_inv_eval(self, *eta):
        """ Compute inverse Jacobian matrix at location eta.
            An exception should be raised if the matrix is singular.
        """

    @abstractmethod
    def metric_eval(self, *eta):
        """ Compute components of metric tensor at location eta. """

    @abstractmethod
    def metric_det_eval(self, *eta):
        """ Compute determinant of metric tensor at location eta. """

    @property
    @abstractmethod
    def ldim(self):
        """ Number of logical/parametric dimensions in mapping
            (= number of eta components).
        """

    @property
    @abstractmethod
    def pdim(self):
        """ Number of physical dimensions in mapping
            (= number of x components)."""