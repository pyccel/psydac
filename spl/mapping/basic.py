# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from abc import ABCMeta, abstractmethod

__all__ = ['Mapping']

#==============================================================================
class Mapping( metaclass=ABCMeta ):
    """
    Transformation of coordinates

    F: R^p -> R^s
    F(eta) = x

    with p <= s

    """
    @abstractmethod
    def __call__( self, eta ):
        """ Evaluate mapping at location eta. """

    @abstractmethod
    def jac_mat( self, eta ):
        """ Compute Jacobian matrix at location eta. """

    @abstractmethod
    def metric( self, eta ):
        """ Compute components of metric tensor at location eta. """

    @abstractmethod
    def metric_det( self, eta ):
        """ Compute determinant of metric tensor at location eta. """

    @property
    @abstractmethod
    def ldim( self ):
        """ Number of logical/parametric dimensions in mapping
            (= number of eta components).
        """

    @property
    @abstractmethod
    def pdim( self ):
        """ Number of physical dimensions in mapping
            (= number of x components).
        """
