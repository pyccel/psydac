# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import numpy as np
import sympy as sym

from spl.mapping.basic import Mapping

__all__ = ['SymbolicMapping','AnalyticalMapping']

#==============================================================================
class SymbolicMapping:
    """ Coordinate transformation from parametric space (eta)
        to physical space (x).

        Object is purely symbolic.

    """
    def __init__( self, eta_symbols, map_expressions ):

        import sympy as sym

        self._eta = eta_symbols
        self._map = map_expressions

    #--------------------------------------------------------------------------
    def compute_derivatives( self, max_order=1 ):

        tensors    = [None]*(max_order+1)
        tensors[0] = self._map

        for i in range( max_order ):
            tensors[i+1] = sym.derive_by_array( tensors[i], self._eta )

        self._derivs_tensors = tensors

    #--------------------------------------------------------------------------
    @property
    def eta( self ):
        return sym.Array( self._eta )

    @property
    def map( self ):
        return sym.Array( self._map )

    @property
    def jac_mat( self ):
#        return self._derivs_tensors[1].tomatrix().T
        return sym.Matrix( self._map ).jacobian( self._eta )

    @property
    def metric( self ):
        jm = self.jac_mat
        return sym.simplify( jm.T * jm )

    @property
    def metric_det( self ):
        metric = self.metric
        return metric.det().simplify()

    @property
    def ndim_param( self ):
        return len( self._eta )

    @property
    def ndim_phys( self ):
        return len( self._map )

#==============================================================================
class AnalyticalMapping( Mapping ):

    def __init__( self, eta_symbols, expressions, **params ):

        # Independent variables and mapping expressions
        eta_symbols = sym.sympify( tuple( eta_symbols ) )
        expressions = sym.sympify( tuple( expressions ) )

        # Symbolic representation of coordinate transformation
        symbolic = SymbolicMapping( eta_symbols, expressions )

        # Callable function: __call__
        expr = sym.simplify( symbolic.map.subs( params ) )
        self._func_eval = sym.lambdify( [eta_symbols], expr, 'numpy' )

        # Callable function: jac_mat
        expr = sym.simplify( symbolic.jac_mat.subs( params ) )
        self._func_jac_mat = sym.lambdify( [eta_symbols], expr, 'numpy' )
    
        # Callable function: metric
        expr = sym.simplify( symbolic.metric.subs( params ) )
        self._func_metric = sym.lambdify( [eta_symbols], expr, 'numpy' )

        # Callable function: metric_det
        expr = sym.simplify( symbolic.metric_det.subs( params ) )
        self._func_metric_det = sym.lambdify( [eta_symbols], expr, 'numpy' )

        # Store symbolic information
        self._symbolic = symbolic
        self._params   = params

    #--------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    def __call__( self, eta ):
        return self._func_eval( eta )

    def jac_mat( self, eta ):
        return self._func_jac_mat( eta )

    def metric( self, eta ):
        return self._func_metric( eta )

    def metric_det( self, eta ):
        return self._func_metric_det( eta )

    @property
    def ndim_param( self ):
        return self._symbolic.ndim_param

    @property
    def ndim_phys( self ):
        return self._symbolic.ndim_phys

    #--------------------------------------------------------------------------
    # Symbolic information
    #--------------------------------------------------------------------------
    @property
    def symbolic( self ):
        return self._symbolic

    @property
    def params( self ):
        return self._params

#==============================================================================
# class AffineMap( Mapping ):
#     """ Linear transformation from parametric to physical space.
#     """
#     def __init__( self, x0, jac_mat ):
# 
#         x0 = np.asarray( x0 )
#         jm = np.asarray( jac_mat )
# 
#         # Check input data
#         assert x0.ndim == 1
#         assert jm.ndim == 2
#         assert jm.shape[0] == x0.shape[0]
#         assert jm.shape[1] >= x0.shape[0]
# 
#         # Number of physical and parametric dimensions
#         ndim_phys, ndim_param = jm.shape
# 
#         # Components of metric tensor and matrix determinant
#         metric     = np.dot( jm.T, jm )
#         metric_det = np.linalg.det( metric )
# 
#         # Store data in object
#         self._x0         = x0
#         self._jm         = jm
#         self._ndim_param = ndim_param
#         self._ndim_phys  = ndim_phys
#         self._metric     = metric
#         self._metric_det = metric_det
# 
#     #--------------------------------------------------------------------------
#     def __call__( self, eta ):
#         return self._x0 + np.dot( self._jm, eta )
# 
#     # ...
#     def jac_mat( self, eta ):
#         return self._jm
# 
#     # ...
#     def metric( self, eta ):
#         return self._metric
# 
#     # ...
#     def metric_det( self, eta ):
#         return self._metric_det
# 
#     # ...
#     @property
#     def ndim_param( self ):
#         return self._ndim_param
# 
#     # ...
#     @property
#     def ndim_phys( self ):
#         return self._ndim_phys
