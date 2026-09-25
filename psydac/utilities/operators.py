class Laplacian:
    """
    Symbolic Laplace operator associated with a mapping F from logical
    to physical coordinates. Builds Laplacian in logical coordinates
    using the metric induced by F.

    Parameters
    ----------
    mapping : sympde.topology.mapping.Mapping
        Mapping defining the physical domain.
    """

    def __init__(self, mapping):
        from sympde.topology.mapping import Mapping

        assert isinstance(mapping, Mapping)

        self._eta = mapping.logical_coordinates
        self._metric = mapping.metric_expr
        self._metric_det = mapping.metric_det_expr

    def __call__(self, phi):
        """
        Compute the Laplacian of a symbolic scalar function.

        Parameters
        ----------
        phi : sympy expression
            Scalar function in logical coordinates.

        Returns
        -------
        sympy expression
            Laplacian of ``phi`` in the mapped physical domain.
        """

        from sympy import sqrt, Matrix

        u = self._eta
        G = self._metric
        sqrt_g = sqrt(self._metric_det)

        # Store column vector of partial derivatives of phi w.r.t. uj
        dphi_du = Matrix([phi.diff(uj) for uj in u])

        # Compute gradient of phi in tangent basis: A = G^(-1) dphi_du
        A = G.LUsolve(dphi_du)

        # Compute Laplacian of phi using formula for divergence of vector A
        lapl = sum((sqrt_g * Ai).diff(ui) for ui, Ai in zip(u, A)) / sqrt_g

        return lapl
