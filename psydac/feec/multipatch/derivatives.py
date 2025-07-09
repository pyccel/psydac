import os
import numpy as np

from psydac.linalg.block     import BlockLinearOperator
from psydac.feec.derivatives import Gradient_2D, ScalarCurl_2D
from psydac.fem.basic        import FemLinearOperator

class BrokenGradient_2D(FemLinearOperator):

    def __init__(self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V0h, fem_codomain=V1h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._linop = BlockLinearOperator(self.linop_domain, self.linop_codomain, blocks={
                                           (i, i): D0i.linop for i, D0i in enumerate(D0s)})

    def transpose(self, conjugate=False):
        # todo (MCP): define as the dual differential operator
        return BrokenTransposedGradient_2D(self.fem_domain, self.fem_codomain)

# ==============================================================================


class BrokenTransposedGradient_2D(FemLinearOperator):

    def __init__(self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V0h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._linop = BlockLinearOperator(self.linop_domain, self.linop_codomain, blocks={
                                           (i, i): D0i.linop.T for i, D0i in enumerate(D0s)})

    def transpose(self, conjugate=False):
        # todo (MCP): discard
        return BrokenGradient_2D(self.fem_codomain, self.fem_domain)


# ==============================================================================
class BrokenScalarCurl_2D(FemLinearOperator):
    def __init__(self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V2h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._linop = BlockLinearOperator(self.linop_domain, self.linop_codomain, blocks={
                                           (i, i): D1i.linop for i, D1i in enumerate(D1s)})

    def transpose(self, conjugate=False):
        return BrokenTransposedScalarCurl_2D(
            V1h=self.fem_domain, V2h=self.fem_codomain)


# ==============================================================================
class BrokenTransposedScalarCurl_2D(FemLinearOperator):

    def __init__(self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V2h, fem_codomain=V1h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._linop = BlockLinearOperator(self.linop_domain, self.linop_codomain, blocks={
                                           (i, i): D1i.linop.T for i, D1i in enumerate(D1s)})

    def transpose(self, conjugate=False):
        return BrokenScalarCurl_2D(V1h=self.fem_codomain, V2h=self.fem_domain)

# ==============================================================================
