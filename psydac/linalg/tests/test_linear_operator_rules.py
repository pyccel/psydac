from psydac.linalg.basic   import IdentityOperator, SumLinearOperator, ScaledLinearOperator, ComposedLinearOperator
#from psydac.linalg.block   import BlockVectorSpace
from psydac.linalg.stencil import StencilVectorSpace, StencilMatrix
from psydac.api.settings   import PSYDAC_BACKEND_GPYCCEL

from .test_linalg import get_StencilVectorSpace


def get_Hcurl_mass_matrix_2d(nc=5, comm=None):

    domain = Square()
    derham = Derham(domain, sequence=['h1', 'hcurl', 'l2'])

    domain_h = discretize(domain, ncells=[nc, nc], periodic=[False, False], comm=comm)
    derham_h = discretize(derham, domain_h, degree=[2, 2])

    u, v = elements_of(derham.V1, names='u, v')
    m1 = BilinearForm((u, v), integral(domain, dot(u, v)))
    M1 = discretize(m1, domain_h, (derham_h.V1, derham_h.V1), backend=PSYDAC_BACKEND_GPYCCEL).assemble()

    return M1


def test_types_and_refs():

  V = get_StencilVectorSpace(10, 5, p1=3, p2=2, P1=True, P2=False)

  I = IdentityOperator(V)  # Immutable
  S = StencilMatrix(V, V)  # Mutable
  P = StencilMatrix(V, V)  # Mutable

  # TODO: set some entries of S, P to non-zero values

  # Example 1
  # ---------

  # Create simple sum
  M = I + S
  a = M
  # New object, with references to [I, S]
  # Type is SumLinearOperator
  # If S is changed, so is M
  assert isinstance(M, SumLinearOperator)

  # += does not modify the object, but creates a new one
  M += 2*P  # M := I + S + 2*P = a + 2*P
  b = M
  # New object, with references to the same atoms [I, S, P]
  assert isinstance(M, SumLinearOperator)
  assert M is not a

  # Store reference to M1
  M *= 2   # M := 2 * (I + S + 2*P) = 2 * b
  # New object, with references to the same atoms [I, S, P]
  assert isinstance(M, ScaledLinearOperator)
  assert M is not b

  # Think about this one... are we OK with creating a new StencilMatrix?
  W = S + 3 * S + P
  assert isinstance(W, StencilMatrix)  # currently...

  X = S @ S
  assert isinstance(X, ComposedLinearOperator)

  # Example 2
  # ---------

  M1 = get_Hcurl_mass_matrix_2d()
  V1 = M1.domain
  A = BlockLinearOperator(V1, V1, ((None, None), (None, M1[1, 1])))
  B = BlockLinearOperator(V1, V1, ((M1[0, 0] + IdentityOperator(V1[0]), None), (None, None)))

  # Sum: should we get a SumLinearOperator or a BlockLinearOperator?
  C = A + B
  r1 = C
  assert isinstance(C, BlockLinearOperator)  # currently...

  # In-place multiplication
  C *= 5  # We want a new object here!
  assert C is not r1
  assert isinstance(C, BlockLinearOperator)  # debatable
