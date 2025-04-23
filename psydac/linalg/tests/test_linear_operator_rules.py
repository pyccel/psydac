from psydac.linalg.basic   import IdentityOperator, SumLinearOperator, ScaledLinearOperator, ComposedLinearOperator
#from psydac.linalg.block   import BlockVectorSpace
from psydac.linalg.stencil import StencilVectorSpace, StencilMatrix

from .test_linalg import get_StencilVectorSpace


def test_types_and_refs():

  V = get_StencilVectorSpace(10, 5, p1=3, p2=2, P1=True, P2=False)

  I = IdentityOperator(V)  # Immutable
  S = StencilMatrix(V, V)  # Mutable
  P = StencilMatrix(V, V)  # Mutable

  # TODO: set some entries of S, P to non-zero values

  # Example 1
  # ---------

  # Create simple sum
  M1 = I + S
  a = M1
  # New object, with references to [I, S]
  # Type is SumLinearOperator
  # If S is changed, so is M1
  assert isinstance(M1, SumLinearOperator)

  # += does not modify the object, but creates a new one
  M1 += 2*P  # M1 := I + S + 2*P = a + 2*P
  b = M1
  # New object, with references to the same atoms [I, S, P]
  assert isinstance(M1, SumLinearOperator)
  assert M1 is not a

  # Store reference to M1
  M1 *= 2   # M1 := 2 * (I + S + 2*P) = 2 * b
  # New object, with references to the same atoms [I, S, P]
  assert isinstance(M1, ScaledLinearOperator)
  assert M1 is not b

  # Think about this one... are we OK with creating a new StencilMatrix?
  M2 = S + 3 * S + P
  assert isinstance(M2, StencilMatrix)  # ?

  M3 = S @ S
  assert isinstance(M3, ComposedLinearOperator)

  # Example 2
  # ---------

  
  
