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

  # 
  M1 += 2*P  # M1 := I + S + 2*P
  b = M1
  # Same object, with references to [I, S, P]
  assert isinstance(M1, SumLinearOperator)
  assert a is b

  # Store reference to M1
  M1 *= 2 # -> 2 * (I + S + 2*P)
  # New object, with references to [I, S, P]
  # Not the same object as b := I + S + 2*P
  assert isinstance(M1, ScaledLinearOperator)
  assert b is not M1

  # Think about this one...
  M2 = S + S
  assert isinstance(M2, StencilMatrix)

  M3 = S @ S
  assert isinstance(M3, ComposedLinearOperator)
