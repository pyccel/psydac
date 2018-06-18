subroutine bsplvb ( t, jhigh, index, x, left, biatx )

!*****************************************************************************80
!
!! BSPLVB evaluates B-splines at a point X with a given knot sequence.
!
!  Discusion:
!
!    BSPLVB evaluates all possibly nonzero B-splines at X of order
!
!      JOUT = MAX ( JHIGH, (J+1)*(INDEX-1) )
!
!    with knot sequence T.
!
!    The recurrence relation
!
!                     X - T(I)               T(I+J+1) - X
!    B(I,J+1)(X) = ----------- * B(I,J)(X) + --------------- * B(I+1,J)(X)
!                  T(I+J)-T(I)               T(I+J+1)-T(I+1)
!
!    is used to generate B(LEFT-J:LEFT,J+1)(X) from B(LEFT-J+1:LEFT,J)(X)
!    storing the new values in BIATX over the old.
!
!    The facts that
!
!      B(I,1)(X) = 1  if  T(I) <= X < T(I+1)
!
!    and that
!
!      B(I,J)(X) = 0  unless  T(I) <= X < T(I+J)
!
!    are used.
!
!    The particular organization of the calculations follows
!    algorithm 8 in chapter X of the text.
!
!  Modified:
!
!    14 February 2007
!
!  Author:
!
!    Carl DeBoor
!
!  Reference:
!
!    Carl DeBoor,
!    A Practical Guide to Splines,
!    Springer, 2001,
!    ISBN: 0387953663.
!
!  Parameters:
!
!    Input, real(8) T(LEFT+JOUT), the knot sequence.  T is assumed to
!    be nondecreasing, and also, T(LEFT) must be strictly less than
!    T(LEFT+1).
!
!    Input, integer JHIGH, INDEX, determine the order
!    JOUT = max ( JHIGH, (J+1)*(INDEX-1) )
!    of the B-splines whose values at X are to be returned.
!    INDEX is used to avoid recalculations when several
!    columns of the triangular array of B-spline values are
!    needed, for example, in BVALUE or in BSPLVD.
!    If INDEX = 1, the calculation starts from scratch and the entire
!    triangular array of B-spline values of orders
!    1, 2, ...,JHIGH is generated order by order, that is,
!    column by column.
!    If INDEX = 2, only the B-spline values of order J+1, J+2, ..., JOUT
!    are generated, the assumption being that BIATX, J,
!    DELTAL, DELTAR are, on entry, as they were on exit
!    at the previous call.  In particular, if JHIGH = 0,
!    then JOUT = J+1, that is, just the next column of B-spline
!    values is generated.
!    Warning: the restriction  JOUT <= JMAX (= 20) is
!    imposed arbitrarily by the dimension statement for DELTAL
!    and DELTAR, but is nowhere checked for.
!
!    Input, real(8) X, the point at which the B-splines
!    are to be evaluated.
!
!    Input, integer LEFT, an integer chosen so that
!    T(LEFT) <= X <= T(LEFT+1).
!
!    Output, real(8) BIATX(JOUT), with BIATX(I) containing the
!    value at X of the polynomial of order JOUT which agrees
!    with the B-spline B(LEFT-JOUT+I,JOUT,T) on the interval
!    (T(LEFT),T(LEFT+1)).
!
  implicit none

  integer, parameter :: jmax = 20

  integer jhigh

  real(8) biatx(jhigh)
  real(8), save, dimension ( jmax ) :: deltal
  real(8), save, dimension ( jmax ) :: deltar
!  real(8), dimension ( jmax ) :: deltal
!  real(8), dimension ( jmax ) :: deltar
  integer i
  integer index
  integer, save :: j = 1
  integer left
  real(8) saved
  real(8) t(left+jhigh)
  real(8) term
  real(8) x

  if ( index == 1 ) then
    j = 1
    biatx(1) = 1.0
    if ( jhigh <= j ) then
      return
    end if
  end if

  if ( t(left+1) <= t(left) ) then
    print*,'x=',x
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'BSPLVB - Fatal error!'
    write ( *, '(a)' ) '  It is required that T(LEFT) < T(LEFT+1).'
    write ( *, '(a,i8)' ) '  But LEFT = ', left
    write ( *, '(a,g14.6)' ) '  T(LEFT) =   ', t(left)
    write ( *, '(a,g14.6)' ) '  T(LEFT+1) = ', t(left+1)
    stop
  end if

  do

    deltar(j) = t(left+j) - x
    deltal(j) = x - t(left+1-j)

    saved = 0.0
    do i = 1, j
      term = biatx(i) / ( deltar(i) + deltal(j+1-i) )
      biatx(i) = saved + deltar(i) * term
      saved = deltal(j+1-i) * term
    end do

    biatx(j+1) = saved
    j = j + 1

    if ( jhigh <= j ) then
      exit
    end if

  end do

  return
end subroutine bsplvb
