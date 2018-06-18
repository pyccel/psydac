! -*- coding: UTF-8 -*-

!> @brief 
!> Module for Splines (use core.external modules) 
!> @details
!> basic functions and routines for B-Splines
!> in 1D, 2D and 3D

! .......................................................
module bsp_ext
contains

  ! .......................................................
  !> @brief     Determine non zero elements 
  !>
  !> @param[in] n number of control points 
  !> @param[in] p spline degree 
  !> @param[in] U Knot vector 
  !> @param[in] n_elements number of non-zero elements 
  !> @param[in] grid the corresponding grid
  subroutine FindNonZeroElements(p,m,U,n_elements,grid)
    use bspline, Find => FindNonZeroElements_bspline
    implicit none
    integer(kind=4), intent(in)  :: p, m
    real   (kind=8), intent(in)  :: U(0:m)
    integer(kind=4), intent(inout) :: n_elements
    real   (kind=8), intent(inout) :: grid(0:m)

    call Find(m-(p+1),p,U,n_elements,grid) 
  end subroutine FindNonZeroElements
  ! .......................................................

  ! .......................................................
  !> @brief     Determine the knot span index 
  !>
  !> @param[in] n number of control points 
  !> @param[in] p spline degree 
  !> @param[in] U Knot vector 
  !> @param[in] uu given knot 
  !> @param[out] span the span index 
  subroutine FindSpan(p,m,U,uu,span)
    use bspline, FindS => FindSpan
    implicit none
    integer(kind=4), intent(in)  :: p, m
    real   (kind=8), intent(in)  :: U(0:m), uu
    integer(kind=4), intent(out) :: span
    span = FindS(m-(p+1),p,uu,U)
  end subroutine FindSpan
  ! .......................................................

  ! .......................................................
  !> @brief     Determine the multiplicity of a given knot starting from a span
  !>
  !> @param[in]    p     spline degree 
  !> @param[in]    m     number of control points - 1
  !> @param[in]    U     Knot vector 
  !> @param[in]    uu    given knot 
  !> @param[inout] span  the span index 
  !> @param[out]   mult  multiplicity of the given knot
  subroutine FindMult(p,m,U,uu,span,mult)
    use bspline, FindM => FindMult
    implicit none
    integer(kind=4), intent(in)  :: p, m
    real   (kind=8), intent(in)  :: U(0:m), uu
    integer(kind=4), intent(inout)  :: span
    integer(kind=4), intent(out) :: mult

    if (span < 0) then
       span = FindSpan(m-(p+1),p,uu,U)
    end if
    mult = FindM(span,uu,p,U)
  end subroutine FindMult
  ! .......................................................

  ! .......................................................
  !> @brief     Determine the multiplicity of a given knot
  !>
  !> @param[in]    p     spline degree 
  !> @param[in]    m     number of control points - 1
  !> @param[in]    U     Knot vector 
  !> @param[in]    uu    given knot 
  !> @param[out]   mult  multiplicity of the given knot
  subroutine FindSpanMult(p,m,U,uu,k,s)
    use bspline, FindSM => FindSpanMult
    implicit none
    integer(kind=4), intent(in)  :: p, m
    real   (kind=8), intent(in)  :: U(0:m), uu
    integer(kind=4), intent(out) :: k, s
    call FindSM(m-(p+1),p,uu,U,k,s)
  end subroutine FindSpanMult
  ! .......................................................

  ! .......................................................
  !> @brief     evaluates all b-splines at a given site 
  !>
  !> @param[in]    p     spline degree 
  !> @param[in]    m     number of control points - 1
  !> @param[in]    U     Knot vector 
  !> @param[in]    uu    given knot 
  !> @param[inout] span  the span index 
  !> @param[out]   N     the p+1 non vanishing b-splines at uu 
  subroutine EvalBasisFuns(p,m,U,uu,span,N)
    use bspline
    implicit none
    integer(kind=4), intent(in) :: p, m
    integer(kind=4), intent(inout) :: span
    real   (kind=8), intent(in) :: U(0:m), uu
    real   (kind=8), intent(out):: N(0:p)

    if (span < 0) then
       span = FindSpan(m-(p+1),p,uu,U)
    end if
    call BasisFuns(span,uu,p,U,N)
  end subroutine EvalBasisFuns
  ! .......................................................

  ! .......................................................
  !> @brief     evaluates all b-splines and their derivatives at a given site 
  !>
  !> @param[in]    p     spline degree 
  !> @param[in]    m     number of control points - 1
  !> @param[in]    U     Knot vector 
  !> @param[in]    uu    given knot 
  !> @param[inout] span  the span index 
  !> @param[out]   dN    the p+1 non vanishing b-splines and their derivatives at uu 
  subroutine EvalBasisFunsDers(p,m,U,uu,d,span,dN)
    use bspline
    implicit none
    integer(kind=4), intent(in) :: p, m, d
    integer(kind=4), intent(inout) :: span
    real   (kind=8), intent(in) :: U(0:m), uu
    real   (kind=8), intent(out):: dN(0:p,0:d)

    if (span < 0) then
       span = FindSpan(m-(p+1),p,uu,U)
    end if
    call DersBasisFuns(span,uu,p,d,U,dN)
  end subroutine EvalBasisFunsDers
  ! .......................................................

  ! .......................................................
  !> @brief     evaluates all b-splines and their derivatives at given sites
  !>
  !> @param[in]    p     spline degree 
  !> @param[in]    m     number of control points - 1
  !> @param[in]    d     number of derivatives
  !> @param[in]    r     size of tau
  !> @param[in]    U     Knot vector 
  !> @param[in]    tau   given knot 
  !> @param[out]   dN    the p+1 non vanishing b-splines and their derivatives at uu 
  subroutine spl_eval_splines_ders(p,m,d,r,U,tau,dN)
    use bspline
    implicit none
    integer(kind=4), intent(in) :: p, m, d, r
    real   (kind=8), intent(in) :: U(0:m), tau(0:r)
    real   (kind=8), intent(out):: dN(0:p,0:d,0:r)
    ! local
    integer(kind=4) :: span
    integer(kind=4) :: i

    do i = 0, r
      span = -1
      call EvalBasisFunsDers(p,m,U,tau(i),d,span,dN(0:p,0:d,i))
    end do
  end subroutine spl_eval_splines_ders
  ! .......................................................

  ! .......................................................
  !> @brief     evaluates all b-splines and their derivatives at given sites 
  !>
  !> @param[in]    p           spline degree 
  !> @param[in]    m           number of control points - 1
  !> @param[in]    U           Knot vector 
  !> @param[in]    uu          given sites, contained in the same element 
  !> @param[in]    n_points    size of uu 
  !> @param[inout] span        the span index 
  !> @param[out]   dN          the p+1 non vanishing b-splines and their derivatives at uu 
  subroutine EvalBasisFunsDers_array(p,m,U,uu,n_points,d,span,dN)
    use bspline
    implicit none
    integer(kind=4), intent(in) :: p, m, d, n_points
    integer(kind=4), intent(inout) :: span
    real   (kind=8), intent(in) :: U(0:m), uu(1:n_points)
    real   (kind=8), intent(out):: dN(0:p,0:d,1:n_points)

    if (span < 0) then
       span = FindSpan(m-(p+1),p,uu(1),U)
    end if
    call DersBasisFuns_array(span,uu,n_points,p,d,U,dN)
  end subroutine EvalBasisFunsDers_array 
  ! .......................................................

  ! .......................................................
  !> @brief     Determine the span indices for every knot 
  !>
  !> @param[in]    p     spline degree 
  !> @param[in]    m     number of control points - 1
  !> @param[in]    U     Knot vector 
  !> @param[inout] r     maximum number of knots 
  !> @param[out]   I     span for every knot 
  subroutine SpanIndex(p,m,U,r,I)
    integer(kind=4), intent(in)  :: p, m
    real   (kind=8), intent(in)  :: U(0:m)
    integer(kind=4), intent(in)  :: r
    integer(kind=4), intent(out) :: I(r)
    integer(kind=4) :: k, s
    s = 1
    do k = p, m-(p+1)
       if (U(k) /= U(k+1)) then
          I(s) = k; s = s + 1
          if (s > r) exit
       end if
    end do
  end subroutine SpanIndex
  ! .......................................................

  ! .......................................................
  !> @brief     returns the Greville abscissae 
  !>
  !> @param[in]    p     spline degree 
  !> @param[in]    m     number of control points - 1
  !> @param[in]    U     Knot vector 
  !> @param[out]   X     Greville abscissae 
  subroutine Greville(p,m,U,X)
    implicit none
    integer(kind=4), intent(in)  :: p, m
    real   (kind=8), intent(in)  :: U(0:m)
    real   (kind=8), intent(out) :: X(0:m-(p+1))
    integer(kind=4) :: i
    do i = 0, m-(p+1)
       X(i) = sum(U(i+1:i+p)) / p
    end do
  end subroutine Greville
  ! .......................................................

  ! .......................................................
  !> @brief     inserts the knot uu r times 
  !>
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] n number of control points 
  !> @param[in] p spline degree 
  !> @param[in] U Initial Knot vector 
  !> @param[in] Pw Initial Control points  
  !> @param[in] uu knot to insert 
  !> @param[in] r number of times uu will be inserted
  !> @param[in] V Final Knot vector 
  !> @param[in] Qw Final Control points  
  subroutine InsertKnot(d,n,p,U,Pw,uu,r,V,Qw)
    use bspline, InsKnt => InsertKnot
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    real   (kind=8), intent(in)  :: uu
    integer(kind=4), intent(in)  :: r
    real   (kind=8), intent(out) :: V(0:n+p+1+r)
    real   (kind=8), intent(out) :: Qw(d,0:n+r)
    integer(kind=4) :: k, s
    if (r == 0) then
       V = U; Qw = Pw; return
    end if
    call FindSpanMult(n,p,uu,U,k,s)
    call InsKnt(d,n,p,U,Pw,uu,k,s,r,V,Qw)
  end subroutine InsertKnot
  ! .......................................................

  ! .......................................................
  !> @brief     removes a knot from a B-Splines curve, given a tolerance 
  !>
  !> @param[in]    d      dimension of the manifold 
  !> @param[in]    n      number of control points  - 1
  !> @param[in]    p      spline degree 
  !> @param[in]    U      Knot vector 
  !> @param[in]    Pw     weighted control points 
  !> @param[in]    uu     knot to remove 
  !> @param[in]    r      maximum number of iterations 
  !> @param[out]   t      requiered number of iterations 
  !> @param[out]   V      new Knot vector 
  !> @param[out]   Qw     new control points 
  !> @param[in]    TOL    tolerance for the distance to the control point 
  subroutine RemoveKnot(d,n,p,U,Pw,uu,r,t,V,Qw,TOL)
    use bspline, RemKnt => RemoveKnot
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    real   (kind=8), intent(in)  :: uu
    integer(kind=4), intent(in)  :: r
    integer(kind=4), intent(out) :: t
    real   (kind=8), intent(out) :: V(0:n+p+1)
    real   (kind=8), intent(out) :: Qw(d,0:n)
    real   (kind=8), intent(in)  :: TOL
    integer(kind=4) :: k, s
    t = 0
    V = U
    Qw = Pw
    if (r == 0) return
    if (uu <= U(p)) return
    if (uu >= U(n+1)) return
    call FindSpanMult(n,p,uu,U,k,s)
    call RemKnt(d,n,p,V,Qw,uu,k,s,r,t,TOL)
  end subroutine RemoveKnot
  ! .......................................................

  ! .......................................................
  !> @brief    clampes a B-spline curve 
  !>
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] n number of control points 
  !> @param[in] p spline degree 
  !> @param[in] U Initial Knot vector 
  !> @param[in] Pw Initial Control points  
  !> @param[in] l apply the algorithm on the left 
  !> @param[in] r apply the algorithm on the right 
  !> @param[in] V Final Knot vector 
  !> @param[in] Qw Final Control points  
  subroutine Clamp(d,n,p,U,Pw,l,r,V,Qw)
    use bspline
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    logical(kind=4), intent(in)  :: l, r
    real   (kind=8), intent(out) :: V(0:n+p+1)
    real   (kind=8), intent(out) :: Qw(d,0:n)
    V  = U
    Qw = Pw
    call ClampKnot(d,n,p,V,Qw,l,r)
  end subroutine Clamp
  ! .......................................................

  ! .......................................................
  !> @brief    unclampes a B-spline curve 
  !>
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] n number of control points 
  !> @param[in] p spline degree 
  !> @param[in] U Initial Knot vector 
  !> @param[in] Pw Initial Control points  
  !> @param[in] l apply the algorithm on the left 
  !> @param[in] r apply the algorithm on the right 
  !> @param[in] V Final Knot vector 
  !> @param[in] Qw Final Control points  
  subroutine Unclamp(d,n,p,U,Pw,l,r,V,Qw)
    use bspline
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    logical(kind=4), intent(in)  :: l, r
    real   (kind=8), intent(out) :: V(0:n+p+1)
    real   (kind=8), intent(out) :: Qw(d,0:n)
    V  = U
    Qw = Pw
    call UnclampKnot(d,n,p,V,Qw,l,r)
  end subroutine Unclamp
  ! .......................................................

  ! .......................................................
  !> @brief     inserts all elements of X into the knot vector 
  !>
  !> @param[in] d     manifold dimension for the control points  
  !> @param[in] n     number of control points 
  !> @param[in] p     spline degree 
  !> @param[in] U     Initial Knot vector 
  !> @param[in] Pw    Initial Control points  
  !> @param[in] X     knots to insert 
  !> @param[in] r     size of X 
  !> @param[in] Ubar  Final Knot vector 
  !> @param[in] Qw    Final Control points  
  subroutine RefineKnotVector(d,n,p,U,Pw,r,X,Ubar,Qw)
    use bspline, RefKnt => RefineKnotVector
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    integer(kind=4), intent(in)  :: r
    real   (kind=8), intent(in)  :: X(0:r)
    real   (kind=8), intent(out) :: Ubar(0:n+r+1+p+1)
    real   (kind=8), intent(out) :: Qw(d,0:n+r+1)
    call RefKnt(d,n,p,U,Pw,r,X,Ubar,Qw)
  end subroutine RefineKnotVector
  ! .......................................................

  ! .......................................................
  !> @brief     elevate the spline degree by t 
  !>
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] n number of control points 
  !> @param[in] p spline degree 
  !> @param[in] U Initial Knot vector 
  !> @param[in] Pw Initial Control points  
  !> @param[in] t number of degree elevation 
  !> @param[in] nh equal to n + t *nrb_internal_knots 
  !> @param[in] Uh Final Knot vector 
  !> @param[in] Qw Final Control points  
  subroutine DegreeElevate(d,n,p,U,Pw,t,nh,Uh,Qw)
    use bspline, DegElev => DegreeElevate
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    integer(kind=4), intent(in)  :: t
    integer(kind=4), intent(in)  :: nh
    real   (kind=8), intent(out) :: Uh(0:nh+p+t+1)
    real   (kind=8), intent(out) :: Qw(d,0:nh)
    call DegElev(d,n,p,U,Pw,t,nh,Uh,Qw)
  end subroutine DegreeElevate
  ! .......................................................

  ! .......................................................
  !> @brief     extracts a B-Spline curve at the knot x 
  !>
  !> @param[in]    d             dimension of the manifold 
  !> @param[in]    n             number of control points  - 1
  !> @param[in]    p             spline degree 
  !> @param[in]    U             Knot vector 
  !> @param[in]    Pw            weighted control points 
  !> @param[in]    x             knot to evaluate at 
  !> @param[inout] Cw            the point on the curve 
  subroutine Extract(d,n,p,U,Pw,x,Cw)
    use bspline, CornerCut => CurvePntByCornerCut
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    real   (kind=8), intent(in)  :: x
    real   (kind=8), intent(out) :: Cw(d)
    call CornerCut(d,n,p,U,Pw,x,Cw)
  end subroutine Extract
  ! .......................................................

  ! .......................................................
  !> @brief     elevate the spline at X 
  !>
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] n number of control points 
  !> @param[in] p spline degree 
  !> @param[in] U Initial Knot vector 
  !> @param[in] Q Initial Control points  
  !> @param[in] r dimension of X - 1 
  !> @param[in] X the positions on wich evaluation is done  
  !> @param[out] Cw Values  
  subroutine Evaluate1(d,n,p,U,Q,weights,r,X,Cw)
    use bspline
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Q(d,0:n)
    real   (kind=8), intent(in)  :: weights(0:n)
    integer(kind=4), intent(in)  :: r
    real   (kind=8), intent(in)  :: X(0:r)
    real   (kind=8), intent(out) :: Cw(d,0:r)
    integer(kind=4) :: i, j, span
    real   (kind=8) :: basis(0:p), C(d)
    real   (kind=8) :: w 
    !
    do i = 0, r
       span = FindSpan(n,p,X(i),U)
       call BasisFuns(span,X(i),p,U,basis)
       !
       ! compute w = sum wi Ni
       w  = 0.0
       do j = 0, p
          w  = w  + basis(j) * weights(span-p+j)
       end do

       !
       C = 0.0
       do j = 0, p
          C = C + basis(j) * weights(span-p+j) * Q(:,span-p+j)
       end do
       Cw(:,i) = C / w
       !
    end do
    !
  end subroutine Evaluate1
  ! .......................................................

  ! .......................................................
  !> @brief     elevate the spline at X, works with M-splines too 
  !>
  !> @param[in] normalize use M-Splines in the 1st direction  
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] n number of control points 
  !> @param[in] p spline degree 
  !> @param[in] U Initial Knot vector 
  !> @param[in] Q Initial Control points  
  !> @param[in] r dimension of X - 1 
  !> @param[in] X the positions on wich evaluation is done  
  !> @param[out] Cw Values  
  subroutine EvaluateNormal1(normalize,d,n,p,U,Q,weights,r,X,Cw)
    use bspline
    implicit none
    logical,         intent(in)  :: normalize
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Q(d,0:n)
    real   (kind=8), intent(in)  :: weights(0:n)
    integer(kind=4), intent(in)  :: r
    real   (kind=8), intent(in)  :: X(0:r)
    real   (kind=8), intent(out) :: Cw(d,0:r)
    integer(kind=4) :: i, j, span, o
    real   (kind=8) :: basis(0:p), C(d)
    real   (kind=8) :: w 
    real   (kind=8) :: x_scale 
    !
    do i = 0, r
       span = FindSpan(n,p,X(i),U)
       call BasisFuns(span,X(i),p,U,basis)
       !

       if (normalize) then
          o = span - p 
          do j = 0, p 
            x_scale =   ( p + 1) &
                    & / ( U(o+j + p + 1) &
                    &   - U(o+j) )
           
            basis(j) = basis(j) * x_scale
          end do
       end if

       !
       C = 0.0
       do j = 0, p
          C = C + basis(j) * weights(span-p+j) * Q(:,span-p+j)
       end do
       Cw(:,i) = C 
       !
    end do
    !
  end subroutine EvaluateNormal1
  ! .......................................................

  ! .......................................................
  !> @brief     elevate the spline at X 
  !>
  !> @param[in] nderiv number of derivatives 
  !> @param[in] N corresponding number of partial derivatives
  !> @param[in] rationalize true if rational B-Splines are to be used
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] nx number of control points 
  !> @param[in] px spline degree 
  !> @param[in] U Initial Knot vector 
  !> @param[in] Q Initial Control points  
  !> @param[in] rx dimension of X - 1 
  !> @param[in] X the positions on wich evaluation is done  
  !> @param[out] Cw Values  
  subroutine EvaluateDeriv1(nderiv,N,d,nx,px,U,Q,weights,rx,X,Cw)
    use bspline
    implicit none
    integer(kind=4), intent(in)  :: nderiv
    integer(kind=4), intent(in)  :: N   
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: nx, px
    real   (kind=8), intent(in)  :: U(0:nx+px+1)
    real   (kind=8), intent(in)  :: Q(d,0:nx)
    real   (kind=8), intent(in)  :: weights(0:nx)
    integer(kind=4), intent(in)  :: rx
    real   (kind=8), intent(in)  :: X(0:rx)
    real   (kind=8), intent(out) :: Cw(0:N,d,0:rx)
    integer(kind=4) :: i, j, span, deriv
    real   (kind=8) :: dbasis(0:px,0:nderiv), C(d), w(0:nderiv)
    real   (kind=8) :: Rdbasis(0:px,0:nderiv)
    real   (kind=8) :: basis(0:px)
    !

    do i = 0, rx
       span = FindSpan(nx,px,X(i),U)
       call DersBasisFuns(span,X(i),px,nderiv,U,dbasis)

       !
       ! compute w = sum wi Ni
       ! and w' = sum wi Ni'
       w  = 0.0
       do j = 0, px
          w  = w  + dbasis(j,:) * weights(span-px+j)
       end do
       ! compute Nurbs
       Rdbasis  = 0.0
       Rdbasis(:,0) = dbasis(:,0) / w(0) 

       if (nderiv >= 1) then
          Rdbasis(:,1) = dbasis(:,1) / w(0) - dbasis(:,0) * w(1) / w(0)**2   
       end if

       if (nderiv>=2) then
          Rdbasis(:,2) = dbasis(:,2) / w(0)               &
                     & - 2 * dbasis(:,1) * w(1) / w(0)**2 &
                     & - dbasis(:,0) * w(2) / w(0)**2     &
                     & + 2 * dbasis(:,0) * w(1)**2 / w(0)**3
       end if

       do deriv = 0, N
          C = 0.0
          do j = 0, px
             C = C + Rdbasis(j,deriv) * weights(span-px+j) * Q(:,span-px+j)
          end do
          Cw(deriv,:,i) = C
       end do
       !
    end do
    !
  end subroutine EvaluateDeriv1
  ! .......................................................

  ! .......................................................
  !> @brief     elevate the spline at X 
  !>
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] nx number of control points 
  !> @param[in] px spline degree 
  !> @param[in] Ux Initial Knot vector 
  !> @param[in] ny number of control points 
  !> @param[in] py spline degree 
  !> @param[in] Uy Initial Knot vector 
  !> @param[in] Q Initial Control points  
  !> @param[in] rx dimension of X - 1 
  !> @param[in] X the positions on wich evaluation is done  
  !> @param[in] ry dimension of Y - 1 
  !> @param[in] Y the positions on wich evaluation is done  
  !> @param[out] Cw Values  
  subroutine Evaluate2(d,nx,px,Ux,ny,py,Uy,Q,weights,rx,X,ry,Y,Cw)
    use bspline
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: nx, ny
    integer(kind=4), intent(in)  :: px, py
    integer(kind=4), intent(in)  :: rx, ry
    real   (kind=8), intent(in)  :: Ux(0:nx+px+1)
    real   (kind=8), intent(in)  :: Uy(0:ny+py+1)
    real   (kind=8), intent(in)  :: Q(d,0:nx,0:ny)
    real   (kind=8), intent(in)  :: weights(0:nx,0:ny)
    real   (kind=8), intent(in)  :: X(0:rx), Y(0:ry)
    real   (kind=8), intent(out) :: Cw(d,0:rx,0:ry)
    integer(kind=4) :: ix, jx, iy, jy, ox, oy
    integer(kind=4) :: spanx(0:rx), spany(0:ry)
    real   (kind=8) :: basisx(0:px,0:rx), basisy(0:py,0:ry)
    real   (kind=8) :: M, C(d)
    real   (kind=8) :: w 

    !
    do ix = 0, rx
       spanx(ix) = FindSpan(nx,px,X(ix),Ux)
       call BasisFuns(spanx(ix),X(ix),px,Ux,basisx(:,ix))
    end do
    do iy = 0, ry
       spany(iy) = FindSpan(ny,py,Y(iy),Uy)
       call BasisFuns(spany(iy),Y(iy),py,Uy,basisy(:,iy))
    end do
    !
    do iy = 0, ry
      oy = spany(iy) - py
      do ix = 0, rx
        ox = spanx(ix) - px
        ! ---
        w = 0.0
        do jy = 0, py
          do jx = 0, px
             M = basisx(jx,ix) * basisy(jy,iy)
             w = w + M * weights(ox+jx,oy+jy)
          end do
        end do
        ! ---
        ! ---
        C = 0.0
        do jy = 0, py
          do jx = 0, px
             M = basisx(jx,ix) * basisy(jy,iy)
             C = C + M * weights(ox+jx,oy+jy) * Q(:,ox+jx,oy+jy)
          end do
        end do
        Cw(:,ix,iy) = C / w
        ! ---
      end do
    end do
    !
  end subroutine Evaluate2
  ! .......................................................

  ! .......................................................
  !> @brief     elevate the spline at X, works with M-splines too 
  !>
  !> @param[in] normalize_x use M-Splines in the 1st direction  
  !> @param[in] normalize_y use M-Splines in the 2nd direction  
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] nx number of control points 
  !> @param[in] px spline degree 
  !> @param[in] Ux Initial Knot vector 
  !> @param[in] ny number of control points 
  !> @param[in] py spline degree 
  !> @param[in] Uy Initial Knot vector 
  !> @param[in] Q Initial Control points  
  !> @param[in] rx dimension of X - 1 
  !> @param[in] X the positions on wich evaluation is done  
  !> @param[in] ry dimension of Y - 1 
  !> @param[in] Y the positions on wich evaluation is done  
  !> @param[out] Cw Values  
  subroutine EvaluateNormal2( normalize_x,normalize_y,&
                            & d,nx,px,Ux,ny,py,Uy,Q,weights,rx,X,ry,Y,Cw)
    use bspline
    implicit none
    logical,         intent(in)  :: normalize_x
    logical,         intent(in)  :: normalize_y
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: nx, ny
    integer(kind=4), intent(in)  :: px, py
    integer(kind=4), intent(in)  :: rx, ry
    real   (kind=8), intent(in)  :: Ux(0:nx+px+1)
    real   (kind=8), intent(in)  :: Uy(0:ny+py+1)
    real   (kind=8), intent(in)  :: Q(d,0:nx,0:ny)
    real   (kind=8), intent(in)  :: weights(0:nx,0:ny)
    real   (kind=8), intent(in)  :: X(0:rx), Y(0:ry)
    real   (kind=8), intent(out) :: Cw(d,0:rx,0:ry)
    integer(kind=4) :: ix, jx, iy, jy, ox, oy
    integer(kind=4) :: spanx(0:rx), spany(0:ry)
    real   (kind=8) :: basisx(0:px,0:rx), basisy(0:py,0:ry)
    real   (kind=8) :: M, C(d)
    real   (kind=8) :: w 
    real   (kind=8) :: x_scale 
    real   (kind=8) :: y_scale 

    !
    do ix = 0, rx
       spanx(ix) = FindSpan(nx,px,X(ix),Ux)
       call BasisFuns(spanx(ix),X(ix),px,Ux,basisx(:,ix))

       if (normalize_x) then
          ox = spanx(ix) - px 
          do jx = 0, px 
            x_scale =   ( px + 1) &
                    & / ( Ux(ox+jx + px + 1) &
                    &   - Ux(ox+jx) )
           
            basisx(jx,ix) = basisx(jx,ix) * x_scale
          end do
       end if
    end do
    do iy = 0, ry
       spany(iy) = FindSpan(ny,py,Y(iy),Uy)
       call BasisFuns(spany(iy),Y(iy),py,Uy,basisy(:,iy))

       if (normalize_y) then
          oy = spany(iy) - py 
          do jy = 0, py 
            y_scale =   ( py + 1) &
                    & / ( Uy(oy+jy + py + 1) &
                    &   - Uy(oy+jy) )
           
            basisy(jy,iy) = basisy(jy,iy) * y_scale
          end do
       end if
    end do
    !
    do iy = 0, ry
      oy = spany(iy) - py
      do ix = 0, rx
        ox = spanx(ix) - px
        ! ---
        C = 0.0
        do jy = 0, py
          do jx = 0, px
             M = basisx(jx,ix) * basisy(jy,iy)
             C = C + M * Q(:,ox+jx,oy+jy)
          end do
        end do
        Cw(:,ix,iy) = C 
        ! ---
      end do
    end do
    !

  end subroutine EvaluateNormal2
  ! .......................................................

  ! .......................................................
  !> @brief     elevate spline derivatives the spline at X 
  !>
  !> @param[in] nderiv number of derivatives 
  !> @param[in] N corresponding number of partial derivatives
  !> @param[in] rationalize true if rational B-Splines are to be used
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] nx number of control points 
  !> @param[in] px spline degree 
  !> @param[in] Ux Initial Knot vector 
  !> @param[in] ny number of control points 
  !> @param[in] py spline degree 
  !> @param[in] Uy Initial Knot vector 
  !> @param[in] Q Initial Control points  
  !> @param[in] rx dimension of X - 1 
  !> @param[in] X the positions on wich evaluation is done  
  !> @param[in] ry dimension of Y - 1 
  !> @param[in] Y the positions on wich evaluation is done  
  !> @param[out] Cw Values  
  subroutine EvaluateDeriv2(nderiv,N,d,nx,px,Ux,ny,py,Uy,Q,weights,rx,X,ry,Y,Cw)
    use bspline
    implicit none
    integer(kind=4), intent(in)  :: nderiv
    integer(kind=4), intent(in)  :: N 
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: nx, ny
    integer(kind=4), intent(in)  :: px, py
    integer(kind=4), intent(in)  :: rx, ry
    real   (kind=8), intent(in)  :: Ux(0:nx+px+1)
    real   (kind=8), intent(in)  :: Uy(0:ny+py+1)
    real   (kind=8), intent(in)  :: Q(d,0:nx,0:ny)
    real   (kind=8), intent(in)  :: weights(0:nx,0:ny)
    real   (kind=8), intent(in)  :: X(0:rx), Y(0:ry)
    real   (kind=8), intent(out) :: Cw(0:N,d,0:rx,0:ry)
    integer(kind=4) :: ix, jx, iy, jy, ox, oy, deriv
    integer(kind=4) :: spanx(0:rx), spany(0:ry)
    real   (kind=8) :: dbasisx(0:px,0:nderiv,0:rx)
    real   (kind=8) :: dbasisy(0:py,0:nderiv,0:ry)
    ! Rdbasis(0) => Rij
    ! Rdbasis(1) => dx Rij
    ! Rdbasis(2) => dy Rij
    real   (kind=8) :: Rdbasis(0:N)  
    real   (kind=8) :: C(0:N,d)
    real   (kind=8) :: weight 
    real   (kind=8) :: M, Mx, My, Mxy, Mxx, Myy
    real   (kind=8) :: w, wx, wy, wxy, wxx, wyy

    Cw = 0.0

    !
    do ix = 0, rx
       spanx(ix) = FindSpan(nx,px,X(ix),Ux)
       call DersBasisFuns(spanx(ix),X(ix),px,nderiv,Ux,dbasisx(:,0:nderiv,ix))
    end do
    do iy = 0, ry
       spany(iy) = FindSpan(ny,py,Y(iy),Uy)
       call DersBasisFuns(spany(iy),Y(iy),py,nderiv,Uy,dbasisy(:,0:nderiv,iy))
    end do

    !
    ! compute 
    ! w   = sum wij Ni   Nj
    ! wx  = sum wij Ni'  Nj
    ! wy  = sum wij Ni   Nj'
    ! wxx = sum wij Ni'' Nj
    ! wxy = sum wij Ni'  Nj'
    ! wyy = sum wij Ni   Nj''
    do iy = 0, ry
    oy = spany(iy) - py
    do ix = 0, rx
    ox = spanx(ix) - px

       ! --- compute w and its Derivatives
       w   = 0.0 ; wx  = 0.0 ; wy  = 0.0
       wxx = 0.0 ; wxy = 0.0 ; wyy = 0.0
       do jy = 0, py
       do jx = 0, px
          weight = weights(ox+jx,oy+jy)

          M   = dbasisx(jx,0,ix) * dbasisy(jy,0,iy)
          w  = w  + M   * weight

          if (nderiv >= 1) then
             Mx  = dbasisx(jx,1,ix) * dbasisy(jy,0,iy)
             My  = dbasisx(jx,0,ix) * dbasisy(jy,1,iy)

             wx = wx + Mx  * weight
             wy = wy + My  * weight
          end if

          if (nderiv >= 2) then
             Mxx = dbasisx(jx,2,ix) * dbasisy(jy,0,iy)
             Mxy = dbasisx(jx,1,ix) * dbasisy(jy,1,iy)
             Myy = dbasisx(jx,0,ix) * dbasisy(jy,2,iy)

             wxx = wxx + Mxx * weight
             wxy = wxy + Mxy * weight
             wyy = wyy + Myy * weight
          end if 
       end do
       end do
       ! ---

       ! compute Nurbs and their derivatives
       C = 0.0     
       do jy = 0, py     
       do jx = 0, px
          M   = dbasisx(jx,0,ix) * dbasisy(jy,0,iy)
          Rdbasis(0) = M / w 

          if (nderiv >= 1) then
             Mx  = dbasisx(jx,1,ix) * dbasisy(jy,0,iy)
             My  = dbasisx(jx,0,ix) * dbasisy(jy,1,iy)

             Rdbasis(1) = Mx / w - M * wx / w**2 
             Rdbasis(2) = My / w - M * wy / w**2 
          end if
       
          if (nderiv >= 2) then
             Mxx = dbasisx(jx,2,ix) * dbasisy(jy,0,iy)
             Mxy = dbasisx(jx,1,ix) * dbasisy(jy,1,iy)
             Myy = dbasisx(jx,0,ix) * dbasisy(jy,2,iy)

             Rdbasis(3) = Mxx / w                 &
                        - 2 * Mx * wx / w**2      &
                        - M * wxx / w**2          &
                        + 2 * M * wx**2 / w**3
          
             Rdbasis(4) = Mxy / w                 &
                        - Mx * wy / w**2          &
                        - My * wx / w**2          &
                        - M * wxy / w**2          &
                        + 2 * M * wx * wy / w**3
          
             Rdbasis(5) = Myy / w                 &
                        - 2 * My * wy / w**2      &
                        - M * wyy / w**2          &
                        + 2 * M * wy**2 / w**3
          end if

          do deriv=0,N
             C(deriv,:) = C(deriv,:) + Rdbasis(deriv) * Q(:,ox+jx,oy+jy) * weights(ox+jx,oy+jy)
          end do
       end do
       end do

       Cw(0:N,1:d,ix,iy) = C(0:N,1:d)
       ! ---

    end do
    end do
    !
  end subroutine EvaluateDeriv2
  ! .......................................................

  ! .......................................................
  !> @brief     evaluate the spline at X 
  !>
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] nx number of control points 
  !> @param[in] px spline degree 
  !> @param[in] Ux Initial Knot vector 
  !> @param[in] ny number of control points 
  !> @param[in] py spline degree 
  !> @param[in] Uy Initial Knot vector 
  !> @param[in] nz number of control points 
  !> @param[in] pz spline degree 
  !> @param[in] Uz Initial Knot vector 
  !> @param[in] Q Initial Control points  
  !> @param[in] rx dimension of X - 1 
  !> @param[in] X the positions on wich evaluation is done  
  !> @param[in] ry dimension of Y - 1 
  !> @param[in] Y the positions on wich evaluation is done  
  !> @param[in] rz dimension of Z - 1 
  !> @param[in] Z the positions on wich evaluation is done  
  !> @param[out] Cw Values  
  subroutine Evaluate3(d,nx,px,Ux,ny,py,Uy,nz,pz,Uz,Q,weights,rx,X,ry,Y,rz,Z,Cw)
    use bspline
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: nx, ny, nz
    integer(kind=4), intent(in)  :: px, py, pz
    integer(kind=4), intent(in)  :: rx, ry, rz
    real   (kind=8), intent(in)  :: Ux(0:nx+px+1)
    real   (kind=8), intent(in)  :: Uy(0:ny+py+1)
    real   (kind=8), intent(in)  :: Uz(0:nz+pz+1)
    real   (kind=8), intent(in)  :: Q(d,0:nx,0:ny,0:nz)
    real   (kind=8), intent(in)  :: weights(0:nx,0:ny,0:nz)
    real   (kind=8), intent(in)  :: X(0:rx), Y(0:ry), Z(0:rz)
    real   (kind=8), intent(out) :: Cw(d,0:rx,0:ry,0:rz)
    integer(kind=4) :: ix, jx, iy, jy, iz, jz, ox, oy, oz
    integer(kind=4) :: spanx(0:rx), spany(0:ry), spanz(0:rz)
    real   (kind=8) :: basisx(0:px,0:rx), basisy(0:py,0:ry), basisz(0:pz,0:rz)
    real   (kind=8) :: M, C(d)
    real   (kind=8) :: w 

    !
    do ix = 0, rx
       spanx(ix) = FindSpan(nx,px,X(ix),Ux)
       call BasisFuns(spanx(ix),X(ix),px,Ux,basisx(:,ix))
    end do
    do iy = 0, ry
       spany(iy) = FindSpan(ny,py,Y(iy),Uy)
       call BasisFuns(spany(iy),Y(iy),py,Uy,basisy(:,iy))
    end do
    do iz = 0, rz
       spanz(iz) = FindSpan(nz,pz,Z(iz),Uz)
       call BasisFuns(spanz(iz),Z(iz),pz,Uz,basisz(:,iz))
    end do
    !
    do iz = 0, rz
    oz = spanz(iz) - pz
      do iy = 0, ry
      oy = spany(iy) - py
        do ix = 0, rx
        ox = spanx(ix) - px
        ! ---
        w = 0.0
        do jx = 0, px
           do jy = 0, py
              do jz = 0, pz
                 M = basisx(jx,ix) * basisy(jy,iy) * basisz(jz,iz)
                 w = w + M * weights(ox+jx,oy+jy,oz+jz)
              end do
           end do
        end do
        ! ---
        ! ---
        C = 0.0
        do jx = 0, px
           do jy = 0, py
              do jz = 0, pz
                 M = basisx(jx,ix) * basisy(jy,iy) * basisz(jz,iz)
                 C = C + M * weights(ox+jx,oy+jy,oz+jz) * Q(:,ox+jx,oy+jy,oz+jz)
              end do
           end do
        end do
        Cw(:,ix,iy,iz) = C / w
        ! ---
        end do
      end do
    end do
    !
  end subroutine Evaluate3
  ! .......................................................

  ! .......................................................
  !> @brief     elevate spline derivatives the spline at X 
  !>
  !> @param[in] nderiv number of derivatives 
  !> @param[in] N corresponding number of partial derivatives
  !> @param[in] rationalize true if rational B-Splines are to be used
  !> @param[in] d manifold dimension for the control points  
  !> @param[in] nx number of control points 
  !> @param[in] px spline degree 
  !> @param[in] Ux Initial Knot vector 
  !> @param[in] ny number of control points 
  !> @param[in] py spline degree 
  !> @param[in] Uy Initial Knot vector 
  !> @param[in] nz number of control points 
  !> @param[in] pz spline degree 
  !> @param[in] Uz Initial Knot vector 
  !> @param[in] Q Initial Control points  
  !> @param[in] rx dimension of X - 1 
  !> @param[in] X the positions on wich evaluation is done  
  !> @param[in] ry dimension of Y - 1 
  !> @param[in] Y the positions on wich evaluation is done  
  !> @param[in] rz dimension of Z - 1 
  !> @param[in] Z the positions on wich evaluation is done  
  !> @param[out] Cw Values  
  subroutine EvaluateDeriv3(nderiv,N,d,nx,px,Ux,ny,py,Uy,nz,pz,Uz,Q,weights,rx,X,ry,Y,rz,Z,Cw)
    use bspline
    implicit none
    integer(kind=4), intent(in)  :: nderiv
    integer(kind=4), intent(in)  :: N 
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: nx, ny, nz
    integer(kind=4), intent(in)  :: px, py, pz
    integer(kind=4), intent(in)  :: rx, ry, rz
    real   (kind=8), intent(in)  :: Ux(0:nx+px+1)
    real   (kind=8), intent(in)  :: Uy(0:ny+py+1)
    real   (kind=8), intent(in)  :: Uz(0:nz+pz+1)
    real   (kind=8), intent(in)  :: Q(d,0:nx,0:ny,0:nz)
    real   (kind=8), intent(in)  :: weights(0:nx,0:ny,0:nz)
    real   (kind=8), intent(in)  :: X(0:rx), Y(0:ry), Z(0:rz) 
    real   (kind=8), intent(out) :: Cw(0:N,d,0:rx,0:ry,0:rz)
    integer(kind=4) :: ix, jx, iy, jy, iz, jz, ox, oy, oz, deriv
    integer(kind=4) :: spanx(0:rx), spany(0:ry), spanz(0:rz)
    real   (kind=8) :: dbasisx(0:px,0:nderiv,0:rx)
    real   (kind=8) :: dbasisy(0:py,0:nderiv,0:ry)
    real   (kind=8) :: dbasisz(0:pz,0:nderiv,0:rz)
    ! Rdbasis(0) => Rij
    ! Rdbasis(1) => dx Rij
    ! Rdbasis(2) => dy Rij
    real   (kind=8) :: Rdbasis(0:N)  
    real   (kind=8) :: C(0:N,d)
    real   (kind=8) :: M, Mx, My, Mz, Mxy, Myz, Mzx, Mxx, Myy, Mzz
    real   (kind=8) :: w, wx, wy, wz, wxy, wyz, wzx, wxx, wyy, wzz
    real   (kind=8) :: weight 

    Cw = 0.0
    Rdbasis = 0.0

    !
    do ix = 0, rx
       spanx(ix) = FindSpan(nx,px,X(ix),Ux)
       call DersBasisFuns(spanx(ix),X(ix),px,nderiv,Ux,dbasisx(:,0:nderiv,ix))
    end do
    do iy = 0, ry
       spany(iy) = FindSpan(ny,py,Y(iy),Uy)
       call DersBasisFuns(spany(iy),Y(iy),py,nderiv,Uy,dbasisy(:,0:nderiv,iy))
    end do
    do iz = 0, rz
       spanz(iz) = FindSpan(nz,pz,Z(iz),Uz)
       call DersBasisFuns(spanz(iz),Z(iz),pz,nderiv,Uz,dbasisz(:,0:nderiv,iz))
    end do

    !
    ! compute 
    do iz = 0, rz
    oz = spanz(iz) - pz
      do iy = 0, ry
      oy = spany(iy) - py
        do ix = 0, rx
        ox = spanx(ix) - px

         ! --- compute w and its Derivatives
         w   = 0.0
         wx  = 0.0 ; wy  = 0.0 ; wz  = 0.0
         wxx = 0.0 ; wyy = 0.0 ; wzz = 0.0
         wxy = 0.0 ; wyz = 0.0 ; wzx = 0.0
         do jz = 0, pz     
         do jy = 0, py     
         do jx = 0, px
            weight = weights(ox+jx,oy+jy,oz+jz)

            M   = dbasisx(jx,0,ix) * dbasisy(jy,0,iy) * dbasisz(jz,0,iz)
            w  = w  + M   * weight

            if (nderiv >= 1) then
              Mx  = dbasisx(jx,1,ix) * dbasisy(jy,0,iy) * dbasisz(jz,0,iz)
              My  = dbasisx(jx,0,ix) * dbasisy(jy,1,iy) * dbasisz(jz,0,iz) 
              Mz  = dbasisx(jx,0,ix) * dbasisy(jy,0,iy) * dbasisz(jz,1,iz) 

              wx = wx + Mx * weight
              wy = wy + My * weight
              wz = wz + Mz * weight
            end if
         
            if (nderiv >= 2) then
              Mxx = dbasisx(jx,2,ix) * dbasisy(jy,0,iy) * dbasisz(jz,0,iz) 
              Myy = dbasisx(jx,0,ix) * dbasisy(jy,2,iy) * dbasisz(jz,0,iz)
              Mzz = dbasisx(jx,0,ix) * dbasisy(jy,0,iy) * dbasisz(jz,2,iz)

              Mxy = dbasisx(jx,1,ix) * dbasisy(jy,1,iy) * dbasisz(jz,0,iz)
              Myz = dbasisx(jx,0,ix) * dbasisy(jy,1,iy) * dbasisz(jz,1,iz)
              Mzx = dbasisx(jx,1,ix) * dbasisy(jy,0,iy) * dbasisz(jz,1,iz)

              wxx = wxx + Mxx * weight
              wyy = wyy + Myy * weight
              wzz = wzz + Mzz * weight

              wxy = wxy + Mxy * weight
              wyz = wyz + Myz * weight
              wzx = wzx + Mzx * weight
            end if

         end do
         end do
         end do
        ! ---

        ! compute Nurbs and their derivatives
         C = 0.0
         do jz = 0, pz     
         do jy = 0, py     
         do jx = 0, px
            M   = dbasisx(jx,0,ix) * dbasisy(jy,0,iy) * dbasisz(jz,0,iz)
            Rdbasis(0) = M / w

            if (nderiv >= 1) then
              Mx  = dbasisx(jx,1,ix) * dbasisy(jy,0,iy) * dbasisz(jz,0,iz)
              My  = dbasisx(jx,0,ix) * dbasisy(jy,1,iy) * dbasisz(jz,0,iz) 
              Mz  = dbasisx(jx,0,ix) * dbasisy(jy,0,iy) * dbasisz(jz,1,iz) 

              Rdbasis(1) = Mx / w - M * wx / w**2 
              Rdbasis(2) = My / w - M * wy / w**2 
              Rdbasis(3) = Mz / w - M * wz / w**2 
            end if
         
            if (nderiv >= 2) then
              Mxx = dbasisx(jx,2,ix) * dbasisy(jy,0,iy) * dbasisz(jz,0,iz) 
              Myy = dbasisx(jx,0,ix) * dbasisy(jy,2,iy) * dbasisz(jz,0,iz)
              Mzz = dbasisx(jx,0,ix) * dbasisy(jy,0,iy) * dbasisz(jz,2,iz)

              Mxy = dbasisx(jx,1,ix) * dbasisy(jy,1,iy) * dbasisz(jz,0,iz)
              Myz = dbasisx(jx,0,ix) * dbasisy(jy,1,iy) * dbasisz(jz,1,iz)
              Mzx = dbasisx(jx,1,ix) * dbasisy(jy,0,iy) * dbasisz(jz,1,iz)

              Rdbasis(4) = Mxx / w                 &
                         - 2 * Mx * wx / w**2      &
                         - M * wxx / w**2          &
                         + 2 * M * wx**2 / w**3
          
              Rdbasis(5) = Myy / w                 &
                         - 2 * My * wy / w**2      &
                         - M * wyy / w**2          &
                         + 2 * M * wy**2 / w**3

              Rdbasis(6) = Mzz / w                 &
                         - 2 * Mz * wz / w**2      &
                         - M * wzz / w**2          &
                         + 2 * M * wz**2 / w**3
          
              Rdbasis(7) = Mxy / w                 &
                         - Mx * wy / w**2          &
                         - My * wx / w**2          &
                         - M * wxy / w**2          &
                         + 2 * M * wx * wy / w**3
          
              Rdbasis(8) = Myz / w                 &
                         - My * wz / w**2          &
                         - Mz * wy / w**2          &
                         - M * wyz / w**2          &
                         + 2 * M * wy * wz / w**3
          
              Rdbasis(9) = Mzx / w                 &
                         - Mz * wx / w**2          &
                         - Mx * wz / w**2          &
                         - M * wzx / w**2          &
                         + 2 * M * wz * wx / w**3
            end if

            do deriv=0,N
               C(deriv,:) = C(deriv,:) + Rdbasis(deriv) * Q(:,ox+jx,oy+jy,oz+jz) &
                                                      & * weights(ox+jx,oy+jy,oz+jz)
            end do
         end do
         end do
         end do

         Cw(0:N,1:d,ix,iy,iz) = C(0:N,1:d)
        ! ---

        end do
      end do
    end do
    !
  end subroutine EvaluateDeriv3
  ! .......................................................
    
  ! .......................................................
  !> @brief returns pp form coeffs of a uniform quadratic spline
  !> pp_form(i,1:2): pp form on i-th element
  function pp_square() &
       result(pp_form)! todo : name of res
    implicit none
    real(kind=8),dimension(3,3)            :: pp_form
    ! LOCAL
    !> 1st element
    pp_form(1,1:2) =  0.
    pp_form(1,3)   =  1.
    !> 2nd element
    pp_form(2,2)   =  1.
    pp_form(2,3)   = -1.
      
  end function pp_square
  ! .......................................................

  ! .......................................................
  !> @brief returns pp form coeffs of uniform cubic spline
  !> pp_form(i,1:4): pp form on i-th element
  function pp_cubic() &
       result(pp_form)! todo : name of res
    implicit none
    real(kind=8),dimension(4,4)            :: pp_form
    ! LOCAL
    !> 1st element
    pp_form(1,1:3) =  0.
    pp_form(1,4)   =  1./6.
    !> 2nd element
    pp_form(2,1)   =  1./6.
    pp_form(2,2)   =  1./2.
    pp_form(2,3)   =  1./2.
    pp_form(2,4)   = -1./2.
    !> 3rd element
    pp_form(3,1)   =  2./3.
    pp_form(3,2)   =  0.
    pp_form(3,3)   = -1.
    pp_form(3,4)   =  1./2.
    !> 4th element
    pp_form(4,1)   =  1./6.
    pp_form(4,2)   = -1./2.   
    pp_form(4,3)   =  1./2.
    pp_form(4,4)   = -1./6.
    
  end function pp_cubic
  ! .......................................................

  ! .......................................................
  !> @brief     computes the refinement matrix corresponding to the insertion of a given knot 
  !>
  !> @param[in]    t             knot to be inserted 
  !> @param[in]    n             number of control points 
  !> @param[in]    p             spline degree 
  !> @param[in]    knots         Knot vector 
  !> @param[out]   mat           refinement matrix 
  !> @param[out]   knots_new     new Knot vector 
  subroutine spl_refinement_matrix_one_stage(t, n, p, knots, mat, knots_new)
  use pppack, only : interv 
  implicit none
    real(8),               intent(in)    :: t
    integer,                    intent(in)    :: n
    integer,                    intent(in)    :: p
    real(8), dimension(:), intent(in)    :: knots
    real(8), dimension(:,:), intent(out)    :: mat 
    real(8), dimension(:), optional, intent(out)    :: knots_new
    ! local
    integer :: i 
    integer :: j
    integer :: k
    integer :: i_err
    real(8) :: alpha

    mat = 0.0d0

    ! ...
    call interv ( knots, n+p+1, t, k, i_err) 
    ! ...

    ! ...
    j = 1
    call alpha_function(j, k, t, n, p, knots, alpha)
    mat(j,j) = alpha 

    do j=2, n
      call alpha_function(j, k, t, n, p, knots, alpha)
      mat(j,j)   = alpha 
      mat(j,j-1) = 1.0d0- alpha 
    end do

    j = n + 1
    call alpha_function(j, k, t, n, p, knots, alpha)
    mat(j,j-1) = 1.0d0 - alpha 
    ! ...

    ! ...
    if (present(knots_new)) then
      knots_new = -100000
      do i = 1, k
        knots_new(i) = knots(i)
      end do
      knots_new(k+1) = t
      do i = k+1, n+p+1
        knots_new(i+1) = knots(i)
      end do
    end if
    ! ...

  contains
    subroutine alpha_function(i, k, t, n, p, knots, alpha)
    implicit none
      integer,                    intent(in)    :: i 
      integer,                    intent(in)    :: k
      real(8),               intent(in)    :: t
      integer,                    intent(in)    :: n
      integer,                    intent(in)    :: p
      real(8), dimension(:), intent(in)    :: knots
      real(8),               intent(inout) :: alpha 
      ! local

      ! ...
      if (i <= k-p) then
        alpha = 1.0d0
      elseif ((k-p < i) .and. (i <= k)) then 
        alpha = (t - knots(i)) / (knots(i+p) - knots(i))
      else
        alpha = 0.0d0
      end if
      ! ...
    end subroutine alpha_function

  end subroutine spl_refinement_matrix_one_stage
  ! .......................................................

  ! .......................................................
  !> @brief     computes the refinement matrix corresponding to the insertion of a given list of knots 
  !>
  !> @param[in]    ts            array of knots to be inserted 
  !> @param[in]    n             number of control points 
  !> @param[in]    p             spline degree 
  !> @param[in]    knots         Knot vector 
  !> @param[out]   mat           refinement matrix 
  subroutine spl_refinement_matrix_multi_stages(ts, n, p, knots, mat)  
  implicit none
    real(8), dimension(:),   intent(in)  :: ts
    integer,                 intent(in)  :: n
    integer,                 intent(in)  :: p
    real(8), dimension(:),   intent(in)  :: knots
    real(8), dimension(:,:), intent(out) :: mat 
    ! local
    integer :: i
    integer :: j 
    integer :: m 
    integer :: k 
    real(8), dimension(:,:), allocatable :: mat_1
    real(8), dimension(:,:), allocatable :: mat_2
    real(8), dimension(:,:), allocatable :: mat_stage
    real(8), dimension(:), allocatable :: knots_1
    real(8), dimension(:), allocatable :: knots_2
   
    m = size(ts,1)
   
    allocate(mat_1(n + m, n + m))
    allocate(mat_2(n + m, n + m))
    allocate(mat_stage(n + m, n + m))
   
    allocate(knots_1(n + p + 1 + m))
    allocate(knots_2(n + p + 1 + m))
    
    ! ... mat is the identity at t=0
    mat_1 = 0.0d0
    do i = 1, n
      mat_1(i,i) = 1.0d0
    end do
    ! ...
   
    knots_1(1:n+p+1) = knots(1:n+p+1) 

    k = n
    do i = 1, m
      call spl_refinement_matrix_one_stage( ts(i), &
                               & k, &
                               & p, &
                               & knots_1, &
                               & mat_stage, & 
                               & knots_new=knots_2) 
   
      mat_2 = 0.0d0
      mat_2(1:k+1, 1:n) = matmul(mat_stage(1:k+1, 1:k), mat_1(1:k, 1:n))
   
      mat_1(1:k+1, 1:n) = mat_2(1:k+1, 1:n)  
      
      k = k + 1
      knots_1(1:k+p+1) = knots_2(1:k+p+1) 
    end do
    mat(1:k, 1:n) = mat_1(1:k, 1:n)  

  end subroutine spl_refinement_matrix_multi_stages
  ! .......................................................

  ! .......................................................
  !> @brief    Computes the derivative matrix for B-Splines 
  !>
  !> @param[in]  n              number of control points 
  !> @param[in]  p              spline degree 
  !> @param[in]  knots          Knot vector 
  !> @param[out] mat            derivatives matrix 
  !>                            where m depends on the boundary condition
  !> @param[in]  normalize      uses normalized B-Splines [optional] (Default: False) 
  subroutine spl_derivative_matrix(n, p, knots, mat, normalize)
  implicit none
    integer,                 intent(in)  :: n
    integer,                 intent(in)  :: p
    real(8), dimension(:),   intent(in)  :: knots
    real(8), dimension(:,:), intent(out) :: mat 
    logical, optional      , intent(in)  :: normalize
    ! local
    integer :: i
    integer :: j 
    real(8) :: alpha
    logical :: l_normalize

    ! ...
    l_normalize = .false.
    if (present(normalize)) then
      l_normalize = normalize
    end if
    ! ...

    ! ...
    mat = 0.0d0
    ! ...

    ! ...
    i = 1
    mat(i,i)   =  1.0d0 

    if (.not. l_normalize) then
      alpha      = p * 1.0d0 / (knots(i+p+1) - knots(i)) 

      mat(i,i)   =   alpha * mat(i,i)
    end if
    ! ...

    ! ...
    do i = 2, n 
      ! ...
      mat(i,i)   =  1.0d0 
      mat(i-1,i) = -1.0d0 
      ! ...

      ! ...
      if (.not. l_normalize) then
        alpha      = p * 1.0d0 / (knots(i+p+1) - knots(i)) 

        mat(i,i)   =   alpha * mat(i,i)
        mat(i-1,i) = - alpha * mat(i-1,i) 
      end if
      ! ...
    end do
    ! ...

  end subroutine spl_derivative_matrix
  ! .......................................................

  ! .......................................................
  !> @brief    Computes the toeplitz matrix associated to the stiffness-preconditioner symbol 
  !>
  !> @param[in]  p              spline degree 
  !> @param[in]  n_points       number of collocation points 
  !> @param[out] mat            mat is a dense matrix of size (n_points, n_points) 
  !>                            where m depends on the boundary condition
  subroutine spl_compute_symbol_stiffness(p, n_points, mat)
  use bspline, finds => findspan
  implicit none
    integer,                 intent(in)  :: p
    integer,                 intent(in)  :: n_points
    real(8), dimension(:,:), intent(out) :: mat 
    ! local
    integer :: i
    integer :: j
    integer :: span
    integer :: p_new
    integer :: n
    real(8) :: x
    real(8), dimension(:), allocatable :: batx
    real(8), dimension(:), allocatable :: knots

    ! ...
    p_new = 2*p - 1
    n     = 2*p_new + 1
    ! ...

    ! ...
    allocate(batx(p_new+1))
    allocate(knots(p_new+n+1))
    ! ...

    ! ...
    knots(1) = -float(p_new)
    do i = 2, p_new+n+1
      knots(i) = knots(i-1) + 1.0d0
    enddo
    ! ...

    ! ...
    ! TODO to fix. we are rewriting on the same array
    do j = 0, p_new
      x = float(j)
      span = finds(n-1,p_new,x,knots)
      call evalbasisfuns(p_new,n,knots,x,span,batx)
    end do
    ! ...
       
    ! ...
    mat = 0.0d0
    do i = 1, n_points
      do j = 1, n_points
        if ((abs(i-j) .le. p) .and. ((p-i+j) .ne. 0)) then
          mat(i,j) = batx(p-i+j)
        endif
      enddo
    enddo
    ! ...
    
    ! ...
    deallocate(batx)
    deallocate(knots)
    ! ...

  end subroutine spl_compute_symbol_stiffness 
  ! .......................................................

  ! .......................................................
  !> @brief    Computes collocation matrix 
  !>
  !> @param[in]  n              number of control points 
  !> @param[in]  p              spline degree 
  !> @param[in]  knots          Knot vector 
  !> @param[in]  arr_x          array of sites for evaluation 
  !> @param[out] mat            mat is a dense matrix of size (n_points, n_points) 
  !>                            where m depends on the boundary condition
  subroutine spl_collocation_matrix(n, p, knots, arr_x, mat)
  use bspline, finds => findspan
  implicit none
    integer,                 intent(in)  :: n
    integer,                 intent(in)  :: p
    real(8), dimension(:),   intent(in)  :: knots
    real(8), dimension(:),   intent(in)  :: arr_x 
    real(8), dimension(:,:), intent(out) :: mat 
    ! local
    integer :: i
    integer :: j
    integer :: span
    integer :: n_points
    real(8) :: x
    real(8), dimension(:,:), allocatable :: batx
    integer, dimension(:), allocatable :: spans

    ! ...
    n_points = size(arr_x, 1)
    ! ...

    ! ...
    allocate(batx(p+1,n_points))
    allocate(spans(n_points))
    ! ...

    ! ...
    do i = 1, n_points
      x = arr_x(i) 
      span = finds(n-1, p, x, knots)
      spans(i) = span
      call BasisFuns(span, x, p, knots, batx(:,i))
    end do
    ! ...
       
    ! ...
    mat = 0.0d0
    do i = 1, n_points
      span = spans(i)
      do j = 0, p
        mat(i,span-p+j+1) = batx(j+1,i)
      enddo
    enddo
    ! ...

    ! ...
    deallocate(spans)
    deallocate(batx)
    ! ...

  end subroutine spl_collocation_matrix 
  ! .......................................................

  ! .......................................................
  !> @brief    Computes collocation matrix using periodic bc.
  !>           mat must be allocatable. we allocate its memory inside the subroutine
  !>
  !> @param[in]  r         spline space regularity at extremities 
  !> @param[in]  n         number of control points 
  !> @param[in]  p         spline degree 
  !> @param[in]  knots     Knot vector 
  !> @param[in]  arr_x     array of sites for evaluation 
  !> @param[out] mat       mat is a dense matrix of size (n_points, n_points) 
  !>                       where m depends on the boundary condition
  subroutine spl_collocation_periodic_matrix(r, n, p, knots, arr_x, mat)
  use bspline, finds => findspan
  implicit none
    integer,                 intent(in)  :: r 
    integer,                 intent(in)  :: n
    integer,                 intent(in)  :: p
    real(8), dimension(:),   intent(in)  :: knots
    real(8), dimension(:),   intent(in)  :: arr_x 
    real(8), dimension(:,:), intent(out) :: mat 
    ! local
    integer :: i
    integer :: j
    integer :: k 
    integer :: span
    integer :: n_points
    integer :: nu
    real(8) :: x
    real(8), dimension(:,:), allocatable :: batx
    integer, dimension(:), allocatable :: spans

    ! ...
    n_points = size(arr_x, 1)
    nu = r + 1
    ! ...

    ! ...
!    allocate(mat(n_points, n-nu))
    allocate(batx(p+1,n_points))
    allocate(spans(n_points))
    ! ...

    ! ...
    do i = 1, n_points
      x = arr_x(i) 
      span = finds(n-1, p, x, knots)
      spans(i) = span
      call BasisFuns(span, x, p, knots, batx(:,i))
    end do
    ! ...
       
    ! ...
    mat = 0.0d0
    do i = 1, n_points
      span = spans(i)
      do k = 1, p+1
        j = span-p+k  
        if (j <= n-nu) then
          mat(i,j) = batx(k,i)
        else
          mat(i,j-(n-nu)) = batx(k,i)
        end if
      enddo
    enddo
    ! ...

    ! ...
    deallocate(spans)
    deallocate(batx)
    ! ...

  end subroutine spl_collocation_periodic_matrix 
  ! .......................................................

  ! .......................................................
  !> @brief     symetrizes a knot vector, needed for periodic interpolation 
  !>
  !> @param[in]    r     spline space regularity at extremities 
  !> @param[in]    n     number of control points
  !> @param[in]    p     spline degree 
  !> @param[inout] knots Knot vector 
  subroutine spl_symetrize_knots(r, n, p, knots)
    implicit none
    integer(kind=4), intent(in)  :: r 
    integer(kind=4), intent(in)  :: n 
    integer(kind=4), intent(in)  :: p
    real   (kind=8), dimension(:), intent(inout)  :: knots
    ! local
    integer(kind=4) :: i
    integer(kind=4) :: nu 
    real(8) :: period
    real(8), dimension(:), allocatable :: arr_u 

    allocate(arr_u(n+p+1))

    nu = r +1

    period = knots(n+1) - knots(p+1)
    do i=1,nu
      arr_u(i) = knots(n+i-nu) - period
    end do
    do i=nu+1, p+1
      arr_u(i) = knots(p+1)
    end do
    arr_u(p+2:n+p+1-nu) = knots(p+2:n+p+1-nu)
    do i=1,nu
      arr_u(n+p+1-nu+i) = knots(p+1+i) + period
    end do

    knots = arr_u
    deallocate(arr_u)

  end subroutine spl_symetrize_knots
  ! .......................................................

  ! ........................................ 
  !> @brief     creates an open knot vector on the interval [0, 1]
  !>
  !> @param[in]    n       number of control points
  !> @param[in]    p       spline degree 
  !> @param[out]   knots   Knot vector 
  subroutine spl_make_open_knots(n, p, knots)
    implicit none
    integer, intent(in)  :: p
    integer, intent(in)  :: n
    real(kind=8), dimension(:), intent(out) :: knots
    ! local
    integer :: n_elements
    integer :: i
    integer :: i_b
    integer :: i_e

    i_b = lbound(knots, 1)
    i_e = ubound(knots, 1)

    n_elements = n - p

    knots = 0.0
    do i = i_b, i_b + p, 1
      knots(i) = 0.0d0
    end do

    do i = i_b + 1 + p, i_b + n, 1
      knots(i) = (i - i_b - p)*1.0d0/n_elements
    end do

    do i = i_b + 1 + n, i_b + n + p, 1
      knots(i) = 1.0d0
    end do

  end subroutine spl_make_open_knots
  ! ........................................ 

  ! .......................................................
  !> @brief     returns the Greville abscissae 
  !>
  !> @param[in]    p     spline degree 
  !> @param[in]    n     number of control points
  !> @param[in]    knots Knot vector 
  !> @param[out]   arr_x Greville abscissae 
  subroutine spl_compute_greville(p, n, knots, arr_x)
    implicit none
    integer(kind=4), intent(in)  :: p
    integer(kind=4), intent(in)  :: n 
    real   (kind=8), dimension(:), intent(in)  :: knots
    real   (kind=8), dimension(:), intent(out) :: arr_x 
    integer(kind=4) :: i

    call Greville(p,n+p,knots,arr_x)

  end subroutine spl_compute_greville
  ! .......................................................

  ! .......................................................
  !> @brief     computes span index for every element 
  !>
  !> @param[in]    n               number of control points
  !> @param[in]    p               spline degree 
  !> @param[in]    knots           Knot vector 
  !> @param[out]   elements_spans  Knot vector 
  subroutine spl_compute_spans(p, n, knots, elements_spans)
  implicit none
    integer,                    intent(in)  :: n
    integer,                    intent(in)  :: p
    real(kind=8), dimension(:), intent(in)  :: knots
    integer,      dimension(:), intent(out) :: elements_spans
    ! local variables
    integer :: i_element
    integer :: i_knot
    integer :: i_b
    integer :: i_e

    ! ...
    elements_spans = -1 
    ! ...

    ! ...
    i_b = lbound(knots, 1)
    i_e = ubound(knots, 1)
    ! ...

    ! ...
    i_element = 0
    do i_knot = i_b, i_e - p - 1 
      ! we check if the element has zero measure
      if ( knots(i_knot) /= knots(i_knot + 1) ) then
        i_element = i_element + 1
        
        elements_spans(i_element) = i_knot
      end if
    end do
    ! ...
     
  end subroutine spl_compute_spans 
  ! .......................................................

  ! .......................................................
  !> @brief     computes the origin element of all splines
  !>
  !> @param[in]    n                number of control points
  !> @param[in]    p                spline degree 
  !> @param[in]    knots            Knot vector 
  !> @param[out]   origins_element  Knot vector 
  subroutine spl_compute_origins_element(p, n, knots, origins_element)
  implicit none
    integer,                    intent(in)  :: n
    integer,                    intent(in)  :: p
    real(kind=8), dimension(:), intent(in)  :: knots
    integer,      dimension(:), intent(out) :: origins_element
    ! local variables
    integer :: i_element
    integer :: i_knot
    integer :: i_b
    integer :: i_e

    ! ...
    origins_element = -1 
    ! ...

    ! ...
    i_b = lbound(knots, 1)
    i_e = ubound(knots, 1)
    ! ...

    ! ...
    i_element = 0
    do i_knot = i_b, i_e - p - 1 
      origins_element(i_knot) = i_element

      ! we check if the element has zero measure
      if ( knots(i_knot) /= knots(i_knot + 1) ) then
        i_element = i_element + 1
      end if
    end do
    ! ...
     
  end subroutine spl_compute_origins_element
  ! .......................................................

  ! ........................................ 
  !> @brief     constructs a grid from knots vector 
  !>
  !> @param[in]    p            spline degree 
  !> @param[in]    n            number of control points
  !> @param[in]    n_elements   number of elements
  !> @param[in]    knots        knot vector 
  !> @param[out]   grid         array containing the grid 
  subroutine spl_construct_grid_from_knots(p, n, n_elements, knots, grid)
    implicit none
    integer, intent(in)  :: p
    integer, intent(in)  :: n
    integer, intent(in)  :: n_elements
    real(kind=8), dimension(:), intent(in)  :: knots
    real(kind=8), dimension(:), intent(out) :: grid
    ! local
    integer :: i
    integer :: i_b
    integer :: i_e

    ! ...
    i_b = lbound(knots, 1)
    i_e = ubound(knots, 1)
    ! ...

    ! ...
    grid = 0.0
    do i = i_b, i_b+n_elements, 1
      grid(i) = knots(i + p)
    end do
    ! ...

  end subroutine spl_construct_grid_from_knots
  ! ........................................ 

  ! ........................................ 
  !> @brief     constructs the quadrature grid
  !>
  !> @param[in]    u        array containing the quadrature rule points on [-1, 1]
  !> @param[in]    w        array containing the quadrature rule weights on [-1, 1] 
  !> @param[in]    grid     array containing the grid 
  !> @param[out]   points   array containing for each element, all quad points
  !> @param[out]   weights  array containing for each element, all quad weights
  subroutine spl_construct_quadrature_grid(u, w, grid, points, weights)
    implicit none
    real(kind=8), dimension(:), intent(in) :: u
    real(kind=8), dimension(:), intent(in) :: w
    real(kind=8), dimension(:), intent(in) :: grid 
    real(kind=8), dimension(:,:), intent(out) :: points
    real(kind=8), dimension(:,:), intent(out) :: weights
    ! local
    integer :: i_b
    integer :: i_e
    integer :: k_b
    integer :: k_e
    integer :: i_element
    integer :: i_point
    real(kind=8) :: a
    real(kind=8) :: b
    real(kind=8) :: half

    ! ...
    i_b = lbound(grid, 1)
    i_e = ubound(grid, 1)

    k_b = lbound(u, 1)
    k_e = ubound(u, 1)
    ! ...

    points  = 0.0
    weights = 0.0

    do i_element = i_b, -1 + i_e, 1
      a = grid(i_element)
      b = grid(i_element + 1)
      half = 0.5d0*(-a + b)

      do i_point = k_b, k_e, 1
        points(i_point, i_element) = (u(i_point) + 1.0d0)*half + a
        weights(i_point, i_element) = w(i_point)*half
      end do
    end do

  end subroutine spl_construct_quadrature_grid
  ! ........................................ 

  ! ........................................ 
  !> @brief     evaluates all splines on a quad grid 
  !>
  !> @param[in]    n       number of control points
  !> @param[in]    p       spline degree 
  !> @param[in]    d       number of derivatives 
  !> @param[in]    knots   Knot vector 
  !> @param[in]    points  array containing for each element, all quad points
  !> @param[out]   basis   array containing the evaluation of splines for every element 
  subroutine spl_eval_on_grid_splines_ders(n, p, d, knots, points, basis)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: p
    integer, intent(in) :: d
    real(kind=8), dimension(:), intent(in)  :: knots
    real(kind=8), dimension(:,:), intent(in) :: points
    real(kind=8), dimension(:,:,:,:), intent(out) :: basis
    ! local
    integer :: i_element
    integer :: i_b
    integer :: i_e
    integer :: k
    real(kind=8), dimension(:,:,:), allocatable :: dN

!    basis_1(0:p, 0:d, 0:k-1, 0:n_elements-1)

    ! ...
    i_b = lbound(points, 2)
    i_e = ubound(points, 2)

    k = ubound(points, 1) - lbound(points, 1) + 1
    ! ...

    ! ...
    allocate(dN(0:p,0:d,0:k-1))
    ! ...

    ! ...
    basis = 0.0d0
    do i_element = i_b, i_e, 1
      dN = 0.0d0
      call spl_eval_splines_ders( p, n+p, d, k-1, knots, &
                                & points(:,i_element), &
                                & dN)
      basis(:,:,:,i_element) = dN(0:p,0:d,0:k-1)
    end do
    ! ...

    ! ...
    deallocate(dN)
    ! ...

  end subroutine spl_eval_on_grid_splines_ders
  ! ........................................ 

end module bsp_ext

