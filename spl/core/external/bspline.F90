! -*- coding: UTF-8 -*-

!> @brief 
!> Module for Splines
!> This module was extracted from igakit library.
!> Toolkit for IsoGeometric Analysis (IGA).
!> Copyright (c) 2014, Lisandro Dalcin and Nathaniel Collier.
!>

module bspline
contains

  ! .......................................................
  !> @brief     Determine non zero elements 
  !>
  !> @param[in]    n             number of control points  - 1
  !> @param[in]    p             spline degree 
  !> @param[in]    U             Knot vector 
  !> @param[inout] n_elements    number of non-zero elements 
  !> @param[inout] grid          the corresponding grid
  subroutine FindNonZeroElements_bspline(n,p,U,n_elements,grid) 
  implicit none
    integer(kind=4), intent(in) :: n, p
    real   (kind=8), intent(in) :: U(0:n+p+1)
    integer(kind=4), intent(inout) :: n_elements
    real   (kind=8), intent(inout) :: grid(0:n+p+1)
    ! local
    integer :: i
    integer :: i_current
    real(kind=8) :: min_current

    grid = -10000000.0 

    i_current = 0 
    grid(i_current) = minval(U)
    do i=1, (n + 1) + p
       min_current = minval(U(i : ))
       if ( min_current > grid(i_current) ) then
               i_current = i_current + 1
               grid(i_current) = min_current
       end if
    end do
    n_elements = i_current 

  end subroutine FindNonZeroElements_bspline
  ! .......................................................

  ! .......................................................
  !> @brief     Determine the knot span index 
  !>
  !> @param[in]  n     number of control points 
  !> @param[in]  p     spline degree 
  !> @param[in]  U     Knot vector 
  !> @param[in]  uu    given knot 
  !> @param[out] span  span index 
  function FindSpan(n,p,uu,U) result (span)
  implicit none
    integer(kind=4), intent(in) :: n, p
    real   (kind=8), intent(in) :: uu, U(0:n+p+1)
    integer(kind=4)             :: span
    integer(kind=4) low, high

    if (uu >= U(n+1)) then
       span = n
       return
    end if
    if (uu <= U(p)) then
       span = p
       return
    end if
    low  = p
    high = n+1
    span = (low + high) / 2
    do while (uu < U(span) .or. uu >= U(span+1))
       if (uu < U(span)) then
          high = span
       else
          low  = span
       end if
       span = (low + high) / 2
    end do
  end function FindSpan
  ! .......................................................

  ! .......................................................
  !> @brief     Computes the multiplicity of a knot  
  !>
  !> @param[in]  n      number of control points 
  !> @param[in]  p      spline degree 
  !> @param[in]  U      Knot vector 
  !> @param[in]  uu     Knot 
  !> @param[in]  i      starting index for search 
  !> @param[out] mult   multiplicit of the given knot 
  function FindMult(i,uu,p,U) result (mult)
  implicit none
    integer(kind=4), intent(in)  :: i, p
    real   (kind=8), intent(in)  :: uu, U(0:i+p+1)
    integer(kind=4)              :: mult
    integer(kind=4) :: j
    
    mult = 0
    do j = -p, p+1
       if (uu == U(i+j)) mult = mult + 1
    end do
  end function FindMult
  ! .......................................................

  ! .......................................................
  !> @brief     Computes the span and multiplicity of a knot  
  !>
  !> @param[in]  n    number of control points 
  !> @param[in]  p    spline degree 
  !> @param[in]  U    Knot vector 
  !> @param[in]  uu   Knot 
  !> @param[out] k    span of a knot 
  !> @param[out] s    multiplicity of a knot 
  subroutine FindSpanMult(n,p,uu,U,k,s)
    implicit none
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: uu, U(0:n+p+1)
    integer(kind=4), intent(out) :: k, s
    k = FindSpan(n,p,uu,U)
    s = FindMult(k,uu,p,U)
  end subroutine FindSpanMult
  ! .......................................................

  ! .......................................................
  !> @brief      Compute the nonvanishing basis functions
  !>
  !> @param[in]  p    spline degree 
  !> @param[in]  U    Knot vector 
  !> @param[in]  uu   Knot 
  !> @param[in]  i    span of a knot 
  !> @param[out] N    all (p+1) Splines non-vanishing at uu 
  subroutine BasisFuns(i,uu,p,U,N)
    implicit none
    integer(kind=4), intent(in) :: i, p
    real   (kind=8), intent(in) :: uu, U(0:i+p)
    real   (kind=8), intent(out):: N(0:p)
    integer(kind=4) :: j, r
    real   (kind=8) :: left(p), right(p), saved, temp
    N(0) = 1.0
    do j = 1, p
       left(j)  = uu - U(i+1-j)
       right(j) = U(i+j) - uu
       saved = 0.0
       do r = 0, j-1
          temp = N(r) / (right(r+1) + left(j-r))
          N(r) = saved + right(r+1) * temp
          saved = left(j-r) * temp
       end do
       N(j) = saved
    end do
  end subroutine BasisFuns
  ! .......................................................

  ! .......................................................
  !> @brief      Compute the nonvanishing basis functions and their derivatives.
  !> @details    First section is A2.2 (The NURBS Book) modified 
  !>             to store functions and knot differences.  
  !>
  !> @param[in]  p      spline degree 
  !> @param[in]  U      Knot vector 
  !> @param[in]  uu     Knot 
  !> @param[in]  i      span of a knot 
  !> @param[in]  n      number of derivatives 
  !> @param[out] ders   all (p+1) Splines non-vanishing at uu and their derivatives
  subroutine DersBasisFuns(i,uu,p,n,U,ders)
    implicit none
    integer(kind=4), intent(in) :: i, p, n
    real   (kind=8), intent(in) :: uu, U(0:i+p)
    real   (kind=8), intent(out):: ders(0:p,0:n)
    integer(kind=4) :: j, k, r, s1, s2, rk, pk, j1, j2
    real   (kind=8) :: saved, temp, d
    real   (kind=8) :: left(p), right(p)
    real   (kind=8) :: ndu(0:p,0:p), a(0:1,0:p)
    ndu(0,0) = 1.0
    do j = 1, p
       left(j)  = uu - U(i+1-j)
       right(j) = U(i+j) - uu
       saved = 0.0
       do r = 0, j-1
          ndu(j,r) = right(r+1) + left(j-r)
          temp = ndu(r,j-1) / ndu(j,r)
          ndu(r,j) = saved + right(r+1) * temp
          saved = left(j-r) * temp
       end do
       ndu(j,j) = saved
    end do
    ders(:,0) = ndu(:,p)
    do r = 0, p
       s1 = 0; s2 = 1;
       a(0,0) = 1.0
       do k = 1, n
          d = 0.0
          rk = r-k; pk = p-k;
          if (r >= k) then
             a(s2,0) = a(s1,0) / ndu(pk+1,rk)
             d =  a(s2,0) * ndu(rk,pk)
          end if
          if (rk > -1) then
             j1 = 1
          else
             j1 = -rk
          end if
          if (r-1 <= pk) then
             j2 = k-1
          else
             j2 = p-r
          end if
          do j = j1, j2
             a(s2,j) = (a(s1,j) - a(s1,j-1)) / ndu(pk+1,rk+j)
             d =  d + a(s2,j) * ndu(rk+j,pk)
          end do
          if (r <= pk) then
             a(s2,k) = - a(s1,k-1) / ndu(pk+1,r)
             d =  d + a(s2,k) * ndu(r,pk)
          end if
          ders(r,k) = d
          j = s1; s1 = s2; s2 = j;
       end do
    end do
    r = p
    do k = 1, n
       ders(:,k) = ders(:,k) * r
       r = r * (p-k)
    end do
  end subroutine DersBasisFuns
  ! .......................................................

  ! .......................................................
  !> @brief      Compute the nonvanishing basis functions and their derivatives.
  !> @details    First section is A2.2 (The NURBS Book) modified 
  !>             to store functions and knot differences.  
  !>
  !> @param[in]  p           spline degree 
  !> @param[in]  U           Knot vector 
  !> @param[in]  uu          Knot array 
  !> @param[in]  n_points    size of the array uu 
  !> @param[in]  i span      of a knot 
  !> @param[in]  n number    of derivatives 
  !> @param[out] ders        all (p+1) Splines non-vanishing at uu and their derivatives
  subroutine DersBasisFuns_array(i,uu,n_points,p,n,U,ders)
    implicit none
    integer(kind=4), intent(in) :: i, p, n, n_points
    real   (kind=8), dimension(1:n_points), intent(in) :: uu
    real   (kind=8), intent(in) :: U(0:i+p)
    real   (kind=8), intent(out):: ders(0:p,0:n,1:n_points)
    integer(kind=4) :: j, k, r, s1, s2, rk, pk, j1, j2
    real   (kind=8), dimension(1:n_points) :: saved
    real   (kind=8), dimension(1:n_points) :: temp
    real   (kind=8), dimension(1:n_points) :: d
    real   (kind=8), dimension(1:p,1:n_points) :: left
    real   (kind=8), dimension(1:p,1:n_points) :: right
    real   (kind=8), dimension(0:p,0:p,1:n_points) :: ndu
    real   (kind=8), dimension(0:1,0:p,1:n_points) :: a
    ndu(0,0,:) = 1.0
    do j = 1, p
       left(j,:)  = uu(:) - U(i+1-j)
       right(j,:) = U(i+j) - uu(:)
       saved = 0.0
       do r = 0, j-1
          ndu(j,r,:) = right(r+1,:) + left(j-r,:)
          temp(:) = ndu(r,j-1,:) / ndu(j,r,:)
          ndu(r,j,:) = saved(:) + right(r+1,:) * temp(:)
          saved(:) = left(j-r,:) * temp(:)
       end do
       ndu(j,j,:) = saved(:)
    end do
    ders(:,0,:) = ndu(:,p,:)
    do r = 0, p
       s1 = 0; s2 = 1;
       a(0,0,:) = 1.0
       do k = 1, n
          d(:) = 0.0
          rk = r-k; pk = p-k;
          if (r >= k) then
             a(s2,0,:) = a(s1,0,:) / ndu(pk+1,rk,:)
             d(:) =  a(s2,0,:) * ndu(rk,pk,:)
          end if
          if (rk > -1) then
             j1 = 1
          else
             j1 = -rk
          end if
          if (r-1 <= pk) then
             j2 = k-1
          else
             j2 = p-r
          end if
          do j = j1, j2
             a(s2,j,:) = (a(s1,j,:) - a(s1,j-1,:)) / ndu(pk+1,rk+j,:)
             d(:) =  d(:) + a(s2,j,:) * ndu(rk+j,pk,:)
          end do
          if (r <= pk) then
             a(s2,k,:) = - a(s1,k-1,:) / ndu(pk+1,r,:)
             d(:) =  d(:) + a(s2,k,:) * ndu(r,pk,:)
          end if
          ders(r,k,:) = d(:)
          j = s1; s1 = s2; s2 = j;
       end do
    end do
    r = p
    do k = 1, n
       ders(:,k,:) = ders(:,k,:) * r
       r = r * (p-k)
    end do
  end subroutine DersBasisFuns_array 
  ! .......................................................

  ! .......................................................
  !> @brief     evaluates a B-Spline curve at the knot uu 
  !>
  !> @param[in]    d             dimension of the manifold 
  !> @param[in]    n             number of control points  - 1
  !> @param[in]    p             spline degree 
  !> @param[in]    U             Knot vector 
  !> @param[in]    Pw            weighted control points 
  !> @param[in]    uu            knot to evaluate at 
  !> @param[inout] C             the point on the curve 
  subroutine CurvePoint(d,n,p,U,Pw,uu,C)
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    real   (kind=8), intent(in)  :: uu
    real   (kind=8), intent(out) :: C(d)
    integer(kind=4) :: j, span
    real   (kind=8) :: basis(0:p)
    span = FindSpan(n,p,uu,U)
    call BasisFuns(span,uu,p,U,basis)
    C = 0.0
    do j = 0, p
       C = C + basis(j)*Pw(:,span-p+j)
    end do
  end subroutine CurvePoint
  ! .......................................................

  ! .......................................................
  !> @brief     evaluates a B-Spline surface at the knot (uu, vv) 
  !>
  !> @param[in]    d             dimension of the manifold 
  !> @param[in]    n             number of control points  - 1
  !> @param[in]    p             spline degree 
  !> @param[in]    U             Knot vector 
  !> @param[in]    m             number of control points  - 1
  !> @param[in]    q             spline degree 
  !> @param[in]    V             Knot vector 
  !> @param[in]    Pw            weighted control points 
  !> @param[in]    uu            knot to evaluate at 
  !> @param[in]    vv            knot to evaluate at 
  !> @param[inout] S             the point on the surface 
  subroutine SurfacePoint(d,n,p,U,m,q,V,Pw,uu,vv,S)
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    integer(kind=4), intent(in)  :: m, q
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: V(0:m+q+1)
    real   (kind=8), intent(in)  :: Pw(d,0:m,0:n)
    real   (kind=8), intent(in)  :: uu, vv
    real   (kind=8), intent(out) :: S(d)
    integer(kind=4) :: uj, vj, uspan, vspan
    real   (kind=8) :: ubasis(0:p), vbasis(0:q)
    uspan = FindSpan(n,p,uu,U)
    call BasisFuns(uspan,uu,p,U,ubasis)
    vspan = FindSpan(m,q,vv,V)
    call BasisFuns(vspan,vv,q,V,vbasis)
    S = 0.0
    do uj = 0, p
       do vj = 0, q
          S = S + ubasis(uj)*vbasis(vj)*Pw(:,vspan-q+vj,uspan-p+uj)
       end do
    end do
  end subroutine SurfacePoint
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
  subroutine CurvePntByCornerCut(d,n,p,U,Pw,x,Cw)
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    real   (kind=8), intent(in)  :: x
    real   (kind=8), intent(out) :: Cw(d)
    integer(kind=4) :: i, j, k, s, r
    real   (kind=8) :: uu, alpha, Rw(d,0:p)
    if (x <= U(p)) then
       uu = U(p)
       k = p
       s = FindMult(p,uu,p,U)
       if (s >= p) then
          Cw(:) = Pw(:,0)
          return
       end if
    elseif (x >= U(n+1)) then
       uu = U(n+1)
       k = n+1
       s = FindMult(n,uu,p,U)
       if (s >= p) then
          Cw(:) = Pw(:,n)
          return
       end if
    else
       uu = x
       k = FindSpan(n,p,uu,U)
       s = FindMult(k,uu,p,U)
       if (s >= p) then
          Cw(:) = Pw(:,k-p)
          return
       end if
    end if
    r = p-s
    do i = 0, r
       Rw(:,i) = Pw(:,k-p+i)
    end do
    do j = 1, r
       do i = 0, r-j
          alpha = (uu-U(k-p+j+i))/(U(i+k+1)-U(k-p+j+i))
          Rw(:,i) = alpha*Rw(:,i+1)+(1-alpha)*Rw(:,i)
       end do
    end do
    Cw(:) = Rw(:,0)
  end subroutine CurvePntByCornerCut
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
  !> @param[in] k span of a knot 
  !> @param[in] s multiplicity of a knot 
  !> @param[in] r number of times uu will be inserted
  !> @param[in] V Final Knot vector 
  !> @param[in] Qw Final Control points  
  subroutine InsertKnot(d,n,p,U,Pw,uu,k,s,r,V,Qw)
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    real   (kind=8), intent(in)  :: uu
    integer(kind=4), intent(in)  :: k, s, r
    real   (kind=8), intent(out) :: V(0:n+p+1+r)
    real   (kind=8), intent(out) :: Qw(d,0:n+r)
    integer(kind=4) :: i, j, idx
    real   (kind=8) :: alpha, Rw(d,0:p)
    ! Load new knot vector
    forall (i = 0:k) V(i) = U(i)
    forall (i = 1:r) V(k+i) = uu
    forall (i = k+1:n+p+1) V(i+r) = U(i)
    ! Save unaltered control points
    forall (i = 0:k-p) Qw(:,i)   = Pw(:,i)
    forall (i = k-s:n) Qw(:,i+r) = Pw(:,i)
    forall (i = 0:p-s) Rw(:,i)   = Pw(:,k-p+i)
    ! Insert the knot r times
    do j = 1, r
       idx = k-p+j
       do i = 0, p-j-s
          alpha = (uu-U(idx+i))/(U(i+k+1)-U(idx+i))
          Rw(:,i) = alpha*Rw(:,i+1)+(1-alpha)*Rw(:,i)
       end do
       Qw(:,idx) = Rw(:,0)
       Qw(:,k+r-j-s) = Rw(:,p-j-s)
    end do
    ! Load remaining control points
    idx = k-p+r
    do i = idx+1, k-s-1
       Qw(:,i) = Rw(:,i-idx)
    end do
  end subroutine InsertKnot
  ! .......................................................

  ! .......................................................
  !> @brief     removes a knot from a B-Splines curve, given a tolerance 
  !>
  !> @param[in]    d      dimension of the manifold 
  !> @param[in]    n      number of control points  - 1
  !> @param[in]    p      spline degree 
  !> @param[inout] U      Knot vector 
  !> @param[inout] Pw     weighted control points 
  !> @param[in]    uu     knot to remove 
  !> @param[in]    r      starting multiplicity to remove 
  !> @param[in]    s      ending multiplicity to remove 
  !> @param[in]    num    maximum number of iterations 
  !> @param[out]   t      requiered number of iterations 
  !> @param[in]    TOL    tolerance for the distance to the control point 
  subroutine RemoveKnot(d,n,p,U,Pw,uu,r,s,num,t,TOL)
    implicit none
    integer(kind=4), intent(in)    :: d
    integer(kind=4), intent(in)    :: n, p
    real   (kind=8), intent(inout) :: U(0:n+p+1)
    real   (kind=8), intent(inout) :: Pw(d,0:n)
    real   (kind=8), intent(in)    :: uu
    integer(kind=4), intent(in)    :: r, s, num
    integer(kind=4), intent(out)   :: t
    real   (kind=8), intent(in)    :: TOL

    integer(kind=4) :: m,ord,fout,last,first,off
    integer(kind=4) :: i,j,ii,jj,k
    logical         :: remflag
    real   (kind=8) :: temp(d,0:2*p)
    real   (kind=8) :: alfi,alfj

    m = n + p + 1
    ord = p + 1
    fout = (2*r-s-p)/2
    first = r - p
    last  = r - s
    do t = 0,num-1
       off = first - 1
       temp(:,0) = Pw(:,off)
       temp(:,last+1-off) = Pw(:,last+1)
       i = first; ii = 1
       j = last;  jj = last - off
       remflag = .false.
       do while (j-i > t)
          alfi = (uu-U(i))/(U(i+ord+t)-U(i))
          alfj = (uu-U(j-t))/(U(j+ord)-U(j-t))
          temp(:,ii) = (Pw(:,i)-(1.0-alfi)*temp(:,ii-1))/alfi
          temp(:,jj) = (Pw(:,j)-alfj*temp(:,jj+1))/(1.0-alfj)
          i = i + 1; ii = ii + 1
          j = j - 1; jj = jj - 1
       end do
       if (j-i < t) then
          if (Distance(d,temp(:,ii-1),temp(:,jj+1)) <= TOL) then
             remflag = .true.
          end if
       else
          alfi = (uu-U(i))/(U(i+ord+t)-U(i))
          if (Distance(d,Pw(:,i),alfi*temp(:,ii+t+1)+(1-alfi)*temp(:,ii-1)) <= TOL) then
             remflag = .true.
          end if
       end if
       if (remflag .eqv. .false.) then
          exit ! break out of the for loop
       else
          i = first
          j = last
          do while (j-i > t)
             Pw(:,i) = temp(:,i-off)
             Pw(:,j) = temp(:,j-off)
             i = i + 1
             j = j - 1
          end do
       end if
       first = first - 1
       last  = last  + 1
    end do
    if (t == 0) return
    do k = r+1,m
       U(k-t) = U(k)
    end do
    j = fout
    i = j
    do k = 1,t-1
       if (mod(k,2) == 1) then
          i = i + 1
       else
          j = j - 1
       end if
    end do
    do k = i+1,n
       Pw(:,j) = Pw(:,k)
       j = j + 1
    enddo
  contains
    function Distance(d,P1,P2) result (dist)
      implicit none
      integer(kind=4), intent(in) :: d
      real   (kind=8), intent(in) :: P1(d),P2(d)
      integer(kind=4) :: i
      real   (kind=8) :: dist
      dist = 0.0
      do i = 1,d
         dist = dist + (P1(i)-P2(i))*(P1(i)-P2(i))
      end do
      dist = sqrt(dist)
    end function Distance
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
  subroutine ClampKnot(d,n,p,U,Pw,l,r)
    implicit none
    integer(kind=4), intent(in)    :: d
    integer(kind=4), intent(in)    :: n, p
    real   (kind=8), intent(inout) :: U(0:n+p+1)
    real   (kind=8), intent(inout) :: Pw(d,0:n)
    logical(kind=4), intent(in)    :: l, r
    integer(kind=4) :: k, s
    if (l) then ! Clamp at left end
       k = p
       s = FindMult(p,U(p),p,U)
       call KntIns(d,n,p,U,Pw,k,s)
       U(0:p-1) = U(p)
    end if
    if (r) then ! Clamp at right end
       k = n+1
       s = FindMult(n,U(n+1),p,U)
       call KntIns(d,n,p,U,Pw,k,s)
       U(n+2:n+p+1) = U(n+1)
    end if
  contains
    subroutine KntIns(d,n,p,U,Pw,k,s)
        implicit none
        integer(kind=4), intent(in)    :: d
        integer(kind=4), intent(in)    :: n, p
        real   (kind=8), intent(in)    :: U(0:n+p+1)
        real   (kind=8), intent(inout) :: Pw(d,0:n)
        integer(kind=4), intent(in)    :: k, s
        integer(kind=4) :: r, i, j, idx
        real   (kind=8) :: uu, alpha, Rw(d,0:p), Qw(d,0:2*p)
        if (s >= p) return
        uu = U(k)
        r = p-s
        Qw(:,0) = Pw(:,k-p)
        Rw(:,0:p-s) = Pw(:,k-p:k-s)
        do j = 1, r
           idx = k-p+j
           do i = 0, p-j-s
              alpha = (uu-U(idx+i))/(U(i+k+1)-U(idx+i))
              Rw(:,i) = alpha*Rw(:,i+1)+(1-alpha)*Rw(:,i)
           end do
           Qw(:,j) = Rw(:,0)
           Qw(:,p-j-s+r) = Rw(:,p-j-s)
        end do
        if (k == p) then ! left end
           Pw(:,0:r-1) = Qw(:,r:r+r-1)
        else             ! right end
           Pw(:,n-r+1:n) = Qw(:,p-r:p-1)
        end if
      end subroutine KntIns
  end subroutine ClampKnot
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
  subroutine UnclampKnot(d,n,p,U,Pw,l,r)
    implicit none
    integer(kind=4), intent(in)    :: d
    integer(kind=4), intent(in)    :: n, p
    real   (kind=8), intent(inout) :: U(0:n+p+1)
    real   (kind=8), intent(inout) :: Pw(d,0:n)
    logical(kind=4), intent(in)    :: l, r
    integer(kind=4) :: i, j, k
    real   (kind=8) :: alpha
    if (l) then ! Unclamp at left end
       do i = 0, p-2
          U(p-i-1) = U(p-i) - (U(n-i+1)-U(n-i))
          k = p-1
          do j = i, 0, -1
             alpha = (U(p)-U(k))/(U(p+j+1)-U(k))
             Pw(:,j) = (Pw(:,j)-alpha*Pw(:,j+1))/(1-alpha)
             k = k-1
          end do
       end do
       U(0) = U(1) - (U(n-p+2)-U(n-p+1)) ! Set first knot
    end if
    if (r) then ! Unclamp at right end
       do i = 0, p-2
          U(n+i+2) = U(n+i+1) + (U(p+i+1)-U(p+i))
          do j = i, 0, -1
             alpha = (U(n+1)-U(n-j))/(U(n-j+i+2)-U(n-j))
             Pw(:,n-j) = (Pw(:,n-j)-(1-alpha)*Pw(:,n-j-1))/alpha
          end do
       end do
       U(n+p+1) = U(n+p) + (U(2*p)-U(2*p-1)) ! Set last knot
    end if
  end subroutine UnclampKnot
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
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    integer(kind=4), intent(in)  :: r
    real   (kind=8), intent(in)  :: X(0:r)
    real   (kind=8), intent(out) :: Ubar(0:n+r+1+p+1)
    real   (kind=8), intent(out) :: Qw(d,0:n+r+1)
    integer(kind=4) :: m, a, b
    integer(kind=4) :: i, j, k, l
    integer(kind=4) :: idx
    real   (kind=8) :: alpha
    if (r < 0) then
       Ubar = U
       Qw = Pw
       return
    end if
    m = n + p + 1
    a = FindSpan(n,p,X(0),U)
    b = FindSpan(n,p,X(r),U)
    b = b + 1
    forall (j = 0:a-p) Qw(:,j)     = Pw(:,j)
    forall (j = b-1:n) Qw(:,j+r+1) = Pw(:,j)
    forall (j =   0:a) Ubar(j)     = U(j)
    forall (j = b+p:m) Ubar(j+r+1) = U(j)
    i = b + p - 1
    k = b + p + r
    do j = r, 0, -1
       do while (X(j) <= U(i) .and. i > a)
          Qw(:,k-p-1) = Pw(:,i-p-1)
          Ubar(k) = U(i)
          k = k - 1
          i = i - 1
       end do
       Qw(:,k-p-1) = Qw(:,k-p)
       do l = 1, p
          idx = k - p + l
          alpha = Ubar(k+l) - X(j)
          if (abs(alpha) == 0.0) then
             Qw(:,idx-1) = Qw(:,idx)
          else
             alpha = alpha / (Ubar(k+l) - U(i-p+l))
             Qw(:,idx-1) = alpha*Qw(:,idx-1) + (1-alpha)*Qw(:,idx)
          end if
       end do
       Ubar(k) = X(j)
       k = k-1
    end do
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
    implicit none
    integer(kind=4), intent(in)  :: d
    integer(kind=4), intent(in)  :: n, p
    real   (kind=8), intent(in)  :: U(0:n+p+1)
    real   (kind=8), intent(in)  :: Pw(d,0:n)
    integer(kind=4), intent(in)  :: t
    integer(kind=4), intent(in)  :: nh
    real   (kind=8), intent(out) :: Uh(0:nh+p+t+1)
    real   (kind=8), intent(out) :: Qw(d,0:nh)

    integer(kind=4) :: i, j, k, kj, tr, a, b
    integer(kind=4) :: m, ph, kind, cind, first, last
    integer(kind=4) :: r, oldr, s, mul, lbz, rbz

    real   (kind=8) :: bezalfs(0:p+t,0:p)
    real   (kind=8) :: bpts(d,0:p), ebpts(d,0:p+t), nextbpts(d,0:p-2)
    real   (kind=8) :: alfs(0:p-2), ua, ub, alf, bet, gam, den
    if (t < 1) then
       Uh = U
       Qw = Pw
       return
    end if
    m = n + p + 1
    ph = p + t
    ! Bezier coefficients
    bezalfs(0,0)  = 1.0
    bezalfs(ph,p) = 1.0
    do i = 1, ph/2
       do j = max(0,i-t), min(p,i)
          bezalfs(i,j) = Bin(p,j)*Bin(t,i-j)*(1.0d+0/Bin(ph,i))
       end do
    end do
    do i = ph/2+1, ph-1
       do j = max(0,i-t), min(p,i)
          bezalfs(i,j) = bezalfs(ph-i,p-j)
       end do
    end do
    kind = ph+1
    cind = 1
    r = -1
    a = p
    b = p+1
    ua = U(a)
    Uh(0:ph) = ua
    Qw(:,0) = Pw(:,0)
    bpts = Pw(:,0:p)
    do while (b < m)
       i = b
       do while (b < m)
          if (U(b) /= U(b+1)) exit
          b = b + 1
       end do
       mul = b - i + 1
       oldr = r
       r = p - mul
       ub = U(b)
       if (oldr > 0) then
          lbz = (oldr+2)/2
       else
          lbz = 1
       end if
       if (r > 0) then
          rbz = ph - (r+1)/2
       else
          rbz = ph
       end if
       ! insert knots
       if (r > 0) then
          do k = p, mul+1, -1
             alfs(k-mul-1) = (ub-ua)/(U(a+k)-ua)
          end do
          do j = 1, r
             s = mul + j
             do k = p, s, -1
                bpts(:,k) = alfs(k-s)  * bpts(:,k) + &
                       (1.0-alfs(k-s)) * bpts(:,k-1)
             end do
             nextbpts(:,r-j) = bpts(:,p)
          end do
       end if
       ! degree elevate
       do i = lbz, ph
          ebpts(:,i) = 0.0
          do j = max(0,i-t), min(p,i)
             ebpts(:,i) = ebpts(:,i) + bezalfs(i,j)*bpts(:,j)
          end do
       end do
       ! remove knots
       if (oldr > 1) then
          first = kind-2
          last = kind
          den = ub-ua
          bet = (ub-Uh(kind-1))/den
          do tr = 1, oldr-1
             i = first
             j = last
             kj = j-kind+1
             do while (j-i > tr)
                if (i < cind) then
                   alf = (ub-Uh(i))/(ua-Uh(i))
                   Qw(:,i) = alf*Qw(:,i) + (1.0-alf)*alf*Qw(:,i-1)
                end if
                if (j >= lbz) then
                   if (j-tr <= kind-ph+oldr) then
                      gam = (ub-Uh(j-tr))/den
                      ebpts(:,kj) = gam*ebpts(:,kj) + (1.0-gam)*ebpts(:,kj+1)
                   else
                      ebpts(:,kj) = bet*ebpts(:,kj) + (1.0-bet)*ebpts(:,kj+1)
                   end if
                end if
                i = i+1
                j = j-1
                kj = kj-1
             end do
             first = first-1
             last = last+1
          end do
       end if
       !
       if (a /= p) then
          do i = 0, ph-oldr-1
             Uh(kind) = ua
             kind = kind+1
          end do
       end if
       do j = lbz, rbz
          Qw(:, cind) = ebpts(:,j)
          cind = cind+1
       end do
       !
       if (b < m) then
          bpts(:,0:r-1) = nextbpts(:,0:r-1)
          bpts(:,r:p) = Pw(:,b-p+r:b)
          a = b
          b = b+1
          ua = ub
       else
          Uh(kind:kind+ph) = ub
       end if
    end do
  contains
    pure function Bin(n,k) result (C)
      implicit none
      integer(kind=4), intent(in) :: n, k
      integer(kind=4) :: i, C
      C = 1
      do i = 0, min(k,n-k) - 1
         C = C * (n - i)
         C = C / (i + 1)
      end do
    end function Bin
  end subroutine DegreeElevate
  ! .......................................................

end module bspline
