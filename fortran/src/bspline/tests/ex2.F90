!
! file: ex1.f90
!
!
! usage:
!   > ./ex1 
!
! authors:
!   ahmed ratnani  - ratnaniahmed@gmail.com
!

! ............................................
subroutine evalbasisfuns_test_1()
use spl_m_bspline, finds => findspan
use spl_m_bsp
implicit none
     integer, parameter :: n = 6 
     integer, parameter :: p = 2
     real*8, dimension(n+p+1) :: u
     integer, parameter :: nx = 6
     real*8, dimension(nx) :: x
     real*8, dimension(nx, p+1) :: expected
     real*8, dimension(p+1) :: batx
     integer :: i
     integer :: span 

     print *, ">>>> evalbasisfuns_test_1: begin "

     u        = (/0.0,  0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0/)

     ! ...
     do i = 1, nx
        x(i) = float(i-1) / float(nx-1) 
     end do
     ! ...

     ! ...
     expected(1,:) = (/ 1.0000000000000000, 0.0000000000000000, 0.0000000000000000/)
     expected(2,:) = (/ 0.5000000000000000, 0.5000000000000000, 0.0000000000000000/)
     expected(3,:) = (/ 0.5000000000000000, 0.5000000000000000, 0.0000000000000000/)
     expected(4,:) = (/ 0.5000000000000000, 0.5000000000000000, 0.0000000000000000/)
     expected(5,:) = (/ 0.0000000000000000, 0.0000000000000000, 1.0000000000000000/)
     ! ...

     ! ...
     open(unit=12, file="bspline_ex2_test_1.txt"&
             & , action="write", status="replace")
     do i = 1, nx
        span = finds(n-1,p,x(i),u)
        call evalbasisfuns(p,n,u,x(i),span,batx)
        print *, maxval(abs(batx - expected(i,:)))

        write(12,*) batx 
     end do
     close(12)
     ! ...

     print *, ">>>> evalbasisfuns_test_1: end "

end subroutine evalbasisfuns_test_1
! ............................................

! ............................................
program main

  implicit none

  call evalbasisfuns_test_1()

end program main
! ............................................
