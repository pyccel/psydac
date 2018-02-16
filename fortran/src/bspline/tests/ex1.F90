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
subroutine span_test_1()
use spl_m_bspline
implicit none
     integer, parameter :: n = 4
     integer, parameter :: p = 2
     real(8), dimension(0:n+p) :: u
     integer, parameter :: nx = 5
     real(8), dimension(nx) :: x
     integer, dimension(nx) :: expected
     integer :: i
     integer :: span 

     print *, ">>>> span_test_1: begin "

     u        = (/0.0,  0.0, 0.0,  0.5, 1.0, 1.0, 1.0/)
     x        = (/0.0, 0.25, 0.5, 0.75, 1.0/)
     expected = (/  2,    2,   3,    3,   3/)

     do i = 1, nx
        span = findspan(n-1,p,x(i),u)
        print *, "x = ", x(i), "span = ", span, " expected = ", expected(i)
     end do

     print *, ">>>> span_test_1: end "
end subroutine span_test_1
! ............................................

! ............................................
subroutine span_test_2()
use spl_m_bspline
implicit none
     integer, parameter :: n = 4
     integer, parameter :: p = 2
     real(8), dimension(0:n+p) :: u
     integer, parameter :: nx = 5
     real(8), dimension(nx) :: x
     integer, dimension(nx) :: expected
     integer :: i
     integer :: span 

     print *, ">>>> span_test_2: begin "

     u        = (/-1.0,0.0,0.0, 0.5, 1.0,1.0, 2.0/)
     x        = (/0.0, 0.25, 0.5, 0.75, 1.0/)
     expected = (/  2,    2,   3,    3,  3/)

     do i = 1, nx
        span = findspan(n-1,p,x(i),u)
        print *, "x = ", x(i), "span = ", span, " expected = ", expected(i)
     end do

     print *, ">>>> span_test_2: end "
end subroutine span_test_2
! ............................................

! ............................................
subroutine span_test_3()
use spl_m_bspline
implicit none
     integer, parameter :: n = 3
     integer, parameter :: p = 2
     real(8), dimension(0:n+p) :: u
     integer, parameter :: nx = 3 
     real(8), dimension(nx) :: x
     integer, dimension(nx) :: expected
     integer :: i
     integer :: span 

     print *, ">>>> span_test_3: begin "

     u        = (/-2.0, -1.0, 0.0, 1.0, 2.0, 3.0/)
     x        = (/0.0, 0.5, 1.0/)
     expected = (/  2,   2,   2/)

     do i = 1, nx
        span = findspan(n-1,p,x(i),u)
        print *, "x = ", x(i), "span = ", span, " expected = ", expected(i)
     end do

     print *, ">>>> span_test_3: end "
end subroutine span_test_3
! ............................................

! ............................................
subroutine multiplicity_test_1()
use spl_m_bspline
implicit none
     integer, parameter :: n = 6
     integer, parameter :: p = 2
     real(8), dimension(0:n+p) :: u
     integer, parameter :: nx = 5 
     real(8), dimension(nx) :: x
     integer, dimension(nx) :: expected
     integer :: i
     integer :: span 
     integer :: mult 

     print *, ">>>> multiplicity_test_1: begin "

     u        = (/0.0,  0.0, 0.0, 0.25, 0.25, 0.5, 1.0, 1.0, 1.0/)
     x        = (/0.0, 0.25, 0.5, 0.75, 1.0/)
     expected = (/  3,    2,   1,    0, 3/)

     do i = 1, nx
        span = findspan(n-1,p,u(i),u)
        mult = findmult(span,x(i),p,u)
        print *, "u = ", x(i), "mult = ", mult, " expected = ", expected(i)
     end do

     print *, ">>>> multiplicity_test_1: end "
end subroutine multiplicity_test_1
! ............................................

! ............................................
subroutine multiplicity_test_2()
use spl_m_bspline
implicit none
     integer, parameter :: n = 6
     integer, parameter :: p = 2
     real(8), dimension(0:n+p) :: u
     integer, parameter :: nx = 5 
     real(8), dimension(nx) :: x
     integer, dimension(nx) :: expected
     integer :: i
     integer :: span 
     integer :: mult 

     print *, ">>>> multiplicity_test_2: begin "

     u        = (/-1.0,  0.0, 0.0, 0.25, 0.25, 0.5, 1.0, 1.0, 2.0/)
     x        = (/0.0, 0.25, 0.5, 0.75, 1.0/)
     expected = (/  2,    2,   1,    0, 2/)

     do i = 1, nx
        span = findspan(n-1,p,u(i),u)
        mult = findmult(span,x(i),p,u)
        print *, "u = ", x(i), "mult = ", mult, " expected = ", expected(i)
     end do

     print *, ">>>> multiplicity_test_2: end "
end subroutine multiplicity_test_2
! ............................................

! ............................................
subroutine multiplicity_test_3()
use spl_m_bspline
implicit none
     integer, parameter :: n = 6
     integer, parameter :: p = 2
     real(8), dimension(0:n+p) :: u
     integer, parameter :: nx = 5 
     real(8), dimension(nx) :: x
     integer, dimension(nx) :: expected
     integer :: i
     integer :: span 
     integer :: mult 

     print *, ">>>> multiplicity_test_3: begin "

     u        = (/-2.0,  -1.0, 0.0, 0.25, 0.25, 0.5, 1.0, 2.0, 3.0/)
     x        = (/0.0, 0.25, 0.5, 0.75, 1.0/)
     expected = (/  1,    2,   1,    0, 1/)

     do i = 1, nx
        span = findspan(n-1,p,u(i),u)
        mult = findmult(span,x(i),p,u)
        print *, "u = ", x(i), "mult = ", mult, " expected = ", expected(i)
     end do

     print *, ">>>> multiplicity_test_3: end "
end subroutine multiplicity_test_3
! ............................................


! ............................................
program main

  implicit none

  call span_test_1()
  call span_test_2()
  call span_test_3()
  call multiplicity_test_1()
  call multiplicity_test_2()
  call multiplicity_test_3()

end program main
! ............................................
