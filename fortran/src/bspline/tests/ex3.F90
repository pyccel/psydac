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
subroutine evalbasisfunsders_test_1()
use spl_m_bsp
implicit none
     integer, parameter :: n = 7 
     integer, parameter :: p = 3 
     real*8, dimension(n+p+1) :: u
     integer, parameter :: nx = 5000
     integer, parameter :: nderiv = 2
     real*8, dimension(nx) :: x
     real*8, dimension(p+1,nderiv+1) :: dbatx
     integer :: i
     integer :: span 

     print *, ">>>> evalbasisfunsders_test_1: begin "

     u        = (/0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0/)

     do i = 1, nx
        x(i) = float(i) / float(nx) 
     end do

     do i = 1, nx
        call findspan(p,n,u,x(i),span)
        call evalbasisfunsders(p,n,u,x(i),nderiv,span,dbatx)
!        print *, ">>>>"
!        print *, x(i)
!        print *, dbatx
     end do

     print *, ">>>> evalbasisfunsders_test_1: end "

end subroutine evalbasisfunsders_test_1
! ............................................

! ............................................
program main

  implicit none

  call evalbasisfunsders_test_1()

end program main
! ............................................
