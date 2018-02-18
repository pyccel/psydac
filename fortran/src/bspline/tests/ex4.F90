!
! file: ex1.f90
!
!
! usage:
!   > ./ex4
!
! authors:
!   ahmed ratnani  - ratnaniahmed@gmail.com
!

! ............................................
program main
implicit none

  ! ...
!  call test_1()
!  call test_2()
!  call test_3()
  call test_4()
  ! ...

contains
  ! ............................................
  subroutine test_1()
  use spl_m_bsp,       only: spl_collocation_matrix
  use spl_m_bsp,       only: spl_compute_greville
  use spl_m_utilities, only: spl_print_matrix
  use spl_m_utilities, only: spl_print_array
  implicit none
    integer, parameter :: n = 6 
    integer, parameter :: p = 2
    integer, parameter :: n_points = n 
    real(8), dimension(1:n+p+1)     :: knots 
    real(8), dimension(n_points)    :: arr_x
    real(8), dimension(n_points, n) :: mat 

    knots = (/0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0/)

    call spl_compute_greville(n, p, knots, arr_x) 
    call spl_print_array( "Greville points", n_points, arr_x )

    call spl_collocation_matrix(n, p, knots, arr_x, mat) 
    call spl_print_matrix( "collocation matrix", n_points, n, mat, n ) 

  end subroutine test_1
  ! ............................................

  ! ............................................
  subroutine test_2()
  use spl_m_bsp,       only: spl_collocation_periodic_matrix
  use spl_m_bsp,       only: spl_compute_greville
  use spl_m_bsp,       only: spl_symetrize_knots
  use spl_m_utilities, only: spl_print_matrix
  use spl_m_utilities, only: spl_print_array
  implicit none
    integer, parameter :: n = 7 
    integer, parameter :: p = 2
    integer, parameter :: nu = p
    integer, parameter :: n_points = n-nu 
    real(8), dimension(1:n+p+1)     :: knots 
    real(8), dimension(n)    :: arr_x
    real(8), dimension(:,:), allocatable :: mat 
    integer :: i

    knots = (/0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0/)

    ! ...
    call spl_compute_greville(n-nu, p, knots(1:n+p+1-nu), arr_x) 
    call spl_symetrize_knots(nu-1, n, p, knots) 
    call spl_print_array( "knots", n+p+1, knots )
    ! ...

    ! ...
    call spl_print_array( "arr_x", n, arr_x )
    ! ...

    ! ...
    call spl_collocation_periodic_matrix(nu-1, n, p, knots, arr_x(2:n-1), mat) 
    call spl_print_matrix( "periodic collocation matrix", n_points, n-nu, mat, n-nu ) 

    deallocate(mat)
    ! ...

  end subroutine test_2
  ! ............................................

  ! ............................................
  subroutine test_3()
  use spl_m_bsp,       only: spl_collocation_periodic_matrix
  use spl_m_bsp,       only: spl_symetrize_knots
  use spl_m_utilities, only: spl_print_matrix
  use spl_m_utilities, only: spl_print_array
  implicit none
    integer, parameter :: n = 8 
    integer, parameter :: p = 3
    integer, parameter :: nu = p
    integer, parameter :: n_points = n-nu 
    real(8), dimension(1:n+p+1)     :: knots 
    real(8), dimension(n)    :: arr_x
    real(8), dimension(:,:), allocatable :: mat 
    integer :: i

    knots = (/0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0/)

    ! ...
    call spl_symetrize_knots(nu-1, n, p, knots) 
    call spl_print_array( "knots", n+p+1, knots )
    ! ...

    ! ...
    do i=1, n_points
      arr_x(i) = knots(p+1) + (i-1)*(knots(n+1) - knots(p+1))/(n_points) 
    end do
    call spl_print_array( "arr_x", n_points, arr_x )
    ! ...

    ! ...
    call spl_collocation_periodic_matrix(nu-1, n, p, knots, arr_x, mat) 
    call spl_print_matrix( "periodic collocation matrix", n_points, n-nu, mat, n-nu ) 

    deallocate(mat)
    ! ...

  end subroutine test_3
  ! ............................................

  ! ............................................
  subroutine test_4()
  use spl_m_bsp,       only: spl_collocation_periodic_matrix
  use spl_m_bsp,       only: spl_symetrize_knots
  use spl_m_utilities, only: spl_print_matrix
  use spl_m_utilities, only: spl_print_array
  implicit none
    integer, parameter :: n = 10 
    integer, parameter :: p = 5 
    integer, parameter :: nu = p
    integer, parameter :: n_points = n-nu 
    real(8), dimension(1:n+p+1)     :: knots 
    real(8), dimension(n)    :: arr_x
    real(8), dimension(:,:), allocatable :: mat 
    integer :: i

    knots = (/0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0/)

    ! ...
    call spl_symetrize_knots(nu-1, n, p, knots) 
    call spl_print_array( "knots", n+p+1, knots )
    ! ...

    allocate(mat(n, n))
    ! ...
    do i=1, n_points
      arr_x(i) = knots(p+1) + (i-1)*(knots(n+1) - knots(p+1))/(n_points+1) 
    end do
    call spl_print_array( "arr_x", n_points, arr_x )
    ! ...

    ! ...
    call spl_collocation_periodic_matrix(nu-1, n, p, knots, arr_x, mat) 
    call spl_print_matrix( "periodic collocation matrix", n_points, n-nu, mat, n-nu ) 

    deallocate(mat)
    ! ...

  end subroutine test_4
  ! ............................................

end program main
! ............................................
