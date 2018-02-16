!> @brief 
!> mapping example in 2d 
!> @details
!> creates a circular map and evaluates on given sites 
!> usage:
!>   $> ./mapping_2d_ex9 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_circle
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping
  real(spl_rk), parameter :: radius = 0.5_spl_rk
  real(spl_rk), dimension(3) :: U
  real(spl_rk), dimension(1) :: V
  real(spl_rk), dimension(2,3,1) :: Y
  real(spl_rk), dimension(2,2,3,1) :: dY
  real(spl_rk), dimension(3,2,3,1) :: d2Y
  integer :: i
  integer :: j 
  real(spl_rk) :: r

  ! ... creates a circular map
  call spl_mapping_circle(mapping)
  ! ...

  ! ... prints info
  call mapping % print_info()
  ! ...

  ! ... mapping evaluation 
  U = (/ 0.2_spl_rk, 0.4_spl_rk, 0.6_spl_rk /)
  V = (/ 0.0_spl_rk /)

  call mapping % evaluate(U, V, Y)
  call mapping % evaluate_deriv(U, V, Y, dY, d2Y)

  print *, "  u,   v,   x,   y,   r"
  j = 1
  do i=1, 3
    r = sqrt(Y(1,i,j)**2 + Y(2,i,j)**2) 
    print *, U(i), V(j), Y(1,i,j), Y(2,i,j), r
  end do
  ! ...

  ! ...
  print *, "  x_u,   y_u"
  j = 1
  do i=1, 3
    print *, dY(1,1,i,j), dY(1,2,i,j)
  end do
  ! ...

  ! ...
  print *, "  x_v,   y_v"
  j = 1
  do i=1, 3
    print *, dY(2,1,i,j), dY(2,2,i,j)
  end do
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
