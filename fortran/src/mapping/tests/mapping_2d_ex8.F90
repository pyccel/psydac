!> @brief 
!> mapping example in 1d 
!> @details
!> creates an annulus map and evaluates on given sites. 
!>  
!> usage:
!>   $> ./mapping_2d_ex8 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_annulus
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping
  real(spl_rk), parameter :: r_min = 0.5_spl_rk
  real(spl_rk), parameter :: r_max = 1.0_spl_rk
  real(spl_rk), dimension(3) :: U
  real(spl_rk), dimension(1) :: V
  real(spl_rk), dimension(2,3,1) :: Y
  real(spl_rk), dimension(2,2,3,1) :: dY
  real(spl_rk), dimension(3,2,3,1) :: d2Y
  integer :: i
  integer :: j 
  real(spl_rk) :: r

  ! ... creates an annulus map
  call spl_mapping_annulus(mapping, r_min, r_max)
  ! ...

  ! ... prints info
  call mapping % print_info()
  ! ...

  ! ... mapping evaluation
  U = (/ 0.2_spl_rk, 0.4_spl_rk, 0.6_spl_rk /)
  V = (/ 0.5_spl_rk /)

  call mapping % evaluate(U, V, Y)
  call mapping % evaluate_deriv(U, V, Y, dY, d2Y)

  print *, "  u,   v,   x,   y,   r"
  j = 1
  do i=1, 3
    r = sqrt(Y(1,i,j)**2 + Y(2,i,j)**2) 
    print *, U(i), V(j), Y(1,i,j), Y(2,i,j), r
  end do
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
