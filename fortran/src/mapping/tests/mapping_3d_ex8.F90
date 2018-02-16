!> @brief 
!> mapping example in 3d 
!> @details
!> creates a cylinder map by extruding an annulus and evaluates on given sites. 
!> usage:
!>   $> ./mapping_3d_ex8 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_3d,      only: spl_t_mapping_3d
use spl_m_mapping_gallery, only: spl_mapping_annulus
use spl_m_mapping_cad,     only: spl_t_mapping_cad
implicit none
  ! local
  real(plf_rk), dimension(2) :: U
  real(plf_rk), dimension(1) :: V
  real(plf_rk), dimension(1) :: W 
  real(plf_rk), dimension(  3,2,1,1) :: Y
  real(plf_rk), dimension(3,3,2,1,1) :: dY
  real(plf_rk), dimension(6,3,2,1,1) :: d2Y
  type(spl_t_mapping_2d), target :: annulus
  type(spl_t_mapping_3d), target :: mapping
  type(spl_t_mapping_cad), target :: cad
  real(plf_rk), parameter :: r_min = 0.5_plf_rk
  real(plf_rk), parameter :: r_max = 1.0_plf_rk
  integer :: i
  integer :: j 
  integer :: k 
  real(plf_rk) :: r

  ! ... creates an annulus
  call spl_mapping_annulus(annulus, r_min, r_max)
  ! ...

  ! ... extrude the annulus to get a cylinder
  call cad % create()
  call cad % extrude(annulus, (/ 0.0_plf_rk, 0.0_plf_rk, 1.0_plf_rk /), mapping)
  ! ...

  ! ... evaluation on the cylinder
  U = (/ 0.2_plf_rk, 0.6_plf_rk /)
  V = (/ 0.0_plf_rk /)
  w = (/ 0.3_plf_rk /)

  call mapping % evaluate_deriv(U, V, W, Y, dY, d2Y)

  print *, "  u,   v,   w,   x,   y,   z,   r"
  j = 1
  k = 1
  do i=1,2 
    r = sqrt(Y(1,i,j,k)**2 + Y(2,i,j,k)**2) 
    print *, U(i), V(j), W(k), Y(1,i,j,k), Y(2,i,j,k), Y(3,i,j,k), r
  end do
  ! ...

  ! ... deallocates memory
  call cad % free()
  call mapping % free() 
  call annulus % free() 
  ! ...

end program main
! ............................................
