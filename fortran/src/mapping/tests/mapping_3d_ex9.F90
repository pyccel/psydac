!> @brief 
!> mapping example in 3d 
!> @details
!> example of derivatives evaluation for a 3d mapping.
!> usage:
!>   $> ./mapping_3d_ex9 

! ............................................
program main
use spl_m_global
use spl_m_mapping_3d,           only: spl_t_mapping_3d
use spl_m_mapping_2d,           only: spl_t_mapping_2d
use spl_m_mapping_cad,          only: spl_t_mapping_cad
use spl_m_mapping_gallery,      only: spl_mapping_circle
implicit none
  ! local
  type(spl_t_mapping_3d),           target      :: mapping
  type(spl_t_mapping_2d),           target      :: circle 
  type(spl_t_mapping_cad),          target      :: cad
  integer, parameter :: n_u = 3 
  integer, parameter :: n_v = 1
  integer, parameter :: n_w = 1 
  real(spl_rk), dimension(n_u) :: U
  real(spl_rk), dimension(n_v) :: V
  real(spl_rk), dimension(n_w) :: W 
  real(spl_rk), dimension(  3,n_u,n_v,n_w) :: Y
  real(spl_rk), dimension(3,3,n_u,n_v,n_w) :: dY
  real(spl_rk), dimension(6,3,n_u,n_v,n_w) :: d2Y
  integer :: n
  integer :: i
  integer :: j 
  integer :: k 
  real(spl_rk) :: r

  ! ... creates a circular mapping 
  call spl_mapping_circle(circle, radius=1.0_spl_rk)
  call circle % export('circle.nml')
  ! ... 

  ! ... extrude the 2d map into a 3d map
  call cad % create()
  call cad % extrude(circle, (/ 0.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /), mapping)
  call mapping % export('mapping.nml')
  ! ... 

  ! ... mapping evaluation
  U = (/ 0.2_spl_rk, 0.4_spl_rk, 0.6_spl_rk /)
  V = (/ 0.0_spl_rk /)
  w = (/ 0.0_spl_rk /)

  call mapping % evaluate_deriv(U, V, W, Y, dY, d2Y)
  ! ...

  ! ...
  print *, "  u,   v,   w,   x,   y,   z,   r"
  j = 1
  k = 1
  do i=1, 3
    r = sqrt(Y(1,i,j,k)**2 + Y(2,i,j,k)**2) 
    print *, U(i), V(j), W(k), Y(1,i,j,k), Y(2,i,j,k), Y(3,i,j,k), r
  end do
  ! ...

  ! ...
  print *, "  x_u,   y_u,   z_u"
  j = 1
  k = 1
  do i=1, 3
    print *, dY(1,1,i,j,k), dY(1,2,i,j,k), dY(1,3,i,j,k)
  end do
  ! ...

  ! ...
  print *, "  x_v,   y_v,   z_v"
  j = 1
  k = 1
  do i=1, 3
    print *, dY(2,1,i,j,k), dY(2,2,i,j,k), dY(2,3,i,j,k)
  end do
  ! ...

  ! ...
  print *, "  x_w,   y_w,   z_w"
  j = 1
  k = 1
  do i=1, 3
    print *, dY(3,1,i,j,k), dY(3,2,i,j,k), dY(3,3,i,j,k)
  end do
  ! ...

  ! ... deallocates memory
  call cad % free()
  call mapping % free()
  call circle % free()
  ! ...

end program main
! ............................................
