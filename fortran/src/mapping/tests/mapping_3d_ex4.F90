!> @brief 
!> mapping example in 3d 
!> @details
!> example of derivatives evaluation for a 3d mapping.
!> usage:
!>   $> ./mapping_3d_ex4 

! ............................................
program main
use spl_m_global
use spl_m_mapping_3d,      only: spl_t_mapping_3d
use spl_m_mapping_gallery, only: spl_mapping_trilinear
implicit none
  ! local
  type(spl_t_mapping_3d), target :: mapping
  real(spl_rk), dimension(3) :: P_111
  real(spl_rk), dimension(3) :: P_121
  real(spl_rk), dimension(3) :: P_211
  real(spl_rk), dimension(3) :: P_221
  real(spl_rk), dimension(3) :: P_112
  real(spl_rk), dimension(3) :: P_122
  real(spl_rk), dimension(3) :: P_212
  real(spl_rk), dimension(3) :: P_222
  real(spl_rk), dimension(2) :: U
  real(spl_rk), dimension(1) :: V
  real(spl_rk), dimension(1) :: W 
  real(spl_rk), dimension(  3,2,1,1) :: Y
  real(spl_rk), dimension(3,3,2,1,1) :: dY
  real(spl_rk), dimension(6,3,2,1,1) :: d2Y
  integer :: n

  ! ... createsa trilinear map
  P_111 = (/ 0.0_spl_rk, 0.0_spl_rk, 0.0_spl_rk /)
  P_211 = (/ 1.0_spl_rk, 0.0_spl_rk, 0.0_spl_rk /)
  P_121 = (/ 0.0_spl_rk, 1.0_spl_rk, 0.0_spl_rk /)
  P_221 = (/ 1.0_spl_rk, 1.0_spl_rk, 0.0_spl_rk /)
  P_112 = (/ 0.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /)
  P_212 = (/ 1.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /)
  P_122 = (/ 0.0_spl_rk, 1.0_spl_rk, 1.0_spl_rk /)
  P_222 = (/ 1.0_spl_rk, 1.0_spl_rk, 1.0_spl_rk /)

  call spl_mapping_trilinear(mapping, &
    & P_111, P_121, P_211, P_221, &
    & P_112, P_122, P_212, P_222, &
    & degrees=(/1,1,1/), n_elements=(/0,0,0/))
  ! ...

  ! ... mapping evaluation
  U = (/ 0.2_spl_rk, 0.6_spl_rk /)
  V = (/ 0.1_spl_rk /)
  w = (/ 0.3_spl_rk /)

  call mapping % evaluate_deriv(U, V, W, Y, dY, d2Y)
  print *, Y
  print *, dY(1,:,:,:,:)
  print *, dY(2,:,:,:,:)
  print *, dY(3,:,:,:,:)
  print *, d2Y(1,:,:,:,:)
  print *, d2Y(2,:,:,:,:)
  print *, d2Y(3,:,:,:,:)
  print *, d2Y(4,:,:,:,:)
  print *, d2Y(5,:,:,:,:)
  print *, d2Y(6,:,:,:,:)
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
