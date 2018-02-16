!> @brief 
!> mapping example in 3d 
!> @details
!> creates a trilinear map and translates it
!> usage:
!>   $> ./mapping_3d_ex10 

! ............................................
program main
use spl_m_global
use spl_m_mapping_cad,     only: spl_t_mapping_cad
use spl_m_mapping_3d,      only: spl_t_mapping_3d
use spl_m_mapping_gallery, only: spl_mapping_trilinear
implicit none
  ! local
  type(spl_t_mapping_3d), target :: mapping
  type(spl_t_mapping_cad), target :: cad
  real(spl_rk), dimension(3) :: P_111
  real(spl_rk), dimension(3) :: P_121
  real(spl_rk), dimension(3) :: P_211
  real(spl_rk), dimension(3) :: P_221
  real(spl_rk), dimension(3) :: P_112
  real(spl_rk), dimension(3) :: P_122
  real(spl_rk), dimension(3) :: P_212
  real(spl_rk), dimension(3) :: P_222

  ! ... creates a trilinear map
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
    & P_112, P_122, P_212, P_222)
  ! ...

  ! ... prints info before translation
  print *, "======= Before ====== "
  call mapping % print_info()
  ! ...

  ! ... mapping translation
  call cad % create()
  call cad % translate(mapping, (/ 0.5_spl_rk, -0.5_spl_rk, 1.0_spl_rk /))
  ! ...

  ! ... prints info after translation
  print *, "======= After  ====== "
  call mapping % print_info()
  ! ...

  ! ... deallocates memory
  call cad % free()
  call mapping % free() 
  ! ...

end program main
! ............................................
