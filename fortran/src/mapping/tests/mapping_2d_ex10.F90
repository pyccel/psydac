!> @brief 
!> mapping example in 2d 
!> @details
!> creates a bilinear map and translates it
!> usage:
!>   $> ./mapping_2d_ex10 

! ............................................
program main
use spl_m_global
use spl_m_mapping_cad,     only: spl_t_mapping_cad
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_bilinear
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping
  type(spl_t_mapping_cad), target :: cad
  real(spl_rk), dimension(2) :: P_11
  real(spl_rk), dimension(2) :: P_12
  real(spl_rk), dimension(2) :: P_21
  real(spl_rk), dimension(2) :: P_22

  ! ... creates a bilinear map
  P_11 = (/ 0.0_spl_rk, 0.0_spl_rk /)
  P_21 = (/ 1.0_spl_rk, 0.0_spl_rk /)
  P_12 = (/ 0.0_spl_rk, 1.0_spl_rk /)
  P_22 = (/ 1.0_spl_rk, 1.0_spl_rk /)

  call spl_mapping_bilinear(mapping, P_11, P_12, P_21, P_22)
  ! ...

  ! ... prints info before translation
  print *, "======= Before ====== "
  call mapping % print_info()
  ! ...

  ! ... mapping translation
  call cad % create()
  call cad % translate(mapping, (/ 0.5_spl_rk, -0.5_spl_rk /))
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
