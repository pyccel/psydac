!> @brief 
!> mapping example in 1d 
!> @details
!> example of mapping translation 
!> usage:
!>   $> ./mapping_1d_ex10 

! ............................................
program main
use spl_m_global
use spl_m_mapping_cad,     only: spl_t_mapping_cad
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_gallery, only: spl_mapping_linear
implicit none
  ! local
  type(spl_t_mapping_1d), target :: mapping
  type(spl_t_mapping_cad), target :: cad
  real(spl_rk), dimension(1) :: P_1
  real(spl_rk), dimension(1) :: P_2

  ! ... create a linear map
  P_1 = (/ 0.0_spl_rk /)
  P_2 = (/ 1.0_spl_rk /)

  call spl_mapping_linear(mapping, P_1, P_2)
  ! ...

  ! ... prints info 
  print *, "======= Before ====== "
  call mapping % print_info()
  ! ...

  ! ... mapping translation
  call cad % create()
  call cad % translate(mapping, (/ 0.5_spl_rk /))
  ! ...

  ! ... prints info 
  print *, "======= After  ====== "
  call mapping % print_info()
  ! ...

  ! ... deallocate memory
  call mapping % free() 
  call cad % free()
  ! ...

end program main
! ............................................
