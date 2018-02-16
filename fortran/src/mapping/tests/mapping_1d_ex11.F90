!> @brief 
!> mapping example in 1d 
!> @details
!> creates a linear map, then extrudes it to a 2d map
!> usage:
!>   $> ./mapping_1d_ex11 

! ............................................
program main
use spl_m_global
use spl_m_mapping_cad,     only: spl_t_mapping_cad
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_linear
implicit none
  ! local
  type(spl_t_mapping_1d), target :: mapping_in
  type(spl_t_mapping_2d), target :: mapping_out
  type(spl_t_mapping_cad), target :: cad
  real(spl_rk), dimension(1) :: P_1
  real(spl_rk), dimension(1) :: P_2

  ! ... create a linear map
  P_1 = (/ 0.0_spl_rk /)
  P_2 = (/ 1.0_spl_rk /)

  call spl_mapping_linear( mapping_in, &
                         & P_1, P_2)
  ! ...

  ! ... extrudes the mapping
  call cad % create()
  call cad % extrude(mapping_in, (/ 0.0_spl_rk, 1.0_spl_rk /), mapping_out)
  ! ...

  ! ... prints info
  call mapping_out % print_info()
  ! ...

  ! ... deallocates memory
  call cad % free()
  call mapping_out % free() 
  call mapping_in % free() 
  ! ...

end program main
! ............................................
