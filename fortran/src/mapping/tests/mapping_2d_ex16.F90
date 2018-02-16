!> @brief 
!> mapping example in 2d 
!> @details
!> creates an annulus map and transpose it. 
!>  
!> usage:
!>   $> ./mapping_2d_ex16 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_annulus
use spl_m_mapping_cad,     only: spl_t_mapping_cad
implicit none
  ! local
  type(spl_t_mapping_cad), target :: cad
  type(spl_t_mapping_2d), target :: mapping_in
  type(spl_t_mapping_2d), target :: mapping_out
  real(spl_rk), parameter :: r_min = 0.5_spl_rk
  real(spl_rk), parameter :: r_max = 1.0_spl_rk

  ! ... creates an annulus map
  call spl_mapping_annulus(mapping_in, r_min, r_max)
  ! ...

  ! ... mapping transposition 
  call cad % create()
  call cad % transpose(mapping_in, mapping_out)
  ! ...

  ! ... prints info
  print *, "===== mapping_in  ====="
  call mapping_in % print_info()

  print *, "===== mapping_out ====="
  call mapping_out % print_info()
  ! ...

  ! ... deallocates memory
  call mapping_in % free() 
  call mapping_out % free() 
  ! ...

end program main
! ............................................
