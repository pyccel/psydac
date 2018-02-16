!> @brief 
!> mapping example in 3d 
!> @details
!> this example shows the construction of a cylinder domain 
!> by extruding an annulus
!> usage:
!>   $> ./mapping_3d_ex11 

! ............................................
program main
use spl_m_global
use spl_m_mapping_cad,     only: spl_t_mapping_cad
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_3d,      only: spl_t_mapping_3d
use spl_m_mapping_gallery, only: spl_mapping_annulus
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping_in
  type(spl_t_mapping_3d), target :: mapping_out
  type(spl_t_mapping_cad), target :: cad
  real(spl_rk), parameter :: r_min = 0.5_spl_rk
  real(spl_rk), parameter :: r_max = 1.0_spl_rk

  ! ... creates a 2d annulus
  call spl_mapping_annulus(mapping_in, r_min, r_max)
  ! ...

  ! ... extrude the annulus to get a cylinder
  call cad % create()
  call cad % extrude(mapping_in, (/ 0.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /), mapping_out)
  ! ...

  ! ... prints info for the cylinder map
  call mapping_out % print_info()
  ! ...

  ! ... deallocates memory
  call cad % free()
  call mapping_out % free() 
  call mapping_in % free() 
  ! ...

end program main
! ............................................
