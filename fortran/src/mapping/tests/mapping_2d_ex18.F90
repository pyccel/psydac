!> @brief 
!> mapping example in 2D 
!> @details
!> creates an eccentric annulus map  
!>  
!> usage:
!>   $> ./mapping_2d_ex18 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_eccentric_annulus
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping
  real(plf_rk), parameter :: r_min = 0.2_plf_rk
  real(plf_rk), parameter :: r_max = 1.0_plf_rk
  real(plf_rk), dimension(2) :: C1, C2
  
  
  ! ... new internal center
  C1 = (/10.0_plf_rk, 0.0_plf_rk /)
  C2 = (/10.5_plf_rk, 0.0_plf_rk /)

  ! ... creates an annulus map
  call spl_mapping_eccentric_annulus(mapping, r_min, r_max, C1, C2)
  ! ...

  ! ... prints info
  call mapping % print_info()
  ! ...

  ! ... exports mapping
  call mapping % export("mapping_2d_ex18.nml")
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
