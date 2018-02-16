!> @brief 
!> mapping example in 2D 
!> @details
!> creates an eccentric annulus map  
!>  
!> usage:
!>   $> ./mapping_2d_ex17 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_eccentric_annulus
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping
  real(spl_rk), parameter :: r_min = 0.2_spl_rk
  real(spl_rk), parameter :: r_max = 1.0_spl_rk
  real(spl_rk), dimension(2) :: C
  
  
  ! ... new internal center
  C = (/0.5_spl_rk, 0.0_spl_rk /)

  ! ... creates an annulus map
  call spl_mapping_eccentric_annulus(mapping, r_min, r_max, C)
  ! ...

  ! ... prints info
  call mapping % print_info()
  ! ...

  ! ... exports mapping
  call mapping % export("mapping_2d_ex17.nml")
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
