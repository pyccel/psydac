!> @brief 
!> mapping example in 2d 
!> @details
!> example of export/read_from_file for a 2d mapping.
!> usage:
!>   $> ./mapping_2d_ex0 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_bilinear
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping_in
  type(spl_t_mapping_2d), target :: mapping_out
  real(spl_rk), dimension(2) :: P_11
  real(spl_rk), dimension(2) :: P_12
  real(spl_rk), dimension(2) :: P_21
  real(spl_rk), dimension(2) :: P_22

  ! ... creates a bilinear map
  P_11 = (/ -2.0_spl_rk, -2.0_spl_rk /)
  P_21 = (/  2.0_spl_rk, -2.0_spl_rk /)
  P_12 = (/ -2.0_spl_rk,  2.0_spl_rk /)
  P_22 = (/  2.0_spl_rk,  2.0_spl_rk /)

  call spl_mapping_bilinear(mapping_in, P_11, P_12, P_21, P_22)
  ! ...

  ! ... exports the mapping
  call mapping_in % export('mapping_2d_ex0_in.nml')
  ! ...

  ! ... read the exported mapping and then re-export it again
  call mapping_out % read_from_file('mapping_2d_ex0_in.nml')
  call mapping_out % export('mapping_2d_ex0_out.nml')
  ! ...

  ! ... deallocates memory
  call mapping_out % free() 
  call mapping_in % free() 
  ! ...

end program main
! ............................................
