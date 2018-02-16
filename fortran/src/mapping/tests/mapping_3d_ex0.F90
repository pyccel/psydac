!> @brief 
!> mapping example in 3d 
!> @details
!> example of export/read_from_file for a 3d mapping.
!> usage:
!>   $> ./mapping_3d_ex0 

! ............................................
program main
use spl_m_global
use spl_m_mapping_3d,      only: spl_t_mapping_3d
use spl_m_mapping_gallery, only: spl_mapping_trilinear
implicit none
  ! local
  type(spl_t_mapping_3d), target :: mapping_in
  type(spl_t_mapping_3d), target :: mapping_out
  real(spl_rk), dimension(3) :: P_111
  real(spl_rk), dimension(3) :: P_121
  real(spl_rk), dimension(3) :: P_211
  real(spl_rk), dimension(3) :: P_221
  real(spl_rk), dimension(3) :: P_112
  real(spl_rk), dimension(3) :: P_122
  real(spl_rk), dimension(3) :: P_212
  real(spl_rk), dimension(3) :: P_222

  ! ... creates a trilinear map
  P_111 = (/ -2.0_spl_rk, -2.0_spl_rk, -2.0_spl_rk /)
  P_211 = (/  2.0_spl_rk, -2.0_spl_rk, -2.0_spl_rk /)
  P_121 = (/ -2.0_spl_rk,  2.0_spl_rk, -2.0_spl_rk /)
  P_221 = (/  2.0_spl_rk,  2.0_spl_rk, -2.0_spl_rk /)
  P_112 = (/ -2.0_spl_rk, -2.0_spl_rk,  2.0_spl_rk /)
  P_212 = (/  2.0_spl_rk, -2.0_spl_rk,  2.0_spl_rk /)
  P_122 = (/ -2.0_spl_rk,  2.0_spl_rk,  2.0_spl_rk /)
  P_222 = (/  2.0_spl_rk,  2.0_spl_rk,  2.0_spl_rk /)

  call spl_mapping_trilinear(mapping_in, &
    & P_111, P_121, P_211, P_221, &
    & P_112, P_122, P_212, P_222)
  ! ...

  ! ... exports the mapping
  call mapping_in % export('mapping_3d_ex0_in.nml')
  ! ...

  ! ... read the exported mapping and then re-export it again
  call mapping_out % read_from_file('mapping_3d_ex0_in.nml')
  call mapping_out % export('mapping_3d_ex0_out.nml')
  ! ...

  ! ... dealloacates memory
  call mapping_out % free() 
  call mapping_in % free() 
  ! ...

end program main
! ............................................
