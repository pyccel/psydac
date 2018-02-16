!> @brief 
!> mapping example in 1d 
!> @details
!> example of export/read_from_file for a 1d mapping.
!> usage:
!>   $> ./mapping_1d_ex0 
   
! ............................................
program main
use spl_m_global
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_gallery, only: spl_mapping_linear 
implicit none
  type(spl_t_mapping_1d), target :: mapping_in
  type(spl_t_mapping_1d), target :: mapping_out
  real(spl_rk), dimension(2) :: P_1
  real(spl_rk), dimension(2) :: P_2

  ! ... creates a linear map between P_1 and P_2
  P_1 = (/ 0.0_spl_rk, 1.0_spl_rk /)
  P_2 = (/ 1.0_spl_rk, 2.0_spl_rk /)

  call spl_mapping_linear( mapping_in, &
                         & P_1, P_2, &
                         & degree=1, n_elements=2)
  ! ...

  ! ... exports the mapping
  call mapping_in % export('mapping_1d_ex0_in.nml')
  ! ...

  ! ... read the exported mapping and then re-export it again
  call mapping_out % read_from_file('mapping_1d_ex0_in.nml')
  call mapping_out % export('mapping_1d_ex0_out.nml')
  ! ...

  ! ...
  call mapping_out % free() 
  call mapping_in % free() 
  ! ...

end program main
! ............................................
