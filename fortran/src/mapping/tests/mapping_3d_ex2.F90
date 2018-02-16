!> @brief 
!> mapping example in 3d 
!> @details
!> example of clamp/unclamp for a 3d mapping.
!> usage:
!>   $> ./mapping_3d_ex2 

! ............................................
program main
use spl_m_global
use spl_m_mapping_3d,      only: spl_t_mapping_3d
use spl_m_mapping_gallery, only: spl_mapping_trilinear
implicit none
  type(spl_t_mapping_3d), target :: mapping
  real(spl_rk), dimension(3) :: P_111
  real(spl_rk), dimension(3) :: P_121
  real(spl_rk), dimension(3) :: P_211
  real(spl_rk), dimension(3) :: P_221
  real(spl_rk), dimension(3) :: P_112
  real(spl_rk), dimension(3) :: P_122
  real(spl_rk), dimension(3) :: P_212
  real(spl_rk), dimension(3) :: P_222

  ! ... creates a trilinear map
  P_111 = (/ 0.0_spl_rk, 0.0_spl_rk, 0.0_spl_rk /)
  P_211 = (/ 1.0_spl_rk, 0.0_spl_rk, 0.0_spl_rk /)
  P_121 = (/ 0.0_spl_rk, 1.0_spl_rk, 0.0_spl_rk /)
  P_221 = (/ 1.0_spl_rk, 1.0_spl_rk, 0.0_spl_rk /)
  P_112 = (/ 0.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /)
  P_212 = (/ 1.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /)
  P_122 = (/ 0.0_spl_rk, 1.0_spl_rk, 1.0_spl_rk /)
  P_222 = (/ 1.0_spl_rk, 1.0_spl_rk, 1.0_spl_rk /)

  call spl_mapping_trilinear(mapping, &
    & P_111, P_121, P_211, P_221, &
    & P_112, P_122, P_212, P_222, &
    & degrees=(/2,2,2/), n_elements=(/2,2,2/))
  ! ...

  ! ... prints info
  print *, ">>>>>>>>>> initial mapping "
  call mapping % print_info()
  ! ...

  ! ... unclamp and prints info
  call mapping % unclamp()
  print *, ">>>>>>>>>> unclamped mapping "
  call mapping % print_info()
  ! ...

  ! ... clamp and prints info
  call mapping % clamp()
  print *, ">>>>>>>>>> clamped mapping "
  call mapping % print_info()
  ! ...

  ! ... deallocate memory
  call mapping % free() 
  ! ...

end program main
! ............................................
