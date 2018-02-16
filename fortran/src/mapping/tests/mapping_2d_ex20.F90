!> @brief 
!> mapping example in 2d 
!> @details
!> example of composition and evaluation for a 2d mapping.
!> usage:
!>   $> ./mapping_2d_ex20 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_bilinear
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping_1
  type(spl_t_mapping_2d), target :: mapping_2
  real(spl_rk), dimension(2) :: P_11
  real(spl_rk), dimension(2) :: P_12
  real(spl_rk), dimension(2) :: P_21
  real(spl_rk), dimension(2) :: P_22
  real(spl_rk), dimension(3) :: U
  real(spl_rk), dimension(1) :: V
  real(spl_rk), dimension(2,3,1) :: Y
  real(spl_rk), dimension(2,3,1) :: Z

  ! ... creates a bilinear map
  P_11 = (/ 0.0_spl_rk, 0.0_spl_rk /)
  P_21 = (/ 2.0_spl_rk, 0.0_spl_rk /)
  P_12 = (/ 0.0_spl_rk, 1.0_spl_rk /)
  P_22 = (/ 2.0_spl_rk, 1.0_spl_rk /)

  call spl_mapping_bilinear( mapping_1, &
                           & P_11, P_12, P_21, P_22, &
                           & degrees=(/1,1/), n_elements=(/2,2/))
  ! ...

  ! ... mapping evaluation
  U = (/ 0.2_spl_rk, 0.4_spl_rk, 0.6_spl_rk /)
  V = (/ 0.2_spl_rk /)

  call mapping_1 % evaluate(U, V, Y)
  print *, Y
  ! ...

  ! ...
  call spl_mapping_bilinear( mapping_2, &
                           & P_11, P_12, P_21, P_22, &
                           & degrees=(/1,1/), n_elements=(/2,2/), &
                           & other = mapping_1)
  
  call mapping_2 % evaluate(U, V, Z)
  print *, Z
  ! ...
  
  ! ... deallocates memory
  call mapping_1 % free() 
  call mapping_2 % free() 
  ! ...

end program main
! ............................................
