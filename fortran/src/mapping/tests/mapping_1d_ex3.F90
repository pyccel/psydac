!> @brief 
!> mapping example in 1d 
!> @details
!> example of evaluation for a 1d mapping.
!> usage:
!>   $> ./mapping_1d_ex3 

! ............................................
program main
use spl_m_global
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_gallery, only: spl_mapping_linear 
implicit none
  type(spl_t_mapping_1d), target :: mapping
  real(spl_rk), dimension(1) :: P_1
  real(spl_rk), dimension(1) :: P_2
  real(spl_rk), dimension(3) :: X
  real(spl_rk), dimension(1,3) :: Y

  ! ... create a linear map
  P_1 = (/ 0.0_spl_rk /)
  P_2 = (/ 1.0_spl_rk /)

  call spl_mapping_linear( mapping, &
                         & P_1, P_2, &
                         & degree=1, n_elements=4)
  ! ...

  ! ... mapping evaluation
  X = (/ 0.2_spl_rk, 0.4_spl_rk, 0.6_spl_rk /)

  call mapping % evaluate(X, Y)
  print *, Y
  ! ...

  ! ... deallocate memory
  call mapping % free() 
  ! ...

end program main
! ............................................
