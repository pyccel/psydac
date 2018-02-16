!> @brief 
!> mapping example in 2d 
!> @details
!> creates a bilinear map and prints its info
!> usage:
!>   $> ./mapping_2d_ex1 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_bilinear 
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping
  real(spl_rk), dimension(2) :: P_11
  real(spl_rk), dimension(2) :: P_12
  real(spl_rk), dimension(2) :: P_21
  real(spl_rk), dimension(2) :: P_22
  integer, parameter :: p_u = 3 
  integer, parameter :: p_v = 2 
  integer, parameter :: n_elements_u = 4 
  integer, parameter :: n_elements_v = 4 
  real(spl_rk), dimension(n_elements_u+p_u) :: us 
  real(spl_rk), dimension(n_elements_v+p_v) :: vs 

  ! ... create a bilinear map
  P_11 = (/ 0.0_spl_rk, 0.0_spl_rk /)
  P_21 = (/ 1.0_spl_rk, 0.0_spl_rk /)
  P_12 = (/ 0.0_spl_rk, 1.0_spl_rk /)
  P_22 = (/ 1.0_spl_rk, 1.0_spl_rk /)

  call spl_mapping_bilinear( mapping, &
                           & P_11, P_12, P_21, P_22, &
                           & degrees=(/p_u,p_v/), &
                           & n_elements=(/n_elements_u,n_elements_v/))
  ! ...

  ! ...
  call mapping % get_greville(us, axis=1)
  call mapping % get_greville(vs, axis=2)

  print *, "Knots-u            : ", mapping % knots_u
  print *, "Greville abscissae : ", us
  print *, "Knots-v            : ", mapping % knots_v
  print *, "Greville abscissae : ", vs
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
