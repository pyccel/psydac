!> @brief 
!> mapping example in 1d 
!> @details
!> creates the matrix refinement for a linear map
!> usage:
!>   $> ./mapping_1d_ex13 

! ............................................
program main
use spl_m_global
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_gallery, only: spl_mapping_linear
implicit none
  ! local
  type(spl_t_mapping_1d), target :: mapping
  real(spl_rk), dimension(1) :: P_1
  real(spl_rk), dimension(1) :: P_2
  integer, parameter :: p_u = 3 
  integer, parameter :: n_elements_u = 4 
  real(spl_rk), dimension(n_elements_u+p_u) :: us 

  ! ... create a linear map
  P_1 = (/ 0.0_spl_rk /)
  P_2 = (/ 1.0_spl_rk /)

  call spl_mapping_linear( mapping, &
                         & P_1, P_2, &
                         & degree=p_u, n_elements=n_elements_u)
  ! ...

  ! ...
  call mapping % get_greville(us)
  print *, "Knots              : ", mapping % knots_u
  print *, "Greville abscissae : ", us
  ! ...

  ! ...
  call mapping % free() 
  ! ...
end program main
! ............................................
