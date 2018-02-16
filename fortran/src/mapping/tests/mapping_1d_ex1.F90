!> @brief 
!> mapping example in 1d 
!> @details
!> this example creates a linear mapping of degrees (1) and (2) elements
!> then prints the corresponding info 
!> usage:
!>   $> ./mapping_1d_ex1 

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

  ! ... create a linear map
  P_1 = (/ 10.0_spl_rk /)
  P_2 = (/ 20.0_spl_rk /)

  call spl_mapping_linear( mapping, &
                         & P_1, P_2, &
                         & degree=1, n_elements=2)
  ! ...

  ! ... prints info 
  call mapping % print_info()
  ! ...

  ! ... deallocate memory
  call mapping % free() 
  ! ...

end program main
! ............................................
