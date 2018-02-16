!> @brief 
!> mapping example in 1d 
!> @details
!> 1d mapping example for the breaks method
!> usage:
!>   $> ./mapping_1d_ex5 
   
! ............................................
program main
use spl_m_global
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_gallery, only: spl_mapping_linear
implicit none
  type(spl_t_mapping_1d), target :: mapping
  real(spl_rk), dimension(1) :: P_1
  real(spl_rk), dimension(1) :: P_2
  real(spl_rk), dimension(:), allocatable :: grid 
  integer, dimension(:), allocatable :: i_spans
  integer :: n_elements

  ! ... create a linear map
  P_1 = (/ 0.0_spl_rk /)
  P_2 = (/ 1.0_spl_rk /)

  call spl_mapping_linear( mapping, &
                         & P_1, P_2, &
                         & degree=1, n_elements=4)
  ! ...

  ! ... prints info
  call mapping % print_info()
  ! ...

  ! ... example of computing a mapping breaks
  allocate(grid(mapping % n_u + mapping % p_u + 1))
  allocate(i_spans(mapping % n_u + mapping % p_u + 1))

  call mapping % breaks(n_elements, grid, i_spans=i_spans)
  print *, "<> n_elements :", n_elements
  print *, "<> grid       :", grid
  print *, "<> i_spans    :", i_spans
  ! ...

  ! ... deallocate memory
  call mapping % free() 
  ! ...

end program main
! ............................................
