!> @brief 
!> mapping example in 2d 
!> @details
!> 2d mapping example for the breaks method
!> usage:
!>   $> ./mapping_2d_ex5 

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_bilinear
implicit none
  type(spl_t_mapping_2d), target :: mapping
  real(spl_rk), dimension(2) :: P_11
  real(spl_rk), dimension(2) :: P_12
  real(spl_rk), dimension(2) :: P_21
  real(spl_rk), dimension(2) :: P_22
  real(spl_rk), dimension(:), allocatable :: grid_u 
  real(spl_rk), dimension(:), allocatable :: grid_v 
  integer, dimension(:), allocatable :: i_spans_u 
  integer, dimension(:), allocatable :: i_spans_v 
  integer :: n_elements_u
  integer :: n_elements_v
  integer :: n_elements

  ! ... creates a bilinear map
  P_11 = (/ 0.0_spl_rk, 0.0_spl_rk /)
  P_21 = (/ 1.0_spl_rk, 0.0_spl_rk /)
  P_12 = (/ 0.0_spl_rk, 1.0_spl_rk /)
  P_22 = (/ 1.0_spl_rk, 1.0_spl_rk /)

  call spl_mapping_bilinear( mapping, &
                           & P_11, P_12, P_21, P_22, &
                           & degrees=(/1,1/), n_elements=(/2,2/))
  ! ...

  ! ... prints info
  call mapping % print_info()
  ! ...

  ! ... computes breaks
  allocate(grid_u(mapping % n_u + mapping % p_u + 1))
  allocate(grid_v(mapping % n_v + mapping % p_v + 1))
  allocate(i_spans_u(mapping % n_u + mapping % p_u + 1))
  allocate(i_spans_v(mapping % n_v + mapping % p_v + 1))

  call mapping % breaks(n_elements_u, grid_u, i_spans=i_spans_u, axis=1)
  call mapping % breaks(n_elements_v, grid_v, i_spans=i_spans_v, axis=2)

  print *, "n_elements_u : ", n_elements_u
  print *, "n_elements_v : ", n_elements_v
  print *, "grid_u       : ", grid_u
  print *, "grid_v       : ", grid_v
  print *, "i_spans_u    : ", i_spans_u
  print *, "i_spans_v    : ", i_spans_v
  ! ...

  ! ... deallocate memory
  call mapping % free() 
  ! ...

end program main
! ............................................
