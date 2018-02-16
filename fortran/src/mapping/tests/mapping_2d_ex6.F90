!> @brief 
!> mapping example in 2d 
!> @details
!> 2d mapping example for conversion to uniform B-Spline 
!> usage:
!>   $> ./mapping_2d_ex6 

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
  real(spl_rk), dimension(:,:,:,:), allocatable :: arr_us
  real(spl_rk), dimension(:), allocatable :: grid_u 
  real(spl_rk), dimension(:), allocatable :: grid_v 
  integer :: n_elements_u 
  integer :: n_elements_v
  integer :: n_elements
  integer :: i_element_u 
  integer :: i_element_v
  integer :: i_element

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

  ! ... computes mapping breaks
  allocate(grid_u(mapping % n_u + mapping % p_u + 1))
  allocate(grid_v(mapping % n_v + mapping % p_v + 1))
  call mapping % breaks(n_elements_u, grid_u, axis=1)
  call mapping % breaks(n_elements_v, grid_v, axis=2)
  ! ...

  ! ... computes the coefficients of the associated uniform B-Splines (unclamped)
  n_elements = n_elements_u * n_elements_v
  allocate(arr_us(mapping % d_dim, mapping % p_u + 1, mapping % p_u + 1, n_elements))

  call mapping % to_us(arr_us)

  print *, "====== arr_us ======"
  i_element = 0
  do i_element_u=1, mapping % n_elements_u
    do i_element_v=1, mapping % n_elements_v
      i_element = i_element + 1
      print *, ">> element ", i_element
      print *, arr_us(:,:,:,i_element)
    end do
  end do
  ! ...

  ! ... deallocate memory
  call mapping % free() 
  ! ...

end program main
! ............................................
