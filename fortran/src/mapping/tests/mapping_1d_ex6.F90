!> @brief 
!> mapping example in 1d 
!> @details
!> 1d mapping example for conversion to uniform B-Spline 
!> usage:
!>   $> ./mapping_1d_ex6 
   
! ............................................
program main
use spl_m_global
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_gallery, only: spl_mapping_linear 
implicit none
  type(spl_t_mapping_1d), target :: mapping
  real(spl_rk), dimension(1) :: P_1
  real(spl_rk), dimension(1) :: P_2
  real(spl_rk), dimension(:,:,:), allocatable :: arr_us
  real(spl_rk), dimension(:), allocatable :: grid 
  integer, dimension(:), allocatable :: i_spans
  integer :: n_elements
  integer :: i_element

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

  ! ... computes mapping breaks
  allocate(grid(mapping % n_u + mapping % p_u + 1))
  allocate(i_spans(mapping % n_u + mapping % p_u + 1))

  call mapping % breaks(n_elements, grid, i_spans=i_spans)
  ! ...

  ! ... computes the coefficients of the associated uniform B-Splines (unclamped)
  allocate(arr_us(mapping % d_dim, mapping % p_u + 1, n_elements))

  call mapping % to_us(arr_us)

  print *, "====== arr_us ======"
  do i_element=1, n_elements
    print *, ">> element ", i_element
    print *, arr_us(:,:,i_element)
  end do
  ! ...

  ! ... memory deallocation
  call mapping % free() 
  ! ...

end program main
! ............................................
