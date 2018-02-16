!> @brief 
!> mapping example in 1d 
!> @details
!> 1d mapping example for conversion to the pp form 
!> usage:
!>   $> ./mapping_1d_ex7 

! ............................................
program main
use spl_m_global
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_gallery, only: spl_mapping_linear 
implicit none
  type(spl_t_mapping_1d), target :: mapping
  real(spl_rk), dimension(1) :: P_1
  real(spl_rk), dimension(1) :: P_2
  real(spl_rk), dimension(:,:,:), allocatable :: arr_pp
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

  ! ... Computes the pp-form of the mapping
  n_elements = mapping % n_elements_u
  allocate(arr_pp(mapping % d_dim, mapping % p_u + 1, n_elements))

  call mapping % to_pp(arr_pp)

  print *, "====== arr_pp ======"
  do i_element=1, n_elements
    print *, ">> element ", i_element
    print *, arr_pp(:,:,i_element)
  end do
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
