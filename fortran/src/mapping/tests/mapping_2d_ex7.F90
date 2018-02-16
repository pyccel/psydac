!> @brief 
!> mapping example in 2d 
!> @details
!> 2d mapping example for conversion to the pp form 
!> usage:
!>   $> ./mapping_2d_ex7 

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
  real(spl_rk), dimension(:,:,:,:,:), allocatable :: arr_pp
  integer :: i_element_u 
  integer :: i_element_v
  integer :: i_u

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

  ! ... Computes the pp-form of the mapping
  allocate(arr_pp(mapping % d_dim, &
       & mapping % p_u + 1       , &
       & mapping % p_v + 1       , &
       & mapping % n_elements_u  , &
       & mapping % n_elements_v))

  call mapping % to_pp(arr_pp)

  print *, "====== arr_pp ======"
 
  do i_element_u=1, mapping % n_elements_u
    do i_element_v=1, mapping % n_elements_v
     
      print *, ">> element ", i_element_u, i_element_v
      print *, arr_pp(:,:,:,i_element_u,i_element_v)
    end do
  end do
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
