!> @brief 
!> mapping example in 1d 
!> @details
!> creates the matrix refinement for a linear map
!> usage:
!>   $> ./mapping_1d_ex12 

! ............................................
program main
use spl_m_global
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_gallery, only: spl_mapping_linear
use spl_m_bsp,             only: spl_refinement_matrix_one_stage 
implicit none
  ! local
  type(spl_t_mapping_1d), target :: mapping
  real(spl_rk), dimension(1) :: P_1
  real(spl_rk), dimension(1) :: P_2
  integer, parameter :: p_u = 3 
  integer, parameter :: n_f =  4 
  real(spl_rk), dimension(:,:), allocatable :: mat
  integer :: i

  ! ... create a linear map
  P_1 = (/ 0.0_spl_rk /)
  P_2 = (/ 1.0_spl_rk /)

  call spl_mapping_linear( mapping, &
                         & P_1, P_2, &
                         & degree=p_u, n_elements=n_f)
  ! ...

  ! ... prints info
  call mapping % print_info()
  print *, "=============================="
  ! ...

  ! ...
  allocate(mat(mapping % n_u + 1, mapping % n_u))
  call spl_refinement_matrix_one_stage( 0.13_spl_rk, &
                                      & mapping % n_u, &
                                      & mapping % p_u, &
                                      & mapping % knots_u, &
                                      & mat) 
  ! ...
  
  ! ...
  do i = 1, mapping % n_u +1
    print *, mat(i, :)
  end do
  ! ...

  ! ...
  call mapping % free() 
  ! ...

end program main
! ............................................
