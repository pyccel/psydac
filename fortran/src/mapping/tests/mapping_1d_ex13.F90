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
use spl_m_bsp,             only: spl_refinement_matrix_multi_stages 
implicit none
  ! local
  type(spl_t_mapping_1d), target :: mapping
  real(spl_rk), dimension(1) :: P_1
  real(spl_rk), dimension(1) :: P_2
  integer, parameter :: p_u = 3 
  integer, parameter :: n_f =  4 
  real(spl_rk), dimension(3) :: ts =(/ 0.13_spl_rk, 0.35_spl_rk, 0.6_spl_rk /) 
  real(spl_rk), dimension(:,:), allocatable :: mat
  integer :: m
  integer :: j

  ! ... create a linear map
  P_1 = (/ 0.0_spl_rk /)
  P_2 = (/ 1.0_spl_rk /)

  call spl_mapping_linear( mapping, &
                         & P_1, P_2, &
                         & degree=p_u, n_elements=n_f)
  ! ...

  ! ...
  m = size(ts, 1)
  allocate(mat(mapping % n_u + m, mapping % n_u))
  call spl_refinement_matrix_multi_stages( ts, &
                                         & mapping % n_u, &
                                         & mapping % p_u, &
                                         & mapping % knots_u, &
                                         & mat) 
  ! ...

  ! ...
  do j = 1, mapping % n_u + m 
    print *, mat(j, 1:mapping % n_u)
  end do
  ! ...

  ! ...
  call mapping % free() 
  ! ...
end program main
! ............................................
