!> @brief 
!> mapping example in 3d 
!> @details
!> 3d mapping example for the breaks method
!> usage:
!>   $> ./mapping_3d_ex5 
   
! ............................................
program main
use spl_m_global
use spl_m_mapping_3d,      only: spl_t_mapping_3d
use spl_m_mapping_gallery, only: spl_mapping_trilinear
implicit none
  ! local
  ! ... number of internal knots is = n - p - 1
  type(spl_t_mapping_3d), target :: mapping
  real(spl_rk), dimension(3) :: P_111
  real(spl_rk), dimension(3) :: P_121
  real(spl_rk), dimension(3) :: P_211
  real(spl_rk), dimension(3) :: P_221
  real(spl_rk), dimension(3) :: P_112
  real(spl_rk), dimension(3) :: P_122
  real(spl_rk), dimension(3) :: P_212
  real(spl_rk), dimension(3) :: P_222
  real(spl_rk), dimension(:), allocatable :: grid_u 
  real(spl_rk), dimension(:), allocatable :: grid_v 
  real(spl_rk), dimension(:), allocatable :: grid_w 
  integer, dimension(:), allocatable :: i_spans_u 
  integer, dimension(:), allocatable :: i_spans_v 
  integer, dimension(:), allocatable :: i_spans_w 
  integer :: n_elements_u
  integer :: n_elements_v
  integer :: n_elements_w
  integer :: n_elements

  ! ... creates a trilinear map
  P_111 = (/ 0.0_spl_rk, 0.0_spl_rk, 0.0_spl_rk /)
  P_211 = (/ 1.0_spl_rk, 0.0_spl_rk, 0.0_spl_rk /)
  P_121 = (/ 0.0_spl_rk, 1.0_spl_rk, 0.0_spl_rk /)
  P_221 = (/ 1.0_spl_rk, 1.0_spl_rk, 0.0_spl_rk /)
  P_112 = (/ 0.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /)
  P_212 = (/ 1.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /)
  P_122 = (/ 0.0_spl_rk, 1.0_spl_rk, 1.0_spl_rk /)
  P_222 = (/ 1.0_spl_rk, 1.0_spl_rk, 1.0_spl_rk /)

  call spl_mapping_trilinear(mapping, &
    & P_111, P_121, P_211, P_221, &
    & P_112, P_122, P_212, P_222, &
    & degrees=(/2,2,2/), n_elements=(/2,2,2/))
  ! ...

  ! ... prints info
  call mapping % print_info()
  ! ...

  ! ... computes mapping breaks
  allocate(grid_u(mapping % n_u + mapping % p_u + 1))
  allocate(grid_v(mapping % n_v + mapping % p_v + 1))
  allocate(grid_w(mapping % n_w + mapping % p_w + 1))
  allocate(i_spans_u(mapping % n_u + mapping % p_u + 1))
  allocate(i_spans_v(mapping % n_v + mapping % p_v + 1))
  allocate(i_spans_w(mapping % n_w + mapping % p_w + 1))

  call mapping % breaks(n_elements_u, grid_u, i_spans_u, axis=1)
  call mapping % breaks(n_elements_v, grid_v, i_spans_v, axis=2)
  call mapping % breaks(n_elements_w, grid_w, i_spans_w, axis=3)

  print *, "n_elements_u : ", n_elements_u
  print *, "n_elements_v : ", n_elements_v
  print *, "n_elements_w : ", n_elements_w
  print *, "grid_u       : ", grid_u
  print *, "grid_v       : ", grid_v
  print *, "grid_w       : ", grid_w
  print *, "i_spans_u    : ", i_spans_u
  print *, "i_spans_v    : ", i_spans_v
  print *, "i_spans_w    : ", i_spans_w
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
