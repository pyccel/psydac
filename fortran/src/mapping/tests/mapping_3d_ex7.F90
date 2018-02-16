!> @brief 
!> mapping example in 3d 
!> @details
!> 3d mapping example for conversion to the pp form 
!> usage:
!>   $> ./mapping_3d_ex7 
   
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
  real(spl_rk), dimension(:,:,:,:,:,:,:), allocatable :: arr_pp
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

  ! ... Computes the pp-form of the mapping
  allocate(arr_pp(mapping % d_dim, &
       & mapping % p_u + 1       , &
       & mapping % p_v + 1       , &
       & mapping % p_w + 1       , &
       & mapping % n_elements_u  , &
       & mapping % n_elements_v  , &
       & mapping % n_elements_w))

  call mapping % to_pp(arr_pp)
  ! ...
  
  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
