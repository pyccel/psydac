!> @brief 
!> mapping example in 2d 
!> @details
!> example of collela mapping 
!> usage:
!>   $> ./mapping_2d_ex15

! ............................................
program main
use spl_m_global
use spl_m_mapping_2d,      only: spl_t_mapping_2d
use spl_m_mapping_gallery, only: spl_mapping_collela
implicit none
  ! local
  type(spl_t_mapping_2d), target :: mapping
  integer, parameter :: p_u = 3 
  integer, parameter :: p_v = 3 
  integer, parameter :: n_elements_u = 8 
  integer, parameter :: n_elements_v = 8 
  real(spl_rk), parameter :: eps = 0.1_spl_rk
  real(spl_rk), parameter :: k1  = 1.0_spl_rk
  real(spl_rk), parameter :: k2  = 1.0_spl_rk

  ! ... creates a collela map
  call spl_mapping_collela( mapping, &
                          & eps=eps, k1=k1, k2=k2, &
                          & degrees=(/p_u,p_v/), &
                          & n_elements=(/n_elements_u,n_elements_v/))
  ! ...

  ! ... exports the mapping
  call mapping % export('mapping_2d_ex15_in.nml')
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
