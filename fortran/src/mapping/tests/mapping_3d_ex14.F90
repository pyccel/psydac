!> @brief 
!> mapping example in 3d 
!> @details
!> creates a trilinear map then prints its info
!> usage:
!>   $> ./mapping_3d_ex1 

! ............................................
program main
use spl_m_global
use spl_m_mapping_3d,      only: spl_t_mapping_3d
use spl_m_mapping_gallery, only: spl_mapping_trilinear
implicit none
  ! local
  type(spl_t_mapping_3d), target :: mapping
  real(spl_rk), dimension(3) :: P_111
  real(spl_rk), dimension(3) :: P_121
  real(spl_rk), dimension(3) :: P_211
  real(spl_rk), dimension(3) :: P_221
  real(spl_rk), dimension(3) :: P_112
  real(spl_rk), dimension(3) :: P_122
  real(spl_rk), dimension(3) :: P_212
  real(spl_rk), dimension(3) :: P_222
  integer, parameter :: p_u = 3 
  integer, parameter :: p_v = 2 
  integer, parameter :: p_w = 2 
  integer, parameter :: n_elements_u = 4 
  integer, parameter :: n_elements_v = 4 
  integer, parameter :: n_elements_w = 4 
  real(spl_rk), dimension(n_elements_u+p_u) :: us 
  real(spl_rk), dimension(n_elements_v+p_v) :: vs 
  real(spl_rk), dimension(n_elements_w+p_w) :: ws 

  ! ... creates a trilinear map
  P_111 = (/ 0.0_spl_rk, 0.0_spl_rk, 0.0_spl_rk /)
  P_211 = (/ 1.0_spl_rk, 0.0_spl_rk, 0.0_spl_rk /)
  P_121 = (/ 0.0_spl_rk, 1.0_spl_rk, 0.0_spl_rk /)
  P_221 = (/ 1.0_spl_rk, 1.0_spl_rk, 0.0_spl_rk /)
  P_112 = (/ 0.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /)
  P_212 = (/ 1.0_spl_rk, 0.0_spl_rk, 1.0_spl_rk /)
  P_122 = (/ 0.0_spl_rk, 1.0_spl_rk, 1.0_spl_rk /)
  P_222 = (/ 1.0_spl_rk, 1.0_spl_rk, 1.0_spl_rk /)

  call spl_mapping_trilinear( mapping, &
                            & P_111, P_121, P_211, P_221, &
                            & P_112, P_122, P_212, P_222, &
                            & degrees=(/p_u,p_v,p_w/), &
                            & n_elements=(/n_elements_u,n_elements_v,n_elements_w/))
  ! ...

  ! ... 
  call mapping % get_greville(us, axis=1)
  call mapping % get_greville(vs, axis=2)
  call mapping % get_greville(ws, axis=3)

  print *, "Knots-u            : ", mapping % knots_u
  print *, "Greville abscissae : ", us
  print *, "Knots-v            : ", mapping % knots_v
  print *, "Greville abscissae : ", vs
  print *, "Knots-w            : ", mapping % knots_w
  print *, "Greville abscissae : ", ws
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
