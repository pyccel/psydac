!> @brief 
!> mapping example in 1d 
!> @details
!> creates an arc map and evaluates on given sites. 
!> usage:
!>   $> ./mapping_1d_ex8 

! ............................................
program main
use spl_m_global
use spl_m_mapping_1d,      only: spl_t_mapping_1d
use spl_m_mapping_gallery, only: spl_mapping_arc
implicit none
  ! local
  type(spl_t_mapping_1d), target :: mapping
  real(spl_rk), dimension(3) :: U 
  real(spl_rk), dimension(2,3) :: X
  real(spl_rk), dimension(2,3) :: dX
  real(spl_rk), dimension(2,3) :: ddX
  integer :: i
  real(spl_rk) :: r

  ! ... creates an arc map
  call spl_mapping_arc(mapping)
  ! ...

  ! ... prints info
  call mapping % print_info()
  ! ...

  ! ... mapping evaluation and its derivatives
  U = (/ 0.2_spl_rk, 0.4_spl_rk, 0.6_spl_rk /)

  call mapping % evaluate_deriv(U, X, dX, ddX)

  print *, "  u,   x,   y,   r"
  do i=1, 3
    r = sqrt(X(1,i)**2 + X(2,i)**2) 
    print *, U(i), X(1,i), X(2,i), r
  end do
  ! ...

  ! ... deallocates memory
  call mapping % free() 
  ! ...

end program main
! ............................................
