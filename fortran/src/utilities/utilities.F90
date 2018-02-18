!> @brief 
!> module for different utilities.
!>
!>
!> Maintainer   ARA	
!> Stability	stable

module spl_m_utilities

use spl_m_global

implicit none

  private
  public  :: spl_print_array, & 
           & spl_print_matrix
contains

  ! ..................................................
  !> @brief  a nice printing of an array of shape 1 
  !>
  !> @param[in]  desc    string containing a description of the matrix  
  !> @param[in]  m       number of rows to print 
  !> @param[in]  a       array to print 
  subroutine spl_print_array( desc, m, a )
  implicit none
  character(len = *),         intent(in) :: desc 
  integer,                    intent(in) :: m
  real(spl_rk), dimension(:), intent(in) :: a
  ! local
  integer          j

  ! ...
  write(*,*)
  write(*,*) desc
  write(*,9997) ( a( j ), j = 1, m )
  ! ...

9997 format( 11(:,1x,f6.2) )

  end subroutine spl_print_array 
  ! ..................................................

  ! ..................................................
  !> @brief  a nice printing of an array of shape 2 
  !>
  !> @param[in]  desc    string containing a description of the matrix  
  !> @param[in]  m       number of rows to print 
  !> @param[in]  n       number of columns to print 
  !> @param[in]  a       matrix to print 
  !> @param[in]  lda     number of rows of the matrix
  subroutine spl_print_matrix( desc, m, n, a, lda )
  implicit none
  character(len = *),           intent(in) :: desc 
  integer,                      intent(in) :: m
  integer,                      intent(in) :: n
  real(spl_rk), dimension(:,:), intent(in) :: a
  integer,                      intent(in) :: lda
  ! local
  integer          i
  integer          j

  ! ...
  write(*,*)
  write(*,*) desc
  do i = 1, m
     write(*,9998) ( a( i, j ), j = 1, n )
  end do
  ! ...

9998 format( 11(:,1x,f6.2) )

  end subroutine spl_print_matrix 
  ! ..................................................

end module spl_m_utilities
