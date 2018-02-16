!> @brief 
!> module for different sort.
module spl_m_sort

use spl_m_global

implicit none

  private
  public  :: spl_sort_qsortc

contains

  ! ..........................................................        
  !> @brief       sort the array a using the quick sort algorithm 
  !>
  !> @param[inout] a the array to sort 
  recursive subroutine spl_sort_qsortc(a)
  implicit none
    real(spl_rk), dimension(:), intent(inout) :: a
    integer :: iq
  
    if(size(a) > 1) then
       call spl_sort_partition(a, iq)
       call spl_sort_qsortc(a(:iq-1))
       call spl_sort_qsortc(a(iq:))
    endif
  end subroutine spl_sort_qsortc
  ! ..........................................................        
  
  ! ..........................................................        
  !> @brief       computes a partition for the array a 
  !>
  !> @param[inout] a the array to sort 
  !> @param[out] marker the span index 
  subroutine spl_sort_partition(a, marker)
  implicit none
    real(spl_rk), dimension(:), intent(inout) :: a
    integer                   , intent(out)    :: marker
    ! local
    integer :: i, j
    real(spl_rk) :: temp
    real(spl_rk) :: x      ! pivot point
  
    x = a(1)
    i= 0
    j= size(a) + 1
  
    do
       j = j-1
       do
          if (a(j) <= x) exit
          j = j-1
       end do
       i = i+1
       do
        if (a(i) >= x) exit
        i = i+1
       end do
       if (i < j) then
          ! exchange a(i) and a(j)
          temp = a(i)
          a(i) = a(j)
          a(j) = temp
       elseif (i == j) then
          marker = i+1
          return
       else
          marker = i
          return
       endif
    end do
  
  end subroutine spl_sort_partition
  ! ..........................................................        
  
end module spl_m_sort
