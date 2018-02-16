!# -*- coding: utf8 -*-
!> @brief 
!> module for spl abstract class
!> @details

module spl_m_abstract
use spl_m_global
implicit none

  ! ..................................................
  !> @brief 
  !> spl abstract class
  type, public, abstract :: spl_t_abstract
    logical  :: is_allocated  = .false.  !< true if memory is allocated 
  contains
    procedure(spl_p_free_abstract), deferred :: free 
  end type spl_t_abstract
  ! ..................................................

  ! ..................................................
  abstract interface
     subroutine spl_p_free_abstract(self)
       import spl_t_abstract

       class(spl_t_abstract), intent(inout)  :: self
     end subroutine spl_p_free_abstract
  end interface
  ! ..................................................

end module spl_m_abstract
