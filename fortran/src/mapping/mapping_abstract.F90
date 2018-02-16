!> @brief 
!> abstract module for mappings.
!> @details
!> A mapping is a 1D, 2D or 3D geometric transformation. 
!> Any mapping is described as B-Spline/NURBS surfaces

module spl_m_mapping_abstract
use spl_m_global
use spl_m_abstract, only: spl_t_abstract
implicit none

  ! .........................................................
  !> @brief 
  !> Abstract class for mappings.
  type, abstract, public, extends(spl_t_abstract) :: spl_t_mapping_abstract
     integer :: n_elements  = 0                  !< number of elements
     integer :: p_dim       = 0                  !< parametric dimension 
     integer :: d_dim       = 1                  !< physical dimension 
     integer :: n_points    = 0                  !< number of control points 
     logical :: rationalize = .FALSE.            !< use rationalize B-Splines
     
    !< pointer to another mapping for the composition 
     class(spl_t_mapping_abstract), pointer :: ptr_other     => null()
    
  contains
    procedure(spl_p_free_mapping_abstract),           deferred :: free 
    procedure(spl_p_print_info_mapping_abstract),     deferred :: print_info 
    procedure(spl_p_read_from_file_mapping_abstract), deferred :: read_from_file
    procedure(spl_p_export_mapping_abstract),         deferred :: export
  end type spl_t_mapping_abstract
  ! .........................................................

  ! ..................................................
  abstract interface
     subroutine spl_p_free_mapping_abstract(self)
       import spl_t_mapping_abstract

       class(spl_t_mapping_abstract), intent(inout)  :: self
     end subroutine spl_p_free_mapping_abstract
  end interface
  ! ..................................................
  
  ! ..................................................
  abstract interface
    subroutine spl_p_print_info_mapping_abstract(self)
      import spl_t_mapping_abstract

      class(spl_t_mapping_abstract), intent(in) :: self
    end subroutine spl_p_print_info_mapping_abstract
  end interface
  ! ..................................................

  ! ..................................................
  abstract interface
     subroutine spl_p_read_from_file_mapping_abstract(self, filename)
      use spl_m_global
      import spl_t_mapping_abstract
      
      class(spl_t_mapping_abstract), intent(inout) :: self
      character(len=*)                   , intent(in)    :: filename
     end subroutine spl_p_read_from_file_mapping_abstract
  end interface
  ! ..................................................

  ! ..................................................
  abstract interface
     subroutine spl_p_export_mapping_abstract(self, filename, i_format)
      use spl_m_global
      import spl_t_mapping_abstract
      
      class(spl_t_mapping_abstract), intent(inout) :: self
      character(len=*)                   , intent(in)    :: filename
      integer      , optional, intent(in)    :: i_format
     end subroutine spl_p_export_mapping_abstract
  end interface
  ! ..................................................

end module spl_m_mapping_abstract
