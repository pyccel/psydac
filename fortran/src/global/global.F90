!> @brief 
!> SPL global constants.
!> @details
!> Global constants for SPL 
module spl_m_global
use plf_m_global
implicit none
    
    ! ... mapping boundary ids 
    integer, parameter  :: spl_mapping_boundary_min      = -1   !< id for min boundary 
    integer, parameter  :: spl_mapping_boundary_max      = 1    !< id for max boundary 
    ! ...

    ! ... boundary conditions
    integer, parameter  :: spl_bc_dirichlet_homogen      = 0    !< Homogeneous Dirichlet Boundary Condition 
    integer, parameter  :: spl_bc_periodic               = 1    !< Periodic Boundary condition	
    integer, parameter  :: spl_bc_none                   = 2    !< No Boundary Condition
    integer, parameter  :: spl_bc_unclamped              = 3    !< Unclamped B-SPlines with no Boundary Condition
    integer, parameter  :: spl_bc_periodic_c0            = 4    !< C0 Periodic Boundary condition	
    ! ...
    ! ...

    ! ... mapping format output
    integer, parameter  :: spl_mapping_format_nml        = 0    !< Mapping in nml format 
    ! ...

contains

  ! .............................................
  !> @brief     initialization of SPL
  !>
  !> @param[inout] i_err    error code id, given by PLAF 
  subroutine spl_initialize(i_err)
  implicit none
  integer, intent(inout) :: i_err
  ! Local

  call plf_initialize(i_err)

  end subroutine spl_initialize
  ! .............................................

  ! .............................................
  !> @brief     finalization of SPL
  !>
  !> @param[inout] i_err    error code id, given by PLAF 
  subroutine spl_finalize(i_err)
  implicit none
  integer, intent(inout) :: i_err
  ! Local

  call plf_finalize(i_err)

  end subroutine spl_finalize
  ! .............................................

  ! .............................................
  !> @brief     compute CPU time 
  !>
  !> @param[inout] time    a real value  
  subroutine spl_time(time)
  implicit none
  real(kind=plf_rk), intent(inout) :: time
  ! Local

  call plf_time(time) 

  end subroutine spl_time
  ! .............................................

end module spl_m_global
