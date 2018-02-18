!> @brief 
!> SPL global constants.
!> @details
!> Global constants for SPL 
module spl_m_global
implicit none

    integer, parameter    :: spl_rk=kind(1.d0)                 !< Real precision    
    integer, parameter    :: spl_int_default=-10000000         !< default integer value
    real(kind=spl_rk)   :: spl_pi=3.1415926535897931            !< Pi definition

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


end module spl_m_global
