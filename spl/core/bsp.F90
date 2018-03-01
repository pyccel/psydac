! -*- coding: UTF-8 -*-
!> @brief 
!> Module for 
!> @details
!> interfaced module with python


!... TODO: add bsp_mapping module
!... TODO: add more utils functions


module bsp_utils

  implicit none
  
  public :: collocation_matrix,         &
          & eval_on_grid_splines_ders,  &
          & compute_spans,              &
          & construct_grid_from_knots,  &
          & construct_quadrature_grid,  &
          & make_open_knots,            &
          & compute_origins_element

contains

  ! .......................................................
  subroutine collocation_matrix(p, n, m, knots, u, mat)
    use spl_m_bsp, only: spl_collocation_matrix
    implicit none
    integer,                    intent(in)  :: p
    integer,                    intent(in)  :: n
    integer,                    intent(in)  :: m
    real(8), dimension(n+p+1),  intent(in)  :: knots
    real(8), dimension(m),      intent(in)  :: u
    real(8), dimension(m, n),   intent(out) :: mat

    ! ...
    call spl_collocation_matrix(n, p, knots, u, mat)
    ! ...

  end subroutine collocation_matrix 
  ! .......................................................

  ! ................................................
  subroutine eval_on_grid_splines_ders(p, n, k, d, knots, points, basis)
  use spl_m_bsp, only: spl_eval_on_grid_splines_ders
  implicit none
    integer,                    intent(in)  :: p
    integer,                    intent(in)  :: n
    integer,                    intent(in)  :: k
    integer,                    intent(in)  :: d
    real(8), dimension(n+p+1),  intent(in)  :: knots
    real(kind=8), dimension(k,n-p), intent(in) :: points
    real(kind=8), dimension(p+1,d+1,k,n-p), intent(out) :: basis

    ! ...
    call spl_eval_on_grid_splines_ders(n, p, d, knots, points, basis)
    ! ...
    
  end subroutine eval_on_grid_splines_ders 
  ! ................................................

  ! ................................................
  subroutine compute_spans(p, n, knots, elements_spans)
  use spl_m_bsp, only: spl_compute_spans
  implicit none
    integer,                        intent(in)  :: p
    integer,                        intent(in)  :: n
    real(8), dimension(n+p+1),      intent(in)  :: knots
    integer, dimension(n-p), intent(out) :: elements_spans

    ! ...
    call spl_compute_spans(p, n, knots, elements_spans)
    ! ...
    
  end subroutine compute_spans 
  ! ................................................

  ! ................................................
  subroutine construct_grid_from_knots(p, n, knots, grid)
  use spl_m_bsp, only: spl_construct_grid_from_knots
  implicit none
    integer,                               intent(in)  :: p
    integer,                               intent(in)  :: n
    real(8), dimension(n+p+1),             intent(in)  :: knots
    real(kind=8), dimension(n-p+1), intent(out) :: grid
    
    integer :: n_elements

    ! ...
    n_elements = n-p
    call spl_construct_grid_from_knots(p, n, n_elements, knots, grid)
    ! ...
    
  end subroutine construct_grid_from_knots 
  ! ................................................
  
  ! ................................................
  subroutine construct_quadrature_grid(n_elements, k, u, w, grid, points, weights)
  use spl_m_bsp, only: spl_construct_quadrature_grid
  implicit none
    integer,                               intent(in)  :: n_elements
    integer,                               intent(in)  :: k
    real(kind=8), dimension(k),            intent(in)  :: u
    real(kind=8), dimension(k),            intent(in)  :: w
    real(kind=8), dimension(n_elements+1), intent(in)  :: grid
    real(kind=8), dimension(k,n_elements), intent(out) :: points
    real(kind=8), dimension(k,n_elements), intent(out) :: weights

    ! ...
    call spl_construct_quadrature_grid(u, w, grid, points, weights)
    ! ...
    
  end subroutine construct_quadrature_grid 
  ! ................................................

  ! ................................................
  subroutine make_open_knots(p, n, knots)
  use spl_m_bsp, only: spl_make_open_knots
  implicit none
    integer,                    intent(in)  :: p
    integer,                    intent(in)  :: n
    real(8), dimension(n+p+1),  intent(out)  :: knots

    ! ...
    call spl_make_open_knots(n, p, knots)
    ! ...
    
  end subroutine make_open_knots 
  ! ................................................  
  
  ! ................................................
  subroutine compute_origins_element(p, n, knots, origins_element)
  use spl_m_bsp, only: spl_compute_origins_element
  implicit none
    integer,                   intent(in)  :: p
    integer,                   intent(in)  :: n
    real(8), dimension(n+p+1), intent(in)  :: knots
    integer, dimension(n),     intent(out) :: origins_element

    ! ...
    call spl_compute_origins_element(p, n, knots, origins_element)
    ! ...
    
  end subroutine compute_origins_element 
  ! ................................................


end module bsp_utils
