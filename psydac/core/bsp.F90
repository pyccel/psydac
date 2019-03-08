! -*- coding: UTF-8 -*-

!> @brief 
!> Module for basic BSplines functions
!> @details
!> specific implementations for f2py interface


module bsp_utils

  implicit none
  
  public :: compute_greville,           &
          & eval_on_grid_splines_ders,  &
          & compute_spans,              &
          & construct_grid_from_knots,  &
          & construct_quadrature_grid,  &
          & make_open_knots,            &
          & make_periodic_knots,        &
          & compute_origins_element,    &
          & collocation_matrix,         &
          & collocation_cardinal_splines, &
          & matrix_multi_stages

contains

  ! .......................................................
  subroutine compute_greville(p, n, knots, arr_x)
    use bsp_ext, only: spl_compute_greville
    implicit none
    integer(kind=4),            intent(in)  :: p
    integer(kind=4),            intent(in)  :: n 
    real(8), dimension(n+p+1),  intent(in)  :: knots
    real(8), dimension(n),      intent(out) :: arr_x 

    ! ...
    call spl_compute_greville(p, n, knots, arr_x)
    ! ...

  end subroutine compute_greville
  ! .......................................................

  ! ................................................
  subroutine eval_on_grid_splines_ders(p, n, k, d, knots, points, basis)
  use bsp_ext, only: spl_eval_on_grid_splines_ders
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
  use bsp_ext, only: spl_compute_spans
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
  use bsp_ext, only: spl_construct_grid_from_knots
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
  use bsp_ext, only: spl_construct_quadrature_grid
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
  use bsp_ext, only: spl_make_open_knots
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
  subroutine make_periodic_knots(p, n, knots)
  use bsp_ext, only: spl_make_open_knots, &
                     & spl_symetrize_knots
  implicit none
    integer,                    intent(in)  :: p
    integer,                    intent(in)  :: n
  real(8), dimension(n+p+1),   intent(out)  :: knots

    ! ...
    call spl_make_open_knots(n, p, knots)
    call spl_symetrize_knots(p-1, n, p, knots) 
    ! ...
    
  end subroutine make_periodic_knots 
  ! ................................................ 

  ! ................................................
  subroutine compute_origins_element(p, n, knots, origins_element)
  use bsp_ext, only: spl_compute_origins_element
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
  
  ! .......................................................
  subroutine collocation_matrix(p, n, m, knots, u, mat)
    use bsp_ext, only: spl_collocation_matrix
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
  subroutine collocation_cardinal_splines(p, n, mat)
  use bsp_ext, only: spl_compute_symbol_stiffness
  implicit none
    integer,                               intent(in)  :: p
    integer,                               intent(in)  :: n
    real(8), dimension(n, n), intent(out) :: mat 

    ! ...
    call spl_compute_symbol_stiffness(p, n, mat)
    ! ...
    
  end subroutine collocation_cardinal_splines 
  ! ................................................

  ! ................................................
  subroutine matrix_multi_stages(m, ts, n, p, knots, mat)
  use bsp_ext, only: spl_refinement_matrix_multi_stages
  implicit none
    integer,                    intent(in)  :: m
    real(8), dimension(m),      intent(in)  :: ts
    integer,                    intent(in)  :: n
    integer,                    intent(in)  :: p
    real(8), dimension(n+p+1),  intent(in)  :: knots
    real(8), dimension(n+m, n), intent(out) :: mat 

    ! ...
    call spl_refinement_matrix_multi_stages(ts, n, p, knots, mat)
    ! ...
    
  end subroutine matrix_multi_stages 
  ! ................................................

end module bsp_utils
