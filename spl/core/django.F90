module m_django_core

use spl_m_mapping_abstract,           only: spl_t_mapping_abstract
use spl_m_mapping_1d,                 only: spl_t_mapping_1d
use spl_m_mapping_2d,                 only: spl_t_mapping_2d
use spl_m_mapping_3d,                 only: spl_t_mapping_3d
                                     
implicit none

  private
  public ::                                              &
          & get_n_max_core_array,                        &
          ! ... SPL/mapping                             
          & mapping_read_from_file,                      &
          & mapping_export,                              &
          & mapping_free,                                &
          & mapping_print_info,                          &
          & mapping_allocate,                            &
          & mapping_create_1d,                           &
          & mapping_create_2d_0,                         &
          & mapping_create_2d_1,                         &
          & mapping_create_3d,                           &
          & mapping_evaluate_1d,                         &
          & mapping_evaluate_2d,                         &
          & mapping_evaluate_3d,                         &
          & mapping_evaluate_deriv_1_1d,                 & ! TODO add evaluate_deriv with 2nd derivatives too
          & mapping_evaluate_deriv_1_2d,                 &
          & mapping_evaluate_deriv_1_3d,                 &
          & mapping_get_d_dim,                           &
          ! ...                                         
          ! ... SPL/utilities                                        
          & utilities_collocation_cardinal_splines,      &
          & utilities_collocation_matrix,                &
          & utilities_make_open_knots,                   &
          & utilities_compute_spans,                     &
          & utilities_compute_origins_element,           &
          & utilities_construct_grid_from_knots,         &
          & utilities_construct_quadrature_grid,         &
          & utilities_eval_on_grid_splines_ders
          ! ...                                         

  ! ... maximum size of arrays for local data to the module
  integer, parameter, private :: n_max_core_array =   50 
  ! ...

  ! ... data structures for mappings
  type, private :: t_mapping
    class(spl_t_mapping_abstract), allocatable :: mapping
  end type t_mapping

  integer, parameter, private :: n_mappings = n_max_core_array
  type(t_mapping), dimension(n_mappings), private, target :: p_mappings
  ! ...

contains

  ! ................................................
  subroutine get_n_max_core_array(n)
  implicit none
    integer, intent(out) :: n
    ! local

    ! ... 
    n = n_max_core_array 
    ! ...
    
  end subroutine get_n_max_core_array 
  ! ................................................

  ! ................................................
  subroutine utilities_collocation_cardinal_splines(p, n_points, mat)
  use spl_m_bsp, only: spl_compute_symbol_stiffness
  implicit none
    integer,                               intent(in)  :: p
    integer,                               intent(in)  :: n_points
    real(8), dimension(n_points,n_points), intent(out) :: mat 
    ! local

    ! ...
    call spl_compute_symbol_stiffness(p, n_points, mat)
    ! ...
    
  end subroutine utilities_collocation_cardinal_splines 
  ! ................................................
  
  ! ................................................
  subroutine utilities_collocation_matrix(p, n, m, knots, u, mat)
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
    
  end subroutine utilities_collocation_matrix 
  ! ................................................
  
  ! ................................................
  subroutine utilities_make_open_knots(p, n, knots)
  use spl_m_bsp, only: spl_make_open_knots
  implicit none
    integer,                    intent(in)  :: p
    integer,                    intent(in)  :: n
    real(8), dimension(n+p+1),  intent(out)  :: knots

    ! ...
    call spl_make_open_knots(n, p, knots)
    ! ...
    
  end subroutine utilities_make_open_knots 
  ! ................................................
  
  ! ................................................
  subroutine utilities_compute_spans(p, n, knots, n_elements, elements_spans)
  use spl_m_bsp, only: spl_compute_spans
  implicit none
    integer,                        intent(in)  :: p
    integer,                        intent(in)  :: n
    real(8), dimension(n+p+1),      intent(in)  :: knots
    integer,                        intent(in)  :: n_elements
    integer, dimension(n_elements), intent(out) :: elements_spans

    ! ...
    call spl_compute_spans(p, n, knots, elements_spans)
    ! ...
    
  end subroutine utilities_compute_spans 
  ! ................................................
  
  ! ................................................
  subroutine utilities_compute_origins_element(p, n, knots, origins_element)
  use spl_m_bsp, only: spl_compute_origins_element
  implicit none
    integer,                   intent(in)  :: p
    integer,                   intent(in)  :: n
    real(8), dimension(n+p+1), intent(in)  :: knots
    integer, dimension(n),     intent(out) :: origins_element

    ! ...
    call spl_compute_origins_element(p, n, knots, origins_element)
    ! ...
    
  end subroutine utilities_compute_origins_element 
  ! ................................................
  
  ! ................................................
  subroutine utilities_construct_grid_from_knots(p, n, n_elements, knots, grid)
  use spl_m_bsp, only: spl_construct_grid_from_knots
  implicit none
    integer,                               intent(in)  :: p
    integer,                               intent(in)  :: n
    integer,                               intent(in)  :: n_elements
    real(8), dimension(n+p+1),             intent(in)  :: knots
    real(kind=8), dimension(n_elements+1), intent(out) :: grid

    ! ...
    call spl_construct_grid_from_knots(p, n, n_elements, knots, grid)
    ! ...
    
  end subroutine utilities_construct_grid_from_knots 
  ! ................................................
  
  ! ................................................
  subroutine utilities_construct_quadrature_grid(n_elements, k, u, w, grid, points, weights)
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
    
  end subroutine utilities_construct_quadrature_grid 
  ! ................................................
  
  ! ................................................
  subroutine utilities_eval_on_grid_splines_ders(p, n, n_elements, k, d, knots, points, basis)
  use spl_m_bsp, only: spl_eval_on_grid_splines_ders
  implicit none
    integer,                    intent(in)  :: p
    integer,                    intent(in)  :: n
    integer,                    intent(in)  :: n_elements
    integer,                    intent(in)  :: k
    integer,                    intent(in)  :: d
    real(8), dimension(n+p+1),  intent(in)  :: knots
    real(kind=8), dimension(k,n_elements), intent(in) :: points
    real(kind=8), dimension(p+1,d+1,k,n_elements), intent(out) :: basis

    ! ...
    call spl_eval_on_grid_splines_ders(n, p, d, knots, points, basis)
    ! ...
    
  end subroutine utilities_eval_on_grid_splines_ders 
  ! ................................................


  ! ................................................
  subroutine mapping_allocate(i_mapping, i_dim)
  implicit none
    integer, intent(in) :: i_mapping
    integer, intent(in) :: i_dim
    ! local

    ! ...
    if (i_dim == 1) then
      allocate(spl_t_mapping_1d::p_mappings(i_mapping) % mapping)
    elseif (i_dim == 2) then
      allocate(spl_t_mapping_2d::p_mappings(i_mapping) % mapping)
    elseif (i_dim == 3) then
      allocate(spl_t_mapping_3d::p_mappings(i_mapping) % mapping)
    else
      stop "mapping_allocate: wrong i_dim"
    end if
    ! ...
    
  end subroutine mapping_allocate
  ! ................................................

  ! ................................................
  subroutine mapping_create_1d( i_mapping, &
                              & d_dim, &
                              & p_u, &
                              & n_u, &
                              & np1_u, &
                              & knots_u, &
                              & control_points, weights)
  implicit none
    integer,                           intent(in) :: i_mapping
    integer,                           intent(in) :: d_dim 
    integer,                           intent(in) :: p_u 
    integer,                           intent(in) :: n_u 
    integer,                           intent(in) :: np1_u 
    real(8), dimension(np1_u),         intent(in) :: knots_u 
    real(8), dimension(d_dim,n_u), intent(in) :: control_points
    real(8), dimension(n_u),       intent(in) :: weights 
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_1d)
      call mapping % create(p_u, knots_u, control_points, weights=weights)
    end select
    ! ...
    
  end subroutine mapping_create_1d
  ! ................................................

  ! ................................................
  subroutine mapping_create_2d_0( i_mapping, &
                              & d_dim, &
                              & p_u, p_v, &
                              & n_u, n_v, &
                              & np1_u, np1_v, &
                              & knots_u, knots_v, &
                              & control_points, weights)
  implicit none
    integer,                           intent(in) :: i_mapping
    integer,                           intent(in) :: d_dim 
    integer,                           intent(in) :: p_u 
    integer,                           intent(in) :: p_v
    integer,                           intent(in) :: n_u 
    integer,                           intent(in) :: n_v
    integer,                           intent(in) :: np1_u 
    integer,                           intent(in) :: np1_v
    real(8), dimension(np1_u),         intent(in) :: knots_u 
    real(8), dimension(np1_v),         intent(in) :: knots_v
    real(8), dimension(d_dim,n_u,n_v), intent(in) :: control_points
    real(8), dimension(n_u,n_v),       intent(in) :: weights 
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_2d)
      call mapping % create(p_u, p_v, knots_u, knots_v, control_points, weights=weights)
    end select
    ! ...
    
  end subroutine mapping_create_2d_0
  ! ................................................
  
  ! ................................................
  subroutine mapping_create_2d_1( i_mapping, &
                              & d_dim, &
                              & p_u, p_v, &
                              & n_u, n_v, &
                              & np1_u, np1_v, &
                              & knots_u, knots_v, &
                              & control_points, weights, &
                              & i_other)
  implicit none
    integer,                           intent(in) :: i_mapping
    integer,                           intent(in) :: d_dim 
    integer,                           intent(in) :: p_u 
    integer,                           intent(in) :: p_v
    integer,                           intent(in) :: n_u 
    integer,                           intent(in) :: n_v
    integer,                           intent(in) :: np1_u 
    integer,                           intent(in) :: np1_v
    real(8), dimension(np1_u),         intent(in) :: knots_u 
    real(8), dimension(np1_v),         intent(in) :: knots_v
    real(8), dimension(d_dim,n_u,n_v), intent(in) :: control_points
    real(8), dimension(n_u,n_v),       intent(in) :: weights 
    integer,                           intent(in) :: i_other
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_2d)
      select type (other => p_mappings(i_other) % mapping)
      class is (spl_t_mapping_2d)
        call mapping % create(p_u, p_v, knots_u, knots_v, control_points, weights=weights, other=other)
      end select
    end select
    ! ...
    
  end subroutine mapping_create_2d_1
  ! ................................................

  ! ................................................
  subroutine mapping_create_3d( i_mapping, &
                              & d_dim, &
                              & p_u, p_v, p_w, &
                              & n_u, n_v, n_w, &
                              & np1_u, np1_v, np1_w, &
                              & knots_u, knots_v, knots_w, &
                              & control_points, weights)
  implicit none
    integer,                           intent(in) :: i_mapping
    integer,                           intent(in) :: d_dim 
    integer,                           intent(in) :: p_u 
    integer,                           intent(in) :: p_v
    integer,                           intent(in) :: p_w
    integer,                           intent(in) :: n_u 
    integer,                           intent(in) :: n_v
    integer,                           intent(in) :: n_w
    integer,                           intent(in) :: np1_u 
    integer,                           intent(in) :: np1_v
    integer,                           intent(in) :: np1_w
    real(8), dimension(np1_u),         intent(in) :: knots_u 
    real(8), dimension(np1_v),         intent(in) :: knots_v
    real(8), dimension(np1_w),         intent(in) :: knots_w
    real(8), dimension(d_dim,n_u,n_v,n_w), intent(in) :: control_points
    real(8), dimension(n_u,n_v,n_w),       intent(in) :: weights 
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_3d)
      call mapping % create(p_u, p_v, p_w, knots_u, knots_v, knots_w, control_points, weights=weights)
    end select
    ! ...
    
  end subroutine mapping_create_3d
  ! ................................................

  ! ................................................
  subroutine mapping_read_from_file(i_mapping, filename)
  implicit none
    integer, intent(in) :: i_mapping
    character(len=256), intent(in) :: filename
    ! local

    ! ...
    call p_mappings(i_mapping) % mapping % read_from_file(filename)
    ! ...
    
  end subroutine mapping_read_from_file 
  ! ................................................

  ! ................................................
  subroutine mapping_evaluate_1d( i_mapping, &
                                & d_dim, &
                                & n_points_u, &
                                & arr_u, &
                                & arr_y)
  implicit none
    integer,                                   intent(in)  :: i_mapping
    integer,                                   intent(in)  :: d_dim 
    integer,                                   intent(in)  :: n_points_u 
    real(kind=8), dimension(n_points_u),       intent(in)  :: arr_u
    real(kind=8), dimension(d_dim,n_points_u), intent(out) :: arr_y
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_1d)
      call mapping % evaluate( arr_u(1:n_points_u), &
                             & arr_y(1:d_dim, 1:n_points_u))
    end select
    ! ...
    
  end subroutine mapping_evaluate_1d
  ! ................................................

  ! ................................................
  subroutine mapping_evaluate_2d( i_mapping, &
                                & d_dim, &
                                & n_points_u, &
                                & n_points_v, &
                                & arr_u, &
                                & arr_v, &
                                & arr_y)
  implicit none
    integer,                                              intent(in)  :: i_mapping
    integer,                                              intent(in)  :: d_dim 
    integer,                                              intent(in)  :: n_points_u 
    integer,                                              intent(in)  :: n_points_v 
    real(kind=8), dimension(n_points_u),                  intent(in)  :: arr_u
    real(kind=8), dimension(n_points_v),                  intent(in)  :: arr_v
    real(kind=8), dimension(d_dim,n_points_u,n_points_v), intent(out) :: arr_y
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_2d)
      call mapping % evaluate( arr_u(1:n_points_u), arr_v(1:n_points_v), &
                             & arr_y(1:d_dim, 1:n_points_u, 1:n_points_v))
    end select
    ! ...
    
  end subroutine mapping_evaluate_2d
  ! ................................................

  ! ................................................
  subroutine mapping_evaluate_3d( i_mapping, &
                                & d_dim, &
                                & n_points_u, &
                                & n_points_v, &
                                & n_points_w, &
                                & arr_u, &
                                & arr_v, &
                                & arr_w, &
                                & arr_y)
  implicit none
    integer,                                                         intent(in)  :: i_mapping
    integer,                                                         intent(in)  :: d_dim 
    integer,                                                         intent(in)  :: n_points_u 
    integer,                                                         intent(in)  :: n_points_v 
    integer,                                                         intent(in)  :: n_points_w 
    real(kind=8), dimension(n_points_u),                             intent(in)  :: arr_u
    real(kind=8), dimension(n_points_v),                             intent(in)  :: arr_v
    real(kind=8), dimension(n_points_w),                             intent(in)  :: arr_w
    real(kind=8), dimension(d_dim,n_points_u,n_points_v,n_points_w), intent(out) :: arr_y
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_3d)
      call mapping % evaluate(arr_u(1:n_points_u), arr_v(1:n_points_v), arr_w(1:n_points_w), &
                            & arr_y(1:d_dim, 1:n_points_u, 1:n_points_v, 1:n_points_w))
    end select
    ! ...
    
  end subroutine mapping_evaluate_3d
  ! ................................................

  ! ................................................
  subroutine mapping_evaluate_deriv_1_1d( i_mapping, &
                                        & d_dim, &
                                        & n_points_u, &
                                        & arr_u, &
                                        & arr_y, &
                                        & arr_dy)
  implicit none
    integer,                                   intent(in)  :: i_mapping
    integer,                                   intent(in)  :: d_dim 
    integer,                                   intent(in)  :: n_points_u 
    real(kind=8), dimension(n_points_u),       intent(in)  :: arr_u
    real(kind=8), dimension(d_dim,n_points_u), intent(out) :: arr_y
    real(kind=8), dimension(d_dim,n_points_u), intent(out) :: arr_dy
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_1d)
      call mapping % evaluate_deriv(  arr_u(1:n_points_u),          &
                                   &  arr_y= arr_y(1:d_dim, 1:n_points_u), &
                                   & arr_dy=arr_dy(1:d_dim, 1:n_points_u))
    end select
    ! ...
    
  end subroutine mapping_evaluate_deriv_1_1d
  ! ................................................

  ! ................................................
  subroutine mapping_evaluate_deriv_1_2d( i_mapping, &
                                        & n_total_deriv, & 
                                        & d_dim, &
                                        & n_points_u, &
                                        & n_points_v, &
                                        & arr_u, &
                                        & arr_v, &
                                        & arr_y, &
                                        & arr_dy)
  implicit none
    integer,                                              intent(in)  :: i_mapping
    integer,                                              intent(in)  :: n_total_deriv 
    integer,                                              intent(in)  :: d_dim 
    integer,                                              intent(in)  :: n_points_u 
    integer,                                              intent(in)  :: n_points_v 
    real(kind=8), dimension(n_points_u),                  intent(in)  :: arr_u
    real(kind=8), dimension(n_points_v),                  intent(in)  :: arr_v
    real(kind=8), dimension(d_dim,n_points_u,n_points_v), intent(out) :: arr_y
    real(kind=8), dimension(n_total_deriv,d_dim,n_points_u,n_points_v), intent(out) :: arr_dy
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_2d)
      call mapping % evaluate_deriv(  arr_u(1:n_points_u), arr_v(1:n_points_v), &
                             &  arr_y= arr_y(1:d_dim, 1:n_points_u, 1:n_points_v), &
                             & arr_dy=arr_dy(1:n_total_deriv, 1:d_dim, 1:n_points_u, 1:n_points_v))
    end select
    ! ...
    
  end subroutine mapping_evaluate_deriv_1_2d
  ! ................................................

  ! ................................................
  subroutine mapping_evaluate_deriv_1_3d( i_mapping, &
                                        & n_total_deriv, & 
                                        & d_dim, &
                                        & n_points_u, &
                                        & n_points_v, &
                                        & n_points_w, &
                                        & arr_u, &
                                        & arr_v, &
                                        & arr_w, &
                                        & arr_y, &
                                        & arr_dy)
  implicit none
    integer,                                                         intent(in)  :: i_mapping
    integer,                                                         intent(in)  :: n_total_deriv 
    integer,                                                         intent(in)  :: d_dim 
    integer,                                                         intent(in)  :: n_points_u 
    integer,                                                         intent(in)  :: n_points_v 
    integer,                                                         intent(in)  :: n_points_w 
    real(kind=8), dimension(n_points_u),                             intent(in)  :: arr_u
    real(kind=8), dimension(n_points_v),                             intent(in)  :: arr_v
    real(kind=8), dimension(n_points_w),                             intent(in)  :: arr_w
    real(kind=8), dimension(d_dim,n_points_u,n_points_v,n_points_w), intent(out) :: arr_y
    real(kind=8), dimension(n_total_deriv,d_dim,n_points_u,n_points_v,n_points_w), intent(out) :: arr_dy
    ! local

    ! ...
    select type (mapping => p_mappings(i_mapping) % mapping)
    class is (spl_t_mapping_3d)
      call mapping % evaluate_deriv( arr_u(1:n_points_u), arr_v(1:n_points_v), arr_w(1:n_points_w), &
                            &  arr_y= arr_y(1:d_dim, 1:n_points_u, 1:n_points_v, 1:n_points_w), &
                            & arr_dy=arr_dy(1:n_total_deriv,1:d_dim, 1:n_points_u, 1:n_points_v, 1:n_points_w))
    end select
    ! ...
    
  end subroutine mapping_evaluate_deriv_1_3d
  ! ................................................

  ! ................................................
  subroutine mapping_get_d_dim(i_mapping, d_dim)
  implicit none
    integer, intent(in)  :: i_mapping
    integer, intent(out) :: d_dim
    ! local

    ! ... 
    d_dim = p_mappings(i_mapping) % mapping % d_dim
    ! ...
    
  end subroutine mapping_get_d_dim
  ! ................................................

  ! ................................................
  ! TODO use i_format
  subroutine mapping_export(i_mapping, filename, i_format)
  implicit none
    integer, intent(in) :: i_mapping
    character(len=256), intent(in) :: filename
    integer, intent(in) :: i_format
    ! local

    ! ...
    call p_mappings(i_mapping) % mapping % export(filename)
    ! ...
    
  end subroutine mapping_export 
  ! ................................................

  ! ................................................
  subroutine mapping_free(i_mapping)
  implicit none
    integer, intent(in) :: i_mapping
    ! local

    ! ...
    call p_mappings(i_mapping) % mapping % free()
    deallocate(p_mappings(i_mapping) % mapping)
    ! ...
    
  end subroutine mapping_free 
  ! ................................................

  ! ................................................
  subroutine mapping_print_info(i_mapping)
  implicit none
    integer, intent(in) :: i_mapping
    ! local

    ! ...
    call p_mappings(i_mapping) % mapping % print_info()
    ! ...
    
  end subroutine mapping_print_info 
  ! ................................................
  
end module m_django_core
