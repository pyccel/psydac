!> @brief 
!> Module for 2D mappings.
!> @details
!> A 2D mapping is defined as a B-Spline/NURBS surface 

module spl_m_mapping_2d 

use spl_m_global
use spl_m_bsp
use spl_m_pp_form,          only: to_pp_form_2d 
use spl_m_mapping_abstract, only: spl_t_mapping_abstract
use spl_m_mapping_1d,       only: spl_t_mapping_1d
use spl_m_calculus,         only: spl_compute_determinants_jacobians_2d, &
                                & spl_compute_inv_jacobians_2d
implicit none

!  private

  ! .........................................................
  !> @brief 
  !> Class for 2D mappings.
  type, public, extends(spl_t_mapping_abstract) :: spl_t_mapping_2d
     integer :: n_u            !< number of control points in the u-direction
     integer :: p_u            !< spline degree in the u-direction
     integer :: n_v            !< number of control points in the v-direction
     integer :: p_v            !< spline degree in the v-direction
     integer :: n_elements_u   !< number of elements in the u-direction 
     integer :: n_elements_v   !< number of elements in the v-direction

     real(spl_rk), dimension(:),     allocatable :: knots_u !< array of knots of size n_u+p_u+1 in the u-direction
     real(spl_rk), dimension(:),     allocatable :: knots_v !< array of knots of size n_v+p_v+1 in the v-direction
     real(spl_rk), dimension(:,:,:), allocatable :: control_points !< array of control points in IR^d
     real(spl_rk), dimension(:,:),   allocatable :: weights !< array of weights

     real(spl_rk), dimension(:), allocatable :: grid_u !< corresponding grid in the u-direction
     real(spl_rk), dimension(:), allocatable :: grid_v !< corresponding grid in the v-direction

     integer, dimension(:), allocatable :: i_spans_u !< knots indices corresponding to the grid in the u-direction  
     integer, dimension(:), allocatable :: i_spans_v !< knots indices corresponding to the grid in the v-direction
  contains
    procedure :: breaks             => spl_breaks_mapping_2d
    procedure :: clamp              => spl_clamp_mapping_2d
    procedure :: create             => spl_create_mapping_2d
    procedure :: duplicate          => spl_duplicate_mapping_2d
    procedure :: elevate            => spl_elevate_mapping_2d
    procedure :: evaluate           => spl_evaluate_mapping_2d
    procedure :: evaluate_deriv     => spl_evaluate_deriv_mapping_2d
    procedure :: export             => spl_export_mapping_2d
    procedure :: extract            => spl_extract_mapping_2d
    procedure :: free               => spl_free_mapping_2d
    procedure :: insert_knot        => spl_insert_knot_mapping_2d
    procedure :: print_info         => spl_print_mapping_2d
    procedure :: read_from_file     => spl_read_mapping_2d
    procedure :: refine             => spl_refine_mapping_2d
    procedure :: set_control_points => spl_set_mapping_control_points_2d
    procedure :: set_weights        => spl_set_mapping_weights_2d
    procedure :: get_greville       => spl_get_greville_mapping_2d
    procedure :: to_pp              => spl_to_pp_form_mapping_2d
    procedure :: to_us              => spl_to_us_form_mapping_2d
    procedure :: unclamp            => spl_unclamp_mapping_2d
  end type spl_t_mapping_2d
  ! .........................................................

contains

  ! .........................................................
  !> @brief      Creates a 2D mapping 
  !>
  !> @param[inout] self             the current object 
  !> @param[in]    p_u              polynomial degree 
  !> @param[in]    p_v              polynomial degree 
  !> @param[in]    knots_u          the knot vector
  !> @param[in]    knots_v          the knot vector
  !> @param[in]    control_points   array containing the control points
  !> @param[in]    weights          array containing the control points weights [optional] 
  !> @param[in]    other            mapping for the composition [optional] 
  subroutine spl_create_mapping_2d( self,            &
                                  & p_u,             &
                                  & p_v,             &
                                  & knots_u,         &
                                  & knots_v,         &
                                  & control_points,  &
                                  & weights,         &
                                  & other)
  implicit none
     class(spl_t_mapping_2d)     , intent(inout) :: self
     integer                     , intent(in)    :: p_u 
     integer                     , intent(in)    :: p_v
     real(spl_rk), dimension (:) , intent(in)    :: knots_u 
     real(spl_rk), dimension (:) , intent(in)    :: knots_v
     real(spl_rk), dimension(:,:,:), intent(in)    :: control_points
     real(spl_rk), optional, dimension(:,:), intent(in) :: weights 
     class(spl_t_mapping_2d), target, optional, intent(in) :: other

     ! local
     integer :: knots_size_u 
     integer :: knots_size_v

     ! ... manifold dimension
     self % p_u   = p_u 
     self % p_v   = p_v
     self % p_dim = 2
     self % d_dim = size(control_points, 1)
     self % n_u   = size(control_points, 2)
     self % n_v   = size(control_points, 3)
     ! ...

     ! ...
     knots_size_u = size(knots_u, 1)
     knots_size_v = size(knots_v, 1)
     allocate(self % knots_u(knots_size_u))
     allocate(self % knots_v(knots_size_v))

     self % knots_u = knots_u
     self % knots_v = knots_v
     ! ...

     ! ...
     allocate(self % control_points(self % d_dim, self % n_u, self % n_v))

     call self % set_control_points(control_points)
     ! ...

     ! ...
     allocate(self % weights(self % n_u, self % n_v))

     if (present(weights)) then
       call self % set_weights(weights)
     else
       self % weights = 1.0_spl_rk
     end if
     ! ...

     ! ...
     allocate(self % grid_u(self % n_u + self % p_u + 1))
     allocate(self % grid_v(self % n_v + self % p_v + 1))
     allocate(self % i_spans_u(self % n_u + self % p_u + 1))
     allocate(self % i_spans_v(self % n_v + self % p_v + 1))

     call self % breaks(self % n_elements_u, self % grid_u, i_spans=self % i_spans_u, axis=1)
     call self % breaks(self % n_elements_v, self % grid_v, i_spans=self % i_spans_v, axis=2)
      
     ! ...
     if (present(other)) then
      self % ptr_other => other
     end if 
     ! ...

  end subroutine spl_create_mapping_2d
  ! .........................................................

  ! .........................................................
  !> @brief      destroys a 2D mapping 
  !>
  !> @param[inout] self the current object 
  subroutine spl_free_mapping_2d(self)
  implicit none
     class(spl_t_mapping_2d)      , intent(inout) :: self
     ! local

     deallocate(self % knots_u)
     deallocate(self % knots_v)
     deallocate(self % control_points)
     deallocate(self % weights)
     deallocate(self % grid_u)
     deallocate(self % grid_v)
     deallocate(self % i_spans_u)
     deallocate(self % i_spans_v)

  end subroutine spl_free_mapping_2d 
  ! .........................................................

  ! .........................................................
  !> @brief      Duplicates a 2D mapping 
  !>
  !> @param[in]    self    the current object 
  !> @param[inout] other   the new object
  subroutine spl_duplicate_mapping_2d(self, other)
  implicit none
     class(spl_t_mapping_2d), intent(in)    :: self
     class(spl_t_mapping_2d), intent(inout) :: other 
     ! local

     ! ...
     call other % create( self % p_u, &
                        & self % p_v, &
                        & self % knots_u, &
                        & self % knots_v, &
                        & self % control_points, &
                        & weights=self % weights)
     ! ...

  end subroutine spl_duplicate_mapping_2d
  ! .........................................................

  ! .........................................................
  !> @brief      sets control points to a 2D mapping 
  !>
  !> @param[inout] self             the current mapping object 
  !> @param[in]    control_points   the array of shape 3 of control points 
  subroutine spl_set_mapping_control_points_2d(self, control_points)
  implicit none
     class(spl_t_mapping_2d)     , intent(inout) :: self
     real(spl_rk), dimension (:,:,:), intent(in)    :: control_points
     ! local

     ! ... control points
     self % control_points (:,:,:) = control_points(:,:,:) 
     ! ...
   
   end subroutine spl_set_mapping_control_points_2d
  ! .........................................................

  ! .........................................................
  !> @brief      sets control points weights to a 2D mapping 
  !>
  !> @param[inout] self      the current mapping object 
  !> @param[in]    weights   the array of shape 2 of weights 
  subroutine spl_set_mapping_weights_2d(self, weights)
  implicit none
     class(spl_t_mapping_2d)     , intent(inout) :: self
     real(spl_rk), dimension (:,:), intent(in)    :: weights
     ! local

     ! ... control points
     self % weights (:,:) = weights(:,:) 
     ! ...

  end subroutine spl_set_mapping_weights_2d
  ! .........................................................

  ! .........................................................
  !> @brief     prints the current mapping 
  !>
  !> @param[inout] self the current mapping object 
  subroutine spl_print_mapping_2d(self)
  implicit none
    class(spl_t_mapping_2d), intent(in) :: self
    ! local
    integer :: i
    integer :: j

    ! ...
    print *, ">>>> mapping_2d"
    print *, "* n_u                     : ", self % n_u
    print *, "* n_v                     : ", self % n_v
    print *, "* p_u                     : ", self % p_u
    print *, "* p_v                     : ", self % p_v
    print *, "* knots_u                 : ", self % knots_u
    print *, "* knots_v                 : ", self % knots_v
    print *, "* control points / weights: "
      do j = 1, self % n_v
        do i = 1, self % n_u
          print *, i,j,self % control_points(:, i,j),self % weights(i,j) 
        end do
      end do
    print *, "<<<< "
    ! ...

  end subroutine spl_print_mapping_2d 
  ! .........................................................

  ! .........................................................
  !> @brief     unclampes the current mapping 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    sides   enables the unclamping algo on sides [optional]   
  subroutine spl_unclamp_mapping_2d(self, sides)
  implicit none
     class(spl_t_mapping_2d), intent(inout) :: self
     logical, dimension(2,2), optional, intent(in) :: sides 
     ! local
     logical, dimension(2,2) :: l_sides 
     integer :: i
     integer :: j 
     integer :: axis
     real(kind=spl_rk), dimension(:), allocatable :: knots

     ! ...
     l_sides = .TRUE.
     if (present(sides)) then
       l_sides = sides
     end if
     ! ...

     ! ...
     axis = 1
     allocate(knots(self % n_u + self % p_u + 1))

     do j = 1, self % n_v
       knots = self % knots_u

       call Unclamp(self % d_dim &
         & , self % n_u - 1, self % p_u &
         & , knots, self % control_points(:,:,j) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,:,j))
     end do

     self % knots_u = knots
     deallocate(knots)
     ! ...

     ! ...
     axis = 2
     allocate(knots(self % n_v + self % p_v + 1))

     do i = 1, self % n_u
       knots = self % knots_v

       call Unclamp(self % d_dim &
         & , self % n_v - 1, self % p_v &
         & , knots, self % control_points(:,i,:) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,i,:))
     end do

     self % knots_v = knots
     deallocate(knots)
     ! ...

  end subroutine spl_unclamp_mapping_2d 
  ! .........................................................

  ! .........................................................
  !> @brief     evaluates the current mapping 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    arr_u   evaluation u-sites. array of dim(1:n_points_u) 
  !> @param[in]    arr_v   evaluation v-sites. array of dim(1:n_points_v) 
  !> @param[out]   arr_y   values.             array of dim(1:d_dim,1:n_points_u,1:n_points_v)  
  subroutine spl_evaluate_mapping_2d(self, arr_u, arr_v, arr_y)
  implicit none
    class(spl_t_mapping_2d), intent(inout) :: self
    real(kind=spl_rk), dimension(:), intent(in) :: arr_u
    real(kind=spl_rk), dimension(:), intent(in) :: arr_v
    real(kind=spl_rk), dimension(:,:,:), intent(out) :: arr_y
    ! local
    integer :: i_point 
    integer :: i_point_u 
    integer :: i_point_v 
    integer :: n_points
    integer :: n_points_u
    integer :: n_points_v
    real(kind=spl_rk), dimension(:,:,:),   allocatable :: y 
    real(kind=spl_rk), dimension(:,:,:),   allocatable :: mat_x
    real(kind=spl_rk), dimension(:,:,:,:), allocatable :: mat_dx    
    real(kind=spl_rk), dimension(:,:),     allocatable :: arr_x_u 
    real(kind=spl_rk), dimension(:,:),     allocatable :: arr_x_v 
    real(kind=spl_rk), dimension(:),       allocatable :: jacobians 
    real(kind=spl_rk), dimension(:,:,:),   allocatable :: pullback
    real(spl_rk) :: j11
    real(spl_rk) :: j12
    real(spl_rk) :: j21
    real(spl_rk) :: j22
    class(spl_t_mapping_2d), pointer :: ptr_mapping => null()
    
    ! ...
    n_points_u = size(arr_u, 1)
    n_points_v = size(arr_v, 1)
    n_points   = n_points_u * n_points_v
    ! ...
    
    ! ...
    allocate(y(self % d_dim, n_points_u, n_points_v))
    ! ...
    
    ! ...
    call Evaluate2(self % d_dim &
       & , self % n_u - 1, self % p_u &
       & , self % knots_u &
       & , self % n_v - 1, self % p_v &
       & , self % knots_v &
       & , self % control_points &
       & , self % weights &
       & , n_points_u-1, arr_u, n_points_v-1, arr_v, y)
    ! ...
    
    ! ... apply other mapping for the comopisition
    if (associated(self % ptr_other)) then
      ! ...
      select type (other => self % ptr_other)
      class is (spl_t_mapping_2d)
        ! ...
        allocate(mat_x (   other % d_dim, n_points_u, n_points_v))
        allocate(mat_dx(2, other % d_dim, n_points_u, n_points_v))
        
        mat_x  = 0.0_spl_rk
        mat_dx = 0.0_spl_rk 
        ! ...
        
        ! ...
        call other % evaluate_deriv( arr_u(1:n_points_u), &
                                     & arr_v(1:n_points_v), & 
                                     & mat_x(:, 1:n_points_u, 1:n_points_v), &
                                     & mat_dx(:, :, 1:n_points_u, 1:n_points_v)) 
        ! ...
        
        ! ...
        allocate(arr_x_u(other % d_dim, n_points))
        allocate(arr_x_v(other % d_dim, n_points))
        
        allocate(jacobians(n_points))
        allocate(pullback(2, 2, n_points))
        ! ...
        
        ! ...
        i_point = 0
        do i_point_u = 1, n_points_u
        do i_point_v = 1, n_points_v
          i_point = i_point + 1
          
          arr_x_u(:, i_point) = mat_dx(1, :, i_point_u, i_point_v)
          arr_x_v(:, i_point) = mat_dx(2, :, i_point_u, i_point_v)
        end do
        end do
        ! ...
        
        ! ...  
        call spl_compute_determinants_jacobians_2d( arr_x_u(:,1:n_points), & 
                                                  & arr_x_v(:,1:n_points), &
                                                  & jacobians(1:n_points))
        ! ...  
        
        ! ...  
        call spl_compute_inv_jacobians_2d( arr_x_u(:,1:n_points), &
                                         & arr_x_v(:,1:n_points), & 
                                         & jacobians(1:n_points), &
                                         & pullback(:,:,1:n_points) )
          ! ...  
        
        ! ...
        i_point = 0
        do i_point_u = 1, n_points_u
        do i_point_v = 1, n_points_v
          i_point = i_point + 1
          
          ! ...
          j11 = pullback(1,1,i_point) 
          j12 = pullback(1,2,i_point) 
          j21 = pullback(2,1,i_point) 
          j22 = pullback(2,2,i_point)
          ! ...
          
          ! ...
          arr_y(1, i_point_u, i_point_v) = j11 * y(1, i_point_u, i_point_v) &
                                       & + j12 * y(2, i_point_u, i_point_v)   
          
          arr_y(2, i_point_u, i_point_v) = j21 * y(1, i_point_u, i_point_v) &
                                       & + j22 * y(2, i_point_u, i_point_v)   
          ! ...
        end do
        end do
        ! ...
        
        ! ...
        deallocate(arr_x_u)
        deallocate(arr_x_v)
        deallocate(mat_x)
        deallocate(mat_dx)
        deallocate(jacobians)
        deallocate(pullback)
        ! ...
      end select
      ! ...
    else
      ! ...
      i_point = 0
      do i_point_u = 1, n_points_u
      do i_point_v = 1, n_points_v
        i_point = i_point + 1
        
        arr_y(1:2, i_point_u, i_point_v) = y(1:2, i_point_u, i_point_v) 
      end do
      end do
      ! ...
    end if
    ! ...

    ! ...
    deallocate(y)
    ! ...
    
  end subroutine spl_evaluate_mapping_2d 
  ! .........................................................

  ! .........................................................
  !> @brief     evaluates derivatives of the current mapping 
  !>
  !> @param[inout] self     the current mapping object 
  !> @param[in]    arr_u    evaluation u-sites. array of dim(1:n_points_u) 
  !> @param[in]    arr_v    evaluation v-sites. array of dim(1:n_points_v) 
  !> @param[out]   arr_y    values.             array of dim(1:d_dim,1:n_points_u,1:n_points_v)  
  !> @param[out]   arr_dy   1st derivatives.    array of dim(1:n_deriv,1:d_dim,1:n_points_u,1:n_points_v). [n_deriv = 2] 
  !> @param[out]   arr_d2y  2nd derivatives.    array of dim(1:n_deriv,1:d_dim,1:n_points_u,1:n_points_v).  [n_deriv = 3] 
  subroutine spl_evaluate_deriv_mapping_2d(self, arr_u, arr_v, arr_y, arr_dy, arr_d2y)
  implicit none
    class(spl_t_mapping_2d), intent(inout) :: self
    real(kind=spl_rk), dimension(:), intent(in) :: arr_u
    real(kind=spl_rk), dimension(:), intent(in) :: arr_v
    real(kind=spl_rk), dimension(:,:,:), intent(out) :: arr_y
    real(kind=spl_rk), optional, dimension(:,:,:,:), intent(out) :: arr_dy
    real(kind=spl_rk), optional, dimension(:,:,:,:), intent(out) :: arr_d2y
    ! local
    integer :: ru
    integer :: rv
    integer :: n_deriv 
    integer :: n_total_deriv 
    real(kind=spl_rk), dimension(:,:,:,:), allocatable :: Cw

    ! ...
    n_deriv = 0

    if (present(arr_dy)) then
      n_deriv = n_deriv + 1
    end if

    if (present(arr_d2y)) then
      n_deriv = n_deriv + 1
    end if
    ! ...

    ! ... in 2d
    n_total_deriv = 1

    if (n_deriv==1) then
      n_total_deriv = 2 + 1 
    end if
    if (n_deriv==2) then
      n_total_deriv = 2 + 3 + 1 
    end if
    ! ...

    ! ...
    ru = size(arr_u, 1)
    rv = size(arr_v, 1)
    allocate(Cw(n_total_deriv, self % d_dim, ru, rv))
    Cw = 0.0_spl_rk

    call EvaluateDeriv2(n_deriv, n_total_deriv-1 &
      & , self % d_dim &
      & , self % n_u - 1, self % p_u &
      & , self % knots_u &
      & , self % n_v - 1, self % p_v &
      & , self % knots_v &
      & , self % control_points &
      & , self % weights &
      & , ru-1, arr_u, rv-1, arr_v, Cw)
    ! ...

    ! ...
    arr_y(:,:,:) = Cw(1,:,:,:)

    if (present(arr_dy)) then
      arr_dy(1:2,:,:,:) = Cw(2:3,:,:,:)
    end if

    if (present(arr_d2y)) then
      arr_d2y(1:3,:,:,:) = Cw(4:6,:,:,:)
    end if
    ! ...

  end subroutine spl_evaluate_deriv_mapping_2d 
  ! .........................................................

  ! .........................................................
  !> @brief     computes the breaks of the knot vector 
  !>
  !> @param[inout] self         the current mapping object 
  !> @param[inout] n_elements   number of non-zero elements 
  !> @param[inout] grid         the corresponding grid, maximum size = size(knots)
  !> @param[inout] i_spans      the span for every element [optional]  
  !> @param[in]    axis         knot vector axis [optional] 
  subroutine spl_breaks_mapping_2d(self, n_elements, grid, i_spans, axis)
  implicit none
     class(spl_t_mapping_2d), intent(inout) :: self
     integer(kind=4), intent(inout) :: n_elements
     real   (kind=spl_rk), dimension(:), intent(inout) :: grid
     integer, dimension(:), optional, intent(inout) :: i_spans
     integer(kind=4), optional, intent(in) :: axis 
     ! local
     integer :: l_axis
     integer :: i_element
     integer :: i_span
     real(spl_rk) :: x
     integer :: n
     integer :: p
     real(spl_rk), dimension(:), allocatable :: knots 

     ! ...
     l_axis = 1
     if (present(axis)) then
       l_axis = axis
     end if
     ! ...

     ! ...
     if (axis == 1) then
       n = self % n_u ; p = self % p_u
       allocate(knots(n+p+1))
       knots = self % knots_u
     elseif (axis == 2) then
       n = self % n_v ; p = self % p_v
       allocate(knots(n+p+1))
       knots = self % knots_v
     else
       stop "spl_breaks_mapping_2d: wrong arguments"
     end if
     ! ...

     ! ...
     call FindNonZeroElements(p, n + p, knots, n_elements, grid) 
     ! ...

     ! ...
     if (present(i_spans)) then
       i_spans = spl_int_default
       do i_element=1, n_elements 
         x = 0.5_spl_rk * (grid(i_element) + grid(i_element+1)) 
      
         call FindSpan(p, n + p, knots, x, i_span)
         i_spans(i_element) = i_span 
       end do
       i_spans = i_spans + 1
     end if
     ! ...
  end subroutine spl_breaks_mapping_2d
  ! .........................................................

  ! .........................................................
  !> @brief     convert to uniform bspline form 
  !>
  !> @param[inout] self     the current mapping object 
  !> @param[inout] arr_us   values 
  subroutine spl_to_us_form_mapping_2d(self, arr_us)
  implicit none
     class(spl_t_mapping_2d), intent(inout) :: self
     real(kind=spl_rk), dimension(:,:,:,:), intent(inout) :: arr_us
     ! local
     type(spl_t_mapping_2d) :: mapping_tmp
     real(spl_rk), dimension(self % n_u + self % p_u + 1) :: grid_u
     real(spl_rk), dimension(self % n_v + self % p_v + 1) :: grid_v
     integer, dimension(self % n_u + self % p_u + 1) :: i_spans_u
     integer, dimension(self % n_v + self % p_v + 1) :: i_spans_v
     integer :: n_elements_u 
     integer :: n_elements_v
     integer :: i_u 
     integer :: i_v 
     integer :: j_u
     integer :: j_v 
     integer :: i_element
     integer :: i_element_u 
     integer :: i_element_v 
     integer :: span_u 
     integer :: span_v

     ! ... create a copy of self
     call self % duplicate(mapping_tmp)
     ! ...

     ! ... first we unclamp the spline
     !     and then we get the control points
     call mapping_tmp % unclamp()

     call self % breaks(n_elements_u, grid_u, i_spans=i_spans_u, axis=1)
     call self % breaks(n_elements_v, grid_v, i_spans=i_spans_v, axis=2)

     arr_us = spl_int_default * 1.0_spl_rk
     i_element = 0
     do i_element_u = 1, n_elements_u
       span_u = i_spans_u(i_element_u) 
       do i_element_v = 1, n_elements_v
         span_v = i_spans_v(i_element_v) 

         i_element = i_element + 1
         do j_u = 1, self % p_u + 1
            i_u = span_u - 1 - self % p_u + j_u 
           do j_v = 1, self % p_v + 1
              i_v = span_v - 1 - self % p_v + j_v 
              arr_us(:, j_u, j_v, i_element) = mapping_tmp % control_points(:, i_u, i_v)
           end do
         end do
       end do
     end do
     ! ...

     ! ... free 
     call mapping_tmp % free()
     ! ...

  end subroutine spl_to_us_form_mapping_2d
  ! .........................................................

  ! .........................................................
  !> @brief     convert to pp_form
  !>
  !> @param[inout] self     the current mapping object 
  !> @param[inout] arr_pp   values 
  subroutine spl_to_pp_form_mapping_2d(self, arr_pp)
  implicit none
     class(spl_t_mapping_2d), intent(inout) :: self
     real(kind=spl_rk), dimension(:,:,:,:,:), intent(inout) :: arr_pp
    ! local
    integer :: i_dim
    integer :: i_element_v
    integer :: i_element_u
    integer :: i_element
    real(kind=spl_rk), dimension(:,:,:,:), allocatable :: pp_coeff 

    allocate(pp_coeff(self % p_u + 1, self % p_v + 1, self % n_elements_u, self % n_elements_v))

    do i_dim = 1, self % d_dim
      call to_pp_form_2d(self % control_points(i_dim, :, :), &
                       & self % knots_u, self % knots_v, &
                       & self % n_u, self % n_v, &
                       & self % p_u, self % p_v, &
                       & self % n_elements_u, self % n_elements_v, &
                       & pp_coeff)

      arr_pp(i_dim, :, :, :, :) = pp_coeff(:, :, :, :)
      
    end do

  end subroutine spl_to_pp_form_mapping_2d
  ! .........................................................

  ! .........................................................
  !> @brief     clampes the current mapping 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    sides   enables the unclamping algo on sides [optional]
  subroutine spl_clamp_mapping_2d(self, sides)
  implicit none
     class(spl_t_mapping_2d), intent(inout) :: self
     logical, dimension(2,2), optional, intent(in) :: sides 
     ! local
     logical, dimension(2,2) :: l_sides 
     integer :: i
     integer :: j 
     integer :: axis
     real(kind=spl_rk), dimension(:), allocatable :: knots

     ! ...
     l_sides = .TRUE.
     if (present(sides)) then
       l_sides = sides
     end if
     ! ...

     ! ...
     axis = 1
     allocate(knots(self % n_u + self % p_u + 1))

     do j = 1, self % n_v
       knots = self % knots_u

       call Clamp(self % d_dim &
         & , self % n_u - 1, self % p_u &
         & , knots, self % control_points(:,:,j) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,:,j))
     end do

     self % knots_u = knots
     deallocate(knots)
     ! ...

     ! ...
     axis = 2
     allocate(knots(self % n_v + self % p_v + 1))

     do i = 1, self % n_u
       knots = self % knots_v

       call Clamp(self % d_dim &
         & , self % n_v - 1, self % p_v &
         & , knots, self % control_points(:,i,:) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,i,:))
     end do

     self % knots_v = knots
     deallocate(knots)
     ! ...

  end subroutine spl_clamp_mapping_2d 
  ! .........................................................

  ! .........................................................
  !> @brief     get the greville abscissae 
  !>
  !> @param[in]    self  the current mapping object 
  !> @param[inout] us    array containing the greville abscissae 
  !> @param[inout] axis  axis for which we compute the greville abscissae (possible values: 1, 2) 
  subroutine spl_get_greville_mapping_2d(self, us, axis)
  use spl_m_bsp, bsp_greville => Greville
  implicit none
    class(spl_t_mapping_2d),    intent(in)    :: self
    real(spl_rk), dimension(:), intent(inout) :: us
    integer,                    intent(in) :: axis
    ! local

    ! ...
    if (axis == 1) then
      call bsp_greville( self % p_u, self % n_u + self % p_u, &
                       & self % knots_u(1:self % n_u + self % p_u + 1), us) 
    elseif (axis == 2) then 
      call bsp_greville( self % p_v, self % n_v + self % p_v, &
                       & self % knots_v(1:self % n_v + self % p_v + 1), us) 
    else
      stop "spl_get_greville_mapping_2d: wrong value for axis. expect: 1 or 2"
    end if
    ! ...

  end subroutine spl_get_greville_mapping_2d
  ! .........................................................

  ! .........................................................
  !> @brief     inserts the knot t (number of insertion = times) 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    t       knot to insert 
  !> @param[in]    axis    first or second direction 
  !> @param[in]    times   number of times t will be inserted [optional]  
  !> @param[inout] other   the new mapping object [optional]  
  subroutine spl_insert_knot_mapping_2d(self, t, axis, times, other)
  implicit none
     class(spl_t_mapping_2d)          , intent(inout) :: self
     real(spl_rk)                     , intent(in)    :: t 
     integer                          , intent(in)    :: axis
     integer                , optional, intent(in)    :: times 
     class(spl_t_mapping_2d), optional, intent(inout) :: other 
     ! local
     integer :: i
     integer :: d
     integer :: k 
     integer :: j 
     integer :: degree
     integer :: n_u
     integer :: n_v
     integer :: d_dim_ini
     integer :: d_dim_new
     real(spl_rk), dimension(:,:), allocatable :: control_points_crv
     real(spl_rk), dimension(:,:,:), allocatable :: control_points_srf
     type(spl_t_mapping_2d) :: mapping_tmp
     type(spl_t_mapping_1d) :: mapping_crv
     type(spl_t_mapping_2d) :: mapping_srf

     call self % duplicate(mapping_tmp)
     d_dim_ini = mapping_tmp % d_dim

     ! ... u direction: step 1 
     if (axis == 1) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_v

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       allocate(control_points_crv(d_dim_new, n_u))
       do i=1, n_u
         k = 0
         do j=1, n_v
           do d=1, d_dim_ini
             k = k + 1
             control_points_crv(k, i) = mapping_tmp % control_points(d, i, j)
           end do
         end do
       end do
       call mapping_crv % create(mapping_tmp % p_u, mapping_tmp % knots_u, control_points_crv)

       call mapping_crv % insert_knot(t, times=times) 
       deallocate(control_points_crv)
       ! ...

       ! ... u direction: step 2 
       n_u = mapping_crv % n_u
       n_v = mapping_tmp % n_v
       allocate(control_points_srf(d_dim_ini, n_u, n_v))
       do i=1, n_u
         k = 0
         do j=1, n_v
           do d=1, d_dim_ini
             k = k + 1
             control_points_srf(d, i, j) = mapping_crv % control_points(k, i)
           end do
         end do
       end do

       call mapping_srf % create(mapping_crv % p_u, mapping_tmp % p_v &
         & , mapping_crv % knots_u, mapping_tmp % knots_v, control_points_srf )

       call mapping_tmp % free()
       call mapping_srf % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_srf % free()
       deallocate(control_points_srf)
     end if
     ! ...

     ! ... v direction: step 1 
     if (axis == 2) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_u

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v

       allocate(control_points_crv(d_dim_new, mapping_tmp % n_v))
       do j=1, n_v
         k = 0
         do i=1, n_u
           do d=1, d_dim_ini
             k = k + 1
             control_points_crv(k, j) = mapping_tmp % control_points(d, i, j)
           end do
         end do
       end do

       call mapping_crv % create(mapping_tmp % p_v, mapping_tmp % knots_v, control_points_crv)

       call mapping_crv % insert_knot(t, times=times) 
       deallocate(control_points_crv)
       ! ...

       ! ... v direction: step 2 
       n_u = mapping_tmp % n_u
       n_v = mapping_crv % n_u
       allocate(control_points_srf(d_dim_ini, n_u, n_v))
       do j=1, n_v
         k = 0
         do i=1, n_u
           do d=1, d_dim_ini
             k = k + 1
             control_points_srf(d, i, j) = mapping_crv % control_points(k, j)
           end do
         end do
       end do

       call mapping_srf % create(mapping_tmp % p_u, mapping_crv % p_u, &
         & mapping_tmp % knots_u, mapping_crv % knots_u, control_points_srf )

       call mapping_tmp % free()
       call mapping_srf % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_srf % free()
       deallocate(control_points_srf)
     end if
     ! ...

     if (present(other)) then
       call mapping_tmp % duplicate(other)
     else
       call self % free()
       call mapping_tmp % duplicate(self)
     end if
     call mapping_tmp % free()

  end subroutine spl_insert_knot_mapping_2d 
  ! .........................................................
  
  ! .........................................................
  !> @brief     elevates the polynomial degree (number of elevation = times) 
  !>
  !> @param[inout]  self     the current mapping object 
  !> @param[in]     times    number of times the spline degree will be raised 
  !> @param[inout]  other    the new mapping object [optional]
  subroutine spl_elevate_mapping_2d(self, times, other)
  implicit none
     class(spl_t_mapping_2d)          , intent(inout) :: self
     integer, dimension(2)            , intent(in)    :: times 
     class(spl_t_mapping_2d), optional, intent(inout) :: other 
     ! local
     integer :: i
     integer :: d
     integer :: k 
     integer :: j 
     integer :: degree
     integer :: n_u
     integer :: n_v
     integer :: d_dim_ini
     integer :: d_dim_new
     real(spl_rk), dimension(:,:), allocatable :: control_points_crv
     real(spl_rk), dimension(:,:,:), allocatable :: control_points_srf
     type(spl_t_mapping_2d) :: mapping_tmp
     type(spl_t_mapping_1d) :: mapping_crv
     type(spl_t_mapping_2d) :: mapping_srf

     call self % duplicate(mapping_tmp)
     d_dim_ini = mapping_tmp % d_dim

     ! ... u direction: step 1 
     degree = times(1)
     if (degree > 0) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_v

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       allocate(control_points_crv(d_dim_new, n_u))
       do i=1, n_u
         k = 0
         do j=1, n_v
           do d=1, d_dim_ini
             k = k + 1
             control_points_crv(k, i) = mapping_tmp % control_points(d, i, j)
           end do
         end do
       end do
       call mapping_crv % create(mapping_tmp % p_u, mapping_tmp % knots_u, control_points_crv)

       call mapping_crv % elevate(degree)
       deallocate(control_points_crv)
       ! ...

       ! ... u direction: step 2 
       n_u = mapping_crv % n_u
       n_v = mapping_tmp % n_v
       allocate(control_points_srf(d_dim_ini, n_u, n_v))
       do i=1, n_u
         k = 0
         do j=1, n_v
           do d=1, d_dim_ini
             k = k + 1
             control_points_srf(d, i, j) = mapping_crv % control_points(k, i)
           end do
         end do
       end do

       call mapping_srf % create(mapping_crv % p_u, mapping_tmp % p_v &
         & , mapping_crv % knots_u, mapping_tmp % knots_v, control_points_srf )

       call mapping_tmp % free()
       call mapping_srf % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_srf % free()
       deallocate(control_points_srf)
     end if
     ! ...

     ! ... v direction: step 1 
     degree = times(2)
     if (degree > 0) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_u

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v

       allocate(control_points_crv(d_dim_new, mapping_tmp % n_v))
       do j=1, n_v
         k = 0
         do i=1, n_u
           do d=1, d_dim_ini
             k = k + 1
             control_points_crv(k, j) = mapping_tmp % control_points(d, i, j)
           end do
         end do
       end do

       call mapping_crv % create(mapping_tmp % p_v, mapping_tmp % knots_v, control_points_crv)

       call mapping_crv % elevate(degree)
       deallocate(control_points_crv)
       ! ...

       ! ... v direction: step 2 
       n_u = mapping_tmp % n_u
       n_v = mapping_crv % n_u
       allocate(control_points_srf(d_dim_ini, n_u, n_v))
       do j=1, n_v
         k = 0
         do i=1, n_u
           do d=1, d_dim_ini
             k = k + 1
             control_points_srf(d, i, j) = mapping_crv % control_points(k, j)
           end do
         end do
       end do

       call mapping_srf % create(mapping_tmp % p_u, mapping_crv % p_u, &
         & mapping_tmp % knots_u, mapping_crv % knots_u, control_points_srf )

       call mapping_tmp % free()
       call mapping_srf % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_srf % free()
       deallocate(control_points_srf)
     end if
     ! ...

     if (present(other)) then
       call mapping_tmp % duplicate(other)
     else
       call self % free()
       call mapping_tmp % duplicate(self)
     end if
     call mapping_tmp % free()

  end subroutine spl_elevate_mapping_2d 
  ! .........................................................

  ! .........................................................
  !> @brief     refines a 2d mapping 
  !>
  !> @param[inout] self         the current mapping object 
  !> @param[in]    n_elements   a new subdivision [optional]
  !> @param[in]    degrees      number of times the spline degree will be raised [optional]
  !> @param[inout] other        the new mapping object [optional]   
  !> @param[in]    verbose      print details about the refinement when True [optional]   
  !> \TODO add knots and grid 
  subroutine spl_refine_mapping_2d(self, degrees, n_elements, other, verbose)
  implicit none
     class(spl_t_mapping_2d), intent(inout) :: self
     integer, dimension(2)  , optional, intent(in)    :: degrees 
     integer, dimension(2)  , optional, intent(in)    :: n_elements 
     class(spl_t_mapping_2d), optional, intent(inout) :: other 
     logical                , optional, intent(in)    :: verbose
     ! local
     integer :: i
     integer :: axis
     logical :: l_verbose
     type(spl_t_mapping_2d)          , target :: mapping_tmp
     real(spl_rk) :: t

     l_verbose = .False.
     if (present(verbose)) then
       l_verbose = verbose
     end if

     call mapping_tmp % create(self % p_u, self % p_v, self % knots_u, self % knots_v, self % control_points)

     ! ...
     if (l_verbose) then
       print *, "==== initial mapping ===="
       call mapping_tmp % print_info()
       print *, "=============="
     end if
     ! ...

     ! ... add degree elevation 
     if (present(degrees)) then
       if (maxval(degrees) > 0) then
         ! ...
         if (l_verbose) then
           print *, "==== before elevation ======"
           call mapping_tmp % print_info()
           print *, "=============="
         end if
         ! ...

         call mapping_tmp % elevate(degrees)
       end if
     end if
     ! ...

     ! ... knots insertion
     if (present(n_elements)) then
       if (maxval(n_elements) > 1) then
         ! ...
         if (l_verbose) then
           print *, "==== before insertion ======"
           call mapping_tmp % print_info()
           print *, "=============="
         end if
         ! ...

         ! ...
         axis = 1 
         do i = 1, n_elements(axis) - 1
           t = (i * 1.0_spl_rk / n_elements(axis)) 
           call mapping_tmp % insert_knot(t, axis)
         end do
         ! ...

         ! ...
         axis = 2
         do i = 1, n_elements(axis) - 1
           t = (i * 1.0_spl_rk / n_elements(axis)) 
           call mapping_tmp % insert_knot(t, axis)
         end do
         ! ...

       end if
     end if
     ! ...

     if (l_verbose) then
       print *, "==== final mapping ======"
       call mapping_tmp % print_info()
       print *, "=============="
     end if

     if (present(other)) then
       call mapping_tmp % duplicate(other)
     else
       call self % free()
       call mapping_tmp % duplicate(self)
     end if

     call mapping_tmp % free() 

  end subroutine spl_refine_mapping_2d
  ! ...................................................
   
  ! ...................................................
  !> @brief     Extracts a B-Spline curve from a B-Spline surface
  !>
  !> @param[inout] self       the current mapping object 
  !> @param[inout] other      a 1d mapping 
  !> @param[in]    axis       direction axis 
  !> @param[in]    face       face id for the given axis 
  subroutine spl_extract_mapping_2d(self, other, axis, face)
  implicit none
    class(spl_t_mapping_2d), intent(inout) :: self
    class(spl_t_mapping_1d), intent(inout) :: other 
    integer                , intent(in)    :: axis 
    integer                , intent(in)    :: face
    ! local
    integer :: p
    integer :: n
    integer :: d_dim 
    real(spl_rk), dimension (:) , allocatable :: knots
    real(spl_rk), dimension(:,:), allocatable :: control_points
    real(spl_rk), dimension(:), allocatable :: weights 
    
    ! ...
    if ((axis > self % p_dim) .or. (axis <= 0)) then
      print *, "spl_extract_mapping_2d: wrong value for axis. Given ", axis
      stop
    end if
    ! ...

    ! ...
    if ((face > 2) .or. (face <= 0)) then
      print *, "spl_extract_mapping_2d: wrong value for face. Given ", face
      stop
    end if
    ! ...

    ! ...

    ! ...
    if (axis == 1) then
      ! ...
      p     = self % p_v
      n     = self % n_v
      d_dim = self % d_dim

      allocate(knots(n+p+1))
      allocate(control_points(d_dim, n))
      allocate(weights(n))
      knots = self % knots_v
      ! ...

      ! ...
      if (face == 1) then
        control_points = self % control_points(:,1,:)
        weights        = self % weights(1,:)
      elseif (face == 2) then                 
        control_points = self % control_points(:,self % n_u,:)
        weights        = self % weights(self % n_u,:)
      end if
      ! ...
    elseif (axis == 2) then
      ! ...
      p     = self % p_u
      n     = self % n_u
      d_dim = self % d_dim

      allocate(knots(n+p+1))
      allocate(control_points(d_dim, n))
      allocate(weights(n))
      knots = self % knots_u
      ! ...

      ! ...
      if (face == 1) then
        control_points = self % control_points(:,:,1)
        weights        = self % weights(:,1)
      elseif (face == 2) then                 
        control_points = self % control_points(:,:,self % n_v)
        weights        = self % weights(:,self % n_v)
      end if
      ! ...
    end if
    ! ...

    ! ...
    call other % create(p, knots, control_points, weights=weights)
    ! ...

  end subroutine spl_extract_mapping_2d
  ! ...................................................
   
  ! ...................................................
  !> @brief     Exports the B-Spline mapping to file 
  !>
  !> @param[inout] self       the current mapping object 
  !> @param[in]    filename   output filename 
  !> @param[in]    i_format   output format 
  subroutine spl_export_mapping_2d(self, filename, i_format)
  implicit none
    class(spl_t_mapping_2d), intent(inout) :: self
    character(len=*)       , intent(in)    :: filename
    integer      , optional, intent(in)    :: i_format
    ! local

    ! ...
    if (present(i_format)) then
       if (i_format == spl_mapping_format_nml) then
          call export_mapping_2d_nml(self, filename)
       else
          stop "spl_export_mapping_2d: format not yet supported"
       end if
    else
       call export_mapping_2d_nml(self, filename)
    end if
    ! ...

  end subroutine spl_export_mapping_2d 
  ! ...................................................
   
  ! ...................................................
  !> @brief     Exports the B-Spline mapping to a namelist file 
  !>
  !> @param[inout] self       the current mapping object 
  !> @param[in]    filename   output filename 
  subroutine export_mapping_2d_nml(self, filename)
  implicit none
    class(spl_t_mapping_2d), intent(inout) :: self
    character(len=*)       , intent(in)    :: filename
    ! local
    integer :: IO_stat
    integer, parameter :: input_file_id = 111
    integer :: ierr
    integer :: spline_deg1
    integer :: spline_deg2
    integer :: num_pts1
    integer :: num_pts2
    character(len=256) :: label
    real(kind=spl_rk), dimension(:), allocatable :: knots1
    real(kind=spl_rk), dimension(:), allocatable :: knots2
    real(kind=spl_rk), dimension(:), allocatable :: control_pts1
    real(kind=spl_rk), dimension(:), allocatable :: control_pts2
    real(kind=spl_rk), dimension(:), allocatable :: control_pts3
    real(kind=spl_rk), dimension(:), allocatable :: control_weights
    real(kind=spl_rk) :: eta1_min_minimal
    real(kind=spl_rk) :: eta1_max_minimal
    real(kind=spl_rk) :: eta2_min_minimal
    real(kind=spl_rk) :: eta2_max_minimal
    integer  :: bc_left
    integer  :: bc_right
    integer  :: bc_bottom
    integer  :: bc_top
    integer  :: number_cells1,number_cells2
    integer :: sz_knots1,sz_knots2
    integer :: i,j,i_current
    integer :: d_dim
  
    namelist /transf_label/  label
    namelist /d_dimension/  d_dim
    namelist /degree/   spline_deg1, spline_deg2
    namelist /shape/    num_pts1, num_pts2 ! it is not the number of points but the number of coeff sdpline in each direction !!
    namelist /knots_1/   knots1
    namelist /knots_2/   knots2
    namelist /control_points_1/ control_pts1
    namelist /control_points_2/ control_pts2
    namelist /control_points_3/ control_pts3
    namelist /weights/  control_weights

    open(unit=input_file_id, file=trim(filename))

    ! write the label
    label = trim("test_patch") 
    write( input_file_id, transf_label )

    ! write the manifold dimension
    d_dim = self % d_dim
    write( input_file_id, d_dimension )

    ! write the degree of spline
    spline_deg1 = self % p_u
    spline_deg2 = self % p_v 
    write( input_file_id, degree )

    ! write ....?
    num_pts1 = self % n_u
    num_pts2 = self % n_v
    write( input_file_id, shape )

!    ! write if we use NURBS or not
!    ! Allocations of knots to construct the splines

    allocate(knots1(num_pts1 + self % p_u + 1))
    allocate(knots2(num_pts2 + self % p_v + 1))
    knots1 = self % knots_u
    knots2 = self % knots_v
    ! write the knots associated to each direction 
    
    ! we don't use the namelist here: problem with repeated factors in the namelist
    write( input_file_id, *) "&KNOTS_1"
    write( input_file_id, *) "KNOTS1= ", self % knots_u
    write( input_file_id, *) "/"
    write( input_file_id, *) "&KNOTS_2"
    write( input_file_id, *) "KNOTS2= ", self % knots_v
    write( input_file_id, *) "/"
    
    ! allocations of tables containing control points in each direction 
    ! here its table 1D
    allocate(control_pts1(num_pts1*num_pts2))
    allocate(control_pts2(num_pts1*num_pts2))
    allocate(control_pts3(num_pts1*num_pts2))
    allocate(control_weights(num_pts1*num_pts2))
    ! ...

    ! ...
    i_current = 0
    do j = 1, num_pts2
    do i = 1, num_pts1
      i_current = i_current + 1
      control_weights(i_current) = self % weights(i,j) 

      control_pts1(i_current) = self % control_points(1,i,j) 
      if (self % d_dim .ge. 2) then
        control_pts2(i_current) = self % control_points(2,i,j) 
        if (self % d_dim .ge. 3) then
          control_pts3(i_current) = self % control_points(3,i,j) 
        end if
      end if
    end do
    end do
    ! ...

    ! we don't use the namelist here: problem with repeated factors in the namelist
    if ( self % d_dim >= 1) then 
      write( input_file_id, *) "&CONTROL_POINTS_1"
      write( input_file_id, *) "CONTROL_PTS1= ", control_pts1
      write( input_file_id, *) "/"
    end if
    if ( self % d_dim >= 2) then 
      write( input_file_id, *) "&CONTROL_POINTS_2"
      write( input_file_id, *) "CONTROL_PTS2= ", control_pts2
      write( input_file_id, *) "/"
    end if
    if ( self % d_dim >= 3) then 
      write( input_file_id, *) "&CONTROL_POINTS_3"
      write( input_file_id, *) "CONTROL_PTS3= ", control_pts3
      write( input_file_id, *) "/"
    end if

    write( input_file_id, *) "&CONTROL_WEIGHTS"
    write( input_file_id, *) "WEIGHTS= ", control_weights
    write( input_file_id, *) "/"

    close(input_file_id)

  end subroutine export_mapping_2d_nml 
  ! ...................................................

  ! ...................................................
  !> @brief     create a new 2d mapping reading from a namelist file
  !> @param[inout] self       the current mapping object 
  !> @param[in]    filename   the name of the file to be read from
  subroutine spl_read_mapping_2d(self, filename)
    implicit none
    class(spl_t_mapping_2d), intent(inout) :: self
    character(len=*)       , intent(in)    :: filename
    ! LOCAL
    integer :: IO_stat
    integer, parameter :: input_file_id = 111112
    character(len=256) :: label
    integer :: spline_deg1
    integer :: spline_deg2
    integer :: num_pts1
    integer :: num_pts2
    integer :: d_dim
    real(spl_rk), dimension(:), allocatable :: knots1
    real(spl_rk), dimension(:), allocatable :: knots2
    real(spl_rk), dimension(:), allocatable :: weights
    real(spl_rk), dimension(:,:), allocatable :: weights_2d
    real(spl_rk), dimension(:), allocatable :: control_pts1
    real(spl_rk), dimension(:), allocatable :: control_pts2
    real(spl_rk), dimension(:), allocatable :: control_pts3
    integer  :: number_cells1
    integer  :: number_cells2
    real(spl_rk), dimension(:,:,:), allocatable  :: control_points3d
        
    namelist /transf_label/  label
    namelist /d_dimension/   d_dim
    namelist /degree/   spline_deg1,spline_deg2
    namelist /shape/    num_pts1, num_pts2 ! it is not the number of points but the number of coeff sdpline in each direction !!
    namelist /knots_1/   knots1
    namelist /knots_2/   knots2
    namelist /control_points_1/ control_pts1
    namelist /control_points_2/ control_pts2
    namelist /control_points_3/ control_pts3
    namelist /control_weights/  weights
    namelist /cartesian_mesh_2d/ number_cells1,number_cells2
    open(unit=input_file_id, file=filename, STATUS="OLD", IOStat=IO_stat)
    !> read label
    read( input_file_id, transf_label )
    !> read manifold dim
    read( input_file_id, d_dimension )
    !> read spline degrees
    read( input_file_id, degree )
    !> read number of points
    read( input_file_id, shape )
    !> Allocations of knots to construct the splines
    allocate(knots1 (num_pts1+spline_deg1+1))
    allocate(knots2 (num_pts2+spline_deg2+1))
    read( input_file_id, knots_1 )
    read( input_file_id, knots_2 )
 
    !>TODO 
    !>Allocation of table containing the weights associated
    !> to each control points
    allocate(weights(num_pts1*num_pts2))
    allocate(weights_2d(num_pts1, num_pts2))
    !> allocations of tables containing control points in each direction
    allocate(control_pts1(num_pts1*num_pts2))
    allocate(control_pts2(num_pts1*num_pts2))
    allocate(control_pts3(num_pts1*num_pts2))
    !>allocation of control_points
    allocate(control_points3d(d_dim,num_pts1,num_pts2))

    if (d_dim >= 1) then
      read( input_file_id, control_points_1)
      control_points3d(1,:,:) = reshape(control_pts1,(/num_pts1,num_pts2/))
    end if
    if (d_dim >= 2) then
      read( input_file_id, control_points_2)
      control_points3d(2,:,:) = reshape(control_pts2,(/num_pts1,num_pts2/))
    end if
    if (d_dim >= 3) then
      read( input_file_id, control_points_3)
      control_points3d(3,:,:) = reshape(control_pts3,(/num_pts1,num_pts2/))
    end if
    read( input_file_id, control_weights)
    weights_2d(:,:) = reshape(weights,(/num_pts1,num_pts2/))

    call spl_create_mapping_2d(   self            &
                                & , spline_deg1         &
                                & , spline_deg2         &
                                & , knots1              &
                                & , knots2              &
                                & , control_points3d    &
                                & , weights=weights_2d  &
                                &  )

    close(input_file_id)

  end subroutine spl_read_mapping_2d
  ! ...................................................

end module spl_m_mapping_2d 
