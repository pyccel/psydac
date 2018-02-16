!> @brief 
!> Module for 1D mappings.
!> @details
!> A 1D mapping is defined as a B-Spline/NURBS curve 

module spl_m_mapping_1d 

use spl_m_global
use spl_m_pp_form,          only: to_pp_form_1d 
use spl_m_mapping_abstract, only: spl_t_mapping_abstract

implicit none

  ! .........................................................
  !> @brief 
  !> Class for 1D mappings.
  type, public, extends(spl_t_mapping_abstract) :: spl_t_mapping_1d
     integer :: n_u           !< number of control points
     integer :: p_u           !< spline degree            
     integer :: n_elements_u  !< number of elements

     real(spl_rk), dimension(:)  , allocatable :: knots_u         !< array of knots of size n_u+p_u+1 
     real(spl_rk), dimension(:,:), allocatable :: control_points  !< array of control points in IR^d
     real(spl_rk), dimension(:)  , allocatable :: weights         !< array of weights 

     real(spl_rk), dimension(:)  , allocatable :: grid_u          !< corresponding grid 
     integer     , dimension(:)  , allocatable :: i_spans_u       !< knots indices corresponding to the grid
  contains
    procedure :: breaks             => spl_breaks_mapping_1d
    procedure :: clamp              => spl_clamp_mapping_1d
    procedure :: create             => spl_create_mapping_1d
    procedure :: duplicate          => spl_duplicate_mapping_1d
    procedure :: elevate            => spl_elevate_mapping_1d
    procedure :: evaluate           => spl_evaluate_mapping_1d
    procedure :: evaluate_deriv     => spl_evaluate_deriv_mapping_1d
    procedure :: export             => spl_export_mapping_1d
    procedure :: free               => spl_free_mapping_1d
    procedure :: insert_knot        => spl_insert_knot_mapping_1d
    procedure :: print_info         => spl_print_mapping_1d
    procedure :: read_from_file     => spl_read_mapping_1d
    procedure :: refine             => spl_refine_mapping_1d
    procedure :: set_control_points => spl_set_mapping_control_points_1d
    procedure :: set_weights        => spl_set_mapping_weights_1d
    procedure :: get_greville       => spl_get_greville_mapping_1d
    procedure :: to_pp              => spl_to_pp_form_mapping_1d
    procedure :: to_us              => spl_to_us_form_mapping_1d
    procedure :: unclamp            => spl_unclamp_mapping_1d
  end type spl_t_mapping_1d
  ! .........................................................

contains

  ! .........................................................
  !> @brief      Creates a 1D mapping 
  !>
  !> @param[inout] self             the current object 
  !> @param[in]    p                polynomial degree 
  !> @param[in]    knots            the knot vector
  !> @param[in]    control_points   array containing the control points
  !> @param[in]    weights          array containing the control points weights [optional]
  subroutine spl_create_mapping_1d( self,           &
                                  & p,              &
                                  & knots,          &
                                  & control_points, &
                                  & weights)        
  implicit none
     class(spl_t_mapping_1d)     , intent(inout) :: self
     integer                     , intent(in)    :: p
     real(spl_rk), dimension (:) , intent(in)    :: knots
     real(spl_rk), dimension(:,:), intent(in)    :: control_points
     real(spl_rk), optional, dimension(:), intent(in) :: weights 
     ! local
     integer :: knots_size

     ! ... manifold dimension
     self % p_u   = p
     self % p_dim = 1
     self % d_dim = size(control_points, 1)
     self % n_u   = size(control_points, 2)
     ! ...

     ! ...
     knots_size = size(knots, 1)
     allocate(self % knots_u(knots_size))

     self % knots_u = knots
     ! ...

     ! ...
     allocate(self % control_points(self % d_dim, self % n_u))

     call self % set_control_points(control_points)
     ! ...

     ! ...
     allocate(self % weights(self % n_u))

     if (present(weights)) then
       call self % set_weights(weights)
     else
       self % weights = 1.0_spl_rk
     end if
     ! ...

     ! ...
     allocate(self % grid_u(self % n_u + self % p_u + 1))
     allocate(self % i_spans_u(self % n_u + self % p_u + 1))

     call self % breaks(self % n_elements_u, self % grid_u, i_spans=self % i_spans_u)
     ! ...
  end subroutine spl_create_mapping_1d
  ! .........................................................

  ! .........................................................
  !> @brief      destroys a 1D mapping 
  !>
  !> @param[inout] self the current object 
  subroutine spl_free_mapping_1d(self)
  implicit none
     class(spl_t_mapping_1d)      , intent(inout) :: self
     ! local

     deallocate(self % knots_u)
     deallocate(self % control_points)
     deallocate(self % weights)
     deallocate(self % grid_u)
     deallocate(self % i_spans_u)

  end subroutine spl_free_mapping_1d 
  ! .........................................................

  ! .........................................................
  !> @brief      Duplicates a 1D mapping 
  !>
  !> @param[in]    self    the current object 
  !> @param[inout] other   the new object
  subroutine spl_duplicate_mapping_1d(self, other)
  implicit none
     class(spl_t_mapping_1d), intent(in)    :: self
     class(spl_t_mapping_1d), intent(inout) :: other 
     ! local

     ! ...
     call other % create( self % p_u, &
                        & self % knots_u, &
                        & self % control_points, &
                        & weights=self % weights)
     ! ...

  end subroutine spl_duplicate_mapping_1d
  ! .........................................................

  ! .........................................................
  !> @brief      sets control points to a 1D mapping 
  !>
  !> @param[inout] self             the current mapping object 
  !> @param[in]    control_points   the array of shape 2 of control points 
  subroutine spl_set_mapping_control_points_1d(self, control_points)
  implicit none
     class(spl_t_mapping_1d)     , intent(inout) :: self
     real(spl_rk), dimension (:,:), intent(in)    :: control_points
     ! local

     ! ... control points
     self % control_points (:,:) = control_points(:,:) 
     ! ...

  end subroutine spl_set_mapping_control_points_1d
  ! .........................................................

  ! .........................................................
  !> @brief      sets control points weights to a 1D mapping 
  !>
  !> @param[inout] self      the current mapping object 
  !> @param[in]    weights   the array of shape 1 of weights 
  subroutine spl_set_mapping_weights_1d(self, weights)
  implicit none
     class(spl_t_mapping_1d)     , intent(inout) :: self
     real(spl_rk), dimension (:), intent(in)    :: weights
     ! local

     ! ... control points
     self % weights (:) = weights(:) 
     ! ...

  end subroutine spl_set_mapping_weights_1d
  ! .........................................................

  ! .........................................................
  !> @brief     prints the current mapping 
  !>
  !> @param[inout] self the current mapping object 
  subroutine spl_print_mapping_1d(self)
  implicit none
    class(spl_t_mapping_1d), intent(in) :: self
    ! local
    integer :: i

    ! ...
    print *, ">>>> mapping_1d"
    print *, "* n_u                     : ", self % n_u
    print *, "* p_u                     : ", self % p_u
    print *, "* knots                   : ", self % knots_u
    print *, "* control points / weights: "
      do i = 1, self % n_u
        print *, i,self % control_points(:, i), self % weights(i) 
      end do
    print *, "<<<< "
    ! ...

  end subroutine spl_print_mapping_1d 
  ! .........................................................

  ! .........................................................
  !> @brief     unclampes the current mapping 
  !>
  !> @param[inout] self  the current mapping object 
  !> @param[in]    sides enables the unclamping algo on sides [optional]  
  subroutine spl_unclamp_mapping_1d(self, sides)
  use spl_m_bsp, bsp_unclamp => Unclamp
  implicit none
     class(spl_t_mapping_1d), intent(inout) :: self
     logical, dimension(2), optional, intent(in) :: sides 
     ! local
     logical, dimension(2) :: l_sides 

     ! ...
     l_sides = (/ .TRUE., .TRUE. /)
     if (present(sides)) then
       l_sides = sides
     end if
     ! ...

     ! ...
     call bsp_unclamp(self % d_dim &
       & , self % n_u - 1, self % p_u &
       & , self % knots_u, self % control_points &
       & , l_sides(1), l_sides(2) &
       & , self % knots_u, self % control_points)
     ! ...

  end subroutine spl_unclamp_mapping_1d 
  ! .........................................................

  ! .........................................................
  !> @brief     clampes the current mapping 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    sides   enables the unclamping algo on sides [optional]
  subroutine spl_clamp_mapping_1d(self, sides)
  use spl_m_bsp, bsp_clamp => Clamp
  implicit none
     class(spl_t_mapping_1d), intent(inout) :: self
     logical, dimension(2), optional, intent(in) :: sides 
     ! local
     logical, dimension(2) :: l_sides 

     ! ...
     l_sides = (/ .TRUE., .TRUE. /)
     if (present(sides)) then
       l_sides = sides
     end if
     ! ...

     ! ...
     call bsp_clamp(self % d_dim &
       & , self % n_u - 1, self % p_u &
       & , self % knots_u, self % control_points &
       & , l_sides(1), l_sides(2) &
       & , self % knots_u, self % control_points)
     ! ...

  end subroutine spl_clamp_mapping_1d 
  ! .........................................................

  ! .........................................................
  !> @brief     evaluates the current mapping 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    arr_u   evaluation sites. array of dim(1:n_points_u) 
  !> @param[out]   arr_y   values.           array of dim(1:d_dim,1:n_points_u)  
  subroutine spl_evaluate_mapping_1d(self, arr_u, arr_y)
  use spl_m_bsp, bsp_evaluate_1d => Evaluate1
  implicit none
     class(spl_t_mapping_1d), intent(inout) :: self
     real(kind=spl_rk), dimension(:), intent(in) :: arr_u
     real(kind=spl_rk), dimension(:,:), intent(out) :: arr_y
     ! local
     integer :: r

     ! ...
     r = size(arr_u, 1)

     call bsp_evaluate_1d(self % d_dim &
       & , self % n_u - 1, self % p_u &
       & , self % knots_u, self % control_points, self % weights &
       & , r-1, arr_u, arr_y)
     ! ...

  end subroutine spl_evaluate_mapping_1d 
  ! .........................................................

  ! .........................................................
  !> @brief     evaluates derivatives of the current mapping 
  !>
  !> @param[inout] self     the current mapping object 
  !> @param[in]    arr_u    evaluation sites. array of dim(1:n_points_u) 
  !> @param[out]   arr_y    values.           array of dim(1:d_dim,1:n_points_u)  
  !> @param[out]   arr_dy   1st derivatives.  array of dim(1:d_dim,1:n_points_u)  
  !> @param[out]   arr_d2y  2nd derivatives.  array of dim(1:d_dim,1:n_points_u)  
  subroutine spl_evaluate_deriv_mapping_1d(self, arr_u, arr_y, arr_dy, arr_d2y)
  use spl_m_bsp, bsp_evaluate_deriv_1d => EvaluateDeriv1
  implicit none
    class(spl_t_mapping_1d), intent(inout) :: self
    real(kind=spl_rk), dimension(:), intent(in) :: arr_u
    real(kind=spl_rk), dimension(:,:), intent(out) :: arr_y
    real(kind=spl_rk), optional, dimension(:,:), intent(out) :: arr_dy
    real(kind=spl_rk), optional, dimension(:,:), intent(out) :: arr_d2y
    ! local
    integer :: r
    integer :: rationalize
    integer :: n_deriv 
    integer :: n_total_deriv 
    real(kind=spl_rk), dimension(:,:,:), allocatable :: Cw

    ! ...
    rationalize = 0
    if (self % rationalize) then
      rationalize = 1
    end if 
    ! ...

    ! ...
    n_deriv = 0

    if (present(arr_dy)) then
      n_deriv = n_deriv + 1
    end if

    if (present(arr_d2y)) then
      n_deriv = n_deriv + 1
    end if
    ! ...

    ! ... in 1d
    n_total_deriv = n_deriv + 1 
    ! ...

    ! ...
    r = size(arr_u, 1)
    allocate(Cw(n_total_deriv, self % d_dim, r))
    Cw = 0.0_spl_rk

    call bsp_evaluate_deriv_1d(n_deriv, n_total_deriv-1 &
      & , self % d_dim &
      & , self % n_u - 1, self % p_u &
      & , self % knots_u, self % control_points, self % weights &
      & , r-1, arr_u, Cw)
    ! ...

    ! ...
    arr_y(:,:) = Cw(1,:,:)

    if (present(arr_dy)) then
      arr_dy(:,:) = Cw(2,:,:)
    end if

    if (present(arr_d2y)) then
      arr_d2y(:,:) = Cw(3,:,:)
    end if
    ! ...

  end subroutine spl_evaluate_deriv_mapping_1d 
  ! .........................................................

  ! .........................................................
  !> @brief     computes the breaks of the knot vector 
  !>
  !> @param[inout] self         the current mapping object 
  !> @param[inout] n_elements   number of non-zero elements 
  !> @param[inout] grid         the corresponding grid, maximum size = size(knots)
  !> @param[inout] i_spans      the span for every element [optional]  
  subroutine spl_breaks_mapping_1d(self, n_elements, grid, i_spans)
  use spl_m_bsp, only : FindNonZeroElements, FindSpan 
  implicit none
     class(spl_t_mapping_1d), intent(inout) :: self
     integer(kind=4), intent(inout) :: n_elements
     real   (kind=spl_rk), dimension(:), intent(inout) :: grid
     integer, dimension(:), optional, intent(inout) :: i_spans
     ! local
     integer :: i_element
     integer :: i_span
     real(spl_rk) :: x

     ! ...
     call FindNonZeroElements( &
       & self % p_u, self % n_u + self % p_u, self % knots_u, &
       & n_elements, grid) 
     ! ...

     ! ...
     if (present(i_spans)) then
       i_spans = spl_int_default
       do i_element=1, n_elements 
         x = 0.5_spl_rk * (grid(i_element) + grid(i_element+1)) 
      
         call FindSpan(self % p_u, self % n_u + self % p_u, self % knots_u, x, i_span)
         i_spans(i_element) = i_span 
       end do
       i_spans = i_spans + 1
     end if
     ! ...

  end subroutine spl_breaks_mapping_1d
  ! .........................................................

  ! .........................................................
  !> @brief     convert to uniform bspline form 
  !>
  !> @param[inout] self     the current mapping object 
  !> @param[inout] arr_us   values 
  subroutine spl_to_us_form_mapping_1d(self, arr_us)
  implicit none
     class(spl_t_mapping_1d), intent(inout) :: self
     real(kind=spl_rk), dimension(:,:,:), intent(inout) :: arr_us
     ! local
     type(spl_t_mapping_1d) :: mapping_tmp
     integer :: n_elements
     real(spl_rk), dimension(self % n_u + self % p_u + 1) :: grid
     integer, dimension(self % n_u + self % p_u + 1) :: i_spans
     integer :: i
     integer :: i_element 
     integer :: j 
     integer :: p 
     integer :: span

     ! ... create a copy of self
     call self % duplicate(mapping_tmp)
     ! ...

     ! ... first we unclamp the spline
     !     and then we get the control points
     call mapping_tmp % unclamp()

     call self % breaks(n_elements, grid, i_spans=i_spans)

     p = self % p_u
     arr_us = spl_int_default * 1.0_spl_rk
     do i_element = 1, n_elements
       span = i_spans(i_element) 
       do j = 1, p+1
          arr_us(:, j, i_element) = mapping_tmp % control_points(:,span-1-p+j)
       end do
     end do
     ! ...

     ! ... free 
     call mapping_tmp % free()
     ! ...

  end subroutine spl_to_us_form_mapping_1d
  ! .........................................................

  ! .........................................................
  !> @brief     convert to pp_form
  !>
  !> @param[inout] self     the current mapping object 
  !> @param[inout] arr_pp   values 
  subroutine spl_to_pp_form_mapping_1d(self, arr_pp)
  implicit none
    class(spl_t_mapping_1d), intent(inout) :: self
    real(kind=spl_rk), dimension(:,:,:), intent(inout) :: arr_pp
    ! local
    integer :: i_dim

    ! ...
    do i_dim = 1, self % d_dim
      call to_pp_form_1d(self % control_points(i_dim, :), &
                       & self % knots_u, &
                       & self % n_u, &
                       & self % p_u, &
                       & self % n_elements_u, &
                       & arr_pp(i_dim, :, :))
    end do
    ! ...

  end subroutine spl_to_pp_form_mapping_1d
  ! .........................................................

  ! .........................................................
  !> @brief     inserts the knot t (number of insertion = times) 
  !>
  !> @param[inout]  self    the current mapping object 
  !> @param[in]     t       knot to insert 
  !> @param[in]     times   number of times t will be inserted [optional]  
  !> @param[inout]  other   the new mapping object [optional]   
  subroutine spl_insert_knot_mapping_1d(self, t, times, other)
  use spl_m_bsp, bsp_insert_knot => InsertKnot
  implicit none
     class(spl_t_mapping_1d)          , intent(inout) :: self
     real(spl_rk)                     , intent(in)    :: t 
     integer                , optional, intent(in)    :: times 
     class(spl_t_mapping_1d), optional, intent(inout) :: other 
     ! local
     integer :: i
     integer :: n_u
     integer :: n_total
     integer :: l_times
     real(spl_rk), dimension (:) , allocatable :: knots_new
     real(spl_rk), dimension (:) , allocatable :: knots_ini
     real(spl_rk), dimension(:,:), allocatable :: control_points_new
     real(spl_rk), dimension(:,:), allocatable :: control_points_ini
     type(spl_t_mapping_1d) :: mapping_tmp

     l_times = 1
     if (present(times)) then
       l_times = times
     end if

     !> \TODO must check if the knot is already in self and then compute its multiplicity and check p >= l_times + mult

     allocate(knots_ini(self % n_u + self % p_u + 1))
     allocate(control_points_ini(self % d_dim, self % n_u))

     knots_ini = self % knots_u
     control_points_ini = self % control_points

     allocate(knots_new(self % n_u + self % p_u + 1 + l_times))
     allocate(control_points_new(self % d_dim, self % n_u + l_times))

     call bsp_insert_knot(self % d_dim &
       & , self % n_u - 1, self % p_u &
       & , knots_ini, control_points_ini &
       & , t, l_times &
       & , knots_new, control_points_new)

     if (present(other)) then
       call other % create(self % p_u, knots_new, control_points_new)
     else
       call mapping_tmp % create(self % p_u, knots_new, control_points_new)
       call self % free()
       call mapping_tmp % duplicate(self)
       call mapping_tmp % free()
     end if

  end subroutine spl_insert_knot_mapping_1d 
  ! .........................................................
  
  ! .........................................................
  !> @brief     elevates the polynomial degree (number of elevation = times) 
  !>
  !> @param[inout] self  the current mapping object 
  !> @param[in]    times number of times the spline degree will be raised 
  !> @param[inout] other the new mapping object [optional]
  subroutine spl_elevate_mapping_1d(self, times, other)
  use spl_m_bsp, bsp_elevate_degree => DegreeElevate 
  implicit none
     class(spl_t_mapping_1d)          , intent(inout) :: self
     integer                          , intent(in)    :: times 
     class(spl_t_mapping_1d), optional, intent(inout) :: other 
     ! local
     integer :: i
     integer :: n_u
     integer :: n_total
     integer :: n_internal_knots
     integer :: nh
     real(spl_rk), dimension (:) , allocatable :: knots_new
     real(spl_rk), dimension (:) , allocatable :: knots_ini
     real(spl_rk), dimension(:,:), allocatable :: control_points_new
     real(spl_rk), dimension(:,:), allocatable :: control_points_ini
     type(spl_t_mapping_1d) :: mapping_tmp

     !> \TODO must check the internal knots 

     call spl_computes_internal_knots_mapping_1d(self, n_internal_knots) 
     nh = self % n_u + times * (n_internal_knots + 1)

     allocate(knots_ini(self % n_u + self % p_u + 1))
     allocate(control_points_ini(self % d_dim, self % n_u))

     knots_ini = self % knots_u
     control_points_ini = self % control_points

     allocate(knots_new(nh + self % p_u + times + 1))
     allocate(control_points_new(self % d_dim, nh))

     call bsp_elevate_degree(self % d_dim &
       & , self % n_u - 1, self % p_u &
       & , knots_ini, control_points_ini &
       & , times, nh - 1 &
       & , knots_new, control_points_new)

     if (present(other)) then
       call other % create(self % p_u + times, knots_new, control_points_new)
     else
       call mapping_tmp % create(self % p_u + times, knots_new, control_points_new)
       call self % free()
       call mapping_tmp % duplicate(self)
       call mapping_tmp % free()
     end if

  end subroutine spl_elevate_mapping_1d 
  ! .........................................................

  ! .........................................................
  !> @brief     refines a 1d mapping 
  !>
  !> @param[inout] self       the current mapping object 
  !> @param[in]    n_elements a new subdivision [optional]
  !> @param[in]    degree     number of times the spline degree will be raised [optional]  
  !> @param[inout] other      the new mapping object [optional]  
  !> @param[in]    verbose    print details about the refinement when True [optional]   
  !> \todo add knots and grid 
  subroutine spl_refine_mapping_1d(self, degree, n_elements, other, verbose)
  implicit none
     class(spl_t_mapping_1d), intent(inout) :: self
     integer                , optional, intent(in)    :: degree 
     integer                , optional, intent(in)    :: n_elements 
     class(spl_t_mapping_1d), optional, intent(inout) :: other 
     logical                , optional, intent(in)    :: verbose
     ! local
     integer :: i
     logical :: l_verbose
     type(spl_t_mapping_1d)          , target :: mapping_tmp
     real(spl_rk) :: t

     l_verbose = .False.
     if (present(verbose)) then
       l_verbose = verbose
     end if

     call mapping_tmp % create(self % p_u, self % knots_u, self % control_points)

     ! ...
     if (l_verbose) then
       print *, "==== initial mapping ===="
       call mapping_tmp % print_info()
       print *, "=============="
     end if
     ! ...

     ! ... add degree elevation 
     if (present(degree)) then
       if (degree > 0) then
         ! ...
         if (l_verbose) then
           print *, "==== before elevation ======"
           call mapping_tmp % print_info()
           print *, "=============="
         end if
         ! ...

         call mapping_tmp % elevate(degree)
       end if
     end if
     ! ...

     ! ... knots insertion
     if (present(n_elements)) then
       if (n_elements > 1) then
         ! ...
         if (l_verbose) then
           print *, "==== before insertion ======"
           call mapping_tmp % print_info()
           print *, "=============="
         end if
         ! ...

         do i = 1, n_elements - 1
           t = (i * 1.0_spl_rk / n_elements) 
           call mapping_tmp % insert_knot(t)
         end do

       end if
     end if
     ! ...

     ! ...
     if (l_verbose) then
       print *, "==== final mapping ===="
       call mapping_tmp % print_info()
       print *, "=============="
     end if
     ! ...

     if (present(other)) then
       call mapping_tmp % duplicate(other)
     else
       call self % free()
       call mapping_tmp % duplicate(self)
     end if

     call mapping_tmp % free() 

  end subroutine spl_refine_mapping_1d
  ! .........................................................

  ! .........................................................
  !> @brief     counts the number of internal knots 
  !>
  !> @param[in]    self               the current mapping object 
  !> @param[inout] n_internal_knots   number of internal knots 
  subroutine spl_computes_internal_knots_mapping_1d(self, n_internal_knots)
  implicit none
     class(spl_t_mapping_1d), intent(in) :: self
     integer, intent(inout) :: n_internal_knots
     ! local
     integer :: i
     integer :: i_current
     real(spl_rk), dimension(self % n_u + self % p_u + 1) :: lpr_grid
     real(spl_rk) :: min_current

     lpr_grid = spl_int_default * 1.0_spl_rk 

     i_current = 1
     lpr_grid(i_current) = minval(self % knots_u)
     do i=1, self % n_u + self % p_u
        min_current = minval(self % knots_u(i : self % n_u + self % p_u + 1))
        if ( min_current > lpr_grid(i_current) ) then
                i_current = i_current + 1
                lpr_grid(i_current) = min_current
        end if
     end do
     n_internal_knots = i_current - 2 

  end subroutine spl_computes_internal_knots_mapping_1d
  ! .........................................................

  ! .........................................................
  !> @brief     get the greville abscissae 
  !>
  !> @param[in]    self  the current mapping object 
  !> @param[inout] us    array containing the greville abscissae 
  subroutine spl_get_greville_mapping_1d(self, us)
  use spl_m_bsp, bsp_greville => Greville
  implicit none
    class(spl_t_mapping_1d),    intent(in)    :: self
    real(spl_rk), dimension(:), intent(inout) :: us
    ! local

    ! ...
    call bsp_greville( self % p_u, self % n_u + self % p_u, &
                     & self % knots_u(1:self % n_u + self % p_u + 1), us) 
    ! ...

  end subroutine spl_get_greville_mapping_1d
  ! .........................................................
   
  ! ...................................................
  !> @brief     Exports the B-Spline mapping to file 
  !>
  !> @param[inout] self       the current mapping object 
  !> @param[in]    filename   output filename 
  !> @param[in]    i_format   output format 
  subroutine spl_export_mapping_1d(self, filename, i_format)
  implicit none
    class(spl_t_mapping_1d), intent(inout) :: self
    character(len=*)       , intent(in)    :: filename
    integer      , optional, intent(in)    :: i_format
    ! local

    ! ...
    if (present(i_format)) then
       if (i_format == spl_mapping_format_nml) then
          call export_mapping_1d_nml(self, filename)
       else
          stop "spl_export_mapping_1d: format not yet supported"
       end if
    else
       call export_mapping_1d_nml(self, filename)
    end if
    ! ...

  end subroutine spl_export_mapping_1d 
  ! ...................................................
   
  ! ...................................................
  !> @brief     Exports the B-Spline mapping to a namelist file 
  !>
  !> @param[inout] self       the current mapping object 
  !> @param[in]    filename   output filename 
  subroutine export_mapping_1d_nml(self, filename)
  implicit none
    class(spl_t_mapping_1d), intent(inout) :: self
    character(len=*)       , intent(in)    :: filename
    ! local
    integer :: IO_stat
    integer, parameter :: input_file_id = 111
    integer :: ierr
    integer :: spline_deg1
    integer :: num_pts1
    integer :: d_dim
    character(len=256) :: label
    real(kind=spl_rk), dimension(:), allocatable :: knots1
    real(kind=spl_rk), dimension(:), allocatable :: control_pts1
    real(kind=spl_rk), dimension(:), allocatable :: control_pts2
    real(kind=spl_rk), dimension(:), allocatable :: control_pts3
    real(kind=spl_rk), dimension(:), allocatable :: control_weights
    real(kind=spl_rk) :: eta1_min_minimal
    real(kind=spl_rk) :: eta1_max_minimal
    integer  :: bc_left
    integer  :: bc_right
    integer  :: bc_bottom
    integer  :: bc_top
    integer  :: number_cells1
    integer :: sz_knots1
    integer :: i,j,i_current
  
    namelist /transf_label/  label
    namelist /d_dimension/  d_dim
    namelist /degree/   spline_deg1
    namelist /shape/    num_pts1 
    namelist /knots_1/   knots1
    namelist /control_points_1/ control_pts1
    namelist /control_points_2/ control_pts2
    namelist /control_points_3/ control_pts3
    namelist /weights/ control_weights

    open(unit=input_file_id, file=trim(filename))

    ! write the label
    label = trim("test_patch") 
    write( input_file_id, transf_label )

    ! write the manifold dimension
    d_dim = self % d_dim
    write( input_file_id, d_dimension )

    ! write the degree of spline
    spline_deg1 = self % p_u
    write( input_file_id, degree )

    ! write ....?
    num_pts1 = self % n_u
    write( input_file_id, shape )

    ! ...  Allocations of knots to construct the splines
    allocate(knots1(num_pts1 + self % p_u + 1))
    knots1 = self % knots_u
    ! write the knots associated to each direction 
    write( input_file_id, *) "&KNOTS_1"
    write( input_file_id, *) "KNOTS1= ", self % knots_u
    write( input_file_id, *) "/"
    
    ! allocations of tables containing control points in each direction 
    ! here its table 1D
    allocate(control_pts1(num_pts1))
    allocate(control_pts2(num_pts1))
    allocate(control_pts3(num_pts1))
    allocate(control_weights(num_pts1))

    i_current = 0
    do i = 1, num_pts1
      i_current = i_current + 1
      control_weights(i_current) = self % weights(i) 

      control_pts1(i_current) = self % control_points(1,i) 
      if (self % d_dim .ge. 2) then
        control_pts2(i_current) = self % control_points(2,i) 
        if (self % d_dim .ge. 3) then
          control_pts3(i_current) = self % control_points(3,i) 
        end if
      end if
    end do

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

  end subroutine export_mapping_1d_nml 
  ! ...................................................

  ! ...................................................
  !> @brief     create a new 1d mapping reading from a namelist file
  !> @param[inout] self       the current mapping object 
  !> @param[in]    filename   the name of the file to be read from
  subroutine spl_read_mapping_1d(self, filename)
    implicit none
    class(spl_t_mapping_1d), intent(inout) :: self
    character(len=*)       , intent(in)    :: filename
    ! LOCAL
    integer :: IO_stat
    integer :: input_file_id
    character(len=256) :: label
    integer :: spline_deg1
    integer :: num_pts1
    integer :: d_dim
    real(spl_rk), dimension(:), allocatable :: knots1
    real(spl_rk), dimension(:), allocatable :: weights
    real(spl_rk), dimension(:), allocatable :: control_pts1
    real(spl_rk), dimension(:), allocatable :: control_pts2
    real(spl_rk), dimension(:), allocatable :: control_pts3
    integer  :: number_cells1
    integer  :: number_cells2
    real(spl_rk), dimension(:,:), allocatable  :: control_points3d
        
    namelist /transf_label/  label
    namelist /d_dimension/   d_dim
    namelist /degree/   spline_deg1
    namelist /shape/    num_pts1
    namelist /knots_1/   knots1
    namelist /control_points_1/ control_pts1
    namelist /control_points_2/ control_pts2
    namelist /control_points_3/ control_pts3
    namelist /control_weights/  weights
    namelist /cartesian_mesh_1d/ number_cells1

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
    read( input_file_id, knots_1 )
 
    !>Allocation of table containing the weights associated
    !> to each control points
    allocate(weights(num_pts1))
    !> allocations of tables containing control points in each direction
    allocate(control_pts1(num_pts1))
    allocate(control_pts2(num_pts1))
    allocate(control_pts3(num_pts1))
    !>allocation of control_points
    allocate(control_points3d(d_dim,num_pts1))

    if (d_dim >= 1) then
      read( input_file_id, control_points_1)
      control_points3d(1,:) = control_pts1
    end if
    if (d_dim >= 2) then
      read( input_file_id, control_points_2)
      control_points3d(2,:) = control_pts2
    end if
    if (d_dim >= 3) then
      read( input_file_id, control_points_3)
      control_points3d(3,:) = control_pts3
    end if

    read( input_file_id, control_weights)

    call spl_create_mapping_1d(   self                  &
                                & , spline_deg1         &
                                & , knots1              &
                                & , control_points3d    &
                                & , weights=weights     &
                                &  )

    close(input_file_id)

  end subroutine spl_read_mapping_1d
  ! ...................................................

end module spl_m_mapping_1d 
