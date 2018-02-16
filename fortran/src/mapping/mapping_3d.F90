!> @brief 
!> Module for 3D mappings.
!> @details
!> A 3D mapping is defined as a B-Spline/NURBS volume 

module spl_m_mapping_3d 

use spl_m_global
use spl_m_bsp
use spl_m_pp_form,          only: to_pp_form_3d 
use spl_m_mapping_abstract, only: spl_t_mapping_abstract
use spl_m_mapping_1d,       only: spl_t_mapping_1d
use spl_m_mapping_2d,       only: spl_t_mapping_2d

implicit none

!  private

  ! .........................................................
  !> @brief 
  !> Class for 3D mappings.
  type, public, extends(spl_t_mapping_abstract) :: spl_t_mapping_3d
     integer :: n_u            !< number of control points in the u-direction
     integer :: p_u            !< spline degree in the u-direction
     integer :: n_v            !< number of control points in the v-direction
     integer :: p_v            !< spline degree in the v-direction
     integer :: n_w            !< number of control points in the w-direction
     integer :: p_w            !< spline degree in the w-direction
     integer :: n_elements_u   !< number of elements in the u-direction 
     integer :: n_elements_v   !< number of elements in the v-direction
     integer :: n_elements_w   !< number of elements in the w-direction

     real(spl_rk), dimension(:),       allocatable :: knots_u !< array of knots of size n_u+p_u+1 in the u-direction
     real(spl_rk), dimension(:),       allocatable :: knots_v !< array of knots of size n_v+p_v+1 in the v-direction
     real(spl_rk), dimension(:),       allocatable :: knots_w !< array of knots of size n_w+p_w+1 in the w-direction
     real(spl_rk), dimension(:,:,:,:), allocatable :: control_points !< array of control points 
     real(spl_rk), dimension(:,:,:),   allocatable :: weights !< array of weights

     real(spl_rk), dimension(:), allocatable :: grid_u !< corresponding grid in the u-direction
     real(spl_rk), dimension(:), allocatable :: grid_v !< corresponding grid in the v-direction
     real(spl_rk), dimension(:), allocatable :: grid_w !< corresponding grid in the w-direction

     integer, dimension(:), allocatable :: i_spans_u !< knots indices corresponding to the grid in the u-direction  
     integer, dimension(:), allocatable :: i_spans_v !< knots indices corresponding to the grid in the v-direction
     integer, dimension(:), allocatable :: i_spans_w !< knots indices corresponding to the grid in the w-direction
  contains
    procedure :: breaks             => spl_breaks_mapping_3d
    procedure :: clamp              => spl_clamp_mapping_3d
    procedure :: create             => spl_create_mapping_3d
    procedure :: duplicate          => spl_duplicate_mapping_3d
    procedure :: elevate            => spl_elevate_mapping_3d
    procedure :: free               => spl_free_mapping_3d
    procedure :: evaluate           => spl_evaluate_mapping_3d
    procedure :: evaluate_deriv     => spl_evaluate_deriv_mapping_3d
    procedure :: export             => spl_export_mapping_3d
    procedure :: extract            => spl_extract_mapping_3d
    procedure :: insert_knot        => spl_insert_knot_mapping_3d
    procedure :: print_info         => spl_print_mapping_3d
    procedure :: read_from_file     => spl_read_mapping_3d
    procedure :: refine             => spl_refine_mapping_3d
    procedure :: set_control_points => spl_set_mapping_control_points_3d
    procedure :: set_weights        => spl_set_mapping_weights_3d
    procedure :: get_greville       => spl_get_greville_mapping_3d
    procedure :: to_pp              => spl_to_pp_form_mapping_3d
    procedure :: to_us              => spl_to_us_form_mapping_3d
    procedure :: unclamp            => spl_unclamp_mapping_3d
  end type spl_t_mapping_3d
  ! .........................................................

contains

  ! .........................................................
  !> @brief      Creates a 3D mapping 
  !>
  !> @param[inout] self             the current object 
  !> @param[in]    p_u              polynomial degree 
  !> @param[in]    p_v              polynomial degree 
  !> @param[in]    p_w              polynomial degree 
  !> @param[in]    knots_u          the knot vector
  !> @param[in]    knots_v          the knot vector
  !> @param[in]    knots_w          the knot vector
  !> @param[in]    control_points   array containing the control points
  !> @param[in]    weights          array containing the control points weights [optional]
  subroutine spl_create_mapping_3d( self,           &
                                  & p_u,            &
                                  & p_v,            &
                                  & p_w,            &
                                  & knots_u,        &
                                  & knots_v,        &
                                  & knots_w,        &
                                  & control_points, &
                                  & weights)
  implicit none
     class(spl_t_mapping_3d)     , intent(inout) :: self
     integer                     , intent(in)    :: p_u 
     integer                     , intent(in)    :: p_v
     integer                     , intent(in)    :: p_w
     real(spl_rk), dimension (:) , intent(in)    :: knots_u 
     real(spl_rk), dimension (:) , intent(in)    :: knots_v
     real(spl_rk), dimension (:) , intent(in)    :: knots_w
     real(spl_rk), dimension(:,:,:,:), intent(in)    :: control_points
     real(spl_rk), optional, dimension(:,:,:), intent(in) :: weights 
     ! local
     integer :: knots_size_u 
     integer :: knots_size_v
     integer :: knots_size_w

     ! ... manifold dimension
     self % p_u   = p_u 
     self % p_v   = p_v
     self % p_w   = p_w
     self % p_dim = 3 
     self % d_dim = size(control_points, 1)
     self % n_u   = size(control_points, 2)
     self % n_v   = size(control_points, 3)
     self % n_w   = size(control_points, 4)
     ! ...

     ! ...
     knots_size_u = size(knots_u, 1)
     knots_size_v = size(knots_v, 1)
     knots_size_w = size(knots_w, 1)
     allocate(self % knots_u(knots_size_u))
     allocate(self % knots_v(knots_size_v))
     allocate(self % knots_w(knots_size_w))

     self % knots_u = knots_u
     self % knots_v = knots_v
     self % knots_w = knots_w
     ! ...

     ! ...
     allocate(self % control_points(self % d_dim, self % n_u, self % n_v, self % n_w))

     call self % set_control_points(control_points)
     ! ...

     ! ...
     allocate(self % weights(self % n_u, self % n_v, self % n_w))

     if (present(weights)) then
       call self % set_weights(weights)
     else
       self % weights = 1.0_spl_rk
     end if
     ! ...

     ! ...
     allocate(self % grid_u(self % n_u + self % p_u + 1))
     allocate(self % grid_v(self % n_v + self % p_v + 1))
     allocate(self % grid_w(self % n_w + self % p_w + 1))
     allocate(self % i_spans_u(self % n_u + self % p_u + 1))
     allocate(self % i_spans_v(self % n_v + self % p_v + 1))
     allocate(self % i_spans_w(self % n_w + self % p_w + 1))

     call self % breaks(self % n_elements_u, self % grid_u, i_spans=self % i_spans_u, axis=1)
     call self % breaks(self % n_elements_v, self % grid_v, i_spans=self % i_spans_v, axis=2)
     call self % breaks(self % n_elements_w, self % grid_w, i_spans=self % i_spans_w, axis=3)
     ! ...

  end subroutine spl_create_mapping_3d
  ! .........................................................

  ! .........................................................
  !> @brief      destroys a 3D mapping 
  !>
  !> @param[inout] self the current object 
  subroutine spl_free_mapping_3d(self)
  implicit none
     class(spl_t_mapping_3d)      , intent(inout) :: self
     ! local

     deallocate(self % knots_u)
     deallocate(self % knots_v)
     deallocate(self % knots_w)
     deallocate(self % control_points)
     deallocate(self % weights)
     deallocate(self % grid_u)
     deallocate(self % grid_v)
     deallocate(self % grid_w)
     deallocate(self % i_spans_u)
     deallocate(self % i_spans_v)
     deallocate(self % i_spans_w)

  end subroutine spl_free_mapping_3d 
  ! .........................................................

  ! .........................................................
  !> @brief      Duplicates a 3D mapping 
  !>
  !> @param[inout] self    the current object 
  !> @param[inout] other   the new object
  subroutine spl_duplicate_mapping_3d(self, other)
  implicit none
     class(spl_t_mapping_3d), intent(in)    :: self
     class(spl_t_mapping_3d), intent(inout) :: other 
     ! local

     ! ...
     call other % create( self % p_u, &
                        & self % p_v, &
                        & self % p_w, &
                        & self % knots_u, &
                        & self % knots_v, &
                        & self % knots_w, &
                        & self % control_points, &
                        & weights=self % weights)
     ! ...

  end subroutine spl_duplicate_mapping_3d
  ! .........................................................

  ! .........................................................
  !> @brief      sets control points to a 3D mapping 
  !>
  !> @param[inout] self             the current mapping object 
  !> @param[in]    control_points   the array of shape 4 of control points 
  subroutine spl_set_mapping_control_points_3d(self, control_points)
  implicit none
     class(spl_t_mapping_3d)     , intent(inout) :: self
     real(spl_rk), dimension (:,:,:,:), intent(in)    :: control_points
     ! local

     ! ... control points
     self % control_points (:,:,:,:) = control_points(:,:,:,:) 
     ! ...

  end subroutine spl_set_mapping_control_points_3d
  ! .........................................................

  ! .........................................................
  !> @brief      sets control points weights to a 3D mapping 
  !>
  !> @param[inout] self      the current mapping object 
  !> @param[in]    weights   the array of shape 3 of weights 
  subroutine spl_set_mapping_weights_3d(self, weights)
  implicit none
     class(spl_t_mapping_3d)     , intent(inout) :: self
     real(spl_rk), dimension (:,:,:), intent(in)    :: weights
     ! local

     ! ... control points
     self % weights (:,:,:) = weights(:,:,:) 
     ! ...

  end subroutine spl_set_mapping_weights_3d
  ! .........................................................

  ! .........................................................
  !> @brief     prints the current mapping 
  !>
  !> @param[inout] self the current mapping object 
  subroutine spl_print_mapping_3d(self)
  implicit none
    class(spl_t_mapping_3d), intent(in) :: self
    ! local
    integer :: i
    integer :: j
    integer :: k

    ! ...
    print *, ">>>> mapping_3d"
    print *, "* n_u                     : ", self % n_u
    print *, "* n_v                     : ", self % n_v
    print *, "* n_w                     : ", self % n_w
    print *, "* p_u                     : ", self % p_u
    print *, "* p_v                     : ", self % p_v
    print *, "* p_w                     : ", self % p_w
    print *, "* knots_u                 : ", self % knots_u
    print *, "* knots_v                 : ", self % knots_v
    print *, "* knots_w                 : ", self % knots_w
    print *, "* control points / weights: "
      do k = 1, self % n_w
        do j = 1, self % n_v
          do i = 1, self % n_u
            print *, i,j,self % control_points(:, i,j,k),self % weights(i,j,k)
          end do
        end do
      end do
    print *, "<<<< "
    ! ...

  end subroutine spl_print_mapping_3d 
  ! .........................................................

  ! .........................................................
  !> @brief     unclampes the current mapping 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    sides   enables the unclamping algo on sides [optional] 
  subroutine spl_unclamp_mapping_3d(self, sides)
  implicit none
     class(spl_t_mapping_3d), intent(inout) :: self
     logical, dimension(3,2), optional, intent(in) :: sides 
     ! local
     logical, dimension(3,2) :: l_sides 
     integer :: i
     integer :: j 
     integer :: k 
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
     do k = 1, self % n_w
       knots = self % knots_u

       call Unclamp(self % d_dim &
         & , self % n_u - 1, self % p_u &
         & , knots, self % control_points(:,:,j,k) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,:,j,k))
     end do
     end do

     self % knots_u = knots
     deallocate(knots)
     ! ...

     ! ...
     axis = 2
     allocate(knots(self % n_v + self % p_v + 1))

     do k = 1, self % n_w
     do i = 1, self % n_u
       knots = self % knots_v

       call Unclamp(self % d_dim &
         & , self % n_v - 1, self % p_v &
         & , knots, self % control_points(:,i,:,k) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,i,:,k))
     end do
     end do

     self % knots_v = knots
     deallocate(knots)
     ! ...

     ! ...
     axis = 3 
     allocate(knots(self % n_w + self % p_w + 1))

     do i = 1, self % n_u
     do j = 1, self % n_v
       knots = self % knots_w

       call Unclamp(self % d_dim &
         & , self % n_w - 1, self % p_w &
         & , knots, self % control_points(:,i,j,:) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,i,j,:))
     end do
     end do

     self % knots_w = knots
     deallocate(knots)
     ! ...

  end subroutine spl_unclamp_mapping_3d 
  ! .........................................................

  ! .........................................................
  !> @brief     clampes the current mapping 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    sides   enables the unclamping algo on sides [optional]
  subroutine spl_clamp_mapping_3d(self, sides)
  implicit none
     class(spl_t_mapping_3d), intent(inout) :: self
     logical, dimension(3,2), optional, intent(in) :: sides 
     ! local
     logical, dimension(3,2) :: l_sides 
     integer :: i
     integer :: j 
     integer :: k 
     real(kind=spl_rk), dimension(:), allocatable :: knots
     integer :: axis 

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
     do k = 1, self % n_w
       knots = self % knots_u

       call Clamp(self % d_dim &
         & , self % n_u - 1, self % p_u &
         & , knots, self % control_points(:,:,j,k) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,:,j,k))
     end do
     end do

     self % knots_u = knots
     deallocate(knots)
     ! ...

     ! ...
     axis = 2
     allocate(knots(self % n_v + self % p_v + 1))

     do k = 1, self % n_w
     do i = 1, self % n_u
       knots = self % knots_v

       call Clamp(self % d_dim &
         & , self % n_v - 1, self % p_v &
         & , knots, self % control_points(:,i,:,k) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,i,:,k))
     end do
     end do

     self % knots_v = knots
     deallocate(knots)
     ! ...

     ! ...
     axis = 3 
     allocate(knots(self % n_w + self % p_w + 1))

     do i = 1, self % n_u
     do j = 1, self % n_v
       knots = self % knots_w

       call Clamp(self % d_dim &
         & , self % n_w - 1, self % p_w &
         & , knots, self % control_points(:,i,j,:) &
         & , l_sides(axis,1), l_sides(axis,2) &
         & , knots, self % control_points(:,i,j,:))
     end do
     end do

     self % knots_w = knots
     deallocate(knots)
     ! ...

  end subroutine spl_clamp_mapping_3d 
  ! .........................................................

  ! .........................................................
  !> @brief     get the greville abscissae 
  !>
  !> @param[in]    self  the current mapping object 
  !> @param[inout] us    array containing the greville abscissae 
  !> @param[inout] axis  axis for which we compute the greville abscissae (possible values: 1, 2, 3) 
  subroutine spl_get_greville_mapping_3d(self, us, axis)
  use spl_m_bsp, bsp_greville => Greville
  implicit none
    class(spl_t_mapping_3d),    intent(in)    :: self
    real(spl_rk), dimension(:), intent(inout) :: us
    integer,                    intent(in)    :: axis
    ! local

    ! ...
    if (axis == 1) then
      call bsp_greville( self % p_u, self % n_u + self % p_u, &
                       & self % knots_u(1:self % n_u + self % p_u + 1), us) 
    elseif (axis == 2) then 
      call bsp_greville( self % p_v, self % n_v + self % p_v, &
                       & self % knots_v(1:self % n_v + self % p_v + 1), us) 
    elseif (axis == 3) then 
      call bsp_greville( self % p_w, self % n_w + self % p_w, &
                       & self % knots_w(1:self % n_w + self % p_w + 1), us) 
    else
      stop "spl_get_greville_mapping_3d: wrong value for axis. expect: 1,2 or 3"
    end if
    ! ...

  end subroutine spl_get_greville_mapping_3d
  ! .........................................................

  ! .........................................................
  !> @brief     inserts the knot t (number of insertion = times) 
  !>
  !> @param[inout] self   the current mapping object 
  !> @param[in]    t      knot to insert 
  !> @param[in]    axis   first or second direction 
  !> @param[in]    times  number of times t will be inserted [optional] 
  !> @param[inout] other  the new mapping object [optional]  
  subroutine spl_insert_knot_mapping_3d(self, t, axis, times, other)
  implicit none
     class(spl_t_mapping_3d)          , intent(inout) :: self
     real(spl_rk)                     , intent(in)    :: t 
     integer                          , intent(in)    :: axis
     integer                , optional, intent(in)    :: times 
     class(spl_t_mapping_3d), optional, intent(inout) :: other 
     ! local
     integer :: i
     integer :: d
     integer :: k 
     integer :: j 
     integer :: r 
     integer :: degree
     integer :: n_u
     integer :: n_v
     integer :: n_w
     integer :: d_dim_ini
     integer :: d_dim_new
     real(spl_rk), dimension(:,:), allocatable :: control_points_crv
     real(spl_rk), dimension(:,:,:,:), allocatable :: control_points_vol
     type(spl_t_mapping_3d) :: mapping_tmp
     type(spl_t_mapping_1d) :: mapping_crv
     type(spl_t_mapping_3d) :: mapping_vol

     call self % duplicate(mapping_tmp)
     d_dim_ini = mapping_tmp % d_dim

     ! ... u direction: step 1 
     if (axis == 1) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_v * mapping_tmp % n_w

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       n_w = mapping_tmp % n_w
       allocate(control_points_crv(d_dim_new, n_u))
       do i=1, n_u
         k = 0
         do j=1, n_v
           do r=1, n_w
             do d=1, d_dim_ini
               k = k + 1
               control_points_crv(k, i) = mapping_tmp % control_points(d, i, j, r)
             end do
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
       n_w = mapping_tmp % n_w
       allocate(control_points_vol(d_dim_ini, n_u, n_v, n_w))
       do i=1, n_u
         k = 0
         do j=1, n_v
           do r=1, n_w
             do d=1, d_dim_ini
               k = k + 1
               control_points_vol(d, i, j, r) = mapping_crv % control_points(k, i)
             end do
           end do
         end do
       end do

       call mapping_vol % create(mapping_crv % p_u, mapping_tmp % p_v, mapping_tmp % p_w &
         & , mapping_crv % knots_u, mapping_tmp % knots_v, mapping_tmp % knots_w &
         & , control_points_vol )

       call mapping_tmp % free()
       call mapping_vol % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_vol % free()
       deallocate(control_points_vol)
     end if
     ! ...

     ! ... v direction: step 1 
     if (axis == 2) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_u * mapping_tmp % n_w

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       n_w = mapping_tmp % n_w

       allocate(control_points_crv(d_dim_new, mapping_tmp % n_v))
       do j=1, n_v
         k = 0
         do r=1, n_w
           do i=1, n_u
             do d=1, d_dim_ini
               k = k + 1
               control_points_crv(k, j) = mapping_tmp % control_points(d, i, j, r)
             end do
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
       n_w = mapping_tmp % n_w
       allocate(control_points_vol(d_dim_ini, n_u, n_v, n_w))
       do j=1, n_v
         k = 0
         do r=1, n_w
           do i=1, n_u
             do d=1, d_dim_ini
               k = k + 1
               control_points_vol(d, i, j, r) = mapping_crv % control_points(k, j)
             end do
           end do
         end do
       end do

       call mapping_vol % create(mapping_tmp % p_u, mapping_crv % p_u, mapping_tmp % p_w, &
         & mapping_tmp % knots_u, mapping_crv % knots_u, mapping_tmp % knots_w, &
         & control_points_vol )

       call mapping_tmp % free()
       call mapping_vol % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_vol % free()
       deallocate(control_points_vol)
     end if
     ! ...

     ! ... w direction: step 1 
     if (axis == 3) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_u * mapping_tmp % n_v

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       n_w = mapping_tmp % n_w

       allocate(control_points_crv(d_dim_new, mapping_tmp % n_w))
       do r=1, n_w
         k = 0
         do i=1, n_u
           do j=1, n_v
             do d=1, d_dim_ini
               k = k + 1
               control_points_crv(k, r) = mapping_tmp % control_points(d, i, j, r)
             end do
           end do
         end do
       end do

       call mapping_crv % create(mapping_tmp % p_w, mapping_tmp % knots_w, control_points_crv)

       call mapping_crv % insert_knot(t, times=times) 
       deallocate(control_points_crv)
       ! ...

       ! ... w direction: step 2 
       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       n_w = mapping_crv % n_u
       allocate(control_points_vol(d_dim_ini, n_u, n_v, n_w))
       do r=1, n_w
         k = 0
         do i=1, n_u
           do j=1, n_v
             do d=1, d_dim_ini
               k = k + 1
               control_points_vol(d, i, j, r) = mapping_crv % control_points(k, r)
             end do
           end do
         end do
       end do

       call mapping_vol % create(mapping_tmp % p_u, mapping_tmp % p_w, mapping_crv % p_u, &
         & mapping_tmp % knots_u, mapping_tmp % knots_v, mapping_crv % knots_u, &
         & control_points_vol )

       call mapping_tmp % free()
       call mapping_vol % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_vol % free()
       deallocate(control_points_vol)
     end if
     ! ...

     if (present(other)) then
       call mapping_tmp % duplicate(other)
     else
       call self % free()
       call mapping_tmp % duplicate(self)
     end if
     call mapping_tmp % free()

  end subroutine spl_insert_knot_mapping_3d 
  ! .........................................................

  ! .........................................................
  !> @brief     evaluates the current mapping 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    arr_u   evaluation u-sites. array of dim(1:n_points_u) 
  !> @param[in]    arr_v   evaluation v-sites. array of dim(1:n_points_v) 
  !> @param[in]    arr_w   evaluation w-sites. array of dim(1:n_points_w) 
  !> @param[out]   arr_y   values.             array of dim(1:d_dim,1:n_points_u,1:n_points_v,1:n_points_w)  
  subroutine spl_evaluate_mapping_3d(self, arr_u, arr_v, arr_w, arr_y)
  implicit none
     class(spl_t_mapping_3d), intent(inout) :: self
     real(kind=spl_rk), dimension(:), intent(in) :: arr_u
     real(kind=spl_rk), dimension(:), intent(in) :: arr_v
     real(kind=spl_rk), dimension(:), intent(in) :: arr_w
     real(kind=spl_rk), dimension(:,:,:,:), intent(out) :: arr_y
     ! local
     integer :: ru
     integer :: rv
     integer :: rw

     ! ...
     ru = size(arr_u, 1)
     rv = size(arr_v, 1)
     rw = size(arr_w, 1)

     call Evaluate3(self % d_dim &
       & , self % n_u - 1, self % p_u &
       & , self % knots_u &
       & , self % n_v - 1, self % p_v &
       & , self % knots_v &
       & , self % n_w - 1, self % p_w &
       & , self % knots_w &
       & , self % control_points &
       & , self % weights &
       & , ru-1, arr_u, rv-1, arr_v, rw-1, arr_w, arr_y)
     ! ...

  end subroutine spl_evaluate_mapping_3d 
  ! .........................................................

  ! .........................................................
  !> @brief     evaluates derivatives of the current mapping 
  !>
  !> @param[inout] self     the current mapping object 
  !> @param[in]    arr_u    evaluation u-sites. array of dim(1:n_points_u) 
  !> @param[in]    arr_v    evaluation v-sites. array of dim(1:n_points_v) 
  !> @param[in]    arr_w    evaluation w-sites. array of dim(1:n_points_w) 
  !> @param[out]   arr_y    values.             array of dim(1:d_dim,1:n_points_u,1:n_points_v,1:n_points_w)  
  !> @param[out]   arr_dy   1st derivatives.    array of dim(1:n_deriv,1:d_dim,1:n_points_u,1:n_points_v,1:n_points_w). [n_deriv = 3] 
  !> @param[out]   arr_d2y  2nd derivatives.    array of dim(1:n_deriv,1:d_dim,1:n_points_u,1:n_points_v,1:n_points_w).  [n_deriv = 6] 
  subroutine spl_evaluate_deriv_mapping_3d(self, arr_u, arr_v, arr_w, arr_y, arr_dy, arr_d2y)
  implicit none
    class(spl_t_mapping_3d), intent(inout) :: self
    real(kind=spl_rk), dimension(:), intent(in) :: arr_u
    real(kind=spl_rk), dimension(:), intent(in) :: arr_v
    real(kind=spl_rk), dimension(:), intent(in) :: arr_w
    real(kind=spl_rk), dimension(:,:,:,:), intent(out) :: arr_y
    real(kind=spl_rk), optional, dimension(:,:,:,:,:), intent(out) :: arr_dy
    real(kind=spl_rk), optional, dimension(:,:,:,:,:), intent(out) :: arr_d2y
    ! local
    integer :: ru
    integer :: rv
    integer :: rw
    integer :: n_deriv 
    integer :: n_total_deriv 
    real(kind=spl_rk), dimension(:,:,:,:,:), allocatable :: Cw

    ! ...
    n_deriv = 0

    if (present(arr_dy)) then
      n_deriv = n_deriv + 1
    end if

    if (present(arr_d2y)) then
      n_deriv = n_deriv + 1
    end if
    ! ...

    ! ... in 3d
    n_total_deriv = 1

    if (n_deriv==1) then
      n_total_deriv = 3 + 1 
    end if
    if (n_deriv==2) then
      n_total_deriv = 3 + 6 + 1 
    end if
    ! ...

    ! ...
    ru = size(arr_u, 1)
    rv = size(arr_v, 1)
    rw = size(arr_w, 1)
    allocate(Cw(n_total_deriv, self % d_dim, ru, rv, rw))
    Cw = 0.0_spl_rk

    call EvaluateDeriv3(n_deriv, n_total_deriv-1 &
      & , self % d_dim &
      & , self % n_u - 1, self % p_u &
      & , self % knots_u &
      & , self % n_v - 1, self % p_v &
      & , self % knots_v &
      & , self % n_w - 1, self % p_w &
      & , self % knots_w &
      & , self % control_points &
      & , self % weights &
      & , ru-1, arr_u, rv-1, arr_v, rw-1, arr_w, Cw)
    ! ...

    ! ...
    arr_y(:,:,:,:) = Cw(1,:,:,:,:)

    if (present(arr_dy)) then
      arr_dy(1:3,:,:,:,:) = Cw(2:4,:,:,:,:)
    end if

    if (present(arr_d2y)) then
      arr_d2y(1:6,:,:,:,:) = Cw(5:10,:,:,:,:)
    end if
    ! ...

  end subroutine spl_evaluate_deriv_mapping_3d 
  ! .........................................................

  ! .........................................................
  !> @brief     computes the breaks of the knot vector 
  !>
  !> @param[inout] self         the current mapping object 
  !> @param[inout] n_elements   number of non-zero elements 
  !> @param[inout] grid         the corresponding grid, maximum size = size(knots)
  !> @param[inout] i_spans      the span for every element [optional]  
  !> @param[in]    axis         knot vector axis [optional] 
  subroutine spl_breaks_mapping_3d(self, n_elements, grid, i_spans, axis)
  implicit none
     class(spl_t_mapping_3d), intent(inout) :: self
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
     elseif (axis == 3) then
       n = self % n_w ; p = self % p_w
       allocate(knots(n+p+1))
       knots = self % knots_w
     else
       stop "spl_breaks_mapping_3d: wrong arguments"
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
  end subroutine spl_breaks_mapping_3d
  ! .........................................................

  ! .........................................................
  !> @brief     convert to uniform bspline form 
  !>
  !> @param[inout] self     the current mapping object 
  !> @param[inout] arr_us   values 
  subroutine spl_to_us_form_mapping_3d(self, arr_us)
  implicit none
     class(spl_t_mapping_3d), intent(inout) :: self
     real(kind=spl_rk), dimension(:,:,:,:,:), intent(inout) :: arr_us
     ! local
     type(spl_t_mapping_3d) :: mapping_tmp
     real(spl_rk), dimension(self % n_u + self % p_u + 1) :: grid_u
     real(spl_rk), dimension(self % n_v + self % p_v + 1) :: grid_v
     real(spl_rk), dimension(self % n_w + self % p_w + 1) :: grid_w
     integer, dimension(self % n_u + self % p_u + 1) :: i_spans_u
     integer, dimension(self % n_v + self % p_v + 1) :: i_spans_v
     integer, dimension(self % n_w + self % p_w + 1) :: i_spans_w
     integer :: n_elements_u 
     integer :: n_elements_v
     integer :: n_elements_w
     integer :: i_u 
     integer :: i_v 
     integer :: i_w 
     integer :: j_u
     integer :: j_v 
     integer :: j_w 
     integer :: i_element
     integer :: i_element_u 
     integer :: i_element_v 
     integer :: i_element_w 
     integer :: span_u 
     integer :: span_v
     integer :: span_w

     ! ... create a copy of self
     call self % duplicate(mapping_tmp)
     ! ...

     ! ... first we unclamp the spline
     !     and then we get the control points
     call mapping_tmp % unclamp()

     call self % breaks(n_elements_u, grid_u, i_spans=i_spans_u, axis=1)
     call self % breaks(n_elements_v, grid_v, i_spans=i_spans_v, axis=2)
     call self % breaks(n_elements_w, grid_w, i_spans=i_spans_w, axis=3)

     arr_us = spl_int_default * 1.0_spl_rk
     i_element = 0
     do i_element_u = 1, n_elements_u
       span_u = i_spans_u(i_element_u) 
       do i_element_v = 1, n_elements_v
         span_v = i_spans_v(i_element_v) 
         do i_element_w = 1, n_elements_w
           span_w = i_spans_w(i_element_w) 

           i_element = i_element + 1
           do j_u = 1, self % p_u + 1
             i_u = span_u - 1 - self % p_u + j_u 
             do j_v = 1, self % p_v + 1
               i_v = span_v - 1 - self % p_v + j_v 
               do j_w = 1, self % p_w + 1
                 i_w = span_w - 1 - self % p_w + j_w 
                 arr_us(:, j_u, j_v, j_w, i_element) = self % control_points(:, i_u, i_v, i_w)
               end do
             end do
           end do
         end do
       end do
     end do
     ! ...

     ! ... free 
     call mapping_tmp % free()
     ! ...

  end subroutine spl_to_us_form_mapping_3d
  ! .........................................................

  ! .........................................................
  !> @brief     convert to pp_form
  !>
  !> @param[inout] self     the current mapping object 
  !> @param[inout] arr_pp   values 
  subroutine spl_to_pp_form_mapping_3d(self, arr_pp)
  implicit none
    class(spl_t_mapping_3d), intent(inout) :: self
    real(kind=spl_rk), dimension(:,:,:,:,:,:,:), intent(inout) :: arr_pp
    ! local
    integer :: i_dim
    integer :: i_element_w
    integer :: i_element_v
    integer :: i_element_u
    integer :: i_element
    real(kind=spl_rk), dimension(:,:,:,:,:,:), allocatable :: pp_coeff 

    allocate(pp_coeff(self % p_u + 1, self % p_v + 1, self % p_w + 1, &
      & self % n_elements_u, self % n_elements_v, self % n_elements_w))

    do i_dim = 1, self % d_dim
      call to_pp_form_3d(self % control_points(i_dim, :, :, :), &
                       & self % knots_u, self % knots_v, self % knots_w, &
                       & self % n_u, self % n_v, self % n_w, &
                       & self % p_u, self % p_v, self % p_w, &
                       & self % n_elements_u, self % n_elements_v, self % n_elements_w, &
                       & pp_coeff)

 

      arr_pp(i_dim, :, :, :, :,  :, :) = pp_coeff(:, :, :, :, :, :)
            
    end do

  end subroutine spl_to_pp_form_mapping_3d
  ! .........................................................
  
  ! .........................................................
  !> @brief     elevates the polynomial degree (number of elevation = times) 
  !>
  !> @param[inout] self    the current mapping object 
  !> @param[in]    times   number of times the spline degree will be raised 
  !> @param[inout] other   the new mapping object [optional]  
  subroutine spl_elevate_mapping_3d(self, times, other)
  implicit none
     class(spl_t_mapping_3d)          , intent(inout) :: self
     integer, dimension(3)            , intent(in)    :: times 
     class(spl_t_mapping_3d), optional, intent(inout) :: other 
     ! local
     integer :: i
     integer :: d
     integer :: r 
     integer :: k 
     integer :: j 
     integer :: degree
     integer :: n_u
     integer :: n_v
     integer :: n_w
     integer :: d_dim_ini
     integer :: d_dim_new
     real(spl_rk), dimension(:,:), allocatable :: control_points_crv
     real(spl_rk), dimension(:,:,:,:), allocatable :: control_points_vol
     type(spl_t_mapping_3d) :: mapping_tmp
     type(spl_t_mapping_1d) :: mapping_crv
     type(spl_t_mapping_3d) :: mapping_vol

     call self % duplicate(mapping_tmp)
     d_dim_ini = mapping_tmp % d_dim

     ! ... u direction: step 1 
     degree = times(1)
     if (degree > 0) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_v * mapping_tmp % n_w

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       n_w = mapping_tmp % n_w
       allocate(control_points_crv(d_dim_new, n_u))
       do i=1, n_u
         k = 0
         do j=1, n_v
           do r=1, n_w
             do d=1, d_dim_ini
               k = k + 1
               control_points_crv(k, i) = mapping_tmp % control_points(d, i, j, r)
             end do
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
       n_w = mapping_tmp % n_w
       allocate(control_points_vol(d_dim_ini, n_u, n_v, n_w))
       do i=1, n_u
         k = 0
         do j=1, n_v
           do r=1, n_w
             do d=1, d_dim_ini
               k = k + 1
               control_points_vol(d, i, j, r) = mapping_crv % control_points(k, i)
             end do
           end do
         end do
       end do

       call mapping_vol % create(mapping_crv % p_u, mapping_tmp % p_v, mapping_tmp % p_w &
         & , mapping_crv % knots_u, mapping_tmp % knots_v, mapping_tmp % knots_w, control_points_vol )

       call mapping_tmp % free()
       call mapping_vol % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_vol % free()
       deallocate(control_points_vol)
     end if
     ! ...

     ! ... v direction: step 1 
     degree = times(2)
     if (degree > 0) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_u * mapping_tmp % n_w

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       n_w = mapping_tmp % n_w

       allocate(control_points_crv(d_dim_new, mapping_tmp % n_v))
       do j=1, n_v
         k = 0
         do r=1, n_w
           do i=1, n_u
             do d=1, d_dim_ini
               k = k + 1
               control_points_crv(k, j) = mapping_tmp % control_points(d, i, j, r)
             end do
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
       n_w = mapping_tmp % n_w
       allocate(control_points_vol(d_dim_ini, n_u, n_v, n_w))
       do j=1, n_v
         k = 0
         do r=1, n_w
           do i=1, n_u
             do d=1, d_dim_ini
               k = k + 1
               control_points_vol(d, i, j, r) = mapping_crv % control_points(k, j)
             end do
           end do
         end do
       end do

       call mapping_vol % create(mapping_tmp % p_u, mapping_crv % p_u, mapping_tmp % p_w, &
         & mapping_tmp % knots_u, mapping_crv % knots_u, mapping_tmp % knots_w, &
         & control_points_vol )

       call mapping_tmp % free()
       call mapping_vol % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_vol % free()
       deallocate(control_points_vol)
     end if
     ! ...

     ! ... w direction: step 1 
     degree = times(3)
     if (degree > 0) then
       d_dim_new = mapping_tmp % d_dim * mapping_tmp % n_u * mapping_tmp % n_v

       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       n_w = mapping_tmp % n_w

       allocate(control_points_crv(d_dim_new, mapping_tmp % n_w))
       do r=1, n_w
         k = 0
         do i=1, n_u
           do j=1, n_v
             do d=1, d_dim_ini
               k = k + 1
               control_points_crv(k, r) = mapping_tmp % control_points(d, i, j, r)
             end do
           end do
         end do
       end do

       call mapping_crv % create(mapping_tmp % p_w, mapping_tmp % knots_w, control_points_crv)

       call mapping_crv % elevate(degree)
       deallocate(control_points_crv)
       ! ...

       ! ... w direction: step 2 
       n_u = mapping_tmp % n_u
       n_v = mapping_tmp % n_v
       n_w = mapping_crv % n_u
       allocate(control_points_vol(d_dim_ini, n_u, n_v, n_w))
       do r=1, n_w
         k = 0
         do i=1, n_u
           do j=1, n_v
             do d=1, d_dim_ini
               k = k + 1
               control_points_vol(d, i, j, r) = mapping_crv % control_points(k, r)
             end do
           end do
         end do
       end do

       call mapping_vol % create(mapping_tmp % p_u, mapping_tmp % p_v, mapping_crv % p_u, &
         & mapping_tmp % knots_u, mapping_tmp % knots_v, mapping_crv % knots_u, &
         & control_points_vol )

       call mapping_tmp % free()
       call mapping_vol % duplicate(mapping_tmp)
       call mapping_crv % free()
       call mapping_vol % free()
       deallocate(control_points_vol)
     end if
     ! ...

     if (present(other)) then
       call mapping_tmp % duplicate(other)
     else
       call self % free()
       call mapping_tmp % duplicate(self)
     end if
     call mapping_tmp % free()

  end subroutine spl_elevate_mapping_3d 
  ! .........................................................

  ! .........................................................
  !> @brief     refines a 3d mapping 
  !>
  !> @param[inout] self         the current mapping object 
  !> @param[in]    n_elements   a new subdivision [optional]  
  !> @param[in]    degrees      number of times the spline degree will be raised [optional]   
  !> @param[inout] other        the new mapping object [optional]   
  !> @param[in]    verbose      print details about the refinement when True [optional]   
  !> \TODO add knots and grid 
  subroutine spl_refine_mapping_3d(self, degrees, n_elements, other, verbose)
  implicit none
     class(spl_t_mapping_3d), intent(inout) :: self
     integer, dimension(3)  , optional, intent(in)    :: degrees 
     integer, dimension(3)  , optional, intent(in)    :: n_elements 
     class(spl_t_mapping_3d), optional, intent(inout) :: other 
     logical                , optional, intent(in)    :: verbose
     ! local
     integer :: i
     integer :: axis
     logical :: l_verbose
     type(spl_t_mapping_3d)          , target :: mapping_tmp
     real(spl_rk) :: t

     l_verbose = .False.
     if (present(verbose)) then
       l_verbose = verbose
     end if

     call mapping_tmp % create(self % p_u, self % p_v, self % p_w, &
       & self % knots_u, self % knots_v, self % knots_w, &
       & self % control_points)

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

         ! ...
         axis = 3 
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

  end subroutine spl_refine_mapping_3d
  ! .........................................................

  ! ...................................................
  !> @brief     Extracts a B-Spline surface from a B-Spline volume
  !>
  !> @param[inout] self       the current mapping object 
  !> @param[inout] other      a 2d mapping 
  !> @param[in]    axis       direction axis 
  !> @param[in]    face       face id for the given axis 
  subroutine spl_extract_mapping_3d(self, other, axis, face)
  implicit none
    class(spl_t_mapping_3d), intent(inout) :: self
    class(spl_t_mapping_2d), intent(inout) :: other 
    integer                , intent(in)    :: axis 
    integer                , intent(in)    :: face
    ! local
    integer :: p_1
    integer :: p_2
    integer :: n_1
    integer :: n_2
    integer :: d_dim 
    real(spl_rk), dimension (:) , allocatable :: knots_1 
    real(spl_rk), dimension (:) , allocatable :: knots_2
    real(spl_rk), dimension(:,:,:), allocatable :: control_points
    real(spl_rk), dimension(:,:), allocatable :: weights
    
    ! ...
    if ((axis > self % p_dim) .or. (axis <= 0)) then
      print *, "spl_extract_mapping_3d: wrong value for axis. Given ", axis
      stop
    end if
    ! ...

    ! ...
    if ((face > 2) .or. (face <= 0)) then
      print *, "spl_extract_mapping_3d: wrong value for face. Given ", face
      stop
    end if
    ! ...
    
    ! ...
    if (axis == 1) then
      ! ...
      p_1   = self % p_v
      p_2   = self % p_w
      n_1   = self % n_v
      n_2   = self % n_w
      d_dim = self % d_dim

      allocate(knots_1(n_1+p_1+1))
      allocate(knots_2(n_2+p_2+1))
      allocate(control_points(d_dim, n_1, n_2))
      allocate(weights(n_1, n_2))
      knots_1 = self % knots_v
      knots_2 = self % knots_w
      ! ...

      ! ...
      if (face == 1) then
        control_points = self % control_points(:,1,:,:)
        weights        = self % weights(1,:,:)
      elseif (face == 2) then                 
        control_points = self % control_points(:,self % n_u,:,:)
        weights        = self % weights(self % n_u,:,:)
      end if
      ! ...
    elseif (axis == 2) then
      ! ...
      p_1   = self % p_u
      p_2   = self % p_w
      n_1   = self % n_u
      n_2   = self % n_w
      d_dim = self % d_dim

      allocate(knots_1(n_1+p_1+1))
      allocate(knots_2(n_2+p_2+1))
      allocate(control_points(d_dim, n_1, n_2))
      allocate(weights(n_1, n_2))
      knots_1 = self % knots_u
      knots_2 = self % knots_w
      ! ...

      ! ...
      if (face == 1) then
        control_points = self % control_points(:,:,1,:)
        weights        = self % weights(:,1,:)
      elseif (face == 2) then                 
        control_points = self % control_points(:,:,self % n_v,:)
        weights        = self % weights(:,self % n_v,:)
      end if
      ! ...
    elseif (axis == 3) then
      ! ...
      p_1   = self % p_u
      p_2   = self % p_v
      n_1   = self % n_u
      n_2   = self % n_v
      d_dim = self % d_dim

      allocate(knots_1(n_1+p_1+1))
      allocate(knots_2(n_2+p_2+1))
      allocate(control_points(d_dim, n_1, n_2))
      allocate(weights(n_1, n_2))
      knots_1 = self % knots_u
      knots_2 = self % knots_v
      ! ...

      ! ...
      if (face == 1) then
        control_points = self % control_points(:,:,:,1)
        weights        = self % weights(:,:,1)
      elseif (face == 2) then                 
        control_points = self % control_points(:,:,:,self % n_w)
        weights        = self % weights(:,:,self % n_w)
      end if
      ! ...
    end if
    ! ...

    ! ...
    call other % create(p_1, p_2, knots_1, knots_2, control_points, weights=weights)
    ! ...

  end subroutine spl_extract_mapping_3d
  ! ...................................................

  ! ...................................................
  !> @brief     Exports the B-Spline mapping to file 
  !>
  !> @param[inout] self       the current mapping object 
  !> @param[in]    filename   output filename 
  !> @param[in]    i_format   output format 
  subroutine spl_export_mapping_3d(self, filename, i_format)
  implicit none
    class(spl_t_mapping_3d), intent(inout) :: self
    character(len=*)       , intent(in)    :: filename
    integer      , optional, intent(in)    :: i_format
    ! local

    ! ...
    if (present(i_format)) then
       if (i_format == spl_mapping_format_nml) then
          call export_mapping_3d_nml(self, filename)
       else
          stop "spl_export_mapping_3d: format not yet supported"
       end if
    else
       call export_mapping_3d_nml(self, filename)
    end if
    ! ...

  end subroutine spl_export_mapping_3d 
  ! ...................................................
   
  ! ...................................................
  !> @brief     Exports the B-Spline mapping to a namelist file 
  !>
  !> @param[inout] self       the current mapping object 
  !> @param[in]    filename   output filename 
  subroutine export_mapping_3d_nml(self, filename)
  implicit none
    class(spl_t_mapping_3d), intent(inout) :: self
    character(len=*)       , intent(in)    :: filename
    ! local
    integer :: IO_stat
    integer, parameter :: input_file_id = 111
    integer :: ierr
    integer :: spline_deg1
    integer :: spline_deg2
    integer :: spline_deg3
    integer :: num_pts1
    integer :: num_pts2
    integer :: num_pts3
    character(len=256) :: label
    real(kind=spl_rk), dimension(:), allocatable :: knots1
    real(kind=spl_rk), dimension(:), allocatable :: knots2
    real(kind=spl_rk), dimension(:), allocatable :: knots3
    real(kind=spl_rk), dimension(:), allocatable :: control_pts1
    real(kind=spl_rk), dimension(:), allocatable :: control_pts2
    real(kind=spl_rk), dimension(:), allocatable :: control_pts3
    real(kind=spl_rk), dimension(:), allocatable :: control_weights
    real(kind=spl_rk) :: eta1_min_minimal
    real(kind=spl_rk) :: eta1_max_minimal
    real(kind=spl_rk) :: eta2_min_minimal
    real(kind=spl_rk) :: eta2_max_minimal
    real(kind=spl_rk) :: eta3_min_minimal
    real(kind=spl_rk) :: eta3_max_minimal
    integer  :: bc_left
    integer  :: bc_right
    integer  :: bc_bottom
    integer  :: bc_top
    integer  :: number_cells1,number_cells2,number_cells3
    integer :: sz_knots1,sz_knots2
    integer :: i,j,k,i_current,d_dim
  
    namelist /transf_label/  label
    namelist /d_dimension/  d_dim
    namelist /degree/   spline_deg1, spline_deg2, spline_deg3
    namelist /shape/    num_pts1, num_pts2, num_pts3
    namelist /knots_1/   knots1
    namelist /knots_2/   knots2
    namelist /knots_3/   knots3
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
    spline_deg3 = self % p_w
    write( input_file_id, degree )

    ! write ....?
    num_pts1 = self % n_u
    num_pts2 = self % n_v
    num_pts3 = self % n_w
    write( input_file_id, shape )

!    ! write if we use NURBS or not
!    ! Allocations of knots to construct the splines

    allocate(knots1(num_pts1 + self % p_u + 1))
    allocate(knots2(num_pts2 + self % p_v + 1))
    allocate(knots3(num_pts3 + self % p_w + 1))
    knots1 = self % knots_u
    knots2 = self % knots_v
    knots3 = self % knots_w
    ! write the knots associated to each direction 
    ! we don't use the namelist here: problem with repeated factors in the namelist
    write( input_file_id, *) "&KNOTS_1"
    write( input_file_id, *) "KNOTS1= ", self % knots_u
    write( input_file_id, *) "/"
    write( input_file_id, *) "&KNOTS_2"
    write( input_file_id, *) "KNOTS2= ", self % knots_v
    write( input_file_id, *) "/"
    write( input_file_id, *) "&KNOTS_3"
    write( input_file_id, *) "KNOTS3= ", self % knots_w
    write( input_file_id, *) "/"
    
    ! allocations of tables containing control points in each direction 
    ! here its table 1D
    allocate(control_pts1(num_pts1*num_pts2*num_pts3))
    allocate(control_pts2(num_pts1*num_pts2*num_pts3))
    allocate(control_pts3(num_pts1*num_pts2*num_pts3))
    allocate(control_weights(num_pts1*num_pts2*num_pts3))

    i_current = 0
    do k = 1, num_pts3
    do j = 1, num_pts2
    do i = 1, num_pts1
      i_current = i_current + 1

      control_weights(i_current) = self % weights(i,j,k) 

      control_pts1(i_current) = self % control_points(1,i,j,k) 
      if (self % d_dim .ge. 2) then
        control_pts2(i_current) = self % control_points(2,i,j,k) 
      end if
      if (self % d_dim .ge. 3) then
        control_pts3(i_current) = self % control_points(3,i,j,k) 
      end if
    end do
    end do
    end do

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

  end subroutine export_mapping_3d_nml 
  ! ...................................................

  ! ...................................................
  !> @brief     create a new 3d mapping reading from a namelist file
  !> @param[inout] self       the current mapping object 
  !> @param[in]    filename   the name of the file to be read from
  subroutine spl_read_mapping_3d(self, filename)
    implicit none
    class(spl_t_mapping_3d), intent(inout) :: self
    character(len=*)           , intent(in) :: filename
    ! LOCAL
    integer :: IO_stat
    integer :: input_file_id
    character(len=256) :: label
    integer :: spline_deg1
    integer :: spline_deg2
    integer :: spline_deg3
    integer :: num_pts1
    integer :: num_pts2
    integer :: num_pts3
    integer :: i,j,k, i_current
    real(spl_rk), dimension(:), allocatable :: knots1
    real(spl_rk), dimension(:), allocatable :: knots2
    real(spl_rk), dimension(:), allocatable :: knots3
    real(spl_rk), dimension(:), allocatable :: weights
    real(spl_rk), dimension(:,:,:), allocatable :: weights_3d
    real(spl_rk), dimension(:), allocatable :: control_pts1
    real(spl_rk), dimension(:), allocatable :: control_pts2
    real(spl_rk), dimension(:), allocatable :: control_pts3
    integer  :: number_cells1
    integer  :: number_cells2
    integer  :: number_cells3
    integer :: d_dim
    real(spl_rk), dimension(:,:,:,:), allocatable  :: control_points3d
        
    namelist /transf_label/  label
    namelist /d_dimension/   d_dim
    namelist /degree/   spline_deg1,spline_deg2,spline_deg3
    namelist /shape/    num_pts1, num_pts2, num_pts3
    namelist /knots_1/   knots1
    namelist /knots_2/   knots2
    namelist /knots_3/   knots3
    namelist /control_points_1/ control_pts1
    namelist /control_points_2/ control_pts2
    namelist /control_points_3/ control_pts3
    namelist /control_weights/  weights
    namelist /cartesian_mesh_3d/ number_cells1,number_cells2,number_cells3
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
    allocate(knots3 (num_pts3+spline_deg3+1))
    read( input_file_id, knots_1 )
    read( input_file_id, knots_2 )
    read( input_file_id, knots_3 )
 
    !>TODO 
    !>Allocation of table containing the weights associated
    !> to each control points
    allocate(weights(num_pts1*num_pts2*num_pts3))
    allocate(weights_3d(num_pts1, num_pts2, num_pts3))
    !> allocations of tables containing control points in each direction
    allocate(control_pts1(num_pts1*num_pts2*num_pts3))
    allocate(control_pts2(num_pts1*num_pts2*num_pts3))
    allocate(control_pts3(num_pts1*num_pts2*num_pts3))
    !>allocation of control_points
    allocate(control_points3d(d_dim,num_pts1,num_pts2,num_pts3))

    if (d_dim >= 1) then
      read( input_file_id, control_points_1)

      i_current = 0
      do k = 1, num_pts3
      do j = 1, num_pts2
      do i = 1, num_pts1
        i_current = i_current + 1
        control_points3d(1,i,j,k) = control_pts1(i_current) 
      end do
      end do
      end do
    end if
    if (d_dim >= 2) then
      read( input_file_id, control_points_2)

      i_current = 0
      do k = 1, num_pts3
      do j = 1, num_pts2
      do i = 1, num_pts1
        i_current = i_current + 1
        control_points3d(2,i,j,k) = control_pts2(i_current) 
      end do
      end do
      end do
    end if
    if (d_dim >= 3) then
      read( input_file_id, control_points_3)

      i_current = 0
      do k = 1, num_pts3
      do j = 1, num_pts2
      do i = 1, num_pts1
        i_current = i_current + 1
        control_points3d(3,i,j,k) = control_pts3(i_current) 
      end do
      end do
      end do
    end if
    read( input_file_id, control_weights)
    weights_3d(:,:,:) = reshape(weights,(/num_pts1,num_pts2,num_pts3/))

    call spl_create_mapping_3d(   self            &
                                & , spline_deg1         &
                                & , spline_deg2         &
                                & , spline_deg3         &
                                & , knots1              &
                                & , knots2              &
                                & , knots3              &
                                & , control_points3d    &
                                & , weights=weights_3d  &
                                &  )

    close(input_file_id)

  end subroutine spl_read_mapping_3d
  ! ...................................................

end module spl_m_mapping_3d 
