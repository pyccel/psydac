!> @brief 
!> Module for Common CAD algorithms.
!> @details
!> The object mapping_cad allows the user to apply some common construction algorithms in Computer Aided Design.
!> More details can be found in the "NURBS Book" 

module spl_m_mapping_cad 

use spl_m_global
use spl_m_abstract,         only: spl_t_abstract
use spl_m_mapping_abstract, only: spl_t_mapping_abstract
use spl_m_mapping_1d,       only: spl_t_mapping_1d
use spl_m_mapping_2d,       only: spl_t_mapping_2d
use spl_m_mapping_3d,       only: spl_t_mapping_3d

implicit none

  private

  ! .........................................................
  !> @brief 
  !> class for CAD mappings.
  type, public, extends(spl_t_abstract) :: spl_t_mapping_cad
  contains
    procedure :: translate => translate_mapping_cad
    procedure :: rotate    => rotate_mapping_cad
    procedure :: extrude   => extrude_mapping_cad
    procedure :: transpose => transpose_mapping_cad
    procedure :: create    => create_mapping_cad
    procedure :: free      => free_mapping_cad
  end type spl_t_mapping_cad
  ! .........................................................

contains

  ! .........................................................
  !> @brief      creates a cad mapping 
  !>
  !> @param[inout] self the current object 
  subroutine create_mapping_cad(self)
  implicit none
     class(spl_t_mapping_cad), intent(inout) :: self
     ! local

  end subroutine create_mapping_cad 
  ! .........................................................

  ! .........................................................
  !> @brief      destroys a cad mapping 
  !>
  !> @param[inout] self the current object 
  subroutine free_mapping_cad(self)
  implicit none
     class(spl_t_mapping_cad), intent(inout) :: self
     ! local

  end subroutine free_mapping_cad 
  ! .........................................................

  ! .........................................................
  !> @brief     translates a mapping given a displacements array [inplace] 
  !>
  !> @param[in]    self           the current object 
  !> @param[inout] mapping        mapping to be translated 
  !> @param[in]    displacements  displacement vector 
  subroutine translate_mapping_cad(self, mapping, displacements)
  implicit none
     class(spl_t_mapping_cad),        intent(in)    :: self
     class(spl_t_mapping_abstract),   intent(inout) :: mapping 
     real(kind=spl_rk), dimension(:), intent(in)    :: displacements
     ! local
     integer :: i

     ! ...
     select type (mapping)
     class is (spl_t_mapping_1d)
       do i = 1, mapping % d_dim 
         mapping % control_points(i,:) = mapping % control_points(i,:) &
                                   & + displacements(i)
       end do
     class is (spl_t_mapping_2d)
       do i = 1, mapping % d_dim 
         mapping % control_points(i,:,:) = mapping % control_points(i,:,:) &
                                   & + displacements(i)
       end do
     class is (spl_t_mapping_3d)
       do i = 1, mapping % d_dim 
         mapping % control_points(i,:,:,:) = mapping % control_points(i,:,:,:) &
                                   & + displacements(i)
       end do
     end select
     ! ...

  end subroutine translate_mapping_cad
  ! .........................................................

  ! .........................................................
  !> @brief     rotates a mapping given an angle and an axis [inplace] 
  !>
  !> @param[in]    self     the current object 
  !> @param[inout] mapping  mapping to be rotated 
  !> @param[in]    angle    rotation angle 
  !> @param[in]    axis     rotation axis 
  subroutine rotate_mapping_cad(self, mapping, angle, axis)
  implicit none
     class(spl_t_mapping_cad),      intent(in)    :: self
     class(spl_t_mapping_abstract), intent(inout) :: mapping 
     real(kind=spl_rk),             intent(in)    :: angle
     integer,                       intent(in)    :: axis
     ! local
     integer :: i_u
     integer :: i_v
     integer :: i_w
     real(kind=spl_rk) :: sin_a
     real(kind=spl_rk) :: cos_a
     real(kind=spl_rk), dimension(:), allocatable :: x
     real(kind=spl_rk), dimension(:), allocatable :: y

     ! ...
     if ((mapping % d_dim .ne. 2) .and. (mapping % d_dim .ne. 3)) then
       stop 'expecting d_dim to be 2 or 3'
     end if
     ! ...

     ! ...
     if (mapping % d_dim .eq. 3) then
       stop 'not yet implemented for d_dim = 3'
     end if
     ! ...

     ! ...
     if (axis .ne. 3) then
       stop 'not yet implemented for axis <> 3'
     end if
     ! ...

     ! ...
     sin_a = sin(angle)
     cos_a = cos(angle)
     ! ...

     ! ...
     allocate(x(mapping % d_dim))
     allocate(y(mapping % d_dim))
     ! ...

     ! ...
     select type (mapping)
     class is (spl_t_mapping_1d)
       do i_u = 1, mapping % n_u 
         x = mapping % control_points(:, i_u)
         mapping % control_points(:, i_u) = y
       end do
     class is (spl_t_mapping_2d)
       do i_u = 1, mapping % n_u 
         do i_v = 1, mapping % n_v
           x = mapping % control_points(:, i_u, i_v)

           y(1) = cos_a * x(1) - sin_a * x(2)
           y(2) = sin_a * x(1) + cos_a * x(2)

           mapping % control_points(:, i_u, i_v) = y
         end do
       end do
     class is (spl_t_mapping_3d)
       do i_u = 1, mapping % n_u 
         do i_v = 1, mapping % n_v
           do i_w = 1, mapping % n_w
             x = mapping % control_points(:, i_u, i_v, i_w)
             mapping % control_points(:, i_u, i_v, i_w) = y
           end do
         end do
       end do
     end select
     ! ...

  end subroutine rotate_mapping_cad
  ! .........................................................

  ! .........................................................
  !> @brief    Construct a NURBS surface/volume by
  !>           extruding a NURBS curve/surface.
  !>
  !> @param[in]    self           the current object 
  !> @param[in]    mapping_in     input mapping to be extruded (curve or surface)
  !> @param[in]    displacements  displacement vector 
  !> @param[inout] mapping_out    output mapping after extrude (surface or volume)
  subroutine extrude_mapping_cad(self, mapping_in, displacements, mapping_out)
  implicit none
     class(spl_t_mapping_cad),         intent(in) :: self
     class(spl_t_mapping_abstract),    intent(in) :: mapping_in 
     real(kind=spl_rk), dimension(:),  intent(in) :: displacements
     class(spl_t_mapping_abstract), intent(inout) :: mapping_out 
     ! local
     real(spl_rk), dimension(:,:,:,:), allocatable :: control_points 
     real(spl_rk), dimension(:,:,:), allocatable :: weights 
     real(spl_rk), dimension(:), allocatable :: knots_u
     real(spl_rk), dimension(:), allocatable :: knots_v   
     real(spl_rk), dimension(:), allocatable :: knots_w   
     integer :: n_u
     integer :: n_v
     integer :: n_w
     integer :: p_u
     integer :: p_v
     integer :: p_w
     integer :: d_dim
     integer :: d

     ! ...
     select type (mapping_in)
     class is (spl_t_mapping_1d)
       select type (mapping_out)
       class is (spl_t_mapping_2d)
         ! ...
         d_dim = max(mapping_in % d_dim, 2) 

         n_u   = mapping_in % n_u
         n_v   = 2
         n_w   = 1 

         p_u   = mapping_in % p_u
         p_v   = 1 
         p_w   = 0 
         ! ...

         ! ...
         allocate(knots_u(n_u+p_u+1))
         allocate(knots_v(n_v+p_v+1))
         allocate(control_points(d_dim, n_u, n_v, n_w))
         allocate(weights(n_u, n_v, n_w))
         ! ...

         ! ...
         knots_u(:) = mapping_in % knots_u(:) 
         ! ...

         ! ...
         knots_v(1:p_v+1) = 0.0
         knots_v(n_v+1:n_v+p_v+1) = 1.0
         ! ...

         ! ...
         d = mapping_in % d_dim
         control_points = 0.0_spl_rk
         control_points(1:d,:,1,1) = mapping_in % control_points(1:d,:)
         control_points(1:d,:,2,1) = mapping_in % control_points(1:d,:)
         do d = 1, d_dim
           control_points(d,:,2,1) = control_points(d,:,2,1) &
                                 & + displacements(d)    
         end do

         weights(:, 1, 1) = mapping_in % weights(:)
         weights(:, 2, 1) = mapping_in % weights(:)
         ! ...

         ! ...
         call mapping_out % create( p_u, p_v, &
                                  & knots_u, knots_v, &
                                  & control_points(:,:,:,1), &
                                  & weights=weights(:,:,1))
         ! ...
       class default 
         stop "extrude_mapping_cad: mapping_out must be of dimension 2."
       end select
     class is (spl_t_mapping_2d)
       select type (mapping_out)
       class is (spl_t_mapping_3d)
         ! ...
         d_dim = max(mapping_in % d_dim, 3) 

         n_u   = mapping_in % n_u
         n_v   = mapping_in % n_v
         n_w   = 2 

         p_u   = mapping_in % p_u
         p_v   = mapping_in % p_v 
         p_w   = 1 
         ! ...

         ! ...
         allocate(knots_u(n_u+p_u+1))
         allocate(knots_v(n_v+p_v+1))
         allocate(knots_w(n_w+p_w+1))
         allocate(control_points(d_dim, n_u, n_v, n_w))
         allocate(weights(n_u, n_v, n_w))
         ! ...

         ! ...
         knots_u(:) = mapping_in % knots_u(:) 
         knots_v(:) = mapping_in % knots_v(:) 
         ! ...

         ! ...
         knots_w(1:p_w+1) = 0.0
         knots_w(n_w+1:n_w+p_w+1) = 1.0
         ! ...

         ! ...
         d = mapping_in % d_dim
         control_points = 0.0_spl_rk
         control_points(1:d,:,:,1) = mapping_in % control_points(1:d,:,:)
         control_points(1:d,:,:,2) = mapping_in % control_points(1:d,:,:)
         do d = 1, d_dim
           control_points(d,:,:,2) = control_points(d,:,:,2) &
                                 & + displacements(d)    
         end do

         weights(:,:,1) = mapping_in % weights(:,:)
         weights(:,:,2) = mapping_in % weights(:,:)
         ! ...

         ! ...
         call mapping_out % create( p_u, p_v, p_w, &
                                  & knots_u, knots_v, knots_w,  &
                                  & control_points(:,:,:,:), &
                                  & weights=weights(:,:,:))
         ! ...
       class default 
         stop "extrude_mapping_cad: mapping_out must be of dimension 3."
       end select
     class default 
       stop "extrude_mapping_cad: mapping_in must be of dimension 1 or 2."
     end select
     ! ...

  end subroutine extrude_mapping_cad
  ! .........................................................

  ! .........................................................
  !> @brief     transposes a mapping given a displacements array [inplace] 
  !>
  !> @param[in]    self           the current object 
  !> @param[in]    mapping_in     input mapping to be transposed
  !> @param[inout] mapping_out    output mapping 
  subroutine transpose_mapping_cad(self, mapping_in, mapping_out)
  implicit none
     class(spl_t_mapping_cad),      intent(in)    :: self
     class(spl_t_mapping_abstract), intent(inout) :: mapping_in 
     class(spl_t_mapping_abstract), intent(inout) :: mapping_out 
     ! local
     real(spl_rk), dimension(:,:,:), allocatable :: control_points 
     real(spl_rk), dimension(:,:), allocatable :: weights 
     real(spl_rk), dimension(:), allocatable :: knots_u
     real(spl_rk), dimension(:), allocatable :: knots_v   
     integer :: n_u
     integer :: n_v
     integer :: p_u
     integer :: p_v
     integer :: d_dim
     integer :: d

     ! ...
     select type (mapping_in)
     class is (spl_t_mapping_1d)
       ! ...
       select type (mapping_out)
       class is (spl_t_mapping_1d)
         print *, "transpose_mapping_cad: do not exist in 1d."
       class default 
         stop "transpose_mapping_cad: mapping_out must be of dimension 1."
       end select
       ! ...
     class is (spl_t_mapping_2d)
       ! ...
       select type (mapping_out)
       class is (spl_t_mapping_2d)
         ! ...
         d_dim = mapping_in % d_dim

         n_u   = mapping_in % n_v
         n_v   = mapping_in % n_u

         p_u   = mapping_in % p_v
         p_v   = mapping_in % p_u
         ! ...

         ! ...
         allocate(knots_u(n_u+p_u+1))
         allocate(knots_v(n_v+p_v+1))
         allocate(control_points(d_dim, n_u, n_v))
         allocate(weights(n_u, n_v))
         ! ...

         ! ...
         knots_u(:) = mapping_in % knots_v(:) 
         knots_v(:) = mapping_in % knots_u(:) 
         ! ...

         ! ...
         control_points = 0.0_spl_rk
         do d = 1, d_dim
           control_points(d,:,:) = transpose(mapping_in % control_points(d,:,:))
         end do
         weights(:, :) = transpose(mapping_in % weights(:,:))
         ! ...

         ! ...
         call mapping_out % create( p_u, p_v, &
                                  & knots_u, knots_v, &
                                  & control_points(:,:,:), &
                                  & weights=weights(:,:))
         ! ...
       class default 
         stop "transpose_mapping_cad: mapping_out must be of dimension 2."
       end select
       ! ...
     class is (spl_t_mapping_3d)
       ! ...
       select type (mapping_out)
       class is (spl_t_mapping_3d)
         print *, "transpose_mapping_cad: not yet implemented in 3d."
         stop 
       class default 
         stop "transpose_mapping_cad: mapping_out must be of dimension 3."
       end select
       ! ...
     end select
     ! ...

  end subroutine transpose_mapping_cad
  ! .........................................................

end module spl_m_mapping_cad
