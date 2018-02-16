!> @brief 
!> Module for predefined mappings.
!> @details
!> In this module are given some predefined mappings that can be directly imported  

!> \todo change the condition for refining the mapping. when one of them is 1, it is not working

module spl_m_mapping_gallery 

use spl_m_global
use spl_m_mapping_abstract, only: spl_t_mapping_abstract
use spl_m_mapping_1d,       only: spl_t_mapping_1d
use spl_m_mapping_2d,       only: spl_t_mapping_2d
use spl_m_mapping_3d,       only: spl_t_mapping_3d
use spl_m_mapping_cad,      only: spl_t_mapping_cad

implicit none

  private
  public :: spl_mapping_linear,            &
          & spl_mapping_arc,               &
          & spl_mapping_bilinear,          &
          & spl_mapping_annulus,           &
          & spl_mapping_eccentric_annulus, &
          & spl_mapping_circle,            &
          & spl_mapping_ellipse,           &
          & spl_mapping_quart_circle,      &
          & spl_mapping_collela,           &
          & spl_mapping_trilinear

contains

  ! .........................................................
  !> @brief      creates a 1D mapping line between the points P_1 and P_2 
  !>
  !> @param[inout] mapping_out   the mapping to create 
  !> @param[in]    P_1           left summet 
  !> @param[in]    P_2           right summet 
  !> @param[in]    degree        the polynomial degree [optional]   
  !> @param[in]    n_elements    number of elements/subdivisions of the logical domain [optional]   
  subroutine spl_mapping_linear(mapping_out, P_1, P_2, degree, n_elements)
  implicit none
     type(spl_t_mapping_1d)          , target, intent(inout) :: mapping_out 
     real(spl_rk), intent(in), dimension(:) :: P_1 
     real(spl_rk), intent(in), dimension(:) :: P_2
     integer, optional, intent(in) :: degree 
     integer, optional, intent(in) :: n_elements 
     ! local
     integer :: n 
     integer :: p 
     integer :: l_degree
     integer :: l_n_elements
     integer :: e
     integer :: i 
     integer :: d_dim
     real(spl_rk), dimension(:), allocatable :: knots_ini   
     real(spl_rk), dimension(:), allocatable :: knots_ref   
     real(spl_rk), dimension(:, :), allocatable :: control_points_ini
     real(spl_rk), dimension(:, :), allocatable :: control_points_ref
     real(spl_rk) :: t

     ! ...
     l_degree = 0
     if (present(degree)) then
       l_degree = degree - 1 
     end if
     ! ...

     ! ...
     l_n_elements = 0
     if (present(n_elements)) then
       l_n_elements = n_elements
     end if
     ! ...

     d_dim = size(P_1, 1)
     !> \TODO test if size(P_1,1) is equal to size(P_2,1)

     ! ...
     p = 1
     n = p + 1 
     ! ...

     ! ...
     allocate(knots_ini(n+p+1))
     allocate(control_points_ini(d_dim, n))
     ! ...

     ! ...
     knots_ini(1:p+1) = 0.0
     knots_ini(n+1:n+p+1) = 1.0
     ! ...

     ! ... control points
     control_points_ini(:,1) = P_1
     control_points_ini(:,2) = P_2
     ! ...

     ! ...
     call mapping_out % create(p, knots_ini, control_points_ini)
     ! ...

     ! ...
     if (l_degree > 0) then
       call mapping_out % refine(degree=l_degree, verbose=.FALSE.)
     end if
     ! ...

     ! ...
     if (l_n_elements > 0) then
       call mapping_out % refine(n_elements=l_n_elements, verbose=.FALSE.)
     end if
     ! ...

  end subroutine spl_mapping_linear
  ! .........................................................

  ! .........................................................
  !> @brief      creates a 1D mapping arc  
  !> TODO add angle, center and radius as optional args
  !>
  !> @param[inout] mapping_out the mapping to create 
  subroutine spl_mapping_arc(mapping_out)
  implicit none
     type(spl_t_mapping_1d)          , target, intent(inout) :: mapping_out 
     ! local
     integer :: n 
     integer :: p 
     integer :: d_dim
     real(spl_rk), dimension(:), allocatable :: knots_ini   
     real(spl_rk), dimension(:, :), allocatable :: control_points_ini
     real(spl_rk), dimension(:), allocatable :: weights_ini

     ! ...
     d_dim = 2
     p = 2 
     n = 3*p + p + 1 
     ! ...

     ! ...
     allocate(knots_ini(n+p+1))
     allocate(control_points_ini(d_dim, n))
     allocate(weights_ini(n))
     ! ...

     ! ...
     knots_ini(1:3) = 0.0_spl_rk 
     knots_ini(4:5) = 0.25_spl_rk
     knots_ini(6:7) = 0.5_spl_rk
     knots_ini(8:9) = 0.75_spl_rk
     knots_ini(10:) = 1.0_spl_rk 
     ! ...

     ! ... control points
     control_points_ini(:,1) = (/  0.0_spl_rk, -1.0_spl_rk /) 
     control_points_ini(:,2) = (/ -1.0_spl_rk, -1.0_spl_rk /) 
     control_points_ini(:,3) = (/ -1.0_spl_rk,  0.0_spl_rk /) 
     control_points_ini(:,4) = (/ -1.0_spl_rk,  1.0_spl_rk /) 
     control_points_ini(:,5) = (/  0.0_spl_rk,  1.0_spl_rk /) 
     control_points_ini(:,6) = (/  1.0_spl_rk,  1.0_spl_rk /) 
     control_points_ini(:,7) = (/  1.0_spl_rk,  0.0_spl_rk /) 
     control_points_ini(:,8) = (/  1.0_spl_rk, -1.0_spl_rk /) 
     control_points_ini(:,9) = (/  0.0_spl_rk, -1.0_spl_rk /) 
     ! ...                                 

     ! ... weights
     weights_ini(1) = 1.0_spl_rk 
     weights_ini(2) = 1.0_spl_rk / sqrt(2.0_spl_rk) 
     weights_ini(3) = 1.0_spl_rk 
     weights_ini(4) = 1.0_spl_rk / sqrt(2.0_spl_rk) 
     weights_ini(5) = 1.0_spl_rk 
     weights_ini(6) = 1.0_spl_rk / sqrt(2.0_spl_rk) 
     weights_ini(7) = 1.0_spl_rk 
     weights_ini(8) = 1.0_spl_rk / sqrt(2.0_spl_rk) 
     weights_ini(9) = 1.0_spl_rk 
     ! ...                                 

     ! ...
     call mapping_out % create(p, knots_ini, control_points_ini, weights=weights_ini)
     ! ...

  end subroutine spl_mapping_arc
  ! .........................................................

  ! .........................................................
  !> @brief      creates a 2D mapping bilinear between 
  !>
  !> @param[inout] mapping_out  the mapping to create 
  !> @param[in]    P_11         left-bottom summet 
  !> @param[in]    P_12         right-bottom summet 
  !> @param[in]    P_21         left-top summet 
  !> @param[in]    P_22         right-top summet 
  !> @param[in]    n_elements   a new subdivision [optional]  
  !> @param[in]    degrees      number of times the spline degree will be raised [optional]   
  !> @param[in]    other        mapping [optional]   
  subroutine spl_mapping_bilinear(mapping_out, P_11, P_12, P_21, P_22, degrees, n_elements, other)
  implicit none
     type(spl_t_mapping_2d)          , target, intent(inout) :: mapping_out 
     real(spl_rk), intent(in), dimension(:) :: P_11 
     real(spl_rk), intent(in), dimension(:) :: P_12
     real(spl_rk), intent(in), dimension(:) :: P_21 
     real(spl_rk), intent(in), dimension(:) :: P_22
     integer, dimension(2)  , optional, intent(in)    :: degrees 
     integer, dimension(2)  , optional, intent(in)    :: n_elements
     class(spl_t_mapping_2d),  optional, target, intent(in) :: other 
     ! local
     integer :: n 
     integer :: p 
     integer :: e
     integer :: i 
     integer :: d_dim
     integer, dimension(2) :: l_degrees 
     integer, dimension(2) :: l_n_elements
     real(spl_rk), dimension(:), allocatable :: knots_ini   
     real(spl_rk), dimension(:, :, :), allocatable :: control_points_ini
     real(spl_rk) :: t

     ! ...
     l_degrees = 0
     if (present(degrees)) then
       l_degrees = degrees - 1 
     end if
     ! ...

     ! ...
     l_n_elements = 0
     if (present(n_elements)) then
       l_n_elements = n_elements
     end if
     ! ...

     d_dim = size(P_11, 1)
     !> \TODO test if size(P_1,1) is equal to size(P_2,1)

     ! ...
     p = 1
     n = p + 1 
     ! ...

     ! ...
     allocate(knots_ini(n+p+1))
     allocate(control_points_ini(d_dim, 2, 2))
     ! ...

     ! ...
     knots_ini(1:p+1) = 0.0
     knots_ini(n+1:n+p+1) = 1.0
     ! ...

     ! ... control points
     control_points_ini(:,1,1) = P_11
     control_points_ini(:,2,1) = P_21
     control_points_ini(:,1,2) = P_12
     control_points_ini(:,2,2) = P_22
     ! ...

     ! ...
     call mapping_out % create(p, p, knots_ini, knots_ini, control_points_ini, other=other)
     ! ...

     ! ...
     if (l_degrees(1) * l_degrees(2) > 0) then
       call mapping_out % refine(degrees=l_degrees, verbose=.FALSE.)
     end if
     ! ...

     ! ...
     if (l_n_elements(1) * l_n_elements(2) > 0) then
       call mapping_out % refine(n_elements=l_n_elements, verbose=.FALSE.)
     end if
     ! ...

  end subroutine spl_mapping_bilinear
  ! .........................................................

  ! .........................................................
  !> @brief      creates a 2D mapping annulus 
  !>
  !> @param[inout] mapping_out     the mapping to create 
  !> @param[in]    r_min           minimal radius 
  !> @param[in]    r_max           maximal radius 
  !> @param[in]    center          array containing the center of the annulus [optional] 
  subroutine spl_mapping_annulus(mapping_out, r_min, r_max, center)
  implicit none
     type(spl_t_mapping_2d), target, intent(inout) :: mapping_out 
     real(spl_rk), intent(in) :: r_min
     real(spl_rk), intent(in) :: r_max
     real(spl_rk), dimension(:), optional, intent(in) :: center
     ! local
     integer :: n_u
     integer :: p_u 
     integer :: n_v 
     integer :: p_v 
     integer :: d_dim
     real(spl_rk), dimension(:), allocatable :: knots_u
     real(spl_rk), dimension(:), allocatable :: knots_v
     real(spl_rk), dimension(:,:,:), allocatable :: control_points
     real(spl_rk), dimension(:,:), allocatable :: weights 
     type(spl_t_mapping_1d) :: arc 
     type(spl_t_mapping_cad) :: cad

     ! ...
     call spl_mapping_arc(arc) 
     ! ...

     ! ...
     d_dim = arc % d_dim 

     n_u   = arc % n_u
     n_v   = 2

     p_u   = arc % p_u
     p_v   = 1 
     ! ...

     ! ...
     allocate(knots_u(n_u+p_u+1))
     allocate(knots_v(n_v+p_v+1))
     allocate(control_points(d_dim, n_u, n_v))
     allocate(weights(n_u, n_v))
     ! ...

     ! ...
     knots_u(:) = arc % knots_u(:) 
     ! ...

     ! ...
     knots_v(1:p_v+1) = 0.0
     knots_v(n_v+1:n_v+p_v+1) = 1.0
     ! ...

     ! ...
     control_points(:, :, 1) = r_min * arc % control_points(:,:) 
     control_points(:, :, 2) = r_max * arc % control_points(:,:)

     weights(:, 1) = arc % weights(:)
     weights(:, 2) = arc % weights(:)
     ! ...

     ! ...
     call mapping_out % create( p_u, p_v, &
                              & knots_u, knots_v, &
                              & control_points, &
                              & weights=weights)
     ! ...

     ! ...
     if (present(center)) then
       call cad % translate(mapping_out, center)
     end if
     ! ...

     ! ...
     call arc % free()
     ! ...

  end subroutine spl_mapping_annulus
  ! .........................................................
  
  ! .........................................................
  !> @brief      creates a 2D mapping eccentric annulus 
  !>
  !> @param[inout] mapping_out     the mapping to create 
  !> @param[in]    r_min           minimal radius 
  !> @param[in]    r_max           maximal radius 
  !> @param[in]    center_int      array containing the center of the internal circle 
  !> @param[in]    center_ext      array containing the center of the annulus [optional] 
  subroutine spl_mapping_eccentric_annulus(mapping_out, r_min, r_max, center_int, center_ext)
  implicit none
     type(spl_t_mapping_2d), target, intent(inout) :: mapping_out 
     real(spl_rk), intent(in) :: r_min
     real(spl_rk), intent(in) :: r_max
     real(spl_rk), dimension(:), intent(in) :: center_int
     real(spl_rk), dimension(:), optional, intent(in) :: center_ext
     ! local
     integer :: i
     integer :: n_u
     integer :: p_u 
     integer :: n_v 
     integer :: p_v 
     integer :: d_dim
     real(spl_rk), dimension(:), allocatable :: knots_u
     real(spl_rk), dimension(:), allocatable :: knots_v
     real(spl_rk), dimension(:,:,:), allocatable :: control_points
     real(spl_rk), dimension(:,:), allocatable :: weights 
     type(spl_t_mapping_1d) :: arc 

     ! ...
     call spl_mapping_arc(arc) 
     ! ...

     ! ...
     d_dim = arc % d_dim 

     n_u   = arc % n_u
     n_v   = 2

     p_u   = arc % p_u
     p_v   = 1 
     ! ...

     ! ...
     allocate(knots_u(n_u+p_u+1))
     allocate(knots_v(n_v+p_v+1))
     allocate(control_points(d_dim, n_u, n_v))
     allocate(weights(n_u, n_v))
     ! ...

     ! ...
     knots_u(:) = arc % knots_u(:) 
     ! ...

     ! ...
     knots_v(1:p_v+1) = 0.0
     knots_v(n_v+1:n_v+p_v+1) = 1.0
     ! ...

     ! ...
     do i = 1, d_dim 
      control_points(i, :, 1) = r_min * arc % control_points(i,:) + center_int(i) 
      if (present(center_ext)) then
        control_points(i, :, 2) = r_max * arc % control_points(i,:) + center_ext(i)
      else
        control_points(i, :, 2) = r_max * arc % control_points(i,:)
      end if
     end do

     weights(:, 1) = arc % weights(:)
     weights(:, 2) = arc % weights(:)
     ! ...
     ! ...
     call mapping_out % create( p_u, p_v, &
                              & knots_u, knots_v, &
                              & control_points, &
                              & weights=weights)
     ! ...

     ! ...
     call arc % free()
     ! ...

  end subroutine spl_mapping_eccentric_annulus
  ! .........................................................

  ! .........................................................
  !> @brief      creates a 2D mapping quart_circle 
  !>
  !> @param[inout] mapping_out   the mapping to create 
  !> @param[in]    r_min         minimal radius 
  !> @param[in]    r_max         maximal radius 
  !> @param[in]    center        circle center [optional] 
  subroutine spl_mapping_quart_circle(mapping_out, r_min, r_max, center)
  implicit none
     type(spl_t_mapping_2d), target, intent(inout) :: mapping_out 
     real(spl_rk), intent(in) :: r_min
     real(spl_rk), intent(in) :: r_max
     real(spl_rk), dimension(:), optional, intent(in) :: center
     ! local
     integer :: n_u 
     integer :: p_u 
     integer :: n_v 
     integer :: p_v 
     integer :: d_dim
     real(spl_rk), dimension(:), allocatable :: knots_u
     real(spl_rk), dimension(:), allocatable :: knots_v
     real(spl_rk), dimension(:,:,:), allocatable :: control_points
     real(spl_rk), dimension(:,:), allocatable :: weights 
     type(spl_t_mapping_cad) :: cad
     real(spl_rk), parameter :: s = 1.0_spl_rk / sqrt(2.0_spl_rk)

     ! ...
     d_dim = 2 

     n_u   = 3 
     n_v   = 2 

     p_u   = 2 
     p_v   = 1 
     ! ...

     ! ...
     allocate(knots_u(n_u+p_u+1))
     allocate(knots_v(n_v+p_v+1))
     allocate(control_points(d_dim, n_u, n_v))
     allocate(weights(n_u, n_v))
     ! ...

     ! ...
     knots_u(1:p_u+1) = 0.0
     knots_u(p_u+2:n_u+p_u+1) = 1.0
     ! ...

     ! ...
     knots_v(1:p_v+1) = 0.0
     knots_v(p_v+2:n_v+p_v+1) = 1.0
     ! ...

     ! ...
     control_points(:,1,1) = (/ 0.0_spl_rk ,     -r_min /)  
     control_points(:,2,1) = (/     -r_min ,     -r_min /)
     control_points(:,3,1) = (/     -r_min , 0.0_spl_rk /)
                                             
     control_points(:,1,2) = (/ 0.0_spl_rk ,     -r_max /)
     control_points(:,2,2) = (/     -r_max ,     -r_max /)
     control_points(:,3,2) = (/     -r_max , 0.0_spl_rk /)

!     control_points(:,:,:) = s * control_points(:,:,:)   

     weights(1,1) = 1.0_spl_rk  
     weights(2,1) = s
     weights(3,1) = 1.0_spl_rk 
     weights(1,2) = 1.0_spl_rk 
     weights(2,2) = s
     weights(3,2) = 1.0_spl_rk 
     ! ...

     ! ...
     call mapping_out % create( p_u, p_v, &
                              & knots_u, knots_v, &
                              & control_points, &
                              & weights=weights)
     ! ...

     ! ...
     if (present(center)) then
       call cad % translate(mapping_out, center)
     end if
     ! ...

  end subroutine spl_mapping_quart_circle
  ! .........................................................

  ! .........................................................
  !> @brief      creates a 2D mapping circle 
  !>
  !> @param[inout] mapping_out   the mapping to create 
  !> @param[in]    radius        circle radius [optional] 
  !> @param[in]    center        circle center [optional] 
  subroutine spl_mapping_circle(mapping_out, radius, center)
  implicit none
     type(spl_t_mapping_2d), target, intent(inout) :: mapping_out 
     real(spl_rk), optional, intent(in) :: radius
     real(spl_rk), dimension(:), optional, intent(in) :: center
     ! local
     integer :: n_u 
     integer :: p_u 
     integer :: n_v 
     integer :: p_v 
     integer :: d_dim
     real(spl_rk), dimension(:), allocatable :: knots_u
     real(spl_rk), dimension(:), allocatable :: knots_v
     real(spl_rk), dimension(:,:,:), allocatable :: control_points
     real(spl_rk), dimension(:,:), allocatable :: weights 
     type(spl_t_mapping_cad) :: cad
     real(spl_rk), parameter :: s = 1.0_spl_rk / sqrt(2.0_spl_rk)

     ! ...
     d_dim = 2 

     n_u   = 3 
     n_v   = 3

     p_u   = 2 
     p_v   = 2 
     ! ...

     ! ...
     allocate(knots_u(n_u+p_u+1))
     allocate(knots_v(n_v+p_v+1))
     allocate(control_points(d_dim, n_u, n_v))
     allocate(weights(n_u, n_v))
     ! ...

     ! ...
     knots_u(1:p_u+1) = 0.0
     knots_u(p_u+2:n_u+p_u+1) = 1.0
     ! ...

     ! ...
     knots_v(1:p_v+1) = 0.0
     knots_v(p_v+2:n_v+p_v+1) = 1.0
     ! ...

     ! ...
     control_points(:,1,1) = (/ -s            ,            -s  /)  
     control_points(:,2,1) = (/ 0.0_spl_rk    , -2.0_spl_rk*s  /)
     control_points(:,3,1) = (/ s             ,            -s  /)
     control_points(:,1,2) = (/ -2.0_spl_rk*s , 0.0_spl_rk     /)
     control_points(:,2,2) = (/ 0.0_spl_rk    , 0.0_spl_rk     /)
     control_points(:,3,2) = (/ 2.0_spl_rk*s  , 0.0_spl_rk     /)
     control_points(:,1,3) = (/ -s            , s              /)
     control_points(:,2,3) = (/ 0.0_spl_rk    , 2.0_spl_rk*s   /)
     control_points(:,3,3) = (/ s             ,            s   /)

     weights(1,1) = 1.0_spl_rk  
     weights(2,1) = s
     weights(3,1) = 1.0_spl_rk 
     weights(1,2) = s
     weights(2,2) = 1.0_spl_rk 
     weights(3,2) = s
     weights(1,3) = 1.0_spl_rk 
     weights(2,3) = s
     weights(3,3) = 1.0_spl_rk 
     ! ...

     ! ...
     if (present(radius)) then
       control_points = radius * control_points  
     end if
     ! ...

     ! ...
     call mapping_out % create( p_u, p_v, &
                              & knots_u, knots_v, &
                              & control_points, &
                              & weights=weights)
     ! ...

     ! ...
     if (present(center)) then
       call cad % translate(mapping_out, center)
     end if
     ! ...

  end subroutine spl_mapping_circle
  ! .........................................................

  ! .........................................................
  !> @brief      creates a 2D mapping ellipse 
  !>
  !> @param[inout] mapping_out   the mapping to create 
  !> @param[in]    minor_axis    seminminor axis [optional] (Default 1.0)
  !> @param[in]    major_axis    seminmajor axis [optional] (Default 1.0)
  !> @param[in]    center        circle center [optional] 
  subroutine spl_mapping_ellipse(mapping_out, minor_axis, major_axis, center)
  implicit none
     type(spl_t_mapping_2d), target, intent(inout) :: mapping_out 
     real(spl_rk), optional, intent(in) :: minor_axis 
     real(spl_rk), optional, intent(in) :: major_axis 
     real(spl_rk), dimension(:), optional, intent(in) :: center
     ! local

     ! ... first we create a circle of radius 1
     call spl_mapping_circle(mapping_out, 1.0_spl_rk, center) 
     ! ...

     ! ... scale the first component
     if (present(minor_axis)) then
       mapping_out % control_points(1,:,:) = minor_axis * mapping_out % control_points(1,:,:)   
     end if
     ! ...

     ! ... scale the second component
     if (present(major_axis)) then
       mapping_out % control_points(2,:,:) = major_axis * mapping_out % control_points(2,:,:)   
     end if
     ! ...

  end subroutine spl_mapping_ellipse
  ! .........................................................

  ! .........................................................
  !> @brief      creates a 2D collela mapping on a rectangle 
  !>             we use the property of the values on the Greville abscissae
  !>             by default, the ractangle is [-1,1]x[-1,1] 
  !>             in genrel, you can construct a rectangle [-a+xc,a+xc]x[-b+yc,b+yc]
  !>             where center is (xc,yc)
  !>
  !> @param[inout] mapping_out  the mapping to create 
  !> @param[in]    eps          the epsilon parameter 
  !> @param[in]    k1           the k1 parameter 
  !> @param[in]    k2           the k2 parameter 
  !> @param[in]    degrees      number of times the spline degree will be raised
  !> @param[in]    n_elements   a new subdivision
  !> @param[in]    center       left-bottom summet of the rectangle [optional] (default: (0,0)) 
  !> @param[in]    a            2a is the x-length [optional] (default: 1)
  !> @param[in]    b            2b is the y-length [optional] (default: 1)
  subroutine spl_mapping_collela(mapping_out, eps, k1, k2, degrees, n_elements, center, a, b)
  implicit none
    type(spl_t_mapping_2d), target,       intent(inout) :: mapping_out 
    real(spl_rk),                         intent(in)    :: eps
    real(spl_rk),                         intent(in)    :: k1
    real(spl_rk),                         intent(in)    :: k2
    integer, dimension(2),                intent(in)    :: degrees 
    integer, dimension(2),                intent(in)    :: n_elements 
    real(spl_rk), dimension(2), optional, intent(in)    :: center 
    real(spl_rk), optional,               intent(in)    :: a 
    real(spl_rk), optional,               intent(in)    :: b 
    ! local
    integer :: i
    integer :: j 
    real(spl_rk) :: u 
    real(spl_rk) :: v
    real(spl_rk) :: x 
    real(spl_rk) :: y
    real(spl_rk) :: ku 
    real(spl_rk) :: kv 
    real(spl_rk), dimension(2) :: P_11
    real(spl_rk), dimension(2) :: P_12
    real(spl_rk), dimension(2) :: P_21
    real(spl_rk), dimension(2) :: P_22
    real(spl_rk), dimension(:), allocatable :: us 
    real(spl_rk), dimension(:), allocatable :: vs 
    type(spl_t_mapping_cad) :: cad

    ! ...
    P_11 = (/ 0.0_spl_rk, 0.0_spl_rk /)
    P_21 = (/ 1.0_spl_rk, 0.0_spl_rk /)
    P_12 = (/ 0.0_spl_rk, 1.0_spl_rk /)
    P_22 = (/ 1.0_spl_rk, 1.0_spl_rk /)
    ! ...

    ! ...
    call spl_mapping_bilinear( mapping_out, P_11, P_12, P_21, P_22, &
                             & degrees=degrees, &
                             & n_elements=n_elements)
    ! ...

    ! ...
    allocate(us(mapping_out % n_u))
    allocate(vs(mapping_out % n_v))

    call mapping_out % get_greville(us, axis=1)
    call mapping_out % get_greville(vs, axis=2)
    ! ...

    ! ...
    do i = 1, mapping_out % n_u
    do j = 1, mapping_out % n_v
      u = us(i)
      v = vs(j)

      ku = 2.0_spl_rk * spl_pi * k1 
      kv = 2.0_spl_rk * spl_pi * k2 

      x = 2.0_spl_rk * (u + eps * sin(ku * u) * sin(kv * v)) - 1.0_spl_rk
      y = 2.0_spl_rk * (v + eps * sin(ku * u) * sin(kv * v)) - 1.0_spl_rk

      mapping_out % control_points(1, i, j) = x 
      mapping_out % control_points(2, i, j) = y 
    end do
    end do
    ! ...

    ! ... make sure the boundary is ok
    mapping_out % control_points(1, 1, :) = -1.0_spl_rk 
    mapping_out % control_points(1, mapping_out % n_u, :) = 1.0_spl_rk 
    mapping_out % control_points(2, :, 1) = -1.0_spl_rk 
    mapping_out % control_points(2, :, mapping_out % n_u) = 1.0_spl_rk 
    ! ...

    ! ... scale the first component
    if (present(a)) then
      mapping_out % control_points(1,:,:) = a * mapping_out % control_points(1,:,:)   
    end if
    ! ...

    ! ... scale the second component
    if (present(b)) then
      mapping_out % control_points(2,:,:) = b * mapping_out % control_points(2,:,:)   
    end if
    ! ...

    ! ...
    if (present(center)) then
      call cad % create()
      call cad % translate(mapping_out, center)
      call cad % free()
    end if
    ! ...

  end subroutine spl_mapping_collela
  ! .........................................................

  ! .........................................................
  !> @brief      creates a 3D mapping trilinear between 
  !>
  !> @param[inout] mapping_out  the mapping to create 
  !> @param[in]    P_111        bottom face left-bottom summet 
  !> @param[in]    P_112        bottom face right-bottom summet 
  !> @param[in]    P_121        bottom face left-top summet 
  !> @param[in]    P_122        bottom face right-top summet 
  !> @param[in]    P_211        top face left-bottom summet 
  !> @param[in]    P_212        top face right-bottom summet 
  !> @param[in]    P_221        top face left-top summet 
  !> @param[in]    P_222        top face right-top summet 
  !> @param[in]    n_elements   a new subdivision [optional]  
  !> @param[in]    degrees      number of times the spline degree will be raised [optional]   
  subroutine spl_mapping_trilinear( mapping_out, &
                                  & P_111, P_121, P_211, P_221, &
                                  & P_112, P_122, P_212, P_222, &
                                  & degrees, n_elements)
  implicit none
     type(spl_t_mapping_3d)          , target, intent(inout) :: mapping_out 
     real(spl_rk), intent(in), dimension(:) :: P_111 
     real(spl_rk), intent(in), dimension(:) :: P_121
     real(spl_rk), intent(in), dimension(:) :: P_211 
     real(spl_rk), intent(in), dimension(:) :: P_221
     real(spl_rk), intent(in), dimension(:) :: P_112 
     real(spl_rk), intent(in), dimension(:) :: P_122
     real(spl_rk), intent(in), dimension(:) :: P_212 
     real(spl_rk), intent(in), dimension(:) :: P_222
     integer, dimension(3)  , optional, intent(in)    :: degrees 
     integer, dimension(3)  , optional, intent(in)    :: n_elements 
     ! local
     integer :: n 
     integer :: p 
     integer :: e
     integer :: i 
     integer :: d_dim
     integer, dimension(3) :: l_degrees 
     integer, dimension(3) :: l_n_elements
     real(spl_rk), dimension(:), allocatable :: knots_ini   
     real(spl_rk), dimension(:, :, :, :), allocatable :: control_points_ini
     real(spl_rk) :: t

     ! ...
     l_degrees = 0
     if (present(degrees)) then
       l_degrees = degrees - 1 
     end if
     ! ...

     ! ...
     l_n_elements = 0
     if (present(n_elements)) then
       l_n_elements = n_elements
     end if
     ! ...

     d_dim = size(P_111, 1)
     !> \TODO test if size(P_1,1) is equal to size(P_2,1)

     ! ...
     p = 1
     n = p + 1 
     ! ...

     ! ...
     allocate(knots_ini(n+p+1))
     allocate(control_points_ini(d_dim, 2, 2, 2))
     ! ...

     ! ...
     knots_ini(1:p+1) = 0.0
     knots_ini(n+1:n+p+1) = 1.0
     ! ...

     ! ... control points
     control_points_ini(:,1,1,1) = P_111 
     control_points_ini(:,2,1,1) = P_211
     control_points_ini(:,1,2,1) = P_121
     control_points_ini(:,2,2,1) = P_221
     control_points_ini(:,1,1,2) = P_112
     control_points_ini(:,2,1,2) = P_212
     control_points_ini(:,1,2,2) = P_122
     control_points_ini(:,2,2,2) = P_222
     ! ...

     ! ...
     call mapping_out % create(p, p, p, &
       & knots_ini, knots_ini, knots_ini, &
       & control_points_ini)
     ! ...

     ! ...
     if (l_degrees(1) * l_degrees(2) * l_degrees(3) > 0) then
       call mapping_out % refine(degrees=l_degrees, verbose=.FALSE.)
     end if
     ! ...

     ! ...
     if (l_n_elements(1) * l_n_elements(2) * l_n_elements(3) > 0) then
       call mapping_out % refine(n_elements=l_n_elements, verbose=.FALSE.)
     end if
     ! ...
  end subroutine spl_mapping_trilinear
  ! .........................................................

end module spl_m_mapping_gallery
