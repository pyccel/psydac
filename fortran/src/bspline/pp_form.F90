!> @brief 
!> Module for converting a Splines to its pp-form
!> @details
!> description of the B-Splines as piece-wise polynomials on every element
!>

module spl_m_pp_form 
use spl_m_pppack
implicit none
contains

  ! .......................................................
  !> @brief     1d to pp form 
  !>
  !> @param[in]    arr_b_coef   array of B-splines coefficients, of size (n_u) 
  !> @param[in]    knots_u      array of knots in the u-direction, of size (n_u+p_u+1) 
  !> @param[in]    n_u          number of non-vanishing B-Splines in the u direction 
  !> @param[in]    p_u          spline degree in the u direction 
  !> @param[in]    n_elements_u number of non-zero elements 
  !> @param[inout] arr_pp       array of pp forms, of size (p_u+1,n_elements_u) 
  subroutine to_pp_form_1d(arr_b_coef, knots_u, n_u, p_u, n_elements_u, arr_pp)
  implicit none
    real(kind=8), dimension(:)  , intent(in)    :: arr_b_coef
    real(kind=8), dimension(:)  , intent(in)    :: knots_u
    integer                     , intent(in)    :: n_u
    integer                     , intent(in)    :: p_u
    integer                     , intent(in)    :: n_elements_u
    real(kind=8), dimension(:,:), intent(inout) :: arr_pp
    ! local
    real(kind=8), dimension(p_u + 1, p_u + 1)          :: scrtch_u 
    real(kind=8), dimension(n_elements_u + 1)          :: break_u
    real(kind=8), dimension(p_u + 1, n_elements_u + 1) :: coef_u 
    integer :: l_u
    integer :: i_dim

    arr_pp = 0.0d0 

    l_u = n_elements_u + 1
    call bsplpp ( knots_u,       &
      & arr_b_coef(:), &
      & n_u, p_u + 1,     &
      & scrtch_u, break_u, coef_u, l_u )

    arr_pp(:, 1:l_u) = coef_u(:,1:l_u)
  end subroutine to_pp_form_1d
  ! .......................................................

  ! .......................................................
  !> @brief     2d to pp form 
  !>
  !> @param[in]    arr_b_coef   array of B-splines coefficients, of size (n_u,n_v) 
  !> @param[in]    knots_u      array of knots in the u-direction, of size (n_u+p_u+1) 
  !> @param[in]    knots_v      array of knots in the u-direction, of size (n_v+p_v+1) 
  !> @param[in]    n_u          number of non-vanishing B-Splines in the u direction 
  !> @param[in]    n_v          number of non-vanishing B-Splines in the u direction  
  !> @param[in]    p_u          spline degree in the u direction 
  !> @param[in]    p_v          spline degree in the v direction
  !> @param[in]    n_elements_u number of non-zero elements 
  !> @param[in]    n_elements_v number of non-zero elements 
  !> @param[inout] arr_pp       array of pp forms, of size (p_u+1,p_v+1,n_elements_u,n_elements_v) 
  subroutine to_pp_form_2d(arr_b_coef, knots_u, knots_v, n_u, n_v, p_u, p_v, n_elements_u, n_elements_v, arr_pp)
  implicit none
    real(kind=8), dimension(:,:)    , intent(in)    :: arr_b_coef
    real(kind=8), dimension(:)      , intent(in)    :: knots_u
    real(kind=8), dimension(:)      , intent(in)    :: knots_v
    integer                         , intent(in)    :: n_u
    integer                         , intent(in)    :: n_v
    integer                         , intent(in)    :: p_u
    integer                         , intent(in)    :: p_v
    integer                         , intent(in)    :: n_elements_u
    integer                         , intent(in)    :: n_elements_v
    real(kind=8), dimension(:,:,:,:), intent(inout) :: arr_pp
    ! local
    real(kind=8), dimension(p_u + 1, p_u + 1)               :: scrtch_u 
    real(kind=8), dimension(p_v + 1, p_v + 1)               :: scrtch_v 
    real(kind=8), dimension(n_elements_u + 1)               :: break_u
    real(kind=8), dimension(n_elements_v + 1)               :: break_v
    real(kind=8), dimension(p_u + 1, n_u + 1)               :: coef_u 
    real(kind=8), dimension(p_v + 1, n_v + 1)               :: coef_v 
    real(kind=8), dimension(p_u + 1, n_u + 1, n_v)          :: coef
    real(kind=8), dimension(p_u + 1, p_v + 1, n_elements_v) :: d_uv
    integer :: l_v
    integer :: l_u
    integer :: i_point_v
    integer :: i_point_u
    integer :: i_v
    integer :: i_u
    integer :: i_element_v
    integer :: i_element_u

    arr_pp = 0.0d0

    l_u = n_elements_u + 1
    l_v = n_elements_v + 1
    do i_point_v = 1, n_v
      call bsplpp ( knots_u,       &
        & arr_b_coef(:, i_point_v), &
        & n_u, p_u + 1,     &
        & scrtch_u, break_u, coef_u, l_u )

      coef(:, 1:l_u, i_point_v) = coef_u(:,1:l_u)
    end do

    do i_element_u = 1, n_elements_u
      do i_u = 1, p_u + 1
        call bsplpp ( knots_v,       &
          & coef(i_u, i_element_u, :), &
          & n_v, p_v + 1,     &
          & scrtch_v, break_v, coef_v, l_v )

        d_uv(i_u, :, 1:l_v) = coef_v(:,1:l_v)
      end do

      do i_element_v=1, n_elements_v
        arr_pp(:, :, i_element_u, i_element_v) = d_uv(:, :, i_element_v)
      end do
    end do
  end subroutine to_pp_form_2d
  ! .......................................................

  ! .......................................................
  !> @brief     3d to pp form 
  !>
  !> @param[in]    arr_b_coef   array of B-splines coefficients, of size (n_u,n_v) 
  !> @param[in]    knots_u      array of knots in the u-direction, of size (n_u+p_u+1) 
  !> @param[in]    knots_v      array of knots in the u-direction, of size (n_v+p_v+1) 
  !> @param[in]    knots_w      array of knots in the u-direction, of size (n_w+p_w+1) 
  !> @param[in]    n_u          number of non-vanishing B-Splines in the u direction 
  !> @param[in]    n_v          number of non-vanishing B-Splines in the v direction  
  !> @param[in]    n_w          number of non-vanishing B-Splines in the w direction  
  !> @param[in]    p_u          spline degree in the u direction 
  !> @param[in]    p_v          spline degree in the v direction
  !> @param[in]    p_w          spline degree in the w direction
  !> @param[in]    n_elements_u number of non-zero elements 
  !> @param[in]    n_elements_v number of non-zero elements 
  !> @param[in]    n_elements_w number of non-zero elements 
  !> @param[inout] arr_pp       array of pp forms, of size (p_u+1,p_v+1,p_w+1,n_elements_u,n_elements_v,n_elements_w) 
  subroutine to_pp_form_3d(arr_b_coef, &
      & knots_u, knots_v, knots_w, &
      & n_u, n_v, n_w, &
      & p_u, p_v, p_w, &
      & n_elements_u, n_elements_v, n_elements_w, &
      & arr_pp)
  implicit none
    real(kind=8), dimension(:,:,:)      , intent(in)    :: arr_b_coef
    real(kind=8), dimension(:)          , intent(in)    :: knots_u
    real(kind=8), dimension(:)          , intent(in)    :: knots_v
    real(kind=8), dimension(:)          , intent(in)    :: knots_w
    integer                             , intent(in)    :: n_u
    integer                             , intent(in)    :: n_v
    integer                             , intent(in)    :: n_w
    integer                             , intent(in)    :: p_u
    integer                             , intent(in)    :: p_v
    integer                             , intent(in)    :: p_w
    integer                             , intent(in)    :: n_elements_u
    integer                             , intent(in)    :: n_elements_v
    integer                             , intent(in)    :: n_elements_w
    real(kind=8), dimension(:,:,:,:,:,:), intent(inout) :: arr_pp
    ! local
    real(kind=8), dimension(:,:,:,:), allocatable :: coef_uv
    real(kind=8), dimension(:,:,:,:,:), allocatable :: coef
    real(kind=8), dimension(p_w + 1, n_w + 1) :: coef_w 
    real(kind=8), dimension(p_w + 1, p_w + 1) :: scrtch_w 
    real(kind=8), dimension(n_elements_w + 1) :: break_w
    integer :: l_w
    integer :: i_point_w
    integer :: i_v
    integer :: i_u
    integer :: i_element_v
    integer :: i_element_u

    allocate(coef_uv(p_u + 1, p_v + 1, n_elements_u, n_elements_v))
    allocate(coef(p_u + 1, p_v + 1, n_elements_u, n_elements_v, n_w))

    l_w = n_elements_w + 1
    do i_point_w = 1, n_w
      call to_pp_form_2d(arr_b_coef(:,:,i_point_w), &
        & knots_u, knots_v, &
        & n_u, n_v, &
        & p_u, p_v, &
        & n_elements_u, n_elements_v, &
        & coef_uv) 

      coef(:,:,:,:,i_point_w) = coef_uv(:,:,:,:)
    end do

    do i_element_v = 1, n_elements_v
      do i_element_u = 1, n_elements_u
        do i_v = 1, p_v + 1
          do i_u = 1, p_u + 1
            call bsplpp ( knots_w,       &
              & coef(i_u, i_v, i_element_u, i_element_v, :), &
              & n_w, p_w + 1,     &
              & scrtch_w, break_w, coef_w, l_w )

            arr_pp(i_u, i_v, :, i_element_u, i_element_v, 1:l_w) = coef_w(:,1:l_w)
          end do
        end do
      end do
    end do

  end subroutine to_pp_form_3d
  ! .......................................................

  ! ..................................................
  !> @brief     compute pp-coeff of the B-Splines basis 
  !>
  !> @param[in]    p               polynomial degree 
  !> @param[in]    knots           knots vector
  !> @param[in]    ien             connectivity array. of dim(p+1,n_elements) indexed as (i_local_basis,i_element)
  !> @param[inout] pp_coeff_basis  pp-coefficients for all B-Splines on all elements
  subroutine compute_pp_coef_basis(p, knots, ien, pp_coeff_basis)
  implicit none
    integer                          , intent(in)    :: p
    real(8), dimension(:)            , intent(in)    :: knots
    integer, dimension(:,:)          , intent(in)    :: ien
    real(8), dimension(:,:,:)        , intent(inout) :: pp_coeff_basis
    ! local
    integer :: i_control_point
    integer :: i_basis
    integer :: i_element
    integer :: n_elements
    integer :: n
    real(8), dimension(:)  , allocatable :: control_points
    real(8), dimension(:,:), allocatable :: pp_coeff 

    ! ... reset the pp coeff of the basis
    pp_coeff_basis = 0.0d0
    ! ... 

    ! ... allocation of the control_points array
    n = ubound(knots,1) - p - 1
    allocate(control_points(n))
    control_points = 0.0d0
    ! ...

    ! ... allocate pp_coeff
    n_elements = ubound(ien,2)
    allocate(pp_coeff(p + 1, n_elements))
    ! ...

    ! ... loop over all control points, set it to one and then convert to the pp-form
    do i_element= 1, n_elements 
      do i_basis = 1, p + 1
        i_control_point = ien(i_basis, i_element)

        control_points = 0.0d0
        control_points(i_control_point) = 1.0d0

        call to_pp_form_1d(control_points, knots, n, p, n_elements, pp_coeff) 

        pp_coeff_basis(:, i_basis, i_element) = pp_coeff_basis(:, i_basis, i_element) &
                                            & + pp_coeff(:, i_element) 
      end do
    end do
    ! ...

    ! ... deallocate the mapping object
    deallocate(pp_coeff)
    deallocate(control_points)
    ! ...

  end subroutine compute_pp_coef_basis
  ! ..................................................

end module spl_m_pp_form 
