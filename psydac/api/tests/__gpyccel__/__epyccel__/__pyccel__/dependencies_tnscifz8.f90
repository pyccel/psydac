module dependencies_tnscifz8

implicit none

contains

!........................................
subroutine assembly(global_test_basis_u_1, global_test_basis_u_2, &
      global_span_u_1, global_span_u_2, global_x1, global_w1, global_x2 &
      , global_w2, test_u_p1, test_u_p2, n_element_1, n_element_2, k1, &
      k2, pad1, pad2, l_el_j6duku, g_el_wzmbdw, global_arr_coeffs_u)

  implicit none

  real(kind=8), intent(in) :: global_test_basis_u_1(0:,0:,0:,0:)
  real(kind=8), intent(in) :: global_test_basis_u_2(0:,0:,0:,0:)
  integer(kind=8), intent(in) :: global_span_u_1(0:)
  integer(kind=8), intent(in) :: global_span_u_2(0:)
  real(kind=8), intent(in) :: global_x1(0:,0:)
  real(kind=8), intent(in) :: global_w1(0:,0:)
  real(kind=8), intent(in) :: global_x2(0:,0:)
  real(kind=8), intent(in) :: global_w2(0:,0:)
  integer(kind=8), value :: test_u_p1
  integer(kind=8), value :: test_u_p2
  integer(kind=8), value :: n_element_1
  integer(kind=8), value :: n_element_2
  integer(kind=8), value :: k1
  integer(kind=8), value :: k2
  integer(kind=8), value :: pad1
  integer(kind=8), value :: pad2
  real(kind=8), intent(inout) :: l_el_j6duku(0:)
  real(kind=8), intent(inout) :: g_el_wzmbdw(0:)
  real(kind=8), intent(in) :: global_arr_coeffs_u(0:,0:)
  real(kind=8), allocatable :: local_x1(:)
  real(kind=8), allocatable :: local_w1(:)
  real(kind=8), allocatable :: local_x2(:)
  real(kind=8), allocatable :: local_w2(:)
  real(kind=8), allocatable :: arr_u(:,:)
  real(kind=8), allocatable :: arr_u_x2(:,:)
  real(kind=8), allocatable :: arr_u_x1(:,:)
  real(kind=8), allocatable :: arr_coeffs_u(:,:)
  integer(kind=8) :: i_element_1
  integer(kind=8) :: span_u_1
  integer(kind=8) :: i_element_2
  integer(kind=8) :: span_u_2
  integer(kind=8) :: i_basis_1
  integer(kind=8) :: i_basis_2
  real(kind=8) :: coeff_u
  integer(kind=8) :: i_quad_1
  real(kind=8) :: u_1
  real(kind=8) :: u_1_x1
  integer(kind=8) :: i_quad_2
  real(kind=8) :: u_2
  real(kind=8) :: u_2_x2
  real(kind=8) :: u
  real(kind=8) :: u_x2
  real(kind=8) :: u_x1
  real(kind=8) :: x1
  real(kind=8) :: w1
  real(kind=8) :: x2
  real(kind=8) :: w2
  real(kind=8) :: wvol_M_Square
  real(kind=8) :: temp0
  real(kind=8) :: temp1
  real(kind=8) :: temp2
  real(kind=8) :: temp3
  real(kind=8) :: temp4
  real(kind=8) :: temp5
  real(kind=8) :: temp6
  real(kind=8) :: temp7

  allocate(local_x1(0:size(global_x1, 1) - 1_8))
  local_x1 = 0.0
  allocate(local_w1(0:size(global_w1, 1) - 1_8))
  local_w1 = 0.0
  allocate(local_x2(0:size(global_x2, 1) - 1_8))
  local_x2 = 0.0
  allocate(local_w2(0:size(global_w2, 1) - 1_8))
  local_w2 = 0.0
  allocate(arr_u(0:k2 - 1_8, 0:k1 - 1_8))
  arr_u = 0.0
  allocate(arr_u_x2(0:k2 - 1_8, 0:k1 - 1_8))
  arr_u_x2 = 0.0
  allocate(arr_u_x1(0:k2 - 1_8, 0:k1 - 1_8))
  arr_u_x1 = 0.0
  allocate(arr_coeffs_u(0:1_8 + test_u_p2 - 1_8, 0:1_8 + test_u_p1 - 1_8 &
      ))
  arr_coeffs_u = 0.0
  do i_element_1 = 0_8, n_element_1-1_8, 1_8
    local_x1(:) = global_x1(:, i_element_1)
    local_w1(:) = global_w1(:, i_element_1)
    span_u_1 = global_span_u_1(i_element_1)
    do i_element_2 = 0_8, n_element_2-1_8, 1_8
      local_x2(:) = global_x2(:, i_element_2)
      local_w2(:) = global_w2(:, i_element_2)
      span_u_2 = global_span_u_2(i_element_2)
      arr_u(:, :) = 0.0d0
      arr_u_x2(:, :) = 0.0d0
      arr_u_x1(:, :) = 0.0d0
      arr_coeffs_u(:, :) = global_arr_coeffs_u(pad2 + span_u_2 - &
      test_u_p2:1_8 + pad2 + span_u_2 - 1_8, pad1 + span_u_1 - &
      test_u_p1:1_8 + pad1 + span_u_1 - 1_8)
      do i_basis_1 = 0_8, 1_8 + test_u_p1-1_8, 1_8
        do i_basis_2 = 0_8, 1_8 + test_u_p2-1_8, 1_8
          coeff_u = arr_coeffs_u(i_basis_2, i_basis_1)
          do i_quad_1 = 0_8, k1-1_8, 1_8
            u_1 = global_test_basis_u_1(i_quad_1, 0_8, i_basis_1, &
      i_element_1)
            u_1_x1 = global_test_basis_u_1(i_quad_1, 1_8, i_basis_1, &
      i_element_1)
            do i_quad_2 = 0_8, k2-1_8, 1_8
              u_2 = global_test_basis_u_2(i_quad_2, 0_8, i_basis_2, &
      i_element_2)
              u_2_x2 = global_test_basis_u_2(i_quad_2, 1_8, i_basis_2, &
      i_element_2)
              u = u_1 * u_2
              u_x2 = u_1 * u_2_x2
              u_x1 = u_1_x1 * u_2
              arr_u(i_quad_2, i_quad_1) = arr_u(i_quad_2, i_quad_1) + u &
      * coeff_u
              arr_u_x2(i_quad_2, i_quad_1) = arr_u_x2(i_quad_2, i_quad_1 &
      ) + u_x2 * coeff_u
              arr_u_x1(i_quad_2, i_quad_1) = arr_u_x1(i_quad_2, i_quad_1 &
      ) + u_x1 * coeff_u
            end do
          end do
        end do
      end do
      l_el_j6duku(0_8) = 0.0d0
      do i_quad_1 = 0_8, k1-1_8, 1_8
        x1 = local_x1(i_quad_1)
        w1 = local_w1(i_quad_1)
        do i_quad_2 = 0_8, k2-1_8, 1_8
          x2 = local_x2(i_quad_2)
          w2 = local_w2(i_quad_2)
          wvol_M_Square = w1 * w2
          u = arr_u(i_quad_2, i_quad_1)
          u_x2 = arr_u_x2(i_quad_2, i_quad_1)
          u_x1 = arr_u_x1(i_quad_2, i_quad_1)
          temp0 = 3.14159265358979d0 * x1
          temp1 = cos(temp0)
          temp2 = 3.14159265358979d0 * x2
          temp3 = sin(temp2)
          temp4 = 2_8 * 3.14159265358979d0
          temp5 = cos(temp2)
          temp6 = sin(temp0)
          temp7 = 3.14159265358979d0 ** 2_8
          l_el_j6duku(0_8) = l_el_j6duku(0_8) + wvol_M_Square * (temp1 &
      ** 2_8 * temp3 ** 2_8 * temp7 - temp1 * temp3 * temp4 * u_x1 - &
      temp4 * temp5 * temp6 * u_x2 + temp5 ** 2_8 * temp6 ** 2_8 * &
      temp7 + u_x1 ** 2_8 + u_x2 ** 2_8)
        end do
      end do
      g_el_wzmbdw(0_8) = g_el_wzmbdw(0_8) + l_el_j6duku(0_8)
    end do
  end do

end subroutine assembly
!........................................

end module dependencies_tnscifz8
