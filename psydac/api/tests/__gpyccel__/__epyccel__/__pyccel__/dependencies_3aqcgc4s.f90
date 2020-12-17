module dependencies_3aqcgc4s

implicit none

contains

!........................................
subroutine assembly(global_test_basis_v1_1, global_trial_basis_u1_1, &
      global_span_v1_1, global_x1, global_w1, test_v1_p1, trial_u1_p1, &
      n_element_1, k1, pad1, l_mat_u1_v1_3aqcgc4s, g_mat_u1_v1_3aqcgc4s &
      )

  implicit none

  real(kind=8), intent(in) :: global_test_basis_v1_1(0:,0:,0:,0:)
  real(kind=8), intent(in) :: global_trial_basis_u1_1(0:,0:,0:,0:)
  integer(kind=8), intent(in) :: global_span_v1_1(0:)
  real(kind=8), intent(in) :: global_x1(0:,0:)
  real(kind=8), intent(in) :: global_w1(0:,0:)
  integer(kind=8), value :: test_v1_p1
  integer(kind=8), value :: trial_u1_p1
  integer(kind=8), value :: n_element_1
  integer(kind=8), value :: k1
  integer(kind=8), value :: pad1
  real(kind=8), intent(inout) :: l_mat_u1_v1_3aqcgc4s(0:,0:)
  real(kind=8), intent(inout) :: g_mat_u1_v1_3aqcgc4s(0:,0:)
  integer(kind=8) :: pad_u1_v1_1
  real(kind=8), allocatable :: local_x1(:)
  real(kind=8), allocatable :: local_w1(:)
  integer(kind=8) :: i_element_1
  integer(kind=8) :: span_v1_1
  integer(kind=8) :: i_basis_1
  integer(kind=8) :: j_basis_1
  integer(kind=8) :: i_quad_1
  real(kind=8) :: x1
  real(kind=8) :: w1
  real(kind=8) :: v1_1
  real(kind=8) :: v1_1_x1
  real(kind=8) :: u1_1
  real(kind=8) :: u1_1_x1
  real(kind=8) :: wvol_M
  real(kind=8) :: v1
  real(kind=8) :: v1_x1
  real(kind=8) :: u1
  real(kind=8) :: u1_x1
  real(kind=8) :: temp0

  pad_u1_v1_1 = maxval([test_v1_p1, trial_u1_p1])
  allocate(local_x1(0:size(global_x1, 1) - 1_8))
  local_x1 = 0.0
  allocate(local_w1(0:size(global_w1, 1) - 1_8))
  local_w1 = 0.0
  do i_element_1 = 0_8, n_element_1-1_8, 1_8
    local_x1(:) = global_x1(:, i_element_1)
    local_w1(:) = global_w1(:, i_element_1)
    span_v1_1 = global_span_v1_1(i_element_1)
    l_mat_u1_v1_3aqcgc4s(:, :) = 0.0d0
    do i_basis_1 = 0_8, 1_8 + test_v1_p1-1_8, 1_8
      do j_basis_1 = 0_8, 1_8 + trial_u1_p1-1_8, 1_8
        do i_quad_1 = 0_8, 3_8-1_8, 1_8
          x1 = local_x1(i_quad_1)
          w1 = local_w1(i_quad_1)
          v1_1 = global_test_basis_v1_1(i_quad_1, 0_8, i_basis_1, &
      i_element_1)
          v1_1_x1 = global_test_basis_v1_1(i_quad_1, 1_8, i_basis_1, &
      i_element_1)
          u1_1 = global_trial_basis_u1_1(i_quad_1, 0_8, j_basis_1, &
      i_element_1)
          u1_1_x1 = global_trial_basis_u1_1(i_quad_1, 1_8, j_basis_1, &
      i_element_1)
          wvol_M = w1
          v1 = v1_1
          v1_x1 = v1_1_x1
          u1 = u1_1
          u1_x1 = u1_1_x1
          temp0 = (0.25d0 * cos(2_8 * 3.14159265358979d0 * x1) + 1.0d0) &
      ** 2_8
          l_mat_u1_v1_3aqcgc4s(-i_basis_1 + j_basis_1 + pad_u1_v1_1, &
      i_basis_1) = l_mat_u1_v1_3aqcgc4s(-i_basis_1 + j_basis_1 + &
      pad_u1_v1_1, i_basis_1) + 1.0d0 * u1 * v1 * wvol_M / sqrt(temp0)
        end do
      end do
    end do
    g_mat_u1_v1_3aqcgc4s(:, pad1 + span_v1_1 - test_v1_p1:1_8 + pad1 + &
      span_v1_1 - 1_8) = g_mat_u1_v1_3aqcgc4s(:, pad1 + span_v1_1 - &
      test_v1_p1:1_8 + pad1 + span_v1_1 - 1_8) + l_mat_u1_v1_3aqcgc4s( &
      :, :)
  end do

end subroutine assembly
!........................................

end module dependencies_3aqcgc4s
