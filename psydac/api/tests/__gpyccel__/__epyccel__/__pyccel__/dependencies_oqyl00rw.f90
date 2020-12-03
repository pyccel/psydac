module dependencies_oqyl00rw

implicit none

contains

!........................................
subroutine assembly(global_test_basis_v_1, global_test_basis_v_2, &
      global_trial_basis_u_1, global_trial_basis_u_2, global_span_v_1, &
      global_span_v_2, global_x1, global_w1, global_x2, global_w2, &
      test_v_p1, test_v_p2, trial_u_p1, trial_u_p2, n_element_1, &
      n_element_2, k1, k2, pad1, pad2, l_mat_u_v_oqyl00rw, &
      g_mat_u_v_oqyl00rw)

  implicit none

  real(kind=8), intent(in) :: global_test_basis_v_1(0:,0:,0:,0:)
  real(kind=8), intent(in) :: global_test_basis_v_2(0:,0:,0:,0:)
  real(kind=8), intent(in) :: global_trial_basis_u_1(0:,0:,0:,0:)
  real(kind=8), intent(in) :: global_trial_basis_u_2(0:,0:,0:,0:)
  integer(kind=8), intent(in) :: global_span_v_1(0:)
  integer(kind=8), intent(in) :: global_span_v_2(0:)
  real(kind=8), intent(in) :: global_x1(0:,0:)
  real(kind=8), intent(in) :: global_w1(0:,0:)
  real(kind=8), intent(in) :: global_x2(0:,0:)
  real(kind=8), intent(in) :: global_w2(0:,0:)
  integer(kind=8), value :: test_v_p1
  integer(kind=8), value :: test_v_p2
  integer(kind=8), value :: trial_u_p1
  integer(kind=8), value :: trial_u_p2
  integer(kind=8), value :: n_element_1
  integer(kind=8), value :: n_element_2
  integer(kind=8), value :: k1
  integer(kind=8), value :: k2
  integer(kind=8), value :: pad1
  integer(kind=8), value :: pad2
  real(kind=8), intent(inout) :: l_mat_u_v_oqyl00rw(0:,0:,0:,0:)
  real(kind=8), intent(inout) :: g_mat_u_v_oqyl00rw(0:,0:,0:,0:)
  integer(kind=8) :: pad_u_v_1
  integer(kind=8) :: pad_u_v_2
  real(kind=8), allocatable :: local_x1(:)
  real(kind=8), allocatable :: local_w1(:)
  real(kind=8), allocatable :: local_x2(:)
  real(kind=8), allocatable :: local_w2(:)
  integer(kind=8) :: i_element_1
  integer(kind=8) :: span_v_1
  integer(kind=8) :: i_element_2
  integer(kind=8) :: span_v_2
  integer(kind=8) :: i_basis_1
  integer(kind=8) :: i_basis_2
  integer(kind=8) :: j_basis_1
  integer(kind=8) :: j_basis_2
  integer(kind=8) :: i_quad_1
  real(kind=8) :: x1
  real(kind=8) :: w1
  real(kind=8) :: v_1
  real(kind=8) :: v_1_x1
  real(kind=8) :: u_1
  real(kind=8) :: u_1_x1
  integer(kind=8) :: i_quad_2
  real(kind=8) :: x2
  real(kind=8) :: w2
  real(kind=8) :: v_2
  real(kind=8) :: v_2_x2
  real(kind=8) :: u_2
  real(kind=8) :: u_2_x2
  real(kind=8) :: wvol_M_Square
  real(kind=8) :: v
  real(kind=8) :: v_x2
  real(kind=8) :: v_x1
  real(kind=8) :: u
  real(kind=8) :: u_x2
  real(kind=8) :: u_x1

  pad_u_v_1 = maxval([test_v_p1, trial_u_p1])
  pad_u_v_2 = maxval([test_v_p2, trial_u_p2])
  allocate(local_x1(0:size(global_x1, 1) - 1_8))
  local_x1 = 0.0
  allocate(local_w1(0:size(global_w1, 1) - 1_8))
  local_w1 = 0.0
  allocate(local_x2(0:size(global_x2, 1) - 1_8))
  local_x2 = 0.0
  allocate(local_w2(0:size(global_w2, 1) - 1_8))
  local_w2 = 0.0
  do i_element_1 = 0_8, n_element_1-1_8, 1_8
    local_x1(:) = global_x1(:, i_element_1)
    local_w1(:) = global_w1(:, i_element_1)
    span_v_1 = global_span_v_1(i_element_1)
    do i_element_2 = 0_8, n_element_2-1_8, 1_8
      local_x2(:) = global_x2(:, i_element_2)
      local_w2(:) = global_w2(:, i_element_2)
      span_v_2 = global_span_v_2(i_element_2)
      l_mat_u_v_oqyl00rw(:, :, :, :) = 0.0d0
      do i_basis_1 = 0_8, 1_8 + test_v_p1-1_8, 1_8
        do i_basis_2 = 0_8, 1_8 + test_v_p2-1_8, 1_8
          do j_basis_1 = 0_8, 1_8 + trial_u_p1-1_8, 1_8
            do j_basis_2 = 0_8, 1_8 + trial_u_p2-1_8, 1_8
              do i_quad_1 = 0_8, 3_8-1_8, 1_8
                x1 = local_x1(i_quad_1)
                w1 = local_w1(i_quad_1)
                v_1 = global_test_basis_v_1(i_quad_1, 0_8, i_basis_1, &
      i_element_1)
                v_1_x1 = global_test_basis_v_1(i_quad_1, 1_8, i_basis_1, &
      i_element_1)
                u_1 = global_trial_basis_u_1(i_quad_1, 0_8, j_basis_1, &
      i_element_1)
                u_1_x1 = global_trial_basis_u_1(i_quad_1, 1_8, j_basis_1 &
      , i_element_1)
                do i_quad_2 = 0_8, 3_8-1_8, 1_8
                  x2 = local_x2(i_quad_2)
                  w2 = local_w2(i_quad_2)
                  v_2 = global_test_basis_v_2(i_quad_2, 0_8, i_basis_2, &
      i_element_2)
                  v_2_x2 = global_test_basis_v_2(i_quad_2, 1_8, &
      i_basis_2, i_element_2)
                  u_2 = global_trial_basis_u_2(i_quad_2, 0_8, j_basis_2, &
      i_element_2)
                  u_2_x2 = global_trial_basis_u_2(i_quad_2, 1_8, &
      j_basis_2, i_element_2)
                  wvol_M_Square = w1 * w2
                  v = v_1 * v_2
                  v_x2 = v_1 * v_2_x2
                  v_x1 = v_1_x1 * v_2
                  u = u_1 * u_2
                  u_x2 = u_1 * u_2_x2
                  u_x1 = u_1_x1 * u_2
                  l_mat_u_v_oqyl00rw(-i_basis_2 + j_basis_2 + pad_u_v_2, &
      -i_basis_1 + j_basis_1 + pad_u_v_1, i_basis_2, i_basis_1) = &
      l_mat_u_v_oqyl00rw(-i_basis_2 + j_basis_2 + pad_u_v_2, -i_basis_1 &
      + j_basis_1 + pad_u_v_1, i_basis_2, i_basis_1) + wvol_M_Square * &
      (u_x1 * v_x1 + u_x2 * v_x2)
                end do
              end do
            end do
          end do
        end do
      end do
      g_mat_u_v_oqyl00rw(:, :, pad2 + span_v_2 - test_v_p2:1_8 + pad2 + &
      span_v_2 - 1_8, pad1 + span_v_1 - test_v_p1:1_8 + pad1 + span_v_1 &
      - 1_8) = g_mat_u_v_oqyl00rw(:, :, pad2 + span_v_2 - test_v_p2:1_8 &
      + pad2 + span_v_2 - 1_8, pad1 + span_v_1 - test_v_p1:1_8 + pad1 + &
      span_v_1 - 1_8) + l_mat_u_v_oqyl00rw(:, :, :, :)
    end do
  end do

end subroutine assembly
!........................................

end module dependencies_oqyl00rw
