!........................................
subroutine assembly(n0_global_test_basis_v_1, n1_global_test_basis_v_1, &
      n2_global_test_basis_v_1, n3_global_test_basis_v_1, &
      global_test_basis_v_1, n0_global_test_basis_v_2, &
      n1_global_test_basis_v_2, n2_global_test_basis_v_2, &
      n3_global_test_basis_v_2, global_test_basis_v_2, &
      n0_global_trial_basis_u_1, n1_global_trial_basis_u_1, &
      n2_global_trial_basis_u_1, n3_global_trial_basis_u_1, &
      global_trial_basis_u_1, n0_global_trial_basis_u_2, &
      n1_global_trial_basis_u_2, n2_global_trial_basis_u_2, &
      n3_global_trial_basis_u_2, global_trial_basis_u_2, &
      n0_global_span_v_1, global_span_v_1, n0_global_span_v_2, &
      global_span_v_2, n0_global_x1, n1_global_x1, global_x1, &
      n0_global_w1, n1_global_w1, global_w1, n0_global_x2, n1_global_x2 &
      , global_x2, n0_global_w2, n1_global_w2, global_w2, test_v_p1, &
      test_v_p2, trial_u_p1, trial_u_p2, n_element_1, n_element_2, k1, &
      k2, pad1, pad2, n0_l_mat_u_v_hy9im0tq, n1_l_mat_u_v_hy9im0tq, &
      n2_l_mat_u_v_hy9im0tq, n3_l_mat_u_v_hy9im0tq, l_mat_u_v_hy9im0tq, &
      n0_g_mat_u_v_hy9im0tq, n1_g_mat_u_v_hy9im0tq, &
      n2_g_mat_u_v_hy9im0tq, n3_g_mat_u_v_hy9im0tq, g_mat_u_v_hy9im0tq &
      )

  use dependencies_hy9im0tq, only: mod_assembly => assembly

  implicit none

  integer(kind=4), intent(in) :: n0_global_test_basis_v_1
  integer(kind=4), intent(in) :: n1_global_test_basis_v_1
  integer(kind=4), intent(in) :: n2_global_test_basis_v_1
  integer(kind=4), intent(in) :: n3_global_test_basis_v_1
  real(kind=8), intent(in) :: global_test_basis_v_1(0: &
      n3_global_test_basis_v_1-1,0:n2_global_test_basis_v_1-1,0: &
      n1_global_test_basis_v_1-1,0:n0_global_test_basis_v_1-1)
  integer(kind=4), intent(in) :: n0_global_test_basis_v_2
  integer(kind=4), intent(in) :: n1_global_test_basis_v_2
  integer(kind=4), intent(in) :: n2_global_test_basis_v_2
  integer(kind=4), intent(in) :: n3_global_test_basis_v_2
  real(kind=8), intent(in) :: global_test_basis_v_2(0: &
      n3_global_test_basis_v_2-1,0:n2_global_test_basis_v_2-1,0: &
      n1_global_test_basis_v_2-1,0:n0_global_test_basis_v_2-1)
  integer(kind=4), intent(in) :: n0_global_trial_basis_u_1
  integer(kind=4), intent(in) :: n1_global_trial_basis_u_1
  integer(kind=4), intent(in) :: n2_global_trial_basis_u_1
  integer(kind=4), intent(in) :: n3_global_trial_basis_u_1
  real(kind=8), intent(in) :: global_trial_basis_u_1(0: &
      n3_global_trial_basis_u_1-1,0:n2_global_trial_basis_u_1-1,0: &
      n1_global_trial_basis_u_1-1,0:n0_global_trial_basis_u_1-1)
  integer(kind=4), intent(in) :: n0_global_trial_basis_u_2
  integer(kind=4), intent(in) :: n1_global_trial_basis_u_2
  integer(kind=4), intent(in) :: n2_global_trial_basis_u_2
  integer(kind=4), intent(in) :: n3_global_trial_basis_u_2
  real(kind=8), intent(in) :: global_trial_basis_u_2(0: &
      n3_global_trial_basis_u_2-1,0:n2_global_trial_basis_u_2-1,0: &
      n1_global_trial_basis_u_2-1,0:n0_global_trial_basis_u_2-1)
  integer(kind=4), intent(in) :: n0_global_span_v_1
  integer(kind=8), intent(in) :: global_span_v_1(0:n0_global_span_v_1-1)
  integer(kind=4), intent(in) :: n0_global_span_v_2
  integer(kind=8), intent(in) :: global_span_v_2(0:n0_global_span_v_2-1)
  integer(kind=4), intent(in) :: n0_global_x1
  integer(kind=4), intent(in) :: n1_global_x1
  real(kind=8), intent(in) :: global_x1(0:n1_global_x1-1,0: &
      n0_global_x1-1)
  integer(kind=4), intent(in) :: n0_global_w1
  integer(kind=4), intent(in) :: n1_global_w1
  real(kind=8), intent(in) :: global_w1(0:n1_global_w1-1,0: &
      n0_global_w1-1)
  integer(kind=4), intent(in) :: n0_global_x2
  integer(kind=4), intent(in) :: n1_global_x2
  real(kind=8), intent(in) :: global_x2(0:n1_global_x2-1,0: &
      n0_global_x2-1)
  integer(kind=4), intent(in) :: n0_global_w2
  integer(kind=4), intent(in) :: n1_global_w2
  real(kind=8), intent(in) :: global_w2(0:n1_global_w2-1,0: &
      n0_global_w2-1)
  integer(kind=8), intent(in) :: test_v_p1
  integer(kind=8), intent(in) :: test_v_p2
  integer(kind=8), intent(in) :: trial_u_p1
  integer(kind=8), intent(in) :: trial_u_p2
  integer(kind=8), intent(in) :: n_element_1
  integer(kind=8), intent(in) :: n_element_2
  integer(kind=8), intent(in) :: k1
  integer(kind=8), intent(in) :: k2
  integer(kind=8), intent(in) :: pad1
  integer(kind=8), intent(in) :: pad2
  integer(kind=4), intent(in) :: n0_l_mat_u_v_hy9im0tq
  integer(kind=4), intent(in) :: n1_l_mat_u_v_hy9im0tq
  integer(kind=4), intent(in) :: n2_l_mat_u_v_hy9im0tq
  integer(kind=4), intent(in) :: n3_l_mat_u_v_hy9im0tq
  real(kind=8), intent(inout) :: l_mat_u_v_hy9im0tq(0: &
      n3_l_mat_u_v_hy9im0tq-1,0:n2_l_mat_u_v_hy9im0tq-1,0: &
      n1_l_mat_u_v_hy9im0tq-1,0:n0_l_mat_u_v_hy9im0tq-1)
  integer(kind=4), intent(in) :: n0_g_mat_u_v_hy9im0tq
  integer(kind=4), intent(in) :: n1_g_mat_u_v_hy9im0tq
  integer(kind=4), intent(in) :: n2_g_mat_u_v_hy9im0tq
  integer(kind=4), intent(in) :: n3_g_mat_u_v_hy9im0tq
  real(kind=8), intent(inout) :: g_mat_u_v_hy9im0tq(0: &
      n3_g_mat_u_v_hy9im0tq-1,0:n2_g_mat_u_v_hy9im0tq-1,0: &
      n1_g_mat_u_v_hy9im0tq-1,0:n0_g_mat_u_v_hy9im0tq-1)

  !f2py integer(kind=8) :: n0_global_test_basis_v_1=shape(global_test_basis_v_1,0)
  !f2py integer(kind=8) :: n1_global_test_basis_v_1=shape(global_test_basis_v_1,1)
  !f2py integer(kind=8) :: n2_global_test_basis_v_1=shape(global_test_basis_v_1,2)
  !f2py integer(kind=8) :: n3_global_test_basis_v_1=shape(global_test_basis_v_1,3)
  !f2py intent(c) global_test_basis_v_1
  !f2py integer(kind=8) :: n0_global_test_basis_v_2=shape(global_test_basis_v_2,0)
  !f2py integer(kind=8) :: n1_global_test_basis_v_2=shape(global_test_basis_v_2,1)
  !f2py integer(kind=8) :: n2_global_test_basis_v_2=shape(global_test_basis_v_2,2)
  !f2py integer(kind=8) :: n3_global_test_basis_v_2=shape(global_test_basis_v_2,3)
  !f2py intent(c) global_test_basis_v_2
  !f2py integer(kind=8) :: n0_global_trial_basis_u_1=shape(global_trial_basis_u_1,0)
  !f2py integer(kind=8) :: n1_global_trial_basis_u_1=shape(global_trial_basis_u_1,1)
  !f2py integer(kind=8) :: n2_global_trial_basis_u_1=shape(global_trial_basis_u_1,2)
  !f2py integer(kind=8) :: n3_global_trial_basis_u_1=shape(global_trial_basis_u_1,3)
  !f2py intent(c) global_trial_basis_u_1
  !f2py integer(kind=8) :: n0_global_trial_basis_u_2=shape(global_trial_basis_u_2,0)
  !f2py integer(kind=8) :: n1_global_trial_basis_u_2=shape(global_trial_basis_u_2,1)
  !f2py integer(kind=8) :: n2_global_trial_basis_u_2=shape(global_trial_basis_u_2,2)
  !f2py integer(kind=8) :: n3_global_trial_basis_u_2=shape(global_trial_basis_u_2,3)
  !f2py intent(c) global_trial_basis_u_2
  !f2py integer(kind=8) :: n0_global_x1=shape(global_x1,0)
  !f2py integer(kind=8) :: n1_global_x1=shape(global_x1,1)
  !f2py intent(c) global_x1
  !f2py integer(kind=8) :: n0_global_w1=shape(global_w1,0)
  !f2py integer(kind=8) :: n1_global_w1=shape(global_w1,1)
  !f2py intent(c) global_w1
  !f2py integer(kind=8) :: n0_global_x2=shape(global_x2,0)
  !f2py integer(kind=8) :: n1_global_x2=shape(global_x2,1)
  !f2py intent(c) global_x2
  !f2py integer(kind=8) :: n0_global_w2=shape(global_w2,0)
  !f2py integer(kind=8) :: n1_global_w2=shape(global_w2,1)
  !f2py intent(c) global_w2
  !f2py integer(kind=8) :: n0_l_mat_u_v_hy9im0tq=shape(l_mat_u_v_hy9im0tq,0)
  !f2py integer(kind=8) :: n1_l_mat_u_v_hy9im0tq=shape(l_mat_u_v_hy9im0tq,1)
  !f2py integer(kind=8) :: n2_l_mat_u_v_hy9im0tq=shape(l_mat_u_v_hy9im0tq,2)
  !f2py integer(kind=8) :: n3_l_mat_u_v_hy9im0tq=shape(l_mat_u_v_hy9im0tq,3)
  !f2py intent(c) l_mat_u_v_hy9im0tq
  !f2py integer(kind=8) :: n0_g_mat_u_v_hy9im0tq=shape(g_mat_u_v_hy9im0tq,0)
  !f2py integer(kind=8) :: n1_g_mat_u_v_hy9im0tq=shape(g_mat_u_v_hy9im0tq,1)
  !f2py integer(kind=8) :: n2_g_mat_u_v_hy9im0tq=shape(g_mat_u_v_hy9im0tq,2)
  !f2py integer(kind=8) :: n3_g_mat_u_v_hy9im0tq=shape(g_mat_u_v_hy9im0tq,3)
  !f2py intent(c) g_mat_u_v_hy9im0tq
  call mod_assembly(global_test_basis_v_1, global_test_basis_v_2, &
      global_trial_basis_u_1, global_trial_basis_u_2, global_span_v_1, &
      global_span_v_2, global_x1, global_w1, global_x2, global_w2, &
      test_v_p1, test_v_p2, trial_u_p1, trial_u_p2, n_element_1, &
      n_element_2, k1, k2, pad1, pad2, l_mat_u_v_hy9im0tq, &
      g_mat_u_v_hy9im0tq)

end subroutine assembly
!........................................
