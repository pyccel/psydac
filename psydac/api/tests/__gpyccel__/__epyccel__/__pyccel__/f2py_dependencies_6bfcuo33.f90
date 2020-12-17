!........................................
subroutine assembly(n0_global_test_basis_v0_1, n1_global_test_basis_v0_1 &
      , n2_global_test_basis_v0_1, n3_global_test_basis_v0_1, &
      global_test_basis_v0_1, n0_global_trial_basis_u0_1, &
      n1_global_trial_basis_u0_1, n2_global_trial_basis_u0_1, &
      n3_global_trial_basis_u0_1, global_trial_basis_u0_1, &
      n0_global_span_v0_1, global_span_v0_1, n0_global_x1, n1_global_x1 &
      , global_x1, n0_global_w1, n1_global_w1, global_w1, test_v0_p1, &
      trial_u0_p1, n_element_1, k1, pad1, n0_l_mat_u0_v0_6bfcuo33, &
      n1_l_mat_u0_v0_6bfcuo33, l_mat_u0_v0_6bfcuo33, &
      n0_g_mat_u0_v0_6bfcuo33, n1_g_mat_u0_v0_6bfcuo33, &
      g_mat_u0_v0_6bfcuo33)

  use dependencies_6bfcuo33, only: mod_assembly => assembly

  implicit none

  integer(kind=4), intent(in) :: n0_global_test_basis_v0_1
  integer(kind=4), intent(in) :: n1_global_test_basis_v0_1
  integer(kind=4), intent(in) :: n2_global_test_basis_v0_1
  integer(kind=4), intent(in) :: n3_global_test_basis_v0_1
  real(kind=8), intent(in) :: global_test_basis_v0_1(0: &
      n3_global_test_basis_v0_1-1,0:n2_global_test_basis_v0_1-1,0: &
      n1_global_test_basis_v0_1-1,0:n0_global_test_basis_v0_1-1)
  integer(kind=4), intent(in) :: n0_global_trial_basis_u0_1
  integer(kind=4), intent(in) :: n1_global_trial_basis_u0_1
  integer(kind=4), intent(in) :: n2_global_trial_basis_u0_1
  integer(kind=4), intent(in) :: n3_global_trial_basis_u0_1
  real(kind=8), intent(in) :: global_trial_basis_u0_1(0: &
      n3_global_trial_basis_u0_1-1,0:n2_global_trial_basis_u0_1-1,0: &
      n1_global_trial_basis_u0_1-1,0:n0_global_trial_basis_u0_1-1)
  integer(kind=4), intent(in) :: n0_global_span_v0_1
  integer(kind=8), intent(in) :: global_span_v0_1(0: &
      n0_global_span_v0_1-1)
  integer(kind=4), intent(in) :: n0_global_x1
  integer(kind=4), intent(in) :: n1_global_x1
  real(kind=8), intent(in) :: global_x1(0:n1_global_x1-1,0: &
      n0_global_x1-1)
  integer(kind=4), intent(in) :: n0_global_w1
  integer(kind=4), intent(in) :: n1_global_w1
  real(kind=8), intent(in) :: global_w1(0:n1_global_w1-1,0: &
      n0_global_w1-1)
  integer(kind=8), intent(in) :: test_v0_p1
  integer(kind=8), intent(in) :: trial_u0_p1
  integer(kind=8), intent(in) :: n_element_1
  integer(kind=8), intent(in) :: k1
  integer(kind=8), intent(in) :: pad1
  integer(kind=4), intent(in) :: n0_l_mat_u0_v0_6bfcuo33
  integer(kind=4), intent(in) :: n1_l_mat_u0_v0_6bfcuo33
  real(kind=8), intent(inout) :: l_mat_u0_v0_6bfcuo33(0: &
      n1_l_mat_u0_v0_6bfcuo33-1,0:n0_l_mat_u0_v0_6bfcuo33-1)
  integer(kind=4), intent(in) :: n0_g_mat_u0_v0_6bfcuo33
  integer(kind=4), intent(in) :: n1_g_mat_u0_v0_6bfcuo33
  real(kind=8), intent(inout) :: g_mat_u0_v0_6bfcuo33(0: &
      n1_g_mat_u0_v0_6bfcuo33-1,0:n0_g_mat_u0_v0_6bfcuo33-1)

  !f2py integer(kind=8) :: n0_global_test_basis_v0_1=shape(global_test_basis_v0_1,0)
  !f2py integer(kind=8) :: n1_global_test_basis_v0_1=shape(global_test_basis_v0_1,1)
  !f2py integer(kind=8) :: n2_global_test_basis_v0_1=shape(global_test_basis_v0_1,2)
  !f2py integer(kind=8) :: n3_global_test_basis_v0_1=shape(global_test_basis_v0_1,3)
  !f2py intent(c) global_test_basis_v0_1
  !f2py integer(kind=8) :: n0_global_trial_basis_u0_1=shape(global_trial_basis_u0_1,0)
  !f2py integer(kind=8) :: n1_global_trial_basis_u0_1=shape(global_trial_basis_u0_1,1)
  !f2py integer(kind=8) :: n2_global_trial_basis_u0_1=shape(global_trial_basis_u0_1,2)
  !f2py integer(kind=8) :: n3_global_trial_basis_u0_1=shape(global_trial_basis_u0_1,3)
  !f2py intent(c) global_trial_basis_u0_1
  !f2py integer(kind=8) :: n0_global_x1=shape(global_x1,0)
  !f2py integer(kind=8) :: n1_global_x1=shape(global_x1,1)
  !f2py intent(c) global_x1
  !f2py integer(kind=8) :: n0_global_w1=shape(global_w1,0)
  !f2py integer(kind=8) :: n1_global_w1=shape(global_w1,1)
  !f2py intent(c) global_w1
  !f2py integer(kind=8) :: n0_l_mat_u0_v0_6bfcuo33=shape(l_mat_u0_v0_6bfcuo33,0)
  !f2py integer(kind=8) :: n1_l_mat_u0_v0_6bfcuo33=shape(l_mat_u0_v0_6bfcuo33,1)
  !f2py intent(c) l_mat_u0_v0_6bfcuo33
  !f2py integer(kind=8) :: n0_g_mat_u0_v0_6bfcuo33=shape(g_mat_u0_v0_6bfcuo33,0)
  !f2py integer(kind=8) :: n1_g_mat_u0_v0_6bfcuo33=shape(g_mat_u0_v0_6bfcuo33,1)
  !f2py intent(c) g_mat_u0_v0_6bfcuo33
  call mod_assembly(global_test_basis_v0_1, global_trial_basis_u0_1, &
      global_span_v0_1, global_x1, global_w1, test_v0_p1, trial_u0_p1, &
      n_element_1, k1, pad1, l_mat_u0_v0_6bfcuo33, g_mat_u0_v0_6bfcuo33 &
      )

end subroutine assembly
!........................................
