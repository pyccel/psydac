!........................................
subroutine assembly(n0_global_test_basis_u_1, n1_global_test_basis_u_1, &
      n2_global_test_basis_u_1, n3_global_test_basis_u_1, &
      global_test_basis_u_1, n0_global_test_basis_u_2, &
      n1_global_test_basis_u_2, n2_global_test_basis_u_2, &
      n3_global_test_basis_u_2, global_test_basis_u_2, &
      n0_global_span_u_1, global_span_u_1, n0_global_span_u_2, &
      global_span_u_2, n0_global_x1, n1_global_x1, global_x1, &
      n0_global_w1, n1_global_w1, global_w1, n0_global_x2, n1_global_x2 &
      , global_x2, n0_global_w2, n1_global_w2, global_w2, test_u_p1, &
      test_u_p2, n_element_1, n_element_2, k1, k2, pad1, pad2, &
      n0_l_el_j6duku, l_el_j6duku, n0_g_el_wzmbdw, g_el_wzmbdw, &
      n0_global_arr_coeffs_u, n1_global_arr_coeffs_u, &
      global_arr_coeffs_u)

  use dependencies_tnscifz8, only: mod_assembly => assembly

  implicit none

  integer(kind=4), intent(in) :: n0_global_test_basis_u_1
  integer(kind=4), intent(in) :: n1_global_test_basis_u_1
  integer(kind=4), intent(in) :: n2_global_test_basis_u_1
  integer(kind=4), intent(in) :: n3_global_test_basis_u_1
  real(kind=8), intent(in) :: global_test_basis_u_1(0: &
      n3_global_test_basis_u_1-1,0:n2_global_test_basis_u_1-1,0: &
      n1_global_test_basis_u_1-1,0:n0_global_test_basis_u_1-1)
  integer(kind=4), intent(in) :: n0_global_test_basis_u_2
  integer(kind=4), intent(in) :: n1_global_test_basis_u_2
  integer(kind=4), intent(in) :: n2_global_test_basis_u_2
  integer(kind=4), intent(in) :: n3_global_test_basis_u_2
  real(kind=8), intent(in) :: global_test_basis_u_2(0: &
      n3_global_test_basis_u_2-1,0:n2_global_test_basis_u_2-1,0: &
      n1_global_test_basis_u_2-1,0:n0_global_test_basis_u_2-1)
  integer(kind=4), intent(in) :: n0_global_span_u_1
  integer(kind=8), intent(in) :: global_span_u_1(0:n0_global_span_u_1-1)
  integer(kind=4), intent(in) :: n0_global_span_u_2
  integer(kind=8), intent(in) :: global_span_u_2(0:n0_global_span_u_2-1)
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
  integer(kind=8), intent(in) :: test_u_p1
  integer(kind=8), intent(in) :: test_u_p2
  integer(kind=8), intent(in) :: n_element_1
  integer(kind=8), intent(in) :: n_element_2
  integer(kind=8), intent(in) :: k1
  integer(kind=8), intent(in) :: k2
  integer(kind=8), intent(in) :: pad1
  integer(kind=8), intent(in) :: pad2
  integer(kind=4), intent(in) :: n0_l_el_j6duku
  real(kind=8), intent(inout) :: l_el_j6duku(0:n0_l_el_j6duku-1)
  integer(kind=4), intent(in) :: n0_g_el_wzmbdw
  real(kind=8), intent(inout) :: g_el_wzmbdw(0:n0_g_el_wzmbdw-1)
  integer(kind=4), intent(in) :: n0_global_arr_coeffs_u
  integer(kind=4), intent(in) :: n1_global_arr_coeffs_u
  real(kind=8), intent(in) :: global_arr_coeffs_u(0: &
      n1_global_arr_coeffs_u-1,0:n0_global_arr_coeffs_u-1)

  !f2py integer(kind=8) :: n0_global_test_basis_u_1=shape(global_test_basis_u_1,0)
  !f2py integer(kind=8) :: n1_global_test_basis_u_1=shape(global_test_basis_u_1,1)
  !f2py integer(kind=8) :: n2_global_test_basis_u_1=shape(global_test_basis_u_1,2)
  !f2py integer(kind=8) :: n3_global_test_basis_u_1=shape(global_test_basis_u_1,3)
  !f2py intent(c) global_test_basis_u_1
  !f2py integer(kind=8) :: n0_global_test_basis_u_2=shape(global_test_basis_u_2,0)
  !f2py integer(kind=8) :: n1_global_test_basis_u_2=shape(global_test_basis_u_2,1)
  !f2py integer(kind=8) :: n2_global_test_basis_u_2=shape(global_test_basis_u_2,2)
  !f2py integer(kind=8) :: n3_global_test_basis_u_2=shape(global_test_basis_u_2,3)
  !f2py intent(c) global_test_basis_u_2
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
  !f2py integer(kind=8) :: n0_global_arr_coeffs_u=shape(global_arr_coeffs_u,0)
  !f2py integer(kind=8) :: n1_global_arr_coeffs_u=shape(global_arr_coeffs_u,1)
  !f2py intent(c) global_arr_coeffs_u
  call mod_assembly(global_test_basis_u_1, global_test_basis_u_2, &
      global_span_u_1, global_span_u_2, global_x1, global_w1, global_x2 &
      , global_w2, test_u_p1, test_u_p2, n_element_1, n_element_2, k1, &
      k2, pad1, pad2, l_el_j6duku, g_el_wzmbdw, global_arr_coeffs_u)

end subroutine assembly
!........................................
