!> @brief 
!> module containing subroutines for calculus and differential geometry 
module spl_m_calculus

use spl_m_global

implicit none

  private
  public  :: spl_compute_cross_product_2d, &
           & spl_compute_cross_product_3d, &
           & spl_compute_determinants_jacobians_1d, &
           & spl_compute_determinants_jacobians_2d, &
           & spl_compute_determinants_jacobians_3d, &
           & spl_compute_inv_hessian_1d, &
           & spl_compute_inv_hessian_2d, &
           & spl_compute_inv_hessian_3d, &
           & spl_compute_inv_jacobians_1d, &
           & spl_compute_inv_jacobians_2d, &
           & spl_compute_inv_jacobians_3d, &
           & spl_compute_pullback_one_form_2d, &
           & spl_compute_pullback_one_form_3d, &
           & spl_compute_pullback_two_form_2d, &
           & spl_compute_pullback_two_form_3d, &
           & spl_map_second_derivate_2d, &
           & spl_map_second_derivate_3d, &
           & spl_map_vector_2d, &
           & spl_map_vector_3d

contains

  ! ..................................................
  !> @brief    computes the cross product of 2d vectors 
  !>
  !> @param[in]    a       array of size 2 
  !> @param[in]    b       array of size 2 
  !> @param[inout] r       scalar that corresponds to the cross product r = a x b 
  subroutine spl_compute_cross_product_2d(a, b, r)
  implicit none
    real(kind=spl_rk), dimension(2), intent(in)    :: a
    real(kind=spl_rk), dimension(2), intent(in)    :: b 
    real(kind=spl_rk),               intent(inout) :: r 

    r = a(1) * b(2) - a(2) * b(1)

  end subroutine spl_compute_cross_product_2d
  ! ..................................................

  ! ..................................................
  !> @brief    computes the cross product of 3d vectors 
  !>
  !> @param[in]    a       array of size 3 
  !> @param[in]    b       array of size 3 
  !> @param[inout] r       array that corresponds to the cross product r = a x b 
  subroutine spl_compute_cross_product_3d(a, b, r)
  implicit none
    real(kind=spl_rk), dimension(3), intent(in)    :: a
    real(kind=spl_rk), dimension(3), intent(in)    :: b 
    real(kind=spl_rk), dimension(3), intent(inout) :: r 

    r(1) = a(2) * b(3) - a(3) * b(2)
    r(2) = a(3) * b(1) - a(1) * b(3)
    r(3) = a(1) * b(2) - a(2) * b(1)

  end subroutine spl_compute_cross_product_3d
  ! ..................................................
    
  ! .......................................................... 
  !> @brief    computes the determinant of inverse of the the jacobian matrix
  !>
  !> @param[in]    arr_x_u        array of dx/du = dF/du(u). array of dim(1:d_dim,1:n_points)
  !> @param[inout] determinants   determinants of the mapping F. array of dim(1:n_points) 
  subroutine spl_compute_determinants_jacobians_1d( arr_x_u, determinants)
  implicit none
    real(spl_rk), dimension(:,:), intent(in)    :: arr_x_u
    real(spl_rk), dimension(:)  , intent(inout) :: determinants
    ! local
    integer :: i
    integer :: d_dim
    integer :: n_points

    ! ...
    d_dim    = size(arr_x_u,1)
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    determinants(1:n_points) = 0.0_spl_rk
    do i = 1, d_dim
        determinants(1:n_points)= determinants(1:n_points) + arr_x_u(i,1:n_points)**2
    end do
    determinants(1:n_points) = sqrt(determinants(1:n_points))
    ! ...

  end subroutine spl_compute_determinants_jacobians_1d    
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    computes the inverse jacobian matrix 
  !>
  !> @param[in]    arr_x_u        array of dx/du = dF/du(u). array of dim(1:2,1:n_points)
  !> @param[in]    determinants   determinants of the mapping F. array of dim(1:n_points) 
  !> @param[inout] inv_jacobians  inverse of the jacobian matrix of F. array of dim(2,2,1:n_points) 
  subroutine spl_compute_inv_jacobians_1d( arr_x_u, determinants, inv_jacobians)
  implicit none
    real(spl_rk), dimension(:,:), intent(in)    :: arr_x_u
    real(spl_rk), dimension(:)  , intent(in)    :: determinants 
    real(spl_rk), dimension(:)  , intent(inout) :: inv_jacobians
    ! local
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    inv_jacobians(1:n_points)= 1.0_spl_rk / determinants(1:n_points)
    ! ...

  end subroutine spl_compute_inv_jacobians_1d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    computes the inverse of the hessian matrix
  !>
  !> @param[in]    arr_x_u       array of dx/du = dF/du(u). array of dim(1:2,1:n_points)
  !> @param[in]    arr_x_uu      array of d2x/d2u = d2F/d2u(u). array of dim(1:2,1:n_points)
  !> @param[inout] inv_hessian   inverse of the hessian matrix of F. array of dim(2,2,1:n_points) 
  subroutine spl_compute_inv_hessian_1d( arr_x_u, arr_x_uu, inv_hessian)
  implicit none
    real(spl_rk), dimension(:,:), intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_uu
    real(spl_rk), dimension(:,:)  , intent(inout) :: inv_hessian
    ! local
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    inv_hessian(1,:n_points)= 1.0_spl_rk / arr_x_uu(1,1:n_points)
    inv_hessian(2,:n_points)= 1.0_spl_rk / (arr_x_u(1,1:n_points)**2)
    ! ...

  end subroutine spl_compute_inv_hessian_1d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    computes the determinant of inverse of the the jacobian matrix
  !>
  !> @param[in]    arr_x_u        array of dx/du = dF/du(u). array of dim(1:2,1:n_points)
  !> @param[in]    arr_x_v        array of dx/dv = dF/dv(v). array of dim(1:2,1:n_points)
  !> @param[inout] determinants   determinants of the mapping F. array of dim(1:n_points) 
  subroutine spl_compute_determinants_jacobians_2d( arr_x_u, arr_x_v, determinants )
  implicit none
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_v
    real(spl_rk), dimension(:)    , intent(inout) :: determinants 
    ! local
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    ! x_eta * y_theta - x_theta * y_eta
    determinants(1:n_points) = arr_x_u(1, 1:n_points) * arr_x_v(2, 1:n_points) &
                           & - arr_x_v(1, 1:n_points) * arr_x_u(2, 1:n_points)
    ! ...

  end subroutine spl_compute_determinants_jacobians_2d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    computes the inverse of the jacobian matrix
  !>
  !> @param[in]    arr_x_u        array of dx/du = dF/du(u). array of dim(1:2,1:n_points)
  !> @param[in]    arr_x_v        array of dx/dv = dF/dv(v). array of dim(1:2,1:n_points)
  !> @param[in]    determinants   determinants of the mapping F. array of dim(1:n_points) 
  !> @param[inout] inv_jacobians  inverse of the jacobian matrix of F. array of dim(2,2,1:n_points) 
  subroutine spl_compute_inv_jacobians_2d( arr_x_u, arr_x_v, determinants, inv_jacobians )
  implicit none
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_v
    real(spl_rk), dimension(:)    , intent(in)    :: determinants 
    real(spl_rk), dimension(:,:,:), intent(inout) :: inv_jacobians
    ! local
    integer :: i_point
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...
    
    ! ...
    inv_jacobians (1, 1, 1:n_points) =   arr_x_v(2, 1:n_points)
    inv_jacobians (1, 2, 1:n_points) = - arr_x_u(2, 1:n_points)
    inv_jacobians (2, 1, 1:n_points) = - arr_x_v(1, 1:n_points)
    inv_jacobians (2, 2, 1:n_points) =   arr_x_u(1, 1:n_points)
    ! ...

    ! ...
    do i_point = 1, n_points
      inv_jacobians (:,:,i_point) = inv_jacobians (:,:,i_point) / determinants(i_point)
    end do
    ! ...

  end subroutine spl_compute_inv_jacobians_2d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    computes the inverse of the hessian matrix
  !>
  !> @param[in]    arr_x_u      array of dx/du = dF/du(u). array of dim(1:2,1:n_points)
  !> @param[in]    arr_x_v      array of dx/dv = dF/dv(v). array of dim(1:2,1:n_points)
  !> @param[in]    arr_x_uu     array of dx/duu = dF/du(u). array of dim(1:2,1:n_points)
  !> @param[in]    arr_x_uv     array of dx/duv = dF/du(u). array of dim(1:2,1:n_points)
  !> @param[in]    arr_x_vv     array of dx/dvv = dF/dv(v). array of dim(1:2,1:n_points)
  !> @param[inout] inv_hessian  inverse of the jacobian matrix of F. array of dim(3,5,1:n_points) 
  subroutine spl_compute_inv_hessian_2d( arr_x_u, arr_x_v, arr_x_uu, arr_x_uv, arr_x_vv, inv_hessian )
    implicit none
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_v
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_uu
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_uv
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_vv
    real(spl_rk), dimension(:,:,:), intent(inout) :: inv_hessian
    ! local
    integer :: i_point
    integer :: n_points
    real(spl_rk) :: det, det_x, det_y
    real(spl_rk) :: R_u, R_v, R_uu, R_uv, R_vv
    real(spl_rk) :: Z_u, Z_v, Z_uu, Z_uv, Z_vv

    n_points = size(arr_x_u,2)
    
    do i_point = 1, n_points

       R_u = arr_x_u(1, i_point)
       R_v = arr_x_v(1, i_point)
       R_uu = arr_x_uu(1, i_point)
       R_uv = arr_x_uv(1, i_point)
       R_vv = arr_x_vv(1, i_point)

       Z_u = arr_x_u(2, i_point)
       Z_v = arr_x_v(2, i_point)
       Z_uu = arr_x_uu(2, i_point)
       Z_uv = arr_x_uv(2, i_point)
       Z_vv = arr_x_vv(2, i_point)
       
       det =  R_u * Z_v - R_v * Z_u
       
       det_x  = (R_uu* Z_v**2 - Z_uu*R_v*Z_v - 2.d0*R_uv*Z_u*Z_v   &       
            + Z_uv*(R_u*Z_v + R_v*Z_u)			       &
            + R_vv* Z_u**2 - Z_vv*R_u*Z_u) / det
       
       det_y  = (Z_vv* R_u**2 - R_vv*Z_u*R_u - 2.d0*Z_uv*R_v*R_u   &       
            + R_uv*(Z_v*R_u + Z_u*R_v)			       &
            + Z_uu* R_v**2 - R_uu*Z_v*R_v) / det
       
       inv_hessian(1,1, i_point) = (Z_uv*Z_v - Z_vv*Z_u ) / det**2  - det_x * Z_v / det**2
       inv_hessian(1,2, i_point) = (Z_uv*Z_u - Z_uu*Z_v ) / det**2  + det_x * Z_u / det**2
       inv_hessian(1,3, i_point) = Z_v**2 / det**2
       inv_hessian(1,4, i_point) = - 2.d0* Z_u*Z_v / det**2
       inv_hessian(1,5, i_point) = Z_u**2 / det**2
       
       inv_hessian(2,1, i_point) = - (R_uv*Z_v - R_vv*Z_u )  /det**2 + det_x * R_v / det**2
       inv_hessian(2,2, i_point) = - (R_uv*Z_u - R_uu*Z_v ) /det**2 - det_x * R_u / det**2
       inv_hessian(2,3, i_point) = - Z_v*R_v  / det**2
       inv_hessian(2,4, i_point) = (Z_u*R_v  + Z_v*R_u  ) / det**2
       inv_hessian(2,5, i_point) = - R_u*Z_u / det**2       
       
       inv_hessian(3,1, i_point) = (R_uv*R_v - R_vv*R_u ) / det**2  - det_y * R_v / det**2
       inv_hessian(3,2, i_point) = (R_uv*R_u - R_uu*R_v ) / det**2  + det_y * R_u / det**2
       inv_hessian(3,3, i_point) = R_v**2 / det**2
       inv_hessian(3,4, i_point) = - 2.d0* R_u*R_v / det**2
       inv_hessian(3,5, i_point) = R_u**2 / det**2
       
    end do
  end subroutine spl_compute_inv_hessian_2d
  ! ..........................................................        
  
  ! ..........................................................        
  !> @brief    computes the inverse of the hessian matrix
  !>
  !> @param[in]    arr_x_u      array of dx/du = dF/du(u).   array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_v      array of dx/dv = dF/dv(v).   array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_w      array of dx/dw = dF/dw(w).   array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_uu     array of dx/duu = dF/duu(u). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_uv     array of dx/duv = dF/duv(u). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_uw     array of dx/duw = dF/duw(u). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_vv     array of dx/dvv = dF/dvv(u). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_vw     array of dx/dvw = dF/dvw(v). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_ww     array of dx/dww = dF/dww(w). array of dim(1:3,1:n_points)
  !> @param[inout] inv_hessian  inverse of the hesssian matrix of F. array of dim(6,9,1:n_points) 
  subroutine spl_compute_inv_hessian_3d( arr_x_u , &
                                       & arr_x_v , &
                                       & arr_x_w , &
                                       & arr_x_uu, &
                                       & arr_x_uv, &
                                       & arr_x_uw, &
                                       & arr_x_vv, &
                                       & arr_x_vw, &
                                       & arr_x_ww, &
                                       & inv_hessian )
    implicit none
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_v
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_w
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_uu
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_uv
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_uw
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_vv
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_vw
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_ww
    real(spl_rk), dimension(:,:,:), intent(inout) :: inv_hessian
    ! local
    integer :: i_point
    integer :: n_points
    integer :: i, j, k, l
    real(spl_rk), dimension(3,3) :: mat_a
    real(spl_rk), dimension(3,3) :: mat_a_inv
    real(spl_rk), dimension(6,3) :: mat_b
    real(spl_rk), dimension(6,6) :: mat_h
    real(spl_rk), dimension(6,6) :: mat_h_inv
    real(spl_rk), dimension(6,3) :: mat_c
    
    n_points = size(arr_x_u, 2)
    
    do i_point = 1, n_points
      ! ...
      mat_h     = 0.0_spl_rk
      mat_h_inv = 0.0_spl_rk
      mat_a     = 0.0_spl_rk
      mat_a_inv = 0.0_spl_rk
      ! ...
      
      ! ... in terms of: dx, dy, dz (indexed by j) 
      do j= 1, 3
        ! du, dv, dw
        mat_a(1, j) = arr_x_u(j, i_point)
        mat_a(2, j) = arr_x_v(j, i_point)
        mat_a(3, j) = arr_x_w(j, i_point)
        
        ! duu, duv, duw, dvv, dvw, dww 
        mat_b(1, j) = arr_x_uu(j, i_point)
        mat_b(2, j) = arr_x_uv(j, i_point)
        mat_b(3, j) = arr_x_uw(j, i_point)
        mat_b(4, j) = arr_x_vv(j, i_point)
        mat_b(5, j) = arr_x_vw(j, i_point)
        mat_b(6, j) = arr_x_ww(j, i_point)
      end do 
      ! ... 
      
      ! ... in terms of: dxx, dxy, dxz, dyy, dyz, dzz 
      ! duu 
      mat_h(1, 1) = arr_x_u(1, i_point)**2
      mat_h(1, 2) = 2.0_spl_rk*arr_x_u(1, i_point)*arr_x_u(2, i_point)
      mat_h(1, 3) = 2.0_spl_rk*arr_x_u(1, i_point)*arr_x_u(3, i_point)
      mat_h(1, 4) = arr_x_u(2, i_point)**2
      mat_h(1, 5) = 2.0_spl_rk*arr_x_u(2, i_point)*arr_x_u(3, i_point)
      mat_h(1, 6) = arr_x_u(3, i_point)**2
      
      ! duv 
      mat_h(2, 1) = arr_x_u(1, i_point)*arr_x_v(1, i_point)
      mat_h(2, 2) = arr_x_u(1, i_point)*arr_x_v(2, i_point) + arr_x_u(2, i_point)*arr_x_v(1, i_point)
      mat_h(2, 3) = arr_x_u(1, i_point)*arr_x_v(3, i_point) + arr_x_u(3, i_point)*arr_x_v(1, i_point)
      mat_h(2, 4) = arr_x_u(2, i_point)*arr_x_v(2, i_point)
      mat_h(2, 5) = arr_x_u(2, i_point)*arr_x_v(3, i_point) + arr_x_u(3, i_point)*arr_x_v(2, i_point)
      mat_h(2, 6) = arr_x_u(3, i_point)*arr_x_v(3, i_point)
     
      ! duw 
      mat_h(3, 1) = arr_x_u(1, i_point)*arr_x_w(1, i_point)
      mat_h(3, 2) = arr_x_u(1, i_point)*arr_x_w(2, i_point) + arr_x_u(2, i_point)*arr_x_w(1, i_point)
      mat_h(3, 3) = arr_x_u(1, i_point)*arr_x_w(3, i_point) + arr_x_u(3, i_point)*arr_x_w(1, i_point)
      mat_h(3, 4) = arr_x_u(2, i_point)*arr_x_w(2, i_point)
      mat_h(3, 5) = arr_x_u(2, i_point)*arr_x_w(3, i_point) + arr_x_u(3, i_point)*arr_x_w(2, i_point)
      mat_h(3, 6) = arr_x_u(3, i_point)*arr_x_w(3, i_point)
      
      ! dvv 
      mat_h(4, 1) = arr_x_v(1, i_point)**2
      mat_h(4, 2) = 2.0_spl_rk*arr_x_v(1, i_point)*arr_x_v(2, i_point)
      mat_h(4, 3) = 2.0_spl_rk*arr_x_v(1, i_point)*arr_x_v(3, i_point)
      mat_h(4, 4) = arr_x_v(2, i_point)**2
      mat_h(4, 5) = 2.0_spl_rk*arr_x_v(2, i_point)*arr_x_v(3, i_point)
      mat_h(4, 6) = arr_x_v(3, i_point)**2

      ! dvw 
      mat_h(5, 1) = arr_x_v(1, i_point)*arr_x_w(1, i_point)
      mat_h(5, 2) = arr_x_v(1, i_point)*arr_x_w(2, i_point) + arr_x_v(2, i_point)*arr_x_w(1, i_point)
      mat_h(5, 3) = arr_x_v(1, i_point)*arr_x_w(3, i_point) + arr_x_v(3, i_point)*arr_x_w(1, i_point)
      mat_h(5, 4) = arr_x_v(2, i_point)*arr_x_w(2, i_point)
      mat_h(5, 5) = arr_x_v(2, i_point)*arr_x_w(3, i_point) + arr_x_v(3, i_point)*arr_x_w(2, i_point)
      mat_h(5, 6) = arr_x_v(3, i_point)*arr_x_w(3, i_point)

      ! dww 
      mat_h(6, 1) = arr_x_w(1, i_point)**2
      mat_h(6, 2) = 2.0_spl_rk*arr_x_w(1, i_point)*arr_x_w(2, i_point)
      mat_h(6, 3) = 2.0_spl_rk*arr_x_w(1, i_point)*arr_x_w(3, i_point)
      mat_h(6, 4) = arr_x_w(2, i_point)**2
      mat_h(6, 5) = 2.0_spl_rk*arr_x_w(2, i_point)*arr_x_w(3, i_point)
      mat_h(6, 6) = arr_x_w(3, i_point)**2
      !... 
      
      !... invert H and A and compute C =  H-1*B*A-1 
      call  spl_direct_inverse(mat_a, mat_a_inv, 3)
      call  spl_direct_inverse(mat_h, mat_h_inv, 6)
      
      mat_c = 0.0_spl_rk
      do i = 1, 6
      do j = 1, 3
          do k =1, 6
          do l =1, 3
            mat_c(i,j) = mat_c(i,j) - mat_h_inv(i, k)*mat_b(k, l)*mat_a_inv(l, j) 
           enddo
          end do 
       end do
      end do
      !...

      !... D2[phi(x,y,z)] = H-1*D2[(phi(u,u,w)] -C*D1[phi(u,v,w)], D1 and D2 are 1st and 2nd derivatives.
      ! dxx, dxy, dxz, dyy, dyz, dzz in terms of: 
      ! du, dv, dw, duu, duv, duw, dvv, dvw, dww
      do j=1, 3
        do i=1, 6
          inv_hessian(i,j,   i_point) =  mat_c(i, j) 
          inv_hessian(i,j+3, i_point) =  mat_h_inv(i, j)
          inv_hessian(i,j+6, i_point) =  mat_h_inv(i, j+3)
        end do
      end do
      !...
      
    end do
    
  end subroutine spl_compute_inv_hessian_3d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief   pullback of one forms 
  !>
  !> @param[in]    arr_x_u            array of dx/du = dF/du(u). array of dim(1:2,1:n_points)
  !> @param[in]    arr_x_v            array of dx/dv = dF/dv(v). array of dim(1:2,1:n_points)
  !> @param[in]    determinants       determinants of the mapping F. array of dim(1:n_points) 
  !> @param[inout] pullback_one_form  inverse of the jacobian matrix of F. array of dim(2,2,1:n_points) 
  subroutine spl_compute_pullback_one_form_2d( arr_x_u, arr_x_v, determinants, pullback_one_form )
  implicit none
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_v
    real(spl_rk), dimension(:)    , intent(in)    :: determinants 
    real(spl_rk), dimension(:,:,:), intent(inout) :: pullback_one_form
    ! local
    integer :: i_point
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    pullback_one_form (1, 1, 1:n_points) =   arr_x_v(2, 1:n_points)
    pullback_one_form (1, 2, 1:n_points) = - arr_x_u(2, 1:n_points)
    pullback_one_form (2, 1, 1:n_points) = - arr_x_v(1, 1:n_points)
    pullback_one_form (2, 2, 1:n_points) =   arr_x_u(1, 1:n_points) 
    ! ...

    ! ...
    do i_point = 1, n_points
      pullback_one_form (:,:,i_point) = pullback_one_form (:,:,i_point) / determinants(i_point)
    end do
    ! ...

  end subroutine spl_compute_pullback_one_form_2d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief   pullback of two forms 
  !>
  !> @param[in]    arr_x_u            array of dx/du = dF/du(u). array of dim(1:2,1:n_points)
  !> @param[in]    arr_x_v            array of dx/dv = dF/dv(v). array of dim(1:2,1:n_points)
  !> @param[in]    determinants       determinants of the mapping F. array of dim(1:n_points) 
  !> @param[inout] pullback_two_form  inverse of the jacobian matrix of F. array of dim(2,2,1:n_points) 
  subroutine spl_compute_pullback_two_form_2d( arr_x_u, arr_x_v, determinants, pullback_two_form )
  implicit none
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:),   intent(in)    :: arr_x_v
    real(spl_rk), dimension(:)    , intent(in)    :: determinants 
    real(spl_rk), dimension(:,:,:), intent(inout) :: pullback_two_form
    ! local
    integer :: i_point
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    pullback_two_form (1, 1, 1:n_points) = arr_x_u(1, 1:n_points)
    pullback_two_form (1, 2, 1:n_points) = arr_x_v(1, 1:n_points)
    pullback_two_form (2, 1, 1:n_points) = arr_x_u(2, 1:n_points)
    pullback_two_form (2, 2, 1:n_points) = arr_x_v(2, 1:n_points)
    ! ...

    ! ...
    do i_point = 1, n_points
      pullback_two_form (:,:,i_point) = pullback_two_form (:,:,i_point) / determinants(i_point)
    end do
    ! ...

  end subroutine spl_compute_pullback_two_form_2d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    computes the determinant of inverse of the the jacobian matrix
  !>
  !> @param[in]    arr_x_u        array of dx/du = dF/du(u). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_v        array of dx/dv = dF/dv(v). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_w        array of dx/dw = dF/dw(w). array of dim(1:3,1:n_points)
  !> @param[inout] determinants   determinants of the mapping F. array of dim(1:n_points) 
  subroutine spl_compute_determinants_jacobians_3d( arr_x_u, arr_x_v, arr_x_w, determinants )
  implicit none
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_v
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_w
    real(spl_rk), dimension(:)    , intent(inout) :: determinants
            ! local
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    determinants(1:n_points) = &
      ! + x_ksi * (y_eta * z_theta - y_theta * z_eta)
      &   arr_x_u(1,1:n_points) * ( arr_x_v(2,1:n_points) * arr_x_w(3,1:n_points)   &
      &                           - arr_x_w(2,1:n_points) * arr_x_v(3,1:n_points) ) &
      ! - y_ksi * (x_eta * z_theta - x_theta * z_eta)
      & - arr_x_u(2,1:n_points) * ( arr_x_v(1,1:n_points) * arr_x_w(3,1:n_points)   &
      &                           - arr_x_w(1,1:n_points) * arr_x_v(3,1:n_points) ) &
      ! + z_ksi * (x_eta * y_theta - x_theta * y_eta)
      & + arr_x_u(3,1:n_points) * ( arr_x_v(1,1:n_points) * arr_x_w(2,1:n_points)   &
      &                           - arr_x_w(1,1:n_points) * arr_x_v(2,1:n_points) )
    ! ...

  end subroutine spl_compute_determinants_jacobians_3d 
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    computes the inverse of the jacobian matrix
  !>
  !> @param[in]    arr_x_u        array of dx/du = dF/du(u). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_v        array of dx/dv = dF/dv(v). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_w        array of dx/dw = dF/dw(w). array of dim(1:3,1:n_points)
  !> @param[in]    determinants   determinants of the mapping F. array of dim(1:n_points) 
  !> @param[inout] inv_jacobians  inverse of the jacobian inv_jacobians of F. array of dim(3,3,1:n_points) 
  subroutine spl_compute_inv_jacobians_3d( arr_x_u, arr_x_v, arr_x_w, determinants, inv_jacobians )
  implicit none
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_v
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_w
    real(spl_rk), dimension(:)    , intent(in)    :: determinants
    real(spl_rk), dimension(:,:,:), intent(inout) :: inv_jacobians
    ! local
    integer :: i_point
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    inv_jacobians(1,1,1:n_points) =   arr_x_v(2,1:n_points) * arr_x_w(3,1:n_points) &
                                  & - arr_x_w(2,1:n_points) * arr_x_v(3,1:n_points)
    inv_jacobians(1,2,1:n_points) = - arr_x_u(2,1:n_points) * arr_x_w(3,1:n_points) &
                                  & + arr_x_w(2,1:n_points) * arr_x_u(3,1:n_points)
    inv_jacobians(1,3,1:n_points) =   arr_x_u(2,1:n_points) * arr_x_v(3,1:n_points) &
                                  & - arr_x_v(2,1:n_points) * arr_x_u(3,1:n_points)
    inv_jacobians(2,1,1:n_points) = - arr_x_v(1,1:n_points) * arr_x_w(3,1:n_points) &
                                  & + arr_x_w(1,1:n_points) * arr_x_v(3,1:n_points)
    inv_jacobians(2,2,1:n_points) =   arr_x_u(1,1:n_points) * arr_x_w(3,1:n_points) &
                                  & - arr_x_w(1,1:n_points) * arr_x_u(3,1:n_points)
    inv_jacobians(2,3,1:n_points) = - arr_x_u(1,1:n_points) * arr_x_v(3,1:n_points) &
                                  & + arr_x_v(1,1:n_points) * arr_x_u(3,1:n_points)
    inv_jacobians(3,1,1:n_points) =   arr_x_v(1,1:n_points) * arr_x_w(2,1:n_points) &
                                  & - arr_x_w(1,1:n_points) * arr_x_v(2,1:n_points)
    inv_jacobians(3,2,1:n_points) =   arr_x_u(1,1:n_points) * arr_x_w(2,1:n_points) &
                                  & - arr_x_w(1,1:n_points) * arr_x_u(2,1:n_points)
    inv_jacobians(3,3,1:n_points) =   arr_x_u(1,1:n_points) * arr_x_v(2,1:n_points) &
                                  & - arr_x_v(1,1:n_points) * arr_x_u(2,1:n_points)
    ! ...

    ! ...
    do i_point = 1, n_points
        inv_jacobians (:,:,i_point) = inv_jacobians (:,:,i_point) / determinants(i_point)
    end do
    ! ...

  end subroutine spl_compute_inv_jacobians_3d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    pullback for one forms 
  !>
  !> @param[in]    arr_x_u            array of dx/du = dF/du(u). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_v            array of dx/dv = dF/dv(v). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_w            array of dx/dw = dF/dw(w). array of dim(1:3,1:n_points)
  !> @param[in]    determinants       determinants of the mapping F. array of dim(1:n_points) 
  !> @param[inout] pullback_one_form  inverse of the jacobian pullback_one_form of F. array of dim(3,3,1:n_points) 
  subroutine spl_compute_pullback_one_form_3d( arr_x_u, arr_x_v, arr_x_w, determinants, pullback_one_form )
  implicit none
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_v
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_w
    real(spl_rk), dimension(:)    , intent(in)    :: determinants
    real(spl_rk), dimension(:,:,:), intent(inout) :: pullback_one_form
    ! local
    integer :: i_point
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    ! (y_eta * z_theta - y_theta * z_eta)
    pullback_one_form(1,1,1:n_points) =   arr_x_v(2,1:n_points) * arr_x_w(3,1:n_points) &
                                    & - arr_x_w(2,1:n_points) * arr_x_v(3,1:n_points)
    ! -(x_eta * z_theta - x_theta * z_eta) 
    pullback_one_form(1,2,1:n_points) = - arr_x_u(2,1:n_points) * arr_x_w(3,1:n_points) &
                                    & + arr_x_w(2,1:n_points) * arr_x_u(3,1:n_points)
    ! (x_eta * y_theta - x_theta * y_eta) 
    pullback_one_form(1,3,1:n_points) =   arr_x_u(2,1:n_points) * arr_x_v(3,1:n_points) &
                                    & - arr_x_v(2,1:n_points) * arr_x_u(3,1:n_points)
                        
    ! -(y_ksi * z_theta - y_theta * z_ksi)
    pullback_one_form(2,1,1:n_points) = - arr_x_v(1,1:n_points) * arr_x_w(3,1:n_points) &
                                    & + arr_x_w(1,1:n_points) * arr_x_v(3,1:n_points)
    ! (x_ksi * z_theta - x_theta * z_ksi)
    pullback_one_form(2,2,1:n_points) =   arr_x_u(1,1:n_points) * arr_x_w(3,1:n_points) &
                                    & - arr_x_w(1,1:n_points) * arr_x_u(3,1:n_points)
    ! -(x_ksi * y_theta - x_theta * y_ksi) 
    pullback_one_form(2,3,1:n_points) = - arr_x_u(1,1:n_points) * arr_x_v(3,1:n_points) &
                                    & + arr_x_v(1,1:n_points) * arr_x_u(3,1:n_points)
                        
    ! (y_ksi * z_eta - y_eta * z_ksi)
    pullback_one_form(3,1,1:n_points) =   arr_x_v(1,1:n_points) * arr_x_w(2,1:n_points) &
                                    & - arr_x_w(1,1:n_points) * arr_x_v(2,1:n_points)
    ! - (x_ksi * z_eta - x_eta * z_ksi)
    pullback_one_form(3,2,1:n_points) =   arr_x_u(1,1:n_points) * arr_x_w(2,1:n_points) &
                                    & - arr_x_w(1,1:n_points) * arr_x_u(2,1:n_points)
    ! (x_ksi * y_eta - x_eta * y_ksi)
    pullback_one_form(3,3,1:n_points) =   arr_x_u(1,1:n_points) * arr_x_v(2,1:n_points) &
                                    & - arr_x_v(1,1:n_points) * arr_x_u(2,1:n_points)
    ! ...
    
    ! ...
    do i_point = 1, n_points
        pullback_one_form (:,:,i_point) = pullback_one_form (:,:,i_point) / determinants(i_point)
    end do
    ! ...

  end subroutine spl_compute_pullback_one_form_3d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    pullback for two forms 
  !>
  !> @param[in]    arr_x_u        array of dx/du = dF/du(u). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_v        array of dx/dv = dF/dv(v). array of dim(1:3,1:n_points)
  !> @param[in]    arr_x_w        array of dx/dw = dF/dw(w). array of dim(1:3,1:n_points)
  !> @param[in]    determinants   determinants of the mapping F. array of dim(1:n_points) 
  !> @param[inout] pullback_two_form  inverse of the jacobian pullback_two_form of F. array of dim(3,3,1:n_points) 
  subroutine spl_compute_pullback_two_form_3d( arr_x_u, arr_x_v, arr_x_w, determinants, pullback_two_form )
  implicit none
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_u
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_v
    real(spl_rk), dimension(:,:)  , intent(in)    :: arr_x_w
    real(spl_rk), dimension(:)    , intent(in)    :: determinants
    real(spl_rk), dimension(:,:,:), intent(inout) :: pullback_two_form
    ! local
    integer :: i_point
    integer :: n_points

    ! ...
    n_points = size(arr_x_u,2)
    ! ...

    ! ...
    pullback_two_form (1, 1, 1:n_points) = arr_x_u(1, 1:n_points)
    pullback_two_form (1, 2, 1:n_points) = arr_x_v(1, 1:n_points)
    pullback_two_form (1, 3, 1:n_points) = arr_x_w(1, 1:n_points)
    pullback_two_form (2, 1, 1:n_points) = arr_x_u(2, 1:n_points)
    pullback_two_form (2, 2, 1:n_points) = arr_x_v(2, 1:n_points)
    pullback_two_form (2, 3, 1:n_points) = arr_x_w(2, 1:n_points)
    pullback_two_form (3, 1, 1:n_points) = arr_x_u(3, 1:n_points)
    pullback_two_form (3, 2, 1:n_points) = arr_x_v(3, 1:n_points)
    pullback_two_form (3, 3, 1:n_points) = arr_x_w(3, 1:n_points)
    ! ...
    
    ! ...
    do i_point = 1, n_points
        pullback_two_form (:,:,i_point) = pullback_two_form (:,:,i_point) / determinants(i_point)
    end do
    ! ...

  end subroutine spl_compute_pullback_two_form_3d
  ! ..........................................................        

  ! ..........................................................        
  !> @brief    map a vector valued function in 2d 
  !>
  !> @param[in]    f_logical      f vector field in the logical domain. array of dim(2,1:n_points)
  !> @param[in]    matrix_jac     jacobian matrix to compute first derivative  array of dim(2,2,1:n_points)
  !> @param[inout] f_physical     f vector field in the physical domain. array of dim(2,1:n_points)
  subroutine spl_map_vector_2d(f_logical, matrix_jac, f_physical)
  implicit none
    real(spl_rk), dimension(:,:)  , intent(in)    :: f_logical  
    real(spl_rk), dimension(:,:,:), intent(in)    :: matrix_jac
    real(spl_rk), dimension(:,:)  , intent(inout) :: f_physical
    ! local
    integer :: i_point
    integer :: n_points
    real(spl_rk) :: j11, j12, j21, j22

    ! ...
    n_points = ubound(f_logical, 2)
    ! ...

    ! ...
    do i_point = 1, n_points
      ! ...
      j11 = matrix_jac(1,1,i_point) 
      j12 = matrix_jac(1,2,i_point) 
      j21 = matrix_jac(2,1,i_point) 
      j22 = matrix_jac(2,2,i_point)

      ! ...
      f_physical(1, i_point) = j11 * f_logical(1,i_point) &
                           & + j12 * f_logical(2,i_point) 

      f_physical(2, i_point) = j21 * f_logical(1,i_point) &
           & + j22 * f_logical(2,i_point)
      
      ! ...
    end do
    ! ...

  end subroutine spl_map_vector_2d
  ! ..........................................................

  ! ..........................................................        
  !> @brief    map a vector valued function in 2d for second derivatives 
  !>
  !> @param[in]    f_logical      f vector field in the logical domain. array of dim(2,1:n_points)
  !> @param[in]    matrix_hes     hessian matrix to compute first derivative  array of dim(2,2,1:n_points) 
  !> @param[inout] f_physical     f vector field in the physical domain. array of dim(2,1:n_points)
  subroutine spl_map_second_derivate_2d(f_logical, matrix_hes, f_physical)
  implicit none
    real(spl_rk), dimension(:,:)  , intent(in)    :: f_logical  
    real(spl_rk), dimension(:,:,:), intent(in)    :: matrix_hes
    real(spl_rk), dimension(:,:)  , intent(inout) :: f_physical
    ! local
    integer :: i_point
    integer :: n_points
    real(spl_rk) :: h11, h12, h13, h14, h15
    real(spl_rk) :: h21, h22, h23, h24, h25
    real(spl_rk) :: h31, h32, h33, h34, h35

    ! ...
    n_points = ubound(f_logical, 2)
    ! ...
    ! ...
    do i_point = 1, n_points

      h11 = matrix_hes(1,1,i_point) 
      h12 = matrix_hes(1,2,i_point) 
      h13 = matrix_hes(1,3,i_point) 
      h14 = matrix_hes(1,4,i_point) 
      h15 = matrix_hes(1,5,i_point)

      h21 = matrix_hes(2,1,i_point) 
      h22 = matrix_hes(2,2,i_point) 
      h23 = matrix_hes(2,3,i_point) 
      h24 = matrix_hes(2,4,i_point) 
      h25 = matrix_hes(2,5,i_point)

      h31 = matrix_hes(3,1,i_point) 
      h32 = matrix_hes(3,2,i_point) 
      h33 = matrix_hes(3,3,i_point) 
      h34 = matrix_hes(3,4,i_point) 
      h35 = matrix_hes(3,5,i_point)      

      f_physical(1, i_point) = h11 * f_logical(1,i_point) + h12 * f_logical(2,i_point) &
           & + h13 * f_logical(3,i_point) + h14 * f_logical(4,i_point) + h15 * f_logical(5,i_point) 
      f_physical(2, i_point) = h21 * f_logical(1,i_point) + h22 * f_logical(2,i_point) &
           & + h23 * f_logical(3,i_point) + h24 * f_logical(4,i_point) + h25 * f_logical(5,i_point) 
      f_physical(3, i_point) = h31 * f_logical(1,i_point) + h32 * f_logical(2,i_point) &
           & + h33 * f_logical(3,i_point) + h34 * f_logical(4,i_point) + h35 * f_logical(5,i_point) 
      ! ...
    end do
    ! ...

  end subroutine spl_map_second_derivate_2d
  ! ..........................................................        
  
  ! ..........................................................        
  !> @brief    map a vector valued function in 3d 
  !>
  !> @param[in]    f_logical      f vector field in the logical domain. array of dim(3,1:n_points)
  !> @param[in]    matrix         array of dim(3,3,1:n_points) 
  !> @param[inout] f_physical     f vector field in the physical domain. array of dim(3,1:n_points)
  subroutine spl_map_vector_3d(f_logical, matrix, f_physical)
  implicit none
    real(spl_rk), dimension(:,:)  , intent(in)    :: f_logical  
    real(spl_rk), dimension(:,:,:), intent(in)    :: matrix
    real(spl_rk), dimension(:,:)  , intent(inout) :: f_physical
    ! local
    integer :: i_point
    integer :: n_points
    real(spl_rk) :: j11
    real(spl_rk) :: j12
    real(spl_rk) :: j13
    real(spl_rk) :: j21
    real(spl_rk) :: j22
    real(spl_rk) :: j23
    real(spl_rk) :: j31
    real(spl_rk) :: j32
    real(spl_rk) :: j33

    ! ...
    n_points = ubound(f_logical, 2)
    ! ...

    ! ...
    do i_point = 1, n_points
      ! ...
      j11 = matrix(1,1,i_point) 
      j12 = matrix(1,2,i_point) 
      j13 = matrix(1,3,i_point) 
      j21 = matrix(2,1,i_point) 
      j22 = matrix(2,2,i_point) 
      j23 = matrix(2,3,i_point) 
      j31 = matrix(3,1,i_point) 
      j32 = matrix(3,2,i_point) 
      j33 = matrix(3,3,i_point) 
      ! ...

      ! ...
      f_physical(1, i_point) = j11 * f_logical(1,i_point) &  
                           & + j12 * f_logical(2,i_point) &
                           & + j13 * f_logical(3,i_point)

      f_physical(2, i_point) = j21 * f_logical(1,i_point) &
                           & + j22 * f_logical(2,i_point) &
                           & + j23 * f_logical(3,i_point)

      f_physical(3, i_point) = j31 * f_logical(1,i_point) &
                           & + j32 * f_logical(2,i_point) &
                           & + j33 * f_logical(3,i_point)
      ! ...
    end do
    ! ...

  end subroutine spl_map_vector_3d
  ! ..........................................................  
  
  ! ..........................................................        
  !> @brief    map a vector valued function in 3d for second derivatives 
  !>
  !> @param[in]    f_logical      f vector field in the logical domain. array of dim(9,1:n_points)
  !> @param[in]    matrix_hes     hessian matrix to compute first derivative  array of dim(6,9,1:n_points) 
  !> @param[inout] f_physical     f vector field in the physical domain. array of dim(6,1:n_points)
  subroutine spl_map_second_derivate_3d(f_logical, matrix_hes, f_physical)
  implicit none
    real(spl_rk), dimension(:,:)  , intent(in)    :: f_logical  
    real(spl_rk), dimension(:,:,:), intent(in)    :: matrix_hes
    real(spl_rk), dimension(:,:)  , intent(inout) :: f_physical
    ! local
    integer :: i, j
    integer :: i_point
    integer :: n_points
    
    ! ...
    n_points = ubound(f_logical, 2)
    ! ...
    
    f_physical = 0.0_spl_rk
    ! ...
    do i_point = 1, n_points
        do i= 1, 6
        do j = 1, 9
          f_physical(i, i_point) = f_physical(i, i_point) +   matrix_hes(i, j, i_point)*f_logical(j, i_point) 
        end do
      end do
    end do
    ! ...
    
  end subroutine spl_map_second_derivate_3d
  ! ..........................................................      
  
  ! ..................................................
  !> @brief  invert a square matrix using LU factorization 
  !>
  !> @param[in]     n       number of rows/columns
  !> @param[in]     a       matrix to invert 
  !> @param[inout]  c       inverse matrix 
  subroutine spl_direct_inverse(a,c,n)
  !============================================================
  ! Inverse matrix
  ! Method: Based on Doolittle LU factorization for Ax=b
  ! Alex G. December 2009
  !-----------------------------------------------------------
  ! input ...
  ! a(n,n) - array of coefficients for matrix A
  ! n      - dimension
  ! output ...
  ! c(n,n) - inverse matrix of A
  ! comments ...
  ! the original matrix a(n,n) will be destroyed 
  ! during the calculation
  !===========================================================
  implicit none 
    integer n
    real(spl_rk) ::  a(n,n), c(n,n)
    real(spl_rk) ::  L(n,n), U(n,n), b(n), d(n), x(n)
    real(spl_rk) ::  coeff
    integer :: i, j, k

    ! step 0: initialization for matrices L and U and b
    ! Fortran 90/95 aloows such operations on matrices
    L=0.0
    U=0.0
    b=0.0

    ! step 1: forward elimination
    do k=1, n-1
       do i=k+1,n
          coeff=a(i,k)/a(k,k)
          L(i,k) = coeff
          do j=k+1,n
             a(i,j) = a(i,j)-coeff*a(k,j)
          end do
       end do
    end do

    ! Step 2: prepare L and U matrices 
    ! L matrix is a matrix of the elimination coefficient
    ! + the diagonal elements are 1.0
    do i=1,n
      L(i,i) = 1.0
    end do
    ! U matrix is the upper triangular part of A
    do j=1,n
      do i=1,j
        U(i,j) = a(i,j)
      end do
    end do

    ! Step 3: compute columns of the inverse matrix C
    do k=1,n
      b(k)=1.0
      d(1) = b(1)
    ! Step 3a: Solve Ld=b using the forward substitution
      do i=2,n
        d(i)=b(i)
        do j=1,i-1
          d(i) = d(i) - L(i,j)*d(j)
        end do
      end do
    ! Step 3b: Solve Ux=d using the back substitution
      x(n)=d(n)/U(n,n)
      do i = n-1,1,-1
        x(i) = d(i)
        do j=n,i+1,-1
          x(i)=x(i)-U(i,j)*x(j)
        end do
        x(i) = x(i)/u(i,i)
      end do
    ! Step 3c: fill the solutions x(n) into column k of C
      do i=1,n
        c(i,k) = x(i)
      end do
      b(k)=0.0
    end do
  end subroutine spl_direct_inverse
  ! ..................................................

end module spl_m_calculus
