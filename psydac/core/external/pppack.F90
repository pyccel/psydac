!> @brief 
!> Module for Splines 
!> @details
!> basic functions and routines for B-Splines
!> in 1D, 2D and 3D
!>
!> This file contains two modules
!> BSPLINE : basic functions and routines
!> BSP     : specific implementations for 1D,2D,3D
!> Externaly, the user should call BSP

module pppack
implicit none
contains
include "interv.F90"
include "bsplpp.F90"
include "bsplvb.F90"
include "ppvalu.F90"
        

! .......................................................
!> @brief     Determine non zero elements 
!>
!> @param[in] n number of control points  - 1
!> @param[in] p spline degree 
!> @param[in] U Knot vector 
!> @param[in] n_elements number of non-zero elements 
!> @param[in] grid the corresponding grid
! .......................................................

end module pppack
