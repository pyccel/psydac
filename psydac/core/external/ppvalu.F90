      real function ppvalu (break, coef, l, k, x, jderiv )
!  from  * a practical guide to splines *  by c. de boor
!alls  interv
!alculates value at  x  of  jderiv-th derivative of pp fct from pp-repr
!
!******  i n p u t  ******
!  break, coef, l, k.....forms the pp-representation of the function  f
!        to be evaluated. specifically, the j-th derivative of  f  is
!        given by
!
!     (d**j)f(x) = coef(j+1,i) + h*(coef(j+2,i) + h*( ... (coef(k-1,i) +
!                             + h*coef(k,i)/(k-j-1))/(k-j-2) ... )/2)/1
!
!        with  h = x - break(i),  and
!
!       i  =  max( 1 , max( j ,  break(j) .le. x , 1 .le. j .le. l ) ).
!
!  x.....the point at which to evaluate.
!  jderiv.....integer giving the order of the derivative to be evaluat-
!        ed.  a s s u m e d  to be zero or positive.
!
!******  o u t p u t  ******
!  ppvalu.....the value of the (jderiv)-th derivative of  f  at  x.
!
!******  m e t h o d  ******
!     the interval index  i , appropriate for  x , is found through a
!  call to  interv . the formula above for the  jderiv-th derivative
!  of  f  is then evaluated (by nested multiplication).
!
      integer jderiv,k,l,   i,m,ndummy
      real(8) break(l+1),coef(k,l),x,   fmmjdr,h
      ppvalu = 0.
      fmmjdr = k - jderiv
!              derivatives of order  k  or higher are identically zero.
      if (fmmjdr .le. 0.)               go to 99
!
!              find index  i  of largest breakpoint to the left of  x .
      call interv ( break, l+1, x, i, ndummy )
      print *, "break : ", break(i)
!
!      Evaluate  jderiv-th derivative of  i-th polynomial piece at  x .
      h = x - break(i)
      m = k
    9    ppvalu = (ppvalu/fmmjdr)*h + coef(m,i)
         m = m - 1
         fmmjdr = fmmjdr - 1.
         if (fmmjdr .gt. 0.)            go to 9
   99                                   return
      end
