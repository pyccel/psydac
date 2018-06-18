      subroutine bsplpp ( t, bcoef, n, k, scrtch, break, coef, l )
!  from  * a practical guide to splines *  by c. de Boor (7 may 92)
!alls  bsplvb
!
!onverts the b-representation  t, bcoef, n, k  of some spline into its
!  pp-representation  break, coef, l, k .
!
!******  i n p u t  ******
!  t.....knot sequence, of length  n+k
!  bcoef.....b-spline coefficient sequence, of length  n
!  n.....length of  bcoef  and  dimension of spline space  spline(k,t)
!  k.....order of the spline
!
!  w a r n i n g  . . .  the restriction   k .le. kmax (= 20)   is impo-
!        sed by the arbitrary dimension statement for  biatx  below, but
!        is  n o w h e r e   c h e c k e d   for.
!
!******  w o r k   a r e a  ******
!  scrtch......of size  (k,k) , needed to contain bcoeffs of a piece of
!        the spline and its  k-1  derivatives
!
!******  o u t p u t  ******
!  break.....breakpoint sequence, of length  l+1, contains (in increas-
!        ing order) the distinct points in the sequence  t(k),...,t(n+1)
!  coef.....array of size (k,l), with  coef(i,j) = (i-1)st derivative of
!        spline at break(j) from the right
!  l.....number of polynomial pieces which make up the spline in the in-
!        terval  (t(k), t(n+1))
!
!******  m e t h o d  ******
!     for each breakpoint interval, the  k  relevant b-coeffs of the
!  spline are found and then differenced repeatedly to get the b-coeffs
!  of all the derivatives of the spline on that interval. the spline and
!  its first  k-1  derivatives are then evaluated at the left end point
!  of that interval, using  bsplvb  repeatedly to obtain the values of
!  all b-splines of the appropriate order at that point.
!
      integer k,l,n,   i,j,jp1,kmax,kmj,left,lsofar
      parameter (kmax = 20)
      real(8) bcoef(n),break(l+1),coef(k,l),t(n+k),   scrtch(k,k) &
                                            ,biatx(kmax),diff,factor,sum
!
      lsofar = 0
      break(1) = t(k)
      do 50 left=k,n
!                                find the next nontrivial knot interval.
         if (t(left+1) .eq. t(left))    go to 50
         lsofar = lsofar + 1
         break(lsofar+1) = t(left+1)
         if (k .gt. 1)                  go to 9
         coef(1,lsofar) = bcoef(left)
                                        go to 50
!        store the k b-spline coeff.s relevant to current knot interval
!                             in  scrtch(.,1) .
    9    do 10 i=1,k
   10       scrtch(i,1) = bcoef(left-k+i)
!
!        for j=1,...,k-1, compute the  k-j  b-spline coeff.s relevant to
!        current knot interval for the j-th derivative by differencing
!        those for the (j-1)st derivative, and store in scrtch(.,j+1) .
         do 20 jp1=2,k
            j = jp1 - 1
            kmj = k - j
            do 20 i=1,kmj
               diff = t(left+i) - t(left+i - kmj)
               if (diff .gt. 0.)  scrtch(i,jp1) = &
                             (scrtch(i+1,j)-scrtch(i,j))/diff
   20          continue
!
!        for  j = 0, ..., k-1, find the values at  t(left)  of the  j+1
!        b-splines of order  j+1  whose support contains the current
!        knot interval from those of order  j  (in  biatx ), then comb-
!        ine with the b-spline coeff.s (in scrtch(.,k-j) ) found earlier
!        to compute the (k-j-1)st derivative at  t(left)  of the given
!        spline.
!           note. if the repeated calls to  bsplvb  are thought to gene-
!        rate too much overhead, then replace the first call by
!           biatx(1) = 1.
!        and the subsequent call by the statement
!           j = jp1 - 1
!        followed by a direct copy of the lines
!           deltar(j) = t(left+j) - x
!                  ......
!           biatx(j+1) = saved
!        from  bsplvb . deltal(kmax)  and  deltar(kmax)  would have to
!        appear in a dimension statement, of course.
!
         call bsplvb ( t, 1, 1, t(left), left, biatx )
         coef(k,lsofar) = scrtch(1,k)
         do 30 jp1=2,k
            call bsplvb ( t, jp1, 2, t(left), left, biatx )
            kmj = k+1 - jp1
            sum = 0.
            do 28 i=1,jp1
   28          sum = biatx(i)*scrtch(i,kmj) + sum
   30       coef(kmj,lsofar) = sum
   50    continue
      l = lsofar
        if (k .eq. 1)                     return
        factor = 1.
        do 60 i=2,k
             factor = factor*float(k+1-i)
             do 60 j=1,lsofar
   60       coef(i,j) = coef(i,j)*factor
                                        return
      end subroutine bsplpp
