#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from typing import TypeVar

T = TypeVar('T', float, complex)

#========================================================================================================
def transpose_1d(M  : "T[:,:]",
                 Mt : "T[:,:]",
                 n  : "int64[:]",
                 nc : "int64[:]",
                 gp : "int64[:]",
                 p  : "int64[:]",
                 dm : "int64[:]",
                 cm : "int64[:]",
                 nd : "int64[:]",
                 ndT: "int64[:]",
                 si : "int64[:]",
                 sk : "int64[:]",
                 sl : "int64[:]"):

    #$omp parallel default(private) shared(Mt,M) firstprivate( n,nc,gp,p,dm,cm,nd,ndT,si,sk,sl)
    d1 = gp[0] - p[0]
    e1 = nd[0] - sl[0]
    #$omp for schedule(static) collapse(1)
    for x1 in range(n[0]):

        j1 = dm[0] * gp[0] + x1
        for l1 in range(nd[0]):

            i1 = si[0] + cm[0] * (x1 // dm[0]) + l1 + d1

            k1 = sk[0] + x1 % dm[0] - dm[0] * (l1 // cm[0])

            if k1 < ndT[0] and k1 > -1 and l1  < e1 and i1 < nc[0]:
                Mt[j1, l1 + sl[0]] = M[i1, k1]
    #$omp end parallel
    return

#========================================================================================================
def transpose_2d(M  : "T[:,:,:,:]",
                 Mt : "T[:,:,:,:]",
                 n  : "int64[:]",
                 nc : "int64[:]",
                 gp : "int64[:]",
                 p  : "int64[:]",
                 dm : "int64[:]",
                 cm : "int64[:]",
                 nd : "int64[:]",
                 ndT: "int64[:]",
                 si : "int64[:]",
                 sk : "int64[:]",
                 sl : "int64[:]"):

    #$omp parallel default(private) shared(Mt,M) firstprivate( n,nc,gp,p,dm,cm,nd,ndT,si,sk,sl)
    d1 = gp[0] - p[0]
    d2 = gp[1] - p[1]

    e1 = nd[0] - sl[0]
    e2 = nd[1] - sl[1]

    #$omp for schedule(static) collapse(2)
    for x1 in range(n[0]):
        for x2 in range(n[1]):

            j1 = dm[0]*gp[0] + x1
            j2 = dm[1]*gp[1] + x2

            for l1 in range(nd[0]):
                for l2 in range(nd[1]):

                    i1 = si[0] + cm[0]*(x1//dm[0]) + l1 + d1
                    i2 = si[1] + cm[1]*(x2//dm[1]) + l2 + d2

                    k1 = sk[0] + x1%dm[0]-dm[0]*(l1//cm[0])
                    k2 = sk[1] + x2%dm[1]-dm[1]*(l2//cm[1])

                    if k1<ndT[0] and k1>-1 and k2<ndT[1] and k2>-1 and l1<e1 and l2<e2 and i1<nc[0] and i2<nc[1]:
                        Mt[j1,j2, l1 + sl[0],l2 + sl[1]] = M[i1,i2, k1,k2]
    #$omp end parallel
    return

#========================================================================================================
def transpose_3d(M  : "T[:,:,:,:,:,:]",
                 Mt : "T[:,:,:,:,:,:]",
                 n  : "int64[:]",
                 nc : "int64[:]",
                 gp : "int64[:]",
                 p  : "int64[:]",
                 dm : "int64[:]",
                 cm : "int64[:]",
                 nd : "int64[:]",
                 ndT: "int64[:]",
                 si : "int64[:]",
                 sk : "int64[:]",
                 sl : "int64[:]"):

    #$omp parallel default(private) shared(Mt,M) firstprivate(n,nc,gp,p,dm,cm,nd,ndT,si,sk,sl)
    d1 = gp[0] - p[0]
    d2 = gp[1] - p[1]
    d3 = gp[2] - p[2]

    e1 = nd[0] - sl[0]
    e2 = nd[1] - sl[1]
    e3 = nd[2] - sl[2]

    #$omp for schedule(static) collapse(3)
    for x1 in range(n[0]):
        for x2 in range(n[1]):
            for x3 in range(n[2]):

                j1 = dm[0]*gp[0] + x1
                j2 = dm[1]*gp[1] + x2
                j3 = dm[2]*gp[2] + x3

                for l1 in range(nd[0]):
                    for l2 in range(nd[1]):
                        for l3 in range(nd[2]):

                            i1 = si[0] + cm[0]*(x1//dm[0]) + l1 + d1
                            i2 = si[1] + cm[1]*(x2//dm[1]) + l2 + d2
                            i3 = si[2] + cm[2]*(x3//dm[2]) + l3 + d3

                            k1 = sk[0] + x1%dm[0]-dm[0]*(l1//cm[0])
                            k2 = sk[1] + x2%dm[1]-dm[1]*(l2//cm[1])
                            k3 = sk[2] + x3%dm[2]-dm[2]*(l3//cm[2])

                            if k1<ndT[0] and k1>-1 and k2<ndT[1] and k2>-1 and k3<ndT[2] and k3>-1\
                                and l1<e1 and l2<e2 and l3<e3 and i1<nc[0] and i2<nc[1] and i3<nc[2]:
                                Mt[j1,j2,j3, l1 + sl[0],l2 + sl[1],l3 + sl[2]] = M[i1,i2,i3, k1,k2,k3]
    #$omp end parallel
    return

#========================================================================================================
def interface_transpose_1d(M  : "T[:,:]",
                           Mt : "T[:,:]",
                           n  : "int64[:]",
                           nc : "int64[:]",
                           gp : "int64[:]",
                           p  : "int64[:]",
                           dm : "int64[:]",
                           cm : "int64[:]",
                           nd : "int64[:]",
                           ndT: "int64[:]",
                           si : "int64[:]",
                           sk : "int64[:]",
                           sl : "int64[:]"):

    #$ omp parallel default(private) shared(Mt,M) firstprivate(n,nc,gp,p,dm,cm,nd,ndT,si,sk,sl)
    d1 = gp[0] - p [0]
    e1 = nd[0] - sl[0]

    #$ omp for schedule(static) collapse(1)
    for x1 in range(n[0]):

                j1 = gp[0] + x1

                for l1 in range(nd[0]):

                            i1 = si[0] + cm[0]*(x1//dm[0]) + l1 + d1

                            k1 = sk[0] + x1%dm[0] - dm[0]*(l1//cm[0])

                            if k1<ndT[0] and k1>-1 and l1<e1 and i1<nc[0]:
                                Mt[j1, l1+sl[0]] = M[i1, k1]

    #$ omp end parallel
    return

#========================================================================================================
def interface_transpose_2d(M  : "T[:,:,:,:]",
                           Mt : "T[:,:,:,:]",
                           n  : "int64[:]",
                           nc : "int64[:]",
                           gp : "int64[:]",
                           p  : "int64[:]",
                           dm : "int64[:]",
                           cm : "int64[:]",
                           nd : "int64[:]",
                           ndT: "int64[:]",
                           si : "int64[:]",
                           sk : "int64[:]",
                           sl : "int64[:]"):

    #$ omp parallel default(private) shared(Mt,M) firstprivate(n,nc,gp,p,dm,cm,nd,ndT,si,sk,sl)
    d1 = gp[0] - p[0]
    d2 = gp[1] - p[1]

    e1 = nd[0] - sl[0]
    e2 = nd[1] - sl[1]

    #$ omp for schedule(static) collapse(2)
    for x1 in range(n[0]):
        for x2 in range(n[1]):

                j1 = gp[0] + x1
                j2 = gp[1] + x2

                for l1 in range(nd[0]):
                    for l2 in range(nd[1]):

                            i1 = si[0] + cm[0]*(x1//dm[0]) + l1 + d1
                            i2 = si[1] + cm[1]*(x2//dm[1]) + l2 + d2

                            k1 = sk[0] + x1%dm[0] - dm[0]*(l1//cm[0])
                            k2 = sk[1] + x2%dm[1] - dm[1]*(l2//cm[1])

                            if k1<ndT[0] and k1>-1 and k2<ndT[1] and k2>-1\
                              and l1<e1 and l2<e2 and i1<nc[0] and i2<nc[1]:
                                Mt[j1,j2, l1+sl[0],l2+sl[1]] = M[i1,i2, k1,k2]

    #$ omp end parallel
    return

#========================================================================================================
def interface_transpose_3d(M  : "T[:,:,:,:,:,:]",
                           Mt : "T[:,:,:,:,:,:]",
                           n  : "int64[:]",
                           nc : "int64[:]",
                           gp : "int64[:]",
                           p  : "int64[:]",
                           dm : "int64[:]",
                           cm : "int64[:]",
                           nd : "int64[:]",
                           ndT: "int64[:]",
                           si : "int64[:]",
                           sk : "int64[:]",
                           sl : "int64[:]"):

    #$ omp parallel default(private) shared(Mt,M) firstprivate(n,nc,gp,p,dm,cm,nd,ndT,si,sk,sl)
    d1 = gp[0] - p[0]
    d2 = gp[1] - p[1]
    d3 = gp[2] - p[2]

    e1 = nd[0] - sl[0]
    e2 = nd[1] - sl[1]
    e3 = nd[2] - sl[2]

    #$ omp for schedule(static) collapse(3)
    for x1 in range(n[0]):
        for x2 in range(n[1]):
            for x3 in range(n[2]):

                j1 = gp[0] + x1
                j2 = gp[1] + x2
                j3 = gp[2] + x3

                for l1 in range(nd[0]):
                    for l2 in range(nd[1]):
                        for l3 in range(nd[2]):

                            i1 = si[0] + cm[0]*(x1//dm[0]) + l1 + d1
                            i2 = si[1] + cm[1]*(x2//dm[1]) + l2 + d2
                            i3 = si[2] + cm[2]*(x3//dm[2]) + l3 + d3

                            k1 = sk[0] + x1%dm[0] - dm[0]*(l1//cm[0])
                            k2 = sk[1] + x2%dm[1] - dm[1]*(l2//cm[1])
                            k3 = sk[2] + x3%dm[2] - dm[2]*(l3//cm[2])

                            if k1<ndT[0] and k1>-1 and k2<ndT[1] and k2>-1 and k3<ndT[2] and k3>-1\
                              and l1<e1 and l2<e2 and l3<e3 and i1<nc[0] and i2<nc[1] and i3<nc[2]:
                                Mt[j1,j2,j3, l1+sl[0],l2+sl[1],l3+sl[2]] = M[i1,i2,i3, k1,k2,k3]

    #$ omp end parallel
    return
