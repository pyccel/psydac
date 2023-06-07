from pyccel.decorators import template

#========================================================================================================
@template(name='T', types=['float[:,:]', 'complex[:,:]'])
def transpose_1d( M:'T', Mt:'T', n1:"int64", nc1:"int64", gp1:"int64", p1:"int64",
                  dm1:"int64", cm1:"int64", nd1:"int64", ndT1:"int64", si1:"int64", sk1:"int64", sl1:"int64"):

    #$omp parallel default(private) shared(Mt,M) firstprivate( n1,nc1,gp1,p1,dm1,cm1,nd1,ndT1,si1,sk1,sl1)

    d1 = gp1-p1

    #$omp for schedule(static)
    for x1 in range(n1):
        j1 = dm1*gp1 + x1
        for l1 in range(nd1):

            i1 = si1 + cm1*(x1//dm1) + l1 + d1
            k1 = sk1 + x1%dm1-dm1*(l1//cm1)

            if k1<ndT1 and k1>-1 and l1+sl1<nd1 and i1<nc1:
                Mt[j1, l1+sl1] = M[i1, k1]
    #$omp end parallel
    return

#========================================================================================================
@template(name='T', types=['float[:,:,:,:]', 'complex[:,:,:,:]'])
def transpose_2d( M:'T', Mt:'T', n1:"int64", n2:"int64", nc1:"int64", nc2:"int64",
                  gp1:"int64", gp2:"int64", p1:"int64", p2:"int64", dm1:"int64", dm2:"int64",
                  cm1:"int64", cm2:"int64", nd1:"int64", nd2:"int64", ndT1:"int64", ndT2:"int64",
                  si1:"int64", si2:"int64", sk1:"int64", sk2:"int64", sl1:"int64", sl2:"int64"):

    #$omp parallel default(private) shared(Mt,M) firstprivate( n1,n2,nc1,nc2,gp1,gp2,p1,p2,dm1,dm2,cm1,cm2,nd1,nd2,ndT1,ndT2,si1,si2,sk1,sk2,sl1,sl2)

    d1 = gp1-p1
    d2 = gp2-p2

    #$omp for schedule(static) collapse(2)
    for x1 in range(n1):
        for x2 in range(n2):

            j1 = dm1*gp1 + x1
            j2 = dm2*gp2 + x2
     
            for l1 in range(nd1):
                for l2 in range(nd2):

                    i1 = si1 + cm1*(x1//dm1) + l1 + d1
                    i2 = si2 + cm2*(x2//dm2) + l2 + d2

                    k1 = sk1 + x1%dm1-dm1*(l1//cm1)
                    k2 = sk2 + x2%dm2-dm2*(l2//cm2)

                    if k1<ndT1 and k1>-1 and k2<ndT2 and k2>-1\
                        and l1+sl1<nd1 and l2+sl2<nd2 and i1<nc1 and i2<nc2:
                        Mt[j1,j2, l1+sl1,l2+sl2] = M[i1,i2, k1,k2]
    #$omp end parallel
    return
#========================================================================================================
@template(name='T', types=['float[:,:,:,:,:,:]', 'complex[:,:,:,:,:,:]'])
def transpose_3d(M:'T', Mt:'T',
                 n1:"int64", n2:"int64", n3:"int64",
                 nc1:"int64", nc2:"int64", nc3:"int64",
                 gp1:"int64", gp2:"int64", gp3:"int64",
                 p1:"int64", p2:"int64", p3:"int64",
                 dm1:"int64", dm2:"int64", dm3:"int64",
                 cm1:"int64", cm2:"int64", cm3:"int64",
                 nd1:"int64", nd2:"int64", nd3:"int64",
                 ndT1:"int64", ndT2:"int64", ndT3:"int64",
                 si1:"int64", si2:"int64", si3:"int64",
                 sk1:"int64", sk2:"int64", sk3:"int64",
                 sl1:"int64", sl2:"int64", sl3:"int64"):

    #$omp parallel default(private) shared(Mt,M) firstprivate( n1,n2,n3,nc1,nc2,nc3,gp1,gp2,gp3,p1,p2,p3,dm1,dm2,dm3,cm1,cm2,cm3,nd1,nd2,nd3,ndT1,ndT2,ndT3,si1,si2,si3,sk1,sk2,sk3,sl1,sl2,sl3)
    d1 = gp1-p1
    d2 = gp2-p2
    d3 = gp3-p3
    #$omp for schedule(static) collapse(3)
    for x1 in range(n1):
        for x2 in range(n2):
            for x3 in range(n3):

                j1 = dm1*gp1 + x1
                j2 = dm2*gp2 + x2
                j3 = dm3*gp3 + x3

                for l1 in range(nd1):
                    for l2 in range(nd2):
                        for l3 in range(nd3):

                            i1 = si1 + cm1*(x1//dm1) + l1 + d1
                            i2 = si2 + cm2*(x2//dm2) + l2 + d2
                            i3 = si3 + cm3*(x3//dm3) + l3 + d3

                            k1 = sk1 + x1%dm1-dm1*(l1//cm1)
                            k2 = sk2 + x2%dm2-dm2*(l2//cm2)
                            k3 = sk3 + x3%dm3-dm3*(l3//cm3)

                            if k1<ndT1 and k1>-1 and k2<ndT2 and k2>-1 and k3<ndT3 and k3>-1\
                                and l1+sl1<nd1 and l2+sl2<nd2 and l3+sl3<nd3 and i1<nc1 and i2<nc2 and i3<nc3:
                                Mt[j1,j2,j3, l1 + sl1,l2 + sl2,l3 + sl3] = M[i1,i2,i3, k1,k2,k3]
    #$omp end parallel
    return

#========================================================================================================
@template(name='T', types=['float[:,:]', 'complex[:,:]'])
def interface_transpose_1d( M:'T', Mt:'T', n1:"int64",
                            nc1:"int64", gp1:"int64", p1:"int64", dm1:"int64", cm1:"int64",
                            nd1:"int64", ndT1:"int64", si1:"int64",
                            sk1:"int64", sl1:"int64"):

    #$omp parallel default(private) shared(Mt,M) firstprivate( n1,nc1,gp1,p1,dm1,cm1,nd1,ndT1,si1,sk1,sl1)

    d1 = gp1-p1

    #$omp for schedule(static)
    for x1 in range(n1):
        j1 = dm1*gp1 + x1
        for l1 in range(nd1):
            i1 = si1 + cm1*(x1//dm1) + l1 + d1
            k1 = sk1 + x1%dm1-dm1*(l1//cm1)
            m1 = l1+sl1
            if k1<ndT1 and k1>-1 and m1<nd1 and i1<nc1:
                Mt[j1,m1] = M[i1,k1]

    #$omp end parallel
    return
#========================================================================================================
@template(name='T', types=['float[:,:,:,:]', 'complex[:,:,:,:]'])
def interface_transpose_2d( M:'T', Mt:'T', n1:"int64", n2:"int64",
                            nc1:"int64", nc2:"int64", gp1:"int64", gp2:"int64",
                            p1:"int64", p2:"int64", dm1:"int64", dm2:"int64",
                            cm1:"int64", cm2:"int64",nd1:"int64", nd2:"int64",
                            ndT1:"int64", ndT2:"int64", si1:"int64", si2:"int64",
                            sk1:"int64", sk2:"int64", sl1:"int64", sl2:"int64"):


    #$omp parallel default(private) shared(Mt,M) firstprivate( n1,n2,nc1,nc2,gp1,gp2,p1,p2,dm1,dm2,cm1,cm2,nd1,nd2,ndT1,ndT2,si1,si2,sk1,sk2,sl1,sl2)
    d1 = gp1-p1
    d2 = gp2-p2

    #$ omp for schedule(static) collapse(2)
    for x1 in range(n1):
        for x2 in range(n2):
            j1 = gp1 + x1
            j2 = gp2 + x2
            for l1 in range(nd1):
                for l2 in range(nd2):

                    i1 = si1 + cm1*(x1//dm1) + l1 + d1
                    i2 = si2 + cm2*(x2//dm2) + l2 + d2

                    k1 = sk1 + x1%dm1 - dm1*(l1//cm1)
                    k2 = sk2 + x2%dm2-dm2*(l2//cm2)

                    m1 = l1+sl1
                    m2 = l2+sl2
                    if k1<ndT1 and k1>-1 and k2<ndT2 and k2>-1\
                        and m1<nd1 and m2<nd2 and i1<nc1 and i2<nc2:
                        Mt[j1,j2, m1,m2] = M[i1,i2, k1,k2]

    #$ omp end parallel
    return
#========================================================================================================
@template(name='T', types=['float[:,:,:,:,:,:]', 'complex[:,:,:,:,:,:]'])
def interface_transpose_3d( M:'T', Mt:'T',
                            n1:"int64", n2:"int64", n3:"int64",
                            nc1:"int64", nc2:"int64", nc3:"int64",
                            gp1:"int64", gp2:"int64", gp3:"int64",
                            p1:"int64", p2:"int64", p3:"int64",
                            dm1:"int64", dm2:"int64", dm3:"int64",
                            cm1:"int64", cm2:"int64", cm3:"int64",
                            nd1:"int64", nd2:"int64", nd3:"int64",
                            ndT1:"int64", ndT2:"int64", ndT3:"int64",
                            si1:"int64", si2:"int64", si3:"int64",
                            sk1:"int64", sk2:"int64", sk3:"int64",
                            sl1:"int64", sl2:"int64", sl3:"int64"):

    #$ omp parallel default(private) shared(Mt,M) firstprivate(n1,n2,n3,nc1,nc2,nc3,gp1,gp2,gp3,p1,p2,p3,dm1,dm2,dm3,&
    #$ cm1,cm2,cm3,nd1,nd2,nd3,ndT1,ndT2,ndT3,si1,si2,si3,sk1,sk2,sk3,sl1,sl2,sl3)
    d1 = gp1-p1
    d2 = gp2-p2
    d3 = gp3-p3

    #$ omp for schedule(static) collapse(3)
    for x1 in range(n1):
        for x2 in range(n2):
            for x3 in range(n3):

                j1 = gp1 + x1
                j2 = gp2 + x2
                j3 = gp3 + x3

                for l1 in range(nd1):
                    for l2 in range(nd2):
                        for l3 in range(nd3):

                            i1 = si1 + cm1*(x1//dm1) + l1 + d1
                            i2 = si2 + cm2*(x2//dm2) + l2 + d2
                            i3 = si3 + cm3*(x3//dm3) + l3 + d3

                            k1 = sk1 + x1%dm1 - dm1*(l1//cm1)
                            k2 = sk2 + x2%dm2-dm2*(l2//cm2)
                            k3 = sk3 + x3%dm3-dm3*(l3//cm3)

                            m1 = l1+sl1
                            m2 = l2+sl2
                            m3 = l3+sl3

                            if k1<ndT1 and k1>-1 and k2<ndT2 and k2>-1 and k3<ndT3 and k3>-1\
                              and m1<nd1 and m2<nd2 and m3<nd3 and i1<nc1 and i2<nc2 and i3<nc3:
                                Mt[j1,j2,j3,m1,m2,m3] = M[i1,i2,i3, k1,k2,k3]

    #$ omp end parallel
    return
