
ncells_list = [((2**k)*10, (2**k)*10, (2**k)*10) for k in range(0,6)]

for ncells in ncells_list:
    import gc
    from mpi4py                         import MPI
    from psydac.api.discretization      import discretize
    from psydac.api.settings            import PSYDAC_BACKENDS
    from sympde.topology                import Cube, element_of, Derham
    from sympde.expr                    import BilinearForm, integral
    
    per = [False, False, False]
    deg = [3,3,3]
    mult = [1,1,1]
    comm = MPI.COMM_WORLD

    D = Cube('D')
    derham = Derham(D)

    Dh = discretize(D, ncells=ncells, periodic=per, comm=comm)
    derham_h = discretize(derham, Dh, degree=deg, multiplicity=mult)
    V0h = derham_h.V0 #TensorFemSpace
    #V1h = derham_h.V1 #VectorFemSpace

    u = element_of(derham.V0, name='u') # trial function
    w = element_of(derham.V0, name='w') # test function

    be=PSYDAC_BACKENDS['pyccel-gcc']


    a = BilinearForm((u,w), integral(D, u*w))
    ah = discretize(a, Dh, [derham_h.V0, derham_h.V0], backend=be)
    M = ah.assemble(sum_factorization = True)
    M.remove_spurious_entries()
    M.update_ghost_regions()

    M.topetsc('performance_petsc')

    comm.Barrier()

    # Delete the reference
    del M

    # Force the Garbage Collector to release unreferenced memory
    gc.collect()

