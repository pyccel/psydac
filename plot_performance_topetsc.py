import numpy as np

nprocs_list = [4,8,16,32]
nrows_list = [1728, 10648, 74088, 551368, 4251528]

time_kernel = np.zeros((len(nprocs_list), len(nrows_list)))
time_setValuesIJV = np.zeros((len(nprocs_list), len(nrows_list)))
time_assemble = np.zeros((len(nprocs_list), len(nrows_list)))

for i in range(len(nprocs_list)):
    for j in range(len(nrows_list)):

        data = np.load(f'performance_petsc/petsc_performance_nprocs={nprocs_list[i]}_nrows={nrows_list[j]}', allow_pickle=True)
        
        time_kernel[i,j] = data['time_kernel']
        time_setValuesIJV[i,j] = data['time_setValuesIJV']
        time_assemble[i,j] = data['time_assemble']

