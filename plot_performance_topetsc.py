import numpy as np
import matplotlib.pyplot as plt

#this is a very messy script, mostly composed of fragments to pass to ipython. 

nprocs_list = [1,4,8,16,32]
nrows_list = [1728, 10648, 74088, 551368, 4251528]
ncells_list = [((2**k)*10, (2**k)*10, (2**k)*10) for k in range(0,5)] #tok cluster only k=0,1,2,3,4, MAX 185GB per node

time_kernel = np.zeros((len(nprocs_list), len(nrows_list)))
time_setValuesIJV = np.zeros((len(nprocs_list), len(nrows_list)))
time_assemble = np.zeros((len(nprocs_list), len(nrows_list)))

for i in range(len(nprocs_list)):
    for j in range(len(nrows_list)):

        data = np.load(f'performance_petsc_serial/petsc_performance_nprocs={nprocs_list[i]}_nrows={nrows_list[j]}.npz', allow_pickle=True)
        
        time_kernel[i,j] = data['time_kernel']
        time_setValuesIJV[i,j] = data['time_setValuesIJV']
        time_assemble[i,j] = data['time_assemble']

time_kernel = np.array([
    [3.44014168e-03, 1.86841488e-02, 1.36070251e-01, 1.05291104e+00, 8.26110077e+00],
    [1.80935860e-03, 6.01798296e-03, 3.86480689e-02, 2.93982983e-01, 2.30626690e+00],
    [5.11139631e-04, 2.96166539e-03, 2.09342837e-02, 1.54421955e-01, 1.20589867e+00],
    [6.44862652e-04, 1.44958496e-03, 1.16356909e-02, 8.47364962e-02, 6.64496660e-01],
    [2.72303820e-04, 7.27772713e-04, 7.04875588e-03, 5.09125888e-02, 4.01131377e-01]
])

time_setValuesIJV = np.array([
    [6.64901733e-02, 4.99193907e-01, 4.21932697e+00, 3.56401129e+01, 3.17807595e+02],
    [2.22587585e-02, 1.25872850e-01, 1.06521857e+00, 8.99232519e+00, 7.71693090e+01],
    [8.34125280e-03, 6.30261004e-02, 5.33503979e-01, 4.47380212e+00, 3.90298276e+01],
    [4.84825671e-03, 3.05047482e-02, 2.65323520e-01, 2.29217426e+00, 1.95350892e+01],
    [2.24738568e-03, 1.51470453e-02, 1.28675453e-01, 1.18095405e+00, 1.00693423e+01]
])

time_assemble = np.array([
    [2.02119350e-02, 1.47285938e-01, 1.17021513e+00, 9.42194057e+00, 9.30189772e+01],
    [2.15317011e-02, 4.92064357e-02, 3.36610734e-01, 2.56101841e+00, 2.39285755e+01],
    [7.41338730e-03, 2.86393166e-02, 1.81091219e-01, 1.38666886e+00, 1.42942967e+01],
    [2.26006955e-02, 1.77826732e-02, 1.08603835e-01, 8.27372923e-01, 6.90622087e+00],
    [6.20394945e-03, 1.05539784e-02, 6.29525781e-02, 4.60187472e-01, 4.19238978e+00]
])

time_kernel_total = np.array([time_kernel[i,:]*nprocs_list[i] for i in range(len(nprocs_list))])
time_setValuesIJV_total = np.array([time_setValuesIJV[i,:]*nprocs_list[i] for i in range(len(nprocs_list))])
time_assemble_total = np.array([time_assemble[i,:]*nprocs_list[i] for i in range(len(nprocs_list))])

# Plotting Configuration
datasets = [
    (time_kernel_total, "Kernel time"),
    (time_setValuesIJV_total, "SetValuesIJV time"),
    (time_assemble_total, "Assembly time")
]
# Change global font size
plt.rcParams.update({'font.size': 11})  # Set the default font size


fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=False)

for i, (data, title) in enumerate(datasets):
    ax = axes[i]

    for j, nproc in enumerate(nprocs_list):
        ax.plot(nrows_list, data[j,:], label=f'#procs={nproc}', marker='^', markerfacecolor='none', markersize=10)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(r'number of rows')
    ax.set_ylabel('Time [s]')
    ax.grid(True, which="major", ls="-", alpha=1)
    ax.grid(True, which="minor", ls="-", alpha=0.5)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.legend()

plt.tight_layout()
plt.show()







for i in range(len(nrows_list)):
    nrows = nrows_list[i]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 1. Strong Scaling Plot ---
    # speedup (T_ref / T_p) for the largest problem size (last column)

    speedup_kernel = time_kernel[0,i]*nprocs_list[0] / (time_kernel[:,i]*nprocs_list[i])
    axes[0].plot(nprocs_list, speedup_kernel, label='kernel', color='b', marker='^', markerfacecolor='none', markersize=10)
    speedup_setValuesIJV = time_setValuesIJV[0,i]*nprocs_list[0] / (time_setValuesIJV[:,i]*nprocs_list[i])
    axes[0].plot(nprocs_list, speedup_setValuesIJV, label='setValuesIJV', color='g', marker='^', markerfacecolor='none', markersize=10)
    speedup_assemble = time_assemble[0,i]*nprocs_list[0] / (time_assemble[:,i]*nprocs_list[i])
    axes[0].plot(nprocs_list, speedup_assemble, label='assemble', color='r', marker='^', markerfacecolor='none', markersize=10)


    axes[0].plot(nprocs_list, np.array(nprocs_list)/nprocs_list[0], '--', color='gray', label='ideal speedup')
    axes[0].set_title(f"Strong scaling (#rows={nrows})")
    axes[0].set_xlabel("Number of processors")
    axes[0].set_ylabel("Speedup")
    axes[0].legend()
    axes[0].grid(True, which="major", ls="-", alpha=1)
    axes[0].grid(True, which="minor", ls="-", alpha=0.5)
    axes[0].set_xticks(nprocs_list, [str(n) for n in nprocs_list])
    #axes[0].set_xticks(nprocs_list)
    #axes[0].set_xscale('log')


    # --- 2. Weak Scaling Plot ---
    # We look at the diagonal or near-diagonal where problem size increases with procs.
    # Since nrows grows much faster than nprocs, we'll pick indices that simulate 
    # a relatively constant work-per-processor if possible.
    # For simplicity, let's plot the time for specific nrows across all procs.


    axes[1].plot(nprocs_list, time_kernel[:, i]*nprocs_list[:], label='kernel', color='b', marker='^', markerfacecolor='none', markersize=10)
    axes[1].plot(nprocs_list, time_setValuesIJV[:, i]*nprocs_list[:], label='setValuesIJV', color='g', marker='^', markerfacecolor='none', markersize=10)
    axes[1].plot(nprocs_list, time_assemble[:, i]*nprocs_list[:], label='assemble', color='r', marker='^', markerfacecolor='none', markersize=10)



    axes[1].set_title(f"Time vs number of processors (#rows={nrows})")
    axes[1].set_xlabel("Number of processors")
    axes[1].set_ylabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True, which="major", ls="-", alpha=1)
    axes[1].grid(True, which="minor", ls="-", alpha=0.5)
    #axes[1].tick_params(axis='x', which='minor', length=0)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    #axes[1].set_xticks(nprocs_list, minor=False)
    #axes[1].tick_params(axis='x', which='minor', length=0)
    
    axes[1].set_xticks(nprocs_list, [str(n) for n in nprocs_list], minor=False)
    axes[1].set_xticks(nprocs_list, ['' for n in nprocs_list], minor=True)
    axes[1].tick_params(axis='x', which='minor', length=0)

    plt.tight_layout()
    plt.savefig(f'scaling_nrows={nrows}.png')
    plt.close()