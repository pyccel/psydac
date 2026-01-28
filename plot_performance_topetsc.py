import numpy as np
import matplotlib.pyplot as plt

#this is a very messy script, mostly composed of fragments to pass to ipython. 

nprocs_list = [1,4,8,16,32]
nrows_list = [1728, 10648, 74088, 551368, 4251528]
ncells_list = [((2**k)*10, (2**k)*10, (2**k)*10) for k in range(0,5)] #tok cluster only k=0,1,2,3,4, MAX 185GB per node

time_kernel = np.empty((len(nprocs_list), len(nrows_list), nprocs_list[-1]), dtype=object)
time_setValuesIJV = np.empty((len(nprocs_list), len(nrows_list), nprocs_list[-1]), dtype=object)
time_assemble = np.empty((len(nprocs_list), len(nrows_list), nprocs_list[-1]), dtype=object)

for i in range(len(nprocs_list)):
    for j in range(len(nrows_list)):
        for k in range(nprocs_list[i]):

            data = np.load(f'performance_petsc/petsc_performance_proc={k}_of_{nprocs_list[i]}_nrows={nrows_list[j]}.npz', allow_pickle=True)
        
            time_kernel[i,j,k] = data['time_kernel'].item()
            time_setValuesIJV[i,j,k] = data['time_setValuesIJV'].item()
            time_assemble[i,j,k] = data['time_assemble'].item()


"""
# Plotting Configuration
datasets = [
    (time_kernel, "Kernel time"),
    (time_setValuesIJV, "SetValuesIJV time"),
    (time_assemble, "Assembly time")
]
# Change global font size
plt.rcParams.update({'font.size': 11})  # Set the default font size


fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=False)

for i, (data, title) in enumerate(datasets):
    ax = axes[i]

    for j, nproc in enumerate(nprocs_list):
        ax.plot(nrows_list, np.sum(data[j,:,:nproc], axis=-1), label=f'#procs={nproc}', marker='^', markerfacecolor='none', markersize=10)
        #for k in range(nproc):
        #    ax.plot(nrows_list, data[j,:,k])#, label=f'#procs={nproc}', marker='^', markerfacecolor='none', markersize=10)
    
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


time_kernel_sum = np.max(np.where(time_kernel == None, 0., time_kernel), axis=-1) #np.sum(np.where(time_kernel == None, 0., time_kernel), axis=-1)
time_setValuesIJV_sum = np.max(np.where(time_kernel == None, 0., time_kernel), axis=-1) #np.sum(np.where(time_setValuesIJV == None, 0., time_setValuesIJV), axis=-1)
time_assemble_sum = np.max(np.where(time_kernel == None, 0., time_kernel), axis=-1) #np.sum(np.where(time_assemble == None, 0., time_assemble), axis=-1)



for i in range(len(nrows_list)):
    nrows = nrows_list[i]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 1. Strong Scaling Plot ---
    # speedup (T_ref / T_p) for the largest problem size (last column)

    speedup_kernel = time_kernel_sum[0,i] / time_kernel_sum[:,i]
    axes[0].plot(nprocs_list, speedup_kernel, label='kernel', color='b', marker='^', markerfacecolor='none', markersize=10)

    speedup_setValuesIJV = time_setValuesIJV_sum[0,i] / time_setValuesIJV_sum[:,i]
    axes[0].plot(nprocs_list, speedup_setValuesIJV, label='setValuesIJV', color='g', marker='^', markerfacecolor='none', markersize=10)

    speedup_assemble = time_assemble_sum[0,i] / time_assemble_sum[:,i]
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


    '''axes[1].plot(nprocs_list, time_kernel_sum[:, i], label='kernel', color='b', marker='^', markerfacecolor='none', markersize=10)
    axes[1].plot(nprocs_list, time_setValuesIJV_sum[:, i], label='setValuesIJV', color='g', marker='^', markerfacecolor='none', markersize=10)
    axes[1].plot(nprocs_list, time_assemble_sum[:, i], label='assemble', color='r', marker='^', markerfacecolor='none', markersize=10)'''
    for k in range(len(nprocs_list)):
        p = np.arange(0, nprocs_list[k])
        axes[1].scatter(p, time_kernel[:, i, k])#, label='kernel', color='b', marker='^', markerfacecolor='none', markersize=10)
    #axes[1].scatter(nprocs_list, time_setValuesIJV_sum[:, i], label='setValuesIJV', color='g', marker='^', markerfacecolor='none', markersize=10)
    #axes[1].scatter(nprocs_list, time_assemble_sum[:, i], label='assemble', color='r', marker='^', markerfacecolor='none', markersize=10)
    '''axes[1].boxplot(time_kernel[:, i,:])#, label='kernel', color='b', marker='^', markerfacecolor='none', markersize=10)
    axes[1].boxplot(time_setValuesIJV[:, i,:])#, label='setValuesIJV', color='g', marker='^', markerfacecolor='none', markersize=10)
    axes[1].boxplot(time_assemble[:, i,:])#, label='assemble', color='r', marker='^', markerfacecolor='none', markersize=10)'''


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
    plt.savefig(f'scaling_nrows={nrows}_scatter.png')
    plt.close()
"""

#######################################

def plot_time_vs_nrows(nrows_list, nprocs_list, array, title):
    plt.rcParams.update({'font.size': 11})  
    fig = plt.figure(figsize=(10, 6))
    
    for i, nprocs in enumerate(nprocs_list):
        x_vals = []
        y_vals = []
        
        for j, nrows in enumerate(nrows_list):
            # Extract the list of times for this specific nprocs and nrows
            process_times = array[i][j]
            
            # Filter out 'None' values and keep only floats
            valid_times = [t for t in process_times if t is not None]
            
            # Append nrows for every valid timing found
            x_vals.extend([nrows] * len(valid_times))
            y_vals.extend(valid_times)
        
        # Plot each nprocs group as a separate scatter series
        plt.scatter(x_vals, y_vals, label=f'nprocs={nprocs}', alpha=0.9, s=40)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of rows')
    plt.ylabel('Time [s]')
    plt.title(f'Time of {title} vs. number of processes')
    #plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.grid(True, which="major", ls="-", alpha=1)
    plt.grid(True, which="minor", ls="-", alpha=0.5)
    
    plt.xticks(nrows_list, [str(n) for n in nrows_list], minor=False)
    plt.xticks(nrows_list, ['' for n in nrows_list], minor=True)
    plt.tick_params(axis='x', which='minor', length=0)

    plt.legend()
    plt.savefig(f'time_vs_nrows_{title}.pdf')

plot_time_vs_nrows(nrows_list, nprocs_list, time_kernel, 'kernel')
plot_time_vs_nrows(nrows_list, nprocs_list, time_setValuesIJV, 'setValuesIJV')
plot_time_vs_nrows(nrows_list, nprocs_list, time_assemble, 'assemble')

def plot_time_vs_nprocs(nrows_list, nprocs_list, array, title):
    plt.rcParams.update({'font.size': 11}) 
    plt.figure(figsize=(10, 6))
    
    # Iterate through nprocs (first dimension)
    for i, nprocs in enumerate(nprocs_list):
        x_vals = []
        y_vals = []
        
        # Iterate through nrows (second dimension)
        for j, nrows in enumerate(nrows_list):
            # Iterate through individual process times (third dimension)
            # We filter out None values here
            times = [t for t in array[i, j] if t is not None]
            
            # Create a list of the current nprocs for every valid time found
            x_vals.extend([nprocs] * len(times))
            y_vals.extend(times)
        
        # Plot each nprocs group with a label
        plt.scatter(x_vals, y_vals, alpha=0.9, label=f'nprocs={nprocs}', s=40)
    plt.title(f'Time of {title} vs. number of rows')
    plt.xlabel('number of processes')
    plt.ylabel('Time [s]')
    plt.yscale('log') # Log scale is recommended due to the wide range in your data
    plt.legend()

    plt.grid(True, which="major", ls="-", alpha=1)
    plt.grid(True, which="minor", ls="-", alpha=0.5)
    
    plt.xticks(nprocs_list, [str(n) for n in nprocs_list], minor=False)
    plt.xticks(nprocs_list, ['' for n in nprocs_list], minor=True)
    plt.tick_params(axis='x', which='minor', length=0)

    plt.savefig(f'time_vs_nprocs_{title}.pdf')
    plt.close()


plot_time_vs_nprocs(nrows_list, nprocs_list, time_kernel, 'kernel')
plot_time_vs_nprocs(nrows_list, nprocs_list, time_setValuesIJV, 'setValuesIJV')
plot_time_vs_nprocs(nrows_list, nprocs_list, time_assemble, 'assemble')

def plot_strong_scaling(nrows_list, nprocs_list, array, title):
    plt.rcParams.update({'font.size': 11})
    # Data preparation: Filtering None values and calculating statistics
    # time_kernel[nprocs_idx][nrows_idx][process_idx]
    data = [] 
    for p_idx in range(len(nprocs_list)):
        p_data = []
        for r_idx in range(len(nrows_list)):
            # Extract times for active processes only
            times = [t for t in array[p_idx][r_idx] if t is not None]
            p_data.append(np.array(times))
        data.append(p_data)


    ## 1. Strong Scaling Plot
    plt.figure(figsize=(10, 6))
    for r_idx, nrows in enumerate(nrows_list):
        # Strong scaling uses the maximum time (bottleneck) among all processes
        max_times = [np.max(data[p_idx][r_idx]) for p_idx in range(len(nprocs_list))]
        
        # Calculate speedup: T(1) / T(N)
        t1 = max_times[0]
        speedup = [t1 / tn for tn in max_times]
        
        plt.plot(nprocs_list, speedup, marker='^', label=f'#rows={nrows}', markerfacecolor='none', markersize=10)

    # Ideal scaling line
    plt.plot(nprocs_list, nprocs_list, '--', color='gray', label='ideal speedup')

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xticks(nprocs_list, nprocs_list)
    plt.yticks([2**k for k in range(len(nprocs_list)+1)], [2**k for k in range(len(nprocs_list)+1)])
    plt.xlabel('number of processes')
    plt.ylabel('speedup')
    plt.title(f'Strong scaling (maximum over processes) {title}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'strong_scaling_{title}.pdf')
    plt.close()

plot_strong_scaling(nrows_list, nprocs_list, time_kernel, 'kernel')
plot_strong_scaling(nrows_list, nprocs_list, time_setValuesIJV, 'setValuesIJV')
plot_strong_scaling(nrows_list, nprocs_list, time_assemble, 'assemble')

def plot_load_imbalance(nrows_list, nprocs_list, array, title):
    data = [] 
    for p_idx in range(len(nprocs_list)):
        p_data = []
        for r_idx in range(len(nrows_list)):
            # Extract times for active processes only
            times = [t for t in array[p_idx][r_idx] if t is not None]
            p_data.append(np.array(times))
        data.append(p_data)
    plt.rcParams.update({'font.size': 11})
    ## 2. Load Imbalance Plot (Boxplots)
    # We will create a subplot for each nrows_list entry to see imbalance evolution
    fig, axes = plt.subplots(1, len(nrows_list), figsize=(20, 6), sharey=False)

    for r_idx, nrows in enumerate(nrows_list):
        # Group data by nprocs for this specific row count
        plot_data = [data[p_idx][r_idx] for p_idx in range(len(nprocs_list))]
        
        axes[r_idx].boxplot(plot_data, labels=nprocs_list)
        axes[r_idx].set_title(f'#rows={nrows}')
        axes[r_idx].set_xlabel('#processes')
        if r_idx == 0:
            axes[r_idx].set_ylabel('Time [s]')
        axes[r_idx].grid(axis='y', alpha=0.3)

    plt.suptitle(f'Load imbalance across processes for {title}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'load_imbalance_{title}.pdf')
    plt.close()

plot_load_imbalance(nrows_list, nprocs_list, time_kernel, 'kernel')
plot_load_imbalance(nrows_list, nprocs_list, time_setValuesIJV, 'setValuesIJV')
plot_load_imbalance(nrows_list, nprocs_list, time_assemble, 'assemble')