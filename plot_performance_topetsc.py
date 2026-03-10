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



def write_time_to_table(nprocs_list, nrows_list, time_kernel, time_setValuesIJV, time_assemble):
    table = open(f'table_petsc_performance.txt', "w")
    print(f'$K$ & $N$ & $T^P(K,N)$ (\\si{{\\second}}) & $T^S(K,N)$ (\\si{{\\second}}) & $T^A(K,N)$ (\\si{{\\second}})' + r'\\ \hline', flush=True, file=table)
    for i in range(len(nprocs_list)):
        print(f'\\SetCell[r=5]{{c}} {nprocs_list[i]}', flush=True, file=table)
        for j in range(len(nrows_list)):
            max_kernel = np.max([t for t in time_kernel[i,j] if t is not None])
            max_setValuesIJV = np.max([t for t in time_setValuesIJV[i,j] if t is not None])
            max_assemble = np.max([t for t in time_assemble[i,j] if t is not None])

            print(f' & {nrows_list[j]} & {'{:.2e}'.format(max_kernel)} & {'{:.2e}'.format(max_setValuesIJV)} & {'{:.2e}'.format(max_assemble)}' + r'\\', flush=True, file=table)

        print(r'\hline', flush=True, file=table)

    table.close()
write_time_to_table(nprocs_list, nrows_list, time_kernel, time_setValuesIJV, time_assemble)





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
            x_vals.extend([nrows] * len(valid_times)) #list with constant nrows, length as valid_times
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
    plt.close()

plot_time_vs_nrows(nrows_list, nprocs_list, time_kernel, 'kernel')
plot_time_vs_nrows(nrows_list, nprocs_list, time_setValuesIJV, 'setValuesIJV')
plot_time_vs_nrows(nrows_list, nprocs_list, time_assemble, 'assemble')

def plot_time_vs_nprocs_scatter(nrows_list, nprocs_list, array, title):
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
    plt.xscale('log', base=2)
    plt.legend()

    plt.grid(True, which="major", ls="-", alpha=1)
    plt.grid(True, which="minor", ls="-", alpha=0.5)
    
    plt.xticks(nprocs_list, [str(n) for n in nprocs_list], minor=False)
    plt.xticks(nprocs_list, ['' for n in nprocs_list], minor=True)
    plt.tick_params(axis='x', which='minor', length=0)

    plt.savefig(f'time_vs_nprocs_{title}.pdf')
    plt.close()

def plot_time_vs_nprocs(nrows_list, nprocs_list, array, title, op):
    plt.figure(figsize=(7, 5))
    plt.rcParams.update({'font.size': 12})  
    
    # Iterate through each problem size (nrows)
    # The array is indexed as [procs_idx, rows_idx, process_idx]
    for r_idx, row_val in enumerate(nrows_list):
        means = []
        mins = []
        maxs = []
        valid_procs = []

        for p_idx, nprocs in enumerate(nprocs_list):
            # Extract times for the specific proc/row combo and remove 'None'
            times = array[p_idx, r_idx]
            clean_times = [t for t in times if t is not None]
            
            if clean_times:
                means.append(np.mean(clean_times))
                mins.append(np.min(clean_times))
                maxs.append(np.max(clean_times))
                valid_procs.append(nprocs)

        # Plotting
        if op == 'mean':
            line, = plt.plot(valid_procs, means, '-o', label=f'nr={row_val}')
            plt.fill_between(
                valid_procs, mins, maxs, 
                color=line.get_color(), alpha=0.3
            )
        elif op == 'max':
            line, = plt.plot(valid_procs, maxs, '-', marker='^', label=f'$N={row_val}$', markerfacecolor='none', markersize=10)

    # Formatting
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel(r'$K$')
    plt.xticks(nprocs_list, nprocs_list)
    if title == 'kernel':
        plt.title(f'$T^P(K,N)$')
    elif title == 'setValuesIJV':
        plt.title(f'$T^S(K,N)$')
    elif title == 'assemble':
        plt.title(f'$T^A(K,N)$')

    #plt.title(f'{title}')
    #plt.legend(loc='lower left', fontsize=10)#bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'time_vs_nprocs_{title}_{op}.pdf')
    plt.close()

plot_time_vs_nprocs(nrows_list, nprocs_list, time_kernel, 'kernel', 'max')
plot_time_vs_nprocs(nrows_list, nprocs_list, time_setValuesIJV, 'setValuesIJV', 'max')
plot_time_vs_nprocs(nrows_list, nprocs_list, time_assemble, 'assemble', 'max')

def plot_strong_scaling(nrows_list, nprocs_list, array, title, op):
    plt.rcParams.update({'font.size': 12})
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
    plt.figure(figsize=(5, 5))

    # Ideal scaling line
    plt.plot(nprocs_list, nprocs_list, '--', color='black', label='linear speedup')

    for r_idx, nrows in enumerate(nrows_list):
        if op == 'max':
            # Strong scaling uses the maximum time (bottleneck) among all processes
            op_times = [np.max(data[p_idx][r_idx]) for p_idx in range(len(nprocs_list))]
        elif op == 'mean':
            op_times = [np.mean(data[p_idx][r_idx]) for p_idx in range(len(nprocs_list))]
        
        # Calculate speedup: T(1) / T(N)
        t1 = op_times[0]
        speedup = [t1 / tn for tn in op_times]
        
        line, = plt.plot(nprocs_list, speedup, marker='^', label=f'$N={nrows}$', markerfacecolor='none', markersize=10)

    for s, nprocs in zip(speedup[1:], nprocs_list[1:]):
        plt.annotate(str(round(s,1)), (nprocs, s), 
            xytext=(-17, -6),      # Offset
            textcoords='offset points', 
            ha='right',            # Horizontal alignment: end of text is at the offset
            va='bottom',
            color=line.get_color(), fontsize=11, 
            bbox=dict(boxstyle="square,pad=0.2", 
                                fc='#F5F5F5', 
                                ec="gray",
                                alpha=1,
                                lw=0))

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xticks(nprocs_list, nprocs_list)
    plt.yticks([2**k for k in range(len(nprocs_list)+1)], [2**k for k in range(len(nprocs_list)+1)])
    #plt.yticks([round(s,1) for s in speedup], [round(s,1) for s in speedup])
    plt.xlabel(r'$K$')
    #plt.ylabel('speedup')
    if title == 'kernel':
        plt.title(f'$\mathcal{{S}}^P(K,N)$')
    elif title == 'setValuesIJV':
        plt.title(f'$\mathcal{{S}}^S(K,N)$')
    elif title == 'assemble':
        plt.title(f'$\mathcal{{S}}^A(K,N)$')
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'strong_scaling_{title}_{op}.pdf')
    plt.close()

plot_strong_scaling(nrows_list, nprocs_list, time_kernel, 'kernel', 'max')
plot_strong_scaling(nrows_list, nprocs_list, time_setValuesIJV, 'setValuesIJV', 'max')
plot_strong_scaling(nrows_list, nprocs_list, time_assemble, 'assemble', 'max')


def plot_efficiency(nrows_list, nprocs_list, array, title, op):
    plt.rcParams.update({'font.size': 12})
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


    ## 1. Efficiency Plot
    plt.figure(figsize=(5, 5))
    # Ideal efficiency
    plt.plot(nprocs_list, [1]*len(nprocs_list), '--', color='black')

    for r_idx, nrows in enumerate(nrows_list):
        if op == 'max':
            op_times = [np.max(data[p_idx][r_idx]) for p_idx in range(len(nprocs_list))]
        elif op == 'mean':
            op_times = [np.mean(data[p_idx][r_idx]) for p_idx in range(len(nprocs_list))]
        
        # Calculate speedup: T(1) / T(N)
        t1 = op_times[0]
        speedup = [t1 / tn for tn in op_times]

        efficiency = [speedup[p_idx]/nprocs_list[p_idx] for p_idx in range(len(nprocs_list))]
        
        line, = plt.plot(nprocs_list, efficiency, marker='^', label=f'$N={nrows}$', markerfacecolor='none', markersize=10)

    for e, nprocs in zip(efficiency[1:], nprocs_list[1:]):
        plt.annotate(f'{int(round(e,2)*1e2)}%', (nprocs, e), 
            xytext=(-15, 15),      # Offset
            textcoords='offset points', color=line.get_color(), fontsize=11, annotation_clip=False, clip_on=True,
            bbox=dict(boxstyle="square,pad=0.2", 
                                fc='#F5F5F5', 
                                ec="gray",
                                alpha=1,
                                lw=0))
    plt.xscale('log', base=2)
    #plt.yscale('log', base=10)
    plt.xticks(nprocs_list, nprocs_list)
    plt.yticks([round(n,1) for n in np.linspace(0,1, 6)] + [1.1], [round(n,1) for n in np.linspace(0,1, 6)] + [''])
    plt.xlabel(r'$K$')
    #plt.ylabel('efficiency')
    if title == 'kernel':
        plt.title(f'$\mathcal{{E}}^P(K,N)$')
    elif title == 'setValuesIJV':
        plt.title(f'$\mathcal{{E}}^S(K,N)$')
    elif title == 'assemble':
        plt.title(f'$\mathcal{{E}}^A(K,N)$')

    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'efficiency_{title}_{op}.pdf')
    plt.ylim(top=1.2, bottom=-0.1)
    plt.close()

plot_efficiency(nrows_list, nprocs_list, time_kernel, 'kernel', 'max')
plot_efficiency(nrows_list, nprocs_list, time_setValuesIJV, 'setValuesIJV', 'max')
plot_efficiency(nrows_list, nprocs_list, time_assemble, 'assemble', 'max')


def plot_load_imbalance(nrows_list, nprocs_list, array_list):
    data = []
    for k in range(len(array_list)):
        p_data = []
        for p_idx in range(len(nprocs_list)):
            p2_data = []
            for r_idx in range(len(nrows_list)):
                # Extract times for active processes only
                times = [t for t in array_list[k][p_idx][r_idx] if t is not None]
                p2_data.append(np.array(times))
            p_data.append(p2_data)
        data.append(p_data)
    plt.rcParams.update({'font.size': 12})
    ## 2. Load Imbalance Plot (Boxplots)
    # We will create a subplot for each nrows_list entry to see imbalance evolution
    fig, axes = plt.subplots(len(array_list), len(nrows_list), figsize=(15, 10), sharey=False)

    for k in range(len(array_list)):
        for r_idx, nrows in enumerate(nrows_list):
            # Group data by nprocs for this specific row count
            plot_data = [data[k][p_idx][r_idx] for p_idx in range(len(nprocs_list))]
            
            axes[k,r_idx].boxplot(plot_data, labels=nprocs_list, 
                                medianprops={"color": f'C{r_idx}', "linewidth": 1.2})
            if k == 0:
                axes[k,r_idx].set_title(f'$N={nrows}$')
            if k == 2:
                axes[k,r_idx].set_xlabel(r'$K$')
            if r_idx == 0:
                if k == 0:
                    axes[k,r_idx].set_ylabel(f'$t^P(K,N,\cdot)$')
                elif k == 1:
                    axes[k,r_idx].set_ylabel(f'$t^S(K,N,\cdot)$')
                elif k == 2:
                    axes[k,r_idx].set_ylabel(f'$t^A(K,N,\cdot)$')

            axes[k,r_idx].grid(axis='y', alpha=0.3)

    plt.suptitle(f'Load imbalance across processes')
    #plt.xscale('log', base=2)
    #plt.yscale('log')
    #plt.xlabel(r'$K$')
    #plt.xticks(nprocs_list, nprocs_list)
    plt.tight_layout()#rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'load_imbalance.pdf')
    plt.close()


plot_load_imbalance(nrows_list, nprocs_list, [time_kernel, time_setValuesIJV, time_assemble])
