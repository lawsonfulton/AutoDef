import json
import numpy
from pprint import pprint
from collections import defaultdict
import sys

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))

def smooth(x, N=10):
    return numpy.convolve(x, numpy.ones((N,))/N, mode='valid')

def main():
    
    log_path = 'full_precon/sim_stats.json'

    with open(log_path) as f:
        sim_stats = json.load(f)

    n_frames = 50
    timesteps = sim_stats['timesteps'][:n_frames]

    print(timesteps[0])
    tot_time_per_step = [t['tot_step_time_s'] for t in timesteps]
    its_per_step = [t['lbfgs_iterations'] for t in timesteps]
    timing_info_per_step = [t['iteration_info']['timing'] for t in timesteps]

    times_per_step = defaultdict(list)
    for step_info in timing_info_per_step:
        for time_section, times in step_info.items():
            times_per_step[time_section].append(numpy.sum(times))

    ax.plot(smooth(tot_time_per_step))

    stack_plots = []
    stack_labels = []
    for time_section, times in times_per_step.items():
        if time_section != 'tot_obj_time_s':
            stack_plots.append(smooth(times))
            stack_labels.append(time_section)

    ax.stackplot(range(len(stack_plots[0])), stack_plots, labels=stack_labels)
    # ax2 = ax.twinx()
    # ax2.plot(its_per_step, 'r')
    
    ax.set_xlabel('Time')
    # ax.set_ylabel('Displacement in Z-axis of vertex 19')

    # ax.set_ylim(0, 0.05)

    # Style
    ax.legend(loc='upper left')
    # ax.spines['bottom'].set_position('center')
    
    gridcolour = 'darkgrey'
    ax.yaxis.grid(color=gridcolour, linestyle='dashed', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor(gridcolour)
    ax.spines['bottom'].set_edgecolor(gridcolour)
    ax.tick_params(axis='x', colors=gridcolour, labelcolor='black')
    ax.tick_params(axis='y', colors=gridcolour, labelcolor='black')

    
    fig.tight_layout()
    fig.savefig('plot.png')#, transparent=True)
    plt.show()

if __name__ == '__main__':
    main()
