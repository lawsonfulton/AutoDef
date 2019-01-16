import json
import numpy
import os


from matplotlib import rc
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})

# Uncomment
rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(8, 4))


def get_last_val_of_each_iteration(obj_vals, iteration_inds):
    final_vals = []

    last_ind = 0
    last_val = -100000
    for val, ind in zip(obj_vals, iteration_inds):
        if ind != last_ind:
            final_vals.append(last_val)
        
        last_val = val
        last_ind = ind

    final_vals.append(last_val)

    return final_vals

def main():
    # Load

    path = 'record.json'
    with open(path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['data'])
    print(df)


    CB_color_cycle = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    best_cols = [2, 6, 5, 1]



    lw = 1
    color_ind = 0


    x_vals = df['n_layers']
    ax2 = ax.twinx()
    lns1 = ax2.plot(x_vals, df['avg_time'], linewidth=lw, color=CB_color_cycle[best_cols[0]], linestyle='-', marker='s', markersize=3)
    lns2 = ax.plot(x_vals, df['max_distance_error'], linewidth=lw, color=CB_color_cycle[3], linestyle='-', marker='o', markersize=3)
    
    # ax.set_xlim(left=0, right=500)
    # ax.set_ylim(top=0.00002)

    # ax.set_ylabel(r'Objective Value', rotation=0)
    # ax.yaxis.set_label_coords(-0.06,1.005)
    ax.set_xlabel(r'Model Depth')

    # ax.legend(["a","b"], loc='upper center')
    # lns = lns1+lns2
    # labs = [r'Step Time', r'Training Error']
    # ax.legend(lns, labs, loc='upper center')

    ax.legend(['Training Error $10^{-2}$'], loc='upper left')
    ax2.legend([r'Step Time (ms)'], loc='upper right')



    def set_ax_params(ax, show_grid=True):
        axes_color = 'darkgrey'
        gridlines_color = 'white'
        background_color = '0.9'#(0.8274509803921568, 0.8274509803921568, 0.8274509803921568, 1.0) * 1.1

        ax.set_facecolor(background_color)
        if show_grid:
            ax.yaxis.grid(color=gridlines_color, linewidth=0.75)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_edgecolor(axes_color)
        ax.spines['bottom'].set_edgecolor(axes_color)
        ax.tick_params(axis='x', colors=axes_color, labelcolor='black')
        ax.tick_params(axis='y', colors=axes_color, labelcolor='black')
        # ax.locator_params(axis='x', nbins=3)
        ax.locator_params(axis='y', nbins=4)
        # ax.get_yaxis().set_tick_params(which='minor', size=0)

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    set_ax_params(ax)
    set_ax_params(ax2, show_grid=False)

    fig.tight_layout()
    fig.savefig('depth_eval_orig.svg')
    plt.show()




if __name__ == '__main__':
    main()
