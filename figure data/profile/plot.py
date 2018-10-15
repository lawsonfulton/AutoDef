import json
import numpy
import os


from matplotlib import rc
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(figsize=(4, 7))




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

import numpy as np
def stacked_bar(ax, data, series_labels, category_labels=None, 
                show_values=False, value_format="{}", y_label=None, 
                grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """
    plt.sca(ax)
    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    CB_color_cycle = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    best_cols = [2, 5, 1, 3, 4, 6]


    # lw = 1
    color_ind = 0
    # for name, sim_stats in names_to_sim_stats.items():
    #     iteration_info = sim_stats['timesteps'][0]['iteration_info']
    #     iteration_inds = iteration_info['lbfgs_iteration']
    #     obj_vals = iteration_info['lbfgs_obj_vals']
    #     final_vals = get_last_val_of_each_iteration(obj_vals, iteration_inds)
    #     print(name, len(final_vals))
    #     ax.plot(final_vals, linewidth=lw, color=CB_color_cycle[best_cols[color_ind]])
    for i, row_data in enumerate(data):
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i], color=CB_color_cycle[best_cols[color_ind]]))
        cum_size += row_data
        color_ind += 1

    if category_labels:
        plt.xticks(ind, category_labels)

    # if y_label:
    #     plt.ylabel(y_label)

    plt.legend(loc='upper left')



    # if show_values:
    #     for axis in axes:
    #         for bar in axis:
    #             w, h = bar.get_width(), bar.get_height()
    #             plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
    #                      value_format.format(h), ha="center", 
    #                      va="center")
def main():
    # Load
    # names_to_paths = {
    # 'Autoencoder (ours) 0.56s': 'ae_precon_rest/sim_stats.json',
    # #'Autoencoder': 'ae_no_precon_static/sim_stats.json',
    
    # 'Linear Subspace 0.96s': 'linear_precon_rest/sim_stats.json', 
    # #'Linear Space Rest Preconditioned': 'linear_rest_precon/sim_stats.json', # 1.01393s
    # # 'Linear Space': 'pca_no_precon_static/sim_stats.json',
    # 'Full Space 4.38s': 'full_precon_rest/sim_stats.json', 
    # # 'Full Space No Preconditioned': 'full_no_precon_static/sim_stats.json', #16.8299s
    # # 'Full Space Rest Preconditioned': 'full_rest_pose_precon/sim_stats.json', #16.8299s

    # # 'Autoencoder Preconditioned Delta': 'delta/sim_stats.json',
    # }

    # # New times
    # # full 4.3763
    # # linear 0.961133
    # # ae 0.5633
    
    # names_to_sim_stats = {}
    # for name, path in names_to_paths.items():
    #     with open(path, 'r') as f:
    #         names_to_sim_stats[name] = json.load(f)


    CB_color_cycle = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    best_cols = [2, 6, 5, 1, 3, 4]


    # lw = 1
    # color_ind = 0
    # for name, sim_stats in names_to_sim_stats.items():
    #     iteration_info = sim_stats['timesteps'][0]['iteration_info']
    #     iteration_inds = iteration_info['lbfgs_iteration']
    #     obj_vals = iteration_info['lbfgs_obj_vals']
    #     final_vals = get_last_val_of_each_iteration(obj_vals, iteration_inds)
    #     print(name, len(final_vals))
    #     ax.plot(final_vals, linewidth=lw, color=CB_color_cycle[best_cols[color_ind]])

    #     color_ind += 1
    



    
    # color_ind = 0
    # lin_bttm = 0
    # ae_bttm = 0
    # for label, lin_t, ae_t in zip(linear_labels, linear_times, ae_times):
    #     ax.bar(["PCA", "AE"], [lin_t, ae_t], bottom=[lin_bttm, ae_bttm], color=CB_color_cycle[best_cols[color_ind]])
    #     lin_bttm += lin_t
    #     ae_bttm += ae_t
    #     color_ind += 1
    series_labels = ['cuba eval time tot', 'cuba decode time tot', 'tf decode time tot', 'tf vjp time tot', 'precon time tot']

    linear_times =np.array([0.000328609, 0.000111729, 0, 0, 0]) # 28its
    ae_times =np.array([0.000388202, 0.000143065,  7.54183e-05, 0.00010022, 0.000174958]) # 9.26 its
    ax.set_ylim(top=1.0)
    ax.locator_params(axis='y', nbins=4)

    linear_times =np.array([0.000328609, 0.000111729, 0, 0, 0]) * 28# 28its
    ae_times =np.array([0.000388202*9.26, 0.000143065*9.26,  7.54183e-05*9.26, 0.00010022*9.26, 0.000174958 * 9.26]) # 9.26 its
    ax.set_ylim(top=15.0)
    ax.locator_params(axis='y', nbins=5)

    data = np.stack([linear_times, ae_times]).transpose() * 1000

    category_labels = ['PCA', 'Autoencoder']

    stacked_bar(
        ax,
        data, 
        series_labels, 
        category_labels=category_labels, 
        show_values=True, 
        value_format="{:.1f}",
        y_label="Quantity (units)"
    )

    # ax.set_xlim(left=0, right=500)
    # ax.set_ylim(top=15.0)

    # ax.set_ylabel(r'Objective Value', rotation=0)
    # ax.yaxis.set_label_coords(-0.06,1.005)
    # ax.set_xlabel(r'Iteration'))

    # ax.legend(linear_labels, loc='upper right')


    axes_color = 'darkgrey'
    ax.set_axisbelow(True)
    background_color = '0.9'#(0.8274509803921568, 0.8274509803921568, 0.8274509803921568, 1.0) * 1.1
    gridlines_color = 'white'
    ax.yaxis.grid(color=gridlines_color, linewidth=0.75)
    ax.set_facecolor(background_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor(axes_color)
    ax.spines['bottom'].set_edgecolor(axes_color)
    ax.tick_params(axis='x', colors=axes_color, labelcolor='black')
    ax.tick_params(axis='y', colors=axes_color, labelcolor='black')
    # ax.locator_params(axis='x', nbins=3)
    
    
    # ax.get_yaxis().set_tick_params(which='minor', size=0)

    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.tight_layout()
    fig.savefig('per_eval.svg')
    plt.show()




if __name__ == '__main__':
    main()
