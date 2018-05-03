import sys, os
import numpy as np
from utils.my_utils import load_obj_verts
from multiprocessing import Pool

import matplotlib.pyplot as plt
# Mess around with this come paper polish time
# params = {'backend': 'ps',
#               'text.latex.preamble': ['\\usepackage{gensymb}'],
#               'axes.labelsize': 16, # fontsize for x and y labels (was 10)
#               'axes.labelweight': 'bold',
#               # 'axes.titlesize': 8,
#               # 'text.fontsize': 8, # was 10
#               # 'legend.fontsize': 8, # was 10
#               'xtick.labelsize': 12,
#               'ytick.labelsize': 12,
#               'text.usetex': True,
#               # 'figure.figsize': [fig_width,fig_height],
#               'font.family': 'sans-serif'
#     }
# plt.rcParams.update(params)

def get_displacements_for_vert(Vs, index):
    V0 = Vs[0][index]
    return Vs[:, index] - V0

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = '/home/lawson/Workspace/research-experiments/fem-sim/models/x-final/simulation_logs/compare_an08_fixed'

    chosen_vert = 19 # Top, right, front corner

    # Load
    obj_Vs = {}
    displacements = {}
    displacement_norms = {}
    log_dir_names = ['ae_all', 'an08_ae_152', 'pred_weights_72', 'pcr_ae_150', 'an08_ae_84'] # os.listdir(root_dir)
    for name in log_dir_names:
        print('Loading', name)
        obj_Vs[name] = load_obj_verts(os.path.join(root_dir, name, 'objs/surface'))

        displacements[name] = get_displacements_for_vert(obj_Vs[name], chosen_vert)
        displacement_norms[name] = np.linalg.norm(displacements[name], axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))   

    names_to_style = {
        'ae_all': '-',
        'an08_ae_152': ':',
        'an08_ae_84': ':',
        'pred_weights_72': '-.',
        'pcr_ae_150': '--',
    }
    for name in names_to_style:
        # ax.plot(displacement_norms[name], label=name, linewidth=2, linestyle=names_to_style[name])
        ax.plot(displacements[name][:,2], label=name, linewidth=2, linestyle=names_to_style[name])

    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement in Z-axis of vertex 19')

    ax.set_ylim(-6.05, 6)

    # Style
    ax.legend(loc='upper left')
    ax.spines['bottom'].set_position('center')
    
    gridcolour = 'darkgrey'
    ax.yaxis.grid(color=gridcolour, linestyle='dashed', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor(gridcolour)
    ax.spines['bottom'].set_edgecolor(gridcolour)
    ax.tick_params(axis='x', colors=gridcolour, labelcolor='black')
    ax.tick_params(axis='y', colors=gridcolour, labelcolor='black')

    
    fig.tight_layout()
    fig.savefig('demo.pdf')#, transparent=True)
    plt.show()
    
if __name__=='__main__':
    main()
