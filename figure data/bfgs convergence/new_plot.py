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

fig, ax = plt.subplots(figsize=(6, 4))


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    if window_len<3:
        return x

    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else: # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise 
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

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
    names_to_paths = {
    'Autoencoder (ours) 0.56s': 'ae_precon_rest/sim_stats.json',
    #'Autoencoder': 'ae_no_precon_static/sim_stats.json',
    
    'Linear Subspace 0.96s': 'linear_precon_rest/sim_stats.json', 
    #'Linear Space Rest Preconditioned': 'linear_rest_precon/sim_stats.json', # 1.01393s
    # 'Linear Space': 'pca_no_precon_static/sim_stats.json',
    'Full Space 4.38s': 'full_precon_rest/sim_stats.json', 
    # 'Full Space No Preconditioned': 'full_no_precon_static/sim_stats.json', #16.8299s
    # 'Full Space Rest Preconditioned': 'full_rest_pose_precon/sim_stats.json', #16.8299s

    # 'Autoencoder Preconditioned Delta': 'delta/sim_stats.json',
    }

    # New times
    # full 4.3763
    # linear 0.961133
    # ae 0.5633
    
    names_to_sim_stats = {}
    for name, path in names_to_paths.items():
        with open(path, 'r') as f:
            names_to_sim_stats[name] = json.load(f)


    CB_color_cycle = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    best_cols = [2, 6, 5, 1]


    lw = 1
    color_ind = 0
    for name, sim_stats in names_to_sim_stats.items():
        iteration_info = sim_stats['timesteps'][0]['iteration_info']
        iteration_inds = iteration_info['lbfgs_iteration']
        obj_vals = iteration_info['lbfgs_obj_vals']
        final_vals = get_last_val_of_each_iteration(obj_vals, iteration_inds)
        print(name, len(final_vals))
        ax.plot(final_vals, linewidth=lw, color=CB_color_cycle[best_cols[color_ind]])

        color_ind += 1
    
    ax.set_xlim(left=0, right=500)
    ax.set_ylim(top=0.00002)

    # ax.set_ylabel(r'Objective Value', rotation=0)
    # ax.yaxis.set_label_coords(-0.06,1.005)
    ax.set_xlabel(r'Iteration')

    ax.legend(names_to_paths.keys(), loc='upper right')


    axes_color = 'darkgrey'
    gridlines_color = 'white'
    background_color = '0.9'#(0.8274509803921568, 0.8274509803921568, 0.8274509803921568, 1.0) * 1.1

    ax.set_facecolor(background_color)
    ax.yaxis.grid(color=gridlines_color, linewidth=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor(axes_color)
    ax.spines['bottom'].set_edgecolor(axes_color)
    ax.tick_params(axis='x', colors=axes_color, labelcolor='black')
    ax.tick_params(axis='y', colors=axes_color, labelcolor='black')
    ax.locator_params(axis='x', nbins=3)
    ax.locator_params(axis='y', nbins=5)
    # ax.get_yaxis().set_tick_params(which='minor', size=0)

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.tight_layout()
    fig.savefig('training_convergence.svg')
    plt.show()




if __name__ == '__main__':
    main()
