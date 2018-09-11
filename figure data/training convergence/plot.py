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

fig, ax = plt.subplots(figsize=(5, 3))


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

def main():
    # Load
    pca_space_loss_history_path = 'trained_models/pca_space_loss/training_history.json' # full space loss, random init, no freeze
    full_space_loss_rand_unfrozen_weights_history_path = 'trained_models/full_space_loss_rand_unfrozen/training_history.json' # PCA space loss
    full_space_loss_pca_frozen_weights_history_path = 'trained_models/full_space_loss_pca_frozen/training_history.json'
    full_space_loss_pca_unfrozen_weights_history_path = 'trained_models/full_space_loss_pca_unfrozen/training_history.json'
    
    other_history_path = ''#'trained_models/full_space_loss_rand_unfrozen_unreduced/training_history.json'
    do_other = os.path.exists(other_history_path)
    # Next load the full space loss, frozen weights

    with open(pca_space_loss_history_path, 'r') as f:
        pca_loss_hist = json.load(f)

    with open(full_space_loss_rand_unfrozen_weights_history_path, 'r') as f:
        full_space_loss_rand_unfrozen_hist = json.load(f)

    with open(full_space_loss_pca_frozen_weights_history_path, 'r') as f:
        full_space_loss_pca_frozen_weights_hist = json.load(f)

    with open(full_space_loss_pca_unfrozen_weights_history_path, 'r') as f:
        full_space_loss_pca_unfrozen_weights_hist = json.load(f)

    if do_other:
        with open(other_history_path, 'r') as f:
            other_hist = json.load(f)

    # Plot
    epochs_to_show = len(pca_loss_hist['mse'])

    pca_loss_mse = pca_loss_hist['mse'][:epochs_to_show]
    full_space_loss_rand_unfrozen_mse = full_space_loss_rand_unfrozen_hist['mse'][:epochs_to_show]
    full_space_loss_pca_frozen_weights_mse = full_space_loss_pca_frozen_weights_hist['mse'][:epochs_to_show]
    full_space_loss_pca_unfrozen_weights_mse = full_space_loss_pca_unfrozen_weights_hist['mse'][:epochs_to_show]
    if do_other:
        other_mse = other_hist['mse'][:epochs_to_show]


    # N = 100
    # print(pca_loss_mse[:10])
    # pca_loss_mse = smooth(pca_loss_mse, N, 'hanning')#numpy.convolve(pca_loss_mse, numpy.ones((N,))/N, mode='valid')
    # full_space_loss_rand_unfrozen_mse = smooth(full_space_loss_rand_unfrozen_mse, N, 'hanning')#numpy.convolve(full_space_loss_rand_unfrozen_mse, numpy.ones((N,))/N, mode='valid')
    # full_space_loss_pca_frozen_weights_mse = smooth(full_space_loss_pca_frozen_weights_mse, N, 'hanning')#numpy.convolve(full_space_loss_pca_frozen_weights_mse, numpy.ones((N,))/N, mode='valid')
    # full_space_loss_pca_unfrozen_weights_mse = smooth(full_space_loss_pca_unfrozen_weights_mse, N, 'hanning')#numpy.convolve(full_space_loss_pca_unfrozen_weights_mse, numpy.ones((N,))/N, mode='valid')

    CB_color_cycle = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    s = 1

    lw = 0.5
    ax.semilogy(pca_loss_mse, linewidth=lw, color=CB_color_cycle[2])
    ax.semilogy(full_space_loss_rand_unfrozen_mse, linewidth=lw, color=CB_color_cycle[6])
    ax.semilogy(full_space_loss_pca_frozen_weights_mse, linewidth=lw, color=CB_color_cycle[5])
    ax.semilogy(full_space_loss_pca_unfrozen_weights_mse, linewidth=lw, color=CB_color_cycle[1])
    if do_other:
        ax.semilogy(other_mse, linewidth=lw)

    ax.axhline(y=7.85752324663e-06, color='gray', linestyle='dashed', linewidth=1)

    ax.set_xlim(left=0)
    ax.set_ylim(top=10e-5)

    ax.set_ylabel(r'Full Space MSE', rotation=0)
    ax.yaxis.set_label_coords(0,1.02)
    #ax.set_xlabel(r'Epoch')

    ax.legend(['Train in PCA Space', 'Full Space Random Weights', 'Full Space Frozen PCA Weights', 'Full Space Unfrozen PCA Weights', '5D PCA Only'], loc='upper right')


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
    ax.get_yaxis().set_tick_params(which='minor', size=0)


    fig.tight_layout()
    # fig.savefig('training_convergence.svg')
    plt.show()




if __name__ == '__main__':
    main()
