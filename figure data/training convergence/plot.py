import json
import numpy
import os


from matplotlib import rc
import matplotlib.pyplot as plt

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(figsize=(12, 5))



def main():
    # Load
    pca_space_loss_history_path = 'training_history_pca_space_loss.json' # full space loss, random init, no freeze
    full_space_loss_rand_unfrozen_weights_history_path = 'training_history_full_space_loss_rand_unfrozen_weights.json' # PCA space loss
    full_space_loss_pca_frozen_weights_history_path = 'training_history_full_space_loss_pca_frozen_weights.json'
    full_space_loss_pca_unfrozen_weights_history_path = 'training_history_full_space_loss_pca_unfrozen_weights.json'
    
    other_history_path = 'training_history.json'
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
    epochs_to_show = 2000

    pca_loss_mse = pca_loss_hist['mse'][:epochs_to_show]
    full_space_loss_rand_unfrozen_mse = full_space_loss_rand_unfrozen_hist['mse'][:epochs_to_show]
    full_space_loss_pca_frozen_weights_mse = full_space_loss_pca_frozen_weights_hist['mse'][:epochs_to_show]
    full_space_loss_pca_unfrozen_weights_mse = full_space_loss_pca_unfrozen_weights_hist['mse'][:epochs_to_show]
    if do_other:
        other_mse = other_hist['mse'][:epochs_to_show]

    lw = 1
    ax.semilogy(pca_loss_mse, linewidth=lw)
    ax.semilogy(full_space_loss_rand_unfrozen_mse, linewidth=lw)
    ax.semilogy(full_space_loss_pca_frozen_weights_mse, linewidth=lw)
    ax.semilogy(full_space_loss_pca_unfrozen_weights_mse, linewidth=lw)
    if do_other:
        ax.semilogy(other_mse, linewidth=lw)

    ax.axhline(y=7.85752324663e-06, color='gray', linestyle='dashed', linewidth=1)

    ax.set_ylabel(r'Full Space MSE')
    ax.set_xlabel(r'Epoch')

    ax.legend(['Train in PCA Space', 'Full Space Random Weights', 'Full Space Frozen PCA Weights', 'Full Space Unfrozen PCA Weights', '5D PCA Only'], loc='upper right')


    gridcolour = 'darkgrey'
    # ax.yaxis.grid(color=gridcolour, linestyle='dashed', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor(gridcolour)
    ax.spines['bottom'].set_edgecolor(gridcolour)
    ax.tick_params(axis='x', colors=gridcolour, labelcolor='black')
    ax.tick_params(axis='y', colors=gridcolour, labelcolor='black')

    fig.tight_layout()
    #fig.savefig('training_mse.svg')
    plt.show()




if __name__ == '__main__':
    main()
