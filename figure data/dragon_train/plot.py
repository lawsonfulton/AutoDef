import json
import numpy

import matplotlib.pyplot as plt


def main():
    # Load
    no_pca_history_path = 'training_history.json'
    # with_pca_history_path = 'with_pca_training_history.json'

    with open(no_pca_history_path, 'r') as f:
        no_pca_hist = json.load(f)

    # with open(with_pca_history_path, 'r') as f:
    #     with_pca_hist = json.load(f)

    # Plot
    epochs_to_show = 2000

    no_pca_mse = no_pca_hist['loss'][:epochs_to_show]
    # with_pca_mse = with_pca_hist['mse'][:epochs_to_show]

    plt.semilogy(no_pca_mse)
    # plt.semilogy(with_pca_mse)

    plt.ylabel('Full Space MSE')
    plt.xlabel('Epoch')

    plt.legend(['No PCA', 'With PCA'], loc='upper right')

    plt.savefig('training_mse.svg')
    plt.show()




if __name__ == '__main__':
    main()
