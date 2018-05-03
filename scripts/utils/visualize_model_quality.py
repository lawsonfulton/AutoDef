import json
import numpy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_autoencoder(model_base_filename):
    import keras

    model_names = ['_encoder', '_decoder']
    models = {}

    for name in model_names:
        keras_model_file = model_base_filename + name + '.hdf5' 
        models[name] = keras.models.load_model(keras_model_file)

    return models['_encoder'], models['_decoder']


def plot_training_history(model_base_filename):
    history_path = model_base_filename + '_history.json'
    with open(history_path, 'r') as f:
        hist = json.load(f)

    loss = hist['loss']
    val_loss = hist['loss']

    plt.plot(loss)
    plt.plot(val_loss)

    plt.ylabel('Loss')
    plt.xlabel('epoch')

def plot_encoded_space(model_base_filename):
    """ Only supports 3dim latent space """
    with open(model_base_filename + '_ae_encoded.json', 'r') as f:
        encoded_vals = numpy.array(json.load(f))

    with open(model_base_filename + '_decoder_jacobian_norms.json', 'r') as f:
        decoder_jacobian_norms = numpy.array(json.load(f))


    ## TODO this normalization should be an absolute scale to compare across models

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    from sklearn.preprocessing import normalize
    # normed_norms = (decoder_jacobian_norms - min(decoder_jacobian_norms)) / max(decoder_jacobian_norms - min(decoder_jacobian_norms))
    normed_norms = (decoder_jacobian_norms - 2.0) / 4.0
    # print(min(decoder_jacobian_norms))
    # print(max(decoder_jacobian_norms))
    print(numpy.mean(decoder_jacobian_norms))
    print(numpy.var(decoder_jacobian_norms))

    ax.plot(encoded_vals[:, 0], encoded_vals[:, 1], encoded_vals[:, 2], label='Encoded space')
    ax.scatter(encoded_vals[:, 0], encoded_vals[:, 1], encoded_vals[:, 2], s = normed_norms*100, c = normed_norms +0.5, vmin=0.0, vmax=1.0,)

def main():
    model_base_filename = 'models/baseline_elu'
    # TODO replicate for pca space

    #plot_training_history(ae_base_filename)
    plot_encoded_space(model_base_filename)

    #encode, decode = load_autoencoder(ae_base_filename)

    plt.show()


if __name__ == '__main__':
    main()
