import os
import re
import itertools
from multiprocessing import Pool

import numpy

import pyigl as igl
from iglhelpers import e2p, p2e

def save_numpy_mat_to_dmat(filename, numpy_mat):
    eigen_mat = p2e(numpy_mat)
    igl.writeDMAT(filename, eigen_mat, False) # Save as binary dmat
    print("Saved dmat to", filename)

def load_base_vert_and_face_dmat_to_numpy(base_path):
    """ Returns a tuple (verts, faces) """
    verts_filename = base_path + 'base_verts.dmat'
    faces_filename = base_path + 'base_faces.dmat'

    verts = igl.eigen.MatrixXd()
    faces = igl.eigen.MatrixXi()
    igl.readDMAT(verts_filename, verts)
    igl.readDMAT(faces_filename, faces)

    return e2p(verts), e2p(faces)

def _read_dmat_helper(args):
    i, base_path = args
    filename = base_path + 'displacements_%d.dmat' % i
    if(i % 13 == 0):
        print('.', end='', flush=True)
    displacements = igl.eigen.MatrixXd()
    igl.readDMAT(filename, displacements)
    return e2p(displacements)

def load_displacement_dmats_to_numpy(base_path, num_samples=None):
    """ Returns a numpy array of displacements for num_samples configurations """

    if num_samples is None:  # Compute number of configurations to load automatically
        filenames = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        numbers_strings = [re.findall('\d+', f) for f in filenames]
        num_samples = max(int(n_str[0]) for n_str in numbers_strings if n_str)

    print('Loading', num_samples, 'samples...', end='', flush=True)
    p = Pool(16)

    displacements_samples = numpy.array(p.map(_read_dmat_helper, zip(range(num_samples), itertools.repeat(base_path))))
    p.terminate()
    print()
    print('Done.')
    return displacements_samples

def decompose_ae(autoencoder):
    """ Takes a Keras autoencoder model and splits it into three seperate models """
    import keras
    from keras.layers import Input, Dense
    from keras.models import Model, load_model
    def get_encoded_layer_and_index(): # Stupid hack
        for i, layer in enumerate(autoencoder.layers):
            if layer.name == 'encoded_layer':
                return layer, i

    encoded_layer, encoded_layer_idx = get_encoded_layer_and_index()
    encoder = Model(inputs=autoencoder.input, outputs=encoded_layer.output)

    decoder_input = Input(shape=(encoded_layer.output_shape[-1],), name="decoder_input")
    old_decoder_layers = autoencoder.layers[encoded_layer_idx+1:] # Need to rebuild the tensor I guess
    decoder_output = decoder_input
    for layer in old_decoder_layers:
        decoder_output = layer(decoder_output)

    decoder = Model(inputs=decoder_input, outputs=decoder_output)

    return autoencoder, encoder, decoder

def get_flattners(data):
    """ Returns two functions flatten and unflatten that will convert to and from a 2d array """
    if len(data.shape) == 2:
        return lambda x: x, lambda x: x

    sample_dim = len(data[0])
    point_dim = len(data[0][0])

    def flatten_data(unflattned_data):
        n_samples = len(unflattned_data)
        return unflattned_data.reshape((n_samples, sample_dim * point_dim))
    def unflatten_data(flattened_data):
        n_samples = len(flattened_data)
        return flattened_data.reshape((n_samples, sample_dim, point_dim))

    return flatten_data, unflatten_data
