import os
import sys
import re
import itertools
from multiprocessing import Pool

import numpy

import pyigl as igl
from utils.iglhelpers import e2p, p2e

def save_numpy_mat_to_dmat(filename, numpy_mat):
    eigen_mat = p2e(numpy_mat)
    igl.writeDMAT(filename, eigen_mat, False) # Save as binary dmat
    print("Saved dmat to", filename)

def load_base_vert_and_face_dmat_to_numpy(base_path):
    """ Returns a tuple (verts, faces) """
    verts_filename = os.path.join(base_path, 'base_verts.dmat')
    faces_filename = os.path.join(base_path, 'base_faces.dmat')

    verts = igl.eigen.MatrixXd()
    faces = igl.eigen.MatrixXi()
    igl.readDMAT(verts_filename, verts)
    igl.readDMAT(faces_filename, faces)

    return e2p(verts), e2p(faces)

def _read_dmat_helper(args):
    i, base_path, dmat_prefix = args
    filename = os.path.join(base_path, dmat_prefix + str(i) + '.dmat')
    if(i % 13 == 0):
        print('.', end='', flush=True)
    displacements = igl.eigen.MatrixXd()
    igl.readDMAT(filename, displacements)
    return e2p(displacements)

def load_dmats(base_path, dmat_prefix, num_samples=None):
    """ Returns a numpy array of displacements for num_samples configurations """

    if num_samples is None:  # Compute number of configurations to load automatically
        filenames = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        fnames_w_prefix = [f for f in filenames if f.startswith(dmat_prefix)]
        numbers_strings = [re.findall('\d+', f) for f in fnames_w_prefix]
        num_samples = max(int(n_str[0]) + 1 for n_str in numbers_strings if n_str)

    print('Loading', num_samples, 'samples...', end='', flush=True)
    p = Pool(16)

    samples = numpy.array(p.map(_read_dmat_helper, zip(range(num_samples), itertools.repeat(base_path), itertools.repeat(dmat_prefix))))
    p.terminate()
    print()
    print('Done.')
    return samples

def load_displacement_dmats_to_numpy(base_path, num_samples=None):
    """ Returns a numpy array of displacements for num_samples configurations """
    return load_dmats(base_path, 'displacements_', num_samples)

def load_energy_dmats_to_numpy(base_path, num_samples=None):
    return load_dmats(base_path, 'energy_', num_samples)

def decompose_ae(autoencoder, do_energy=False):
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
    #old_decoder_layers = autoencoder.layers[encoded_layer_idx+1:] # Need to rebuild the tensor I guess
    decoder_output = decoder_input
    for layer in autoencoder.layers:
        if 'decode' in layer.name:
            decoder_output = layer(decoder_output)

    decoder = Model(inputs=decoder_input, outputs=decoder_output)

    if do_energy:
        energy_input = Input(shape=(encoded_layer.output_shape[-1],), name="energy_model_input")
        energy_output = energy_input
        for layer in autoencoder.layers:
            if 'energy' in layer.name:
                energy_output = layer(energy_output)

        energy_model = Model(inputs=energy_input, outputs=energy_output)

        return autoencoder, encoder, decoder, energy_model

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

def fd_jacobian(f_orig, x, eps=None, is_keras=False):
    """
    Computes the jacobian matrix of f at x with finite diffences.
    If is_keras is true, then x will be wrapped in an additional array before being passed to f.
    """
    if eps is None:
        eps = numpy.sqrt(numpy.finfo(float).eps)

    n_x = len(x)
    if is_keras:
        f = lambda x: f_orig(numpy.array([x])).flatten()
    else:
        f = f_orig

    jac = numpy.zeros([n_x, len(f(x))])
    dx = numpy.zeros(n_x)
    for i in range(n_x): # TODO can do this without for loop
       dx[i] = eps
       jac[i] = (f(x + dx ) - f(x - dx)) / (2.0 * eps)
       dx[i] = 0.0

    return jac.transpose()

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
