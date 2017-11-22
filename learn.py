import time

import numpy
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pyigl as igl
from iglhelpers import e2p, p2e

import my_utils

# Global Vars
modes = {0: 'baseline', 1:'pca', 2:'autoencoder'}
current_mode = 0
current_frame = 0
base_path = 'training_data/first_interaction/'

def autoencoder_analysis(data, latent_dim=3, epochs=100, batch_size=100, layers=[32, 16], pca_weights=None, pca_object=None):
    """
    Returns and encoder and decoder for going into and out of the reduced latent space.
    If pca_weights is given, then do a weighted mse.
    If pca_object is given, then the first and final layers will do a pca transformation of the data.
    """
    assert not((pca_weights is not None) and (pca_object is not None))  # pca_weights incompatible with pca_object

    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model, load_model

    flatten_data, unflatten_data = my_utils.get_flattners(data)

    # TODO: Do I need to shuffle?
    train_data = flatten_data(data)  # No test data for now
    test_data = train_data[:10] # Basically skip testing TODO

    ## Preprocess the data
    # mean = numpy.mean(train_data, axis=0)
    # std = numpy.std(train_data, axis=0)
    
    mean = numpy.mean(train_data)
    std = numpy.std(train_data)
    s_min = numpy.min(train_data)
    s_max = numpy.max(train_data)

    def normalize(data):
        return data
        return numpy.nan_to_num((data - mean) / std)
        # return numpy.nan_to_num((train_data - s_min) / (s_max - s_min))
    def denormalize(data):
        return data
        return data * std + mean
        # return data * (s_max - s_min) + s_mi

    ## Custom layer if we need it
    # def pca_layer(input):
    #     pca_object.transform

    ## Set up the network
    activation = keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'
    
    input = Input(shape=(len(train_data[0]),))
    output = input
    
    # if pca_object is not None:
    #     output = Lambda(pca_layer)

    for layer_width in layers:
        output = Dense(layer_width, activation=activation)(output)
    output = Dense(latent_dim, activation=activation, name="encoded")(output)  # TODO Tanh into encoded layer to bound vars?
    for layer_width in reversed(layers):
        output = Dense(layer_width, activation=activation)(output)
    
    output = Dense(len(train_data[0]), activation='linear')(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    autoencoder = Model(input, output)


    ## Set the optimization parameters
    weights = pca_weights / pca_weights.sum() if pca_weights else None
    def pca_weighted_mse(y_pred, y_true):
        mse = K.mean(weights * K.square(y_true - y_pred), axis=1)
        return mse

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer=optimizer,
        loss='mean_squared_error' if pca_weights is None else pca_weighted_mse
    )
    
    model_start_time = time.time()
    autoencoder.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(test_data, test_data)
    )

    # output_path = 'trained_models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    # autoencoder.save(output_path)

    print("Total training time: ", time.time() - model_start_time)
    
    autoencoder,encoder, decoder = my_utils.decompose_ae(autoencoder)

    def encode(decoded_data):
        return encoder.predict(normalize(flatten_data(decoded_data)))
    def decode(encoded_data):
        return unflatten_data(denormalize(decoder.predict(encoded_data)))

    ae_mse = mean_squared_error(flatten_data(data), flatten_data(decode(encode(data))))
    return encode, decode, ae_mse


def pca_analysis(data, n_components, plot_explained_variance=False):
    """ Returns and encoder and decoder for projecting into and out of the PCA reduced space """
    print("Doing PCA with", n_components, "components...")
    # numpy.random.shuffle(numpy_displacements_sample) # Necessary?
    pca = PCA(n_components=n_components)

    flatten_data, unflatten_data = my_utils.get_flattners(data)

    flattened_data = flatten_data(data)
    pca.fit(flattened_data)

    if plot_explained_variance:
        explained_var = pca.explained_variance_ratio_
        plt.xlabel('Component')
        plt.ylabel('Explained Variance')
        plt.plot(list(range(1, n_components + 1)), explained_var)
        plt.show(block=False)

        print("Total explained variance =", sum(explained_var))

    mse = mean_squared_error(flattened_data, pca.inverse_transform(pca.transform(flattened_data)))

    def encode(decoded_data):
        return pca.transform(flatten_data(decoded_data))        
    def decode(encoded_data):
        return unflatten_data(pca.inverse_transform(encoded_data))

    return encode, decode, pca.explained_variance_ratio_, mse

def get_pca_mse(data, n_components):
    pca_encode, pca_decode, explained_var, pca_mse = pca_analysis(data, n_components)
    return pca_mse

def pca_evaluation(data):
    min_dim = 5
    max_dim = 50
    dims = list(range(min_dim, max_dim + 1))
    mse_list = []
    explained_var_list = []

    for pca_dim in dims:
        pca_mse = get_pca_mse(data, pca_dim)
        mse_list.append(pca_mse)

    plt.xlabel('# Components')
    plt.ylabel('MSE')
    plt.plot(dims, mse_list)
    plt.show(block=False)

# def autoencoder_evaluation(data):
#     pca_encode, pca_decode, explained_var, pca_mse = pca_analysis(displacements, pca_dim)
#     encoded_pca_displacements = pca_encode(displacements)
#     decoded_pca_displacements = pca_decode(encoded_pca_displacements)

#     ae_encode, ae_decode = autoencoder_analysis(encoded_pca_displacements, latent_dim=ae_dim, epochs=800, batch_size=len(displacements), layers=[128, 32])
#     decoded_autoencoder_displacements = pca_decode(ae_decode(ae_encode(encoded_pca_displacements)))

def main():
    # Loading the rest pose
    base_verts, face_indices = my_utils.load_base_vert_and_face_dmat_to_numpy(base_path)
    base_verts_eig = p2e(base_verts)
    face_indices_eig = p2e(face_indices)

    # Loading displacements for training data
    displacements = my_utils.load_displacement_dmats_to_numpy(base_path)

    # autoencoder_analysis(data)

    # Do the PCA analysis
    pca_dim = 20
    ae_dim = 3
    train_in_pca_space = True

    if train_in_pca_space:
        pca_encode, pca_decode, explained_var, good_pca_mse = pca_analysis(displacements, pca_dim)
        encoded_pca_displacements = pca_encode(displacements)
        ae_encode, ae_decode, ae_mse = autoencoder_analysis(
                                        encoded_pca_displacements,
                                        latent_dim=ae_dim,
                                        epochs=2000,
                                        batch_size=len(displacements),
                                        layers=[128, 32],
                                        #pca_weights=explained_var,
                                    )
        decoded_autoencoder_displacements = pca_decode(ae_decode(ae_encode(encoded_pca_displacements)))
    else:
        ae_encode, ae_decode, ae_mse = autoencoder_analysis(displacements, latent_dim=ae_dim, epochs=1000, batch_size=len(displacements), layers=[128, 32])
        decoded_autoencoder_displacements = ae_decode(ae_encode(displacements))

    # Compare it with PCA of same dimension
    pca_encode, pca_decode, explained_var, pca_mse = pca_analysis(displacements, ae_dim)
    encoded_pca_displacements = pca_encode(displacements)
    decoded_pca_displacements = pca_decode(encoded_pca_displacements)
    print("PCA MSE =", pca_mse)
    print("Autoencoder MSE =", ae_mse)

    # Set up
    viewer = igl.viewer.Viewer()
    viewer.data.set_mesh(base_verts_eig, face_indices_eig)


    def pre_draw(viewer):
        global current_frame, current_mode
        
        if viewer.core.is_animating:
            if modes[current_mode] == 'baseline':
                viewer.data.set_vertices(p2e(base_verts + displacements[current_frame]))
            elif modes[current_mode] == 'pca':
                viewer.data.set_vertices(p2e(base_verts + decoded_pca_displacements[current_frame]))
                # viewer.data.set_colors(colours[current_frame])
            elif modes[current_mode] == 'autoencoder':
                viewer.data.set_vertices(p2e(base_verts + decoded_autoencoder_displacements[current_frame]))
            else:
                assert False  # Shouldn't happen
                

            viewer.data.compute_normals()
            current_frame = (current_frame + 1) % len(displacements)

        return False

    def key_down(viewer, key, mods):
        global current_mode
        if key == ord(' '):
            viewer.core.is_animating = not viewer.core.is_animating
        elif key == ord('D') or key == ord('d'):
            current_mode = (current_mode + 1) % len(modes)

        return False


    viewer.callback_pre_draw = pre_draw
    viewer.callback_key_down = key_down
    viewer.core.is_animating = False
    viewer.core.invert_normals = True
    viewer.data.face_based = True
    # viewer.core.camera_zoom = 2.5
    viewer.core.animation_max_fps = 30.0

    viewer.launch()

if __name__ == "__main__":
    main()
