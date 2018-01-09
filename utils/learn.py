import time
import subprocess
import json
import os

import numpy
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pyigl as igl
from utils.iglhelpers import e2p, p2e
from utils import my_utils

# Global Vars
modes = {0: 'baseline', 1:'pca', 2:'autoencoder'}
current_mode = 0
current_frame = 0
visualize_test_data = True
# training_data_root = 'training_data/first_interaction/'
# test_data_root = 'training_data/test_interaction/'
training_data_root = 'training_data/fixed_material_model/'
test_data_root = 'training_data/fixed_material_model_test/'

def autoencoder_analysis(
    data,
    test_data,
    activation='relu',
    latent_dim=3,
    epochs=100,
    batch_size=100,
    layers=[32, 16],
    pca_weights=None,
    pca_object=None,
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    energy_model_config=None): # TODO should probably just pass in both configs here 
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
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(data)

    # TODO: Do I need to shuffle?
    train_data = flatten_data(data)
    test_data = flatten_data(data[:10] if test_data is None else test_data)

    if energy_model_config['enabled']:
        energy_train_path = os.path.join(model_root, 'training_data/training/energy.json')
        energy_test_path = os.path.join(model_root, 'training_data/validation/energy.json')
        with open(energy_train_path, 'r') as f:
            energy_json = json.load(f)
            energy_train_data = energy_json['potential_energy_per_frame']
        with open(energy_test_path, 'r') as f:
            energy_json = json.load(f)
            energy_test_data = energy_json['potential_energy_per_frame']

#        normalize_factor = 1e6 # TODO should get this from simulation data?
        normalize_factor = 40000.0
        energy_train_data = numpy.array(energy_train_data) / normalize_factor 
        energy_test_data =  numpy.array(energy_test_data) / normalize_factor 

    ## Preprocess the data
    # mean = numpy.mean(train_data, axis=0)
    # std = numpy.std(train_data, axis=0)
    
    mean = numpy.mean(train_data)
    std = numpy.std(train_data)
    s_min = numpy.min(train_data)
    s_max = numpy.max(train_data)
    print(mean)
    print(std)

    # TODO dig into this. Why does it mess up?
    def normalize(data):
        return data
        #return (data - mean) / std
        # return numpy.nan_to_num((train_data - s_min) / (s_max - s_min))
    def denormalize(data):
        return data
        #return data * std + mean
        # return data * (s_max - s_min) + s_mi

    # TODO loss in full space. Different results?
    # Custom layer if we need it
    if pca_object:
        class PCALayer(Layer):
            def __init__(self, pca_object, is_inv, fine_tune=True, **kwargs):
                self.pca_object = pca_object
                self.is_inv = is_inv
                self.fine_tune = fine_tune
                super(PCALayer, self).__init__(**kwargs)

            def build(self, input_shape):
                # Create a trainable weight variable for this layer.
                self.kernel = self.add_weight(name='kernel', 
                                              shape=None, # Inferred 
                                              initializer=lambda x: K.variable(self.pca_object.components_) if self.is_inv else K.variable(self.pca_object.components_.T),
                                              trainable=self.fine_tune)
                self.mean = self.add_weight(name='mean', 
                                              shape=None, # Inferred
                                              initializer=lambda x: K.variable(self.pca_object.mean_),
                                              trainable=self.fine_tune)
                super(PCALayer, self).build(input_shape)  # Be sure to call this somewhere!

            def call(self, x):
                if self.is_inv:
                    return K.dot(x, self.kernel) + self.mean
                else:
                    return K.dot(x - self.mean, self.kernel)

            def compute_output_shape(self, input_shape):
                if self.is_inv:
                    return (input_shape[0], len(self.pca_object.mean_))
                else:
                    return (input_shape[0], len(self.pca_object.components_))

    ## Set up the network
    
    input = Input(shape=(len(train_data[0]),), name="encoder_input")
    output = input
    
    if pca_object is not None:
        # output = Lambda(pca_transform_layer)(output)
        # output = PCALayer(pca_object, is_inv=False, fine_tune=do_fine_tuning)(output)

        W = pca_object.components_.T
        b =  -pca_object.mean_ @ pca_object.components_.T
        output = Dense(pca_object.n_components_, activation='linear', weights=[W,b], trainable=do_fine_tuning, name="pca_encode_layer")(output)

    for i, layer_width in enumerate(layers):
        output = Dense(layer_width, activation=activation, name="dense_encode_layer_" + str(i))(output)

    # -- Encoded layer
    output = Dense(latent_dim, activation=activation, name="encoded_layer")(output)  # TODO Tanh into encoded layer to bound vars?

    if energy_model_config['enabled']:
        energy_output = output
        for i, layer_width in enumerate(energy_model_config['layer_sizes']):
            energy_output = Dense(layer_width, activation=energy_model_config['activation'], name="dense_energy_layer_" + str(i))(energy_output)
        # TODO what kind of activation on the last layer?            
        energy_output = Dense(1, activation='linear', name="energy_output")(energy_output)


    for i, layer_width in enumerate(reversed(layers)):
        output = Dense(layer_width, activation=activation, name="dense_decode_layer_" + str(i))(output)
    
    if pca_object is not None:
        output = Dense(len(pca_object.components_), activation='linear', name="to_pca_decode_layer")(output) ## TODO is this right?? Possibly should just change earlier layer width
        # output = Lambda(pca_inv_transform_layer)(output)
        # output = PCALayer(pca_object, is_inv=True, fine_tune=do_fine_tuning)(output)

        W = pca_object.components_
        b = pca_object.mean_
        output = Dense(len(train_data[0]), activation='linear', weights=[W,b], trainable=do_fine_tuning, name="pca_decode_layer")(output)
    else:
        output = Dense(len(train_data[0]), activation='linear', name="decoder_output_layer")(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    # TODO make it work when energy_model_config['enabled'] is false
    if energy_model_config['enabled']:
        autoencoder = Model(input, outputs=[output, energy_output])
    else:
        autoencoder = Model(input, output)


    ## Set the optimization parameters
    pca_weights = pca_weights / pca_weights.sum() if pca_weights is not None else None
    def pca_weighted_mse(y_pred, y_true):
        mse = K.mean(pca_weights * K.square(y_true - y_pred), axis=1)
        return mse

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    if energy_model_config['enabled']:
        autoencoder.compile(
            optimizer=optimizer,
            loss={
                'pca_decode_layer': 'mean_squared_error',
                'energy_output': 'mean_squared_error'
            },
            loss_weights={
                'pca_decode_layer': autoencoder_config['loss_weight'],
                'energy_output': energy_model_config['loss_weight']
            }
        )
    else:
        autoencoder.compile(
            optimizer=optimizer,
            loss='mean_squared_error' if pca_weights is None else pca_weighted_mse,
        )
    
    
    hist = History()
    model_start_time = time.time()
    if energy_model_config['enabled']:
        autoencoder.fit(
            {'encoder_input': train_data},
            {'pca_decode_layer': train_data, 'energy_output': energy_train_data},
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            #validation_split=0.02,
            validation_data=[ # TODO get this working with my validation set
                test_data,
                [test_data, energy_test_data],
            ],
            callbacks=[hist]
        )
    else:
        autoencoder.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_data, test_data),
            callbacks=[hist]
            )

    # output_path = 'trained_models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    # autoencoder.save(output_path)

    print("Total training time: ", time.time() - model_start_time)
    
    if energy_model_config['enabled']:
        autoencoder,encoder, decoder, energy_model = my_utils.decompose_ae(autoencoder, do_energy=True)
    else:
        autoencoder,encoder, decoder = my_utils.decompose_ae(autoencoder, do_energy=False)

    # Save the models in both tensorflow and keras formats
    if model_root:
        print("Saving model...")
        if energy_model_config['enabled']:
            models_with_names = [
                (autoencoder, "autoencoder"),
                (encoder, "encoder"),
                (decoder, "decoder"),
                (energy_model, "energy_model")
            ]
        else:
            models_with_names = [
                (autoencoder, "autoencoder"),
                (encoder, "encoder"),
                (decoder, "decoder"),
            ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  # Save the keras model

        history_path = os.path.join(keras_models_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(hist.history, f, indent=2)

        print ("Finished saving model.")

    def encode(decoded_data):
        return encoder.predict(normalize(flatten_data(decoded_data))) 
    def decode(encoded_data):
        return unflatten_data(denormalize(decoder.predict(encoded_data)))

    return encode, decode, hist


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
        #return pca.transform(flatten_data(decoded_data))        
        return pca.components_ @ flatten_data(decoded_data).T # We don't need to subtract the mean before going to/from reduced space?
    def decode(encoded_data):
        # return unflatten_data(pca.inverse_transform(encoded_data))
        return unflatten_data((pca.components_.T @ encoded_data).T)

    return pca, encode, decode, pca.explained_variance_ratio_, mse

def get_pca_mse(data, n_components):
    pca, pca_encode, pca_decode, explained_var, pca_mse = pca_analysis(data, n_components)
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


def generate_model(
    model_root, # This is the root of the standard formatted directory created by build-model.py
    learning_config, # Comes from the training config file
    ):
    autoencoder_config = learning_config['autoencoder_config']
    energy_model_config = learning_config['energy_model_config']

    training_data_path = os.path.join(model_root, 'training_data/training')
    validation_data_path = os.path.join(model_root, 'training_data/validation')
    # Loading the rest pose
    base_verts, face_indices = my_utils.load_base_vert_and_face_dmat_to_numpy(training_data_path)
    base_verts_eig = p2e(base_verts)
    face_indices_eig = p2e(face_indices)

    # Loading displacements for training data
    displacements = my_utils.load_displacement_dmats_to_numpy(training_data_path)
    test_displacements = my_utils.load_displacement_dmats_to_numpy(validation_data_path)
    flatten_data, unflatten_data = my_utils.get_flattners(displacements)

    # Do the training
    pca_ae_train_dim = autoencoder_config['pca_layer_dim']
    pca_compare_dims = autoencoder_config['pca_compare_dims']
    ae_dim = autoencoder_config['ae_encoded_dim']
    ae_epochs = autoencoder_config['training_epochs']
    batch_size = autoencoder_config['batch_size'] if autoencoder_config['batch_size'] > 0 else len(displacements)
    layers = autoencoder_config['non_pca_layer_sizes']
    do_fine_tuning = autoencoder_config['do_fine_tuning']
    activation = autoencoder_config['activation']
    train_autoencoder = True
    train_in_pca_space = False
    save_pca_components = True
    save_autoencoder = True

    training_results = { 
        'autoencoder': {},
        'pca': {},
    }

    # Normal low dim pca first
    for pca_dim in pca_compare_dims:
        pca, pca_encode, pca_decode, explained_var, pca_mse = pca_analysis(displacements, pca_dim)
        encoded_pca_displacements = pca_encode(displacements)
        decoded_pca_displacements = pca_decode(encoded_pca_displacements)
        encoded_pca_test_displacements = pca_encode(test_displacements)
        decoded_pca_test_displacements = pca_decode(encoded_pca_test_displacements)

        if save_pca_components:
            pca_results_filename = os.path.join(model_root, 'pca_results/pca_components_' + str(pca_dim) + '.dmat')
            print('Saving pca results to', pca_results_filename)
            my_utils.create_dir_if_not_exist(os.path.dirname(pca_results_filename))
            my_utils.save_numpy_mat_to_dmat(pca_results_filename, numpy.ascontiguousarray(pca.components_))

            training_mse = mean_squared_error(flatten_data(decoded_pca_displacements), flatten_data(displacements))
            validation_mse = mean_squared_error(flatten_data(decoded_pca_test_displacements), flatten_data(test_displacements))
            training_results['pca'][str(pca_dim) + '-components'] = {}
            training_results['pca'][str(pca_dim) + '-components']['training-mse'] = training_mse
            training_results['pca'][str(pca_dim) + '-components']['validation-mse'] = validation_mse

            print(str(pca_dim) + ' training MSE: ', training_mse)

    # print("PCA MSE =", pca_mse)
    # print("PCA Test MSE =", mean_squared_error(flatten_data(decoded_pca_test_displacements), flatten_data(test_displacements)))


    # if train_autoencoder:
    #     if train_in_pca_space: # TODO enable for faster training
    #         # High dim pca to train autoencoder
    #         high_dim_pca, high_dim_pca_encode, high_dim_pca_decode, explained_var, good_pca_mse = pca_analysis(displacements, pca_ae_train_dim)
    #         encoded_high_dim_pca_displacements = high_dim_pca_encode(displacements)
    #         encoded_high_dim_pca_test_displacements = high_dim_pca_encode(test_displacements)

    #         ae_encode, ae_decode, hist = autoencoder_analysis(
    #                                         encoded_high_dim_pca_displacements,
    #                                         encoded_high_dim_pca_test_displacements,
    #                                         latent_dim=ae_dim,
    #                                         epochs=ae_epochs,
    #                                         batch_size=len(displacements),
    #                                         layers=[200, 200],
    #                                         #pca_weights=explained_var,
    #                                     )

    #         decoded_autoencoder_displacements = high_dim_pca_decode(ae_decode(ae_encode(encoded_high_dim_pca_displacements)))
    #         decoded_autoencoder_test_displacements = high_dim_pca_decode(ae_decode(ae_encode(high_dim_pca_encode(test_displacements))))

    #     else:
    # High dim pca to train autoencoder
    high_dim_pca, high_dim_pca_encode, high_dim_pca_decode, explained_var, good_pca_mse = pca_analysis(displacements, pca_ae_train_dim)
    encoded_high_dim_pca_displacements = high_dim_pca_encode(displacements)
    encoded_high_dim_pca_test_displacements = high_dim_pca_encode(test_displacements)

    ae_encode, ae_decode, hist = autoencoder_analysis(
                                    displacements,
                                    test_displacements,
                                    activation=activation,
                                    latent_dim=ae_dim,
                                    epochs=ae_epochs,
                                    batch_size=batch_size,
                                    layers=layers, # [200, 200, 50] First two layers being wide seems best so far. maybe an additional narrow third 0.0055 see
                                    pca_object=high_dim_pca,
                                    do_fine_tuning=do_fine_tuning,
                                    model_root=model_root,
                                    autoencoder_config=autoencoder_config,
                                    energy_model_config=energy_model_config,
                                )

    decoded_autoencoder_displacements = ae_decode(ae_encode(displacements))
    decoded_autoencoder_test_displacements = ae_decode(ae_encode(test_displacements))
    training_results['autoencoder']['training-mse'] = mean_squared_error(flatten_data(decoded_autoencoder_displacements), flatten_data(displacements))
    training_results['autoencoder']['validation-mse'] = mean_squared_error(flatten_data(decoded_autoencoder_test_displacements), flatten_data(test_displacements))

    # TODO output energy loss as well
    with open(os.path.join(model_root, 'training_results.json'), 'w') as f:
        json.dump(training_results, f, indent=2)
    # print("Autoencoder Test MSE =", mean_squared_error(flatten_data(decoded_autoencoder_test_displacements), flatten_data(test_displacements)))
    # print("PCA Test MSE =", mean_squared_error(flatten_data(decoded_pca_test_displacements), flatten_data(test_displacements)))

    # print("Autoencoder Train MSE =", mean_squared_error(flatten_data(displacements), flatten_data(decoded_autoencoder_displacements)))
    # print("PCA Train MSE =", pca_mse)

    # mse_per_pose = [mean_squared_error(d, dt) for d, dt in zip(decoded_autoencoder_test_displacements, test_displacements)]
    
    

    # encoded_output_pca = model_base_filename + '_pca_encoded.json'
    # encoded_output_ae = model_base_filename + '_ae_encoded.json'
    # decoder_jacobian_norms_path = model_base_filename + '_decoder_jacobian_norms.json'
    # encoded_output_pca_data = encoded_pca_test_displacements.T
    # encoded_output_ae_data = ae_encode(test_displacements)
    
    # print("Computing jacobians...")
    # decoder_jacobian_norms = [numpy.linalg.norm(my_utils.fd_jacobian(ae_decode, x, 0.0005, is_keras=True), ord='fro') for x in encoded_output_ae_data]
    # fig, ax1 = plt.subplots()
    # plt.plot(mse_per_pose)
    # ax2 = ax1.twinx()
    # ax2.plot(decoder_jacobian_norms)
    # print("Done.")

    # with open(encoded_output_pca, 'w') as f:
    #     json.dump(encoded_output_pca_data.tolist(), f)
    # with open(encoded_output_ae, 'w') as f:
    #     json.dump(encoded_output_ae_data.tolist(), f)
    # with open(decoder_jacobian_norms_path, 'w') as f:
    #     json.dump(decoder_jacobian_norms, f)

def main():
    # Loading the rest pose
    base_verts, face_indices = my_utils.load_base_vert_and_face_dmat_to_numpy(training_data_root)
    base_verts_eig = p2e(base_verts)
    face_indices_eig = p2e(face_indices)

    # Loading displacements for training data
    displacements = my_utils.load_displacement_dmats_to_numpy(training_data_root)
    test_displacements = my_utils.load_displacement_dmats_to_numpy(test_data_root)
    flatten_data, unflatten_data = my_utils.get_flattners(displacements)

    # Do the training
    pca_ae_train_dim = 20
    pca_dim = 3
    ae_dim = 3
    ae_epochs = 1000
    train_autoencoder = True
    train_in_pca_space = False
    save_pca_components = False
    save_autoencoder = True
    # model_base_filename = training_data_root + "elu_model" #
    model_base_filename = "models/baseline_elu" #

    # Normal low dim pca first
    pca, pca_encode, pca_decode, explained_var, pca_mse = pca_analysis(displacements, pca_dim)
    encoded_pca_displacements = pca_encode(displacements)
    decoded_pca_displacements = pca_decode(encoded_pca_displacements)
    encoded_pca_test_displacements = pca_encode(test_displacements)
    decoded_pca_test_displacements = pca_decode(encoded_pca_test_displacements)

    if save_pca_components:
        my_utils.save_numpy_mat_to_dmat(training_data_root + "pca_components_" + str(pca_dim) + ".dmat", numpy.ascontiguousarray(pca.components_))

    print("PCA MSE =", pca_mse)
    print("PCA Test MSE =", mean_squared_error(flatten_data(decoded_pca_test_displacements), flatten_data(test_displacements)))


    if train_autoencoder:
        if train_in_pca_space:
            # High dim pca to train autoencoder
            high_dim_pca, high_dim_pca_encode, high_dim_pca_decode, explained_var, good_pca_mse = pca_analysis(displacements, pca_ae_train_dim)
            encoded_high_dim_pca_displacements = high_dim_pca_encode(displacements)
            encoded_high_dim_pca_test_displacements = high_dim_pca_encode(test_displacements)

            ae_encode, ae_decode, hist = autoencoder_analysis(
                                            encoded_high_dim_pca_displacements,
                                            encoded_high_dim_pca_test_displacements,
                                            latent_dim=ae_dim,
                                            epochs=ae_epochs,
                                            batch_size=len(displacements),
                                            layers=[200, 200],
                                            #pca_weights=explained_var,
                                        )

            decoded_autoencoder_displacements = high_dim_pca_decode(ae_decode(ae_encode(encoded_high_dim_pca_displacements)))
            decoded_autoencoder_test_displacements = high_dim_pca_decode(ae_decode(ae_encode(high_dim_pca_encode(test_displacements))))

        else:
            # High dim pca to train autoencoder
            high_dim_pca, high_dim_pca_encode, high_dim_pca_decode, explained_var, good_pca_mse = pca_analysis(displacements, pca_ae_train_dim)
            encoded_high_dim_pca_displacements = high_dim_pca_encode(displacements)
            encoded_high_dim_pca_test_displacements = high_dim_pca_encode(test_displacements)

            ae_encode, ae_decode, hist = autoencoder_analysis(
                                            displacements,
                                            test_displacements,
                                            latent_dim=ae_dim,
                                            epochs=ae_epochs,
                                            batch_size=len(displacements),
                                            layers=[200, 200], # [200, 200, 50] First two layers being wide seems best so far. maybe an additional narrow third 0.0055 see
                                            pca_object=high_dim_pca,
                                            do_fine_tuning=False,
                                            model_root=model_base_filename
                                        )

            decoded_autoencoder_displacements = ae_decode(ae_encode(displacements))
            decoded_autoencoder_test_displacements = ae_decode(ae_encode(test_displacements))

        print("Autoencoder Test MSE =", mean_squared_error(flatten_data(decoded_autoencoder_test_displacements), flatten_data(test_displacements)))
        print("PCA Test MSE =", mean_squared_error(flatten_data(decoded_pca_test_displacements), flatten_data(test_displacements)))

        print("Autoencoder Train MSE =", mean_squared_error(flatten_data(displacements), flatten_data(decoded_autoencoder_displacements)))
        print("PCA Train MSE =", pca_mse)

        mse_per_pose = [mean_squared_error(d, dt) for d, dt in zip(decoded_autoencoder_test_displacements, test_displacements)]
        
        

        encoded_output_pca = model_base_filename + '_pca_encoded.json'
        encoded_output_ae = model_base_filename + '_ae_encoded.json'
        decoder_jacobian_norms_path = model_base_filename + '_decoder_jacobian_norms.json'
        encoded_output_pca_data = encoded_pca_test_displacements.T
        encoded_output_ae_data = ae_encode(test_displacements)
        
        print("Computing jacobians...")
        decoder_jacobian_norms = [numpy.linalg.norm(my_utils.fd_jacobian(ae_decode, x, 0.0005, is_keras=True), ord='fro') for x in encoded_output_ae_data]
        fig, ax1 = plt.subplots()
        plt.plot(mse_per_pose)
        ax2 = ax1.twinx()
        ax2.plot(decoder_jacobian_norms)
        print("Done.")

        with open(encoded_output_pca, 'w') as f:
            json.dump(encoded_output_pca_data.tolist(), f)
        with open(encoded_output_ae, 'w') as f:
            json.dump(encoded_output_ae_data.tolist(), f)
        with open(decoder_jacobian_norms_path, 'w') as f:
            json.dump(decoder_jacobian_norms, f)

        plt.show(block=False)
    # libigl Set up
    viewer = igl.viewer.Viewer()
    viewer.data.set_mesh(base_verts_eig, face_indices_eig)

    def pre_draw(viewer):
        global current_frame, current_mode, visualize_test_data
        
        if viewer.core.is_animating:
            n_frames = len(test_displacements) if visualize_test_data else len(displacements)

            if modes[current_mode] == 'baseline':
                ds = test_displacements if visualize_test_data else displacements
                viewer.data.set_vertices(p2e(base_verts + ds[current_frame % n_frames]))
            elif modes[current_mode] == 'pca':
                ds = decoded_pca_test_displacements if visualize_test_data else decoded_pca_displacements
                viewer.data.set_vertices(p2e(base_verts + ds[current_frame % n_frames]))
            elif modes[current_mode] == 'autoencoder':
                ds = decoded_autoencoder_test_displacements if visualize_test_data else decoded_autoencoder_displacements
                viewer.data.set_vertices(p2e(base_verts + ds[current_frame % n_frames]))
            else:
                assert False  # Shouldn't happen
            
            current_frame = (current_frame + 1) % n_frames
            viewer.data.compute_normals()
        return False

    def key_down(viewer, key, mods):
        global current_mode, visualize_test_data
        if key == ord(' '):
            viewer.core.is_animating = not viewer.core.is_animating
        elif key == ord('D') or key == ord('d'):
            current_mode = (current_mode + 1) % len(modes)
            print("Current mode", modes[current_mode])
        elif key == ord('V') or key == ord('v'):
            visualize_test_data = not visualize_test_data
            print("Visualize test data:", visualize_test_data)

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
