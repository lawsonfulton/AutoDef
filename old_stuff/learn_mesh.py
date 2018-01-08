import time
from multiprocessing import Pool

import pyigl as igl
import numpy
import scipy
from sklearn.decomposition import PCA

from iglhelpers import e2p, p2e

# Unused placeholders
_d = igl.eigen.MatrixXd()
_i = igl.eigen.MatrixXi()

# num_samples = 1033
current_frame = 0

verts_sample = []

show_decoded = False



def do_training(training_data):
    import keras
    from keras.layers import Input, Dense
    from keras.models import Model, load_model
    import datetime

    start_time = time.time()
    
    training_sample_size = 1000
    test_sample_size = 33

    print("Loading training data...")
    data = training_data.reshape((len(training_data), 3 * len(training_data[0])))
    numpy.random.shuffle(data)
    
    train_data = numpy.array(data[:training_sample_size])
    test_data = numpy.array(data[training_sample_size:training_sample_size+test_sample_size])

    print("Done loading: ", time.time()-start_time)
    
    # this is the size of our encoded representations
    encoded_dim = 20

    # Single autoencoder
    # initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.01, seed=5)
    # bias_initializer = initializer
    activation = 'relu' #keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'
    
    input = Input(shape=(len(train_data[0]),))
    output = input
    # output = Dense(200, activation=activation)(input)
    output = Dense(1000, activation=activation)(output)
    output = Dense(encoded_dim, activation=activation, name="encoded")(output)
    output = Dense(1000, activation=activation)(output)
    # output = Dense(200, activation=activation)(output)
    output = Dense(len(train_data[0]), activation='linear')(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    autoencoder = Model(input, output)

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer=optimizer,
        loss='mean_squared_error'
    )
    
    model_start_time = time.time()
    autoencoder.fit(
        train_data, train_data,
        epochs=4,
        batch_size=16,
        shuffle=True,
        validation_data=(test_data, test_data)
    )

    output_path = 'models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    autoencoder.save(output_path)

    print("Total model time: ", time.time() - model_start_time)

    # Display
    predict_start = time.time()
    test_data = test_data
    decoded_samples = autoencoder.predict(test_data)
    print('Predict took: ', time.time() - predict_start)
    
    print("Total runtime: ", time.time() - start_time)

base_path = 'training_data/first_interaction/'
#base_path = 'data-armadillo/verts'
# base_path = 'data-armadillo/verts' # Need to change to i+1 for this one
def read_dmat(i):
    #filename = 'data-armadillo/verts%d.ply' % (i + 1)
    filename = base_path + 'displacements_%d.dmat' % i
    if(i % 13 == 0):
        print('.', end='', flush=True)
    verts = igl.eigen.MatrixXd()
    igl.readDMAT(filename, verts, _i, _d, _d)
    return e2p(verts)

def load_samples(num_samples):
    print('Loading', num_samples, 'samples...', end='', flush=True)
    p = Pool(16)
    numpy_verts_sample = numpy.array(p.map(read_dmat, range(num_samples)))
    p.terminate()
    print()
    print('Done.')
    return numpy_verts_sample

def get_initial_verts_and_faces():
    initial_verts = igl.eigen.MatrixXd()
    initial_faces = igl.eigen.MatrixXi()
    #path = base_path + '_and_faces.ply'
    path = base_path + 'base_mesh.ply'

    igl.readPLY(path, initial_verts, initial_faces, _d, _d)

    return initial_verts, initial_faces


# TODO convert it to offsets instead of absolute positions
def main():
    global verts_sample

    initial_verts, initial_faces = get_initial_verts_and_faces()
    numpy_base_verts = e2p(initial_verts).flatten()

    start = time.time()
    num_samples = 150
    numpy_displacements_sample = load_samples(num_samples)
    #numpy_displacements_sample = numpy_verts_sample - initial_verts

    num_verts = len(numpy_displacements_sample[0])
    print(num_verts)

    print('Loading...')
    #verts_sample = [p2e(m) for m in numpy_verts_sample]
    displacements_sample = [p2e(m) for m in numpy_displacements_sample]
    print(e2p(initial_verts))
    print(len(e2p(initial_verts)))
    position_sample = [p2e(m) for m in numpy_displacements_sample]
    print("Took:", time.time() - start)


    use_pca = True
    do_analysis = False
    if do_analysis:
        if use_pca:
            ### PCA Version
            print("Doing PCA...")
            test_data = numpy_displacements_sample[:test_size] * 1.0
            test_data_eigen = displacements_sample[:test_size]
            numpy.random.shuffle(numpy_displacements_sample)
            # train_data = numpy_verts_sample[test_size:test_size+train_size]
            train_data = numpy_displacements_sample[0:train_size]
            print(train_data[1][:30])
            print(num_verts)
            pca = PCA(n_components=3)
            pca.fit(train_data.reshape((train_size, 3 * num_verts)))

            # def encode(q):
            #     return pca.transform(numpy.array([q.flatten() - numpy_base_verts]))[0]

            # def decode(z):
            #     return (numpy_base_verts + pca.inverse_transform(numpy.array([z]))[0]).reshape((num_verts, 3))

            # print(numpy.equal(test_data[0].flatten().reshape((len(test_data[0]),3)), test_data[0]))
            # print(encode(test_data[0]))

            test_data_encoded = pca.transform(test_data.reshape(test_size, 3 * num_verts))
            test_data_decoded = (numpy_base_verts + pca.inverse_transform(test_data_encoded)).reshape(test_size, num_verts, 3)
            test_data_decoded_eigen = [p2e(m) for m in test_data_decoded]
            ### End of PCA version
        else:
            ### Autoencoder
            import keras
            from keras.layers import Input, Dense
            from keras.models import Model, load_model
            import datetime

            start_time = time.time()
            
            train_size = num_samples
            test_size = num_samples
            test_data = numpy_displacements_sample[:test_size].reshape(test_size, 3 * num_verts)
            test_data_eigen = numpy_displacements_sample[:test_size]
            # numpy.random.shuffle(numpy_displacements_sample)
            # train_data = numpy_verts_sample[test_size:test_size+train_size]
            train_data = numpy_displacements_sample[0:train_size].reshape((train_size, 3 * num_verts))

            mean = numpy.mean(train_data, axis=0)
            std = numpy.std(train_data, axis=0)
            
            mean = numpy.mean(train_data)
            std = numpy.std(train_data)

            s_min = numpy.min(train_data)
            s_max = numpy.max(train_data)
            

            def normalize(data):
                return numpy.nan_to_num((data - mean) / std)
                # return numpy.nan_to_num((train_data - s_min) / (s_max - s_min))
            def denormalize(data):
                return data * std + mean
                # return data * (s_max - s_min) + s_min

            train_data = normalize(train_data)
            test_data = normalize(test_data)

            # print(train_data)
            # print(mean)
            # print(std)
            # exit()
            # this is the size of our encoded representations
            encoded_dim = 3

            # Single autoencoder
            # initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.01, seed=5)
            # bias_initializer = initializer
            activation = keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'
            
            input = Input(shape=(len(train_data[0]),))
            output = input
            output = Dense(30, activation=activation)(input)
            output = Dense(512, activation=activation)(output)
            output = Dense(64, activation=activation)(output)
            output = Dense(encoded_dim, activation=activation, name="encoded")(output)
            output = Dense(64, activation=activation)(output)
            output = Dense(512, activation=activation)(output)
            output = Dense(30, activation=activation)(output)
            output = Dense(len(train_data[0]), activation='linear')(output)#'linear',)(output) # First test seems to indicate no change on output with linear

            autoencoder = Model(input, output)

            optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
            autoencoder.compile(
                optimizer=optimizer,
                loss='mean_squared_error'
            )
            
            model_start_time = time.time()
            autoencoder.fit(
                train_data, train_data,
                epochs=1000,
                batch_size=num_samples,
                shuffle=True,
                validation_data=(test_data, test_data)
            )

            # output_path = 'trained_models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
            # autoencoder.save(output_path)

            print("Total model time: ", time.time() - model_start_time)

            # Display
            
            decoded_samples = denormalize(autoencoder.predict(test_data))
            #decoded_samples = autoencoder.predict(test_data) * std + mean

            test_data_decoded = (numpy_base_verts + decoded_samples).reshape(test_size, num_verts, 3)
            test_data_decoded_eigen = [p2e(m) for m in test_data_decoded]
            ### End of Autoencoder

        # Error colours 
        error = numpy.sum((test_data_decoded - numpy_verts_sample) ** 2, axis=2)
        colours = [igl.eigen.MatrixXd() for _ in range(num_samples)]
        for i in range(num_samples):
            igl.jet(p2e(error[i]), True, colours[i])

    # Set up
    viewer = igl.viewer.Viewer()

    #viewer.data.set_mesh(initial_verts, initial_faces)
    # viewer.data.set_face_based(False)
    colours = igl.eigen.MatrixXd(num_verts)
    viewer.data.set_points(initial_verts, colours)
    def pre_draw(viewer):
        global current_frame, verts_sample, show_decoded
        
        if viewer.core.is_animating:
            print(current_frame)
            print(show_decoded)
            if show_decoded:
                viewer.data.set_points(test_data_decoded_eigen[current_frame], colours)
            else:
                viewer.data.set_points(position_sample[current_frame], colours)


            viewer.data.compute_normals()
            current_frame = (current_frame + 1) % num_samples

        return False


    viewer.callback_pre_draw = pre_draw
    viewer.callback_key_down = key_down
    viewer.core.is_animating = False
    # viewer.core.camera_zoom = 2.5
    viewer.core.animation_max_fps = 30.0

    viewer.launch()




def key_down(viewer, key, mods):
    global show_decoded
    if key == ord(' '):
        viewer.core.is_animating = not viewer.core.is_animating
    elif key == ord('D') or key == ord('d'):
        show_decoded = not show_decoded
        print(show_decoded)

    return False


if __name__=='__main__':
    main()