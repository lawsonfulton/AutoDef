import my_keras_to_tensorflow as k2tf
from keras.models import load_model
import numpy
from keras import backend as K

training_data_root = "./training_data/fixed_material_model/"
model_base_filename = training_data_root + "elu_model"

autoencoder = load_model(model_base_filename + "_autoencoder.hdf5")
autoencoder = k2tf.save_keras_model_as_tf(autoencoder, model_base_filename + "_autoencoder.pb")
K.clear_session()

decoder = load_model("./training_data/fixed_material_model/model_decoder.hdf5")
decoder = k2tf.save_keras_model_as_tf(decoder, model_base_filename + "_decoder.pb")
K.clear_session()

encoder = load_model(model_base_filename + "_encoder.hdf5")
encoder = k2tf.save_keras_model_as_tf(encoder, model_base_filename + "_encoder.pb")
K.clear_session()
