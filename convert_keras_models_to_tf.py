import my_keras_to_tensorflow as k2tf
from keras.models import load_model
import numpy
from keras import backend as K

autoencoder = load_model("./training_data/fixed_material_model/model_autoencoder.hdf5")
autoencoder = k2tf.save_keras_model_as_tf(autoencoder, "./training_data/fixed_material_model/model_autoencoder.pb")
K.clear_session()

decoder = load_model("./training_data/fixed_material_model/model_decoder.hdf5")
decoder = k2tf.save_keras_model_as_tf(decoder, "./training_data/fixed_material_model/model_decoder.pb")
K.clear_session()

encoder = load_model("./training_data/fixed_material_model/model_encoder.hdf5")
encoder = k2tf.save_keras_model_as_tf(encoder, "./training_data/fixed_material_model/model_encoder.pb")
K.clear_session()
