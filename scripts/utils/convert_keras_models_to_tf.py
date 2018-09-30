import os

from utils import my_keras_to_tensorflow as k2tf
from utils import my_utils

import numpy

def convert_keras_models_to_tf(model_root):
    import keras
    from keras.models import load_model, model_from_json
    from keras import backend as K

    keras_models_dir = os.path.join(model_root, 'keras_models')

    tf_models_dir = os.path.join(model_root, 'tf_models')
    my_utils.create_dir_if_not_exist(tf_models_dir)

    model_names = ['autoencoder', 'decoder', 'encoder', 'energy_model', 'l1_discrete_energy_model', 'direct_energy_model']

    K.clear_session()
    for model_name in model_names:
        model_path = os.path.join(keras_models_dir, model_name + '.hdf5')
        if os.path.exists(model_path):
            
            # Custom stuff
            keras.losses.energy_loss = lambda x,y: x
            keras.losses.UTMU_loss = lambda x,y: x
            # keras.losses.contractive_loss = lambda x,y: x
            keras.regularizers.reg = lambda : (lambda x: x)
            keras.activations.my_elu = my_utils.create_my_elu()

            # def make_UTMU_loss():
            #     K_UTMU = K.constant(value=numpy.random.random((30,30)))
            #     def UTMU_loss(y_true, y_pred):
            #         u = y_true - y_pred
            #         return K.mean(K.dot(u, K.dot(K_UTMU, K.transpose(u))), axis=-1) # TODO should mean be over an axis?

            #     return UTMU_loss
            # keras.losses.UTMU_loss = make_UTMU_loss()


            model = model_from_json(open(os.path.join(keras_models_dir, model_name + '.json')).read())
            model.load_weights(os.path.join(keras_models_dir, model_name + '.h5'))
            # model = load_model(model_path,  custom_objects={'contractive_loss': lambda x,y: x, 'lam':1e-4})
            k2tf.save_keras_model_as_tf(model, os.path.join(tf_models_dir, model_name + '.pb'))
            K.clear_session()

if __name__ == '__main__':
    import sys
    convert_keras_models_to_tf(sys.argv[1])