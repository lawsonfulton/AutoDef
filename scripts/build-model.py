#!/usr/bin/env python3

# This script is the master script which takes a configuration file, a mesh, 
# (and optionally existing training data) and generates the appropriate models
# for simulation. PCA, and Autoencoder

import json
import os
import shutil
from optparse import OptionParser

from utils import my_utils
from utils import learn
from utils.convert_keras_models_to_tf import convert_keras_models_to_tf
from utils.compute_tf_jacobian_models import generate_jacobian_for_tf_model, generate_vjp, generate_jvp


# TODO
# - Compute vjp during build
# - Run an08 cubature optimization (if enabled), create energy_model directory

def make_output_dir(path):
    if os.path.exists(path):
        overwrite = my_utils.query_yes_no(
            'The model output directory "'+ path + '" already exists. Would you like to overwrite it?'
        )
        if not overwrite:
            exit()
    else:
        os.makedirs(path)


def get_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def copy_files_into_out_dir(config_path, config, config_dir, model_root):
    # -- Mesh
    if 'mesh' in config:
        mesh_in = os.path.join(config_dir, config['mesh'])
    else:
        mesh_in = os.path.join(config_dir, config['training_dataset'], 'tets.mesh')

    mesh_out = os.path.join(model_root, 'tets.mesh')
    shutil.copy(mesh_in, mesh_out)

    # -- Training data TODO: If no training data specified, start generation routine
    training_in = os.path.join(config_dir, config['training_dataset'])
    validation_in = os.path.join(config_dir, config['validation_dataset']) if config['validation_dataset'] else None

    training_out = os.path.join(model_root, 'training_data/training')
    validation_out = os.path.join(model_root, 'training_data/validation')
    training_data_out_root = os.path.join(model_root, 'training_data')

    if os.path.exists(training_data_out_root):
        shutil.rmtree(training_data_out_root)

    shutil.copytree(training_in, training_out)
    if validation_in:
        shutil.copytree(validation_in, validation_out)

    # -- Training config file as a record
    shutil.copy(config_path, os.path.join(model_root, 'model_config.json'))


def main():
    # Parse the command line options
    config_parser = OptionParser()
    config_parser.add_option('-c', '--config_path', dest='config_path',
                  help='Training config file')
    config_parser.add_option('-o', '--out_dir', dest='out_dir',
                  help='Directory where model will be written')

    options, args = config_parser.parse_args()
    config_path = options.config_path
    model_root = options.out_dir

    if not config_path:
        print('A configuration file must be supplied.')
        exit()
    if not model_root:
        print('An output directory must be supplied.')
        exit()

    # Make the output directory and load the config file
    make_output_dir(model_root)
    config = get_config(config_path)
    config_dir = os.path.dirname(config_path) # All paths in the config are relative to the config location

    training_params_path = os.path.join(config_dir, config['training_dataset'], 'parameters.json')
    with open(training_params_path, 'r') as f:
        training_data_params = json.load(f)

    # Copy the needed files into the output dir
    print('Copying files into output directory...')
    copy_files_into_out_dir(config_path, config, config_dir, model_root)

    # Train model
    # TODO Record other model history, seed, etc
    if not config['learning_config']['skip_training']:
        outer_layer_dim = learn.generate_model(model_root, config)

        # Convert to TF
        print('Converting Keras models to Tensorflow...')
        convert_keras_models_to_tf(model_root)
        print('Done')

    # Generate jacobians
    if not config['learning_config']['skip_jacobian']:
        print('Computing jacobian of decoder network...')
        tf_decoder_path = os.path.join(model_root, 'tf_models/decoder.pb') # TODO these path defn's shouldn't be repeated everywhere
        tf_decoder_jac_path = os.path.join(model_root, 'tf_models/decoder_jac.pb')
        tf_vjp_path = os.path.join(model_root, 'tf_models/decoder_vjp.pb')
        tf_jvp_path = os.path.join(model_root, 'tf_models/decoder_jvp.pb')
        generate_jacobian_for_tf_model(tf_decoder_path, tf_decoder_jac_path)
        generate_vjp(tf_decoder_path, tf_vjp_path)
        generate_jvp(tf_decoder_path, tf_jvp_path)
        print('Done.')

    energy_model_config = config['learning_config']['energy_model_config']
    if energy_model_config['enabled']:
        learn.build_energy_model(model_root, energy_model_config)

    # Output simulation config
    print('\nGenerating simulation config file...')
    simulation_config = {
        'mesh': os.path.join(model_root, 'tets.mesh'),
        'logging_enabled': False,
        'save_objs': False,
        "alternative_full_space_mesh": "",
        'material_config': {
            'density': training_data_params['density'], # TODO these numbers should probably match whatever the training data was by default.
            'youngs_modulus': training_data_params['YM'],
            'poissons_ratio': training_data_params['Poisson'],
        },

        'integrator_config': {
            'reduced_space_type': 'autoencoder', # Options are one of ['autoencoder, linear, full']
            'use_reduced_energy': config['learning_config']['energy_model_config']['enabled'],
            'use_partial_decode': True,
            'reduced_energy_method': "pcr", # options: an08, pcr, and not fullyimplemented: pred_weights_l1
            'use_preconditioner': True,
            'pca_dim': config['learning_config']['autoencoder_config']['pca_compare_dims'][0], # Only used if reduced_space_type is linear
            'ae_encoded_dim': config['learning_config']['autoencoder_config']['ae_encoded_dim'], # Shouldn't be change. Kind of a hack.
            'ae_decoded_dim': outer_layer_dim, # Shouldn't be change. Kind of a hack.
            'timestep': training_data_params['time_step'],
            'lbfgs_config': {
                'lbfgs_max_iterations': 1000,
                'lbfgs_epsilon': 1e-8, # Should this be smaller?
                'lbfgs_m': 8,
            },
            'gravity': -9.8,
            'gravity_axis': 1,
            'start_pose_from_training_data': -1,
            'quasi_static': False,
            'save_obj_every_iteration': False
        },

        'visualization_config' : {
            'gpu_decode': True,
            'show_stress': False,
            'show_energy': False,
            'interaction_spring_stiffness': 100,#training_data_params['spring_strength'],
            'full_space_constrained_axis': training_data_params['fixed_axis'],
            'flip_constrained_axis': training_data_params['flip_fixed_axis'],
            'print_every_n_frames': 10,
            'max_frames': 0,
        },
    }
    with open(os.path.join(model_root, 'sim_config.json'), 'w') as f:
        json.dump(simulation_config, f, indent=2)
    print('Done.')
    
if __name__ == '__main__':
    main()
