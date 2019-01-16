#!/usr/bin/env python3

# This script is the master script which takes a configuration file, a mesh, 
# (and optionally existing training data) and generates the appropriate models
# for simulation. PCA, and Autoencoder

import json
import os
import shutil
import subprocess
from optparse import OptionParser

from utils import my_utils
from utils import learn
from utils.convert_keras_models_to_tf import convert_keras_models_to_tf
from utils.compute_tf_jacobian_models import generate_jacobian_for_tf_model, generate_vjp, generate_jvp


def get_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def copy_files_into_out_dir(config, model_root):
    # -- Mesh
    mesh_in = os.path.join(config['training_dataset'], 'tets.mesh')

    mesh_out = os.path.join(model_root, 'tets.mesh')
    shutil.copy(mesh_in, mesh_out)

    # -- Training data TODO: If no training data specified, start generation routine
    # training_in = os.path.join(config_dir, config['training_dataset'])
    # validation_in = os.path.join(config_dir, config['validation_dataset']) if config['validation_dataset'] else None

    # training_out = os.path.join(model_root, 'training_data/training')
    # validation_out = os.path.join(model_root, 'training_data/validation')
    # training_data_out_root = os.path.join(model_root, 'training_data')

    # if os.path.exists(training_data_out_root):
    #     shutil.rmtree(training_data_out_root)

    # shutil.copytree(training_in, training_out)
    # if validation_in:
    #     shutil.copytree(validation_in, validation_out)

    # -- Training config file as a record

    with open(os.path.join(model_root, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def main():
    # Parse the command line options
    config_parser = OptionParser()
    config_parser.add_option('-c', '--config_path', dest='config_path',
                  help='Training config file')
    config_parser.add_option('-o', '--out_dir', dest='out_dir',
                  help='Directory where model will be written')
    config_parser.add_option('-f', '--force', dest='force',
                  help='Automatically overwrite', default=False)

    options, args = config_parser.parse_args()
    config_path = options.config_path
    model_root = options.out_dir
    force = options.force

    if not config_path:
        print('A configuration file must be supplied.')
        exit()
    if not model_root:
        print('An output directory must be supplied.')
        exit()

    config = get_config(config_path)
    build_model(config, model_root, force)

# Everything is relative to AutoDef/ root
def build_model(config, model_root, force=False, mem={}, components=None, files=None, do_pca_compare = True):
    # Make the output directory and load the config file
    my_utils.make_dir_with_confirmation(model_root, force)

    training_params_path = os.path.join(config['training_dataset'], 'parameters.json')
    with open(training_params_path, 'r') as f:
        training_data_params = json.load(f)

    # Copy the needed files into the output dir
    print('Copying files into output directory...')
    copy_files_into_out_dir(config, model_root)

    # Train model
    # TODO Record other model history, seed, etc
    if not config['learning_config']['skip_training']:
        outer_layer_dim, best_match_pca_dim, ae_encoded_dim, training_results, mem, components, files = learn.generate_model(model_root, config, mem, components, files, do_pca_compare)

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

    # Output simulation config
    print('\nGenerating simulation config file...')
    simulation_config = {
        'mesh': os.path.join(model_root, 'tets.mesh'),
        'logging_enabled': False,
        'save_objs': False,
        'save_pngs': False,
        'save_training_data': False,
        'save_training_data_path': '', # If this path is filled in, then displacements, mouse, params, will be recorded.
        'alternative_full_space_mesh': '',
        'material_config': {
            'density': training_data_params['density'], # TODO these numbers should probably match whatever the training data was by default.
            'youngs_modulus': training_data_params['YM'],
            'poissons_ratio': training_data_params['Poisson'],
        },

        'integrator_config': {
            'reduced_space_type': 'autoencoder', # Options are one of ['autoencoder, linear, full']
            'use_reduced_energy': config['learning_config']['energy_model_config']['enabled'],
            'use_partial_decode': True,
            'reduced_energy_method': config['learning_config']['energy_model_config'].get('type', 'full'), # options: an08, pcr, and not fullyimplemented: pred_weights_l1
            'use_preconditioner': True,
            'pca_dim': best_match_pca_dim, # Only used if reduced_space_type is linear
            'ae_encoded_dim': ae_encoded_dim, # Shouldn't be change. Kind of a hack.
            'ae_decoded_dim': outer_layer_dim, # Shouldn't be change. Kind of a hack.
            'timestep': training_data_params['time_step'],
            'lbfgs_config': {
                'lbfgs_max_iterations': 1000,
                'lbfgs_epsilon': 1e-8, # Should this be smaller?
                'lbfgs_delta': 1e-8,
                'lbfgs_delta_past': 0,
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
            'show_lines': False,
            'interaction_spring_stiffness': training_data_params.get('spring_strength', 100),
            'spring_grab_radius': training_data_params.get('spring_grab_radius', 0.03), # Note this is ignored for reduced spaces
            'use_spring_grab_radius_for_reduced': False,
            'full_space_constrained_axis': training_data_params['fixed_axis'],
            'constrained_axis_eps': training_data_params['constrained_axis_eps'],
            'flip_constrained_axis': training_data_params['flip_fixed_axis'],
            'fixed_point_constraint': training_data_params.get('fixed_point_constraint', [0,0,0]),
            'fixed_point_radius': training_data_params.get('fixed_point_radius', -1),
            'print_every_n_frames': 10,
            'max_frames': 0,
        },
    }
    with open(os.path.join(model_root, 'sim_config.json'), 'w') as f:
        json.dump(simulation_config, f, indent=2)

    energy_model_config = config['learning_config']['energy_model_config']
    if energy_model_config['enabled']:
        energy_type = energy_model_config.get('type', 'an08')
        num_sample_tets = energy_model_config['num_sample_tets']

        if(energy_type == 'pcr'):
            learn.build_energy_model(model_root, config)
        elif(energy_type == 'an08'):
            for dim in [outer_layer_dim, best_match_pca_dim, ae_encoded_dim]:
                print(dim)
                subprocess.run(['cubacode/build/bin/Cubacode', model_root, str(num_sample_tets), str(dim)])
        else:
            raise("Energy type doesn't exist")

    print('Done.')
    return training_results, mem,components, files
    
if __name__ == '__main__':
    main()
