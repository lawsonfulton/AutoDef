import sys
import os
import subprocess

import numpy

import pyigl as igl
from utils.iglhelpers import e2p, p2e
from utils import my_utils

current_frame = 0

def sample_more_encoded_displacements(encoded_displacements, num_extra_per_pose=10):
    max_diff = numpy.zeros(encoded_displacements.shape[1])

    for i in range(len(encoded_displacements) - 1):
        abs_diff = numpy.abs(encoded_displacements[i] - encoded_displacements[i+1])
        max_diff = numpy.max(numpy.stack((abs_diff, max_diff)), 0)

    mu = 0
    sigma = max(max_diff / 6.0) # Just take max scaled by a factor for now

    extra_encoded_displacements = numpy.repeat(encoded_displacements, num_extra_per_pose, axis=0)
    extra_encoded_displacements += numpy.random.normal(mu, sigma, extra_encoded_displacements.shape)

    # print(extra_encoded_displacements[:len(encoded_displacements)] - encoded_displacements)
    return extra_encoded_displacements

def save_mat_with_prefix(path, prefix, mat):    
    dmat_path = os.path.join(path, '%s.dmat' % (prefix))
    my_utils.save_numpy_mat_to_dmat(dmat_path, mat)
    return dmat_path

def reencode_and_augment_training_data(model_root, num_extra_per_poses=0):
    """ 
    Loads existing traing data and generates new encoding / energy vector pairs for 
    1. Energy evalutated on decoded displacements of training data.
    2. Energy evaluated on poses sampled around the encoded training poses.
    """
    training_data_path = os.path.join(model_root,'training_data/training')
    U = igl.eigen.MatrixXd()
    igl.readDMAT(os.path.join(model_root, 'pca_results/ae_pca_components.dmat'), U)

    displacements = my_utils.load_displacement_dmats_to_numpy(training_data_path)
    flatten_displ, unflatten_displ = my_utils.get_flattners(displacements)

    from keras.models import Model, load_model
    encoder = load_model(os.path.join(model_root,'keras_models/encoder.hdf5'))
    decoder = load_model(os.path.join(model_root,'keras_models/decoder.hdf5'))

    encoded_displacements = encoder.predict(flatten_displ(displacements) @ U)
    decoded_displacements = decoder.predict(encoded_displacements) @ U.transpose()

    print('Generating extra samples...')
    extra_encoded_displacements = sample_more_encoded_displacements(encoded_displacements, num_extra_per_poses)
    extra_decoded_displacements = decoder.predict(extra_encoded_displacements) @ U.transpose()
    
    sampled_training_data_path = os.path.join(model_root, 'augmented_training_data/sampled/')
    reencoded_training_data_path = os.path.join(model_root, 'augmented_training_data/reencoded/')
    my_utils.create_dir_if_not_exist(sampled_training_data_path)
    my_utils.create_dir_if_not_exist(reencoded_training_data_path)

    extra_displacements_path = save_mat_with_prefix(sampled_training_data_path, 'displacements', extra_decoded_displacements)
    save_mat_with_prefix(sampled_training_data_path, 'enc_displacements', extra_encoded_displacements)
    reencoded_displacements_path = save_mat_with_prefix(reencoded_training_data_path, 'displacements', decoded_displacements)
    save_mat_with_prefix(reencoded_training_data_path, 'enc_displacements', encoded_displacements)

    tet_mesh_path = os.path.join(model_root, 'tets.mesh')
    parameters_path = os.path.join(model_root, 'training_data/training/parameters.json')

    print('Computing energies for reencoded poses...')
    subprocess.call(['./generate_data_for_pose/build/bin/GenerateDataForPose', reencoded_displacements_path, tet_mesh_path, parameters_path])

    print('Computing energies for samples...')
    subprocess.call(['./generate_data_for_pose/build/bin/GenerateDataForPose', extra_displacements_path, tet_mesh_path, parameters_path])

    ## TODO 
    # Save them all as one matrix.. It's more efficient that way
    # Now I just have to get the energies and I can plop this into my pipeline

    # scale = numpy.max(energies)
    # print(numpy.apply_along_axis(numpy.linalg.norm, 1, energies).shape)
    # print("scale: ", scale)
    # # energies = energies.reshape((len(energies), len(energies[0]))) / scale
    # # energies_test = energies_test.reshape((len(energies_test), len(energies_test[0]))) / scale
    # energies = energies / numpy.apply_along_axis(numpy.linalg.norm, 1, energies)[:, None]
    # # energies_test = energies_test / scale



    # # print(energies)
    # # energies = numpy.sum(energies,axis=1)
    # # print(energies)
    # # energies_test = numpy.sum(energies_test,axis=1)
    # flatten_data, unflatten_data = my_utils.get_flattners(energies)

    # Set up drawings
    # np_verts, np_faces = my_utils.load_base_vert_and_face_dmat_to_numpy(training_data_path)
    # viewer = igl.viewer.Viewer()

    # viewer.data.set_mesh(p2e(np_verts), p2e(np_faces))

    # def pre_draw(viewer):
    #     global current_frame
        
    #     if viewer.core.is_animating:
    #         idx = current_frame % len(extra_decoded_displacements)
            
    #         verts = extra_decoded_displacements[current_frame].reshape(np_verts.shape) + np_verts
    #         viewer.data.set_vertices(p2e(verts))
    #         viewer.data.compute_normals()

    #         current_frame += 1

    #     return False

    # viewer.callback_pre_draw = pre_draw
    # # viewer.callback_key_down = key_down
    # viewer.core.is_animating = False
    # # viewer.core.camera_zoom = 2.5
    # viewer.core.animation_max_fps = 3

    # viewer.launch()

    

if __name__ == '__main__':
    model_root = sys.argv[1]
    augment_training_data(model_root)
