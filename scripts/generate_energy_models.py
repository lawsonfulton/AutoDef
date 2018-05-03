import sys
import subprocess

from utils import learn
from utils.convert_keras_models_to_tf import convert_keras_models_to_tf
from augment_training_data import reencode_and_augment_training_data

def build_energy_model(model_root, energy_type, n_tets=None):
    do_augmentation = True
    if do_augmentation:
        reencode_and_augment_training_data(model_root, 3)

    if energy_type == 'pred_weights_l1':
        print("Building energy model pred_weights_l1...")
        learn.sparse_learn_weights_l1(model_root, use_reencoded=True, use_extra_samples=True)
        convert_keras_models_to_tf(model_root)
    elif energy_type == 'pred_energy_direct':
        print("Building energy model for direct prediction...")
        learn.learn_direct_energy(model_root, use_reencoded=True, use_extra_samples=do_augmentation)
        convert_keras_models_to_tf(model_root)
    elif energy_type == 'an08':
        print("Building energy model an08...")
        subprocess.call(['/home/lawson/Workspace/cubacode/build/bin/Cubacode', model_root, n_tets])


if __name__ == '__main__':
    model_root = sys.argv[1]
    energy_type = sys.argv[2]

    n_tets = None
    if len(sys.argv) == 4:
        n_tets = sys.argv[3]

    build_energy_model(model_root, energy_type, n_tets)

    