import time
from multiprocessing import Pool

import numpy
import scipy
from sklearn.decomposition import PCA

import pyigl as igl
from iglhelpers import e2p, p2e

import my_utils

base_path = 'training_data/first_interaction/'
def main():
	displacements_data = my_utils.load_displacement_dmats_to_numpy(base_path)
	print(displacements_data)


if __name__ == "__main__":
	main()
