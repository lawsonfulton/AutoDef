import os
import re
import itertools
from multiprocessing import Pool

import numpy

import pyigl as igl
from iglhelpers import e2p, p2e

def load_base_vert_and_face_dmat_to_numpy(base_path):
    """ Returns a tuple (verts, faces) """
    verts_filename = base_path + 'base_verts.dmat'
    faces_filename = base_path + 'base_faces.dmat'

    verts = igl.eigen.MatrixXd()
    faces = igl.eigen.MatrixXi()
    igl.readDMAT(verts_filename, verts)
    igl.readDMAT(faces_filename, faces)

    return e2p(verts), e2p(faces)

def _read_dmat_helper(args):
    i, base_path = args
    filename = base_path + 'displacements_%d.dmat' % i
    if(i % 13 == 0):
        print('.', end='', flush=True)
    displacements = igl.eigen.MatrixXd()
    igl.readDMAT(filename, displacements)
    return e2p(displacements)

def load_displacement_dmats_to_numpy(base_path, num_samples=None):
    """ Returns a numpy array of displacements for num_samples configurations """

    if num_samples is None:  # Compute number of configurations to load automatically
        filenames = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        numbers_strings = [re.findall('\d+', f) for f in filenames]
        num_samples = max(int(n_str[0]) for n_str in numbers_strings if n_str)

    print('Loading', num_samples, 'samples...', end='', flush=True)
    p = Pool(16)

    displacements_samples = numpy.array(p.map(_read_dmat_helper, zip(range(num_samples), itertools.repeat(base_path))))
    p.terminate()
    print()
    print('Done.')
    return displacements_samples
