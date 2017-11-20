
def load_base_vert_and_face_dmat_to_numpy(base_path):
    """ Returns a tuple (verts, faces) """
    verts_filename = base_path + 'base_verts.dmat'
    faces_filename = base_path + 'base_faces.dmat'

    verts = igl.eigen.MatrixXd()
    faces = igl.eigen.MatrixXi()
    igl.readDMAT(verts_filename, verts)
    igl.readDMAT(faces_filename, faces)

    return e2p(verts), e2p(faces)


def load_displacement_dmats_to_numpy(base_path, num_samples):
    """ Returns a numpy array of displacements for num_samples configurations """
    print('Loading', num_samples, 'samples...', end='', flush=True)
    p = Pool(16)

    def read_dmat(i):
        filename = base_path + 'displacements_%d.dmat' % i
        if(i % 13 == 0):
            print('.', end='', flush=True)
        displacements = igl.eigen.MatrixXd()
        igl.readDMAT(filename, displacements)
        return e2p(displacements)

    displacements_samples = numpy.array(p.map(read_dmat, range(num_samples)))
    p.terminate()
    print()
    print('Done.')
    return displacements_samples
