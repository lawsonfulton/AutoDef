import matlab.engine
from scipy.sparse import csc_matrix
import numpy
import time

from utils.my_utils import save_numpy_mat_to_dmat, read_double_dmat_to_numpy

def create_or_connect_to_matlab_engine(eng=None):
    if eng is None:
        names = matlab.engine.find_matlab()
        if len(names) > 0:
            print("Found active MATLAB session: ", names[0])
            eng = matlab.engine.connect_matlab(names[0])
        else:
            print("Starting matlab...")
            eng = matlab.engine.start_matlab()
            eng.cd('../')
            # shared_name = eng.matlab.engine.shareEngine('MATLAB_Engine%d' % int(time.time()), nargout=0)
    
    print("MATLAB started in directory:", eng.pwd())

    return eng

def get_mass_matrix(mesh_path, density, eng=None):
    youngs = 1e6 # TODO doesn't depend on these right?
    poisson = 0.45

    eng = create_or_connect_to_matlab_engine(eng)

    print("Setting up mesh")

    eng.evalc(
        "[V, T, F]  = readMESH('%s');\
         fem = WorldFEM('neohookean_linear_tetrahedra', V, T);\
         setMeshParameters(fem, 1e6, 0.45, %d);"
         % (mesh_path, density)
    )

    print("Getting mass matrix")
    row_d, col_d, data_d = eng.eval("find(mass(fem))", nargout=3)
    rows = numpy.array([int(j[0]) - 1 for j in row_d])
    cols = numpy.array([int(j[0]) - 1 for j in col_d]) # Need to decrease index by one for python
    data = numpy.array([j[0] for j in data_d])

    return csc_matrix((data, (rows, cols)))


def mass_pca(mesh_path, density, samples, pca_dim, eng=None):    
    # Python version that didn't work. where does it differ from above?
    # >>> LLT = cholesky(M)
    # >>> L = LLT.L()
    # >>> svd_samples = spsolve(L, displacements.T) # or replace with LLT.solve_L(displacements.T, use_LDLt_decomposition=False)
    # >>> LU, C, _ = linalg.svd(svd_samples, full_matrices=False)
    # >>> U = spsolve(L, LU[:,:30])
    # >>> numpy.max(numpy.abs(U.T @ M @ U - numpy.eye(30)))

    eng = create_or_connect_to_matlab_engine(eng) 

    print("Sending samples to MATLAB...")
    samples_path = '/tmp/samples.dmat'
    basis_path = '/tmp/U.dmat'

    save_numpy_mat_to_dmat(samples_path, samples)

    print("Setting up mesh and sending sample to MATLAB...")
    eng.evalc(
        "samples = readDMAT('%s');\
         [V, T, F] = readMESH('%s');\
         fem = WorldFEM('neohookean_linear_tetrahedra', V, T);\
         setMeshParameters(fem, 1e6, 0.45, %d);\
         M = mass(fem);" % (samples_path, mesh_path, density)
    )

    print("Doing Mass PCA...")
    start = time.time()
    eng.evalc(
        "L = chol(M);\
         [LU,C,~] = svd(L*(samples'),'econ');\
         U = L\\LU(:,1:%d);" % pca_dim
    )
    duration = time.time() - start

    print("Took: %ds" % duration)
    eng.evalc(
        "writeDMAT('%s', U, false);" % basis_path
    )

    error = eng.eval("max(max(abs(U'*M*U - eye(%d))))" % pca_dim)
    if error > 1e-5:
        print("Mass PCA did not give U^TMU=I with max error:", error)
        exit()

    U = read_double_dmat_to_numpy(basis_path)

    return U

if __name__ == '__main__':
    M = get_mass_matrix("~/Workspace/AutoDef/meshes/X.1.mesh", 1.0)

    print(M[0])