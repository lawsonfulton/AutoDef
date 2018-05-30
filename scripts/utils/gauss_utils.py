import matlab.engine

def get_mass_matrix(V, T, density):
    youngs = 1e6 # TODO doesn't depend on these right?
    poisson = 0.45

    print("Starting matlab...")
    eng = matlab.engine.start_matlab()

    print("Getting mass matrix")
    fem = eng.WorldFEM('neohookean_linear_tetrahedra', V, T);
    _ = eng.setMeshParameters(fem, youngs, poisson, density);

    return eng.mass(fem)


if __name__ == '__main__':
    print("Starting Matlab...")
    eng = matlab.engine.start_matlab()
    V, T, F  = eng.readMESH('~/Workspace/AutoDef/meshes/X.1.mesh', nargout=3)

    M = get_mass_matrix(V, T, 1.0)

    print(M)