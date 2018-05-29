import matlab.engine

def get_mass_matrix(V, T, density):
    eng = matlab.engine.start_matlab()

    fem = eng.WorldFEM('neohookean_linear_tetrahedra', V, T);