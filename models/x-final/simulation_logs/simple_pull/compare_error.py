import os

import numpy
from sklearn.metrics import mean_squared_error

import pyigl as igl
from iglhelpers import e2p, p2e

def load_obj_verts(dir):
    Vs = []
    filenames = sorted(os.listdir(dir))

    for filename in filenames:
        path = os.path.join(dir, filename)
        V = igl.eigen.MatrixXd()
        F = igl.eigen.MatrixXi()
        igl.readOBJ(path, V, F)

        Vs.append(e2p(V))

    return numpy.array(Vs)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true_2 = y_true + 0.0001
    return numpy.mean(numpy.abs((y_true_2 - y_pred) / y_true_2)) * 100

def main():
    full_space_path = './full-space/objs/surface/'
    ae_pcr_path = './pred_l1_90/objs/surface/'
    ae_an08_path = './an08_30/objs/surface/'

    print('Loading objs...')
    full_space_Vs = load_obj_verts(full_space_path)
    ae_pcr_Vs = load_obj_verts(ae_pcr_path)
    ae_an08_Vs = load_obj_verts(ae_an08_path)
    v0 = full_space_Vs[1]


    print("PCR")
    print(max(
        mean_squared_error(full_space_V.flatten(),  ae_pcr_V.flatten()) for 
        full_space_V, ae_pcr_V in zip(full_space_Vs, ae_pcr_Vs)))
    print(
        mean_squared_error(full_space_Vs.flatten(),  ae_pcr_Vs.flatten()))

    print("\nAN08")
    print(max(
        mean_squared_error(full_space_V.flatten(),  ae_an08_V.flatten()) for 
        full_space_V, ae_an08_V in zip(full_space_Vs, ae_an08_Vs)))
    print(
        mean_squared_error(full_space_Vs.flatten(),  ae_an08_Vs.flatten()))

if __name__ == '__main__':
    main()
