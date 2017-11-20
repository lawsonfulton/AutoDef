import time

import numpy
import scipy
from sklearn.decomposition import PCA

import pyigl as igl
from iglhelpers import e2p, p2e

import my_utils

current_frame = 0


base_path = 'training_data/first_interaction/'
def main():
    # Loading the rest pose
    base_verts, face_indices = my_utils.load_base_vert_and_face_dmat_to_numpy(base_path)
    base_verts_eig = p2e(base_verts)
    face_indices_eig = p2e(face_indices)

    # Loading displacements for training data
    displacements_data = my_utils.load_displacement_dmats_to_numpy(base_path)


    
    # Set up
    viewer = igl.viewer.Viewer()
    viewer.data.set_mesh(base_verts_eig, face_indices_eig)


    def pre_draw(viewer):
        global current_frame
        
        if viewer.core.is_animating:
            viewer.data.set_vertices(p2e(base_verts + displacements_data[current_frame]))
            # if show_decoded:
            #     viewer.data.set_vertices(test_data_decoded_eigen[current_frame])
            #     viewer.data.set_colors(colours[current_frame])
            # else:
            #     viewer.data.set_vertices(test_data_eigen[current_frame])


            viewer.data.compute_normals()
            current_frame = (current_frame + 1) % len(displacements_data)

        return False

    def key_down(viewer, key, mods):
        global show_decoded
        if key == ord(' '):
            viewer.core.is_animating = not viewer.core.is_animating
        elif key == ord('D') or key == ord('d'):
            show_decoded = not show_decoded
            print(show_decoded)

        return False


    viewer.callback_pre_draw = pre_draw
    viewer.callback_key_down = key_down
    viewer.core.is_animating = False
    viewer.core.invert_normals = True
    viewer.data.face_based = True
    # viewer.core.camera_zoom = 2.5
    viewer.core.animation_max_fps = 30.0

    viewer.launch()

if __name__ == "__main__":
    main()
