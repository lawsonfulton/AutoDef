import sys

from utils.my_utils import read_MESH_to_numpy

if __name__ == '__main__':
    mesh_path = sys.argv[1]

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        output_path = mesh_path.replace('.mesh', '.veg')

    print("Reading mesh...")
    V, T, F = read_MESH_to_numpy(mesh_path)

    print("Converting to .veg...")
    with open(output_path, 'w') as fout:
        fout.write('# Vega mesh file.\n')
        fout.write('# %d vertices, %d elements\n\n' % (len(V), len(T)))

        fout.write('*VERTICES\n')
        fout.write('%d  3  0  0\n' % len(V))

        for i, v in enumerate(V):
            fout.write('   %d    %f  %f  %f\n' % (i + 1, v[0], v[1], v[2]))
        fout.write('\n')

        fout.write('*ELEMENTS\n')
        fout.write('TET\n')
        fout.write('%d  4  0\n' % (len(T)))

        for i, t in enumerate(T):
            fout.write('   %d    %d  %d  %d %d\n' % (i + 1, 1 + t[0], 1 + t[1], 1 + t[2], 1 + t[3]))
        fout.write('\n\n')

    print("Wrote", output_path)
