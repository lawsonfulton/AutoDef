import sys
import subprocess
import os

def main():
    ply_file = sys.argv[1]
    
    out_dir, basename = os.path.split(ply_file)
    basename, extension = basename.split(".")

    name_to_params = {
    "very_low": (10, 200), # edge length, epsilon
    "low": (60, 250),
    "med": (120, 500),
    "high": (250, 1000),
    "very_high": (500, 2000)
    }
    
    processes = []
    for name, params in name_to_params.items():
        el = params[0]
        eps = params[1]

        filename = basename + "_" + name + ".mesh"
        out_path = os.path.join(out_dir, filename)

        proc = subprocess.Popen([
            "/home/lawson/Workspace/TetWild/build/TetWild", ply_file,
            '--output', out_path,
            '--ideal-edge-length', str(el),
            '--epsilon', str(eps)
            ]
        )

        processes.append(proc)

    for proc in processes:
        proc.wait()


if __name__ == '__main__':
    main()
