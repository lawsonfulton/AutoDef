import os
import sys
from collections import defaultdict

def main():
    target_dir = sys.argv[1]

    paths = sorted(os.listdir(target_dir))

    paths_by_it = defaultdict(list)

    for p in paths:
        it, rest = p.split('_')
        # sub_it, _ = p.split('.')
        paths_by_it[int(it)].append(p)
    
    paths_to_remove = []
    for it, paths in paths_by_it.items():
        for p in paths[:-1]:
            paths_to_remove.append(p)

    for p in paths_to_remove:
        os.remove(os.path.join(target_dir, p))
        print("Removed", p)



if __name__ == "__main__":
    main()
