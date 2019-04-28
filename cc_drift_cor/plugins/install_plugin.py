from PYME import config
import os
import sys
from distutils.dir_util import copy_tree

def main():
    this_dir = os.path.dirname(__file__)

    if len(sys.argv) > 1 and sys.argv[1] == 'dist':
        copy_tree(os.path.join(this_dir, '_etc', 'PYME'), config.dist_config_directory)
    else:  # no argument provided or is not 'dist', default to user config directory
        copy_tree(os.path.join(this_dir, '_etc', 'PYME'), config.user_config_dir)

if __name__ == '__main__':
    main()