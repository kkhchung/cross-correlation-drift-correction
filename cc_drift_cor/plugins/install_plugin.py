from PYME import config
import os
import sys
from distutils.dir_util import copy_tree
import pkgutil
from cc_drift_cor.plugins import recipes

def main():
#    this_dir = os.path.dirname(__file__)

    if len(sys.argv) > 1 and sys.argv[1] == 'dist':
#        copy_tree(os.path.join(this_dir, '_etc', 'PYME'), config.dist_config_directory)
        target_dir = config.dist_config_directory
    else:  # no argument provided or is not 'dist', default to user config directory
#        copy_tree(os.path.join(this_dir, '_etc', 'PYME'), config.user_config_dir)
        target_dir = config.user_config_dir
    
    target_filename =  recipes.__name__.split(".")[0] + ".txt"
    target_path = os.path.join(target_dir, "plugins", "recipes", target_filename)
    print("writing addons to file: {}".format(target_path))
    with open(target_path, 'w') as f:
        f.writelines(create_module_list())            
        
def create_module_list():
    modules = list()    
    for _, name, _ in pkgutil.iter_modules([os.path.dirname(recipes.__file__)]):
        modules.append(".".join([recipes.__name__, name])+'\n')
    print(modules)
    return modules

if __name__ == '__main__':
#    main()
    create_module_list()