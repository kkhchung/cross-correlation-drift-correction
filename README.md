# cross-correlation-drift-correction
To install, run this from the project folder:
'python setup.py develop
PYME module for cross correlation-based drift correction with support for both image and localization data.
PYME package for cross correlation-based drift correction.
Supports:
- Self correction of localization data
- Self correction of image dataset
- Correction of localization data with raw images
Cross correlation-based methods require the same object to be visible throughout a number of frames. Sparse labelling without fiducials is unlikely to ever work well.
This was designed with large datasets in mind and on a single computer so intermediate results and cached to files on the harddisk. Unlikely to work on the cluster.
There are also various hacks that deviate from PYME intended design, e.g. dummy inputs/outputs so that the module will execute in dh5view / visgui.
Another major issue is that this runs slow enough that you would not want to run it every time the recipe pipeline is updated. To turn off auto-execute, enter in the shell:
`pipeline.recipe.trait_set(execute_on_invalidation=False)
After building the recipe, enter the following to execute it once:
'pipeline.recipe.execute()
## Recipe modules
Currently this package only adds recipe modules. There are no pre-built recipes, nothing is added to the menus.
###
Recipe module for localization inputs. Responsible for generation of histogram, filtering.
###
Recipe module for image inputs. Responsible for downsampling, filtering.
###
Backend for ??? and ???. Don't need to use this directly unless for debugging.
### SaveDrift
### LoadDrift
### InterpolateDrift
### ApplyDriftCorrection
### ShiftImage
#Todo's
- [] Add metadata for the parameters used
- [] Support for 4D (time and z) image data
- [] Remove the apply shift part of locs rcc
- [] Use reST syntax