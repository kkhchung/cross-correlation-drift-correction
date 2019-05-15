# cross-correlation-drift-correction
PYME package for cross correlation-based drift correction.

To install, run this from the project folder:

```
python setup.py develop
```

Supports:
- Self correction of localization data
- Self correction of image dataset
- Correction of localization data with raw images


Cross correlation-based methods require the same object to be visible throughout a number of frames. Sparse labelling without fiducials is unlikely to ever work well.

This was designed with large datasets in mind and on a single computer so intermediate results are cached to files on the harddisk. Unlikely to work on the cluster as is.

There are also various hacks that deviate from PYME intended design, e.g. dummy inputs/outputs so that the module will execute in dh5view / visgui.

Another issue in visgui is that this runs slow enough that you would not want to run it every time the recipe pipeline is updated. To turn off auto-execute for all recipe modules, enter in the shell:
```
pipeline.recipe.trait_set(execute_on_invalidation=False)
```
After building the recipe, enter the following to execute it once:
```
pipeline.recipe.execute()
```
## Recipe modules
Currently this package only adds recipe modules. There are no pre-built recipes, nothing is added to the menus.

For now, modules are named so they all appear in the node `chung_cc`.

### To calculate drift:
1. from images
	1. `Image_Pre_Clip&Filter`: Image preprocessing, clip data by intensity and applies Tukey filter to dim edges.
	1. `Image_Pre_Downsample`: Simple down sampling, sums pixels together.
	1. `Image_RCC`: Takes images and calculate drift.

1.  from localisation data
	1.  `Locs_RCC`: Generate images from localisation data and calculates drift.

1.  and save to file
	1.  `Drift_Save`: Saves drift data to numpy file.

### To apply drift correction:
1.  from file
	1.  `Drift_Load`: Load drift data from file.
	1.  `Drift_Interpolate`: Create interpolator from drift data

1.  for images
	1.  `Image_Post_Shift`: Performs FT based subpixel shift of images.

1.  For localisation data
	1.  `Locs_Post_Shift`: Add mapping filter to pipeline for correction.

## To-do list
* Add metadata for the parameters used
* Support for 4D (time and z) image data
* Remove the apply shift part of `Locs_RCC`
* Add more documentation