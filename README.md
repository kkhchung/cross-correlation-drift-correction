# cross-correlation-drift-correction
Cross correlation-based drift correction recipe module written for [PYME](https://python-microscopy.org/).

Supports 2/3D localization or 2D image data.


## System requirements
* Windows, Linux or OS X
* Python 2.7, 3.6, or 3.7
* PYME (>18.7.18) and dependencies

- Tested on Windows 10 with PYME (18.7.18)


## Installation

1. Clone repository.
2. Run the following in the project folder. 
	```
		python setup.py develop
	```
3. Start `dh5view` or `VisGUI` (PYME).

(Runtime < 5 min)


## Demo

### To correct drift in localization data and save the measured drift:
1. Open this simulated dataset ([wormlike_simulated_locs_with_drift.hdf](/cc_drift_cor/example/wormlike_simulated_locs_with_drift.hdf)) with `VisGUI` (PYME).
2. Load and run this demo recipe ([correct_drift_locs.yaml](/cc_drift_cor/example/correct_drift_locs.yaml)).
3. The measured drift will display in a new window.
4. Select the new output (`corrected_localizations`) in the Data Pipeline pane to see the drift-corrected data.
5. The drift data will be saved as `drift.npz`

(Runtime < 5 min)

### To correct drift in localization data from previously saved drift data:
1. Open this simulated dataset ([wormlike_simulated_locs_with_drift.hdf](/cc_drift_cor/example/wormlike_simulated_locs_with_drift.hdf)) with `VisGUI` (PYME).
2. Load and run this demo recipe ([correct_drift_locs_load.yaml](/cc_drift_cor/example/correct_drift_locs_load.yaml)).
3. The drift data will be loaded from [drift.npz](/cc_drift_cor/example/drift.npz)
4. The loaded drift will display in a new window.
5. Select the new output (`corrected_localizations`) in the Data Pipeline pane to see the drift-corrected data.

(Runtime < 5 min)

### To correct drift in image data:
1. Open this simulated image ([wormlike_simulated_images_with_drift.h5](/cc_drift_cor/example/wormlike_simulated_images_with_drift.h5)) with `dh5view` (PYME).
2. Load and run this demo recipe ([correct_drift_images.yaml](/cc_drift_cor/example/correct_drift_images.yaml)).
3. The measured drift will display in a new window.
4. Open the drift-corrected images by clicking the output (`drift_corrected_image`) on the final **Image_Post_Shift** module.

(Runtime < 5 min)


## Instructions

### For localization data:
1. Refer to [PYME documentation](https://python-microscopy.org/doc/index.html) for general use of PYME.
2. Detailed description of each module and their inputs, outputs and parameters are accessible in PYME.
3. Open localization file with `VisGUI` (PYME).
4. To correct drift, chain the modules in this order:
	1. **Locs_RCC**
	2. **Drift_Interpolate**
	3. **Locs_Post_Shift**
	
5. Alternatively, to save/load drift:
	To save:
	1. **Locs_RCC**
	2. **Drift_Save**
	
	To load:
	1. **Drift_Load_Interpolate**
	2. **Locs_Post_Shift**
6. Recipes in `VisGUI' auto-run when modules are added. Since drift correction can be slow, it may be desirable to suspend this behaviour by entering this in the Shell tab.
	```
	pipeline.recipe.trait_set(execute_on_invalidation=False)
	```
	Then after building the complete recipe, enter the following to execute it once:
	```
	pipeline.recipe.execute()
	```
	This behavior may have changed in more recent PYME versions.
	
### For image data:
1. Refer to [PYME documentation](https://python-microscopy.org/doc/index.html) for general use of PYME.
2. Detailed description of each module and their inputs, outputs and parameters are accessible in PYME.
3. Open image file with `dh5view` (PYME).
4. *(Optional)* Standard image processing methods to clean up the images may be required depending on their quality.
	1. **Image_Pre_Clip&Filter**
	2. **Image_Pre_Downsample**
5. To correct drift, chain the modules in this order:
	1. **Image_RCC**
	2. **Drift_Interpolate**
	3. **Image_Post_Shift**
6. Alternatively, to save/load drift:
	To save:
	1. **Image_RCC**
	2. **Drift_Save**
	
	To load:
	1. **Drift_Load_Interpolate**
	2. **Image_Post_Shift**


## Notes

* Cross correlation-based methods require the same object to be visible throughout a number of frames. Sparse labelling without fiducials is unlikely to ever work well.

* This was designed with large datasets in mind and on a single computer so intermediate results are cached to files on the hard disk. Cached files may need to be removed manually for some of the modules or when errors occur.

* Runtime can vary hugely depending on the size of the dataset, 2D/3D, pixel size, cross-correlation window size, etc. It is probably worth adjusting the settings if runtime is over 30 mins. Longer runtime may not coincide with better drift correction.


## To do's
* Add more metadata
* Support for 4D (time and z) image data
