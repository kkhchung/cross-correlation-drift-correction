# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 14:15:49 2019

@author: kkc29
"""

from PYME.recipes.base import register_module, ModuleBase, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr, File
from PYME.recipes.graphing import Plot

import numpy as np
from PYME.IO import tabular
from PYME.LMVis import renderers
from scipy import ndimage, signal, interpolate

from functools import partial
from .io import generate_drift_plot

import multiprocessing
import time

def calc_fft_from_locs_helper(args):
    """
        Wrapper
    """
    return (args[0], calc_fft_from_locs(*args[1:]))

# trying to avoid redunant code
from .processing import calc_fft_from_image
def calc_fft_from_locs(xyz, bxyz, cache_fft=None, filter_size=None):
    # module level for multiprocessing
    """
        Creates histogram and applies Tukey filter.
        Results passed to calc_fft_from_image in the base class
    """
    im = np.histogramdd(xyz, bxyz)[0]
    
    if not filter_size is None:
        mask_shape = np.ones(len(im.shape), dtype=np.int)
        for i, d in enumerate(im.shape):
            if d <= 1:
                continue
            mask_shape[:] = 1
            mask_shape[i] = d    
            mask = np.empty(mask_shape)
            mask.squeeze()[:] = signal.tukey(d, filter_size)
            im *= mask
        
    del xyz, bxyz
    
    return calc_fft_from_image(im, cache_fft)
    
from .processing import RCCDriftCorrectionBase
#@register_module('RCCDriftCorrection')
class RCCDriftCorrection(RCCDriftCorrectionBase):
    """
    For localization data.
    
    Performs drift correction using cross-correlation, including redundant RCC from
    Wang et al. Optics Express 2014 22:13 (Bo Huang's RCC algorithm).
    
    Runtime will vary hugely depending on size of dataset and settings.
    
    ``cache_fft`` is necessary for large datasets.
        
    Inputs
    ------
    input_for_correction : Tabular
        Dataset used to calculate drift.
    input_for_mapping : Tabular
        *Deprecated.*   Dataset to correct.
    
    Outputs
    -------
    output_drift : Tuple of arrays
        Drift results.
    output_drift_plot : Plot
        *Deprecated.*   Plot of drift results.
    output_cross_cor : ImageStack
        Cross correlation images if ``debug_cor_file`` is not blank.
    outputName : Tabular
        *Deprecated.*   Drift-corrected dataset.
    
    Parameters
    ----------
    step : Int
        Setting for image construction. Step size between images
    window : Int
        Setting for image construction. Number of frames used per image. Should be equal or larger than step size.
    binsize : Float
        Setting for image construction. Pixel size.
    flatten_z : Bool
        Setting for image construction. Ignore z information if enabled.
    tukey_size : Float
        Setting for image construction. Shape parameter for Tukey filter (``scipy.signal.tukey``).
    cache_fft : File
        Use file as disk cache if provided.
    method : String
        Redundant, mean, or direct cross-correlation.
    shift_max : Float
        Rejection threshold for RCC.
    corr_window : Float
        Size of correlation window. Frames are only compared if within this frame range. N/A for DCC.
    multiprocessing : Float
        Enables multiprocessing.
    debug_cor_file : File
        Enables debugging. Use file as disk cache if provided.
    """
    
    input_for_correction = Input('Localizations')
    input_for_mapping = Input('Localizations')
    # redundant cross-corelation, mean cross-correlation, direction cross-correlation
    step = Int(2500)
    window = Int(2500)
    binsize = Float(30)
    flatten_z = Bool()
    tukey_size = Float(0.25)

    outputName = Output('corrected_localizations')
    
    def calc_corr_drift_from_locs(self, x, y, z, t):

        # bin edges for histogram
        bx = np.arange(x.min(), x.max() + self.binsize + 1, self.binsize)
        by = np.arange(y.min(), y.max() + self.binsize + 1, self.binsize)
        bz = np.arange(z.min(), z.max() + self.binsize + 1, self.binsize)
        
        # pad bin length to odd number so image size is even
        if bx.shape[0] % 2 == 0:
            bx = np.concatenate([bx, [bx[-1] + bx[1] - bx[0]]])
        if by.shape[0] % 2 == 0:
            by = np.concatenate([by, [by[-1] + by[1] - by[0]]])
        if bz.shape[0] > 2 and bz.shape[0] % 2 == 0:
            bz = np.concatenate([bz, [bz[-1] + bz[1] - bz[0]]])
        assert (bx.shape[0] % 2 == 1) and (by.shape[0] % 2 == 1), "Ops. Image not correctly padded to even size."

        # start time of all windows, allow partial window near end of pipeline
        time_values = np.arange(t.min(), t.max() + 1, self.step)
        # 2d array, start and end time of windows
        time_values = np.stack([time_values, np.clip(time_values + self.window, None, t.max())], axis=1)        
        n_steps = time_values.shape[0]
        # center time of center for returning. last window may have different spacing
        time_values_mid = time_values.mean(axis=1)

        if (np.any(np.diff(t) < 0)): # in case pipeline is not sorted for whatever reason
            t_sort_arg = np.argsort(t)
            t = t[t_sort_arg]
            x = x[t_sort_arg]
            y = y[t_sort_arg]
            z = z[t_sort_arg]
            
        time_indexes = np.zeros_like(time_values, dtype=int)
        time_indexes[:, 0] = np.searchsorted(t, time_values[:, 0], side='left')
        time_indexes[:, 1] = np.searchsorted(t, time_values[:, 1]-1, side='right')
        
#        print('time indexes')
#        print(time_values)
#        print(time_values_mid)
#        print(time_indexes)

        # Fourier transformed (and binned) set of images to correlate against
        # one another
        # Crude way of swaping longest axis to the last for optimizing rfft performance.
        # Code changed for this is limited to this method.
        xyz = np.asarray([x, y, z])
        bxyz = np.asarray([bx, by, bz])
        dims_order = np.arange(len(xyz))
        dims_length = np.asarray([len(b) for b in bxyz])
        dims_largest_index = np.argmax(dims_length)
        dims_order[-1], dims_order[dims_largest_index] = dims_order[dims_largest_index], dims_order[-1]
        xyz = xyz[dims_order]
        bxyz = bxyz[dims_order]
        dims_length = dims_length[dims_order]
        
        # use memmap for caching if ft_cache is defined
        if self.cache_fft == "":
            ft_images = np.zeros((n_steps, dims_length[0]-1, dims_length[1]-1, (dims_length[2]-1)//2 + 1, ), dtype=np.complex)
        else:
            ft_images = np.memmap(self.cache_fft, dtype=np.complex, mode='w+', shape=(n_steps, dims_length[0]-1, dims_length[1]-1, (dims_length[2]-1)//2 + 1, ))
        
        print(ft_images.shape)
        print("{:,} bytes".format(ft_images.nbytes))
        
        print("{:.2f} s. About to start heavy lifting.".format(time.time() - self._start_time))
        
        # fill ft_images
        # if multiprocessing, can either use or not caching
        # if not multiprocessing, don't pass filenames for caching, just the memmap array is fine
        if self.multiprocessing:
            dt = ft_images.dtype
            sh = ft_images.shape
            args = [(i, xyz[:,slice(*ti)].T, bxyz, (self.cache_fft, dt, sh, i), self.tukey_size) for i, ti in enumerate(time_indexes)]

            for i, (j, res) in enumerate(self._pool.imap_unordered(calc_fft_from_locs_helper, args)):                
                if self.cache_fft == "":
                    ft_images[j] = res
                 
                if ((i+1) % max(n_steps//5, 1) == 0):
                    print("{:.2f} s. Completed calculating {} of {} total ft images.".format(time.time() - self._start_time, i+1, n_steps))

        else:
            # For each window we wish to correlate...
            for i, ti in enumerate(time_indexes):
    
                # .. we generate an image and store ft of image
                t_slice = slice(*ti)
                ft_images[i] = calc_fft_from_locs(xyz[:,t_slice].T, bxyz, filter_size=self.tukey_size)
                
                if ((i+1) % max(n_steps//5, 1) == 0):
                    print("{:.2f} s. Completed calculating {} of {} total ft images.".format(time.time() - self._start_time, i+1, n_steps))
        
        print("{:.2f} s. Finished generating ft array.".format(time.time() - self._start_time))
        print("{:,} bytes".format(ft_images.nbytes))
        
        shifts, coefs = self.calc_corr_drift_from_ft_images(ft_images)
        
        # clean up of ft_images, potentially really large array
        if isinstance(ft_images, np.memmap):
            ft_images.flush()
        del ft_images
        
#        print(shifts)
#        print(coefs)
        return time_values_mid, self.binsize * shifts[:, dims_order], coefs

    def _execute(self, namespace):
#        from PYME.util import mProfile
        
#        self._start_time = time.time()
        self.trait_setq(**{"_start_time": time.time()})
        print("Starting drift correction module.")
        
        if self.multiprocessing:
            proccess_count = np.clip(multiprocessing.cpu_count()-1, 1, None)            
            self.trait_setq(**{"_pool": multiprocessing.Pool(processes=proccess_count)})
        
        locs = namespace[self.input_for_correction]

#        mProfile.profileOn(['localisations.py', 'processing.py'])
        drift_res = self.calc_corr_drift_from_locs(locs['x'], locs['y'], locs['z'] * (0 if self.flatten_z else 1), locs['t'])
        t_shift, shifts = self.rcc(self.shift_max,  *drift_res)
        
#        mProfile.profileOff()
#        mProfile.report()

        if self.multiprocessing:
            self._pool.close()
            self._pool.join()
            
        # convert frame-to-frame drift to drift from origin
        shifts = np.cumsum(shifts, 0)

        out = tabular.mappingFilter(namespace[self.input_for_mapping])
        t_out = out['t']
        # cubic interpolate with no smoothing
        dx = interpolate.CubicSpline(t_shift, shifts[:, 0])(t_out)
        dy = interpolate.CubicSpline(t_shift, shifts[:, 1])(t_out)
        dz = interpolate.CubicSpline(t_shift, shifts[:, 2])(t_out)

        if 'dx' in out.keys():
            # getting around oddity with mappingFilter
            # addColumn adds a new column but also keeps the old column
            # __getitem__ returns the new column
            # but mappings usues the old column
            # Wrap with another level of mappingFilter so the new column becomes the 'old column'
            out.addColumn('dx', dx)
            out.addColumn('dy', dy)
            out.addColumn('dz', dz)
            out = tabular.mappingFilter(out)
#            out.mdh = namespace[self.input_localizations].mdh
            out.setMapping('x', 'x + dx')
            out.setMapping('y', 'y + dy')
            out.setMapping('z', 'z + dz')
        else:
            out.addColumn('dx', dx)
            out.addColumn('dy', dy)
            out.addColumn('dz', dz)
            out.setMapping('x', 'x + dx')
            out.setMapping('y', 'y + dy')
            out.setMapping('z', 'z + dz')

        # propagate metadata, if present
        try:
            out.mdh = locs.mdh
        except AttributeError:
            pass

        namespace[self.outputName] = out
        namespace[self.output_drift] = t_shift, shifts
        
        # non essential, only for plotting out drift data
        namespace[self.output_drift_plot] = Plot(partial(generate_drift_plot, t_shift, shifts))
        
        namespace[self.output_cross_cor] = self._cc_image

#@register_module('ApplyDrift')
class ApplyDrift(ModuleBase):
    """
    Correct localization data using values from the drift interpolator.
        
    Inputs
    ------
    input_localizations : Tabular
        Dataset to correct.
    input_drift_interpolator : 
        Returns drift when called with frame number / time.
    
    Outputs
    -------
    output_name : Tabular
        Drift-corrected dataset.
    """
    
    input_localizations = Input('Localizations')
#    input_drift = Input('drift')
    input_drift_interpolator = Input('drift_interpolator')

    output_name = Output('corrected_localizations')
    
    def execute(self, namespace):
#        t_shift, shifts = namespace[self.input_drift]
        out = tabular.mappingFilter(namespace[self.input_localizations])
        out.mdh = namespace[self.input_localizations].mdh
        
        t_out = out['t']

        dx = namespace[self.input_drift_interpolator][0](t_out)
        dy = namespace[self.input_drift_interpolator][1](t_out)
        dz = namespace[self.input_drift_interpolator][2](t_out)
        
        if 'dx' in out.keys():
            # getting around oddity with mappingFilter
            # addColumn adds a new column but also keeps the old column
            # __getitem__ returns the new column
            # but mappings usues the old column
            # Wrap with another level of mappingFilter so the new column becomes the 'old column'
            out.addColumn('dx', dx)
            out.addColumn('dy', dy)
            out.addColumn('dz', dz)
            out = tabular.mappingFilter(out)
            out.mdh = namespace[self.input_localizations].mdh
            out.setMapping('x', 'x + dx')
            out.setMapping('y', 'y + dy')
            out.setMapping('z', 'z + dz')
        else:
            out.addColumn('dx', dx)
            out.addColumn('dy', dy)
            out.addColumn('dz', dz)
            out.setMapping('x', 'x + dx')
            out.setMapping('y', 'y + dy')
            out.setMapping('z', 'z + dz')
        
        try:
            out.mdh = self.input_localizations.mdh
        except AttributeError:
            pass

        namespace[self.output_name] = out