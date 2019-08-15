# -*- coding: utf-8 -*-
"""
Created on Tue Apr 02 14:22:15 2019

@author: kkc29


based off PYME recipes processing.py
"""



from PYME.recipes.base import ModuleBase, register_module, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, File

import numpy as np
from scipy import ndimage, optimize, signal, interpolate
from PYME.IO.image import ImageStack
from PYME.IO.dataWrap import ListWrap

from PYME.recipes import processing
import time

import logging
logger=logging.getLogger(__name__)


## Can't use. Output wrapped with bugged ListWrap
#@register_module('Clip')
#class ClipFilter(Filter):
#    """
#        Bypass saving the masks in memory c.f. thresholding & (invert) & multiplication.
#        Replaces out of range values to zeros.
#    """
#    threshold_lower = Float(0)
#    threshold_upper = Float(0)
#    
#    def applyFilter(self, data, chanNum, frNum, im):
#        
##        return np.clip(data, self.threshold_lower, self.threshold_upper)
#        data = np.asarray(data)
#        data[data < self.threshold_lower] = 0
#        data[data > self.threshold_upper] = 0
#        
#        return data
#
#    def completeMetadata(self, im):
#        im.mdh['Processing.Clipping.LowerBounds'] = self.threshold_lower
#        im.mdh['Processing.Clipping.UpperBounds'] = self.threshold_upper

#class DerivedModule(ModuleBase):
#    """
#    In the absent of build-in abstract methods support
#    """
#    
#    def execute(self, namespace):
#        self.complete_metadata()
#        self.cleanup_caches()
#    
#    def complete_metadata(self):
#        pass
#    
#    def cleanup_caches(self):
#        pass

#@register_module('Preprocessing')
class PreprocessingFilter(ModuleBase):
    """
        Bypass saving the masks in memory c.f. thresholding & (invert) & multiplication, etc
        Replaces out of range values to defined values.
        Applies Tukey filter to dampen potential edge artifacts.
    """
    input_name = Input('input')
    threshold_lower = Float(0)
    clip_to_lower = Float(0)
    threshold_upper = Float(65535)
    clip_to_upper = Float(0)
    median_filter_size = Int(3)
    tukey_size = Float(0.25)
    cache = File("clip_cache.bin")
    output_name = Output('clipped_images')
    
    def execute(self, namespace):
        self._start_time = time.time()
        ims = namespace[self.input_name]
        
        dtype = ims.data[:,:,0].dtype
        
        # Somewhat arbitrary way to decide on chunk size 
        chunk_size = 100000000 / ims.data.shape[0] / ims.data.shape[1] / dtype.itemsize
        chunk_size = max(1, chunk_size)
#        print chunk_size
        
        tukey_mask_x = signal.tukey(ims.data.shape[0], self.tukey_size)
        tukey_mask_y = signal.tukey(ims.data.shape[1], self.tukey_size)
        self._tukey_mask_2d = np.multiply(*np.meshgrid(tukey_mask_x, tukey_mask_y, indexing='ij'))[:,:,None]

        raw_data = np.memmap(self.cache, dtype=dtype, mode='w+', shape=tuple(np.asarray(ims.data.shape[:3], dtype=np.long)))
        progress = 0.2 * ims.data.shape[2]
        for f in np.arange(0, ims.data.shape[2], chunk_size):
            raw_data[:,:,f:f+chunk_size] = self.applyFilter(ims.data[:,:,f:f+chunk_size])
            
            
            if (f+chunk_size >= progress):
                raw_data.flush()
                progress += 0.2 * ims.data.shape[2]
                print("{:.2f} s. Completed clipping {} of {} total images.".format(time.time() - self._start_time, min(f+chunk_size, ims.data.shape[2]), ims.data.shape[2]))
        
#        clipped_images = ImageStack(self.applyFilter(raw_data), mdh=ims.mdh)
        clipped_images = ImageStack(raw_data, mdh=ims.mdh)
        self.completeMetadata(clipped_images)
        
        namespace[self.output_name] = clipped_images
    
    def applyFilter(self, data):
        """
            Performs the actual filtering here.
        """
        if self.median_filter_size > 0:
            data = ndimage.median_filter(data, self.median_filter_size, mode='nearest')
        data[data >= self.threshold_upper] = self.clip_to_upper        
        data[data <= self.threshold_lower] = self.clip_to_lower        
        data -= self.clip_to_lower        
        if self.tukey_size > 0:
            data = data * self._tukey_mask_2d
        return data

    def completeMetadata(self, im):
        im.mdh['Processing.Clipping.LowerBounds'] = self.threshold_lower
        im.mdh['Processing.Clipping.LowerSetValue'] = self.clip_to_lower
        im.mdh['Processing.Clipping.UpperBounds'] = self.threshold_upper
        im.mdh['Processing.Clipping.UpperSetValue'] = self.clip_to_upper
        im.mdh['Processing.Tukey.Size'] = self.tukey_size
        
        
#@register_module('Binning')
class Binning(ModuleBase):
    """
        Downsample data (mean) in x, y, t. (Doesn't support z.)
        X, Y pixels does't fill a full bin are dropped.
        Z pixels can have a partially filled bin
        Via numpy reshape function.
    """
    
    inputName = Input('input')
    x_start = Int(0)
    x_end = Int(-1)
    y_start = Int(0)
    y_end = Int(-1)
#    z_start = Int(0)
#    z_end = Int(-1)
    binsize = List([1,1,1], minlen=3, maxlen=3)
#    cache_1 = File("binning_cache_1.bin")
    cache_2 = File("binning_cache_2.bin")
    outputName = Output('binned_image')
    
    def execute(self, namespace):
        self._start_time = time.time()
        ims = namespace[self.inputName]
        
        binsize = np.asarray(self.binsize, dtype=np.int)
#        print (binsize)

        # unconventional, end stop in inclusive
        x_slice = np.arange(ims.data.shape[0]+1)[slice(self.x_start, self.x_end, 1)]
        y_slice = np.arange(ims.data.shape[1]+1)[slice(self.y_start, self.y_end, 1)]
        x_slice = x_slice[:x_slice.shape[0] // binsize[0] * binsize[0]]
        y_slice = y_slice[:y_slice.shape[0] // binsize[1] * binsize[1]]
#        print x_slice, len(x_slice)
#        print y_slice, len(y_slice)
        bincounts = np.asarray([len(x_slice)//binsize[0], len(y_slice)//binsize[1], -(-ims.data.shape[2]//binsize[2])], dtype=np.long)
        
        x_slice_ind = slice(x_slice[0], x_slice[-1]+1)
        y_slice_ind = slice(y_slice[0], y_slice[-1]+1)
        
#        print (bincounts)
        new_shape = np.stack([bincounts, binsize], -1).flatten()
#        print(new_shape)
        
        # need to wrap this to work for multiply color channel images
#        binned_image = ims.data[:,:,:].reshape(new_shape)
        dtype = ims.data[:,:,0].dtype
        
        
#        ### This was for manual caching the data first, too slow
#        print(ims.data.shape[:3])
#        chunk_size = 100000000 / ims.data.shape[0] / ims.data.shape[1] / dtype.itemsize
#        chunk_size = max(1, chunk_size)
#        print chunk_size
#        raw_data = np.memmap(self.cache_1, dtype=dtype, mode='w+', shape=tuple(np.asarray(ims.data.shape[:3], dtype=np.long)))

#        
#        progress = 0.2 * ims.data.shape[2]     
#        for f in np.arange(0, ims.data.shape[2], chunk_size):
#            raw_data[:,:,f:f+chunk_size] = ims.data[:,:,f:f+chunk_size].squeeze()
#            

#            if (f+chunk_size > progress):
#                raw_data.flush()
#                progress += 0.2 * ims.data.shape[2]  
#                print("{:.2f} s. Completed binning {} of {} total images.".format(time.time() - self._start_time, min(f+chunk_size, ims.data.shape[2]), ims.data.shape[2]))
#                    
#        raw_data.shape = new_shape
        
#        binned_image = binned_image.mean((1,3,5))
#        print bincounts
        binned_image = np.memmap(self.cache_2, dtype=dtype, mode='w+', shape=tuple(np.asarray(bincounts, dtype=np.long)))
#        print binned_image.shape
#        print raw_data.shape
        
        new_shape_one_chunk = new_shape.copy()
        new_shape_one_chunk[4] = 1
        new_shape_one_chunk[5] = -1
#        print new_shape_one_chunk
        progress = 0.2 * ims.data.shape[2]
#        print 
        for i, f in enumerate(np.arange(0, ims.data.shape[2], binsize[2])):
            raw_data_chunk = ims.data[x_slice_ind,y_slice_ind,f:f+binsize[2]].squeeze()
            
            binned_image[:,:,i] = raw_data_chunk.reshape(new_shape_one_chunk).mean((1,3,5)).squeeze()
            
            if (f+binsize[2] >= progress):
                binned_image.flush()
                progress += 0.2 * ims.data.shape[2]
                print("{:.2f} s. Completed binning {} of {} total images.".format(time.time() - self._start_time, min(f+binsize[2], ims.data.shape[2]), ims.data.shape[2]))
        
        
#        binned_image = raw_data.mean((1,3,5))

#        print(type(binned_image))
        im = ImageStack(binned_image, titleStub=self.outputName)
#        print(type(im.data))
        im.mdh.copyEntriesFrom(ims.mdh)
        im.mdh['Parent'] = ims.filename
        try:
            ### Metadata must be logged correctly for the measured drift to be applicable to the source image
            im.mdh['voxelsize.x'] *= binsize[0] 
            im.mdh['voxelsize.y'] *= binsize[1]
#            im.mdh['voxelsize.z'] *= binsize[2]
            if 'recipe.binning' in im.mdh.keys():
                im.mdh['recipe.binning'] = binsize * im.mdh['recipe.binning']
            else:
                im.mdh['recipe.binning'] = binsize
        except:
            pass
        
        namespace[self.outputName] = im


def calc_shift_helper(args):
    """
        Wrappers needed for imap_unordered functions, etc.
    """
    return (args[0], calc_shift(*args[1:]))

def calc_shift(index_1, index_2, origin=0, cache_fft=None, debug_cross_cor=None):
    """
        Are the actual ft images passed? If not, fetch them from file cache.
    """
    if not cache_fft is None and cache_fft[0] != "":
        path, dtype, shape = cache_fft
        ft_images = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        ft_1 = ft_images[index_1]
        ft_2 = ft_images[index_2]
        del ft_images
    else:
        ft_1 = index_1
        ft_2 = index_2
        
    return calc_shift_direct(ft_1, ft_2, origin, debug_cross_cor)

def calc_shift_direct(ft_1, ft_2, origin=0, debug_cross_cor=None):
    """
        Does the actual fft cross correlation.
        Clean up - including cropping, thresholding, mask dilation.
        Performs n dimension gaussian fit and returns center.
    """
    ft_1 = ndimage.fourier_gaussian(ft_1, 0.5)
    ft_2 = ndimage.fourier_gaussian(ft_2, 0.5)
    # module level for multiprocessing
    tmp = ft_1 * np.conj(ft_2)    
    del ft_1, ft_2
    cross_corr = np.abs(np.fft.ifftshift(np.fft.irfftn(tmp)))
    flat_dims = np.where(np.asarray(cross_corr.shape) == 1)[0]
    if len(flat_dims) > 0:
        cross_corr = cross_corr.squeeze()

    if cross_corr.sum() == 0:
        return origin * np.nan
    
#    threshold = np.ptp(cross_corr) * 0.5 + np.min(cross_corr)
    
    # cheat and striaght up crop out 3/4 of the image if it's large
    # i.e. drift not allow to span 1/4 the image width
    cropping = [slice(dim*6//16, -dim*6//16) if dim >= 16 else slice(None, None) for dim in cross_corr.shape]
    cross_corr_mask = np.zeros(cross_corr.shape)
    cross_corr_mask[cropping] = True
    
#    threshold = np.percentile(cross_corr[cropping], 95)
    
    threshold = np.ptp(cross_corr[cross_corr_mask.astype(bool)]) * 0.75 + np.min(cross_corr[cross_corr_mask.astype(bool)])
    
    cross_corr_mask *= cross_corr > threshold
    
    # difficult to adjust for complete despeckling. slow?
#    cross_corr_mask = ndimage.binary_erosion(cross_corr_mask, structure=np.ones((1,)*cross_corr_mask.ndim), iterations=1, border_value=1, )
#    cross_corr_mask = ndimage.binary_dilation(cross_corr_mask, structure=np.ones((3,)*cross_corr_mask.ndim), iterations=1, border_value=0, )
#    print("mask {}".format(cross_corr_mask.sum()))
    
    labeled_image, labeled_counts = ndimage.label(cross_corr_mask)
    if labeled_counts > 1: 
        max_index = np.argmax(ndimage.mean(cross_corr_mask, labeled_image, range(1, labeled_counts+1))) + 1
        cross_corr_mask = labeled_image == max_index
    
    cross_corr_mask = ndimage.binary_dilation(cross_corr_mask, structure=np.ones((5,)*cross_corr_mask.ndim), iterations=1, border_value=0, )
    
    cross_corr_thresholded = cross_corr * cross_corr_mask
#    
    if not debug_cross_cor is None:
        i, (path, dtype, shape) = debug_cross_cor
        cc_images = np.memmap(path, mode="r+", dtype=dtype, shape=shape)
        cross_corr_thresholded_repadded = cross_corr_thresholded.view()
        for d in flat_dims:
            cross_corr_thresholded_repadded = np.expand_dims(cross_corr_thresholded_repadded, d)
        short_axis = np.argmin(cross_corr_thresholded_repadded.shape)
        cc_images[i] = cross_corr_thresholded_repadded.mean(axis=short_axis)
        del cc_images
    
    dims = range(len(cross_corr.shape))
    
    # crop out masked area
    bounds = np.zeros((len(dims), 2), dtype=np.int)
    for d in dims:
        dims_tmp = list(dims)
        dims_tmp.remove(d)
        mask_1d = np.any(cross_corr_mask, axis=tuple(dims_tmp))
        bounds[d] = np.where(mask_1d)[0][[0, -1]] + [0, 1]
        cross_corr_thresholded = cross_corr_thresholded.take(np.arange(*bounds[d]), axis=d)    

#    offset = np.zeros(len(dims))
#    for d, length in enumerate(cross_corr_thresholded.shape):
#        dims_tmp = list(dims)
#        dims_tmp.remove(d)
#        offset[d] = np.sum(np.arange(length) * cross_corr_thresholded.sum(axis=tuple(dims_tmp)))
#    offset /= cross_corr_thresholded.sum()
#    offset += bounds[:, 0]
    
    cross_corr_thresholded[cross_corr_thresholded==0] = np.nan
#    cross_corr_thresholded -= np.nanmin(cross_corr_thresholded)
    cross_corr_thresholded /= np.nanmax(cross_corr_thresholded)

#    ### Gaussian fit
##    p0 = [np.nanmax(cross_corr_thresholded), np.nanmin(cross_corr_thresholded)]
#    p0 = [1, 0]
#    grids = list()
#    for i, d in enumerate(cross_corr_thresholded.shape):
#        grids.append(np.arange(d))
#        p0.extend([(d-1)*0.5, 0.5*d])
##    print grids
##    print p0
#    res = optimize.least_squares(guassian_nd_error, p0, args=(grids, cross_corr_thresholded))
##    print res.x 
    
    ### Rbf peak finding
    p0 = []
    grids = list()
    for i, d in enumerate(cross_corr_thresholded.shape):
        grids.append(np.arange(d))
        p0.append(0.5*d)
    rbf_interpolator = build_rbf(grids, cross_corr_thresholded)
    res = optimize.minimize(rbf_nd_error, p0, args=rbf_interpolator)
    
    offset = list()
    for i in xrange(len(cross_corr_thresholded.shape)):
#        offset.append(res.x[2*i+2])
        offset.append(res.x[i])
    offset += bounds[:, 0]
#    print offset
    
    if len(flat_dims) > 0:
        offset = np.insert(offset, flat_dims, 0)
        
    return offset - origin

def guassian_nd_error(p, dims, data):
    """
        Calculates mask size normalized error. Protected against nan's.
    """
    mask = ~np.isnan(data)
    return (data - gaussian_nd(p, dims))[mask]/mask.sum()

def gaussian_nd(p, dims):
    """
        Creates n dimension gaussian with background.
        p: tuple of variable length (A, bg, dim_0, sig_0, dim_1, sig_1, dim_2, sig_2, ...)
        dims: 1d axis. need to match length with p
    """
    A, bg = p[:2]
    dims_nd = np.meshgrid(*dims, indexing='ij')
    exponent = 0
    for i, dim in enumerate(dims_nd):
#        print 2+2*i, 2+2*i+1
        exponent += (dim-p[2+2*i])**2/(2*p[2+2*i+1]**2)
    return A * np.exp(-exponent) + bg

def build_rbf(grids, data):
    grid_nd_list = np.meshgrid(*grids, indexing='ij')
    data = data.flatten()
    mask = ~np.isnan(data)
    data = data[mask]
    grid_nd_list_cleaned = [grid_nd.flatten()[mask] for grid_nd in grid_nd_list]
    grid_nd_list_cleaned.append(data)
    return interpolate.Rbf(*grid_nd_list_cleaned, function='multiquadric', epsilon=1.)

def rbf_nd_error(p, rbf_interpolator):
    return -rbf_interpolator(*p)

def rbf_nd(rbf_interpolator, dims):
    out_shape = [len(d) for d in dims]
    dims_nd_list = np.meshgrid(*dims, indexing='ij')
    dims_nd_list_cleaned = [dim_nd.flatten() for dim_nd in dims_nd_list]
    return rbf_interpolator(*dims_nd_list_cleaned).reshape(out_shape)
    

#@register_module('RCCDriftCorrectionFromCachedFFT')
class RCCDriftCorrectionBase(ModuleBase):
    """    
    Performs drift correction using redundant cross-correlation from
    Wang et al. Optics Express 2014 22:13 (Bo Huang's RCC algorithm).
    Base class for other RCC recipes.
    Can take cached fft input (as filename, not an 'input').
    Only output drift as tuple of time points, and drift amount.
    Currently not registered by itself since not very usefule.
    """
    
    ft_cache = File("rcc_cache.bin")
    method = Enum(['RCC', 'MCC', 'DCC'])
    # redundant cross-corelation, mean cross-correlation, direction cross-correlation
    shift_max = Float(5)  # nm
    corr_window = Int(5)
    multiprocessing = Bool()
    debug_cor_file = File()

    output_drift = Output('drift')
    output_drift_plot = Output('drift_plot')
    
    # if debug_cor_file not blank, filled with imagestack of cross correlation
    output_cross_cor = Output('cross_cor')

    def calc_corr_drift_from_ft_images(self, ft_images):
        import time
        n_steps = ft_images.shape[0]
        
        # Matrix equation coefficient matrix
        # Shape can be predetermined based on method
        if self.method == "DCC":
            coefs_size = n_steps - 1
        elif self.corr_window > 0:
            coefs_size = n_steps * self.corr_window - self.corr_window * (self.corr_window + 1) / 2
        else:
            coefs_size = n_steps * (n_steps-1) / 2
        coefs = np.zeros((coefs_size, n_steps-1))
        shifts = np.zeros((coefs_size, 3))

        counter = 0

        ft_1_cache = list()
        ft_2_cache = list()
        autocor_shift_cache = list()
        
#        print self.debug_cor_file
        if not self.debug_cor_file == "":
            cc_file_shape = [shifts.shape[0], ft_images.shape[1], ft_images.shape[2], (ft_images.shape[3]-1)*2]
            
            # flatten shortest dimension to reduce cross correlation to 2d images for easier debugging
            min_arg = min(enumerate(cc_file_shape[1:]), key=lambda x: x[1])[0] + 1
            cc_file_shape.pop(min_arg)
            cc_file_args = (self.debug_cor_file, np.float, tuple(cc_file_shape))
            cc_file = np.memmap(cc_file_args[0], dtype=cc_file_args[1], mode="w+", shape=cc_file_args[2])
#            del cc_file
            cc_args = zip(range(shifts.shape[0]), (cc_file_args, )*shifts.shape[0])
        else:
            cc_args = (None,) * shifts.shape[0]

        # For each ft image, calculate correlation
        for i in np.arange(0, n_steps-1):
            if self.method == "DCC" and i > 0:
                break
            
            ft_1 = ft_images[i, :, :]
            
            autocor_shift = calc_shift(ft_1, ft_1)

            for j in np.arange(i+1, n_steps):                
                if (self.method != "DCC") and (self.corr_window > 0) and (j-i > self.corr_window):
                    break
                
                ft_2 = ft_images[j, :, :]

                coefs[counter, i:j] = 1
                
                # if multiprocessing, use cache when defined
                if self.multiprocessing:
                    # if reading ft_images from cache, replace ft_1 and ft_2 with their indices
                    if not self.ft_cache == "":
                        ft_1 = i
                        ft_2 = j
#                    pool_results.append(self._pool.apply_async(calc_shift, args=(ft_1, ft_2, autocor_shift, (self.ft_cache, ft_images.dtype, ft_images.shape),)))
                    ft_1_cache.append(ft_1)
                    ft_2_cache.append(ft_2)
                    autocor_shift_cache.append(autocor_shift)
                else:
                    shifts[counter, :] = calc_shift(ft_1, ft_2, autocor_shift, None, cc_args[counter])
                    
                    if ((counter+1) % (coefs_size//5) == 0):
                        print("{:.2f} s. Completed calculating {} of {} total shifts.".format(time.time() - self._start_time, counter+1, coefs_size))
                
                counter += 1
                
        if self.multiprocessing:
            args = zip(range(len(autocor_shift_cache)),
                       ft_1_cache,
                       ft_2_cache,
                       autocor_shift_cache,
                       len(ft_1_cache) * ((self.ft_cache, ft_images.dtype, ft_images.shape),),
                       cc_args
                       )
            for i, (j, res) in enumerate(self._pool.imap_unordered(calc_shift_helper, args)):
                shifts[j,] = res
                
                if ((i+1) % (coefs_size//5) == 0):
                    print("{:.2f} s. Completed calculating {} of {} total shifts.".format(time.time() - self._start_time, i+1, coefs_size))
                
        print("{:.2f} s. Finished calculating all shifts.".format(time.time() - self._start_time))
        print("{:,} bytes".format(coefs.nbytes))
        print("{:,} bytes".format(shifts.nbytes))
        
        if not self.debug_cor_file == "":
            # move time axis for ImageStack
            cc_file = np.moveaxis(cc_file, 0, 2)
            self._cc_image = ImageStack(data=cc_file.copy())
            del cc_file
        else:
            self._cc_image = None

        assert (np.all(np.any(coefs, axis=1))), "Coefficient matrix filled less than expected."

        mask = np.where(~np.isnan(shifts).any(axis=1))[0]
        if len(mask) < shifts.shape[0]:
            print("Removed {} cross correlations due to bad/missing data?".format(shifts.shape[0]-len(mask)))            
            coefs = coefs[mask, :]
            shifts = shifts[mask, :]
        
        assert (coefs.shape[0] > 0) and (np.linalg.matrix_rank(coefs) == n_steps - 1), "Something went wrong with coefficient matrix. Not full rank."
                
        return shifts, coefs  # shifts.shape[0] is n_steps - 1

    def rcc(self, shift_max, t_shift, shifts, coefs, ):
        """
            Should probably rename function.
            Takes cross correlation results and calculates shifts.
        """
        import time
        
        print("{:.2f} s. About to start solving shifts array.".format(time.time() - self._start_time))

        # Estimate drift
        drifts = np.matmul(np.linalg.pinv(coefs), shifts)
        
        print("{:.2f} s. Done solving shifts array.".format(time.time() - self._start_time))
        
        if self.method == "RCC":
        
            # Calculate residual errors
            residuals = np.matmul(coefs, drifts) - shifts
            residuals_dist = np.linalg.norm(residuals, axis=1)
    
            # Sort and mask residual errors
            residuals_arg = np.argsort(-residuals_dist)
            residuals_arg = residuals_arg[residuals_dist[residuals_arg] > shift_max]
    
            # Remove coefs rows
            # Descending from largest residuals to small
            # Only if matrix remains full rank
            coefs_temp = np.empty_like(coefs)
            counter = 0
            for i, index in enumerate(residuals_arg):
                coefs_temp[:] = coefs
                coefs_temp[index, :] = 0
                if np.linalg.matrix_rank(coefs_temp) == coefs.shape[1]:
                    coefs[:] = coefs_temp
    #                print("index {} with residual of {} removed".format(index, residuals_dist[index]))
                    counter += 1
                else:
                    print("Could not remove all residuals over shift_max threshold.")
                    break
            print("removed {} in total".format(counter))
            
            # Estimate drift again
            drifts = np.matmul(np.linalg.pinv(coefs), shifts)
            
            print("{:.2f} s. RCC completed. Repeated solving shifts array.".format(time.time() - self._start_time))

        # pad with 0 drift for first time point
        drifts = np.pad(drifts, [[1,0],[0,0]], 'constant', constant_values=0)

        return t_shift, drifts

    def execute(self, namespace):
        # dervied versions of RCC need to override this method
        # 'execute' of this RCC base class is not throughly tested as its use is probably quite limited.
        
#        from PYME.IO import tabular
        import time
        import multiprocessing

#        from PYME.util import mProfile
        
        self._start_time = time.time()
        print("Starting drift correction module.")
        
        if self.multiprocessing:
            proccess_count = np.clip(multiprocessing.cpu_count()-1, 1, None)
            self._pool = multiprocessing.Pool(processes=proccess_count)
        
#        mProfile.profileOn(['localisations.py'])

        drift_res = self.calc_corr_drift_from_ft_images(self.ft_cache)
        t_shift, shifts = self.rcc(self.shift_max,  *drift_res)
#        mProfile.profileOff()
#        mProfile.report()

        if self.multiprocessing:
            self._pool.close()
            self._pool.join()
            
        # convert frame-to-frame drift to drift from origin
        shifts = np.cumsum(shifts, 0)

        namespace[self.output_drift] = t_shift, shifts
        
    def generate_drift_plot(self, t, shifts):
        """
            Generates plot of drift and returns matplotlib figure object
        """
        from matplotlib import pyplot
        pyplot.ioff()
        fig, ax = pyplot.subplots(1, 1)
        lines = ax.plot(t, shifts, marker='.', )
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Drift (nm)")
        ax.legend(lines, ['x', 'y', 'z'][:shifts.shape[1]])
        fig.tight_layout()
        pyplot.ion()
        
        return fig

def calc_fft_from_image_helper(args):
    """
        Wrapper for working with multiprocessing functions.
    """
    return (args[0], calc_fft_from_image(*args[1:]))

def calc_fft_from_image(im, cache_fft=None):
    # module level for multiprocessing
    """
        Reals real fft from passed or cached image
    """
    
    if not cache_fft is None and cache_fft[0] != "":
        path, dtype, shape, index = cache_fft
        ft_images = np.memmap(path, mode="r+", dtype=dtype, shape=shape)
        ft_images[index] = np.fft.rfftn(im)
        ft_images.flush()
        del ft_images
        return
    
    return np.fft.rfftn(im)

def shift_image_helper(args):
    """
        Wrapper for working with multiprocessing functions.
    """
    return (args[2], shift_image(*args))

def shift_image(ft_image, shifts, index=None, cache_fft=None, cache_image=None, cache_kxyz=None):
    """
        Handles file caching issues.
    """
    if not cache_fft is None and cache_fft[0] != "":
        path, dtype, shape = cache_fft
        ft_images = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        ft_image = ft_images[index]
        del ft_images
#    else:
#        ft_image = index_or_img
        
    if not cache_image is None and cache_image[0] != "":
        path, dtype, shape = cache_image
        images = np.memmap(path, mode="r+", dtype=dtype, shape=shape)
        images[index] = shift_image_direct(ft_image, shifts, cache_kxyz)
        images.flush()
        del images
        return
    
    return shift_image_direct(ft_image, shifts, cache_kxyz)

def shift_image_direct(source_ft, shifts, kxyz=None):
    """
        Performs fft based sub-pixel shifts. Can accept cached kxyz
    """
    if kxyz is None:
        kx = np.fft.fftfreq(source_ft.shape[0])
        ky = np.fft.fftfreq(source_ft.shape[1])
        kz = np.fft.fftfreq(source_ft.shape[2]) * 0.5
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    else:
        kx, ky, kz = kxyz
        
    return np.abs(np.fft.irfftn(source_ft*np.exp(-2j*np.pi*(kx*shifts[0] + ky*shifts[1] + kz*shifts[2]))))

#@register_module('RCCDriftCorrection')
class RCCDriftCorrection(RCCDriftCorrectionBase):
    """
    Performs drift correction using cross-correlation, includes redundant CC from
    Wang et al. Optics Express 2014 22:13 (Bo Huang's RCC algorithm).
    Derived class for the image based verison of RCC.    
    Images are not padded prior to FT. Use the preprocessing modules
    """
    
    input_image = Input('input')
#    binning = List([1,1,1], minlen=3, maxlen=3)
    
    image_cache = File("rcc_shifted_image.bin")
    outputName = Output('drift_corrected_image')
    output_drift = Output('drift')
    
    class WrappedImage(object):
        """
            light wrapper around image data in imagestack
            trying to preserve the buffer and avoid making a copy in memory
            trys to be intelligent based on info gathered from PYME.IO.DataSources.BaseDataSource.py
            Shuffles the dimensions around if necessary to maximize gain from using the real FT.
        """
        ims = None
        dims_order = None
        swaped_axes = (0,0)
        
        def __init__(self, ims):
#            print ims.data.__class__
#            print isinstance(ims.data, ListWrap)
#            
#            print 'strange'
#            ## Wrapper to avoid bug in ListWrap
#            if isinstance(ims.data, ListWrap):                
#                self.ims = ImageStack(np.stack([ims.data[:,:,:,i] for i in range(ims.data.shape[3])], -1), mdh=ims.mdh)
#                print self.ims.data.shape
#            else:
            self.ims = ims
#            
#            print self.ims.data.__class__
#            print 'stranger'
            
            nDims = self.ims.data.nTrueDims
            if nDims < 3 or nDims > 4:
                raise Exception("Don't know how to deal with data less than 3 or more than 4 dimensions.")
            elif nDims == 3:
                # don't care if 3rd dim is Z/T/C, just use it
                cross_cor_dim = 2                
            elif nDims == 4:
                # Assign T as cross_cor_dim if it exists, otherwise use Z
                pos_t = self.ims.data.additionalDims.find('T')
                if pos_t != -1:
                    cross_cor_dim = pos_t
                else:
                    pos_z = self.ims.data.additionalDims.find('Z')
                    if pos_z != -1:
                        cross_cor_dim = pos_z
                    else:
                        assert True, "This shouldn't happen. No T or Z defined in imagestack. Don't know what to do."                    
            else:
                assert True, "This shouldn't happen. IF statements falling through."
                
            xyz_dims = range(0, 4)
            xyz_dims.remove(cross_cor_dim)
            self.dims_order = [cross_cor_dim,] + xyz_dims
            
        def swapaxes(self, a, b):
            a = max(self.dims_order) if a == -1 else a
            b = max(self.dims_order) if b == -1 else b
            a = self.dims_order.index(a)
            b = self.dims_order.index(b)            
            self.swaped_axes = (a-1, b-1)
            self.dims_order[a], self.dims_order[b] = self.dims_order[b], self.dims_order[a]
            
        def __getitem__(self, slices):
            # first dim is for cross correlation
            # fetch data depending on self.dims_order
            # also swap axes from data array on return            
            sorting_order = np.argsort(self.dims_order)
#            print self.ims.data.shape
#            print slices
#            print sorting_order
#            print [slices[sorting_order[i]] for i in np.arange(len(slices))]
#            print type(self.ims.data)
#            print self.ims.data.__class__
            data = self.ims.data[[slices[sorting_order[i]] for i in np.arange(len(slices))]]
            return np.swapaxes(data, *self.swaped_axes)            
        
        @property
        def shape(self):
            raw_shape = np.asarray(self.ims.data.shape)
#            print raw_shape
#            print raw_shape[self.dims_order]
            return tuple(raw_shape[self.dims_order])                
    
    def calc_corr_drift_from_imagestack(self, ims):
        """
            Calculates fft images from source image.
            Feeds fft images to calc_corr_drift_from_ft_images (in base class).
            Returns shifts in pixels (i think).
        """
        import time
        
        images = self.WrappedImage(ims)        
        
        dims_order = np.arange(0, len(images.shape)-1)
        dims_length = np.asarray(images.shape[1:])
        
        dims_largest_index = np.argmax(dims_length)
        dims_order[-1], dims_order[dims_largest_index] = dims_order[dims_largest_index], dims_order[-1]
#        images = np.swapaxes(images, -2, dims_largest_index)
        images.swapaxes(-1, dims_largest_index)
        
        images_shape = images.shape
#        print(images_shape)
        
        ft_images_shape = tuple([long(i) for i in [images_shape[0], images_shape[1], images_shape[2], images_shape[3]//2 + 1]])
        
        # use memmap for caching if ft_cache is defined
        if self.ft_cache == "":
            ft_images = np.zeros(ft_images_shape, dtype=np.complex)
        else:
            ft_images = np.memmap(self.ft_cache, dtype=np.complex, mode='w+', shape=ft_images_shape)
            
#        print(ft_images.shape)
        print("{:,} bytes".format(ft_images.nbytes))
        
        print("{:.2f} s. About to start heavy lifting.".format(time.time() - self._start_time))
            
        if self.multiprocessing:            
            
            dt = ft_images.dtype
            sh = ft_images.shape
            args = [(i, images[i,:,:,:], (self.ft_cache, dt, sh, i)) for i in np.arange(images.shape[0])]

            for i, (j, res) in enumerate(self._pool.imap_unordered(calc_fft_from_image_helper, args)):
                if self.ft_cache == "":
                    ft_images[j] = res
                    
                if ((i+1) % (images_shape[0]//5) == 0):
                    print("{:.2f} s. Completed calculating {} of {} total ft images.".format(time.time() - self._start_time, i+1, images_shape[0]))
        else:
            
            for i in np.arange(images.shape[0]):
    
                # .. we store ft of image                
                ft_images[i] = calc_fft_from_image(images[i,:,:,:])
                
                if ((i+1) % (images_shape[0]//5) == 0):
                    print("{:.2f} s. Completed calculating {} of {} total ft images.".format(time.time() - self._start_time, i+1, images_shape[0]))
        
        print("{:.2f} s. Finished generating ft array.".format(time.time() - self._start_time))
        print("{:,} bytes".format(ft_images.nbytes))
        
        shifts, coefs = self.calc_corr_drift_from_ft_images(ft_images)
        
        self._ft_images = ft_images
        self._images = images
        
        return np.arange(images.shape[0]), shifts[:, dims_order], coefs
    
#    def shift_images(self, shifts):
#        """
#            Moved to separate module. But this is still better, 3D vs 2D?
#        """
#        import time
#        raw_shifts = shifts[:, np.argsort(self._images.dims_order[1:])]
#        
##        print(self._ft_images.shape)
##        padding = np.stack((self._ft_images.shape,)*2, -1)/2
##        padding[0, :] = 0
##        print(padding)
##        self._ft_images = np.pad(self._ft_images, padding, mode="constant", constant_values=0)
#        
#        kx = (np.fft.fftfreq(self._ft_images.shape[1])) 
#        ky = (np.fft.fftfreq(self._ft_images.shape[2]))
#        kz = (np.fft.fftfreq(self._ft_images.shape[3])) * 0.5
#        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
#        
#        if self.image_cache == "":
#            data_shifted = np.empty((self._ft_images.shape[0], self._ft_images.shape[1], self._ft_images.shape[2], 2*(self._ft_images.shape[3]-1)))
#        else:
#            data_shifted = np.memmap(self.image_cache, dtype=np.float, mode='w+', shape=(self._ft_images.shape[0], self._ft_images.shape[1], self._ft_images.shape[2], 2*(self._ft_images.shape[3]-1)))            
#        
#        if self.multiprocessing:
#
#            none_args = ((None), ) * self._ft_images.shape[0]
#            fft_cache_args = ((self.ft_cache, self._ft_images.dtype, self._ft_images.shape), ) * self._ft_images.shape[0]
#            image_cache_args = ((self.image_cache, data_shifted.dtype, data_shifted.shape), ) * self._ft_images.shape[0]
#            args = zip(none_args, raw_shifts, np.arange(self._ft_images.shape[0]), fft_cache_args, image_cache_args)
#            
#            for i, (j, res) in enumerate(self._pool.imap_unordered(shift_image_helper, args)):
#                if self.image_cache == "":
#                    data_shifted[j] = res
#                    
#                if ((i+1) % (self._ft_images.shape[0]//5) == 0):
#                    print("{:.2f} s. Completed shifting {} of {} total images.".format(time.time() - self._start_time, i+1, self._ft_images.shape[0]))
#        else:
#            for i in np.arange(self._ft_images.shape[0]):
#                data_shifted[i] = shift_image(self._ft_images[i], raw_shifts[i], cache_kxyz=(kx, ky, kz))
#                
#                if ((i+1) % (self._ft_images.shape[0]//5) == 0):
#                    print("{:.2f} s. Completed shifting {} of {} total images.".format(time.time() - self._start_time, i+1, self._ft_images.shape[0]))
#            
#        data_shifted = np.moveaxis(data_shifted, 0, self._images.dims_order[0])
#        
#        axes_a = self._images.dims_order[self._images.swaped_axes[0]+1]
#        axes_b = self._images.dims_order[self._images.swaped_axes[1]+1]
#        data_shifted = np.swapaxes(data_shifted, axes_a, axes_b)
#        
#        return data_shifted
    
    def execute(self, namespace):
        try:
            del self._ft_images
            del self.image_cache
        except:
            pass
        
        import time
        import multiprocessing

#        from PYME.util import mProfile
        
        self._start_time = time.time()
        print("Starting drift correction module.")
        
        if self.multiprocessing:
            proccess_count = np.clip(multiprocessing.cpu_count()-1, 1, None)
            self._pool = multiprocessing.Pool(processes=proccess_count)
        
        ims = namespace[self.input_image]

#        try:
#            shift_max = 1. * self.shift_max / ims.mdh.voxelsize.x
#            if ims.mdh.voxelsize.units == "um":
#                self.shift_max = self.shift_max / 1E3
##            print("shift_max converted {:.2f} nm to {:.2f} pixel".format(self.shift_max, shift_max))
#        except:
#            print "Parsing metadata for voxel size failed. Shift_max will remain in units of pixels"
        shift_max = self.shift_max

#        mProfile.profileOn(['processing.py'])
        
#        print(ims.data)
#        print(ims.data.__class__)
        drift_res = self.calc_corr_drift_from_imagestack(ims)
        t_shift, shifts = self.rcc(shift_max,  *drift_res)
        
#        mProfile.profileOff()
#        mProfile.report()
            
        # convert frame-to-frame drift to drift from origin
        shifts = np.cumsum(shifts, 0)
        
        # image drift correction based on drift measured
#        drift_corrected_image = self.shift_images(shifts)
        
        del self._ft_images
#        del self.image_cache
        
        if self.multiprocessing:
            self._pool.close()
            self._pool.join()

#        out = ImageStack(drift_corrected_image.copy(), titleStub = self.outputName)
#        del drift_corrected_image
#        
#        out.mdh.copyEntriesFrom(ims.mdh)
#        out.mdh['Parent'] = ims.filename

#        namespace[self.outputName] = out
        
#        print shifts
        
        try:
            shifts[:, 0] *= ims.mdh.voxelsize.x
            shifts[:, 1] *= ims.mdh.voxelsize.y
            shifts[:, 2] *= ims.mdh.voxelsize.z
            
            if ims.mdh.voxelsize.units == 'um':
#                print('um units')
                shifts *= 1E3
        except:
            Warning("Failed at converting shifts to real distances")
        
#        print shifts
        
        if 'recipe.binning' in ims.mdh.keys():
#            print 'binning detected'
            t_shift = t_shift.astype(np.float)
            t_shift *= ims.mdh['recipe.binning'][2]
            t_shift += 0.5 * ims.mdh['recipe.binning'][2]
            
        namespace[self.output_drift] = t_shift, shifts
#        print shifts
        
        # non essential, only for plotting out drift data
        # converts figure object into an ImageStack to allow clicking in the GUI...
        fig = self.generate_drift_plot(t_shift, shifts)
        
        image_drift_shape = fig.canvas.get_width_height()
        fig.canvas.draw()
        image_drift = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image_drift = image_drift.reshape(image_drift_shape[1], image_drift_shape[0], 1, 3)
        image_drift = np.swapaxes(image_drift, 0, 1)
        namespace[self.output_drift_plot] = ImageStack(image_drift)
        
        namespace[self.output_cross_cor] = self._cc_image
        
        import gc
        gc.collect()
        
#@register_module('ShiftImage')
class ShiftImage(ModuleBase):
    """
        Performs FT based image shift.
        Currently only 2D. Shouldn't be too much work for 3D.
    """
    
    input_image = Input('input')
#    input_shift = Input('drift')
    input_drift_interpolator = Input('drift_interpolator')
    padding_multipler = Int(1)
    
#    ft_cache = File("ft_images.bin")
    image_cache = File("rcc_shifted_image_1.bin")
#    image_cache_2 = File("rcc_shifted_image_2.bin")
    outputName = Output('drift_corrected_image')
    
    def execute(self, namespace):
        self._start_time = time.time()
        try:
            del self._ft_images
            del self.image_cache
        except:
            pass
        
        ims = namespace[self.input_image]
        
#        #quick and dirty, assume dim 1 is t, ignore z, c
#        padding = np.stack((ims.data.shape[:-1],)*2, -1)
#        padding[-1,:] = 0
#        
#        padding *= self.padding_multipler
##        self._padding = padding        
#        
##        padded_image = np.pad(ims.data[:,:,:], padding, mode='constant')
##        images_shape = padded_image.shape
##        print images_shape
#        
#        dtype = ims.data[:,:,0].dtype
#        images_shape = np.asarray(ims.data.shape[:3], dtype=np.long) + padding.sum((1))
#        images_shape = tuple(images_shape)
##        print images_shape
#        padded_image = np.memmap(self.image_cache_2, dtype=dtype, mode='w+', shape=images_shape)
#        
#        chunk_size = 100000000 / ims.data.shape[0] / ims.data.shape[1] / dtype.itemsize
#        chunk_size = max(1, chunk_size)
##        print chunk_size
#        
#        for f in np.arange(0, images_shape[2], chunk_size):
#            padded_image[padding[0,0]:padding[0,0]+ims.data.shape[0],padding[1,0]:padding[1,0]+ims.data.shape[1],f:f+chunk_size] = ims.data[:,:,f:f+chunk_size].squeeze()
#            
##            print (f+1)
##            print images_shape[2]//5
#            progress = 0.2 * images_shape[2]
#            if (f+chunk_size > progress):
#                padded_image.flush()
#                progress += 0.2 * ims.data.shape[2]
#                print("{:.2f} s. Completed padding {} of {} total images.".format(time.time() - self._start_time, min(f+chunk_size, images_shape[2]), images_shape[2]))
#        
#        # quick and dirty coding, no optimizing
#        if self.ft_cache == "":
#            ft_images = np.zeros(images_shape, dtype=np.complex)
#        else:
#            ft_images = np.memmap(self.ft_cache, dtype=np.complex, mode='w+', shape=images_shape)
#            
#        for i in np.arange(images_shape[2]):
#            ft_images[:,:,i] = np.fft.fftn(padded_image[:,:,i])
#            
#            if ((i+1) % (images_shape[2]//5) == 0):
#                ft_images.flush()
#                print("{:.2f} s. Completed fft {} of {} total images.".format(time.time() - self._start_time, i+1, images_shape[2]))
#            
#        self._ft_images = ft_images

#        shifts_t, shifts_xyz = namespace[self.input_shift]
#        print shifts_t
#        print shifts_xyz

        t_out = np.arange(ims.data.shape[2], dtype=np.float)
        
        if 'recipe.binning' in ims.mdh.keys():
            t_out *= ims.mdh['recipe.binning'][2]
            t_out += 0.5*ims.mdh['recipe.binning'][2]
#        print
#        print shifts_xyz
#        print t_out
#        print shifts_t
        # linear interpolate
#        dx = np.interp(t_out, shifts_t, shifts_xyz[:, 0])
#        dy = np.interp(t_out, shifts_t, shifts_xyz[:, 1])
#        print dx.shape
#        print shifts_xyz[:, 0]
#        print dx

        dx = namespace[self.input_drift_interpolator][0](t_out)
        dy = namespace[self.input_drift_interpolator][1](t_out)
        
        shifted_images = self.shift_images(ims, np.stack([dx, dy], 1), ims.mdh)
#        shifted_image = shifted_image[padding[0,0]:padding[0,0]+ims.data.shape[0],padding[1,0]:padding[1,0]+ims.data.shape[1], :]
#        print padding
#        print shifted_image.shape
        
        namespace[self.outputName] = ImageStack(shifted_images, titleStub = self.outputName, mdh=ims.mdh)
            
    def shift_images(self, ims, shifts, mdh):
        
        padding = np.stack((ims.data.shape[:2],)*2, -1) #2d only
        padding *= self.padding_multipler
        
        padded_image_shape = np.asarray(ims.data.shape[:2], dtype=np.long) + padding.sum((1))
        
        dtype = ims.data[:,:,0].dtype
        padded_image = np.zeros(padded_image_shape, dtype=dtype)
        
#        raw_shifts = shifts[:, np.argsort(self._images.dims_order[1:])]        
        
        kx = (np.fft.fftfreq(padded_image_shape[0])) 
        ky = (np.fft.fftfreq(padded_image_shape[1]))
#        kz = (np.fft.fftfreq(self._ft_images.shape[3])) * 0.5
#        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        
        images_shape = np.asarray(ims.data.shape[:3], dtype=np.long)
        images_shape = tuple(images_shape)
        
        if self.image_cache == "":
            shifted_images = np.empty(images_shape)
        else:
            shifted_images = np.memmap(self.image_cache, dtype=np.float, mode='w+', shape=images_shape)
        
#        if self.multiprocessing:
#
#            none_args = ((None), ) * self._ft_images.shape[0]
#            fft_cache_args = ((self.ft_cache, self._ft_images.dtype, self._ft_images.shape), ) * self._ft_images.shape[0]
#            image_cache_args = ((self.image_cache, data_shifted.dtype, data_shifted.shape), ) * self._ft_images.shape[0]
#            args = zip(none_args, raw_shifts, np.arange(self._ft_images.shape[0]), fft_cache_args, image_cache_args)
#            
#            for i, (j, res) in enumerate(self._pool.imap_unordered(shift_image_helper, args)):
#                if self.image_cache == "":
#                    data_shifted[j] = res
#                    
#                if ((i+1) % (self._ft_images.shape[0]//5) == 0):
#                    print("{:.2f} s. Completed shifting {} of {} total images.".format(time.time() - self._start_time, i+1, self._ft_images.shape[0]))
#        else:
            
#        print(self._ft_images.shape)
#        print(shifts.shape)
#        print(kx.shape, ky.shape)
        
#        print shifts
        shifts_in_pixels = np.copy(shifts)
        
        try:
            shifts_in_pixels[:, 0] = shifts[:, 0] / mdh.voxelsize.x
            shifts_in_pixels[:, 1] = shifts[:, 1] / mdh.voxelsize.y
#            shifts_in_pixels[:, 2] = shifts[:, 2] / mdh.voxelsize.z
            
#            shifts_in_pixels[np.isnan(shifts_in_pixels)] = 0
            
#            print mdh
            if mdh.voxelsize.units == 'um':
#                print('um units')
                shifts_in_pixels /= 1E3
        except Exception as e:
            Warning("Failed at converting drift in pixels to real distances")
            repr(e)
        
#        print shifts_in_pixels
        

            
        for i in np.arange(ims.data.shape[2]):
#            print i
            
            padded_image[padding[0,0]:padding[0,0]+ims.data.shape[0],padding[1,0]:padding[1,0]+ims.data.shape[1]] = ims.data[:,:,i].squeeze()
            
            ft_image = np.fft.fftn(padded_image)
            
            data_shifted = shift_image_direct_rough(ft_image, shifts_in_pixels[i], kxy=(kx, ky))
            
            shifted_images[:,:,i] = data_shifted[padding[0,0]:padding[0,0]+ims.data.shape[0],padding[1,0]:padding[1,0]+ims.data.shape[1]]
            
            if ((i+1) % (shifted_images.shape[-1]//5) == 0):
                shifted_images.flush()
                print("{:.2f} s. Completed shifting {} of {} total images.".format(time.time() - self._start_time, i+1, shifted_images.shape[-1]))
            
#        data_shifted = np.moveaxis(data_shifted, 0, self._images.dims_order[0])
        
#        axes_a = self._images.dims_order[self._images.swaped_axes[0]+1]
#        axes_b = self._images.dims_order[self._images.swaped_axes[1]+1]
#        data_shifted = np.swapaxes(data_shifted, axes_a, axes_b)
        
        return shifted_images
    
def shift_image_direct_rough(source_ft, shifts, kxy=None):
    if kxy is None:
        kx = np.fft.fftfreq(source_ft.shape[0])
        ky = np.fft.fftfreq(source_ft.shape[1])
#        kz = np.fft.fftfreq(source_ft.shape[2]) * 0.5
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
    else:
        kx, ky = kxy
    
#    print(source_ft.dtype)
#    print(kx.dtype)
#    print(ky.dtype)
#    print(shifts.dtype)
    return np.abs(np.fft.ifftn(source_ft*np.exp(-2j*np.pi*(kx*shifts[0] + ky*shifts[1]))))