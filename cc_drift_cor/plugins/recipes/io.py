# -*- coding: utf-8 -*-
"""
Created on Tue Apr 02 14:22:15 2019

@author: kkc29


based off PYME recipes processing.py
"""



from PYME.recipes.base import ModuleBase, register_module, Filter, OutputModule
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, File
from PYME.recipes import processing
from PYME.recipes.graphing import Plot

import numpy as np
from scipy import ndimage, interpolate
from PYME.IO.image import ImageStack

from functools import partial

import logging
logger=logging.getLogger(__name__)

#class DriftCombine(ModuleBase):
#    input_drift = Input("drift")
#    input_drift1 = Input("drift1")
#    
#    save_path = File('')
#    
#    output_drift_combined = Output('drift_combined')
#    
#    def execute(self, namespace):
#        tIndex, drift = namespace[self.input_drifts]
#        
#        for i in range(10):
#            drift_item = getattr(self, "drift{}".format(i+1))
#            print (drift_item)
#            if drift_item != "":
#                drift_t, drift_val = drift_item
#                assert np.allclose(tIndex, drift_t), 'different time indexes'
#                
#                drift += drift_val
#                
#        namespace[self.output_drift_combined] = (tIndex, drift)
#        
#        if self.save_path != "":
#            np.savez_compressed(self.save_path, tIndex=tIndex, drift=drift)
#            print('saved')
        
            
#@register_module('SaveDrift')
class DriftOutput(ModuleBase):
    """
    Save drift data to a file.
        
    Inputs
    ------
    input_name : 
        Drift measured from localization or image dataset.
    
    Outputs
    -------
    output_dummy : None
        Blank output. Required to run correctly.
        
    Parameters
    ----------
    save_path : File
        Filepath to save drift data.    
    """
    
    input_name= Input('drift')
    save_path = File('drift')
    output_dummy = Output('dummy') # will not run execute without this
    
    def execute(self, namespace, context={}):
#        out_filename = self.filePattern.format(**context)
        out_filename = self.save_path
        
        tIndex, drift = namespace[self.input_name]
        
        np.savez_compressed(out_filename, tIndex=tIndex, drift=drift)
        print('saved')

#@register_module('LoadDrift')
class LoadDrift(ModuleBase):
    """
    *Deprecated.*   Use ``LoadDriftandInterp`` instead.
    
    Load drift from a file.
    """
    
    input_dummy = Input('input') # breaks GUI without this???
    load_path = File()
    output_drift_raw= Input('drift_raw')
    output_drift_plot = Output('drift_plot')
    
    def execute(self, namespace):
        data = np.load(self.load_path)
        tIndex = data['tIndex']
        drift = data['drift']
        namespace[self.output_drift_raw] = (tIndex, drift)
        
        # non essential, only for plotting out drift data
        namespace[self.output_drift_plot] = Plot(partial(generate_drift_plot, tIndex, drift))

        
#@register_module('InterpolateDrift')
class InterpolateDrift(ModuleBase):
    """
    Creates a spline interpolator from drift data. (``scipy.interpolate.UnivariateSpline``)
        
    Inputs
    ------
    input_drift_raw : Tuple of arrays
        Drift measured from localization or image dataset.
    
    Outputs
    -------
    output_drift_interpolator :
        Drift interpolator. Returns drift when called with frame number / time.
    output_drift_plot : Plot
        Plot of the original and interpolated drift.
        
    Parameters
    ----------
    degree_of_spline : int
        Degree of the smoothing spline.
    smoothing_factor : float
        Smoothing factor.
    """
    
#    input_dummy = Input('input') # breaks GUI without this???
#    load_path = File()
    degree_of_spline = Int(3) # 1 for linear, 3 for cubic
    smoothing_factor = Float(-1) # 0 for no smoothing. set to negative for UnivariateSpline defulat
    input_drift_raw = Input('drift_raw')
    output_drift_interpolator= Output('drift_interpolator')
    output_drift_plot = Output('drift_plot')
    
    def execute(self, namespace):
#        data = np.load(self.load_path)
#        tIndex = data['tIndex']
#        drift = data['drift']
#        namespace[self.output_drift_raw] = (tIndex, drift)
        tIndex, drift = namespace[self.input_drift_raw]
        
        spl = interpolate_drift(tIndex, drift, self.degree_of_spline, self.smoothing_factor)
        
        namespace[self.output_drift_interpolator] = spl
        
#        # non essential, only for plotting out drift data
        namespace[self.output_drift_plot] = Plot(partial(generate_drift_plot, tIndex, drift, spl))
        namespace[self.output_drift_plot].plot()
        
def interpolate_drift(tIndex, drift, degree_of_spline, smoothing_factor):
    if smoothing_factor < 0:
        smoothing_factor = None
                
    spl = list()
    for i in range(drift.shape[1]):
        spl_1d = interpolate.UnivariateSpline(tIndex, drift[:,i], k=degree_of_spline, s=smoothing_factor)
        spl.append(spl_1d)
    
    return spl


class LoadDriftandInterp(ModuleBase):
    """
    Loads drift data from file(s) and use them to create a spline interpolator (``scipy.interpolate.UnivariateSpline``).
        
    Inputs
    ------
    input_dummy : None
       Blank input. Required to run correctly.
    
    Outputs
    -------
    output_drift_interpolator :
        Drift interpolator. Returns drift when called with frame number / time.
    output_drift_plot : Plot
        Plot of the original and interpolated drift.
        
    Parameters
    ----------
    load_paths : list of File
        List of files to load.
    degree_of_spline : int
        Degree of the smoothing spline.
    smoothing_factor : float
        Smoothing factor.
    """
    
    input_dummy = Input('input') # breaks GUI without this???
#    load_path = File()
    load_paths = List(File, [""], 1)
    degree_of_spline = Int(3) # 1 for linear, 3 for cubic
    smoothing_factor = Float(-1) # 0 for no smoothing. set to negative for UnivariateSpline defulat
#    input_drift_raw = Input('drift_raw')
    output_drift_interpolator= Output('drift_interpolator')
    output_drift_plot = Output('drift_plot')
#    output_drift_raw= Input('drift_raw')
    
    def execute(self, namespace):
        spl_array = list()
        t_min = np.inf
        t_max = 0
        tIndexes = list()
        drifts = list()
        for fil in self.load_paths:            
            data = np.load(fil)
            tIndex = data['tIndex']
            t_min = min(t_min, tIndex[0])
            t_max = max(t_max, tIndex[-1])
            drift = data['drift']
            
            tIndexes.append(tIndex)
            drifts.append(drift)
                
            spl = interpolate_drift(tIndex, drift, self.degree_of_spline, self.smoothing_factor)
            spl_array.append(spl)

#        print(len(spl_array))
#        print(spl_array[0])
        spl_array = zip(*spl_array)
#        print(len(spl_final))
#        print(spl_final[0])
        def spl_method(funcs, t):
            return np.sum([f(t) for f in funcs], axis=0)
        
        spl_combined = list()
        for spl in spl_array:
#            print(spl)
#            spl_combined.append(lambda x: np.sum([f(x) for f in spl], axis=0))
            spl_combined.append(partial(spl_method, spl))
        
        namespace[self.output_drift_interpolator] = spl_combined
        
        # non essential, only for plotting out drift data
        namespace[self.output_drift_plot] = Plot(partial(generate_drift_plot, tIndexes, drifts, spl_combined))
        namespace[self.output_drift_plot].plot()
        

def generate_drift_plot(t, shifts, interpolators=None):
    from matplotlib import pyplot

    if not isinstance(t, list):
        t = [t]
        shifts = [shifts]
    dims = max(s.shape[1] for s in shifts)
    dim_name = ['x', 'y', 'z']    
    
    t_min = np.asarray(t).min()
    t_max = np.asarray(t).max()
    step_size = max([(t_max - t_min + 1) // 100, 1])
    t_full = np.arange(t_min, t_max+1, step_size)
    
    n_row = np.ceil(dims*0.5).astype(int)
    n_col = min([dims, 2])
    fig = pyplot.figure(figsize=(n_col*4, n_row*3))
    
    for i in range(dims):
        ax = fig.add_subplot(n_row, n_col, i+1)
        for j in range(len(t)):
            ax.plot(t[j], shifts[j][:, i], marker='.', linestyle=None)
        
        if not interpolators is None:
            ax.plot(t_full, interpolators[i](t_full))
    
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Drift (nm)")
        ax.set_title(dim_name[i])
        ax.legend()

    fig.tight_layout()
    
    return fig         