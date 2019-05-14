# -*- coding: utf-8 -*-
"""
Created on Tue Apr 02 14:22:15 2019

@author: kkc29


based off PYME recipes processing.py
"""



from PYME.recipes.base import ModuleBase, register_module, Filter, OutputModule
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, File

import numpy as np
from scipy import ndimage, interpolate
from PYME.IO.image import ImageStack

from PYME.recipes import processing
import Kenny

import logging
logger=logging.getLogger(__name__)

#@register_module('SaveDrift')
class DriftOutput(ModuleBase):
    """
    Save drift as numpy array file. Expects tuple of 2 arrays, one for frame number (N) and one for drift (Nx3).
    No error checks. Writes to file with key 'tIndex' and 'drift'
    Not derived from OutputModule. Will save on execution.
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
        Load drift from numpy array file. Expects numpy object containing 2 arrays, one for frame number ('tIndex', of size N) and one for drift ( 'drift' of size Nx3).
        No error checks.
        Interpolation with scipy.interpolate.UnivariateSpline.
        Returns a tuple: (tIndex, drift)
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
        fig = self.generate_drift_plot(tIndex, drift)
        
        image_drift_shape = fig.canvas.get_width_height()
        fig.canvas.draw()
        image_drift = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image_drift = image_drift.reshape(image_drift_shape[1], image_drift_shape[0], 1, 3)
        image_drift = np.swapaxes(image_drift, 0, 1)
        namespace[self.output_drift_plot] = ImageStack(image_drift)
        
    def generate_drift_plot(self, t, shifts):
        from matplotlib import pyplot
        pyplot.ioff()
        fig, ax = pyplot.subplots(1, 1)
        lines = ax.plot(t, shifts, marker='.', linestyle=None)
        
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Drift (nm)")
        ax.legend(lines, ['x', 'y', 'z'][:shifts.shape[1]])
        fig.tight_layout()
        pyplot.ion()
        
        return fig
        
#@register_module('InterpolateDrift')
class InterpolateDrift(ModuleBase):
    """
        Return interpolator objects (1 per dim) created from 'raw' drift data.
        No error checking for extrapolation.
    """    
    
#    input_dummy = Input('input') # breaks GUI without this???
#    load_path = File()
    degree_of_spline = Int(3) # 1 for linear, 3 for cubic
    smoothing_factor = Float(-1) # 0 for no smoothing. set to negative for UnivariateSpline defulat
    input_drift_raw = Input('drift_raw')
    output_drift_interpolated= Output('drift_interpolated')
    output_drift_interpolator= Output('drift_interpolator')
    output_drift_plot = Output('drift_plot')
    
    def execute(self, namespace):
#        data = np.load(self.load_path)
#        tIndex = data['tIndex']
#        drift = data['drift']
#        namespace[self.output_drift_raw] = (tIndex, drift)
        tIndex, drift = namespace[self.input_drift_raw]
        
        if self.smoothing_factor < 0:
            smoothing_factor = None
        else:
            smoothing_factor = self.smoothing_factor
        
        t_full = np.arange(tIndex.min(), tIndex.max()+1)
        drift_full = np.empty((t_full.shape[0], drift.shape[1]))
        spl = list()
        for i in xrange(drift.shape[1]):
            spl_1d = interpolate.UnivariateSpline(tIndex, drift[:,i], k=self.degree_of_spline, s=smoothing_factor)
            drift_full[:, i] = spl_1d(t_full)
            spl.append(spl_1d)
            
        namespace[self.output_drift_interpolated] = (t_full, drift_full)
        namespace[self.output_drift_interpolator] = spl
        
        # non essential, only for plotting out drift data
        fig = self.generate_drift_plot(tIndex, drift, t_full, drift_full)
        
        image_drift_shape = fig.canvas.get_width_height()
        fig.canvas.draw()
        image_drift = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image_drift = image_drift.reshape(image_drift_shape[1], image_drift_shape[0], 1, 3)
        image_drift = np.swapaxes(image_drift, 0, 1)
        namespace[self.output_drift_plot] = ImageStack(image_drift)
        
    def generate_drift_plot(self, t, shifts, t_full, shifts_full):
        from matplotlib import pyplot
        pyplot.ioff()
        fig, ax = pyplot.subplots(1, 1)
        lines = ax.plot(t, shifts, marker='.', linestyle=None)

        ax.plot(t_full, shifts_full)
        
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Drift (nm)")
        ax.legend(lines, ['x', 'y', 'z'][:shifts.shape[1]])
        fig.tight_layout()
        pyplot.ion()
        
        return fig
         