# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:15:07 2019

@author: kkc29
"""

from PYME.recipes.base import ModuleBase, register_module, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, File
from PYME.IO.image import ImageStack

@register_module('Preprocessing')
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
    tukey_size = Float(0.25)
    cache = File("clip_cache.bin")
    output_name = Output('clipped_images')