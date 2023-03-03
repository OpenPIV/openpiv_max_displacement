# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:42:13 2019

@author: Theo
"""

# add two directories that include the new files
# note that we need to import openpiv in a separate, original namespace
# so we can use everything from openpiv as openpiv.filters and whatever is 
# going to replace it will be just filteers (for example)

import os
import sys

sys.path.append(os.path.abspath('./openpiv'))
print(sys.path)

from OpenPIV_windef_func import PIV_windef


class Settings(object):
    pass


settings = Settings()

# settingss sounds more like a dictionary where you have keys (parameter names)  and values


settings.filepath_images = './Pre_Pro_PIV_IMAGES/'
settings.save_path = './Results_PIV/'
settings.save_folder_suffix = 'Test'
settings.frame_pattern_a = 'A*a.tif'
settings.frame_pattern_b = 'A*b.tif'
settings.ROI = 'full'  # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
settings.dynamic_masking_method = 'None'  # 'None' for no masking, 'edges' for edges masking, 'intensity' for intensity masking
settings.dynamic_masking_threshold = 0.005
settings.dynamic_masking_filter_size = 7
settings.iterations = 2
settings.windowsizes = (64, 32, 16)
settings.overlap = (32, 16, 8)
settings.interpolation_order = 3  # order of the image interpolation for the window deformation
settings.scaling_factor = 1000 / 0.045  # scaling pixel ->meter
settings.dt = 1.2e-6  # time between to frames (in seconds)
settings.subpixel_method = 'gaussian'
settings.extract_sig2noise = True  # 'True' or 'False' only for the first part
settings.sig2noise_method = 'peak2mean'  # peak2peak or peak2mean
settings.sig2noise_width = 2
settings.validation_first_pass = True  # 'True' or 'False'
settings.MinMax_U_vel = (-60, 30)  # global validation by minimum and maximum velocity
settings.MinMax_V_vel = (-30, 30)
settings.std_threshold = 10  # std validation
settings.median_threshold = 2  # median validation
settings.sig2noise_threshold = 0  # sig2noise validation only for the first pass
settings.filter_method = 'localmean'
settings.max_filter_iteration = 10
settings.filter_kernel_size = 2
settings.scale_plot = 4000

# run the script with the given settingss
PIV_windef(settings)
