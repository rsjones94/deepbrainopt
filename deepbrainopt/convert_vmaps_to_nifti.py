
from glob import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io

import interchange_funcs as ifc



vmap_folder = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/voltage_maps'
template_image = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/repositories/dbsopt/assets/mni_icbm_2009b_nlin_asym/t1.nii'

resolution = (0.5, 0.5, 0.5)

####

globber = os.path.join(vmap_folder, '*.mat')
files = glob(globber)

for f in files:
    bname = os.path.basename(os.path.normpath(f))
    print(f'Converting {bname}')
    ifc.s4lmat_to_nifti(f, template_image, outname=None, res=resolution)