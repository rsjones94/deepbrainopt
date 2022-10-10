
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io

import interchange_funcs as ifc


'''

['__header__',
 '__version__',
 '__globals__',
 'ea_fibformat',
 'fibers',
 'fourindex',
 'idx']

'''

vmap_folder = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/voltage_maps'

tract_file = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/repositories/dbsopt/assets/dbs_tractography_atlas_middlebrooks_2020/lh/STN_associative_tract.mat'
tract_mat = scipy.io.loadmat(tract_file)


glob