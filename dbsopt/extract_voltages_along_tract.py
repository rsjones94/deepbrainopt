#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""


import sys
import os
from glob import glob

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from scipy import spatial
from scipy import interpolate
import scipy.io
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt

import interchange_funcs as icf



def euc(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    d = np.sqrt(np.sum((p2-p1)**2))
    return d


def dists_from_origin(pts):
    starts = pts[:-1]
    stops = pts[1:]
    
    inter_dists = [0]
    inter_dists.extend([euc(start,stop) for start,stop in zip(starts,stops)])
    
    total_dists = [np.sum(inter_dists[:i+1]) for i in range(len(inter_dists))]
    
    return np.array(total_dists)


def interp(exes,whys,step=0.25):
    f = interpolate.interp1d(exes, whys, kind='linear')
    exes_new = np.arange(min(exes), max(exes), step=step)
    whys_new = f(exes_new)
    
    return(exes_new, whys_new)
    

def dydx(whys,exes):
    res = np.diff(whys) / np.diff(exes)
    
    return res


def pwz(li):
    #pairwise zip
    return zip(li[:-1], li[1:])

vmap_folder = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/voltage_maps'
tract_folder = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/atlases/dbs_tractography_atlas_middlebrooks_2020'
resolution = (0.5, 0.5, 0.5)
mni_starts = (-98,-134,-72)

vmap_globber = os.path.join(vmap_folder, '*.nii.gz')
vmap_files = glob(vmap_globber)

tract_globber = os.path.join(tract_folder, '*/', '*.mat')
tract_files = glob(tract_globber)

data_out_folder = os.path.join(vmap_folder, 'tractographic_activation')
if not os.path.exists(data_out_folder):
    os.mkdir(data_out_folder)

for tn, tract_f in enumerate(tract_files):
    
    
    side = os.path.basename(os.path.dirname(tract_f))
    tract = os.path.basename(tract_f)[:-4]
    
    
    dataframe_outname = os.path.join(data_out_folder, f'{side}_{tract}.xlsx')
    if os.path.exists(dataframe_outname):
        print(f'\t\t\t{tract} ({side}) exists. Skipping....')
        continue
    
    
    print(f'On {tract} ({side}), {tn+1} of {len(tract_files)}')
    
    tracts = scipy.io.loadmat(tract_f)
    fibers_coords = tracts['fibers']
    master_df = pd.DataFrame(fibers_coords, columns=['x_coord', 'y_coord', 'z_coord',
                                                     'fiber_number'])
    
    master_df['fiber_number'] = master_df['fiber_number'].astype(int)
    
    
    fibers_indices = fibers_coords.copy()[:,:-1]
    for i,row in enumerate(fibers_indices):
        for j,val in enumerate(row):
            fibers_indices[i,j] = icf.float_to_index(val,resolution[j],mni_starts[j])

    master_df[['x_ind', 'y_ind', 'z_ind']] = fibers_indices.astype(int)
    
    all_subs = []
    unique_fibers = np.unique(master_df['fiber_number'])
    for uf in unique_fibers:
        print(f'\tOn fiber {uf} of {len(unique_fibers)}')
        sub_df = master_df[master_df['fiber_number']==uf].copy()
        dfo = dists_from_origin(np.array(sub_df[['x_coord', 'y_coord', 'z_coord']]))
        sub_df['distance_from_origin'] = dfo
                                       
        for vn, vmap_f in enumerate(vmap_files):
            voltage_im = nib.load(vmap_f)
            voltage_data = voltage_im.get_fdata()
            
            contact_name = '_'.join(os.path.basename(vmap_f).split('_')[1:])[:-7]
            print(f'\t\t({contact_name})')
            
            contact_voltage_name = f'{contact_name}_voltage'
            contact_field_name = f'{contact_name}_field'
            contact_activation_name = f'{contact_name}_activation'
            
            sub_df[contact_voltage_name] = np.nan
            
            for rn, row in sub_df.iterrows():
                xi = int(row['x_ind'])
                yi = int(row['y_ind'])
                zi = int(row['z_ind'])
                
                voltage = voltage_data[xi,yi,zi]
                sub_df.at[rn,contact_voltage_name] = voltage
                
            field = list(dydx(sub_df[contact_voltage_name],
                         sub_df['distance_from_origin']))
            
            activation = list(dydx(field,
                              sub_df['distance_from_origin'][1:]))
            
            field.append(np.nan)
            
            activation.append(np.nan)
            activation.append(np.nan)
            
            sub_df[contact_field_name] = field
            sub_df[contact_activation_name] = activation
        all_subs.append(sub_df)

    ultra_df = pd.concat(all_subs)
    ultra_df.to_excel(dataframe_outname)
    
        
    
### cumulative activation function?
### find centerline projection of all tracts
### project each track's activation onto centerline and sum
        
        
        
        
        
        
        
        