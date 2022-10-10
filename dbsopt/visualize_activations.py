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
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import open3d as o3d

import interchange_funcs as icf



parent_folder = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/voltage_maps/tractographic_activation'
conditions_file = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/spectmean_clean.xlsx'
anatomy_file = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/export/ply/anatomy.ply'
lead_file = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/export/ply/left_electrode.ply'

n_max_fibers = 100
skip_existing = True

###
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

anat = np.asarray(o3d.io.read_point_cloud(anatomy_file).points) # Read the point cloud
n_anat = len(anat)
max_anat = 6000
spacer = n_anat/max_anat
anat_slicer = np.arange(0,n_anat-1,spacer).astype(int)
sparse_anat = anat[anat_slicer,:]

lead_pts = np.asarray(o3d.io.read_point_cloud(lead_file).points) # Read the point cloud
n_lead = len(lead_pts)
max_lead = 3000
spacer_lead = n_lead/max_lead
lead_slicer = np.arange(0,n_lead-1,spacer_lead).astype(int)
sparse_lead = lead_pts[lead_slicer,:]

act_globber = os.path.join(parent_folder, '*.xlsx')
activation_files = glob(act_globber)

def pwz(li):
    #pairwise zip
    return zip(li[:-1], li[1:])


def pwz_df(df):
    #pairwise zip but for a df
    return zip(df[:-1].iterrows(),df[1:].iterrows())

conditions_df = pd.read_excel(conditions_file)

view_theta = 30
view_phi = -45
for activation_file in activation_files:
    
    structure_name = os.path.basename(activation_file)[:-5]
    out_file = os.path.join(parent_folder, f'{structure_name}_activation.pdf')
    out_file_localized = os.path.join(parent_folder, f'{structure_name}_activation_localized.pdf')
    out_file_1d = os.path.join(parent_folder, f'{structure_name}_activation_1d.pdf')
    
    if all([os.path.exists(fi) for fi in [out_file, out_file_1d]]) and skip_existing:
        print(f'\tSkipping {structure_name}')
        continue
    
    
    print(f'Structure: {structure_name}')
    
    in_df = pd.read_excel(activation_file, index_col=0)
    in_df = in_df.dropna() # drop the rows that don't have activation function (due numerical derivative losses)

    
    
    activation_cols = [i for i in in_df.columns if 'activation' in i]
    
    #ac_col = activation_cols[0]con
    
    threedfig, threedaxs = plt.subplots(4,2,subplot_kw=dict(projection='3d'), figsize=(8,12))
    fig, axs = plt.subplots(4,2, figsize=(8,12))
    #cmap = matplotlib.cm.get_cmap('inferno')
    cmap = matplotlib.cm.get_cmap('coolwarm')
    
    # determine norm / limits
    acts = []
    for jj,cond_row in conditions_df.iterrows():
        conditions = str(cond_row['active_contacts']).split(',')
        cmod = cond_row['bounds']
        condition_cols = [f'contact_{c}_activation' for c in conditions]
        condition_activation = in_df[condition_cols].sum(axis=1) * cmod
        acts.append(condition_activation)
    acts = np.array(acts)
    #ranger = np.abs(np.array([np.percentile(acts, 10), np.percentile(acts, 90)]))
    ranger = np.abs([np.percentile(acts,0.025), np.percentile(acts,99.75)])
    ranger_max = np.max(ranger)
    vminner = -ranger_max
    vmaxxer = ranger_max
    
    norm_ranger = np.abs([np.percentile(acts,5), np.percentile(acts,95)])
    norm_ranger_max = np.max(norm_ranger)
    norm_vminner = -norm_ranger_max
    norm_vmaxxer = norm_ranger_max
    
    
    norm = mpl.colors.Normalize(vmin=norm_vminner, vmax=norm_vmaxxer)
    for threedax, ax, (ii,cond_row) in zip(np.ravel(threedaxs), np.ravel(axs), conditions_df.iterrows()):

        
        conditions = str(cond_row['active_contacts']).split(',')
        cmod = cond_row['bounds']
        
        condition_cols = [f'contact_{c}_activation' for c in conditions]
        condition_activation = in_df[condition_cols].sum(axis=1) * cmod
        in_df['condition_activation'] = condition_activation
        print(f'\tPlotting {conditions}')
        
        highb = cond_row['high_beta']
        lowb = cond_row['low_beta']
        title = f"Condition {', '.join(conditions)}\nhigh/low {chr(946)} = {round(highb,2), round(lowb,2)}"
        
        if not os.path.exists(out_file):
            threedax.set_title(title)
            '''
            threedax.scatter3D(sparse_lead[:,0],sparse_lead[:,1],sparse_lead[:,2],
                               c='green',alpha=0.1,zorder=8,s=1)
            threedax.scatter3D(sparse_anat[:,0],sparse_anat[:,1],sparse_anat[:,2],
                               c='black',alpha=0.1,zorder=5,s=1)
            '''
        
        unique_fibers = in_df['fiber_number'].unique()
        n_fibs = unique_fibers.max()
        if n_fibs > n_max_fibers:
            spacing = n_fibs / n_max_fibers
            fiber_nums = np.arange(0,n_fibs,spacing).astype(int)
        else:
            fiber_nums = unique_fibers
        for fn in fiber_nums:
            sub_df = in_df[in_df['fiber_number']==fn]
            
            if not os.path.exists(out_file):
                for start,stop in pwz_df(sub_df):
                    exes = [start[1]['x_coord'], stop[1]['x_coord']]
                    whys = [start[1]['y_coord'], stop[1]['y_coord']]
                    zees = [start[1]['z_coord'], stop[1]['z_coord']]
                    fun = start[1]['condition_activation']
                    
                    lii,=threedax.plot3D(exes, whys, zees, c=cmap(norm(fun)), lw=1.5, alpha=0.3,zorder=10)
                    lii.set_solid_capstyle('round')
                
            if not os.path.exists(out_file_1d):
                dfo = sub_df['distance_from_origin']
                funs = sub_df['condition_activation']
                ax.plot(dfo,
                        funs,
                        c='black',
                        alpha=1/len(fiber_nums)*8,
                        lw=0.5)
                ax.set_title(title)
                ax.set_xlabel('Distance from origin (mm)')
                ax.set_ylabel('Activation (V/$m^2$)')
                ax.set_ylim(round(vminner,4), round(vmaxxer,4))
                
        threedax.view_init(elev=view_theta, azim=view_phi)
        threedax.set_box_aspect((np.ptp(sub_df['x_coord']),
                                 np.ptp(sub_df['y_coord']),
                                 np.ptp(sub_df['z_coord'])))
    
    if not os.path.exists(out_file):
        threedfig.show()
        threedfig.tight_layout()
        np.ravel(threedaxs)[-1].axis('off')
        threedfig.savefig(out_file)
        
        for threedax in np.ravel(threedaxs)[:-1]:
            threedax.scatter3D(sparse_lead[:,0],sparse_lead[:,1],sparse_lead[:,2],
                               c='green',alpha=0.1,zorder=8,s=0.7)
            threedax.scatter3D(sparse_anat[:,0],sparse_anat[:,1],sparse_anat[:,2],
                               c='black',alpha=0.1,zorder=5,s=0.7)
            
            threedax.view_init(elev=view_theta, azim=view_phi)
            threedax.set_box_aspect((np.ptp(sparse_anat[:,0]),
                                     np.ptp(sparse_anat[:,1]),
                                     np.ptp(sparse_anat[:,2])))
        threedfig.savefig(out_file_localized)
        
    plt.close(threedfig)
       
    if not os.path.exists(out_file_1d):
        fig.show()
        fig.tight_layout()
        np.ravel(axs)[-1].axis('off')
        fig.savefig(out_file_1d)
    plt.close(fig)
        
    
    
    
    
        
        
        