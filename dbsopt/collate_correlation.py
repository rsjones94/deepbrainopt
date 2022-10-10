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

import interchange_funcs as icf



parent_folder = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/voltage_maps/tractographic_activation'
conditions_file = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/x010/spectmean_clean.xlsx'


###
correlations_folder = os.path.join(parent_folder, 'correlation')
if not os.path.exists(correlations_folder):
    os.mkdir(correlations_folder)

act_globber = os.path.join(parent_folder, '*.xlsx')
activation_files = glob(act_globber)

def pwz(li):
    #pairwise zip
    return zip(li[:-1], li[1:])


def pwz_df(df):
    #pairwise zip but for a df
    return zip(df[:-1].iterrows(),df[1:].iterrows())

    
cmap = matplotlib.cm.get_cmap('coolwarm')
norm = mpl.colors.Normalize(vmin=-0.015, vmax=0.015)
conditions_df = pd.read_excel(conditions_file)

out_df_name = os.path.join(correlations_folder, 'raw.xlsx')
all_rows = []
for activation_file in activation_files:
    
    structure_name = os.path.basename(activation_file)[:-5]
    print(f'Structure: {structure_name}')
    
    in_df = pd.read_excel(activation_file, index_col=0)
    in_df = in_df.dropna() # drop the rows that don't have activation function (due numerical derivative losses)

    for i,cond_row in conditions_df.iterrows():
        conditions = str(cond_row['active_contacts']).split(',')
        print(f'\tGenerating {conditions}')
        
        cmod = cond_row['bounds']
        condition_usecols = [f'contact_{c}_activation' for c in conditions]
        condition_activation = in_df[condition_usecols].sum(axis=1) * cmod
        
        cond_join = f"condition_{''.join(conditions)}"
        cond_col = f'{cond_join}_activation'
        
        cond_text = ','.join(conditions)
        
    
        #mean_act = np.mean(condition_activation)
        mean_act = np.mean(condition_activation)
        median_act = np.median(condition_activation)
        topdecile_act = np.percentile(condition_activation,90)
        bottomdecile_act = np.percentile(condition_activation,10)
        std_act = np.std(condition_activation)
        absmean_act = np.mean(np.abs(condition_activation))
        absmedian_act = np.median(np.abs(condition_activation))
        
        highb = cond_row['high_beta']
        lowb = cond_row['low_beta']


        data = [structure_name,
                cond_text,
                highb,
                lowb,
                mean_act,
                median_act,
                topdecile_act,
                bottomdecile_act,
                std_act,
                absmean_act,
                absmedian_act]
        
        data_names = ['structure',
                    'condition',
                    'high_beta',
                    'low_beta',
                    'mean_act',
                    'median_act',
                    'topdecile_act',
                    'bottomdecile_act',
                    'std_act',
                    'absmean_act',
                    'absmedian_act']
        
        data_dict = {key:val for key,val in zip(data_names,data)}

        row = pd.Series(data_dict)
        all_rows.append(row)
        
ultra_df = pd.concat(all_rows, axis=1).T
ultra_df.to_excel(out_df_name)
        
tendencies = data_names[4:]
for te in tendencies:
    for us in ultra_df['structure'].unique():
        plot_outname = os.path.join(correlations_folder, f'{te}_{us}.pdf')
        fig, axs = plt.subplots(2,1,figsize=(6,8))
        
        sub = ultra_df[ultra_df['structure']==us]
        exes = sub[te]
        
        why_cols = ['high_beta','low_beta']
        for i, (ax,yc) in enumerate(zip(axs,why_cols)):
            whys = sub[yc]
            ax.scatter(exes,whys,ec='black',alpha=0.5,label=us)
            ax.set_ylabel(yc)
            ax.set_xlabel(te)
            ax.legend()
            
            #if i == 0:
            #    ax.set_title(te)
        
        fig.tight_layout()
        fig.show()
        fig.savefig(plot_outname)
        plt.close(fig)
        
        