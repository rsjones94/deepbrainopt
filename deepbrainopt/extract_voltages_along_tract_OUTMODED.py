#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""


import sys

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


    
 

voltage_file = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/repositories/dbsopt/assets/voltage_maps/mni_p05mm.mat'
streamfile = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/repositories/dbsopt/assets/tractography_samples/hcp842/tracks/commisural/PC.trk.gz'


voltage_mat = scipy.io.loadmat(voltage_file)
voltage_vals, voltage_points = icf.unpack_mat_data(voltage_mat)



trak = nib.streamlines.load(streamfile)
streamlines = trak.streamlines

results = []

ax = plt.axes(projection='3d')
cmap = matplotlib.cm.get_cmap('Spectral')
for ix, stream in enumerate(streamlines):
    print(f'On stream {ix} of {len(streamlines)}')
    # if all we want is the mean 2nd deriv, we can just use 3 points
    mid_index = int(len(stream)/2)
    positions = dists_from_origin(stream)
    
    cut_stream = np.array([stream[0], stream[mid_index], stream[-1]])
    cut_positions = np.array([positions[0], positions[mid_index], positions[-1]])
    '''
    #int_coords = np.round(stream,0).astype(int)
    voltages = [voltage_vals[icf.closest_to((x,y,z), voltage_points)[0]].real[0] for x,y,z in cut_stream]
    
    # these are numerical derivatives
    deriv_first = dydx(voltages, cut_positions)
    # we will center on the first point, even if that's slightly inaccurate
    deriv_second = dydx(deriv_first, cut_positions[:-1])
    
    result = deriv_second.mean()
    '''
    result = np.random.random()
    results.append(result)
    rbga = cmap(result)
    
    exes = stream[:,0]
    whys = stream[:,1]
    zees = stream[:,2]
    ax.plot3D(exes, whys, zees, c=rbga)

plt.show()
    
    
    
    
    
    
    
