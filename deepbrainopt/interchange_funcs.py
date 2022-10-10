
import os
import itertools

import nibabel as nib
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


mat_file = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/repositories/dbsopt/assets/voltage_maps/mni_p05mm.mat'
template = 'C:/Users/jonesr36/OneDrive - Cleveland Clinic/Documents/repositories/dbsopt/assets/mni152/MNI152_T1_1mm_brain.nii.gz'
res = (0.5, 0.5, 0.5)

def mat_to_matrix(mat, real_only=True):
    data = mat['Snapshot0']
    exes = mat['Axis0'][0]
    whys = mat['Axis1'][0]
    zees = mat['Axis2'][0]
    
    nx = len(exes)-1 
    ny = len(whys)-1
    nz = len(zees)-1
    
    if real_only:
        use_data = data.real
    else:
        use_data = data
    
    matrix = np.reshape(use_data, (nx,ny,nz), order='F')
    matrix = np.flip(matrix, axis=1)
    matrix = np.flip(matrix, axis=0)
    
    return matrix
    

def rescale_affine(input_affine, voxel_dims=[1, 1, 1], target_center_coords= None):
    """
    This function uses a generic approach to rescaling an affine to arbitrary
    voxel dimensions. It allows for affines with off-diagonal elements by
    decomposing the affine matrix into u,s,v (or rather the numpy equivalents)
    and applying the scaling to the scaling matrix (s).
    
    Courtesy leej3 on the nipy github

    Parameters
    ----------
    input_affine : np.array of shape 4,4
        Result of nibabel.nifti1.Nifti1Image.affine
    voxel_dims : list
        Length in mm for x,y, and z dimensions of each voxel.
    target_center_coords: list of float
        3 numbers to specify the translation part of the affine if not using the same as the input_affine.

    Returns
    -------
    target_affine : 4x4matrix
        The resampled image.
    """
    # Initialize target_affine
    target_affine = input_affine.copy()
    # Decompose the image affine to allow scaling
    u,s,v = np.linalg.svd(target_affine[:3,:3],full_matrices=False)
    
    # Rescale the image to the appropriate voxel dimensions
    s = voxel_dims
    
    # Reconstruct the affine
    target_affine[:3,:3] = u @ np.diag(s) @ v

    # Set the translation component of the affine computed from the input
    # image affine if coordinates are specified by the user.
    if target_center_coords is not None:
        target_affine[:3,3] = target_center_coords
    return target_affine


def s4lmat_to_nifti(mat_file, template, outname=None, res=(1,1,1)):
    '''
    Converts 3d data in a regularly sampled S4L-generated .mat file to a zipped NiFTI
    in the same space as a template NiFTI (usually in MNI152 space)

    Parameters
    ----------
    mat_file : str
        path to the matrix output by S4L.
    res : tuple of floats, optional
        xyz resolution of the data. The default is (1,1,1).
    outname : str
        name of the output .nii.gz file. If None, then the basename is used, changing
        .mat to .nii.gz
    template : str
        path to template NiFTI

    Returns
    -------
    None.

    '''
    
    voltage_mat = scipy.io.loadmat(mat_file)
    mx = mat_to_matrix(voltage_mat)
    
    te_in = nib.load(template)
    head = te_in.header
    head['pixdim'][1:4] = res
    
    new_affine = rescale_affine(te_in.affine, res)
    
    new_im = nib.Nifti1Image(mx, new_affine, header=head)
    
    if not outname:
        pname = os.path.dirname(os.path.normpath(mat_file))
        bname = os.path.basename(os.path.normpath(mat_file))
        stripname = bname[:-4]
        new_bname = f'{stripname}.nii.gz'
        outname = os.path.join(pname, new_bname)
        
    nib.save(new_im, outname)


def float_to_index(fl, res=1, start=0):
    '''
    Converts a float representing a position on an axis to the index
    on a regularly discretized version of that axis
    
    Note that MNI152 starts a -98,-134,-72
    MNI152 space (0,0,0) is at index 196,268,144 for 0.5mm isotropic
        (i.e., MNI152 center is not at voxel center!)
    '''
    idx = int(round((fl-start)/res))
    return idx


def closest_to(pt, pt_list):
    # returns index of the coordinates that are closest to pt
    A = np.array(pt_list)
    leftbottom = np.array(pt)
    distances = np.linalg.norm(A-leftbottom, axis=1)
    min_index = np.argmin(distances)
    
    close_pt = A[min_index]
    min_dist = distances[min_index]
    
    return min_index, close_pt, min_dist


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def unpack_mat_data(mat):
    # so slow. deprecated
    data = mat['Snapshot0']
    exes = mat['Axis0'][0]
    whys = mat['Axis1'][0]
    zees = mat['Axis2'][0]
    
    exes_r = moving_average(exes, 2)
    whys_r = moving_average(whys, 2)
    zees_r = moving_average(zees, 2)
    
    #idex = 0
    #vals = []
    #coords = []
    
    grid = np.array(list(itertools.product(zees_r, whys_r, exes_r)))
    g_copy = grid.copy()
    grid[:,0], grid[:,2] = g_copy[:,2], g_copy[:,0]
    #grid[:,0], grid[:,2] = grid[:,2], grid[:,0]
    '''
    #grid = np.array(list(itertools.product(zees, whys, exes)))
    
    #grid = np.vstack(np.meshgrid(zees_r,whys_r,exes_r)).reshape(3,-1)
    #grid = np.swapaxes(grid, 0, 1)
            
    print('correct')    
    for iz, z_coord in enumerate(zees_r):
        if iz==2:
            break
        for iy, y_coord in enumerate(whys_r):
            for ix, x_coord in enumerate(exes_r):
                print(ix,iy,iz)
                
                val = float(data[idex].real) # note that we will lose complex component
                coord = np.array([x_coord, y_coord, z_coord])
                
                vals.append(val)
                coords.append(coord)
                
                idex += 1
    '''
                
    return np.array(data), grid





    