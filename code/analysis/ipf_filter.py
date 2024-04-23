"""This script filters and post proccess the result that is the grain map.

Written by: Axel Henningssson, Lund University, Nov 2023.

"""

import os

os.environ['OMP_NUM_THREADS'] = '1'

import scipy
import skimage
import ImageD11
import xfab
import xfab.symmetry
import xfab.tools
from scipy.ndimage import median_filter

import numpy as np
import matplotlib.pyplot as plt
import ImageD11
import ImageD11.grain
import ImageD11.columnfile
import ImageD11.unitcell
from multiprocessing import Pool
import ImageD11.sym_u
ImageD11.indexing.loglevel = 2
np.random.seed(0)

xfab.CHECKS.activated = False

def crystal_direction_cubic( ubi, axis ):
    hkl = np.dot( ubi, axis )
    # cubic symmetry implies:
    #      24 permutations of h,k,l
    #      one has abs(h) <= abs(k) <= abs(l)
    hkl = abs(hkl)
    hkl.sort()
    return hkl

def hkl_to_color_cubic( hkl ):
    """
    https://mathematica.stackexchange.com/questions/47492/how-to-create-an-inverse-pole-figure-color-map
        [x,y,z]=u⋅[0,0,1]+v⋅[0,1,1]+w⋅[1,1,1].
            These are:
                u=z−y, v=y−x, w=x
                This triple is used to assign each direction inside the standard triangle

    makeColor[{x_, y_, z_}] :=
         RGBColor @@ ({z - y, y - x, x}/Max@{z - y, y - x, x})
    """
    x,y,z = hkl
    assert x<=y<=z
    assert z>=0
    u,v,w = z-y, y-x, x
    m = max( u, v, w )
    r,g,b = u/m, v/m, w/m
    return (r,g,b)

def ubi_from_rgb_cubic( rgb, axes ):
    hkl = np.zeros_like(rgb)
    hkl[0,:] = rgb[2,:]
    hkl[1,:] = rgb[1,:] + hkl[0,:]
    hkl[2,:] = rgb[0,:] + hkl[1,:]
    # hkl = rgb @ axes
    ubi = ( hkl @ np.linalg.inv(axes) )
    if np.dot( np.cross(ubi.T[:,0], ubi.T[:,1]), ubi.T[:,2] ) < 0:
        ubi[:,2] *= -1
    return ubi

def hkl_to_pf_cubic( hkl ):
    x,y,z = hkl
    assert x<=y<=z
    assert z>=0
    m = np.sqrt((hkl**2).sum())
    return x/(z+m), y/(z+m)

def triangle(  ):
    """ compute a series of point on the edge of the triangle """
    xy = [ np.array(v) for v in ( (0,1,1), (0,0,1), (1,1,1) ) ]
    xy += [ xy[2]*(1-t) + xy[0]*t for t in np.linspace(0.1,1,15)]
    return np.array( [hkl_to_pf_cubic( np.array(p) ) for p in xy] )

if __name__ == "__main__":

    gmap  = np.load('gmap.npy', allow_pickle=True)
    refined_ubi_map  = np.load('refined_ubi_map.npy', allow_pickle=True)
    X     = np.load('X_coord_gmap.npy')
    Y     = np.load('Y_coord_gmap.npy')

    gmap_ori = np.zeros((*gmap.shape, 3, 3))
    gmap_npks = np.zeros(gmap.shape)
    gmap_ipf = np.zeros((*gmap.shape, 3, 3))
    gmap_xypf = np.zeros((*gmap.shape, 2, 3))

    for i in range(gmap.shape[0]):
        for j in range(gmap.shape[1]):
            if len(gmap[i,j][0])!=0:
                all_ubis, scores  = gmap[i,j]
                gmap_ori[i,j,:,:] = all_ubis[np.argmax(scores)]
                gmap_npks[i,j] = np.sum(scores)
                for ii in range(3):
                    axis = np.array([0.,0.,0.])
                    axis[ii] = 1.
                    hkl = crystal_direction_cubic( gmap_ori[i,j,:,:], axis=axis )
                    gmap_ipf[i,j,:,ii]  = hkl_to_color_cubic( hkl )
                    gmap_xypf[i,j,:,ii] = hkl_to_pf_cubic( hkl )

    np.save( 'gmap_ori.npy' ,  gmap_ori  )
    np.save( 'gmap_npks.npy' , gmap_npks )
    np.save( 'gmap_ipf.npy' ,  gmap_ipf  )
    np.save( 'gmap_xypf.npy' , gmap_xypf )

    refined_ori = np.zeros((*refined_ubi_map.shape, 3, 3))
    refined_npks = np.zeros(refined_ubi_map.shape)
    refined_ipf = np.zeros((*refined_ubi_map.shape, 3, 3))
    refined_xypf = np.zeros((*refined_ubi_map.shape, 2, 3))

    for i in range(refined_ubi_map.shape[0]):
        for j in range(refined_ubi_map.shape[1]):
            if len(refined_ubi_map[i,j][0])!=0:
                all_ubis, scores  = refined_ubi_map[i,j]
                refined_ori[i,j,:,:] = all_ubis[np.argmax(scores)]
                refined_npks[i,j] = np.sum(scores)
                for ii in range(3):
                    axis = np.array([0.,0.,0.])
                    axis[ii] = 1.
                    hkl = crystal_direction_cubic( refined_ori[i,j,:,:], axis=axis )
                    refined_ipf[i,j,:,ii]  = hkl_to_color_cubic( hkl )
                    refined_xypf[i,j,:,ii] = hkl_to_pf_cubic( hkl )

    np.save( 'refined_ori.npy'  , refined_ori   )
    np.save( 'refined_npks.npy' , refined_npks  )
    np.save( 'refined_ipf.npy'  , refined_ipf   )
    np.save( 'refined_xypf.npy' , refined_xypf  )

    median_ipf_s6 = refined_ipf.copy()
    for i in range(3):
        for j in range(3):
            median_ipf_s6[:,:,i,j] = median_filter(median_ipf_s6[:,:,i,j], size=6)

    ubi_median_filt_s6 = np.zeros_like(refined_ori)
    ipf_median_filt_s6  = np.zeros_like(refined_ipf)
    xypf_median_filt_s6  = np.zeros_like(refined_xypf)

    for i in range(ubi_median_filt_s6.shape[0]):
        for j in range(ubi_median_filt_s6.shape[1]):
            ubis, _ = refined_ubi_map[i,j]
            if len(ubis) > 0:
                scores = []
                for ubi in ubis:
                    rgb = np.zeros_like(ubi)
                    for ii in range(3):
                        axis = np.array([0.,0.,0.])
                        axis[ii] = 1.
                        hkl = crystal_direction_cubic(ubi, axis=axis )
                        rgb[:,ii]  = hkl_to_color_cubic( hkl )
                    scores.append( np.linalg.norm( rgb -  median_ipf_s6[i,j,:,:] ) )

                ubi_median_filt_s6[i,j,:,:] = ubis[np.argmin(scores)]
                for ii in range(3):
                    axis = np.array([0.,0.,0.])
                    axis[ii] = 1.
                    hkl = crystal_direction_cubic(ubi_median_filt_s6[i,j,:,:], axis=axis )
                    ipf_median_filt_s6[i,j,:,ii]  = hkl_to_color_cubic( hkl )
                    xypf_median_filt_s6[i,j,:,ii] = hkl_to_pf_cubic( hkl )

    np.save( 'ipf_median_filt_s6.npy' , ipf_median_filt_s6  )
    np.save( 'ubi_median_filt_s6.npy' , ubi_median_filt_s6  )
    np.save( 'xypf_median_filt_s6.npy' , xypf_median_filt_s6  )

    median_ipf_s4 = refined_ipf.copy()
    for i in range(3):
        for j in range(3):
            median_ipf_s4[:,:,i,j] = median_filter(median_ipf_s4[:,:,i,j], size=4)

    ubi_median_filt_s4 = np.zeros_like(refined_ori)
    ipf_median_filt_s4  = np.zeros_like(refined_ipf)
    xypf_median_filt_s4  = np.zeros_like(refined_xypf)

    for i in range(ubi_median_filt_s4.shape[0]):
        for j in range(ubi_median_filt_s4.shape[1]):
            ubis, _ = refined_ubi_map[i,j]
            if len(ubis) > 0:
                scores = []
                for ubi in ubis:
                    rgb = np.zeros_like(ubi)
                    for ii in range(3):
                        axis = np.array([0.,0.,0.])
                        axis[ii] = 1.
                        hkl = crystal_direction_cubic(ubi, axis=axis )
                        rgb[:,ii]  = hkl_to_color_cubic( hkl )
                    scores.append( np.linalg.norm( rgb -  median_ipf_s4[i,j,:,:] ) )

                ubi_median_filt_s4[i,j,:,:] = ubis[np.argmin(scores)]
                for ii in range(3):
                    axis = np.array([0.,0.,0.])
                    axis[ii] = 1.
                    hkl = crystal_direction_cubic(ubi_median_filt_s4[i,j,:,:], axis=axis )
                    ipf_median_filt_s4[i,j,:,ii]  = hkl_to_color_cubic( hkl )
                    xypf_median_filt_s4[i,j,:,ii] = hkl_to_pf_cubic( hkl )

    np.save( 'ipf_median_filt_s4.npy' , ipf_median_filt_s4  )
    np.save( 'ubi_median_filt_s4.npy' , ubi_median_filt_s4  )
    np.save( 'xypf_median_filt_s4.npy' , xypf_median_filt_s4  )

    pars = np.load('pars.npy', allow_pickle=True)[()]
    dzero_cell = [pars['cell_'+key] for key in ('_a','_b','_c','alpha','beta','gamma')]

    strain_map_s6 = np.zeros(ubi_median_filt_s6.shape)
    for i in range(ubi_median_filt_s6.shape[0]):
        for j in range(ubi_median_filt_s6.shape[1]):
            g = ImageD11.grain.grain( ubi_median_filt_s6[i,j,:,:] )
            strain_map_s6[i,j,:,:] = g.eps_sample_matrix(dzero_cell)
    np.save( 'strain_map_s6.npy' , strain_map_s6  )

    strain_map_s4 = np.zeros(ubi_median_filt_s4.shape)
    for i in range(ubi_median_filt_s4.shape[0]):
        for j in range(ubi_median_filt_s4.shape[1]):
            g = ImageD11.grain.grain( ubi_median_filt_s4[i,j,:,:] )
            strain_map_s4[i,j,:,:] = g.eps_sample_matrix(dzero_cell)
    np.save( 'strain_map_s4.npy' , strain_map_s4  )



