"""This script refines a multichannel grain map by collecting peaks for the voxels and doing a LSQ
fit for each possible UBI matrix. The full peak set is used and the final fit uses peaks merged across
the detector plane as well as omega.

Written by: Axel Henningssson, Lund University, Nov 2023.

"""

import os

os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import numpy as np
import matplotlib.pyplot as plt
import ImageD11
import ImageD11.grain
import ImageD11.columnfile
import ImageD11.unitcell
import xfab.tools
import copy
import time
from multiprocessing import Pool, set_start_method
import ImageD11.sym_u
ImageD11.indexing.loglevel = 2
np.random.seed(0)

def get_point_data(params, xi0, yi0, peak_dict):
    xpos = xi0*peak_dict['cosomega'] - yi0*peak_dict['sinomega']
    y = xi0*peak_dict['sinomega'] + yi0*peak_dict['cosomega']
    ydist = np.abs( y + peak_dict['dty'] )
    m = ydist <= params['TRANSLATION_STAGE_DY_STEP']
    return xpos, m, ydist

def get_gve(params, local_mask, xpos, peak_dict):
    D = params['distance']
    params['distance'] = params['distance'] - xpos[local_mask]
    tth, eta = ImageD11.transform.compute_tth_eta(
                        (peak_dict['sc'][local_mask], peak_dict['fc'][local_mask]),
                        **params)
    gve = ImageD11.transform.compute_g_vectors(tth,
                        eta,
                        peak_dict['omega'][local_mask],
                        params['wavelength'],
                        wedge=params['wedge'],
                        chi=params['chi'])
    params['distance'] = D
    return gve


def refine(xi0, yi0, ubis, params, peak_dict, lattice):
    _, local_mask, ydist = get_point_data(params, xi0, yi0, peak_dict)

    gve = get_gve(params, local_mask, peak_dict['xpos_refined'], peak_dict)

    grains = [ImageD11.grain.grain(ubi) for ubi in ubis]

    tol = 0.1
    merge_tol = 0.05

    for i,g in enumerate(grains):
        
        labels = np.zeros( gve.shape[1],'i')-1
        drlv2 = np.ones( gve.shape[1], 'd') 
    
        j = ImageD11.cImageD11.score_and_assign( g.ubi, gve.T, tol, drlv2, labels, i)
        li = np.round(labels).astype(int).copy()
    
        g.mask = g.m = (li == i)
        g.npks = np.sum(g.mask)
    
        hkl_double = np.dot( g.ubi, gve[:, g.mask] ) 
        g.hkl = np.round ( hkl_double ).astype( int )
        g.etasign = np.sign( peak_dict['eta'][ local_mask ][ g.m ] )

        merged = {'sc':[], 'fc':[], 'dty':[], 'omega':[], 'sum_intensity':[], 'xpos_refined':[]}

        peaktags = np.vstack( (g.hkl, g.etasign, peak_dict['iy'][ local_mask ][ g.m ]) )
        unitags, labels  = np.unique(peaktags, axis=1, return_inverse=True )
        wI = peak_dict['sum_intensity'][ local_mask ][g.m]
        sI = np.bincount( labels, weights=wI )
        merged['sum_intensity']=sI
        sc = peak_dict['sc'][ local_mask ][g.m]
        fc = peak_dict['fc'][ local_mask ][g.m]
        om = peak_dict['omega'][ local_mask ][g.m]
        dty = peak_dict['dty'][ local_mask ][g.m]
        xpos_refined = peak_dict['xpos_refined'][ local_mask ][g.m]
        merged['sc'].extend( list( np.bincount( labels, weights=sc*wI  ) / sI ) )
        merged['fc'].extend( list( np.bincount( labels, weights=fc*wI  ) / sI ) )
        merged['omega'].extend( list( np.bincount( labels, weights=om*wI  ) / sI ) )
        merged['dty'].extend( list( np.bincount( labels, weights=dty*wI  ) / sI ) )
        merged['xpos_refined'].extend( list( np.bincount( labels, weights=xpos_refined*wI  ) / sI ) )

        for k in merged.keys():
            merged[k] = np.array(merged[k])

        merged['sinomega'] = np.sin( np.radians(merged['omega']) )
        merged['cosomega'] = np.cos( np.radians(merged['omega']) )
        _, local_mask_of_grain, ydist = get_point_data(params, xi0, yi0, merged)
        gve_grain = get_gve(params, local_mask_of_grain, merged['xpos_refined'], merged)

        labels = np.zeros( gve_grain.shape[1],'i')-1
        drlv2 = np.ones( gve_grain.shape[1], 'd') 
        j = ImageD11.cImageD11.score_and_assign( g.ubi, gve_grain.T, merge_tol, drlv2, labels, i)
        li = np.round(labels).astype(int).copy()
        g.mask = g.m = (li == i)
        g.npks = np.sum(g.m)
        g.gve = gve_grain[:,g.mask]
        hkl_double = np.dot( g.ubi, gve_grain[:, g.mask] ) 
        g.hkl = np.round ( hkl_double ).astype( int )
        g.ydist = ydist[local_mask_of_grain][g.mask]

        if np.sum(g.mask) > 3:

            # a.T @ gve = h =>  gve.T @ a = h.T => a = np.linalg.pinv(gve.T) @ h.T, same for b and c 
            w = (1. / (g.ydist + 1) ).reshape(g.gve.shape[1], 1)
            ubifit = np.linalg.lstsq( w * g.gve.T, w * g.hkl.T, rcond=None )[0].T
            if ubifit is not None:
                if np.linalg.det(ubifit) >= 0:
                    g.set_ubi( ubifit )

            labels = np.zeros( gve_grain.shape[1],'i')-1
            drlv2 = np.ones( gve_grain.shape[1], 'd') 

            j = ImageD11.cImageD11.score_and_assign( g.ubi, gve_grain.T, merge_tol, drlv2, labels, i)
            li = np.round(labels).astype(int).copy()
            g.mask = g.m = (li == i)
            g.npks = np.sum(g.m)

            g.eps_tensor  = None
            if ubifit is not None:
                if np.linalg.det(ubifit) >= 0:
                    ####################################################################################################
                    # And now fit a strain tensor decoupling strain and orientation for precision.
                    g.gve = gve_grain[:,g.mask]
                    hkl_double = np.dot( g.ubi, gve_grain[:, g.mask] ) 
                    g.hkl = np.round ( hkl_double ).astype( int )
                    g.ydist = ydist[local_mask_of_grain][g.mask]

                    dzero_cell = [params['cell_'+key] for key in ('_a','_b','_c','alpha','beta','gamma')]
                    B0 = xfab.tools.form_b_mat(dzero_cell) / (np.pi*2)

                    g.gve0 = g.u @ B0 @ g.hkl

                    gTg0    = np.sum(g.gve*g.gve0, axis=0)
                    gTg   = np.sum(g.gve*g.gve, axis=0)
                    g.directional_strain  = (gTg0/gTg) - 1

                    kappa = g.gve / np.linalg.norm(g.gve, axis=0)
                    kx, ky, kz = kappa
                    M = np.array( [ kx*kx, ky*ky, kz*kz, 2*kx*ky, 2*kx*kz, 2*ky*kz ] ).T

                    w = (1. / (g.ydist + 1) ).reshape(g.gve.shape[1], 1)
                    # The noise in the directional strain now propagates according to the linear transform
                    gnoise_std = 1e-4
                    a  = np.sum(g.gve0*(gnoise_std**2)*g.gve0, axis=0)
                    strain_noise_std = np.sqrt( np.divide(a, gTg**2, out=np.ones_like(gTg), where=gTg!=0) )
                    w = w * (1. / strain_noise_std.reshape(w.shape) )

                    w[ g.directional_strain > np.mean(g.directional_strain) + np.std(g.directional_strain)*3.5 ] = 0 # outliers
                    w[ g.directional_strain < np.mean(g.directional_strain) - np.std(g.directional_strain)*3.5 ] = 0 # outliers

                    try:
                        w = w / np.max(w)
                        eps_vec = np.linalg.lstsq( w * M, w.flatten() * g.directional_strain, rcond=None )[0].T
                        sxx, syy, szz, sxy, sxz, syz = eps_vec
                        g.eps_tensor = np.array([[sxx, sxy, sxz],[sxy, syy, syz],[sxz, syz, szz]])
                    except:
                        pass

                    ###############################################################################################################

        else:
            g.eps_tensor  = None

    grains = np.array(grains)[np.argsort([g.npks for g in grains])]
    ubis = [ImageD11.sym_u.find_uniq_u(g.ubi, lattice) for g in grains]
    npks = [g.npks for g in grains]
    eps  = [g.eps_tensor for g in grains]

    return ubis, eps, npks


def load_colf(pksfile, parfile, TRANSLATION_STAGE_DY_STEP):
    peaks = ImageD11.columnfile.columnfile( pksfile )
    peaks.dty = 1e3 * ( peaks.dty - np.min( peaks.dty ) ) # now in microns
    peaks.dty = peaks.dty - np.median(np.unique(peaks.dty))
    peaks.addcolumn( np.sin( np.radians(peaks.omega) ) , 'sinomega')
    peaks.addcolumn( np.cos( np.radians(peaks.omega) ) , 'cosomega')
    peaks.addcolumn( np.round( peaks.dty / TRANSLATION_STAGE_DY_STEP ).astype(int) , 'iy')
    peaks.parameters.loadparameters(parfile)
    peaks.updateGeometry()
    peaks.filter( np.abs(peaks.eta) > 5 )
    peaks.filter( np.abs(peaks.eta) < 175 )
    return peaks

def refine_chunk(args):
    params, gmap_chunk, X_chunk, Y_chunk, peaks_dict, idp = args
    refined_ubi_map_chunk = np.empty( (len(X_chunk),), dtype=object )
    t1 = time.perf_counter()
    kk = 0

    cubic = ImageD11.sym_u.cubic()

    if idp==0: 
        outstr = 'This is fork number '+str(idp)+' charged with mapping '+str(len(X_chunk))+' x-y points\n'
        with open('refine_log.txt', "a") as f: f.write(outstr)

    L = [len(gmap_chunk[i][0]) for i in range(len(gmap_chunk))]
    index = np.flip( np.argsort( L ) ) # lets do big voxels first

    log_time = 10

    for i, (xi0,yi0) in enumerate( zip(X_chunk, Y_chunk) ):
        ii = index[i]
        candidate_ubis, _ = gmap_chunk[ii]
        xi0, yi0 = X_chunk[ii], Y_chunk[ii]

        if len(candidate_ubis)==0: 
            refined_ubi_map_chunk[ii] = [], [], []
        else: 
            kk += 1
            refined_ubi_map_chunk[ii] = refine(xi0, yi0, candidate_ubis, params, peaks_dict, cubic)

        t2 = time.perf_counter()

        if t2-t1 > log_time and idp==0:
            log_time += 10
            outstr = 'Map-speed @ reference fork is : '+str( np.round((t2-t1)/(kk+1), 3) )+' s / per point,   done '+str(i)+' of '+str(len(gmap_chunk))+' voxels\n'
            with open('refine_log.txt', "a") as f: f.write(outstr)

    return refined_ubi_map_chunk

parser = argparse.ArgumentParser(description='Point by point texture mapping')
parser.add_argument('-pksfile', help='absolute path to column file', required=True)
parser.add_argument('-parfile', help='absolute path to parameters file', required=True)
parser.add_argument('-TRANSLATION_STAGE_DY_STEP', help='y-step in microns', required=True)

args = parser.parse_args()

if __name__ == "__main__":

    args = parser.parse_args()

    if not args.pksfile.startswith('/home/ax8427he/'):
        raise ValueError(args.pksfile+' must be an absolute path starting with /home/ax8427he/ or ~') 
    if not args.parfile.startswith('/home/ax8427he/'):
        raise ValueError(args.pksfile+' must be an absolute path starting with /home/ax8427he/ or ~') 

    with open('pbp_log.txt', "w") as f: 
        f.write('Reading data from :\n')
    
    with open('pbp_log.txt', "a") as f: 
        f.write('gmap.npy' + '\n')
        f.write('X_coord_gmap.npy'+ '\n')
        f.write('Y_coord_gmap.npy'+ '\n')

    gmap  = np.load('gmap.npy', allow_pickle=True)
    X     = np.load('X_coord_gmap.npy')
    Y     = np.load('Y_coord_gmap.npy')

    # r1,r2 = 190,300
    # c1,c2 = 190,330
    # gmap = gmap[r1:r2,c1:c2]
    # X = X[r1:r2,c1:c2]
    # Y = Y[r1:r2,c1:c2]

    TRANSLATION_STAGE_DY_STEP =  int(args.TRANSLATION_STAGE_DY_STEP)

    nthreads    = len(os.sched_getaffinity(0))
    X_chunks    = np.array_split(X.flatten(), nthreads )
    Y_chunks    = np.array_split(Y.flatten(), nthreads )
    gmap_chunks = np.array_split(gmap.flatten(), nthreads )

    colf = load_colf(args.pksfile, args.parfile, TRANSLATION_STAGE_DY_STEP)

    sc       = colf.sc[:].copy()
    fc       = colf.fc[:].copy()
    eta       = colf.eta[:].copy()
    omega    = colf.omega[:].copy()
    dty      = colf.dty[:].copy()
    sinomega = colf.sinomega[:].copy()
    cosomega = colf.cosomega[:].copy()
    iy = colf.cosomega[:].copy()
    xpos_refined = np.load('XPOS.npy')
    sum_intensity = colf.sum_intensity[:].copy()
    Number_of_pixels = colf.Number_of_pixels[:].copy() 
    pks_dict = {'xpos_refined':xpos_refined , 'sc':sc , 'eta':eta, 'fc':fc, 'Number_of_pixels': Number_of_pixels, 'omega':omega, 'dty':dty, 'sinomega':sinomega, 'cosomega':cosomega, 'iy':iy, 'sum_intensity': sum_intensity}
    params     = colf.parameters.parameters
    params['TRANSLATION_STAGE_DY_STEP'] = TRANSLATION_STAGE_DY_STEP

    args = [(params, copy.deepcopy(gmap_chunks[i]), X_chunks[i], Y_chunks[i], pks_dict, i) for i in range(nthreads)]

    set_start_method('forkserver')

    t1 = time.perf_counter()
    with Pool(processes = nthreads) as p:
        refined_ubi_map_chunks = p.map(refine_chunk, args)

    t2 = time.perf_counter()
    refined_ubi_map = np.concatenate( refined_ubi_map_chunks ).reshape(gmap.shape)    

    with open('refine_log.txt', "a") as f: 
        f.write( 'Total elapsed time: '+str(t2-t1)+' s\n' )
    	
    #np.save('refined_ubi_map.npy', refined_ubi_map)
    np.save('refined_ubi_and_strain_map_com_corrected_test.npy', refined_ubi_map)

    with open('refine_log.txt', "a") as f: 
    	f.write(' Done !\n' )


