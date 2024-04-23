import os

# we do not want the processes to conflict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import numpy as np
import matplotlib.pyplot as plt
import ImageD11
import ImageD11.columnfile
import ImageD11.unitcell
import copy
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
import ImageD11.indexing as indexing
import ImageD11.sym_u
ImageD11.indexing.loglevel = 2
np.random.seed(0)

def get_grid(peaks):
    y_translations = -np.unique(peaks.dty.round(1))
    yg =  y_translations
    xg = -y_translations
    X, Y = np.meshgrid( xg, yg, indexing='ij' )
    gmap = np.empty(X.shape, dtype=object)
    return X, Y, gmap

def load_colf(pksfile, parfile):
    peaks = ImageD11.columnfile.columnfile( pksfile )
    peaks.dty = 1e3 * ( peaks.dty - np.min( peaks.dty ) ) # now in microns
    peaks.dty = peaks.dty - np.median(np.unique(peaks.dty))
    peaks.addcolumn( np.sin( np.radians(peaks.omega) ) , 'sinomega')
    peaks.addcolumn( np.cos( np.radians(peaks.omega) ) , 'cosomega')
    peaks.parameters.loadparameters(parfile)
    peaks.updateGeometry()
    return peaks

def get_point_data(params, xi0, yi0, peak_dict):
    xpos = xi0*peak_dict['cosomega'] - yi0*peak_dict['sinomega']
    y = xi0*peak_dict['sinomega'] + yi0*peak_dict['cosomega']
    m = np.abs( y + peak_dict['dty'] ) <= params['TRANSLATION_STAGE_DY_STEP']
    return xpos, m

def get_local_indexer(params, unitcell, local_mask, xpos, peak_dict):
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
    ind = indexing.indexer( unitcell = unitcell,
            wavelength = params['wavelength'],
            gv = gve.T )
    params['distance'] = D
    return ind

def index_point(params, ind, lattice):
    ind.ds_tol=0.01
    ind.assigntorings()
    ind.ring_1  = 3
    ind.ring_2  = 3
    ind.max_grains = 10
    empty_voxel = False
    iteration = 0
    hkl_tols = np.linspace(0.03, 0.05, 5)
    max_min_pks = 24 * 2 * 3 * 3 // params['DATA_DOWNSAMPLING_FACTOR']
    min_pks_tols = np.linspace(max_min_pks, max_min_pks//2, 5).astype(int)
    while( len(ind.ubis) < 2 and not empty_voxel):
        ind.hkl_tol = hkl_tols[iteration]
        ind.minpks = min_pks_tols[iteration]
        ind.find() # finds trial orientations
        ind.scorethem() # scores them trial orientations
        iteration += 1
        if iteration==len(hkl_tols):  empty_voxel = True
    unique_ubis = [ ImageD11.sym_u.find_uniq_u(ubi, lattice) for ubi in ind.ubis ]
    return (unique_ubis, ind.scores)

def map_chunk(args):
    params, X_chunk, Y_chunk, peak_dict, idp = args

    print('This is fork number ', idp,' charged with mapping ', len(X_chunk), ' x-y points')
    if idp==0: 
        outstr = 'This is fork number '+str(idp)+' charged with mapping '+str(len(X_chunk))+' x-y points\n'
        with open('pbp_log.txt', "w") as f: f.write(outstr)

    unitcell = ImageD11.unitcell.unitcell_from_parameters(params)
    gmap_chunk = np.empty( (len(X_chunk),), dtype=object )
    cubic = ImageD11.sym_u.cubic()

    t1 = time.perf_counter()

    log_time = 30

    for i, (xi0,yi0) in enumerate( zip(X_chunk, Y_chunk) ):

        xpos, local_mask = get_point_data(params, xi0, yi0, peak_dict)
        ind = get_local_indexer(params, unitcell, local_mask, xpos, peak_dict)
        gmap_chunk[i] = index_point(params, ind, cubic)

        t2 = time.perf_counter()

        if (i!=0 and i%10==0) and idp==0:
           print( 'Map-speed @ reference fork is : ', np.round((t2-t1)/(i+1), 3), 's / per point,   done ', i, 'of', len(gmap_chunk), ' voxels', end='\r' )
        if t2-t1 > log_time and idp==0:
            log_time += 30
            outstr = 'Map-speed @ reference fork is : '+str( np.round((t2-t1)/(i+1), 3) )+' s / per point,   done '+str(i)+' of '+str(len(gmap_chunk))+' voxels\n'
            with open('pbp_log.txt', "a") as f: f.write(outstr)

    return gmap_chunk

parser = argparse.ArgumentParser(description='Point by point texture mapping')
parser.add_argument('-pksfile', help='absolute path to column file', required=True)
parser.add_argument('-parfile', help='absolute path to parameters file', required=True)
parser.add_argument('-TRANSLATION_STAGE_DY_STEP', help='y-step in microns', required=True)
parser.add_argument('-DATA_DOWNSAMPLING_FACTOR', help='fracion of peaks to use for indexing (speeds things up)', required=True)
parser.add_argument('-GRAIN_MAP_PADDING', help='number of voxels to pad the grain map', required=True)
parser.add_argument('-GRAIN_MAP_SUPER_SAMPLING', help='Use voxels smaller than the beamsize (e.g 2 gives 4 voxels per original voxel etc.)', required=True)

args = parser.parse_args()


if __name__ == "__main__":


    args = parser.parse_args()

    if not args.pksfile.startswith('/home/ax8427he/'):
        raise ValueError(args.pksfile+' must be an absolute path starting with /home/ax8427he/ or ~') 
    if not args.parfile.startswith('/home/ax8427he/'):
        raise ValueError(args.pksfile+' must be an absolute path starting with /home/ax8427he/ or ~') 

    print('')
    print('Reading data from :')
    print(args.pksfile)
    print(args.parfile)
    print('')

    TRANSLATION_STAGE_DY_STEP =  int(args.TRANSLATION_STAGE_DY_STEP)
    DATA_DOWNSAMPLING_FACTOR  =  int(args.DATA_DOWNSAMPLING_FACTOR)
    GRAIN_MAP_PADDING         =  int(args.GRAIN_MAP_PADDING)
    GRAIN_MAP_SUPER_SAMPLING  =  int(args.GRAIN_MAP_SUPER_SAMPLING)

    colf = load_colf(args.pksfile, args.parfile)
    X, Y, gmap = get_grid(colf)
    print('Gridshape is: ', X.shape)

    index = np.random.permutation(colf.nrows)
    keep  = index[0:colf.nrows//DATA_DOWNSAMPLING_FACTOR]
    peak_mask = np.zeros( (colf.nrows,), dtype=bool )
    peak_mask[keep] = True
    colf.filter(peak_mask)

    sc       = colf.sc[:].copy()
    fc       = colf.fc[:].copy()
    omega    = colf.omega[:].copy()
    dty      = colf.dty[:].copy()
    sinomega = colf.sinomega[:].copy()
    cosomega = colf.cosomega[:].copy()
    pks_dict = {'sc':sc ,'fc':fc, 'omega':omega, 'dty':dty, 'sinomega':sinomega, 'cosomega':cosomega}

    pars     = colf.parameters.parameters
    pars['filename'] = None
    pars['TRANSLATION_STAGE_DY_STEP'] = TRANSLATION_STAGE_DY_STEP
    pars['DATA_DOWNSAMPLING_FACTOR'] = DATA_DOWNSAMPLING_FACTOR
    pars['GRAIN_MAP_PADDING'] = GRAIN_MAP_PADDING
    pars['GRAIN_MAP_SUPER_SAMPLING'] = GRAIN_MAP_SUPER_SAMPLING

    np.save('pks_dict.npy', pks_dict)
    np.save('pars.npy', pars)
    np.save('gmap_orig.npy', gmap)
    np.save('X_orig_coord_gmap.npy', X)
    np.save('Y_orig_coord_gmap.npy', Y)

    pks_dict = np.load('pks_dict.npy', allow_pickle=True)[()]
    for k in pks_dict.keys():
        pks_dict[k] = np.ascontiguousarray(pks_dict[k])
    pars = np.load('pars.npy', allow_pickle=True)[()]

    gmap = np.load('gmap_orig.npy', allow_pickle=True)
    X = np.load('X_orig_coord_gmap.npy')
    Y = np.load('Y_orig_coord_gmap.npy')

    if GRAIN_MAP_PADDING < 0:
        pad = -GRAIN_MAP_PADDING
        xistart, xiend = pad, X.shape[0]-pad
        yistart, yiend = pad, Y.shape[1]-pad
        X, Y, gmap = X[xistart:xiend, yistart:yiend] , Y[xistart:xiend, yistart:yiend] , gmap[xistart:xiend, yistart:yiend] 
    elif GRAIN_MAP_PADDING>1:
        raise NotImplementedError()

    if GRAIN_MAP_SUPER_SAMPLING < 0:
        ss   = -GRAIN_MAP_SUPER_SAMPLING
        X    = X[0::ss, 0::ss]
        Y    = Y[0::ss, 0::ss]
        gmap = gmap[0::ss, 0::ss]
    elif GRAIN_MAP_SUPER_SAMPLING>1:
        ss   = -GRAIN_MAP_SUPER_SAMPLING
        xg = np.linspace(X.min(), X.max(), X.shape[0]*GRAIN_MAP_SUPER_SAMPLING)
        yg = np.linspace(Y.min(), Y.max(), Y.shape[1]*GRAIN_MAP_SUPER_SAMPLING)
        X, Y = np.meshgrid( xg, yg, indexing='ij' )
        gmap = np.empty(X.shape, dtype=object)

    print('')
    print('TRANSLATION_STAGE_DY_STEP  :', TRANSLATION_STAGE_DY_STEP)
    print('DATA_DOWNSAMPLING_FACTOR   :', DATA_DOWNSAMPLING_FACTOR)
    print('GRAIN_MAP_PADDING          :', GRAIN_MAP_PADDING)
    print('GRAIN_MAP_SUPER_SAMPLING   :', GRAIN_MAP_SUPER_SAMPLING)
    print('os.sched_getaffinity()     :',  len(os.sched_getaffinity(0)) )
    print('')
    print('Gridshape is: ', X.shape)
    print('X-bounds are : ', X.min(), X.max())
    print('Y-bounds are : ', Y.min(), Y.max())
    print('')

    nthreads    = len(os.sched_getaffinity(0))
    X_chunks    = np.array_split(X.flatten(), nthreads )
    Y_chunks    = np.array_split(Y.flatten(), nthreads )
    args = [(pars, X_chunks[i], Y_chunks[i], pks_dict, i) for i in range(nthreads)]

    t1 = time.perf_counter()
    with Pool(processes = nthreads) as p:
        gmap_chunks = p.map(map_chunk, args)
    t2 = time.perf_counter()
    gmap = np.concatenate( gmap_chunks ).reshape(gmap.shape)    
    print( ' Total elapsed time: ', (t2-t1) , 's' ) # 4 cores gives: 35s, 1 core gives: 46

    np.save('gmap.npy', gmap)
    np.save('X_coord_gmap.npy', X)
    np.save('Y_coord_gmap.npy', Y)