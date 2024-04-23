import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
import numpy as np
from multiprocessing import Pool, set_start_method
import ImageD11.columnfile
import copy

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


def get_origins(args):
    params, ubi_map, sample, X, Y, indices, peak_dict, i = args

    print('Proccess ', i, 'started...')
    xpos=np.zeros((len(indices)))
    for kk,peak_index in enumerate(indices):
        g_vector = np.array([peak_dict['gx'][peak_index], peak_dict['gy'][peak_index], peak_dict['gz'][peak_index]])
        #xgrid = X*peak_dict['cosomega'][peak_index] - Y*peak_dict['sinomega'][peak_index]
        ygrid = X*peak_dict['sinomega'][peak_index] + Y*peak_dict['cosomega'][peak_index]
        ydist = np.abs( ygrid + peak_dict['dty'][peak_index] )
        m = ydist <= params['TRANSLATION_STAGE_DY_STEP']
        if m.sum() > 0:
            m = m * sample
            orientations = ubi_map[m,:,:]

            hklf = orientations @ g_vector
            hklr = hklf.round()
            err = ((hklf-hklr)**2).sum(axis=1)
            hkl_tol = 0.05
            m2 = np.zeros_like(sample)
            m2[m] = err < hkl_tol

            if m2.sum() > 0:
                weights = ( 1/ (ydist + 1 )  )

                xc_sample = (X[m2]*weights[m2]).sum() / weights[m2].sum()
                yc_sample = (Y[m2]*weights[m2]).sum() / weights[m2].sum()

                xc_lab = xc_sample*peak_dict['cosomega'][peak_index] - yc_sample*peak_dict['sinomega'][peak_index]
                #yc_lab = xc_sample*peak_dict['sinomega'][peak_index] + yc_sample*peak_dict['cosomega'][peak_index]

                xpos[kk] = xc_lab

        if i==-1 and kk%2000==0:
            print(end='\n Proccess '+str(i)+' at '+str(kk)+' of total '+str(len(indices)))
    return np.array(xpos)


if __name__=='__main__':


    pksfile = '/home/ax8427he/workspace/AL_DTU/data/experiment/sam3/layer_bottom/sam3_pointscan_60N_fine_sparse_z0.h5'
    parfile = '/home/ax8427he/workspace/AL_DTU/data/experiment/sam3/layer_bottom/aluS3.par'
    TRANSLATION_STAGE_DY_STEP=3
    colf = load_colf(pksfile, parfile, TRANSLATION_STAGE_DY_STEP)

    sc       = colf.sc[:].copy()
    fc       = colf.fc[:].copy()
    eta       = colf.eta[:].copy()
    omega    = colf.omega[:].copy()
    dty      = colf.dty[:].copy()
    sinomega = colf.sinomega[:].copy()
    cosomega = colf.cosomega[:].copy()
    iy = colf.cosomega[:].copy()
    sum_intensity = colf.sum_intensity[:].copy()
    Number_of_pixels = colf.Number_of_pixels[:].copy() 
    peak_dict = {'sc':sc , 'eta':eta, 'fc':fc, 'Number_of_pixels': Number_of_pixels, 'omega':omega, 'dty':dty, 'sinomega':sinomega, 'cosomega':cosomega, 'iy':iy, 'sum_intensity': sum_intensity}
    peak_dict['gx'] = colf.gx[:].copy() 
    peak_dict['gy'] = colf.gy[:].copy() 
    peak_dict['gz'] = colf.gz[:].copy() 
    params     = colf.parameters.parameters
    params['TRANSLATION_STAGE_DY_STEP'] = TRANSLATION_STAGE_DY_STEP

    X     = np.load('X_coord_gmap.npy')
    Y     = np.load('Y_coord_gmap.npy')
    sample = np.fliplr(  np.load('sample_mask_layer_bottom.npy')[2:-2, 2:-2] )
    ubi_map = np.load('ubi_median_filt_s6.npy', allow_pickle=True ) 

    nthreads    = len(os.sched_getaffinity(0))
    index = np.array_split( np.array( range(colf.nrows) ), nthreads )
    args = [[copy.deepcopy(params), ubi_map.copy(), sample.copy(), X.copy(), Y.copy(), index[i].copy(), copy.deepcopy(peak_dict), i] for i in range(nthreads)]
    args[-1][-1]=-1

    set_start_method('forkserver')
    
    t1 = time.perf_counter()
    with Pool(processes = nthreads) as p:
        XPOS = p.map(get_origins, args)
    XPOS = np.concatenate(XPOS)
    t2 = time.perf_counter()

    print(" ")
    print(" ")
    print("DONE")
    print(" ")
    print('TIME: ', (t2-t1)/60, 'min')

    np.save('XPOS.npy', XPOS)  

