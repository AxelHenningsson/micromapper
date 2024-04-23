# About

This repository contains code and data associated to the paper titled *"Microstructure and Stress Mapping in 3D at Industrially Relevant Degrees of Plastic Deformation"*. This is a scientific storage repository primarily to be used for reference.

# Contents

The code/analysis folder contains the primary python code that was used to process the data (from the data folder) resulting in a series of numerical arrays stored in the results folder. Each of the three analysed z-layers (z1-layer_bottom, z2-layer_center, z3-layer_top) are represented as sub-folders in results.

Further post proccessing including stress conversion, KAM filtering, grain boundary identification, error analysis, etc was implemented through a series of jupyter notebooks (code/post). Code used to produce the figures found in the published paper, and the associated supplementary materials, was likewise produced with the notebooks in code/post.

# Documentation
The code provided in this repository reflect a snapshot of a codebase orginally implemented to run at cpu cluster. To maximise transparancy the code is provided as is, and we provide some overarching documentation below:

## code/analysis


* pfun.py 
  
  * *code used to segment raw diffraction in conjunction with ImageD11.*

* sample.ipynb
  * *Computes a sinogram over summed diffracted intenseties used to reconstruct the sample outline shape*
  
* pbp.py

  * *Finds a list of strain-orientation states for each voxel in a 2D grid using the indexer of ImageD11. This code produces the multi-channel strain-orientation map. To be called from command line. e.g*
   
    ```
    python pbp.py -pksfile .../sam3_pointscan_60N_fine_sparse_z0.h5 -parfile .../aluS3.par -TRANSLATION_STAGE_DY_STEP 3 -DATA_DOWNSAMPLING_FACTOR 10 -GRAIN_MAP_PADDING 0 -GRAIN_MAP_SUPER_SAMPLING 2
    ```

* refine.py 
  * *Refines the 2D voxel grid of candidate strain-orientation states. i.e refines the multi-channel strain-orientation map. To be called from command line. e. g*

    ```
    python refine.py -pksfile .../sam3_pointscan_60N_fine_sparse_z0.h5 -parfile .../aluS3.par -TRANSLATION_STAGE_DY_STEP 3
    ```

* compute_diffraction_origins.py
  * *Computes diffraction origins along the beam path based on a primary reconstruction (e.g. ubi_median_filt_s6.npy) and sample mask (e.g. sample_mask_layer_bottom.npy). Data paths to be modified as needed.*


* ipf_filter.py
  * *Converts the multi-channel strain-orientation map to inverse pole figure color maps. Deploys spatial median filters. Selects a solution  strain-orientation map from the multi-channel options based on the closest match to a filtered map.*

## code/post

* load_curve.ipynb
  * *Read and plot corrected stress-strain data from loading device*
  
* kam.ipynb
  * *Kernel average misorientation filter and grain boundary identification. This script also includes spatial error estimates based on grain boundary displacements.*

* misori.ipynb
  * *Segment grains based on a flood-fill approach using misorientation thresholds. Analyses per-grain statistics.*

* strain_maps.ipynb
  * *Plots the strain fields.*

* convert_to_stress.ipynb
  * *Converts elastic strain to stress taking the local orientations into acccount.*

* stress.ipynb
  * *Plots the stress fields. Computes stress metrics (effective stress etc.). Analyses correlations between stress and orientation.*

* residuals.ipynb
  * *Stress error analysis based on out-of-balance forces.*

# Usage

This is not intended to be a packaged library. To use the code you may create a new python environment and install the dependecies:

* **numpy**
* **scipy**
* **matplotlib**
* **skimage**
* **xfab**
* **ImageD11**

The code provided in this repository reflect a snapshot of a codebase orginally implemented to run at cpu cluster. To maximise transparancy the code is provided as is. To reproduce the results please refactor file-paths of to data arrays to match your locally downloaded data structure and refer to the Contents section for documentation.

The analysis and algorihtm is described in the associated paper. The corresponding code in code/analysis contains the cpu-heavy computations and was executed in a few hours on two AMD 7413 processors (Milan) - offering 48 compute cores. The code was run in sequence as as:

**A) - Pre-proccessing:**
  1. pfun.py
  2. sample.ipynb

**B) - Primary mapping**
  1. pbp.py
  2. refine.py
  3. ipf_filter.py
   
**C) - Centroid calibration**
  1. compute_diffraction_origins.py
  2. refine.py
  3. ipf_filter.py

The code in code/post was then used to explore the various dimensions of the reconstructed data.

