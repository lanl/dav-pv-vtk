# data-driven-sampling

Includes code for data-driven sampling methods for halo reconstruction in Nyx dataset.

The base code was taken from: https://github.com/lanl/alpine-lanl

## Usage

To convert from a hdf5 dataset to a vti dataset and vice-versa, use the `vtitoh5.py` and `h5tovti.py` scripts, e.g.: `python vtitoh5.py --infile rho_1.vti` and `python h5tovti.py --input rho_1.h5`

To sample from a given dataset, use the `sample.py` script, e.g.: `python sample.py --input rho_1.vti --percentage 0.1 --method max --contrast_points 0.01`.

`python sample.py --help` will display all available command-line options.

To reconstruct the dataset from the sampled data, use the `reconstruct.py` file, e.g.: `python reconstruct.py --infile rho_1/max_0.01_0.1.vtp --datafile rho_1.vti --samp_method max --recon_method linear`.

`python reconstruct.py --help` will display all available command-line options.

### Pipeline

Typically, usage follows the following pipeline:

1. Obtain Nyx simulation dataset at a given timestep, in hdf5 format.

2. Convert the data from hdf5 format into vti format for sampling (`h5tovti.py`). Note: Technically, this doesn't need to happen - the sampling functions only require a value array, but this is useful for visualizing the data and for compatibility with the original codebase.

3. Run sampling on the vti data to generate a vtk point cloud (`sample.py`).

4. Reconstruct a vti 3d image from the point cloud (`reconstruct.py`).

5. Convert the reconstructed image into an hdf5 dataset (currently done automatically as part of `reconstruct.py`, but can be done manually using `vtitoh5.py`).

6. Run the reeber amr-connected-components code to identify halos in the dataset. Example command:

```${HOME}/reeber/build/examples/amr-connected-components/amr_connected_components_float -b 64 -n -w -f native_fields/baryon_density ${INPUT_FILE} none none ${OUTPUT_FILE} -i ${ISO_THRESHOLD}```
