import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import vtk
from vtk.util import numpy_support as VN


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", action="store", required=True,
                        help="The sampled file from which to reconstruct the data.")
    parser.add_argument("--noexp", action="store_true", required=False, default=False,
                        help="If set, will not apply the exponent function to vti grid. By default, this is done because the data simulation data is log-scaled to aid in visualization.")
    return parser.parse_args()
# End of parse_arguments()


def save_to_h5(grid_3d, h5filename, noexp=False):
    print("writing hdf5 file")
    exp_grid = grid_3d
    if not noexp:
        exp_grid = np.exp(grid_3d)
    # h5filename = 'recons/' + args.dataset + '_' + args.samp_method + '_' + args.recon_method + args.tag + '.h5'
    # os.system("cp template.h5 ${0}".format(h5filename))
    hdf_file = h5py.File(h5filename, "w")
    # del hdf_file["native_fields/baryon_density"]
    dataset = hdf_file.create_dataset("native_fields/baryon_density", data=exp_grid)
    hdf_file.close()

    print("verifying data")
    hdf_file = h5py.File(h5filename, 'r')
    assert(np.allclose(hdf_file["native_fields/baryon_density"][:], exp_grid))
    hdf_file.close()
# End of save_to_h5()


if __name__ == "__main__":
    args = parse_arguments()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(args.infile)
    reader.Update()

    data = reader.GetOutput()
    dim = data.GetDimensions()

    name = data.GetPointData().GetArrayName(0)
    vals = data.GetPointData().GetArray(name)
    vals_np = VN.vtk_to_numpy(vals)
    print(np.shape(vals_np))
    print(np.min(vals_np))
    print(np.max(vals_np))
    grid_3d = vals_np.reshape(dim)
    h5filename = args.infile[:-4] + ".h5"

    save_to_h5(grid_3d, h5filename, args.noexp)
