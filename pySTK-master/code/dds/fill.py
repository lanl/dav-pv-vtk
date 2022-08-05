import argparse
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import vtk

from vtitoh5 import save_to_h5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store", required=False, default="nyx",
                        help="The kind of dataset this is (nyx|isabel). Default = nyx")
    parser.add_argument("--samp_method", action="store", required=False, default="hist_grad",
                        help="The sampling method used. Default = hist_grad")
    parser.add_argument("--recon_method", action="store", required=False, default="nearest",
                        help="Reconstruction method (nearest|linear|cubic). Default = nearest. linear|cubic are slow.")
    parser.add_argument("--infile", action="store", required=True,
                        help="The sampled file from which to reconstruct the data.")
    parser.add_argument("--datafile", action="store", required=False, default="rho_1.vti",
                        help="The original data file, used for dimension information. Default = rho_1.vti")
    parser.add_argument("--tag", action="store", required=False, default="",
                        help="A tag to add to the output file name. Default = \"\"")
    parser.add_argument("--mean", action="store", required=False, default=11.867812041881846, type=np.double,
                        help="If set, will fill the non-sampled points with a value such that the mean equals this value.")
    parser.add_argument("--constant", action="store", required=False, default=None, type=float,
                        help="If mean not set, will fill the non-sampled points with this value.")
    parser.add_argument("--percentile", action="store", required=False, default=1.125, type=float,
                        help="If constant and mean not set, will fill the non-sampled points with this percentile of data values.")
    return parser.parse_args()
# End of parse_arguments()

args = parse_arguments()

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(args.datafile)
reader.Update()
data = reader.GetOutput()

if args.dataset == "nyx":
    var_name = "logField"
elif args.dataset == "isabel":
    var_name = 0

print("Spacing: ", data.GetSpacing(), " origin: ", data.GetOrigin(), " dimensions: ", data.GetDimensions())

XDIM, YDIM, ZDIM = np.array(data.GetDimensions())
spacing = np.array(data.GetSpacing())
origin = np.array(data.GetOrigin())

poly_reader = vtk.vtkXMLPolyDataReader()
poly_reader.SetFileName(args.infile)
poly_reader.Update()

data = poly_reader.GetOutput()

print("total points:", data.GetNumberOfPoints(), data.GetNumberOfElements(0))

pts = data.GetPoints()

pt_data = data.GetPointData().GetArray(var_name).GetTuple1(100)
print('data:', pt_data)

print(pts.GetPoint(0))

tot_pts = data.GetNumberOfPoints()
feat_arr = np.zeros((tot_pts,3))

print('total points:', tot_pts)

data_vals = np.zeros(tot_pts)


for i in range(tot_pts):
    loc = pts.GetPoint(i)
    feat_arr[i, :] = np.asarray(loc)
    pt_data = data.GetPointData().GetArray(var_name).GetTuple1(i)
    data_vals[i] = pt_data

range_min = np.min(feat_arr, axis=0)
range_max = np.max(feat_arr, axis=0)

print("range:", range_min, range_max)

# cur_loc = np.zeros((XDIM * YDIM * ZDIM, 3), dtype='double')

z_idx = np.rint(np.divide(feat_arr[:, 2], spacing[0]))
y_idx = np.rint(np.divide(feat_arr[:, 1], spacing[1]))
x_idx = np.rint(np.divide(feat_arr[:, 0], spacing[2]))

idx = np.asarray((z_idx * XDIM * YDIM) + (y_idx * XDIM) + x_idx, dtype=np.int32)
assert(idx.size == data_vals.size)
grid_size = (XDIM * YDIM * ZDIM)

if args.mean:
    expected_sum = grid_size * args.mean
    value_sum = data_vals.sum()
    remaining_sum = expected_sum - value_sum
    remaining_size = grid_size - data_vals.size
    fill_value = np.divide(remaining_sum, remaining_size, dtype='double')
    grid = np.full(grid_size, fill_value, dtype='double')
elif args.constant:
    grid = np.full(grid_size, args.constant, dtype='double')
else:
    grid = np.full(grid_size, np.percentile(data_vals, args.percentile), dtype='double')

# insert sampled data values
grid[idx] = data_vals

print("interpolating")
# grid_z0 = griddata(feat_arr, data_vals, cur_loc, method=args.recon_method)
grid_z0_3d = grid.reshape((ZDIM, YDIM, XDIM))
# write to a vti file
output_filename = 'recons/' + args.dataset + '_' + args.samp_method + '_' + args.recon_method + args.tag + '_zfill.vti'
print("writing reconstructed file to: ", output_filename)
Path(output_filename).parent.mkdir(exist_ok=True, parents=True)
imageData = vtk.vtkImageData()
imageData.SetDimensions(XDIM, YDIM, ZDIM)

imageData.SetOrigin(origin)
imageData.SetSpacing(spacing)


if vtk.VTK_MAJOR_VERSION <= 5:
    imageData.SetNumberOfScalarComponents(1)
    imageData.SetScalarTypeToDouble()
else:
    imageData.AllocateScalars(vtk.VTK_DOUBLE, 1)

dims = imageData.GetDimensions()
print(dims)
# Fill every entry of the image data with "2.0"
for z in range(dims[2]):
    for y in range(dims[1]):
        for x in range(dims[0]):
            imageData.SetScalarComponentFromDouble(x, y, z, 0, grid_z0_3d[z, y, x])

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_filename)
if vtk.VTK_MAJOR_VERSION <= 5:
    writer.SetInputConnection(imageData.GetProducerPort())
else:
    writer.SetInputData(imageData)
writer.Write()

h5filename = 'recons/' + args.dataset + '_' + args.samp_method + '_' + args.recon_method + args.tag + '_zfill.h5'
save_to_h5(grid_z0_3d, h5filename, False)
