# coding: utf-8
from __future__ import division

import argparse
import lzma
import os

import pymp
import vtk
from vtk.util import numpy_support as VN

from dds.hist_based_sampling import *
from dds.graph_based_sampling import *


def getVtkImageData(origin, dimensions, extents, spacing):
    localDataset = vtk.vtkImageData()
    localDataset.SetOrigin(origin)
    localDataset.SetDimensions(dimensions)
    localDataset.SetExtent(extents)
    localDataset.SetSpacing(spacing)
    # print(origin,dimensions,spacing)
    return localDataset  


def stencil_to_vtk(stencil, name, x, y, z, values, j_list, fill):
    # now use this stencil array to store the locations
    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    if fill:
        for i in range(0, np.size(stencil)):
            Points.InsertNextPoint(x[i], y[i], z[i])
            if stencil[i] <= 0.5:
                val_arr.InsertNextValue(0)
            else:
                val_arr.InsertNextValue(values[i])
    else:
        indices = np.where(stencil > 0.5)
        for i in indices[0]:
            Points.InsertNextPoint(x[i], y[i], z[i])
            val_arr.InsertNextValue(values[i])

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(values[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    return polydata
# End of stencil_to_vtk()


def save_sample(polydata, filename, sampling_ratio, sample_name):
    writer = vtk.vtkXMLPolyDataWriter()
    output_filename = filename + "/" + filename + "_" + sample_name + "_" + str(sampling_ratio) + ".vtp"
    print("writing file:", output_filename)
    writer.SetFileName(output_filename)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()
# End of save_sample()


def load_vti(filename):
    if file_extension == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif file_extension == '.vtk':
        reader = vtk.vtkGenericDataObjectReader()
    elif file_extension == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif file_extension == '.vti':
        reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(args.input)
    reader.Update()

    if not os.path.exists(filename):
        os.makedirs(filename)

    data = reader.GetOutput()
    dim = data.GetDimensions()

    x = np.zeros(data.GetNumberOfPoints())
    y = np.zeros(data.GetNumberOfPoints())
    z = np.zeros(data.GetNumberOfPoints())

    x = pymp.shared.array(np.shape(x), dtype='float32')
    y = pymp.shared.array(np.shape(y), dtype='float32')
    z = pymp.shared.array(np.shape(z), dtype='float32')
    # decide which bin this point goes to
    with pymp.Parallel(args.nthreads) as p:
        for i in p.range(data.GetNumberOfPoints()):
            x[i], y[i], z[i] = data.GetPoint(i)

    vtk_array_name = data.GetPointData().GetArrayName(0)
    vals = data.GetPointData().GetArray(vtk_array_name)
    vals_np = VN.vtk_to_numpy(vals)
    return vals_np, x, y , z, dim
# End of load_vti()


## arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', action="store", required=False, default="yA31_v02_300x300x300_99.vtk",
                    help="input file name")
parser.add_argument('--output', action="store", required=False, help="output folder name")
parser.add_argument('--percentage', action="store", type=float, default=0.001, required=False,
                    help="what fraction of samples to keep")
parser.add_argument('--nbins', action="store", required=False, type=int, default=32, help="how many bins to use")
parser.add_argument('--nthreads', action="store", required=False, type=int, help="how many threads to use")
parser.add_argument('--method', action="store", required=True,
                    help="which sampling method to use. hist, grad, hist_grad, random, mixed")
parser.add_argument('--method1', action="store", required=False, help="for mix: method1=hist, grad, hist_grad, random ")
parser.add_argument('--method2', action="store", required=False, help="for mix: method2=hist, grad, hist_grad, random ")
parser.add_argument('--frac1', action="store", required=False,
                    help="for mix: fraction for method1= must be between 0 and 1")
parser.add_argument('--fill', action="store_true", required=False, default=False,
                    help="If set, non-sampled points get filled with 0")
parser.add_argument('--grad_power', action="store", required=False, type=float, default=1.0,
                    help="For hist_grad_rand_sampling: change gradient effect. higher means more gradient effect, lower means more random effect")
parser.add_argument("--sparse", action="store", required=False, default=0.25, type=float,
                    help="For grpah-based methods; if set to a value > 0, will create a sparse graph by removing edges to points with value less than 'sparse' percentile.")
parser.add_argument("--contrast_points", action="store", required=False, type=float, default=0.00,
                    help="The amount of low value points to include for contrast purposes. Default is 0.00 or 0 percent of the total sampled points.")
parser.add_argument("--logscaled", action="store_true", required=False, default=False, help="If set, will assume data is in log scale")
args = parser.parse_args()
print(args)


outPath = getattr(args, 'output')
args.percentage = getattr(args, 'percentage')

# method = getattr(args, 'method')

method1 = getattr(args, 'method1')
method2 = getattr(args, 'method2')
frac1 = getattr(args, 'frac1')

grad_power = getattr(args, 'grad_power')

filename, file_extension = os.path.splitext(os.path.basename(args.input))

if filename == None:
    outPath = filename

if args.method == "mixed":
    if method1 == None:
        method1 = "hist"
    if method2 == None:
        method2 = "grad"
    if frac1 == None:
        frac1 = 0.5
    else:
        frac1 = float(frac1)

if file_extension == '.vtp':
    reader = vtk.vtkXMLPolyDataReader()
elif file_extension == '.vtk':
    reader = vtk.vtkGenericDataObjectReader()
elif file_extension == '.vtu':
    reader = vtk.vtkXMLUnstructuredGridReader()
elif file_extension == '.vti':
    reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(args.input)
reader.Update()

if not os.path.exists(filename):
    os.makedirs(filename)

data = reader.GetOutput()
dim = data.GetDimensions()

x = np.zeros(data.GetNumberOfPoints())
y = np.zeros(data.GetNumberOfPoints())
z = np.zeros(data.GetNumberOfPoints())

x = pymp.shared.array(np.shape(x), dtype='float32')
y = pymp.shared.array(np.shape(y), dtype='float32')
z = pymp.shared.array(np.shape(z), dtype='float32')
# decide which bin this point goes to
with pymp.Parallel(args.nthreads) as p:
    for i in p.range(data.GetNumberOfPoints()):
        x[i], y[i], z[i] = data.GetPoint(i)

vtk_array_name = data.GetPointData().GetArrayName(0)
vals = data.GetPointData().GetArray(vtk_array_name)
vals_np = VN.vtk_to_numpy(vals)
print(np.shape(vals_np))

## create boundary points
A = [0, dim[0] - 1]
B = [0, dim[1] - 1]
C = [0, dim[2] - 1]

j_vals = (((zval * dim[0] * dim[1]) + (yval * dim[0]) + xval) for xval in A for yval in B for zval in C)
boundary_list = np.asanyarray(list(j_vals))

print("Calling the function::")

start = time.time()

stencil = None
if args.method == "hist":
    stencil = hist_based_sampling_pymp(args, vals_np, x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "grad":
    stencil = grad_based_sampling_pymp(args, dim, vals_np, x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "grad_lcc":
    stencil = gradient_lcc_sampling(args, dim, vals_np)
elif args.method == "hist_grad":
    stencil = hist_grad_sampling_pymp(args, dim, vals_np, x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "hist_grad_random":
    stencil = hist_grad_rand_sampling_pymp(args, dim, vals_np, x, y, z, vtk_array_name, boundary_list, filename, args.grad_power)
elif args.method == "hist_lcc":
    stencil = histogram_lcc_sampling(args, dim, vals_np)
elif args.method == "random":
    stencil = random_sampling(args, vals_np, x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "lcc":
    stencil = lcc_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "mixed":
    stencil = fused_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename, method1, method2, frac1)
elif args.method == "max":
    stencil = max_sampling(args, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "similarity":
    stencil = similarity_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "max_neighbor":
    stencil = max_neighbor_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "random_walk":
    stencil = random_walk_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "value_weighted_random_walk":
    stencil = value_weighted_random_walk_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "similarity_weighted_random_walk":
    stencil = similarity_weighted_random_walk_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "max_seeded_random_walk":
    stencil = max_seeded_random_walk_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
# elif method == "max_seeded_random_walk_v2":
#     max_seeded_random_walk_sampling_v2(args, dim, vals_np, x, y, z, name, j_list, filename)
elif args.method == "max_seeded_weighted_random_walk":
    stencil = max_seeded_weighted_random_walk_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
elif args.method == "complex_max_seeded_random_walk":
    stencil = complex_max_seeded_random_walk_sampling(args, dim, vals_np)  # , x, y, z, vtk_array_name, boundary_list, filename)
else:
    print("Unknown sampling type. Exiting")
    

polydata = stencil_to_vtk(stencil, vtk_array_name, x, y, z, vals_np, boundary_list, args.fill)
save_sample(polydata, filename, args.percentage, "{0}".format(args.method))

end = time.time()
print("total time taken:", end - start)
## write the time out to a file
# filename+"/"+filename+"_random_pymp_"+str(args.percentage)+".vtp"
if not os.path.exists("timing"):
    os.makedirs("timing")
text_file = open("timing" + "/" + "timing_" + args.method + "_" + filename + "_" + str(args.percentage) + ".txt", "w")
text_file.write("total time taken: %s seconds" % str(end - start))
text_file.close()
