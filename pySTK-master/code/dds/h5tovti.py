import argparse

import h5py
import numpy as np
import vtk
from vtk.util import numpy_support


## arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', action="store", required=True, help="input file name")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    f = h5py.File(args.input)
    print("read in hdf5 file")
    grid_3d = f["native_fields"]["baryon_density"][:]
    # print(grid_3d)
    print("extracted 3d grid")
    print("===== Processing the unchanged data")
    XDIM, YDIM, ZDIM = grid_3d.shape
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(XDIM, YDIM, ZDIM)

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
                imageData.SetScalarComponentFromDouble(x, y, z, 0, grid_3d[z, y, x])

    print("point data: ", imageData.GetPointData())
    name = imageData.GetPointData().GetArrayName(0)
    print("name: ", name)
    vals = imageData.GetPointData().GetArray(name)
    print("vals: ", vals)
    vals_np = numpy_support.vtk_to_numpy(vals)
    print("vals_np shape: ", np.shape(vals_np))

    writer = vtk.vtkXMLImageDataWriter()
    output_filename = args.input[:-3] + ".vti"
    writer.SetFileName(output_filename)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInputConnection(imageData.GetProducerPort())
    else:
        writer.SetInputData(imageData)
    print("writing image to file: ", output_filename)
    writer.Write()

    # Writing the log version of th edata

    print("===== Processing the log-scaled data")
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(XDIM, YDIM, ZDIM)
    print("scaling the data")
    grid_3d = np.log(grid_3d)

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
                imageData.SetScalarComponentFromDouble(x, y, z, 0, grid_3d[z, y, x])

    print("point data: ", imageData.GetPointData())
    name = imageData.GetPointData().GetArrayName(0)
    print("name: ", name)
    vals = imageData.GetPointData().GetArray(name)
    print("vals: ", vals)
    vals_np = numpy_support.vtk_to_numpy(vals)
    print("vals_np shape: ", np.shape(vals_np))

    writer = vtk.vtkXMLImageDataWriter()
    output_filename = args.input[:-3] + "_log.vti"
    writer.SetFileName(output_filename)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInputConnection(imageData.GetProducerPort())
    else:
        writer.SetInputData(imageData)
    print("writing log image to file: ", output_filename)
    writer.Write()
    print("done")

