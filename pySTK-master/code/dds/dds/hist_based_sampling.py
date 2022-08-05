from __future__ import division

import os
import random
import time

import lzma
import numpy as np
from numpy import linalg as LA
import pymp
import vtk


def hist_based_sampling(args, vals_np, x, y, z, name, j_list, filename):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    # Histogram based feature driven sampling
    count, bin_edges = np.histogram(vals_np, bins=args.nbins)

    # In[ ]:

    # count, bin_edges = np.histogram(vals_np,bins=args.nbins)
    print(count)

    # In[ ]:

    # percentage=0.02
    frac = args.percentage
    tot_samples = frac * samples
    print('looking for', tot_samples, 'samples')
    # create a dictionary first
    my_dict = dict()
    ind = 0
    for i in count:
        my_dict[ind] = i
        ind = ind + 1
    print(my_dict)

    sorted_count = sorted(my_dict, key=lambda k: my_dict[k])
    print("here:", sorted_count)

    # now distribute across bins
    target_bin_vals = int(tot_samples / args.nbins)
    print('ideal cut:', target_bin_vals)
    new_count = np.copy(count)
    ind = 0
    remain_tot_samples = tot_samples
    for i in sorted_count:
        if my_dict[i] > target_bin_vals:
            val = target_bin_vals
        else:
            val = my_dict[i]
            remain = target_bin_vals - my_dict[i]
        new_count[i] = val
        print(new_count[i], target_bin_vals)
        ind = ind + 1
        remain_tot_samples = remain_tot_samples - val
        if ind < args.nbins:
            target_bin_vals = int(remain_tot_samples / (args.nbins - ind))
    print(new_count)
    print(count)
    acceptance_hist = new_count / count
    print("acceptance histogram:", new_count / count)

    # In[ ]:

    bound_min = np.min(vals_np)
    bound_max = np.max(vals_np)

    # decide which bin this point goes to
    for i in range(samples):
        loc = vals_np[i]
        x_id = int(args.nbins * (loc - bound_min) / (bound_max - bound_min))
        if x_id == args.nbins:
            x_id = x_id - 1
        prob_vals[i] = acceptance_hist[x_id]
        # generate a random number
        rand_vals[i] = random.uniform(0, 1)
    #     if rand_vals[i]<prob_vals[i]:
    #         stencil[i] = 1

    # In[ ]:

    # inds = np.digitize(vals_np, bin_edges)
    # inds[inds>=args.nbins]=args.nbins-1
    # a = np.zeros_like(vals_np)
    # aa = acceptance_hist[inds]

    stencil[rand_vals < prob_vals] = 1

    print("actually generating samples: ", np.sum(stencil))

    # now use this stencil array to store the locations
    int_inds = np.where(stencil > 0.5)

    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    for i in int_inds[0]:
        #     print(i)
        Points.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename + "/" + filename + "_hist_sampled_" + str(args.percentage) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

    return stencil


def hist_based_sampling_pymp(args, vals_np, x, y, z, name, j_list, filename):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    # rand_vals = np.zeros_like(vals_np)
    # Histogram based feature driven sampling
    count, bin_edges = np.histogram(vals_np, bins=args.nbins)

    # In[ ]:

    # count, bin_edges = np.histogram(vals_np,bins=args.nbins)
    print(count)

    # In[ ]:

    # percentage=0.02
    frac = args.percentage
    tot_samples = frac * samples
    print('looking for', tot_samples, 'samples')
    # create a dictionary first
    my_dict = dict()
    ind = 0
    for i in count:
        my_dict[ind] = i
        ind = ind + 1
    print(my_dict)

    sorted_count = sorted(my_dict, key=lambda k: my_dict[k])
    print("here:", sorted_count)

    # now distribute across bins
    target_bin_vals = int(tot_samples / args.nbins)
    print('ideal cut:', target_bin_vals)
    new_count = np.copy(count)
    ind = 0
    remain_tot_samples = tot_samples
    for i in sorted_count:
        if my_dict[i] > target_bin_vals:
            val = target_bin_vals
        else:
            val = my_dict[i]
            remain = target_bin_vals - my_dict[i]
        new_count[i] = val
        print(new_count[i], target_bin_vals)
        ind = ind + 1
        remain_tot_samples = remain_tot_samples - val
        if ind < args.nbins:
            target_bin_vals = int(remain_tot_samples / (args.nbins - ind))
    print(new_count)
    print(count)
    acceptance_hist = new_count / count
    print("acceptance histogram:", new_count / count)

    # In[ ]:

    bound_min = np.min(vals_np)
    bound_max = np.max(vals_np)

    # decide which bin this point goes to
    prob_vals = pymp.shared.array(np.shape(prob_vals), dtype='float32')
    # rand_vals = pymp.shared.array(np.shape(rand_vals), dtype='float32')
    # decide which bin this point goes to
    with pymp.Parallel(args.nthreads) as p:
        for i in p.range(samples):
            loc = vals_np[i]
            x_id = int(args.nbins * (loc - bound_min) / (bound_max - bound_min))
            if x_id == args.nbins:
                x_id = x_id - 1
            prob_vals[i] = acceptance_hist[x_id]

    # generate a random number 
    rand_vals = np.random.random_sample(samples)

    # In[ ]:

    # inds = np.digitize(vals_np, bin_edges)
    # inds[inds>=args.nbins]=args.nbins-1
    # a = np.zeros_like(vals_np)
    # aa = acceptance_hist[inds]

    stencil[rand_vals < prob_vals] = 1

    print("actually generating samples: ", np.sum(stencil))

    # now use this stencil array to store the locations
    int_inds = np.where(stencil > 0.5)

    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    for i in int_inds[0]:
        #     print(i)
        Points.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename + "/" + filename + "_hist_pymp_" + str(args.percentage) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

    return stencil


def hist_grad_sampling(args, dim, vals_np, x, y, z, name, j_list, filename):
    start = time.time()

    print("Starting 1D acceptance histogram creation")
    # Histogram based feature driven sampling
    count, bin_edges = np.histogram(vals_np, bins=args.nbins)

    samples = np.size(vals_np)
    stencil_new = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)

    frac = args.percentage
    tot_samples = frac * samples
    print('looking for', tot_samples, 'samples')
    # create a dictionary first
    my_dict = dict()
    ind = 0
    for i in count:
        my_dict[ind] = i
        ind = ind + 1
    print(my_dict)

    sorted_count = sorted(my_dict, key=lambda k: my_dict[k])
    print("here:", sorted_count)

    # now distribute across bins
    target_bin_vals = int(tot_samples / args.nbins)
    print('ideal cut:', target_bin_vals)
    new_count = np.copy(count)
    ind = 0
    remain_tot_samples = tot_samples
    for i in sorted_count:
        if my_dict[i] > target_bin_vals:
            val = target_bin_vals
        else:
            val = my_dict[i]
            remain = target_bin_vals - my_dict[i]
        new_count[i] = val
        print(new_count[i], target_bin_vals)
        ind = ind + 1
        remain_tot_samples = remain_tot_samples - val
        if ind < args.nbins:
            target_bin_vals = int(remain_tot_samples / (args.nbins - ind))
    print(new_count)
    print(count)
    acceptance_hist = new_count / count
    print("acceptance histogram:", new_count / count)

    print("Done 1D acceptance histogram creation")

    print("Computing gradient")
    # compute gradient on the 3D data
    vals_3d = vals_np.reshape((dim[2], dim[1], dim[0]))
    print(np.shape(vals_3d))

    vals_3d_grad = np.gradient(vals_3d)
    print(np.shape(vals_3d_grad))

    vals_3d_grad_mag = LA.norm(vals_3d_grad, axis=0)

    np.shape(vals_3d_grad_mag)

    print("Starting 2D acceptance histogram creation")

    hist_2d, xedges, yedges = np.histogram2d(np.ndarray.flatten(vals_3d), np.ndarray.flatten(vals_3d_grad_mag),
                                             bins=args.nbins)

    a_1s = np.ones_like(hist_2d)
    print(np.shape((np.add(hist_2d, a_1s))))
    hist_mod = np.add(hist_2d, a_1s)
    np.log
    print(hist_2d)

    # create 2D acceptance histogram
    # for each bin in 1D histogram, we now have the counts
    # now start assiging points to 2D bins for each bin of 1D bin, start with largest gradients

    acceptance_hist_2D = np.zeros_like(hist_2d)
    acceptance_hist_2D_prob = np.zeros_like(hist_2d, dtype='float32')
    print(np.sum(count[0]), np.sum(hist_2d[:, 0]), np.sum(hist_2d[0, :]), new_count[0])

    for jj in range(args.nbins):
        dist_counts = new_count[jj]
        remain_counts = dist_counts
        cur_count = 0
        for ii in range(args.nbins - 1, -1, -1):  # looping from the most grad to least
            if remain_counts < hist_2d[jj, ii]:
                cur_count = remain_counts
            else:
                cur_count = hist_2d[jj, ii]
            acceptance_hist_2D[jj, ii] = cur_count
            remain_counts = remain_counts - cur_count
            if hist_2d[jj, ii] > 0.00000005:
                acceptance_hist_2D_prob[jj, ii] = acceptance_hist_2D[jj, ii] / hist_2d[jj, ii]

    print("expected number of samples", np.sum(np.multiply(acceptance_hist_2D_prob, hist_2d)))

    # sample using the 2D acceptance histogram
    bound_min = np.min(vals_np)
    bound_max = np.max(vals_np)

    grad_flattened = np.ndarray.flatten(vals_3d_grad_mag)

    bound_min_grad = np.min(grad_flattened)
    bound_max_grad = np.max(grad_flattened)

    prob_vals = pymp.shared.array(np.shape(prob_vals), dtype='float32')
    rand_vals = pymp.shared.array(np.shape(rand_vals), dtype='float32')
    # decide which bin this point goes to
    for i in range(samples):
        loc = vals_np[i]
        x_id = int(args.nbins * (loc - bound_min) / (bound_max - bound_min))
        if x_id == args.nbins:
            x_id = x_id - 1

        grad = grad_flattened[i]
        y_id = int(args.nbins * (grad - bound_min_grad) / (bound_max_grad - bound_min_grad))
        if y_id == args.nbins:
            y_id = y_id - 1

        prob_vals[i] = acceptance_hist_2D_prob[x_id, y_id]
        # generate a random number
        rand_vals[i] = random.uniform(0, 1)

    # stencil_new = np.zeros_like(stencil)
    stencil_new[rand_vals < prob_vals] = 1
    print("total sampled points now: ", np.sum(stencil_new))

    # now use this stencil array to store the locations
    int_inds = np.where(stencil_new > 0.5)
    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    for i in int_inds[0]:
        Points.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename + "/" + filename + "_hist_grad_" + str(args.percentage) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

    # In[ ]:

    # vals_np = VN.vtk_to_numpy(vals)
    # print(np.shape(vals_np))
    # vo2_arr = vals_np[stencil_new>0.5]
    # vo2_arr.tofile(filename+"/"+filename+"_hist_grad_"+str(percentage)+".bin")

    return stencil_new


def hist_grad_sampling_pymp(args, dim, vals_np, x, y, z, name, j_list, filename):
    start = time.time()

    print("Starting 1D acceptance histogram creation")
    # Histogram based feature driven sampling
    count, bin_edges = np.histogram(vals_np, bins=args.nbins)

    # print(count)

    # In[ ]:

    # percentage=0.02
    samples = np.size(vals_np)
    stencil_new = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    # rand_vals = np.zeros_like(vals_np)

    frac = args.percentage
    tot_samples = frac * samples
    print('looking for', tot_samples, 'samples')
    # create a dictionary first
    my_dict = dict()
    ind = 0
    for i in count:
        my_dict[ind] = i
        ind = ind + 1
    print(my_dict)

    sorted_count = sorted(my_dict, key=lambda k: my_dict[k])
    print("here:", sorted_count)

    # now distribute across bins
    target_bin_vals = int(tot_samples / args.nbins)
    print('ideal cut:', target_bin_vals)
    new_count = np.copy(count)
    ind = 0
    remain_tot_samples = tot_samples
    for i in sorted_count:
        if my_dict[i] > target_bin_vals:
            val = target_bin_vals
        else:
            val = my_dict[i]
            remain = target_bin_vals - my_dict[i]
        new_count[i] = val
        print(new_count[i], target_bin_vals)
        ind = ind + 1
        remain_tot_samples = remain_tot_samples - val
        if ind < args.nbins:
            target_bin_vals = int(remain_tot_samples / (args.nbins - ind))
    print(new_count)
    print(count)
    acceptance_hist = new_count / count
    print("acceptance histogram:", new_count / count)

    print("Done 1D acceptance histogram creation")

    print("Computing gradient")
    # compute gradient on the 3D data
    vals_3d = vals_np.reshape((dim[2], dim[1], dim[0]))
    print(np.shape(vals_3d))

    vals_3d_grad = np.gradient(vals_3d)
    print(np.shape(vals_3d_grad))

    vals_3d_grad_mag = LA.norm(vals_3d_grad, axis=0)

    np.shape(vals_3d_grad_mag)

    # dataset = getVtkImageData(data.GetOrigin(),data.GetDimensions(),data.GetExtent(),data.GetSpacing())

    # dataset.AllocateScalars(vtk.VTK_DOUBLE, 1)

    # dims = dataset.GetDimensions()

    # # Fill every entry of the image data with gradient information
    # for zz in range(dim[2]):
    #     for yy in range(dim[1]):
    #         for xx in range(dim[0]):
    #             dataset.SetScalarComponentFromDouble(xx, yy, zz, 0, vals_3d_grad_mag[zz,yy,xx])

    # writer = vtk.vtkXMLImageDataWriter()
    # writer.SetFileName('asteroid_gradient.vti')
    # if vtk.VTK_MAJOR_VERSION <= 5:
    #     writer.SetInputConnection(dataset.GetProducerPort())
    # else:
    #     writer.SetInputData(dataset)
    # writer.Write()

    # In[ ]:
    print("Starting 2D acceptance histogram creation")

    hist_2d, xedges, yedges = np.histogram2d(np.ndarray.flatten(vals_3d), np.ndarray.flatten(vals_3d_grad_mag),
                                             bins=args.nbins)

    # In[ ]:

    a_1s = np.ones_like(hist_2d)
    print(np.shape((np.add(hist_2d, a_1s))))
    hist_mod = np.add(hist_2d, a_1s)
    np.log
    # plt.figure()
    # plt.imshow(np.log10(np.add(hist_2d,a_1s)),origin='lower',cmap='Blues',vmax=4)
    # plt.show()
    print(hist_2d)

    # create 2D acceptance histogram
    # for each bin in 1D histogram, we now have the counts
    # now start assiging points to 2D bins for each bin of 1D bin, start with largest gradients

    acceptance_hist_2D = np.zeros_like(hist_2d)
    acceptance_hist_2D_prob = np.zeros_like(hist_2d, dtype='float32')
    print(np.sum(count[0]), np.sum(hist_2d[:, 0]), np.sum(hist_2d[0, :]), new_count[0])

    for jj in range(args.nbins):
        dist_counts = new_count[jj]
        remain_counts = dist_counts
        cur_count = 0
        for ii in range(args.nbins - 1, -1, -1):  # looping from the most grad to least
            if remain_counts < hist_2d[jj, ii]:
                cur_count = remain_counts
            else:
                cur_count = hist_2d[jj, ii]
            acceptance_hist_2D[jj, ii] = cur_count
            remain_counts = remain_counts - cur_count
            if hist_2d[jj, ii] > 0.00000005:
                acceptance_hist_2D_prob[jj, ii] = acceptance_hist_2D[jj, ii] / hist_2d[jj, ii]

    # In[ ]:

    # print(np.sum(count[0]), np.sum(acceptance_hist_2D[0,:]),new_count[0])
    # print(np.sum(count[:]), np.sum(acceptance_hist_2D[:,:]),np.sum(new_count[:]))

    # # In[ ]:

    # print(np.sum(count[:]), np.sum(acceptance_hist_2D[1,:]),np.sum(new_count[1]))
    # print(hist_2d[1,:])
    # acceptance_hist_2D_prob[1,:]

    # In[ ]:

    print("expected number of samples", np.sum(np.multiply(acceptance_hist_2D_prob, hist_2d)))

    # In[ ]:

    # sample using the 2D acceptance histogram
    bound_min = np.min(vals_np)
    bound_max = np.max(vals_np)

    grad_flattened = np.ndarray.flatten(vals_3d_grad_mag)

    bound_min_grad = np.min(grad_flattened)
    bound_max_grad = np.max(grad_flattened)

    prob_vals = pymp.shared.array(np.shape(prob_vals), dtype='float32')
    # rand_vals = pymp.shared.array(np.shape(rand_vals), dtype='float32')
    # decide which bin this point goes to
    with pymp.Parallel(args.nthreads) as p:
        for i in p.range(samples):
            loc = vals_np[i]
            x_id = int(args.nbins * (loc - bound_min) / (bound_max - bound_min))
            if x_id == args.nbins:
                x_id = x_id - 1

            grad = grad_flattened[i]
            y_id = int(args.nbins * (grad - bound_min_grad) / (bound_max_grad - bound_min_grad))
            if y_id == args.nbins:
                y_id = y_id - 1

            prob_vals[i] = acceptance_hist_2D_prob[x_id, y_id]

    # generate a random number
    rand_vals = np.random.random_sample(samples)

    # In[ ]:

    # stencil_new = np.zeros_like(stencil)
    stencil_new[rand_vals < prob_vals] = 1
    print("total sampled points now: ", np.sum(stencil_new))

    # In[ ]:

    # now use this stencil array to store the locations
    int_inds = np.where(stencil_new > 0.5)
    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    for i in int_inds[0]:
        #     print(i)
        Points.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename + "/" + filename + "_hist_grad_pymp_" + str(args.percentage) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

    return stencil_new

    # In[ ]:

    # vals_np = VN.vtk_to_numpy(vals)
    # print(np.shape(vals_np))
    # vo2_arr = vals_np[stencil_new>0.5]
    # vo2_arr.tofile(filename+"/"+filename+"_hist_grad_"+str(args.percentage)+".bin")


def hist_grad_rand_sampling_pymp(args, dim, vals_np, x, y, z, name, j_list, filename, grad_power=1):
    start = time.time()

    print("Starting 1D acceptance histogram creation")
    # Histogram based feature driven sampling
    count, bin_edges = np.histogram(vals_np, bins=args.nbins)
    samples = np.size(vals_np)
    stencil_new = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    print('looking for', tot_samples, 'samples')
    # create a dictionary first
    my_dict = dict()
    ind = 0
    for i in count:
        my_dict[ind] = i
        ind = ind + 1
    # print(my_dict)

    sorted_count = sorted(my_dict, key=lambda k: my_dict[k])
    # now distribute across bins
    target_bin_vals = int(tot_samples / args.nbins)
    # print('ideal cut:',target_bin_vals)
    new_count = np.copy(count)
    ind = 0
    remain_tot_samples = tot_samples
    for i in sorted_count:
        if my_dict[i] > target_bin_vals:
            val = target_bin_vals
        else:
            val = my_dict[i]
            remain = target_bin_vals - my_dict[i]
        new_count[i] = val
        # print(new_count[i], target_bin_vals)
        ind = ind + 1
        remain_tot_samples = remain_tot_samples - val
        if ind < args.nbins:
            target_bin_vals = int(remain_tot_samples / (args.nbins - ind))
    acceptance_hist = np.zeros(args.nbins, dtype="float32")
    for i in range(args.nbins):
        if count[i] == 0:
            acceptance_hist[i] = 0.0
        else:
            acceptance_hist[i] = new_count[i] / count[i]

    # acceptance_hist = new_count/count
    print("acceptance histogram:", acceptance_hist)

    print("Done 1D acceptance histogram creation")

    print("Computing gradient")
    # compute gradient on the 3D data
    vals_3d = vals_np.reshape((dim[2], dim[1], dim[0]))
    vals_3d_grad = np.gradient(vals_3d)
    vals_3d_grad_mag = LA.norm(vals_3d_grad, axis=0)

    print("Starting 2D acceptance histogram creation")
    hist_2d, xedges, yedges = np.histogram2d(np.ndarray.flatten(vals_3d), np.ndarray.flatten(vals_3d_grad_mag),
                                             bins=args.nbins)

    ybin_centers = np.zeros(args.nbins)

    for bc in range(args.nbins):
        ybin_centers[bc] = (yedges[bc] + yedges[bc + 1]) / 2.0

    bc_probs = ybin_centers / np.sum(ybin_centers)

    cur_bc = np.zeros_like(bc_probs)

    # print("bc probs: ", bc_probs, np.sum(bc_probs))
    print("total samples should be picked: ", np.sum(new_count))

    # In[ ]:

    a_1s = np.ones_like(hist_2d)
    hist_mod = np.add(hist_2d, a_1s)

    # create 2D acceptance histogram
    # for each bin in 1D histogram, we now have the counts
    # now start assiging points to 2D bins for each bin of 1D bin, start with largest gradients

    acceptance_hist_2D = np.zeros_like(hist_2d)
    acceptance_hist_2D_prob = np.zeros_like(hist_2d, dtype='float32')
    # print(np.sum(count[0]),np.sum(hist_2d[:,0]), np.sum(hist_2d[0,:]),new_count[0])

    for jj in range(args.nbins):
        dist_counts = new_count[jj]
        inds = np.where(hist_2d[jj, :] > 0.000001)
        cur_bc = np.zeros_like(bc_probs)
        cur_bc[inds] = ybin_centers[inds]
        if np.sum(cur_bc) == 0:
            cur_bc_prob = 0
        else:
            cur_bc_prob = np.power(cur_bc, 1.0 / grad_power) / np.sum(cur_bc)
        remain_counts = dist_counts
        cur_count = 0
        samp_vals = cur_bc_prob * dist_counts
        samp_vals = np.minimum(samp_vals, hist_2d[jj, :])
        # print("wanted samples: ",new_count[jj], "getting sample: ", samp_vals, np.sum(samp_vals), "from: ",hist_2d[jj,:],np.sum(hist_2d[jj,:]))
        remain_samp_count = new_count[jj] - np.sum(samp_vals)
        remaining_samp_vals = hist_2d[jj, :] - samp_vals
        # sort again
        # create a dictionary first
        my_dict1 = dict()
        ind = 0
        for i in remaining_samp_vals:
            my_dict1[ind] = i
            ind = ind + 1
        # print(my_dict1)
        new_sorted_count = sorted(my_dict1, key=lambda k: my_dict1[k])
        # print("sorted count:",new_sorted_count)
        ind = 0
        target_bin_vals = remain_samp_count / (args.nbins - ind)
        for i in new_sorted_count:
            if my_dict1[i] > target_bin_vals:
                val = target_bin_vals
            else:
                val = my_dict1[i]
                # remain = target_bin_vals-my_dict[i]
            samp_vals[i] = samp_vals[i] + val
            # print(samp_vals[i], target_bin_vals)
            ind = ind + 1
            remain_samp_count = remain_samp_count - val
            if ind < args.nbins:
                target_bin_vals = remain_samp_count / (args.nbins - ind)
        # print("Getting samples now: ",samp_vals,np.sum(samp_vals))
        # distribute these many samples over the bins equally
        for ii in range(args.nbins):
            if hist_2d[jj, ii] > 0.00000001:
                acceptance_hist_2D_prob[jj, ii] = samp_vals[ii] / hist_2d[jj, ii]
        acceptance_hist_2D_prob[acceptance_hist_2D_prob > 1.0] = 1.0

    print("expected number of samples", np.sum(np.multiply(acceptance_hist_2D_prob, hist_2d)))
    # sample using the 2D acceptance histogram
    bound_min = np.min(vals_np)
    bound_max = np.max(vals_np)

    grad_flattened = np.ndarray.flatten(vals_3d_grad_mag)

    bound_min_grad = np.min(grad_flattened)
    bound_max_grad = np.max(grad_flattened)

    prob_vals = pymp.shared.array(np.shape(prob_vals), dtype='float32')
    # rand_vals = pymp.shared.array(np.shape(rand_vals), dtype='float32')
    # decide which bin this point goes to
    with pymp.Parallel(args.nthreads) as p:
        for i in p.range(samples):
            loc = vals_np[i]
            x_id = int(args.nbins * (loc - bound_min) / (bound_max - bound_min))
            if x_id == args.nbins:
                x_id = x_id - 1

            grad = grad_flattened[i]
            y_id = int(args.nbins * (grad - bound_min_grad) / (bound_max_grad - bound_min_grad))
            if y_id == args.nbins:
                y_id = y_id - 1

            prob_vals[i] = acceptance_hist_2D_prob[x_id, y_id]

    # generate a random number
    rand_vals = np.random.random_sample(samples)
    stencil_new[rand_vals < prob_vals] = 1
    print("total sampled points now: ", np.sum(stencil_new), "from: ", np.sum(prob_vals))

    # now use this stencil array to store the locations
    int_inds = np.where(stencil_new > 0.5)
    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    for i in int_inds[0]:
        #     print(i)
        Points.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    print("writing file:", filename + "/" + filename + "_hist_grad_rand_pymp_" + str(args.percentage) + ".vtp")
    writer.SetFileName(filename + "/" + filename + "_hist_grad_rand_pymp_" + str(args.percentage) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

    # save the indices
    np.save(filename + "/" + filename + "_hist_grad_rand_pymp_" + str(args.percentage) + ".npy", int_inds[0])

    # compress the indices
    outfile_name_lzma = filename + "/" + filename + "_hist_grad_rand_pymp_" + str(args.percentage) + ".lzma"
    file = lzma.LZMAFile(outfile_name_lzma, "wb")
    file.write(int_inds[0])
    file.close()
    index_size = os.path.getsize(outfile_name_lzma)

    # print stats
    float_size = 4
    orig_total_size = dim[0] * dim[1] * dim[2] * float_size
    sampled_total_size = np.sum(stencil_new) * float_size
    sampled_final_size = sampled_total_size + index_size
    print("printing stats:", orig_total_size, sampled_total_size, index_size, index_size / sampled_total_size,
          sampled_final_size, sampled_final_size / orig_total_size)

    return stencil_new


def grad_based_sampling(args, dim, vals_np, x, y, z, name, j_list, filename):
    start = time.time()
    # grad based sampling

    # compute gradient on the 3D data
    vals_3d = vals_np.reshape((dim[2], dim[1], dim[0]))
    print(np.shape(vals_3d))

    vals_3d_grad = np.gradient(vals_3d)
    print(np.shape(vals_3d_grad))

    vals_3d_grad_mag = LA.norm(vals_3d_grad, axis=0)

    np.shape(vals_3d_grad_mag)

    grad_hist, edges = np.histogram(np.ndarray.flatten(vals_3d_grad_mag), bins=args.nbins)

    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil_grad = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)

    # In[25]:

    # create 1D acceptance histogram
    # now start assiging points to 1D bins for each bin of 1D bin, start with largest gradients

    acceptance_hist_grad = np.zeros_like(grad_hist)
    acceptance_hist_grad_prob = np.zeros_like(grad_hist, dtype='float32')

    dist_counts = int(samples * args.percentage)
    remain_counts = dist_counts
    cur_count = 0
    for ii in range(args.nbins - 1, -1, -1):  # looping from the most grad to least
        # print(ii)
        if remain_counts < grad_hist[ii]:
            cur_count = remain_counts
        else:
            cur_count = grad_hist[ii]
        acceptance_hist_grad[ii] = cur_count
        remain_counts = remain_counts - cur_count
        if grad_hist[ii] > 0.00000005:
            acceptance_hist_grad_prob[ii] = acceptance_hist_grad[ii] / grad_hist[ii]

    # In[32]:

    print(np.shape(grad_hist), grad_hist, acceptance_hist_grad_prob)

    # In[27]:

    vals_3d_grad_mag_flattened = np.ndarray.flatten(vals_3d_grad_mag)
    bound_min = np.min(vals_3d_grad_mag_flattened)
    bound_max = np.max(vals_3d_grad_mag_flattened)

    # decide which bin this point goes to
    for i in range(samples):
        loc = vals_3d_grad_mag_flattened[i]
        x_id = int(args.nbins * (loc - bound_min) / (bound_max - bound_min))
        if x_id == args.nbins:
            x_id = x_id - 1
        prob_vals[i] = acceptance_hist_grad_prob[x_id]
        # generate a random number
        rand_vals[i] = random.uniform(0, 1)

    # In[35]:

    np.histogram(rand_vals, bins=args.nbins)

    # In[37]:

    np.size(np.where(rand_vals < prob_vals))

    # In[40]:

    stencil_grad = np.zeros_like(vals_np)
    stencil_grad[rand_vals < prob_vals] = 1

    # In[41]:

    np.sum(stencil_grad)

    # In[42]:

    # now use this stencil array to store the locations
    int_inds = np.where(stencil_grad > 0.5)

    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    for i in int_inds[0]:
        #     print(i)
        Points.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename + "/" + filename + "_grad_sampled_" + str(args.percentage) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

    return stencil_grad


def grad_based_sampling_pymp(args, dim, vals_np, x, y, z, name, j_list, filename):
    start = time.time()
    # grad based sampling

    # compute gradient on the 3D data
    vals_3d = vals_np.reshape((dim[2], dim[1], dim[0]))
    print(np.shape(vals_3d))

    vals_3d_grad = np.gradient(vals_3d)
    print(np.shape(vals_3d_grad))

    vals_3d_grad_mag = LA.norm(vals_3d_grad, axis=0)

    np.shape(vals_3d_grad_mag)

    grad_hist, edges = np.histogram(np.ndarray.flatten(vals_3d_grad_mag), bins=args.nbins)

    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil_grad = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    # rand_vals = np.zeros_like(vals_np)

    # In[25]:

    # create 1D acceptance histogram
    # now start assiging points to 1D bins for each bin of 1D bin, start with largest gradients

    acceptance_hist_grad = np.zeros_like(grad_hist)
    acceptance_hist_grad_prob = np.zeros_like(grad_hist, dtype='float32')

    dist_counts = int(samples * args.percentage)
    remain_counts = dist_counts
    cur_count = 0
    for ii in range(args.nbins - 1, -1, -1):  # looping from the most grad to least
        # print(ii)
        if remain_counts < grad_hist[ii]:
            cur_count = remain_counts
        else:
            cur_count = grad_hist[ii]
        acceptance_hist_grad[ii] = cur_count
        remain_counts = remain_counts - cur_count
        if grad_hist[ii] > 0.00000005:
            acceptance_hist_grad_prob[ii] = acceptance_hist_grad[ii] / grad_hist[ii]

    # In[32]:

    print(np.shape(grad_hist), grad_hist, acceptance_hist_grad_prob)

    # In[27]:

    vals_3d_grad_mag_flattened = np.ndarray.flatten(vals_3d_grad_mag)
    bound_min = np.min(vals_3d_grad_mag_flattened)
    bound_max = np.max(vals_3d_grad_mag_flattened)

    # decide which bin this point goes to
    prob_vals = pymp.shared.array(np.shape(prob_vals), dtype='float32')
    # rand_vals = pymp.shared.array(np.shape(rand_vals), dtype='float32')
    # decide which bin this point goes to
    with pymp.Parallel(args.nthreads) as p:
        for i in p.range(samples):
            loc = vals_3d_grad_mag_flattened[i]
            x_id = int(args.nbins * (loc - bound_min) / (bound_max - bound_min))
            if x_id == args.nbins:
                x_id = x_id - 1
            prob_vals[i] = acceptance_hist_grad_prob[x_id]

    # generate a random number
    rand_vals = np.random.random_sample(samples)

    # In[35]:

    np.histogram(rand_vals, bins=args.nbins)

    # In[37]:

    np.size(np.where(rand_vals < prob_vals))

    # In[40]:

    stencil_grad = np.zeros_like(vals_np)
    stencil_grad[rand_vals < prob_vals] = 1

    # In[41]:

    np.sum(stencil_grad)

    # In[42]:

    # now use this stencil array to store the locations
    int_inds = np.where(stencil_grad > 0.5)

    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    for i in int_inds[0]:
        #     print(i)
        Points.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename + "/" + filename + "_grad_sampled_pymp_" + str(args.percentage) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

    return stencil_grad


def random_sampling(args, vals_np, x, y, z, name, j_list, filename):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    print('looking for', tot_samples, 'samples')

    prob_vals[:] = args.percentage
    rand_vals = np.random.random_sample(samples)

    stencil[rand_vals < prob_vals] = 1
    print("Collecting samples:", np.sum(stencil))

    # now use this stencil array to store the locations
    int_inds = np.where(stencil > 0.5)

    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    # x_3d = x.reshape((dim[2],dim[1],dim[0]))
    # y_3d = y.reshape((dim[2],dim[1],dim[0]))
    # z_3d = z.reshape((dim[2],dim[1],dim[0]))

    for i in int_inds[0]:
        #     print(i)
        Points.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    print("writing file:", filename + "/" + filename + "_random_pymp_" + str(args.percentage) + ".vtp")
    writer.SetFileName(filename + "/" + filename + "_random_pymp_" + str(args.percentage) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

    return stencil


def fused_sampling(args, dim, vals_np, x, y, z, name, j_list, filename, method1, method2, frac1):
    start = time.time()
    # find which methods to fuse
    frac2 = 1 - frac1

    if method1 == 'hist':
        stencil1 = hist_based_sampling_pymp(frac1 * args.percentage, vals_np, x, y, z, name, j_list, filename)
    elif method1 == 'grad':
        stencil1 = grad_based_sampling_pymp(frac1 * args.percentage, dim, vals_np, x, y, z, name, j_list, filename)
    elif method1 == 'hist_grad':
        stencil1 = hist_grad_sampling_pymp(frac1 * args.percentage, dim, vals_np, x, y, z, name, j_list, filename)
    elif method1 == 'hist_grad_random':
        stencil1 = hist_grad_rand_sampling_pymp(frac1 * args.percentage, dim, vals_np, x, y, z, name, j_list, filename)
    elif method1 == 'random':
        stencil1 = random_sampling(frac1 * args.percentage, vals_np, x, y, z, name, j_list, filename)

    if method2 == 'hist':
        stencil2 = hist_based_sampling_pymp(frac2 * args.percentage, vals_np, x, y, z, name, j_list, filename)
    elif method2 == 'grad':
        stencil2 = grad_based_sampling_pymp(frac2 * args.percentage, dim, vals_np, x, y, z, name, j_list, filename)
    elif method2 == 'hist_grad':
        stencil2 = hist_grad_sampling_pymp(frac2 * args.percentage, dim, vals_np, x, y, z, name, j_list, filename)
    elif method2 == 'hist_grad_random':
        stencil2 = hist_grad_rand_sampling_pymp(frac2 * args.percentage, dim, vals_np, x, y, z, name, j_list, filename)
    elif method2 == 'random':
        stencil2 = random_sampling(frac2 * args.percentage, vals_np, x, y, z, name, j_list, filename)

    stencil = stencil1 + stencil2

    # now use this stencil array to store the locations
    int_inds = np.where(stencil > 0.5)

    Points = vtk.vtkPoints()
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    # x_3d = x.reshape((dim[2],dim[1],dim[0]))
    # y_3d = y.reshape((dim[2],dim[1],dim[0]))
    # z_3d = z.reshape((dim[2],dim[1],dim[0]))

    for i in int_inds[0]:
        #     print(i)
        Points.InsertNextPoint(x[i], y[i], z[i])
        val_arr.InsertNextValue(vals_np[i])

    # add boundary points
    for j in j_list:
        Points.InsertNextPoint(x[j], y[j], z[j])
        val_arr.InsertNextValue(vals_np[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(
        filename + "/" + filename + "_mix_" + method1 + "" + str(frac1) + method2 + "" + str(frac2) + "_sampled_" + str(
            args.percentage) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

    return stencil