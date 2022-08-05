#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import seaborn as sns
from termcolor import colored
import vtk
from sklearn.neighbors import NearestNeighbors
from vtk.util import numpy_support as VN
import os
import time
import sys
sys.path.append('.')
import FeatureSampler as FS
import importlib
importlib.reload(FS)
import pickle
import argparse
#import lzma
import shutil

def dump_with_pickle(pyObj, filename):
    fobj1 = open(filename, 'wb')
    pickle.dump(pyObj, fobj1)
    fobj1.close()
    
def load_with_pickle(filename):
    fobj1 = open(filename, 'rb')
    pyObj = pickle.load(fobj1)
    fobj1.close()
    return pyObj

def run(infile, fb_sr,rand_sr, sampling_method, args):

    #sampling_method_val = np.int(enum_dict[sampling_method])
    #print('sampling_method_val=',sampling_method_val)

    print('Reading data.')
    dm = FS.DataManager(infile, 0) 

    data = dm.get_datafield()

    XDIM, YDIM, ZDIM = dm.get_dimension()
    d_spacing = dm.get_spacing()
    d_origin = dm.get_origin()
    d_extent = dm.get_extent()
    print(dm)

    vhist_nbins = 128
    ghist_nbins = 512

    #rand_sr = 0.0
    #fb_sr = 0.005
    sampling_rate = fb_sr+rand_sr

    #get the global acceptance histogram (provide sample rate) and min/max
    ghist = dm.get_acceptance_histogram(fb_sr, ghist_nbins)
    gmin = dm.getMin()
    gmax = dm.getMax()

    bm_paramter_list = [32, 32, 32, XDIM, YDIM, ZDIM, d_origin, d_spacing, d_extent]

    bm = FS.BlockManager('regular', bm_paramter_list)

    nob = bm.numberOfBlocks()

    sm = FS.SampleManager()


    sm.set_global_properties(ghist, gmin, gmax)


    list_sampled_lid = []
    list_sampled_data = []

    array_void_hist = np.zeros((nob, vhist_nbins))
    array_ble = np.zeros((nob,))
    array_delta = np.zeros((nob,))

    ## for pymp utilization

    t0 = time.time()
    bd_dims = [16,16,16]
    blk_dims = [32,32,32]
    #sampling_method = 'hist'
    freq=1
    whichBlock = 0
    numPieces = len(range(whichBlock,nob,freq))
    multiblock = vtk.vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(numPieces)
    pc_cnt = 0
    tot_points = 0

    print('Starting block processing.')

    if not os.path.exists("vtu_outputs/sampled_"+sampling_method+'_pymp/'):
        os.makedirs("vtu_outputs/sampled_"+sampling_method+'_pymp/')

    for bid in range(whichBlock,nob,freq):
        block_data = bm.get_blockData(data, bid)
        #print(bid,'.',end='')
        
        if sampling_method=='hist':
            fb_stencil = sm.global_hist_based_sampling(block_data)
        elif sampling_method=='hist_grad': 
            fb_stencil = sm.global_hist_grad_based_sampling(block_data,blk_dims)
        elif sampling_method=='hist_grad_rand':
            fb_stencil = sm.global_hist_grad_rand_based_sampling(block_data,blk_dims)
        elif sampling_method == "lcc":
            fb_stencil = sm.lcc_sampling(block_data, blk_dims, args.percentage)
        else:
            print('Unknown sampling method; Not implemented yet.')
        rand_stencil = sm.rand_sampling(block_data, rand_sr)
        
        comb_stencil = fb_stencil + rand_stencil
        comb_stencil = np.where(comb_stencil > 1, 1, comb_stencil)
        
        void_hist, ble, delta = sm.get_void_histogram(block_data, comb_stencil, vhist_nbins)
            
        #sampled_fid, sampled_data = sm.get_samples(block_data, block_fid, comb_stencil)
        sampled_lid, sampled_data = sm.get_samples(block_data, comb_stencil)
        
        #sampled_fid = block_fid[comb_stencil > 0.5]
        sampled_fid= np.zeros_like(sampled_lid,dtype=np.int)
        sampled_locs = np.where(comb_stencil > 0.5)[0]
        #print(np.shape(sampled_locs))

        list_sampled_lid.append(sampled_lid)
        list_sampled_data.append(sampled_data)

        array_void_hist[bid] = void_hist
        array_ble[bid] = ble
        array_delta[bid] = delta
        
        # write out a vtp file
        # now use this stencil array to store the locations
        outfiles = 'out_vals/'
        name='dm_density'
        #int_inds = np.where(stencil_new>0.5)
        #poly_data = vtk.vtkPolyData()
        Points = vtk.vtkPoints()
        val_arr = vtk.vtkDoubleArray()
        val_arr.SetNumberOfComponents(1)
        val_arr.SetName(name)

        
        bd_idx = np.unravel_index(bid,bd_dims)
        sampled_locs_xyz = np.unravel_index(sampled_locs,blk_dims)
        #print(np.multiply(bd_idx,blk_dims),np.shape(sampled_locs_xyz),sampled_locs_xyz)
        pt_locs_np = np.add(np.transpose(sampled_locs_xyz), np.multiply(bd_idx,blk_dims))
        #pt_locs_np = pt_locs_np.T
        pt_locs_np[:,[0,2]] = pt_locs_np[:,[2,0]]
        #print('here:',pt_locs_np)
        Points.SetData(VN.numpy_to_vtk(pt_locs_np))
            
        val_arr.SetArray(sampled_data,sampled_data.size,True)
        val_arr.array = sampled_data
        
        tot_points += sampled_data.size
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(Points)

        polydata.GetPointData().AddArray(val_arr)
        
        ## write the vtp file
        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName("vtu_outputs/sampled_"+sampling_method+'_pymp/'+"sampled_"+sampling_method+"_"+str(bid)+".vtp");
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.Write()
        
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[%-50s] %d%%" % ('='*int((bid-whichBlock)/freq*50/(numPieces-1)), 100/(numPieces-1)*((bid-whichBlock)/freq)))
        sys.stdout.flush()

        
    #print('\n')    

    filename = "vtu_outputs/sampled_"+sampling_method+"_pymp.vtm"

    file = open(filename, "w")
    top_string = '<?xml version="1.0"?> \n <VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor"> \n <vtkMultiBlockDataSet>'
    bottom_string = '\n </vtkMultiBlockDataSet> \n</VTKFile>'
    file.write(top_string)
    file_count = 0
    for bid in range(whichBlock,nob,freq):
        middle_string = '\n  <DataSet index="'+str(file_count)+'" file="sampled_'+sampling_method+'_pymp/sampled_'+sampling_method+'_'+str(bid)+'.vtp"/>'
        file.write(middle_string)
        file_count+=1
    file.write(bottom_string)
    file.close()

    ## store the samples and related information
    sampling_method_val = np.int64(enum_dict[sampling_method])
    #print('sampling_method_val:',sampling_method_val)
    out_folder = 'sampled_output'
    os.makedirs(out_folder, exist_ok=True)
    dump_with_pickle(bm_paramter_list, out_folder+'/'+"bm_paramter_list.pickle")
    dump_with_pickle(list_sampled_lid, out_folder+'/'+"list_sampled_lid.pickle")
    dump_with_pickle(list_sampled_data, out_folder+'/'+"list_sampled_data.pickle")
    array_void_hist.tofile(out_folder+'/'+"array_void_hist.raw")
    array_ble.tofile(out_folder+'/'+"array_ble.raw")
    array_delta.tofile(out_folder+'/'+"array_delta.raw")
    #np.asarray(sampling_rate).tofile(out_folder+"sampling_rate.raw")
    #np.asarray(sampling_method_val).tofile(out_folder+"sampling_method_val.raw")
    np.save(out_folder+'/'+"sampling_rate.npy",np.asarray(sampling_rate))
    np.save(out_folder+'/'+"sampling_method_val.npy",np.asarray(sampling_method_val))

    shutil.make_archive(out_folder+'_archive'+str(sampling_ratio), 'zip', out_folder)
    print('\nSize of the compressed data:',os.path.getsize(out_folder+'_archive'+str(sampling_ratio)+'.zip')/1000000,' MB')

    print('Total points stored:',np.sum(tot_points))
    print('Time taken %.2f secs' %(time.time()-t0))

parser = argparse.ArgumentParser()

# if len(sys.argv) != 5:
#     parser.error("incorrect number of arguments")

parser.add_argument('--input', action="store", required=True, help="input file name")
parser.add_argument('--output', action="store", required=False,help="output folder name")
parser.add_argument('--percentage', action="store", type=np.float32, default=0.01, required=False,
                    help="what fraction of samples to keep")
parser.add_argument('--nbins', action="store", required=False,help="how many bins to use")
parser.add_argument('--nthreads', action="store", required=False,help="how many threads to use")
parser.add_argument('--method', action="store", required=True,help="which sampling method to use. hist, grad, hist_grad, random, mixed")

args = parser.parse_args()

infile = getattr(args, 'input')
outPath = getattr(args, 'output')
sampling_ratio = args.percentage
nthreads = getattr(args, 'nthreads')
nbins = getattr(args, 'nbins')

method = getattr(args, 'method')

if method==None:
    method = 'hist'

enum_dict = {'rand':0,'hist':1,'hist_grad':2,'hist_grad_rand':3, 'lcc':4}

run(infile, sampling_ratio, 0.0, method, args)

## python nyx_test_ayan.py --input=nyx_data_dark_matter_density.vti --method='hist' --percentage=0.005