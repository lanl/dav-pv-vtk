#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import ctypes as c
import importlib
from pathlib import Path
import pickle
import shutil
import sys
import time

import multiprocessing as mp
import numpy as np
import pymp
import scipy.stats as ST
import os
import vtk


from vtk.util import numpy_support as VN
from common import enum_dict

import FeatureSampler as FS

sys.path.append('.')
importlib.reload(FS)

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def dump_with_pickle(pyObj, filename):
    fobj1 = open(filename, 'wb')
    pickle.dump(pyObj, fobj1)
    fobj1.close()


def load_with_pickle(filename):
    fobj1 = open(filename, 'rb')
    pyObj = pickle.load(fobj1)
    fobj1.close()
    return pyObj


def sample_rates(nob, args, bm, data, blk_dims):
    # block_sample_rates = np.full(nob, args.percentage, dtype=np.float64)
    block_sample_rates = pymp.shared.array(nob, dtype=np.float64)
    block_sample_rates[:] = args.percentage
    print(block_sample_rates)
    if args.adaptive:
        block_sample_rates = pymp.shared.array(nob, dtype=np.float64)
        with pymp.Parallel(nthreads) as p:
            for bid in p.range(nob):
                block_data = bm.get_blockData(data, bid)
                if args.function == "max":
                    block_sample_rates[bid] = np.max(block_data)
                elif args.function == "min":
                    block_sample_rates[bid] = np.min(block_data)
                elif args.function == "entropy":
                    block_sample_rates[bid] = ST.entropy(block_data)
        block_sample_rates = block_sample_rates / block_sample_rates.max()
        if args.function in ["min", "entropy"]:  # for weightings where lower is more interesting, invert weights
            block_sample_rates = 1.0 - block_sample_rates
        print("initial block sample rates: {0} {1} {2}".format(np.min(block_sample_rates), np.mean(block_sample_rates), np.max(block_sample_rates)))
        num_iterations = 0
        while True:
            num_iterations = num_iterations + 1
            factor = args.percentage / np.mean(block_sample_rates)
            block_sample_rates = block_sample_rates * factor
            if np.max(block_sample_rates) <= 1.0:
                break
            block_sample_rates[block_sample_rates > 1.0] = 1.0
        print("final block sample rates: {0} {1} {2}".format(np.min(block_sample_rates), np.mean(block_sample_rates), np.max(block_sample_rates)))
        print("num fully sampled blocks: ", np.sum(block_sample_rates >= 1.0))
        print("num iterations = ", num_iterations)
    return block_sample_rates
# End of sample_rates()


def sample(args, block_data, blk_dims, grad, bid, sm, bm):
    if args.percentage == 0.0:
        return np.zeros_like(block_data)
    if args.method == 'hist':
        fb_stencil = sm.global_hist_based_sampling(block_data)
    if args.method == 'hist_lcc':
        fb_stencil = GS.histogram_lcc_sampling(args, blk_dims, block_data)
    elif args.method == 'hist_grad':
        fb_stencil = sm.global_hist_grad_based_sampling(block_data, blk_dims)
    elif args.method == 'hist_grad_rand':
        fb_stencil = sm.global_hist_grad_rand_based_sampling(block_data, blk_dims)
    elif args.method == 'grad':
        block_grad_data = bm.get_blockData(grad, bid)
        fb_stencil = sm.global_grad_based_sampling(block_grad_data, blk_dims)
    elif args.method == 'grad_lcc':
        fb_stencil = GS.gradient_lcc_sampling(args, blk_dims, block_data)
    elif args.method == 'lcc':
        fb_stencil = GS.lcc_sampling(args, blk_dims, block_data)
    elif args.method == 'max':
        fb_stencil = GS.max_sampling(args, block_data)
    elif args.method == 'similarity':
        fb_stencil = GS.similarity_sampling(args, blk_dims, block_data)
    elif args.method == 'max_neighbor':
        fb_stencil = GS.max_neighbor_sampling(args, blk_dims, block_data)
    elif args.method == 'random_walk':
        fb_stencil = GS.random_walk_sampling(args, blk_dims, block_data)
    elif args.method == 'value_weighted_random_walk':
        fb_stencil = GS.value_weighted_random_walk_sampling(args, blk_dims, block_data)
    elif args.method == 'similarity_weighted_random_walk':
        fb_stencil = GS.similarity_weighted_random_walk_sampling(args, blk_dims, block_data)
    elif args.method == 'max_seeded_random_walk':
        fb_stencil = GS.max_seeded_random_walk_sampling(args, blk_dims, block_data)
    elif args.method == 'max_seeded_weighted_random_walk':
        fb_stencil = GS.max_seeded_weighted_random_walk_sampling(args, blk_dims, block_data)
    elif args.method == 'complex_max_seeded_random_walk':
        fb_stencil = GS.complex_max_seeded_random_walk_sampling(args, blk_dims, block_data)
    else:
        print('Unknown sampling method; Not implemented yet.')
    return fb_stencil
# End of sample()


def stencil(bid, bm, block_data, block_sample_rates, blk_dims, grad, sm, rand_sr):
    if args.adaptive:
        myargs = copy.deepcopy(args)
        myargs.percentage = min(1.0, block_sample_rates[bid])
        fb_stencil = sample(myargs, block_data, blk_dims, grad, bid, sm, bm)
    else:
        fb_stencil = sample(args, block_data, blk_dims, grad, bid, sm, bm)
    rand_stencil = sm.rand_sampling(block_data, rand_sr)

    comb_stencil = fb_stencil + rand_stencil
    comb_stencil = np.where(comb_stencil > 1, 1, comb_stencil)

    # pick at least one sample
    if np.sum(comb_stencil) == 0:
        comb_stencil[0] = 1

    # create boundary points
    dim = blk_dims
    A = [0, dim[0] - 1]
    B = [0, dim[1] - 1]
    C = [0, dim[2] - 1]
    j_vals = (((zval * dim[0] * dim[1]) + (yval * dim[0]) + xval) for xval in A for yval in B for zval in C)
    j_list = np.asarray(list(j_vals))
    comb_stencil[j_list] = 1
    return comb_stencil
# End of get_stencil()


def process_block(bid, bm, data, block_sample_rates, blk_dims, grad, sm, list_sampled_lid, list_sampled_data, bd_dims):
    global tot_points
    global array_delta
    global array_ble
    global array_void_hist
    block_data = bm.get_blockData(data, bid)
    comb_stencil = stencil(bid, bm, block_data, block_sample_rates, blk_dims, grad, sm, 0.0)
    void_hist, ble, delta = sm.get_void_histogram(block_data, comb_stencil, 16)
    sampled_lid, sampled_data = sm.get_samples(block_data, comb_stencil)
    

    sampled_locs = np.where(comb_stencil > 0.5)[0]
    list_sampled_lid[bid] = sampled_lid
    list_sampled_data[bid] = sampled_data

    #print(list_sampled_lid.shape)
    #print(list_sampled_data.shape)

    ncols = void_hist.size
    array_void_hist[bid * ncols : (bid + 1) * ncols] = void_hist
    array_ble[bid] = ble
    array_delta[bid] = delta

    # write out a vtp file
    # now use this stencil array to store the locations
    name = 'dm_density'
    Points = vtk.vtkPoints()
    if sampled_data.dtype == 'float64':
        val_arr = vtk.vtkDoubleArray()
    elif sampled_data.dtype == 'float32':
        val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    bd_idx = np.unravel_index(bid, bd_dims)
    sampled_locs_xyz = np.unravel_index(sampled_locs, blk_dims)
    pt_locs_np = np.add(np.transpose(sampled_locs_xyz), np.multiply(bd_idx, blk_dims))
    pt_locs_np[:, [0, 2]] = pt_locs_np[:, [2, 0]]
    #print(pt_locs_np)
    Points.SetData(VN.numpy_to_vtk(pt_locs_np))

    val_arr.SetArray(sampled_data, sampled_data.size, True)
    val_arr.array = sampled_data

    tot_points[bid] = sampled_data.size

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    # write the vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("vtu_outputs/sampled_" + args.method + '_pymp/' +
                        "sampled_" + args.method + "_" + str(bid) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()
# End of process_blocks()


############################### GNN Sampling ########################################

def gnn_sample_from_block(block_data, sample_rate_block, bid, nodeScore):

    block_rate = sample_rate_block[bid]*block_data.shape[0]
    comb_stencil = np.zeros_like(block_data)
    #print(block_data.shape[0] - int(block_rate))
    #indices = np.random.choice(comb_stencil.shape[0], int(block_rate), replace = False)
    #comb_stencil[indices] = 1
    
    if int(block_rate) != 0:
        #Get top n nodes and their indices
        indices = np.argpartition(nodeScore, nodeScore.shape[0] - int(block_rate))[nodeScore.shape[0] - int(block_rate):]
        comb_stencil[indices] = 1
    
    return comb_stencil

def gnn_process_block(bid, bm, data, blk_dims, sm, list_sampled_lid, list_sampled_data, bd_dims, block_score, sample_rate_block):
    #bid = int(block_score[i, 0])
    #bm = process_args[bid][1]
    #data = process_args[bid][2]
    #block_sample_rates = process_args[bid][3]
    #blk_dims = process_args[bid][4]
    #grad = process_args[bid][5]
    #sm = process_args[bid][6]
    #list_sampled_lid = process_args[bid][7]
    #list_sampled_data = process_args[bid][8]
    #bd_dims = process_args[bid][9]
    global tot_points
    global array_delta
    global array_ble
    global array_void_hist       
    
    block_data = bm.get_blockData(data, bid)  
    nodeScore = block_score[bid, 4:]

    comb_stencil = gnn_sample_from_block(block_data, sample_rate_block, bid, nodeScore)

    # pick at least one sample
    if np.sum(comb_stencil) == 0:
        comb_stencil[0] = 1
    
    void_hist, ble, delta = sm.get_void_histogram(block_data, comb_stencil, 16)
    sampled_lid, sampled_data = sm.get_samples(block_data, comb_stencil)

    sampled_locs = np.where(comb_stencil > 0.5)[0]
    list_sampled_lid[bid] = sampled_lid
    list_sampled_data[bid] = sampled_data

    ncols = void_hist.size
    array_void_hist[bid * ncols : (bid + 1) * ncols] = void_hist
    array_ble[bid] = ble
    array_delta[bid] = delta

    # write out a vtp file
    # now use this stencil array to store the locations
    name = 'dm_density'
    Points = vtk.vtkPoints()
    if sampled_data.dtype == 'float64':
        val_arr = vtk.vtkDoubleArray()
    elif sampled_data.dtype == 'float32':
        val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName(name)

    bd_idx = np.unravel_index(bid, bd_dims)
    sampled_locs_xyz = np.unravel_index(sampled_locs, blk_dims)
    pt_locs_np = np.add(np.transpose(sampled_locs_xyz), np.multiply(bd_idx, blk_dims))
    pt_locs_np[:, [0, 2]] = pt_locs_np[:, [2, 0]]
    #print(pt_locs_np)
    Points.SetData(VN.numpy_to_vtk(pt_locs_np))

    val_arr.SetArray(sampled_data, sampled_data.size, True)
    val_arr.array = sampled_data

    tot_points[bid] = sampled_data.size

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)

    polydata.GetPointData().AddArray(val_arr)

    # write the vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("vtu_outputs/sampled_" + args.method + '_pymp/' +
                        "sampled_" + args.method + "_" + str(bid) + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()  


def gnn_sampling(process_args, tot_points, array_void_hist, array_ble, array_delta):
    #process_args:
    #bid, bm, data, block_sample_rates, blk_dims, grad, sm, list_sampled_lid, list_sampled_data, bd_dims
    #0  , 1 ,  2  ,        3          ,     4   ,  5  ,  6,         7       ,         8        ,   9
    
    sample_rate = process_args[0][3]
    stime = time.time()
    #Load the block score
    block_score = np.load('block_score.npy')
    #Sort by block scores  
    block_score = block_score[(block_score[:, 3]).argsort()[::-1]]
    #Number of blocks
    nBlocks = block_score[:,3].shape[0]
    currentBlocksSum = nBlocks*sample_rate[0]
    block_score_sum = np.sum(block_score[:,3])
    sample_rate_block = np.zeros(nBlocks)
    
    for i in range(nBlocks):
        sample_rate_current = min(currentBlocksSum/block_score_sum*block_score[i,3], 0.9)     
        currentBlocksSum = currentBlocksSum - sample_rate_current
        block_score_sum =  np.sum(block_score[i+1:,3])
        bid = int(block_score[i, 0])

        sample_rate_block[bid] = sample_rate_current
    
    #Sorting the block scores by bid
    block_score = block_score[(block_score[:, 0]).argsort()] 
    print("Time taken for block scoring: "+str(time.time() - stime))
    
    
    sampling_args = []

    for i in range(block_score.shape[0]):
        bid = int(block_score[i, 0])
        bm = process_args[bid][1]
        data = process_args[bid][2]
        blk_dims = process_args[bid][4]
        sm = process_args[bid][6]
        list_sampled_lid = process_args[bid][7]
        list_sampled_data = process_args[bid][8]
        bd_dims = process_args[bid][9]  
   
        sampling_args.append((
            bid, bm, data, blk_dims, sm, list_sampled_lid, list_sampled_data, bd_dims, block_score, sample_rate_block
        ))

    def init_arrays(tot_points, array_void_hist, array_ble, array_delta):
        globals()['tot_points'] = tot_points
        globals()['array_void_hist'] = array_void_hist
        globals()['array_ble'] = array_ble
        globals()['array_delta'] = array_delta
    
    with mp.Pool(nthreads, initializer=init_arrays, initargs=(tot_points, array_void_hist, array_ble, array_delta)) as pool:
            pool.starmap(gnn_process_block, sampling_args)
            
############################### GNN Sampling Ends ########################################
        

def run(infile, fb_sr, rand_sr, nthreads, ghist_params_list, args):
    print('Reading data.')
    dm = FS.DataManager(infile, 0)

    data = dm.get_datafield()

    XDIM, YDIM, ZDIM = dm.get_dimension()
    d_spacing = dm.get_spacing()
    d_origin = dm.get_origin()
    d_extent = dm.get_extent()
    print(dm)

    vhist_nbins = ghist_params_list[0]
    ghist_nbins = ghist_params_list[1]
    sampling_rate = fb_sr + rand_sr

    # get the global acceptance histogram (provide sample rate) and min/max
    ghist = dm.get_acceptance_histogram(fb_sr, ghist_nbins)
    gmin = dm.getMin()
    gmax = dm.getMax()

    blk_dims = ghist_params_list[2:]

    bm_paramter_list = [blk_dims[0], blk_dims[1], blk_dims[2], XDIM, YDIM, ZDIM, d_origin, d_spacing, d_extent]

    bm = FS.BlockManager('regular', bm_paramter_list)

    nob = bm.numberOfBlocks()

    sm = FS.SampleManager()

    sm.set_global_properties(ghist, gmin, gmax)

    grad = None
    if args.method == 'grad':
        t0 = time.time()
        ghist_grad = dm.get_acceptance_histogram_grad(fb_sr, ghist_nbins)
        gmin_grad = dm.getGradMin()
        gmax_grad = dm.getGradMax()
        sm.set_global_properties_grad(ghist_grad, gmin_grad, gmax_grad)
        grad = dm.get_gradfield()
        print('Time taken for grad computation %.2f secs' % (time.time() - t0))

    list_sampled_lid = []
    list_sampled_data = []

    array_void_hist = np.zeros((nob, vhist_nbins))
    array_ble = np.zeros((nob,))
    array_delta = np.zeros((nob,))

    t0 = time.time()
    offset = 0.0000005
    bd_dims = [int(XDIM / blk_dims[0] + offset), int(YDIM / blk_dims[1] + offset), int(ZDIM / blk_dims[2] + offset)]
    freq = 1
    whichBlock = 0
    numPieces = len(range(whichBlock, nob, freq))
    multiblock = vtk.vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(numPieces)
    tot_points = 0

    print('Starting block processing. Total blocks=', nob, bd_dims)

    if not os.path.exists("vtu_outputs/sampled_"+args.method+'_pymp/'):
        os.makedirs("vtu_outputs/sampled_"+args.method+'_pymp/')

    tot_points = mp.Array(c.c_int32, nob, lock=False)
    array_void_hist = mp.Array(c.c_int64, nob * vhist_nbins, lock=False)
    array_ble = mp.Array(c.c_double, nob, lock=False)
    array_delta = mp.Array(c.c_double, nob, lock=False)

    list_sampled_lid = pymp.shared.list(nob * [None])
    list_sampled_data = pymp.shared.list(nob * [None])
    block_sample_rates = sample_rates(nob, args, bm, data, blk_dims)

    process_args = list()
    for bid in range(nob):
        process_args.append((
            bid, bm, data, block_sample_rates, blk_dims, grad, sm, list_sampled_lid, list_sampled_data, bd_dims
        ))

    def init_arrays(tot_points, array_void_hist, array_ble, array_delta):
        globals()['tot_points'] = tot_points
        globals()['array_void_hist'] = array_void_hist
        globals()['array_ble'] = array_ble
        globals()['array_delta'] = array_delta

    #GNN sampling + prediction serially
    if args.method == 'gnn':
        gnn_sampling(process_args, tot_points, array_void_hist, array_ble, array_delta)
    else:
        with mp.Pool(nthreads, initializer=init_arrays, initargs=(tot_points, array_void_hist, array_ble, array_delta)) as pool:
            pool.starmap(process_block, process_args)

    tot_points = np.asarray(tot_points[:], dtype=np.int32)
    array_void_hist = np.asarray(array_void_hist[:], dtype=np.int64).reshape((nob, vhist_nbins))
    array_ble = np.asarray(array_ble[:], dtype=np.double)
    array_delta = np.asarray(array_delta[:], dtype=np.double)

    filename = "vtu_outputs/sampled_" + args.method + "_pymp.vtm"

    file = open(filename, "w")
    top_string = '<?xml version="1.0"?> \n <VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor"> \n <vtkMultiBlockDataSet>'
    bottom_string = '\n </vtkMultiBlockDataSet> \n</VTKFile>'
    file.write(top_string)
    file_count = 0
    for bid in range(whichBlock, nob, freq):
        middle_string = '\n  <DataSet index="' + str(file_count) + '" file="sampled_' + \
            args.method + '_pymp/sampled_' + args.method + '_' + str(bid) + '.vtp"/>'
        file.write(middle_string)
        file_count += 1
    file.write(bottom_string)
    file.close()

    # store the samples and related information
    sampling_method_val = np.int(enum_dict[args.method])

    print("Clearing output path")
    os.system("rm -rf {0}".format(args.outpath))  # remove args.outpath first so the compression ratio is correct
    os.makedirs(args.outpath)
    print("writing sample info to: ", args.outpath)

    dump_with_pickle(ghist_params_list[0:2], args.outpath + '/' + "ghist_params_list.pickle")
    dump_with_pickle(bm_paramter_list, args.outpath + '/' + "bm_paramter_list.pickle")
    dump_with_pickle(list(list_sampled_lid), args.outpath + '/' + "list_sampled_lid.pickle")
    dump_with_pickle(list(list_sampled_data), args.outpath + '/' + "list_sampled_data.pickle")
    np.asarray(array_void_hist).tofile(args.outpath + '/' + "array_void_hist.raw")
    np.asarray(array_ble).tofile(args.outpath + '/' + "array_ble.raw")
    np.asarray(array_delta).tofile(args.outpath + '/' + "array_delta.raw")
    np.save(args.outpath + '/' + "sampling_rate.npy", np.asarray(sampling_rate))
    np.save(args.outpath + '/' + "sampling_method_val.npy", np.asarray(sampling_method_val))

    zip_folder = args.outpath + '_archive' + str(sampling_ratio) + '_' + args.method
    shutil.make_archive(zip_folder, 'zip', args.outpath)
    print('\nSize of the compressed data:', os.path.getsize(zip_folder + '.zip') / 1000000, ' MB')
    print('Size of the original data:', os.path.getsize(infile) / 1000000, ' MB')  # infile
    print('Effective Compression Ratio:', os.path.getsize(zip_folder + '.zip')/os.path.getsize(infile))

    print('Total points stored:', np.sum(tot_points))
    print('Time taken %.2f secs' % (time.time() - t0))
# End of run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', action="store", required=True, help="input file name")
    parser.add_argument('--outpath', action="store", required=False, help="output folder name")
    parser.add_argument('--percentage', action="store", type=np.float32, default=0.01, required=False,
                        help="what fraction of samples to keep")
    parser.add_argument('--nbins', action="store", type=int, required=False, default=32, help="how many bins to use")
    parser.add_argument('--blk_dims', action="store", required=False,
                        help="block dimensions. Supporting cubing blocks for now. blk_dim --> [blk_dim,blk_dim,blk_dim] ")
    parser.add_argument('--nthreads', action="store", type=int, required=False, default=8, help="how many threads to use")
    parser.add_argument('--method', action="store", required=True,
                        help="which sampling method to use. hist, grad, hist_grad, random, mixed")
    parser.add_argument("--contrast_points", action="store", required=False, type=float, default=0.00,
                        help="The amount of low value points to include for contrast purposes. Default is 0.00 or 0 percent of the total sampled points.")
    parser.add_argument("--sparse", action="store", required=False, default=0.25, type=float,
                        help="For grpah-based methods; if set to a value > 0, will create a sparse graph by removing edges to points with value less than 'sparse' percentile.")
    parser.add_argument("--adaptive", action="store_true", required=False, default=False,
                        help="""If true, the number of samples taken from every block will be proportional to some
                                function of the values in the block""")
    parser.add_argument("--function", action="store", type=str, default="entropy", 
                        help="""The function to use to assign weights for adaptive sampling. [max, min, entropy].
                                For min and entropy, more samples are taken from blocks with lower values""")

    args = parser.parse_args()

    infile = getattr(args, 'input')
    sampling_ratio = args.percentage
    nthreads = getattr(args, 'nthreads')
    nbins = getattr(args, 'nbins')
    nblk_dims = getattr(args, 'blk_dims')

    method = getattr(args, 'method')

    sampling_ratio = args.percentage

    if method == None:
        method = 'hist'

    nthreads = args.nthreads

    if nblk_dims == None:
        nblk_dims = 10
    else:
        nblk_dims = int(getattr(args, 'blk_dims'))

    if args.outpath == None:
        args.outpath = str(Path("sampled_output_{0}_{1}_{2}_{3}".format(
            method, args.adaptive, args.function, str(args.percentage))))
        print("outPath == ", args.outpath)

    if args.method == 'similarity':
        print("WARNING: similarity method has a bug with inf values and may not work")

    ghist_params_list = [16, 16, nblk_dims, nblk_dims, nblk_dims]
    print(args)
    run(infile, sampling_ratio, 0.0, nthreads, ghist_params_list, args)