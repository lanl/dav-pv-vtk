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
import time
import sys
sys.path.append('.')
import FeatureSampler as FS
import importlib
importlib.reload(FS)
import pickle
import vtk
from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type
import argparse


def dump_with_pickle(pyObj, filename):
    fobj1 = open(filename, 'wb')
    pickle.dump(pyObj, fobj1)
    fobj1.close()
    
def load_with_pickle(filename):
    fobj1 = open(filename, 'rb')
    pyObj = pickle.load(fobj1)
    fobj1.close()
    return pyObj

def write_vti_output(full_recon_data, vti_folder, recon_method):
    print('Starting vti write.')
    t0 = time.time()
    #vti_folder = 'recons_output/'
    filename = vti_folder+'recons_nyx_data_'+str(sampling_rate)+'_'+sampling_method+'_'+str(recon_method)+'.vti'
    print("Writing output to: ", filename)
    vtkArray = numpy_to_vtk(num_array=full_recon_data.flatten('C'), deep=False,
                            array_type=get_vtk_array_type(full_recon_data.dtype))

    imageData = vtk.vtkImageData()
    #imageData.SetOrigin(origin)
    #imageData.SetSpacing(spacing)
    imageData.SetDimensions(full_recon_data.shape)
    imageData.GetPointData().SetScalars(vtkArray)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    writer.Write()
    print('Time taken %.2f secs' %(time.time()-t0))

def power_spectrum_calculation(full_recon_data):
    print('Starting gimlet power spectrum computation.')
    t0 = time.time()
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py

    #run gimlet code
    #cmd = 'rm NVB_C009_l10n512_S12345T692_z5_mod.hdf5'
    #os.system(cmd)
    hdf5_file_num = '42'
    cmd = 'cp NVB_C009_l10n512_S12345T692_z'+hdf5_file_num+'.hdf5 NVB_C009_l10n512_S12345T692_z'+hdf5_file_num+'_mod.hdf5'
    os.system(cmd)
    #cp 'NVB_C009_l10n512_S12345T692_z5.hdf5' 'NVB_C009_l10n512_S12345T692_z5_mod.hdf5' 
    print('Done copying hdf5 file.')

    infile = 'NVB_C009_l10n512_S12345T692_z'+hdf5_file_num+'_mod.hdf5'
    f = h5py.File(infile, 'r+')
    # dset = f['native_fields']
    # dset_bdensity = dset['dark_matter_density']
    # np_data = np.asarray(dset_bdensity)
    data = f['native_fields/dark_matter_density']
    print(data.shape)
    data[...] = full_recon_data     # assign new values to data
    f.close()

    ## verify the write
    f1 = h5py.File(infile, 'r')
    np.allclose(f1['native_fields/dark_matter_density'][()], full_recon_data)

    ## run this command from the command shelll
    # /Users/biswas/Work/gimlet2/apps/sim_stats/sim_stats.ex NVB_C009_l10n512_S12345T692_z5_mod.hdf5 power_spectrum/_5_recon_
    # sampling_rate = 0.005
    # sampling_method = 'hist'
    name_string = '_'+hdf5_file_num+'_recon_'+str(sampling_rate)+'_'+sampling_method+'_'
    cmd = '/Users/biswas/Work/gimlet2/apps/sim_stats/sim_stats.ex NVB_C009_l10n512_S12345T692_z'+hdf5_file_num+'_mod.hdf5 power_spectrum/'+name_string
    os.system(cmd)

    infile = 'power_spectrum/_'+hdf5_file_num+'_rhodm_ps3d.txt'
    df = pd.read_csv(infile,sep=' ',header=None)
    x_vals = np.asarray(df[2])
    y_vals = np.asarray(df[3])

    #infile = 'power_spectrum/_5_recon_rhodm_ps3d.txt'
    infile = 'power_spectrum/'+name_string+'rhodm_ps3d.txt'
    df_recon = pd.read_csv(infile,sep=' ',header=None)
    x_vals_recon = np.asarray(df_recon[2])
    y_vals_recon = np.asarray(df_recon[3])

    plt.figure()
    plt.loglog(x_vals,y_vals,c='k',label='Original')
    plt.loglog(x_vals_recon,y_vals_recon,c='b',label='Reconstructed')
    plt.loglog(x_vals_recon,y_vals_recon/y_vals,c='r',label='Ratio')
    plt.xlim([1,12])
    #plt.ylim([0,1.5])
    #plt.tight_layout()
    plt.legend()
    plt.title('Power Spectrum of Nyx')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.savefig('power_spectrum/power_spec_comparison_'+str(sampling_rate)+'_'+sampling_method+'.pdf')
    plt.show()

    print('Time taken %.2f secs' %(time.time()-t0))

def run_reconstruction(recontype):
    full_recon_data = np.full((ZDIM, YDIM, XDIM), np.nan)

    rm = FS.ReconstructionManager()
    print('Starting reconstruction.')
    t0 = time.time()
    for bid in range(nob):
        #print(bid,end='.')
        if recontype==1:
            recon_block = rm.reconstruction_1(XBLOCK*YBLOCK*ZBLOCK, list_sampled_data[bid], list_sampled_lid[bid], array_void_hist[bid], array_ble[bid], array_delta[bid])
        elif recontype==2:
            recon_block = rm.reconstruction_2(XBLOCK*YBLOCK*ZBLOCK, list_sampled_data[bid], list_sampled_lid[bid], array_void_hist[bid], array_ble[bid], array_delta[bid])
        elif recontype==3:
            recon_block = rm.reconstruction_3(XBLOCK*YBLOCK*ZBLOCK, list_sampled_data[bid], list_sampled_lid[bid], array_void_hist[bid], array_ble[bid], array_delta[bid])
        else:
            recon_block = rm.reconstruction_1(XBLOCK*YBLOCK*ZBLOCK, list_sampled_data[bid], list_sampled_lid[bid], array_void_hist[bid], array_ble[bid], array_delta[bid])
        
        recon_block = recon_block.reshape((ZBLOCK,YBLOCK,XBLOCK))

        fid, tx, ty, tz = bm.func_block_2_full(bid,0)
        full_recon_data[tz:tz+ZBLOCK, ty:ty+YBLOCK, tx:tx+XBLOCK] = recon_block
        #full_recon_data[tx:tx+XBLOCK, ty:ty+YBLOCK, tz:tz+ZBLOCK] = recon_block
    print('Time taken %.2f secs' %(time.time()-t0))
    return full_recon_data
        




##Main program
parser = argparse.ArgumentParser()

parser.add_argument('--sampledir', action="store", required=True,help="sampled data directory")
parser.add_argument('--outputdir', action="store", required=False,help="output folder name")
parser.add_argument('--recontype', action="store", required=False,help="specifiy the reconstruction method")
parser.add_argument('--vtiout', action="store", required=False,help="write out vti file?")
parser.add_argument('--powerspectrum', action="store", required=False,help="run power spectrum computation?")

args = parser.parse_args()

sampledir = getattr(args, 'sampledir')
outputdir = getattr(args, 'outputdir')
recontype = getattr(args, 'recontype')
vtiout = getattr(args, 'vtiout')
powerspectrum = getattr(args, 'powerspectrum')

if outputdir==None:
    outputdir = sampledir

if powerspectrum==None:
    powerspectrum = False
else:   
    powerspectrum = bool(getattr(args, 'powerspectrum'))

if vtiout==None:
    vtiout = True
else:   
    vtiout = bool(getattr(args, 'vtiout'))


if recontype==None:
    recontype = 1
else:
    recontype = int(getattr(args, 'recontype'))

## set some global variables (TODO: need to store it separately)
vhist_nbins = 128
ghist_nbins = 512



#read the sampling algorithm parameters
enum_dict = {'rand':0,'hist':1,'hist_grad':2,'hist_grad_rand':3,'grad':4}
sampling_method_val = np.load(sampledir+"sampling_method_val.npy")
sampling_rate = np.load(sampledir+"sampling_rate.npy")
sampling_method = list(enum_dict.keys())[list(enum_dict.values()).index(sampling_method_val)]
print('Sampling method and rate:',sampling_method, sampling_rate)


#load partition parameters
bm_paramter_list = load_with_pickle(sampledir+"bm_paramter_list.pickle")
XBLOCK, YBLOCK, ZBLOCK = bm_paramter_list[0:3]
XDIM, YDIM , ZDIM = bm_paramter_list[3:6]
bm = FS.BlockManager('regular', bm_paramter_list)
nob = bm.numberOfBlocks()
print('processing total',nob,'blocks')

#load sampled data
list_sampled_lid = load_with_pickle(sampledir+"list_sampled_lid.pickle")
list_sampled_data = load_with_pickle(sampledir+"list_sampled_data.pickle")
array_void_hist = np.fromfile(sampledir+"array_void_hist.raw",dtype=int)
array_void_hist = array_void_hist.reshape((nob, vhist_nbins))
array_ble = np.fromfile(sampledir+"array_ble.raw")
array_delta = np.fromfile(sampledir+"array_delta.raw")



# print("sampledir=", sampledir)
# print("outputdir=", outputdir)
# print("recontype=", recontype)
# print("nthreads=", nthreads)
# print("vtiout=", vtiout)
# print("powerspectrum=", powerspectrum)

## run the reconstruction code
full_recon_data = run_reconstruction(recontype)

## vti output needed?
if vtiout:
    write_vti_output(full_recon_data, outputdir, recontype)

## power spectrum via gimlet needed?
if powerspectrum:
    power_spectrum_calculation(full_recon_data)


