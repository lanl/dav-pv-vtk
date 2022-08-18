#### Import modules
import argparse
import os
import h5py
import numpy as np
import pandas as pd

#### Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', action="store", required=True, help="hdf5 format Nyx dataset file path")
parser.add_argument('--Reeberoutfile', action="store", required=True, help="Reeber halo finder output for the input Nyx dataset: will be used to generate labels")
parser.add_argument('--blk_size', action="store", required=True, type=int, help="block size to divide the input into")
parser.add_argument('--use', action="store", required=True, help="create a train or prediction set for GNN?")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    #### Save the input filename to later track the generated training and prediction dataset as per timestep
    input_filename = os.path.basename(args.input)
    f = h5py.File(args.input)
    
    
    #### Read the h5 format Nyx input file and extract important fields (byron density for Halos) ####
    print("Reading the hdf5 file")
    grid_3d = f["native_fields"]["baryon_density"][:]
    print("Extracted the 3d grid")
    

    #### Convert the grid into a 3D numpy array ####
    np_grid_3d = np.asarray(grid_3d)
    print("Converted the 3D grid to a 3D numpy array")
    dim = np.shape(np_grid_3d)
    print("Dimensions of 3D numpy array:", dim)

    #### Convert to 4D numpy array (blk_size, blk_size, blk_size, No_of_blocks)
    #### to be converted to 'No_of_blocks' 3D graphs later each of size (blk_size, blk_size, blk_size) as inputs to the GNN
    blk_size = int(args.blk_size)
    #length of each axis after block processing 
    n_blocks_dim = dim[0]//blk_size
    #Total blocks the input is divided into
    total_blocks = dim[0] * dim[1] * dim[2] // (blk_size * blk_size * blk_size)
    #Creating the 4D numpy array
    np_grid_4d = np.zeros((blk_size, blk_size, blk_size, total_blocks))
    #Determining the range of coordinates within each block and assigning block IDs (bid) to blocks
    for z in range(n_blocks_dim):
        for y in range(n_blocks_dim):
            for x in range(n_blocks_dim):
                X = int(x*blk_size)
                Y = int(y*blk_size)
                Z = int(z*blk_size)
                bid = int(z*n_blocks_dim**2+ y*n_blocks_dim + x)
                np_grid_4d[:, :, :, bid]  =  np_grid_3d[X:X+blk_size, Y:Y+blk_size, Z:Z+blk_size]
                
    print("Converted into a 4D numpy array (blk_size, blk_size, blk_size, No_of_blocks) to be used as 'No_of_blocks' 3D graphs later each of size (blk_size, blk_size, blk_size) as inputs to the GNN")
    dim_4D = np.shape(np_grid_4d)
    print("Dimensions of 4D numpy array:", dim_4D)
    #Checking if the bids are correct
    #print(np.sum(np_grid_3d[472:480,464:472,400:408]-np_grid_4d[:,:,:,208571]))
    

    #### Label creation via Reeber output file (1D array in the shape of [No_of_blocks] to assign block label to each block)
    print("Starting block label assignment (to be used as graph labels later for supervised GNN after the blocks have been converted to graphs)")
    label = np.zeros(total_blocks)
    #Assign a header to appropriately read the Reeber output file.
    header_list = ["ID", "x", "y", "z", "No_0levelCells", "No_Vertices", "HaloMass", "IntegralAllFields"]
    df = pd.read_csv(args.Reeberoutfile, names=header_list, delim_whitespace=True)
    print("Reading the Reeber output file for block label generation")

    #Creating 1,0 labels for the blocks (based on whether they contain halo centers or not as identified by Reeber)
    for index, row in df.iterrows():
    #For all coordinates identified as Halo centers by Reeber, find the corresponding input block and mark its label 1
        X = int(row["x"])
        Y = int(row["y"])
        Z = int(row["z"])

        x = X//blk_size
        y = Y//blk_size
        z = Z//blk_size

        bid = int(z*n_blocks_dim**2+ y*n_blocks_dim + x)
        label[bid] = 1
    print("Finished block label generation")
    #Print how many blocks are labelled '1'
    print("Number of blocks labelled as '1' (have Halo centers in them):",np.count_nonzero(label == 1))

    #### Creating a class-balanced training data and label if training data is needed
    if args.use == "train":
        print("Starting class-balanced training data and label generation")
        #Total blocks and associated block labels from class '1'
        training_data_1 = np_grid_4d[:,:,:,label==1]
        training_label_1 = label[label==1]
        #print(np.shape(training_data_1))
        #print(np.shape(training_label_1))

        #Total blocks and associated block labels from class '0'
        training_data_0 = np_grid_4d[:,:,:,label==0]
        training_label_0 = label[label==0]
        #print(np.shape(training_data_0))
        #print(np.shape(training_label_0))

        #Taking random samples from class '0' blocks and associated block labels making it equal to total class '1' instances
        idx = np.random.randint(len(training_label_0), size=len(training_label_1))
        training_data_0 = training_data_0[:,:,:,idx]
        training_label_0 = training_label_0[idx]
        #print(np.shape(training_data_0))
        #print(np.shape(training_label_0))

        #Merging equal instances of both the classes together to make the final training dataset
        training_data = np.concatenate((training_data_1, training_data_0), axis = 3)
        training_label = np.concatenate((training_label_1, training_label_0))
        #print(np.shape(training_data))
        #print(np.shape(training_label))

        #Convert to a 4D numpy array input and 1D label to .npy files for graph conversation and subsequent GNN training
        training_data_filename = "training_data_" + str(args.blk_size) + "cub_" + str(input_filename[input_filename.find('z')+1 : input_filename.find('.')]) + ".npy"
        training_label_filename = "training_label_" + str(args.blk_size) + "cub_" + str(input_filename[input_filename.find('z')+1 : input_filename.find('.')]) + ".npy"
        print("Finished generating class-balanced training data and label")
        np.save(training_data_filename, training_data)
        np.save(training_label_filename, training_label)
        print("Saving training data to:",training_data_filename)
        print("Saving training label to:",training_label_filename)

    #### Creating a full cube prediction data and label for GNN
    if args.use == "predict":
        print("Starting prediction data and label generation")
        prediction_data_filename = "prediction_data_" + str(args.blk_size) + "cub_"+ str(input_filename[input_filename.find('z')+1 : input_filename.find('.')]) + ".npy"
        prediction_label_filename = "prediction_label_" + str(args.blk_size) + "cub_"+ str(input_filename[input_filename.find('z')+1 : input_filename.find('.')]) + ".npy"
        print("Finished generating prediction data and label")
        np.save(prediction_data_filename, np_grid_4d)
        np.save(prediction_label_filename, label)
        print("Saving prediction data to:",prediction_data_filename)
        print("Saving prediction label to:",prediction_label_filename)





