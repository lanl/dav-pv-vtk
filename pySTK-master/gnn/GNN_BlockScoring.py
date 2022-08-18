#### Import modules
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch.nn import Linear, Sequential, Dropout, Flatten, BCEWithLogitsLoss
from torch_geometric.loader import DataLoader
import torch.utils.data as data
import copy
import psutil
import time
import multiprocessing as mp

#### Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', action="store", required=True, help="input prediction data 4D array .npy file")
parser.add_argument('--label', action="store", required=True, help="input prediction block labels 1D array .npy file")

#### Function to create graph edge list ####
#### Takes blk_size denoted by 'n' as argument to create a 3D graph's (blk_size, blk_size, blk_size) edge list

def createEL(n):
    edgeList = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                #Assign unique numbers as an identifier to the nodes
                nodeNos = (x)*n*n + (y)*n + z
                
                #Construct the edgelist: Edges with every neighbour (26 total) with the exception of corner nodes
                #1 X, Y, Z+1
                if z < n-1:
                    edgeList.append([nodeNos, nodeNos+1])
                    edgeList.append([nodeNos+1, nodeNos])
                    
                #2 X, Y, Z-1
                if z > 0:
                    edgeList.append([nodeNos, nodeNos-1])
                    edgeList.append([nodeNos-1, nodeNos])
                    
                #3 X, Y+1, Z
                if y < n-1:
                    edgeList.append([nodeNos, nodeNos+n])
                    edgeList.append([nodeNos+n, nodeNos])
                    
                #4 X, Y+1, Z+1
                if  z < n-1 and y < n-1:
                    edgeList.append([nodeNos, nodeNos+n+1])
                    edgeList.append([nodeNos+n+1, nodeNos])
                    
                #5 X, Y+1, Z-1
                if z > 0 and y < n-1:
                    edgeList.append([nodeNos, nodeNos+n-1])
                    edgeList.append([nodeNos+n-1, nodeNos])
                    
                #6 X, Y-1, Z
                if y > 0:
                    edgeList.append([nodeNos, nodeNos-n])
                    edgeList.append([nodeNos-n, nodeNos])
                    
                #7 X, Y-1, Z+1
                if z < n-1 and y > 0:
                    edgeList.append([nodeNos, nodeNos-n+1])
                    edgeList.append([nodeNos-n+1, nodeNos])
                    
                #8 X, Y-1, Z-1
                if z > 0 and y > 0:
                    edgeList.append([nodeNos, nodeNos-n-1])
                    edgeList.append([nodeNos-n-1, nodeNos])
                    
                #9 X+1, Y, Z      
                if x < n-1:
                    edgeList.append([nodeNos, nodeNos+n*n])
                    edgeList.append([nodeNos+n*n, nodeNos])
                    
                #10 X+1, Y, Z+1
                if z < n-1 and x < n-1:
                    edgeList.append([nodeNos, nodeNos+n*n+1])
                    edgeList.append([nodeNos+n*n+1, nodeNos])
                    
                #11 X+1, Y, Z-1
                if z > 0 and x < n-1:
                    edgeList.append([nodeNos, nodeNos+n*n-1])
                    edgeList.append([nodeNos+n*n-1, nodeNos])
                    
                #12 X+1, Y+1, Z
                if y < n-1 and x < n-1:
                    edgeList.append([nodeNos, nodeNos+n*n+n])
                    edgeList.append([nodeNos+n*n+n, nodeNos])
                    
                #13 X+1, Y+1, Z+1
                if z < n-1 and y < n-1 and x < n-1:
                    edgeList.append([nodeNos, nodeNos+n*n+n+1])
                    edgeList.append([nodeNos+n*n+n+1, nodeNos])

                #14 X+1, Y+1, Z-1
                if z > 0 and x < n-1 and y < n-1:
                    edgeList.append([nodeNos, nodeNos+n*n+n-1])
                    edgeList.append([nodeNos+n*n+n-1, nodeNos])

                #15 X+1, Y-1, Z
                if y > 0 and x < n-1:
                    edgeList.append([nodeNos, nodeNos+n*n-n])
                    edgeList.append([nodeNos+n*n-n, nodeNos])

                 #16 X+1, Y-1, Z+1
                if z < n-1 and y > 0 and x < n-1:
                    edgeList.append([nodeNos, nodeNos+n*n-n+1])
                    edgeList.append([nodeNos+n*n-n+1, nodeNos])

                #17 X+1, Y-1, Z-1
                if z > 0 and y > 0 and x < n-1:
                    edgeList.append([nodeNos, nodeNos+n*n-n-1])
                    edgeList.append([nodeNos+n*n-n-1, nodeNos])
                    
                #18 X-1, Y, Z      
                if x > 0:
                    edgeList.append([nodeNos, nodeNos-n*n])
                    edgeList.append([nodeNos-n*n, nodeNos])
                    
                #19 X-1, Y, Z+1
                if z < n-1 and x > 0:
                    edgeList.append([nodeNos, nodeNos-n*n+1])
                    edgeList.append([nodeNos-n*n+1, nodeNos])
                    
                #20 X-1, Y, Z-1
                if z > 0 and x > 0:
                    edgeList.append([nodeNos, nodeNos-n*n-1])
                    edgeList.append([nodeNos-n*n-1, nodeNos])
                    
                #21 X-1, Y+1, Z
                if y < n-1 and x > 0:
                    edgeList.append([nodeNos, nodeNos-n*n+n])
                    edgeList.append([nodeNos-n*n+n, nodeNos])
                    
                #22 X-1, Y+1, Z+1
                if z < n-1 and y < n-1 and x > 0:
                    edgeList.append([nodeNos, nodeNos-n*n+n+1])
                    edgeList.append([nodeNos-n*n+n+1, nodeNos])

                #23 X-1, Y+1, Z-1
                if z > 0 and y < n-1 and x > 0:
                    edgeList.append([nodeNos, nodeNos-n*n+n-1])
                    edgeList.append([nodeNos-n*n+n-1, nodeNos])

                #24 X-1, Y-1, Z
                if y > 0 and x > 0:
                    edgeList.append([nodeNos, nodeNos-n*n-n])
                    edgeList.append([nodeNos-n*n-n, nodeNos])

                #25 X-1, Y-1, Z+1
                if z < n-1 and y > 0 and x > 0:
                    edgeList.append([nodeNos, nodeNos-n*n-n+1])
                    edgeList.append([nodeNos-n*n-n+1, nodeNos])

                #26 X-1, Y-1, Z-1
                if z > 0 and y > 0 and x > 0:
                    edgeList.append([nodeNos, nodeNos-n*n-n-1])
                    edgeList.append([nodeNos-n*n-n-1, nodeNos])
                    
    #Convert edge list to tensor
    edgeListTensor = torch.tensor(edgeList)

    return edgeListTensor

#### Function to construct 3D graphs to be fed to the GNN from the 4D array of the input data
#### Size of each 3D graph would be blk_size x blk_size x blk_size
#### 'blk_data' is a collection of node data within the block which are 'byron's density' values of the corresponding original Nyx data grid points
#### templateGraph would provide edge list and graph structure information as derived from the function 'createEL'
#### 'bid' is the block identifier

def constructGraphs(bid, blk_data, blk_label, templateGraph):
    #Center the blk_data
    blk_data = (blk_data - np.mean(blk_data))/np.std(blk_data)
    blk_data = blk_data.flatten()  
    #Convert the blk_data to tensor after reshaping it suitably for the GNN
    nodeData = torch.tensor(blk_data.reshape((blk_data.shape[0], 1))).float()
    #Create a 3D graph using the node data and edgelist structure
    g = Data(x = nodeData, edge_index=templateGraph.t().contiguous(), bid=bid, label=blk_label);
    
    return g

#### Function to create an 'input graph dataset' for GNN constituting multiple 3D graphs with node data/features

def GNNDataset(data, label, blk_size):
    #For storing all the graphs for the input graph dataset
    XGraphs = []
    #Create an edgelist template 
    gTemplate = createEL(blk_size)
    #Construct graphs out of all the blocks the input has been divided into
    for bid in range(data.shape[3]):
        #Copy from the template
        currentTemplate = copy.deepcopy(gTemplate)
        g = constructGraphs(bid, data[:,:,:,bid], label[bid], currentTemplate)  
        XGraphs.append(g)
     
    return XGraphs

#### Function to implement the GNN Model

class GCN(torch.nn.Module):
    def __init__(self, nin, nhidden, nout):
        super().__init__()
        #GCN(nin=>nhidden)
        self.conv1 = GCNConv(nin, nhidden)
        #GCN(nhidden=>nhidden)
        self.conv2 = GCNConv(nhidden, nhidden)  
        #Dense(nhidden=>nout)
        self.drop = Dropout(0.3)
        self.dense = Linear(nhidden, nout)

    def forward(self, data, batch):
        #Get node data and edge list
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #First convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x) 
        #Second convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        y, _ = torch.max(x, 1, True)
        
        #Max pool
        x = global_max_pool(x, batch)
        x = self.drop(x)
        
        #Dense layer
        x = self.dense(x)
        x = F.softmax(x, dim=-1)

        return x,y
    
    

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    #### Load prediction data and labels
    prediction_data = np.load(args.input)
    prediction_label = np.load(args.label)
    dim = np.shape(prediction_data)
    print("Loading data")
    print("Dimensions:", dim)

    predict_data = GNNDataset(prediction_data, prediction_label, 8)
    predict_loader = DataLoader(predict_data, batch_size=64)

    #### GNN Prediction for block scoring ####
    trainedModel = GCN(1, 64, 2)
    trainedModel.load_state_dict(torch.load('trainedGNN.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainedModel = trainedModel.to(device)
    print(device)

    stime = time.time()
    print("Starting GNN prediction for block scoring")
    #The GNN is supposed to provide importance scores to the blocks of input data based on Halo presence 
    #For a trained supervised classifier, the importance scores are the classification probablilities.

    block_score = []
    i = 0
    batch_size = 64
    trainedModel.eval()


    for graphs in predict_loader:
        graphs = graphs.to(device)
        with torch.no_grad():
            out, out_nodes = trainedModel.forward(graphs, batch_size)
            
        out = out.to('cpu').detach().numpy()
        out_nodes = out_nodes.to('cpu').detach().numpy()
        bid = graphs.bid.to('cpu').detach().numpy()
        label = graphs.label.to('cpu').detach().numpy()
    
        bid = bid.reshape(batch_size, 1)
        label = label.reshape(batch_size, 1)
        out_nodes = out_nodes.reshape((batch_size, 8*8*8))
    
        block_batch = np.concatenate((bid, label), axis=1)
        block_batch = np.concatenate((block_batch, out), axis=1)
        block_batch = np.concatenate((block_batch, out_nodes), axis=1)
    
        if i == 0:
            block_score = block_batch
        else:
            block_score = np.concatenate((block_score, block_batch), axis=0)
    
        i = i+1
            
    print("Time taken for GNN block scoring:",(time.time() - stime))

    #### Sort the importance scores and store them for sampling ####

    block_score = block_score[(block_score[:, 0]).argsort()]
    print(block_score)
    np.save('block_score.npy', block_score)

