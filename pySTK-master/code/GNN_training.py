#### Import modules
import argparse
import numpy as np
import pandas as pd
import math
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import copy
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch.nn import Linear, Sequential, Dropout, Flatten, BCEWithLogitsLoss
import torch.utils.data as data

#### Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--training_data', action="store", required=True, help="input training data 4D array .npy file")
parser.add_argument('--training_label', action="store", required=True, help="input block labels 1D array .npy file")
parser.add_argument('--prediction_data', action="store", required=False, help="input prediction data 4D array .npy file")
parser.add_argument('--prediction_label', action="store", required=False, help="input prediction block labels 1D array .npy file")

#### Function to create graph edge list
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
#### Graph 'blk_label' would be based on the presence of Halo centers in them ####
#### templateGraph would provide edge list and graph structure information as derived from the function 'createEL'
#### 'bid' is the block identifier

def constructGraphs(blk_size, blk_data, blk_label, templateGraph, bid):
    #Center the blk_data
    blk_data = (blk_data - np.mean(blk_data))/np.std(blk_data)
    blk_data = blk_data.flatten()  
    #Convert the blk_data to tensor and make it node data for the graph after reshaping it suitably for the GNN
    nodeData = torch.tensor(blk_data.reshape((blk_data.shape[0], 1))).float()
    #Similarly create GNN suitable graph labels (GNN outputs both '0' and '1' classification probabilities)
    if blk_label == 1:
        probLabel = np.array([0.0, 1.0]).reshape((1, 2))
    else:
        probLabel = np.array([1.0, 0.0]).reshape((1, 2))
    probLabel = torch.tensor(probLabel)
    #Create a 3D graph
    g = Data(x = nodeData, edge_index=templateGraph.t().contiguous(), y = probLabel);

    return g   

#### Function to create an 'input graph dataset' for GNN constituting multiple 3D graphs with node data/features

def GNNDataset(data, label, sGraphs, eGraphs):
    #size of each 3D graph is the block size of each 3D block the input data has been divided into
    blk_size = np.shape(data)[0] 
    #For storing graphs
    XGraphs = []
    #Create template edgelist
    gTemplate = createEL(blk_size)
    #Construct graphs out of all the blocks input has been divided into
    for bid in range(sGraphs,eGraphs):
        #Copy from the template
        currentTemplate = copy.deepcopy(gTemplate)
        g = constructGraphs(blk_size, data[:,:,:,bid], label[bid], currentTemplate, bid)  
        XGraphs.append(g)
        
    return XGraphs

#### Function to calculate the 'Accuracy' metric for the model

def accuracy(tp, tn, fp, fn):
    if (tp + tn + fp + fn) != 0:
        return (tp + tn)/(tp + tn + fp + fn)
    else:
        return 0

#### Function to calculate the 'Recall' metric for the model

def recall(tp, tn, fp, fn):
    if (tp + fn) != 0:
        return (tp)/(tp + fn)
    else:
        return 0

#### Function to calculate the 'Precision' metric for the model

def precision(tp, tn, fp, fn):
    if (tp + fp) != 0:
        return (tp)/(tp + fp)
    else:
        return 0

#### Function to calculate the 'F1 score' metric for the model

def f1_score(pn, rc):
    if (pn + rc) != 0:
        return (2 * pn * rc)/(pn + rc)
    else:
        return 0

#### Function to calculate the confusion matrix for the model

def confusionMatrix(y_true, y_predict):
    tp = y_true[ (y_true == 1) & (y_predict == 1)].shape[0]
    tn = y_true[ (y_true == 0) & (y_predict == 0)].shape[0]
    fp = y_true[ (y_true == 0) & (y_predict == 1)].shape[0]
    fn = y_true[ (y_true == 1) & (y_predict == 0)].shape[0]
    
    return tp, tn, fp, fn

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
        
        y = torch.mean(x, 1, True)
        
        #Max pool
        x = global_max_pool(x, batch)
        x = self.drop(x)
        
        #Dense layer
        x = self.dense(x)
        x = F.softmax(x, dim=-1)

        return x,y

#### Function for GNN training

def train(train_data):
    lr = 0.005          # learning rate
    batchsize = 64     # batch size (number of graphs in each batch)
    epochs = 500         # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    nin = 1             # Input 
    nout = 2            # Output
    nhidden = 64        # dimension of hidden features
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #LOAD TRAIN DATA 
    training_ratio = 0.8
    training_size = round(len(train_data)*training_ratio)
    testing_size = len(train_data) - training_size
    #train and test split
    train_data, test_data = data.random_split(train_data, [training_size, testing_size])
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=True)
  
    #DEFINE MODEL
    model = GCN(nin, nhidden, nout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
       
    #TRAIN
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        ntot_train = 0
        for graphs in train_loader:
            graphs = graphs.to(device)
            optimizer.zero_grad()
            out, out_y = model(graphs, batchsize)
            loss = F.binary_cross_entropy_with_logits(out, graphs.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            ntot_train += 1
              
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        #TEST
        model.eval()
        ntot_val = 0
        val_loss = 0
        for graphs in test_loader:
            graphs = graphs.to(device)
            out, out_y = model(graphs, batchsize)
            loss = F.binary_cross_entropy_with_logits(out, graphs.y)
            val_loss += loss.item()
            out = out.to('cpu').detach().numpy()
            labels = graphs.y.to('cpu').detach().numpy()
            ntot_val += 1
            pred = np.argmax(out, axis=1)
            gt = np.argmax(labels, axis=1)
            tp, tn, fp, fn = confusionMatrix(gt, pred)
            TP = TP + tp
            TN = TN + tn
            FP = FP + fp
            FN = FN + fn
        
        pn = precision(TP, TN, FP, FN)
        rc = recall(TP, TN, FP, FN)
        print("Epoch = "+str(epoch)+"|| Train Loss="+str(round(total_loss/ntot_train, 4))+"|| Val Loss="+str(round(val_loss/ntot_val, 4)) + "|| Accuracy = "+str(round(accuracy(TP, TN, FP, FN), 2))+"|| Recall Score = "+str(round(rc, 2)))

    return model

    
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    #### Create GNN training input graph dataset ####
    training_data = np.load(args.training_data)
    training_label = np.load(args.training_label)

    stime = time.time()
    train_data = GNNDataset(training_data, training_label, 0, training_label.shape[0])
    etime = time.time()
    print("Time taken to create an input graph dataset for GNN training:",etime-stime)

    #### Training the GNN ####

    print("Starting GNN training:")
    stime = time.time()
    trainedModel = train(train_data)
    etime = time.time()
    print("Time taken for GNN training:",etime-stime)

    #### Save the trained GNN ####

    torch.save(trainedModel.state_dict(), 'trainedGNN.pt')
    
    '''

    #### Create GNN prediction input graph dataset ####

    prediction_data = np.load(args.prediction_data)
    prediction_label = np.load(args.prediction_label)

    stime = time.time()
    predict_data = GNNDataset(prediction_data, prediction_label, 1, prediction_label.shape[0])
    predict_loader = DataLoader(predict_data, batch_size=64)
    etime = time.time()
    print("Time taken to create an input graph dataset for GNN prediction:",etime-stime)

    #### GNN prediction and performance metrics ####

    stime = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    #PREDICTION
    for graphs in predict_loader:
        graphs = graphs.to(device)
        out, out_y = trainedModel(graphs, 16)
        out = out.to('cpu').detach().numpy()
        labels = graphs.y.to('cpu').detach().numpy()
        pred = np.argmax(out, axis=1)
        gt = np.argmax(labels, axis=1)
        tp, tn, fp, fn = confusionMatrix(gt, pred)
        TP = TP + tp
        TN = TN + tn
        FP = FP + fp
        FN = FN + fn
            
    etime = time.time()
    print("Time to predict: "+str(etime-stime))
    print("Accuracy = "+str(accuracy(TP, TN, FP, FN)))  
    pn = precision(TP, TN, FP, FN)
    rc = recall(TP, TN, FP, FN)
    print("F1 Score = "+str(f1_score(pn, rc)))
    print("Precision = "+str(pn))
    print("Recall = "+str(rc))
    print("True Positives: "+str(TP))
    print("True Negatives: "+str(TN))
    print("False Positives: "+str(FP))
    print("False Negatives: "+str(FN))
    
    '''




