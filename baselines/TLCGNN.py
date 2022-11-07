import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
from torch.nn import Softmax
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from loaddatas import get_edges_split, compute_persistence_image

class Net(torch.nn.Module):
    def __init__(self,data,num_features,num_classes,PI,dimension=5):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 100, cached=True)
        self.conv2 = GCNConv(100, 16, cached=True)
        self.PI = PI
        self.leakyrelu = torch.nn.LeakyReLU(0.2, True)
        self.linear = torch.nn.Linear(dimension * dimension, 1, bias=True)
        self.linear_1 = torch.nn.Linear(dimension * dimension + 16, dimension * dimension, bias=True)
        self.softmax = Softmax(dim=1)
    def encode(self,data):
        # can set p = 0.8 for Cora and Citeseer, the results can be higher
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x,p=0.5,training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x
    def decode(self,data,emb,type="train"):
        if type == 'train':
            edges_pos = data.total_edges[:data.train_pos]
            index = np.random.randint(0, data.train_neg, data.train_pos)
            edges_neg = data.total_edges[data.train_pos:data.train_pos + data.train_neg][index]
            total_edges = np.concatenate((edges_pos,edges_neg))
            edges_y = torch.cat((data.total_edges_y[:data.train_pos],data.total_edges_y[data.train_pos:data.train_pos + data.train_neg][index]))

            PI = np.concatenate(
                (self.PI[:data.train_pos], self.PI[data.train_pos:data.train_pos + data.train_neg][index]))
        elif type == 'val':
            total_edges =  data.total_edges[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            edges_y = data.total_edges_y[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            PI = self.PI[data.train_pos + data.train_neg:data.train_pos + data.train_neg + data.val_pos + data.val_neg]
        elif type == 'test':
            total_edges = data.total_edges[
                          data.train_pos + data.train_neg + data.val_pos + data.val_neg:]
            edges_y = data.total_edges_y[
                      data.train_pos + data.train_neg + data.val_pos + data.val_neg :]
            PI = self.PI[data.train_pos + data.train_neg + data.val_pos + data.val_neg:]
        #linear to gather edge features
        emb = emb.renorm_(2,0,1)


        #pair wise PI
        new_x = torch.Tensor(
            PI.reshape((len(total_edges), -1))).cuda()

        emb_in = emb[total_edges[:, 0]]
        emb_out = emb[total_edges[:, 1]]
        sqdist = (emb_in - emb_out).pow(2)#.sum(dim=-1)
        sqdist = self.leakyrelu(self.linear_1(torch.cat((sqdist, new_x), dim=1)))
        sqdist = torch.abs(self.linear(sqdist)).reshape(-1)
        sqdist = torch.clamp(sqdist, min=0, max=40)
        prob = 1. / (torch.exp((sqdist - 2.0) / 1.0) + 1.0)
        return prob, edges_y.float()


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(data,name,num_features,num_classes,data_cnt):
    # to generate data and models
    if name in ['PPI']:
        val_prop = 0.2
        test_prop = 0.2
    else:
        val_prop = 0.05
        test_prop = 0.1
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_edges_split(data, val_prop = val_prop, test_prop = test_prop)
    total_edges = np.concatenate((train_edges,train_edges_false,val_edges,val_edges_false,test_edges,test_edges_false))
    data.train_pos,data.train_neg = len(train_edges),len(train_edges_false)
    data.val_pos, data.val_neg = len(val_edges), len(val_edges_false)
    data.test_pos, data.test_neg = len(test_edges), len(test_edges_false)
    data.total_edges = total_edges
    data.total_edges_y = torch.cat((torch.ones(len(train_edges)), torch.zeros(len(train_edges_false)), torch.ones(len(val_edges)), torch.zeros(len(val_edges_false)),torch.ones(len(test_edges)), torch.zeros(len(test_edges_false)))).long()

    # delete val_pos and test_pos
    edge_list = np.array(data.edge_index).T.tolist()
    for edges in val_edges:
        edges = edges.tolist()
        if edges in edge_list:
            # if not in edge_list, mean it is a self loop
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    for edges in test_edges:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    data.edge_index = torch.Tensor(edge_list).long().transpose(0, 1)

    hop = 2 if name in ["PubMed"] else 1
    if name in ['PPI']:
        f1 = compute_persistence_image(data, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, name + "_" + str(data_cnt), hop = hop)
    else:
        f1 = compute_persistence_image(data, train_edges, train_edges_false, val_edges, val_edges_false, test_edges,
                                       test_edges_false, name, hop = hop)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.total_edges_y.to(device)
    model, data = Net(data,num_features,num_classes, PI = f1).to(device), data.to(device)
    return model, data

