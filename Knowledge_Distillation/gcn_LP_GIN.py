import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import networkx as nx
from torch_geometric.utils import remove_self_loops, add_self_loops
import learnable_filter.loaddatas_LP as lds
from loaddatas_LP_arxiv import get_edges_split
from Knowledge_Distillation.data_utils_LP import compute_persistence_image, compute_ricci_curvature
from Knowledge_Distillation.Teacher_model import Teacher_Model
import os
from tqdm import tqdm


class Net(torch.nn.Module):
    def __init__(self,data, name, num_features,num_classes, total_edges, hop, g, dimension=5, data_cnt = 0):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 100, cached=True)
        self.conv2 = GCNConv(100, 16, cached=True)
        self.leakyrelu = torch.nn.LeakyReLU(0.2, True)
        self.linear = torch.nn.Linear(dimension * dimension, 1, bias=True)
        self.linear_1 = torch.nn.Linear(dimension * dimension + 16, dimension * dimension, bias=True)
        self.hop = hop
        self.total_edges = total_edges
        self.g = g
        self.name = name
        self.modelGIN =  Teacher_Model(hidden_dim = 32, type = 'GAT', num_models=1, dropout = 0, new_node_feat = True, use_edge_attn = True).cuda()
        self.modelGIN.load_state_dict(
            torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/" + name + ".pt"))
        for param in self.modelGIN.parameters():
            param.requires_grad = False
        save_GIN_PI = "/data1/curvGN_LP/data/data/KD/predicted/" + name + "_LP.pt"
        if not os.path.exists(save_GIN_PI):
            self.compute_PI(data, name)
            print("PI compute finished")
            torch.save(self.PI, save_GIN_PI)
        else:
            print("PI already exists")
            self.PI = torch.load(save_GIN_PI)


    def compute_PI(self, data, name):
        #self.modelGIN = self.modelGIN.cpu()
        self.modelGIN.eval()
        self.PI = torch.zeros(len(self.total_edges), 25)
        hop = 2 if name in ["Cora", "Citeseer", "PubMed"] else 1
        ricci_curv = compute_ricci_curvature(data, name)
        pbar_edge = tqdm(total=len(self.total_edges))
        for i in range(len(self.total_edges)):

            u, v = self.total_edges[i]
            pbar_edge.update(1)
            filt_value, edge_index = compute_persistence_image(self.g, u, v, filt = 'ricci', hks_time = 10, hop = hop, ricci_curv = ricci_curv, mode = 'filtration')
            if filt_value is None:
                continue
            filt_value, edge_index = torch.FloatTensor(filt_value).cuda().view(-1, 1), torch.LongTensor(
                edge_index).cuda()
            if filt_value.size()[0] > 1:
                with torch.no_grad():
                    self.PI[i] = self.modelGIN(filt_value, edge_index, None, p=2, kernel='wasserstein', pair_diagonal=True, compute_loss = False, grad_PI = False)[1]
            if i == 0 or i == 1:
                print(self.PI[i])
        pbar_edge.close()

    def encode(self,data):
        if self.name in ['Cora', 'Citeseer']:
            dropout = 0.8
        else:
            dropout = 0.5
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x,p=dropout,training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x

    def decode(self,data,emb,type="train"):
        if type == 'train':
            edges_pos = data.total_edges[:data.train_pos]
            index = np.random.randint(0, data.train_neg, data.train_pos)
            edges_neg = data.total_edges[data.train_pos:data.train_pos + data.train_neg][index]
            total_edges = np.concatenate((edges_pos,edges_neg))
            edges_y = torch.cat((data.total_edges_y[:data.train_pos],data.total_edges_y[data.train_pos:data.train_pos + data.train_neg][index]))
            #PI1 = self.PI[:data.train_pos] + [self.PI[data.train_pos:data.train_pos + data.train_neg][index_i] for index_i in index]
            PI1 = torch.cat((self.PI[:data.train_pos], self.PI[data.train_pos:data.train_pos + data.train_neg][index]))
        elif type == 'val':
            total_edges =  data.total_edges[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            edges_y = data.total_edges_y[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            PI1 = self.PI[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
        elif type == 'test':
            total_edges = data.total_edges[
                          data.train_pos + data.train_neg + data.val_pos + data.val_neg:]
            edges_y = data.total_edges_y[
                      data.train_pos + data.train_neg + data.val_pos + data.val_neg :]
            PI1 = self.PI[
                          data.train_pos + data.train_neg + data.val_pos + data.val_neg:]

        #linear to gather edge features
        #self.PI1 = torch.nn.Parameter(self.PI1, requires_grad = True).cuda()
        emb = emb.renorm_(2,0,1)

        #pair wise PI

        self.new_x = PI1.cuda()

        emb_in = emb[total_edges[:, 0]]
        emb_out = emb[total_edges[:, 1]]
        self.sqdist = (emb_in - emb_out).pow(2)#.sum(dim=-1)
        self.sqdist = self.leakyrelu(self.linear_1(torch.cat((self.sqdist, self.new_x), dim=1)))
        # sqdist = self.softmax(self.linear(sqdist)).reshape(-1)
        self.sqdist = torch.abs(self.linear(self.sqdist)).reshape(-1)
        self.sqdist = torch.clamp(self.sqdist, min=0, max=40)
        self.sqdist.retain_grad()
        #sqdist = (emb_in - emb_out).pow(2).sum(dim=-1)
        self.prob = 1. / (torch.exp((self.sqdist - 2.0) / 1.0) + 1.0)
        self.prob.retain_grad()
        return self.prob, edges_y.float()


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(data,name,num_features,num_classes, data_cnt = 0):
    hop = 2 if name in ["Cora", "Citeseer", "PubMed"] else 1


    if name in ['Rand_nnodes_github1000', 'PPI']:
        val_prop = 0.2
        test_prop = 0.2
    else:
        val_prop = 0.05
        test_prop = 0.1
    # to load saved edges, same setting as hgcn

    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_edges_split(data,
                                                                                                               val_prop=val_prop,
                                                                                                           test_prop=test_prop)
    '''
    if name == "photo":
        name = "Photo"
    elif name == 'computers':
        name = "Computers"
    
    if name != "PPI":
        train_edges = np.load('./data/LP/train_set/' + name + '/' + name + '_train_edges.npy')
        train_edges_false = np.load('./data/LP/train_set/' + name + '/' + name + '_train_edges_false.npy')
        val_edges = np.load('./data/LP/train_set/' + name + '/' + name + '_val_edges.npy')
        val_edges_false = np.load('./data/LP/train_set/' + name + '/' + name + '_val_edges_false.npy')
        test_edges = np.load('./data/LP/train_set/' + name + '/' + name + '_test_edges.npy')
        test_edges_false = np.load('./data/LP/train_set/' + name + '/' + name + '_test_edges_false.npy')
    else:
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_edges_split(data,
                                                                                                                   val_prop=val_prop,
                                                                                                                   test_prop=test_prop)
    '''
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


    g = nx.Graph()
    g.add_nodes_from([i for i in range(data.num_nodes)])
    ricci_edge_index_ = np.array(remove_self_loops((data.edge_index.cpu()))[0])
    ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                        range(np.shape(ricci_edge_index_)[1])]
    g.add_edges_from(ricci_edge_index)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.total_edges_y.to(device)
    '''
    if name == 'photo':
        name = "Photo"
    elif name == 'computers':
        name = 'Computers'
    '''
    model, data = Net(data, name, num_features,num_classes, total_edges, hop, g, data_cnt = data_cnt).to(device), data.to(device)
    return model, data

if __name__ == "__main__":
    d_name = 'Cora'
    d_loader = 'Planetoid'
    dataset = lds.loaddatas(d_loader, d_name)
    data = dataset[0]
    call(data,dataset.name,data.x.size(1),dataset.num_classes)
