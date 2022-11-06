import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import networkx as nx
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, PReLU, ELU
import loaddatas as lds
from torch_geometric.utils import softmax,degree
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
from Knowledge_Distillation.data_utils_NC import compute_persistence_image, compute_ricci_curvature
from Knowledge_Distillation.Teacher_model import Teacher_Model
import os
from tqdm import tqdm


class Net(torch.nn.Module):
    def __init__(self,data,name,num_features,num_classes, hop, g, dimension=5, skip_cat = False, skip_sum = False):
        super(Net, self).__init__()
        self.dimension = dimension
        if name in ['Physics', 'computers']:
            hidden_dim = 64
        else:
            hidden_dim = 256
        self.conv1 = curvGN(num_features, hidden_dim, dimension=dimension, skip_cat = skip_cat, skip_sum = skip_sum)
        if skip_cat:
            self.conv2 = curvGN(hidden_dim * 2, num_classes, dimension=dimension, skip_cat = False, skip_sum = skip_sum)
        else:
            self.conv2 = curvGN(hidden_dim, num_classes, dimension=dimension, skip_cat = False, skip_sum = skip_sum)
        self.skip_cat = skip_cat
        self.skip_sum = skip_sum
        self.leakyrelu = torch.nn.LeakyReLU(0.2, True)
        self.linear = torch.nn.Linear(dimension * dimension, 1, bias=True)
        self.linear_1 = torch.nn.Linear(dimension * dimension + 16, dimension * dimension, bias=True)
        self.hop = hop
        self.g = g
        self.name = name
        self.modelGIN = Teacher_Model(hidden_dim = 32, type = 'GAT', num_models=1, dropout = 0, new_node_feat = True, use_edge_attn = True).cuda()


        # evaluating influence of training samples / transferable of models
        self.modelGIN.load_state_dict(
            torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/" + name + "_0.2.pt"))
        for param in self.modelGIN.parameters():
            param.requires_grad = False
        save_GIN_PI = "/data1/curvGN_LP/data/data/KD/predicted/" + name + "_temp.pt"
        if not os.path.exists(save_GIN_PI):
            self.compute_NodeFeat(data, name)
            print("node feature compute finished")
            self.compute_PI(data, name)
            print("PI compute finished")
            torch.save(self.w_mul, save_GIN_PI)
        else:
            print("PI already exists")
            self.w_mul = torch.load(save_GIN_PI)
        '''
        # standard settings
        if name not in ["CS", "Physics"]:
            self.modelGIN.load_state_dict(torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/" + name + ".pt"))
        else:
            self.modelGIN.load_state_dict(
                torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/photo.pt"))
        for param in self.modelGIN.parameters():
            param.requires_grad = False
        save_GIN_PI = "/data1/curvGN_LP/data/data/KD/predicted/" + name + "_NC.pt"
        if not os.path.exists(save_GIN_PI):
            self.compute_NodeFeat(data, name)
            print("node feature compute finished")
            self.compute_PI(data, name)
            print("PI compute finished")
            torch.save(self.w_mul, save_GIN_PI)
        else:
            print("PI already exists")
            self.w_mul = torch.load(save_GIN_PI)
        '''

    def compute_NodeFeat(self, data, name):
        self.dict_feat = {}
        hop = 2 if name in ["Cora", "Citeseer", "PubMed"] else 1

        g = nx.Graph()
        g.add_nodes_from([i for i in range(data.num_nodes)])
        ricci_edge_index_ = np.array(remove_self_loops((data.edge_index.cpu()))[0])
        ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                            range(np.shape(ricci_edge_index_)[1])]
        g.add_edges_from(ricci_edge_index)

        ricci_curv = compute_ricci_curvature(data, name)

        num_nodes = data.num_nodes
        pbar_edge = tqdm(total=num_nodes)
        for u in range(num_nodes):
            self.dict_feat[u] = \
                compute_persistence_image(g, u, filt='ricci', hks_time=10, hop=hop, ricci_curv=ricci_curv, mode='filtration')
            pbar_edge.update(1)
        pbar_edge.close()


    def compute_PI(self,data, name):
        num_nodes = data.num_nodes
        self.modelGIN.eval()
        PI = torch.zeros(num_nodes, 25).cuda()
        pbar_node = tqdm(total=num_nodes)
        for u in range(num_nodes):
            pbar_node.update(1)
            if self.dict_feat[u][0] is None:
                continue
            filt_value, edge_index = torch.FloatTensor(self.dict_feat[u][0]).cuda().view(-1, 1), torch.LongTensor(self.dict_feat[u][1]).cuda()
            if filt_value.size()[0] > 1:
                with torch.no_grad():
                    PI[u] = self.modelGIN(filt_value, edge_index, None, p=2, kernel='wasserstein', pair_diagonal=True, compute_loss = False, grad_PI = False)[1]
                    if name in ['photo']:
                        PI[u] = F.normalize(PI[u], dim = 0)
            if u == 0 or u == 1:
                print(PI[u])
        pbar_node.close()

        pbar_edge = tqdm(total=data.edge_index.size()[1])
        self.w_mul = torch.zeros(data.edge_index.size()[1], 50).cuda()
        for i in range(data.edge_index.size()[1]):
            pbar_edge.update(1)
            u = int(data.edge_index[0][i])
            v = int(data.edge_index[1][i])
            if u == v:
                continue
            else:
                self.w_mul[i] = torch.cat((PI[u].view(1,-1), PI[v].view(1,-1)), dim=1).cuda()
        pbar_edge.close()

    def forward(self,data):
        if self.name in ['Cora']:
            dropout = 0.6
        elif self.name in ['Physics']:
            dropout = 0.8
        elif self.name in ['CS']:
            dropout = 0.2
        else:
            dropout = 0.4
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x,p=dropout,training=self.training)
        x = self.conv1(x, edge_index, self.w_mul)
        x = F.elu(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv2(x, edge_index, self.w_mul)
        return F.log_softmax(x, dim=1)


class  curvGN(MessagePassing):
    def __init__(self, in_channels, out_channels, dimension = 5, skip_cat = False, skip_sum = False):
        super(curvGN, self).__init__(aggr='add') # "Add" aggregation.
        self.lin=Linear(in_channels,out_channels)
        self.skip_cat = skip_cat
        self.skip_sum = skip_sum
        if skip_cat or skip_sum:
            self.lin1 = Linear(in_channels, out_channels)
        widths=[dimension * dimension * 2,out_channels]
        self.w_mlp_out=create_wmlp(widths,out_channels,1)
    def forward(self, x, edge_index, w_mul):
        if self.skip_cat or self.skip_sum:
            x1 = self.lin1(x)
        x = self.lin(x)
        out_weight=self.w_mlp_out(w_mul)
        out_weight=softmax(out_weight,edge_index[0])
        if self.skip_cat:
            return torch.cat((self.propagate(x=x,edge_index=edge_index,out_weight=out_weight), x1), dim = -1)
        elif self.skip_sum:
            return self.propagate(x=x,edge_index=edge_index,out_weight=out_weight) +  x1
        else:
            return self.propagate(x=x,edge_index=edge_index,out_weight=out_weight)
    def message(self,x_j,edge_index,out_weight):
        return out_weight*x_j
    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out

def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k],widths[k+1],bias=False))
        #mlp_modules.append(LeakyReLU(0.2,True))
        mlp_modules.append(PReLU(widths[k+1], 0.2))
    mlp_modules.append(Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return seq(*mlp_modules)

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(data,name,num_features,num_classes):
    hop = 2 if name in ["Cora", "Citeseer", "PubMed"] else 1
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))

    g = nx.Graph()
    g.add_nodes_from([i for i in range(data.num_nodes)])
    ricci_edge_index_ = np.array(remove_self_loops((data.edge_index.cpu()))[0])
    ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                        range(np.shape(ricci_edge_index_)[1])]
    g.add_edges_from(ricci_edge_index)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(data,name,num_features,num_classes, hop, g).to(device), data.to(device)
    return model, data

