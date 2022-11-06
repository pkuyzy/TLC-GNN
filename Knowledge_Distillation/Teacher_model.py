import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool, SAGEConv
#from Knowledge_Distillation.gat_conv import GATConv
#from torch_geometric.nn import GATConv
from scipy.spatial.distance import cityblock
import Knowledge_Distillation.wasserstein as gdw
from Knowledge_Distillation.PD_conv import PDConv
#from kmeans_pytorch_my import kmeans
from Knowledge_Distillation.visualize_PD import draw_PD
from Knowledge_Distillation.pimg import PersistenceImager
from sg2dgm.PersistenceImager import PersistenceImager as PersistenceImager_nograd
import time


class Teacher_Model(torch.nn.Module):
    def __init__(self, hidden_dim = 32, out_dim = 25, num_models = 3, dropout = 0.2, type = 'GIN', max_loop_len = 10, new_node_feat = True, use_edge_attn = True):
        super(Teacher_Model, self).__init__()

        self.DIM0_Model = Base_Model(1, hidden_dim, dropout, type, out_dim = hidden_dim, new_node_feat = new_node_feat, use_edge_attn = use_edge_attn)
        #self.DIM0_Model = Base_Model(1, hidden_dim, dropout, type, out_dim=2)

        # original wrong, only contain max / min value
        #self.DIM1_Model = Base_Model(2, hidden_dim, dropout, type, out_dim = 2)

        # now contain all filter values in the loop
        #self.DIM1_Model = Base_Model(2 * max_loop_len, hidden_dim, dropout, type, out_dim = 2, use_rnn = use_rnn)


        self.lin5 = Linear(2 * hidden_dim, hidden_dim)
        #self.lin5 = Linear(2 * hidden_dim, 2)
        self.lin6 = Linear(hidden_dim, 2)
        self.num_models = num_models
        self.lin1 = Linear(2, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.lin3 = Linear(hidden_dim, hidden_dim)
        self.lin4 = Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.pers_imager = PersistenceImager(resolution=5)
        self.pers_imager_nograd = PersistenceImager_nograd(resolution=5)

    def forward(self, x0, edge_index0, PD, kernel = 'sliced', M = 50, p = 1, pair_diagonal = False, draw_fig = False, fig_name = '', compute_loss = True, grad_PI = True):
        # this process rely on the ground truth PD, therefore only used for training

        t1 = time.time()
        #print(x0.device)
        #print(edge_index0.device)
        #print(next(self.DIM0_Model.parameters()).is_cuda)#False
        x = self.DIM0_Model(x0, edge_index0)
        x_in = x[edge_index0[0, : -len(x0)]]
        x_out = x[edge_index0[1, : -len(x0)]]
        x = self.lin5(torch.cat((x_in, x_out), dim = -1))
        x = F.prelu(x, weight = torch.tensor(0.1).cuda())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin6(x)
        t2 = time.time()

        if compute_loss:
            if not pair_diagonal:
                loss0, _, loss_xy0, loss_xd0, loss_yd0 = self.compute_PD_loss(x, PD, kernel = kernel, M = M, p = p, num_models = 1)
            else:
                loss0, _, loss_xy0, loss_xd0, loss_yd0 = self.compute_PD_loss(x, PD, kernel=kernel, M=M, p=p, num_models=1, type = 'inference')
        else:
            loss0 = None; loss_xy0 = None; loss_xd0 = None; loss_yd0 = None


        if draw_fig:
            if not pair_diagonal:
                draw_PD(PD1=x.detach().cpu().numpy(), save_name='./train_' + fig_name + '.png', PD2=PD.cpu().numpy())
            else:
                draw_PD(PD1=x.detach().cpu().numpy(), save_name='./test_' + fig_name + '.png', PD2=PD.cpu().numpy())


        x0 = x

        if grad_PI:
            x = self.pers_imager.transform(x).reshape(-1)
            #x = self.pers_imager.transform(x.detach().cpu(), use_cuda = True).reshape(-1).cuda()
        else:
            x = torch.tensor(self.pers_imager_nograd.transform(np.array(x.detach().cpu())).reshape(-1)).cuda()
        t3 = time.time()
        '''
        x = self.lin1(x)
        x = F.relu(x)
        #x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        x = F.relu(x)
        #x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.mean(dim = 0)
        x = self.lin4(x)
        x = F.relu(x)
        #x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        '''
        #x = F.normalize(x, dim = -1)

        return x0, x, loss0, loss_xy0, loss_xd0, loss_yd0, t2 - t1, t3 - t2


    def compute_PD_loss(self, PD1, PD2, p = 1, M = 50, kernel = 'wasserstein', num_models = 1, type = 'train'):
        # M is for sliced
        # p denotes p-wassersein
        if kernel == 'sliced':
            diag_theta = torch.FloatTensor([np.cos(0.25 * np.pi), np.sin(0.25 * np.pi)]).cuda()
            l_theta1 = [torch.dot(diag_theta, x) for x in PD1]
            l_theta2 = [torch.dot(diag_theta, x) for x in PD2]
            PD_delta1 = [[torch.sqrt(x ** 2 / 2.0)] * 2 for x in l_theta1]
            PD_delta2 = [[torch.sqrt(x ** 2 / 2.0)] * 2 for x in l_theta2]
            loss = torch.FloatTensor([0]).cuda()
            theta = 0.5
            step = 1.0 / M
            for i in range(M):
                l_theta = torch.FloatTensor([np.cos(theta * np.pi), np.sin(theta * np.pi)]).cuda()
                V1 = [torch.dot(l_theta, x) for x in PD1] + [l_theta[0] * x[0] + l_theta[1] * x[1] for x in PD_delta2]
                V2 = [torch.dot(l_theta, x) for x in PD2] + [l_theta[0] * x[0] + l_theta[1] * x[1] for x in PD_delta1]
                loss += step * cityblock(sorted(V1), sorted(V2))
                theta += step
        elif kernel == 'wasserstein':
            loss = torch.FloatTensor([0]).cuda()
            loss_xy = torch.FloatTensor([0]).cuda(); loss_xd = torch.FloatTensor([0]).cuda(); loss_yd = torch.FloatTensor([0]).cuda();
            # first choice
            #loss += gdw.wasserstein_distance(PD1, PD2, order=p, enable_autodiff=True)[0]
            # second choice
            if type == 'train':
                temp_loss, ind_tmp_test, wxy, wxd, wyd = gdw.wasserstein_distance(PD1, PD2, order=p, enable_autodiff=True, num_models = num_models)
                loss += temp_loss
                loss_xy += wxy; loss_xd += wxd; loss_yd += wyd
            else:
                temp_loss, ind_tmp_test, wxy, wxd, wyd = gdw.wasserstein_distance_inference(PD1, PD2, order=p, enable_autodiff=True)
                loss += temp_loss
                loss_xy += wxy; loss_xd += wxd; loss_yd += wyd
        return loss, ind_tmp_test, loss_xy, loss_xd, loss_yd





class Base_Model(torch.nn.Module):
    def __init__(self, in_dim = 1, hidden_dim = 32, dropout = 0.2, type = 'GCN', out_dim = 2, new_node_feat = True, use_edge_attn = True):
        super(Base_Model, self).__init__()
        if type == 'GCN':
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, out_dim)
            self.conv4 = GCNConv(hidden_dim, hidden_dim)
            self.conv5 = GCNConv(hidden_dim, hidden_dim)
            '''
            self.l1 = Linear(in_dim, hidden_dim)
            self.l2 = Linear(hidden_dim, hidden_dim)
            self.l3 = Linear(hidden_dim, out_dim)
            self.l4 = Linear(hidden_dim, hidden_dim)
            self.l5 = Linear(hidden_dim, hidden_dim)
            self.BN = BatchNorm1d(hidden_dim)
            '''
        elif type == 'GIN':
            #self.conv1 = GINConv(
            #    Sequential(Linear(in_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
            #               Linear(hidden_dim, hidden_dim), ReLU()))
            self.conv1 = GINConv(
                    Sequential(Linear(in_dim, hidden_dim), ReLU()))
            self.conv2 = GINConv(
                    Sequential(Linear(hidden_dim, hidden_dim), ReLU()))
            self.conv3 = GINConv(
                    Sequential(Linear(hidden_dim, out_dim)))
            self.conv4 = GINConv(
                Sequential(Linear(hidden_dim, hidden_dim), ReLU()))
            self.conv5 = GINConv(
                Sequential(Linear(hidden_dim, hidden_dim), ReLU()))
        elif type == 'PDGNN':
            self.conv1 = PDConv(in_dim, hidden_dim, new_node_feat = new_node_feat)
            self.conv2 = PDConv(hidden_dim, hidden_dim, double_input = True, new_node_feat = new_node_feat)
            self.conv3 = PDConv(hidden_dim, int(out_dim / 2), double_input = True, new_node_feat = new_node_feat)
            self.conv4 = PDConv(hidden_dim, hidden_dim, double_input=True, new_node_feat = new_node_feat)
            self.conv5 = PDConv(hidden_dim, hidden_dim, double_input=True, new_node_feat = new_node_feat)
        elif type == 'GAT':
            from Knowledge_Distillation.gat_conv import GATConv
            # from torch_geometric.nn import GATConv
            self.conv1 = GATConv(in_dim, hidden_dim, concat = False, new_node_feat = new_node_feat, use_edge_attn = use_edge_attn)
            self.conv2 = GATConv(hidden_dim, hidden_dim, double_input = True, concat = False, new_node_feat = new_node_feat, use_edge_attn = use_edge_attn)
            self.conv3 = GATConv(hidden_dim, int(out_dim / 2), double_input = True, concat = False, new_node_feat = new_node_feat, use_edge_attn = use_edge_attn)
            self.conv4 = GATConv(hidden_dim, hidden_dim, double_input = True, concat = False, new_node_feat = new_node_feat, use_edge_attn = use_edge_attn)
            self.conv5 = GATConv(hidden_dim, hidden_dim, double_input = True, concat = False, new_node_feat = new_node_feat, use_edge_attn = use_edge_attn)
            #self.conv1 = GATConv(in_dim, hidden_dim, concat=False)
            #self.conv2 = GATConv(hidden_dim, hidden_dim, concat=False)
            #self.conv3 = GATConv(hidden_dim, out_dim, concat=False)
            #self.conv4 = GATConv(hidden_dim, hidden_dim, concat=False)
            #self.conv5 = GATConv(hidden_dim, hidden_dim, concat=False)
        elif type == 'GAT_original':
            from torch_geometric.nn import GATConv
            self.conv1 = GATConv(in_dim, hidden_dim, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, concat=False)
            self.conv3 = GATConv(hidden_dim, out_dim, concat=False)
            self.conv4 = GATConv(hidden_dim, hidden_dim, concat=False)
            self.conv5 = GATConv(hidden_dim, hidden_dim, concat=False)
        elif type == 'SAGE':
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, out_dim)
            self.conv4 = SAGEConv(hidden_dim, hidden_dim)
            self.conv5 = SAGEConv(hidden_dim, hidden_dim)
            self.BN = BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.type = type
        self.out_dim = out_dim

    def forward(self, x, edge_index):
        x_birth = x
        x_size = x.size()
        if x_size[0] == 0:
            return torch.zeros([0, 2]).cuda()
        if self.type != 'GIN':
            x = F.dropout(x, p = self.dropout, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.prelu(x, weight = torch.tensor(0.1).cuda())
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.prelu(x, weight = torch.tensor(0.1).cuda())
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv4(x, edge_index)
            x = F.prelu(x, weight = torch.tensor(0.1).cuda())
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
            #x += x1


        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv4(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
        #x = global_add_pool(x, batch = torch.LongTensor([0 for _ in range(len(x))]).cuda()
        #)
        #x = global_mean_pool(x, batch = torch.LongTensor([0 for _ in range(len(x))])#.cuda()
        #)

        #if self.out_dim == 1:
        #    x = torch.cat((x_birth, x), dim = 1)
        # in case x <= 0 to get nan grad
        #x = torch.abs(x)
        #x[:, 1] += 1e-4 * (x[:, 0] == x[:, 1])
        return x
