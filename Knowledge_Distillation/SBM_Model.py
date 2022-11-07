import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


#create sbm graph dataset

# set density unchanged, and change node number
def create_SBM_Model_for_node(node_start = 100, node_step = 10, node_end = 310, p_intra = 0.6, p_inter = 0.9 ,f_dim = 100, seed = 1234):
    data_list = [Data() for _ in np.arange(node_start, node_end, node_step)]
    np.random.seed(seed)
    for dataid, node_num in enumerate(np.arange(node_start, node_end, node_step)):
        #print(dataid)
        c1,c2,c3,c4,c5 = np.arange(0, node_num, int(node_num / 5))
        p_list=[0.1*i for i in range(10)]
        adj=np.random.rand(node_num,node_num)
        adj=(adj+adj.T)/2
        x=torch.tensor(np.random.rand(node_num,f_dim),dtype=torch.float)
        y=torch.cat((torch.tensor([0]*c1),torch.tensor([1]*c1),torch.tensor([2]*c1),torch.tensor([3]*c1),torch.tensor([4]*c1)))
        end=node_num*node_num
        inter_idx=adj>p_inter
        intra_idx=adj>p_intra
        inter_idx[c4:,c3:c4],inter_idx[c3:c4,c4:]=False,False
        inter_idx[c1:c3,c3:c4],inter_idx[c3:c4,c1:c3]=False,False
        inter_idx[c1:c4,c4:],inter_idx[c4:,c1:c4]=False,False
        inter_idx[:c1,:c1]=intra_idx[:c1,:c1]
        inter_idx[c1:c2,c1:c2]=intra_idx[c1:c2,c1:c2]
        inter_idx[c2:c3,c2:c3]=intra_idx[c2:c3,c2:c3]
        inter_idx[c3:c4,c3:c4]=intra_idx[c3:c4,c3:c4]
        inter_idx[c4:c5,c4:c5]=intra_idx[c4:c5,c4:c5]
        adj_b=inter_idx
        adj_b.flat[:end:node_num+1]=False
        edge_index=[(i,j) for i in range(node_num) for j in range(node_num) if adj_b[i,j]]
        data=Data(x=x,edge_index=torch.tensor(edge_index).transpose(0,1),y=y)
        data_list[dataid]=data
    return data_list

# set node number unchanged, and change density
def create_SBM_Model(node_num = 250, p_intra_start = 0.5, p_inter_start = 0.85, f_dim = 100, seed = 1234):
    data_list = [Data() for _ in range(11)]
    np.random.seed(seed)
    for dataid in range(11):
        #print(dataid)
        p_intra = p_intra_start + dataid * 0.02
        p_inter = p_inter_start + dataid * 0.01
        c1,c2,c3,c4,c5 = np.arange(0, node_num, int(node_num / 5))
        p_list=[0.1*i for i in range(10)]
        adj=np.random.rand(node_num,node_num)
        adj=(adj+adj.T)/2
        x=torch.tensor(np.random.rand(node_num,f_dim),dtype=torch.float)
        y=torch.cat((torch.tensor([0]*c1),torch.tensor([1]*c1),torch.tensor([2]*c1),torch.tensor([3]*c1),torch.tensor([4]*c1)))
        end=node_num*node_num
        inter_idx=adj>p_inter
        intra_idx=adj>p_intra
        inter_idx[c4:,c3:c4],inter_idx[c3:c4,c4:]=False,False
        inter_idx[c1:c3,c3:c4],inter_idx[c3:c4,c1:c3]=False,False
        inter_idx[c1:c4,c4:],inter_idx[c4:,c1:c4]=False,False
        inter_idx[:c1,:c1]=intra_idx[:c1,:c1]
        inter_idx[c1:c2,c1:c2]=intra_idx[c1:c2,c1:c2]
        inter_idx[c2:c3,c2:c3]=intra_idx[c2:c3,c2:c3]
        inter_idx[c3:c4,c3:c4]=intra_idx[c3:c4,c3:c4]
        inter_idx[c4:c5,c4:c5]=intra_idx[c4:c5,c4:c5]
        adj_b=inter_idx
        adj_b.flat[:end:node_num+1]=False
        edge_index=[(i,j) for i in range(node_num) for j in range(node_num) if adj_b[i,j]]
        data=Data(x=x,edge_index=torch.tensor(edge_index).transpose(0,1),y=y)
        data_list[dataid]=data
    return data_list