import torch_geometric.datasets
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import sys
import networkx as nx
import os
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import remove_self_loops
import torch_geometric.datasets
from sg2dgm import riccidist2dgm as sg2dgm
#from sg2dgm import riccidist2dgm_c as sg2dgm

def loaddatas(d_name):
    if d_name in ["PPI"]:
        dataset = torch_geometric.datasets.PPI('./data/' + d_name)
    elif d_name == 'Cora':
        dataset = torch_geometric.datasets.Planetoid('./data/'+d_name,d_name,transform=T.NormalizeFeatures())
    elif d_name in ['Citeseer', 'PubMed']:
        dataset = torch_geometric.datasets.Planetoid('./data/' + d_name, d_name)
    elif d_name in ["Computers", "Photo"]:
        dataset = torch_geometric.datasets.Amazon('./data/'+d_name,d_name)
    return dataset

def get_edges_split(data, val_prop = 0.2, test_prop = 0.2, seed = 1234):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    ricci_edge_index_ = np.array((data.edge_index))
    ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                        range(np.shape(ricci_edge_index_)[1])]
    g.add_edges_from(ricci_edge_index)
    adj = nx.adjacency_matrix(g)

    return get_adj_split(adj,val_prop = val_prop, test_prop = test_prop, seed = seed)

#def get_adj_split(adj, val_prop = 0.05, test_prop = 0.1, seed=1234):
def get_adj_split(adj, val_prop=0.05, test_prop=0.1, seed=1234):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def compute_persistence_image(data, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, data_name, hop = 1):
    if data_name == "photo":
        data_name = "Photo"
    if data_name == "computers":
        data_name = "Computers"

    filename = './data/TLCGNN/' + data_name + '.npy'
    if os.path.exists(filename):
        return np.load(filename)
    total_edges = np.concatenate(
        (train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false))
    data.train_pos, data.train_neg = len(train_edges), len(train_edges_false)
    data.val_pos, data.val_neg = len(val_edges), len(val_edges_false)
    data.test_pos, data.test_neg = len(test_edges), len(test_edges_false)
    data.total_edges = total_edges

    # delete val_pos and test_pos
    edge_list = np.array(data.edge_index).T.tolist()
    for edges in val_edges:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    for edges in test_edges:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    data.edge_index = torch.Tensor(edge_list).long().transpose(0, 1)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    # generate graph for computing persistence diagram
    g = nx.Graph()
    ricci_edge_index_ = np.array(remove_self_loops((data.edge_index.cpu()))[0])
    ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                        range(np.shape(ricci_edge_index_)[1])]
    g.add_edges_from(ricci_edge_index)
    print(len(g.edges()))

    # ricci_cur = compute_ricci_flow(data, d_name)
    ricci_cur = compute_ricci_curvature(data)

    # compute sg2dgm and save in a dict
    pi = sg2dgm.graph2pi(g, ricci_curv=ricci_cur)
    pi.get_pimg_for_all_edges(total_edges, cores=16, hop=hop, norm=True, extended_flag=True,
                                  resolution=5, descriptor='sum')
    np.save(filename,pi.pi_sg)
    return pi.pi_sg

def compute_ricci_curvature(data):
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    print("start writing ricci curvature")
    Gd = nx.Graph()
    ricci_edge_index_ = np.array(data.edge_index)
    ricci_edge_index = [(ricci_edge_index_[0, i],
                         ricci_edge_index_[1, i]) for i in
                        range(np.shape(data.edge_index)[1])]
    Gd.add_edges_from(ricci_edge_index)
    Gd_OT = OllivierRicci(Gd, alpha=0.5, method="Sinkhorn", verbose="INFO")
    print("adding edges finished")
    Gd_OT.compute_ricci_curvature()
    ricci_list = []
    for n1, n2 in Gd_OT.G.edges():
        ricci_list.append([n1, n2, Gd_OT.G[n1][n2]['ricciCurvature']])
        ricci_list.append([n2, n1, Gd_OT.G[n1][n2]['ricciCurvature']])
    ricci_list = sorted(ricci_list)
    print("computing ricci curvature finished")
    return ricci_list


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)



