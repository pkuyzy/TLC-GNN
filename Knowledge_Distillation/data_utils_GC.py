import os.path
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gudhi as gd
import networkx as nx
from scipy.sparse import csgraph
from scipy.io import loadmat
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from torch_geometric.utils import remove_self_loops
import sg2dgm.PersistenceImager as pimg
import learnable_filter.loaddatas_LP as lds
from loaddatas_LP_arxiv import get_edges_split
import torch
import sys
#from spectral import SpectralClustering
from tqdm import tqdm
import random
import pickle
#from new_PD import perturb_filter_function, Union_find
from Knowledge_Distillation.accelerated_PD import perturb_filter_function, Union_find, Accelerate_PD
from torch_geometric.datasets import TUDataset, ZINC
import time
from ogb.graphproppred import PygGraphPropPredDataset
from Knowledge_Distillation.SBM_Model import create_SBM_Model

class ricci_filtration():
    def __init__(self, g, u, hop, ricci_curv):
        self.g = g
        self.n = len(g)
        self.root = u
        self.hop = hop
        self.ricci_curv = ricci_curv

    def build_fv(self, weight_graph=True, norm = False):
        for x in self.g.nodes():
            if x in [self.root]:
                self.g.nodes[x]['sum'] = 0
            else:
                if weight_graph:
                    try:
                        path_1 = nx.dijkstra_path(self.g, x, self.root, weight='weight')
                        dist_1 = sum([self.ricci_curv[(path_1[y], path_1[y + 1])] + 1 for y in range(len(path_1) - 1)])
                    except BaseException:
                        dist_1 = 100
                else:
                    try:
                        dist_1 = nx.shortest_path_length(self.g, x, self.root)
                    except BaseException:
                        dist_1 = 100
                self.g.nodes[x]['sum'] = dist_1
        if norm:
            norm_scaler_sum = float(max([self.g.nodes[x]['sum'] for x in self.g.nodes()]))
            for x in self.g.nodes():
                self.g.nodes[x]['sum'] /= (norm_scaler_sum + 1e-10)
        return self.g

def apply_graph_extended_persistence(num_vertices, xs, ys, filtration_val):
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]
    dgmOrd0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmOrd0 if p[0] == 0]) if len(dgmOrd0) else np.empty([0,2])
    dgmRel1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmRel1 if p[0] == 1]) if len(dgmRel1) else np.empty([0,2])
    dgmExt0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmExt0 if p[0] == 0]) if len(dgmExt0) else np.empty([0,2])
    dgmExt1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) for p in dgmExt1 if p[0] == 1]) if len(dgmExt1) else np.empty([0,2])
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1

def original_extended_persistence(subgraph, filtration_val):
    simplex_filter = perturb_filter_function(subgraph, filtration_val)
    dgmOrd0, dgmExt0, dgmRel1, Pos_edges, Neg_edges = Union_find(simplex_filter)
    dgmExt1 = Accelerate_PD(Pos_edges, Neg_edges, simplex_filter)
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1

def new_extended_persistence(subgraph, filtration_val):
    simplex_filter = perturb_filter_function(subgraph, filtration_val)
    dgmOrd0 ,dgmExt0, dgmRel1, dgmExt1 = Union_find(simplex_filter)

    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1

def hks_signature(subgraph, time):
    A = nx.adjacency_matrix(subgraph)
    L = csgraph.laplacian(A, normed=True)
    egvals, egvectors = eigh(L.toarray())
    return np.square(egvectors).dot(np.diag(np.exp(-time * egvals))).sum(axis=1)



def compute_persistence_image(g, filt = 'hks', hks_time = 0.1, hop = 2, ricci_curv = None, mode = 'PI', num_models = 5, max_loop_len = 10, cycle_the = 2):
    subgraph = g

    # prepare computation of extended persistence
    if len(subgraph.edges()) == 0 or not nx.is_connected(subgraph):
        return None, None
    #num_vertices = len(subgraph.nodes())
    #edge_list = np.array([i for i in subgraph.edges()])
    #xs = edge_list[:, 0]
    #ys = edge_list[:, 1]
    if len(subgraph.edges()) > 0:
        edge_index = torch.Tensor([[e[0], e[1]] for e in subgraph.edges()]).transpose(0, 1).long()
    else:
        edge_index = torch.Tensor([[0], [0]]).long()

    # compute filter function
    if filt == 'hks':
        filtration_val = hks_signature(subgraph, time=hks_time)
        filtration_val /= (max(filtration_val) + + 1e-10)
    elif filt == 'degree':
        filtration_val = [subgraph.degree()[i] for i in subgraph.nodes()]
        filtration_val = [fv / (max(filtration_val) + 1e-10) for fv in filtration_val]
    elif filt == 'ricci':
        for i in ricci_curv:
            subgraph[i[0]][i[1]]['weight'] = i[2] + 1
            subgraph[i[1]][i[0]]['weight'] = i[2] + 1
        fil = ricci_filtration(subgraph, 0, hop, ricci_curv = ricci_curv)
        new_g = fil.build_fv(weight_graph=True, norm=True)
        filtration_val = [new_g.nodes[i]['sum'] for i in new_g.nodes()]
    else:
        print("Error: 'filt' should be 'hks', 'degree' or 'ricci'! ")
        sys.exit()

    # generate edge number
    cnt_edge = 0
    for edge in subgraph.edges():
        u1, v1 = edge[0], edge[1]
        if 'num' not in subgraph[u1][v1]:
            subgraph[u1][v1]['num'] = cnt_edge
            subgraph[v1][u1]['num'] = cnt_edge
            cnt_edge += 1

    if mode == 'PI':
        t = time.time()
        '''
        num_vertices = len(subgraph.nodes())
        edge_list = np.array([i for i in subgraph.edges()])
        xs = edge_list[:, 0]
        ys = edge_list[:, 1]
        dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(num_vertices, xs, ys, filtration_val)
        '''
        #dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = new_extended_persistence(subgraph, filtration_val)

        dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = original_extended_persistence(subgraph, filtration_val)

        t1 = time.time()
        PD_time = t1 - t
        pers_imager = pimg.PersistenceImager(resolution=5)
        PI0 = pers_imager.transform(dgmOrd0).reshape(-1) if len(dgmOrd0) > 0 else np.zeros(25)
        PI1 = pers_imager.transform(dgmExt1).reshape(-1) if len(dgmExt1) > 0 else np.zeros(25)
        if len(dgmOrd0) == 0:
            pers_img = PI1
        elif len(dgmExt1) == 0:
            pers_img = PI0
        else:
            pers_img = pers_imager.transform(np.concatenate((dgmOrd0, dgmExt1))).reshape(-1)
        PI_time = time.time() - t1
        #return np.concatenate((dgmOrd0, dgmExt0)), np.array(dgmExt1), pers_img.reshape(-1), filtration_val, edge_index, Loop_features, Loop_edge_indices
        return np.array(dgmOrd0), np.array(dgmExt1), pers_img, filtration_val, edge_index, PI0, PI1, PD_time, PI_time

    elif mode == 'filtration':
        #return filtration_val, edge_index, Loop_features, Loop_edge_indices
        return filtration_val, edge_index

def compute_ricci_curvature(data, data_name):
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    filename = '/data1/curvGN_LP/data/data/KD/curvature/graph_' + data_name + '_removevaltest.edge_list'
    #filename = './data/curvature/graph_' + data_name + '.edge_list'
    if False:
        print("curvature file exists, directly loading")
        ricci_list = load_ricci_file(filename)
    else:
        #print("start writing ricci curvature")
        Gd = nx.Graph()
        ricci_edge_index_ = np.array(data.edge_index)
        ricci_edge_index = [(ricci_edge_index_[0, i],
                         ricci_edge_index_[1, i]) for i in
                        range(np.shape(data.edge_index)[1])]
        Gd.add_edges_from(ricci_edge_index)
        Gd_OT = OllivierRicci(Gd, alpha=0.5, method="Sinkhorn", verbose="INFO")
        #Gd_OT = OllivierRicci(Gd, alpha=0.5, method="OTD", verbose="INFO")
        #print("adding edges finished")
        Gd_OT.compute_ricci_curvature()
        ricci_list = []
        for n1, n2 in Gd_OT.G.edges():
            ricci_list.append([n1, n2, Gd_OT.G[n1][n2]['ricciCurvature']])
            ricci_list.append([n2, n1, Gd_OT.G[n1][n2]['ricciCurvature']])
        ricci_list = sorted(ricci_list)
        #print("computing ricci curvature finished")
        '''
        ricci_file = open(filename, 'w')
        for ricci_i in range(len(ricci_list)):
            ricci_file.write(
                str(ricci_list[ricci_i][0]) + " " +
                str(ricci_list[ricci_i][1]) + " " +
                str(ricci_list[ricci_i][2]) + "\n")
        ricci_file.close()
        '''
    return ricci_list

def load_ricci_file(filename):
    if os.path.exists(filename):
        f = open(filename)
        cur_list = list(f)
        ricci_list = [[] for i in range(len(cur_list))]
        for i in range(len(cur_list)):
            ricci_list[i] = [num(s) for s in cur_list[i].split(' ', 2)]
        ricci_list = sorted(ricci_list)
        return ricci_list
    else:
        print("Error: no curvature files found")

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)


def call(dataset,name, filt = 'degree', hks_time = 10, mode = 'PI', num_models = 5, gn = 0):


    dict_store = {}
    pbar_edge = tqdm(total=gn)
    total_time_PD = 0
    total_time_PI = 0
    for tt in range(gn):
        data = dataset[tt]
        g = nx.Graph()
        g.add_nodes_from([i for i in range(data.num_nodes)])
        ricci_edge_index_ = np.array(remove_self_loops((data.edge_index.cpu()))[0])
        ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                            range(np.shape(ricci_edge_index_)[1])]
        g.add_edges_from(ricci_edge_index)
        #ricci_curv = compute_ricci_curvature(data, name)
        ricci_curv = None

        # only for cora and citeseer
        largest_cc = max(nx.connected_components(g), key=len)
        g = nx.convert_node_labels_to_integers(g.subgraph(largest_cc))
        print("nodes: {}, edges: {}".format(len(g.nodes()), len(g.edges())))

        dict_store[tt] = \
            compute_persistence_image(g, filt = filt, hks_time = hks_time, hop = 2, ricci_curv = ricci_curv, mode = mode, num_models = num_models)
        print(dict_store[tt][-2:])
        if len(dict_store[tt]) > 2:
            total_time_PD += dict_store[tt][-2]
            total_time_PI += dict_store[tt][-1]
        pbar_edge.update(1)

    pbar_edge.close()


    # for time evaluation
    save_name = '/data1/curvGN_LP/data/data/KD/' + name + '_' + filt + '_total_test.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)


    '''
    # original save name, for further evaluation
    if filt != 'hks':
        save_name = '/data1/curvGN_LP/data/data/KD/' + name + '_' + filt + '.pkl'
    else:
        save_name = '/data1/curvGN_LP/data/data/KD/' + name + '_' + filt + str(hks_time) + '.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)
    '''


    return total_time_PD, total_time_PI

if __name__ == "__main__":
    #d_names = ['MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY' ,'REDDIT-BINARY', 'ogbg-molhiv']
    #d_names = ['ZINC']
    #GN = [188, 600, 1113, 5000, 1000, 2000, 41127]
    #GN = [10000]
    #d_names = ['Cora', 'Citeseer', 'PubMed']
    #GN = [1, 1, 1]
    d_names = ['SBM']
    #GN = [21] # for node
    GN = [11] # for density
    for (d_name, gn) in zip(d_names, GN):
        if d_name in ['MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY' ,'REDDIT-BINARY']:
            dataset = TUDataset('/data1/curvGN_LP/data/data/', d_name)
            save_name = dataset.name
            data_name = dataset.name
        elif d_name in ['ogbg-molhiv', 'ogbg-molpcba']:
            dataset = PygGraphPropPredDataset(name = d_name)
            save_name = d_name
            data_name = d_name
        elif d_name in ['ZINC']:
            dataset = ZINC(root = './dataset', subset = True)
            save_name = d_name
            data_name = d_name
        elif d_name in ['Cora', 'Citeseer', 'PubMed']:
            dataset = lds.loaddatas('Planetoid', d_name)
            save_name = d_name
            data_name = d_name
        elif d_name in ['SBM']:
            dataset = create_SBM_Model()
            save_name = d_name
            data_name = d_name + '_density'

        '''
        # original, for data generation
        for filt in ['degree', 'hks']:
            print(data_name)
            print(filt)
            if filt == 'hks':
                for hks_time in [0.1, 10]:
                    call(dataset, data_name, filt, hks_time = hks_time, num_models = 10, gn = gn)
            else:
                call(dataset, data_name, filt, hks_time=10, num_models=10, gn = gn)
        '''
        # for large sparse graphs, cora/citeseer/pubmed, and SBM
        print(call(dataset, data_name, 'degree', hks_time=10, gn=gn))