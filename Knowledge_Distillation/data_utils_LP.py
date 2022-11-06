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
from Knowledge_Distillation.spectral import SpectralClustering
from tqdm import tqdm
import random
import pickle
#from new_PD import perturb_filter_function, Union_find
from Knowledge_Distillation.accelerated_PD import perturb_filter_function, Union_find, Accelerate_PD
import time

class ricci_filtration():
    def __init__(self, g, u, v, hop, ricci_curv):
        self.g = g
        self.n = len(g)
        self.root_1 = u
        self.root_2 = v
        self.hop = hop
        self.ricci_curv = ricci_curv

    def build_fv(self, weight_graph=True, norm = False):
        for x in self.g.nodes():
            if x in [self.root_1, self.root_2]:
                self.g.nodes[x]['sum'] = 0
            else:
                if weight_graph:
                    try:
                        path_1 = nx.dijkstra_path(self.g, x, self.root_1, weight='weight')
                        dist_1 = sum([self.ricci_curv[(path_1[y], path_1[y + 1])] + 1 for y in range(len(path_1) - 1)])
                    except BaseException:
                        dist_1 = 100
                    try:
                        path_2 = nx.dijkstra_path(self.g, x, self.root_2, weight='weight')
                        dist_2 = sum([self.ricci_curv[(path_2[y], path_2[y + 1])] + 1 for y in range(len(path_2) - 1)])
                    except BaseException:
                        dist_2 = 100
                else:
                    try:
                        dist_1 = nx.shortest_path_length(self.g, x, self.root_1)
                    except BaseException:
                        dist_1 = 100
                    try:
                        dist_2 = nx.shortest_path_length(self.g, x, self.root_2)
                    except BaseException:
                        dist_2 = 100
                self.g.nodes[x]['sum'] = dist_1 + dist_2
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



def compute_persistence_image(g, u, v, filt = 'hks', hks_time = 0.1, hop = 2, ricci_curv = None, mode = 'PI', num_models = 5, max_loop_len = 10, cycle_the = 2):
    # extract subgraph
    root = u
    nodes_u = [root] + [x for u, x in nx.bfs_edges(g, root, depth_limit=hop)]
    root = v
    nodes_v = [root] + [x for v, x in nx.bfs_edges(g, root, depth_limit=hop)]
    nodes = list(set(nodes_u) & set(nodes_v)) + [u] + [v]
    subgraph = g.subgraph(nodes)
    subgraph = nx.convert_node_labels_to_integers(subgraph, label_attribute="old_label")

    # prepare computation of extended persistence
    if len(subgraph.edges()) == 0:
        return None, None
    # num_vertices = len(subgraph.nodes())
    # edge_list = np.array([i for i in subgraph.edges()])
    # xs = edge_list[:, 0]
    # ys = edge_list[:, 1]
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
        dict_node = {}
        for new_label in subgraph._node:
            dict_node[subgraph._node[new_label]['old_label']] = new_label
        new_ricci_curv = {}
        for i in ricci_curv:
            if i[0] in dict_node and i[1] in dict_node:
                new_ricci_curv[(dict_node[i[0]], dict_node[i[1]])] = i[2]
                new_ricci_curv[(dict_node[i[1]], dict_node[i[0]])] = i[2]
                subgraph[dict_node[i[0]]][dict_node[i[1]]]['weight'] = i[2] + 1
                subgraph[dict_node[i[1]]][dict_node[i[0]]]['weight'] = i[2] + 1
        fil = ricci_filtration(subgraph, dict_node[u], dict_node[v], hop, ricci_curv = new_ricci_curv)
        new_g = fil.build_fv(weight_graph=True, norm=True)
        filtration_val = [new_g.nodes[i]['sum'] for i in new_g.nodes()]
    else:
        print("Error: 'filt' should be 'hks', 'degree' or 'ricci'! ")
        sys.exit()

    # generate edge number
    cnt_edge = 0
    for edge in subgraph.edges():
        u, v = edge[0], edge[1]
        if 'num' not in subgraph[u][v]:
            subgraph[u][v]['num'] = cnt_edge
            subgraph[v][u]['num'] = cnt_edge
            cnt_edge += 1

    '''
    # compute edge_index and feature for cycles
    Loop_features = []; Loop_edge_indices = []
    root_nodes = select_node_graph_cluster(subgraph, num_models = num_models)
    for cnt_model in range(num_models):
        if root_nodes is not None:
            tmp_root = root_nodes[cnt_model].tolist()
        else:
            tmp_root = None
        # in case changing subgraph
        subgraph_clone = subgraph.copy()
        Cycle2edge, Cycle2pers = generate_bfs_tree(subgraph_clone, filter_value = filtration_val, max_loop_len = max_loop_len, pos_root = tmp_root)
        loop_edge_index =  generate_edge_index(Cycle2edge, cycle_the = cycle_the)
        Loop_features.append(Cycle2pers)
        Loop_edge_indices.append(loop_edge_index)
    '''

    if mode == 'PI':
        #dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(num_vertices, xs, ys, filtration_val)
        #dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = new_extended_persistence(subgraph, filtration_val)
        t = time.time()
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
    if os.path.exists(filename):
        print("curvature file exists, directly loading")
        ricci_list = load_ricci_file(filename)
    else:
        print("start writing ricci curvature")
        Gd = nx.Graph()
        ricci_edge_index_ = np.array(data.edge_index)
        ricci_edge_index = [(ricci_edge_index_[0, i],
                         ricci_edge_index_[1, i]) for i in
                        range(np.shape(data.edge_index)[1])]
        Gd.add_edges_from(ricci_edge_index)
        Gd_OT = OllivierRicci(Gd, alpha=0.5, method="Sinkhorn", verbose="INFO")
        #Gd_OT = OllivierRicci(Gd, alpha=0.5, method="OTD", verbose="INFO")
        print("adding edges finished")
        Gd_OT.compute_ricci_curvature()
        ricci_list = []
        for n1, n2 in Gd_OT.G.edges():
            ricci_list.append([n1, n2, Gd_OT.G[n1][n2]['ricciCurvature']])
            ricci_list.append([n2, n1, Gd_OT.G[n1][n2]['ricciCurvature']])
        ricci_list = sorted(ricci_list)
        print("computing ricci curvature finished")
        ricci_file = open(filename, 'w')
        for ricci_i in range(len(ricci_list)):
            ricci_file.write(
                str(ricci_list[ricci_i][0]) + " " +
                str(ricci_list[ricci_i][1]) + " " +
                str(ricci_list[ricci_i][2]) + "\n")
        ricci_file.close()
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


def select_node_graph_cluster(g, num_models = 10):
    # need to modify SpectralClustering
    center = []
    for sub_c in nx.connected_components(g):
        sub_g = g.subgraph(sub_c)
        adj_mat = nx.to_numpy_matrix(sub_g)
        n_cluster = num_models if len(sub_g.nodes()) >= num_models else len(sub_g.nodes())
        if len(sub_g.nodes()) < num_models:
            sc = SpectralClustering(n_cluster, affinity='precomputed', n_init=100)
        else:
            sc = SpectralClustering(n_cluster, affinity='precomputed', n_init=100)
        try:
            sc.fit(adj_mat)
        except BaseException:
            print("Do not find root node")
            return None

        tmp_center = np.array(np.array(sub_g.nodes())[sc.centers_].tolist() + [random.choice([__ for __ in sub_g.nodes()]) for _ in
                                                                  range(num_models - n_cluster)])
        center.append(tmp_center)
    center = np.array(center)
    return center.T

def find_loop(g, filter_value, u, v, node2father, node2tree, root_node, max_loop_len = 10):
    path_u = [u]; path_v = [v]; path = []
    path_num = []; path_num_u = []; path_num_v = []
    path_pers = []; path_pers_u = [filter_value[u]]; path_pers_v = [filter_value[v]]
    root = root_node[node2tree[u]]
    if root != root_node[node2tree[v]]:
        return [], []
    node = u
    cnt_path_len = 0
    while node != root:
        if cnt_path_len > max_loop_len:
            return [], []
        next_node = node2father[node]
        path_u.append(next_node)
        path_num_u.append(g[node][next_node]['num'])
        path_pers_u.append(filter_value[next_node])
        node = next_node
        cnt_path_len += 1

    node = v
    cnt_path_len = 0
    while node != root:
        if cnt_path_len > max_loop_len:
            return [], []
        next_node =  node2father[node]
        path_v.append(next_node)
        path_num_v.append(g[node][next_node]['num'])
        path_pers_v.append(filter_value[next_node])
        node = next_node
        cnt_path_len += 1

    len_u = len(path_u); len_v = len(path_v)
    if len_u > len_v:
        for v_i in range(len_v):
            if path_u[v_i + len_u - len_v] == path_v[v_i]:
                break
        u_i = v_i + len_u - len_v
    else:
        for u_i in range(len_u):
            if path_u[u_i] == path_v[u_i + len_v - len_u]:
                break
        v_i = u_i + len_v - len_u

    path.append(u)
    path_pers.append(filter_value[u])
    for i in range(1, u_i + 1):
        path.append(path_u[i])
        path_num.append(path_num_u[i - 1])
        path_pers.append(path_pers_u[i])
    for i in range(v_i - 1, -1, -1):
        path.append(path_v[i])
        path_num.append(path_num_v[i])
        path_pers.append(path_pers_v[i])

    return path_num, path_pers

def select_node(sub_g):
    Nodes = [i for i in sub_g.nodes()]
    return random.choice(Nodes)

def make_matrix_tensor(path_num, path_pers, len_edges, max_loop_len = 10):
    cycle2edge = torch.zeros(len_edges)

    # get max / min filter value, wrong
    #cycle2pers = torch.zeros(2)
    # get all filter vaule, right
    cycle2pers = torch.zeros(2 * max_loop_len)


    if len(path_num) != 0:
        for i in path_num:
            cycle2edge[i] = 1

        # get max / min filter value, wrong
        #cycle2pers[0] = min(path_pers)
        #cycle2pers[1] = max(path_pers)

        # get all filter value, right
        for i in range(len(path_pers)):
            cycle2pers[i] = path_pers[i]
    return cycle2edge.long(), cycle2pers


def generate_bfs_tree(g, filter_value, max_loop_len = 10, pos_root = None):
    edge_list = []
    root_node = []
    node2tree = {}
    node2father = {}

    #generate the bfs tree for the undirected graph
    cnt_tree = 0
    for sub_c in nx.connected_components(g):
        sub_g = g.subgraph(sub_c)
        if pos_root == None:
            node = select_node(sub_g)
        else:
            node = pos_root[cnt_tree]
        bfs_tree = list(nx.bfs_edges(sub_g, node))
        edge_list += bfs_tree

        root_node.append(node)
        for sub_node in sub_g.nodes():
            node2tree[sub_node] = cnt_tree
        cnt_tree += 1

    #find the tree edges and find father for every nodes
    for cnt in range(len(edge_list)):
        u, v = edge_list[cnt][0], edge_list[cnt][1]
        # find father for every nodes
        node2father[v] = u
        # find the tree edges
        g[u][v]['tree'] = 1
        g[v][u]['tree'] = 1


    # compute the length of the matrix
    len_matrix = len(g.edges()) - len(edge_list)


    # get max / min value, but it is wrong
    #Cycle2pers = torch.zeros(len_matrix, 2)
    # get all the filter value, is right
    Cycle2pers = torch.zeros(len_matrix, 2 * max_loop_len)

    Cycle2edge = torch.zeros(len_matrix, len(g.edges()))

    #find all the loop generated by non-tree edges
    #print("start find all the loop generated by non-tree edges")
    #pbar_matrix = tqdm(total = len_matrix)
    cnt_matrix = 0
    for edge in g.edges():
        u, v = edge[0], edge[1]
        if 'tree' not in g[u][v]:
            path_num, path_pers = find_loop(g, filter_value, u, v, node2father, node2tree, root_node, max_loop_len)
            if len(path_num) > 0:
                path_num.append(g[u][v]['num'])
            Cycle2edge[cnt_matrix], Cycle2pers[cnt_matrix] = make_matrix_tensor(path_num, path_pers, len(g.edges()), max_loop_len)
            #pbar_matrix.update(1)
            cnt_matrix += 1
    #pbar_matrix.close()

    return Cycle2edge, Cycle2pers

def generate_edge_index(Cycle2edge, cycle_the = 2):
    # generate the edge index for CBGNN

    #print("start to generate edge index for CBGNN")
    Cycle2edge_T = Cycle2edge.T.float()

    if Cycle2edge.size()[0] == 0:
        return torch.LongTensor([[0], [0]])

    from torch_scatter import scatter_add
    Cycle2edge_index = torch.nonzero(Cycle2edge)
    len_loop = len(Cycle2edge)
    out = scatter_add(Cycle2edge_T[Cycle2edge_index[:, 1]], Cycle2edge_index[:, 0], dim = 0, dim_size = len_loop)

    #print(out)
    # new one
    if out.size()[1] < cycle_the:
        return torch.LongTensor([[0], [0]])
    topk = torch.topk(out, cycle_the, dim = 1)[1]
    edge_index = torch.LongTensor(2, cycle_the * len(topk))
    for i in range(cycle_the * len(topk)):
        u = int(i / cycle_the); v = int(i % cycle_the)
        edge_index[0, i] = u; edge_index[1, i] = topk[u, v]

    #print(edge_index.size())

    return edge_index

def call(data,name, filt = 'degree', hks_time = 10, mode = 'PI', num_models = 5):

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
    total_edges = np.concatenate((train_edges,train_edges_false,val_edges,val_edges_false,test_edges,test_edges_false))
    pi_sg_0 = np.zeros((len(total_edges), 25))
    pi_sg_1 = np.zeros((len(total_edges), 25))

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

    ricci_curv = compute_ricci_curvature(data, name)


    dict_store = {}
    g_edges = [edge for edge in g.edges()]
    pbar_edge = tqdm(total=len(g_edges))
    total_time_PD = 0
    total_time_PI = 0
    for tt in range(len(g_edges)):
    #for tt in range(10):
        u, v = g_edges[tt]
        dict_store[tt] = \
            compute_persistence_image(g, u, v, filt = filt, hks_time = hks_time, hop = hop, ricci_curv = ricci_curv, mode = mode, num_models = num_models)
        total_time_PD += dict_store[tt][-2]
        total_time_PI += dict_store[tt][-1]
        pbar_edge.update(1)
        #if tt > 100:
        #    break

    pbar_edge.close()

    '''
    save_name = '/data1/curvGN_LP/data/data/KD/test.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)
    '''
    '''
    if filt != 'hks':
        save_name = '/data1/curvGN_LP/data/data/KD/' + name + '_' + filt + '_new_new.pkl'
    else:
        save_name = '/data1/curvGN_LP/data/data/KD/' + name + '_' + filt + str(hks_time) + '_new_new.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)
    '''
    return total_time_PD, total_time_PI

if __name__ == "__main__":
    '''
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edges_from([(0, 3), (1, 3), (2, 3)])
    root_list = select_node_graph_cluster(g, num_models=1)
    print(root_list)
    needed_root = root_list[0].tolist()
    print(needed_root)
    '''
    '''
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edges_from([(0, 2), (1, 2), (0, 3), (1, 3), (2,3)])
    A = nx.adjacency_matrix(g)
    dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(A, [0, 1, 2, 3])
    print(dgmOrd0)
    print(dgmExt0)
    print(dgmRel1)
    print(dgmExt1)
    '''

    global compute_time
    cmopute_time = 0
    #d_names = ['Cora', 'Citeseer', 'PubMed', 'Photo', 'Computers']
    d_names = ['Photo']
    for d_name in d_names:
        if d_name == 'Cora' or d_name == 'Citeseer' or d_name == 'PubMed':
            d_loader = 'Planetoid'
        elif d_name == 'Computers' or d_name == 'Photo':
            d_loader = 'Amazon'
        elif d_name == 'CS' or d_name == 'Physics':
            d_loader = 'Coauthor'
        else:
            d_loader = 'PPI'
        dataset = lds.loaddatas(d_loader, d_name)
        save_name = dataset.name
        data_name = dataset.name
        for filt in ['ricci', 'degree', 'hks']:
        #for filt in ['hks']:
            if filt == 'hks':
                for hks_time in [0.1, 10]:
                    data = dataset[0]
                    print(call(data, data_name, filt, hks_time = hks_time, num_models = 10))
            else:
                data = dataset[0]
                print(call(data, data_name, filt, hks_time=10, num_models=10))

    '''
    with open('/data1/curvGN_LP/data/data/KD/Cora_hks10.pkl', 'rb') as f:
        dict_save = pickle.load(f)
        print(dict_save)
    '''
    '''
    d_name = 'PPI'
    for cnt in range(1, 20):
        if d_name == 'Cora' or d_name == 'Citeseer' or d_name == 'PubMed':
            d_loader = 'Planetoid'
        elif d_name == 'Computers' or d_name == 'Photo':
            d_loader = 'Amazon'
        elif d_name == 'CS' or d_name == 'Physics':
            d_loader = 'Coauthor'
        else:
            d_loader = 'PPI'
        dataset = lds.loaddatas(d_loader, d_name)
        data = dataset[cnt]
        data_name = "PPI"
        save_name = "PPI_" + str(cnt)
        call(data, data_name, save_name)
    '''