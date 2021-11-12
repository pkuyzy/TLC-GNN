import numpy as np
import networkx as nx
import dionysus as d
import time
import sg2dgm.PersistenceImager as pimg
from sg2dgm.dgformat import *
from multiprocessing.dummy import Pool as ThreadPool
import sys
from sg2dgm.accelerated_PD import perturb_filter_function, Union_find, Accelerate_PD

class filtration():
    def __init__(self, g, u, v, hop, ricci_curv):
        self.g = g
        self.n = len(g)
        self.root_1 = u
        self.root_2 = v
        self.hop = hop
        self.ricci_curv = ricci_curv

    def build_fv(self, weight_graph=False, norm = True):
        for x in self.g.nodes():
            if x in [self.root_1, self.root_2]:
                self.g.nodes[x]['max'] = 0
                self.g.nodes[x]['min'] = 0
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
                self.g.nodes[x]['min'] = min(dist_1, dist_2)
                self.g.nodes[x]['max'] = max(dist_1, dist_2)
                self.g.nodes[x]['sum'] = dist_1 + dist_2
        if norm:
            norm_scaler = float(max([self.g.nodes[x]['max'] for x in self.g.nodes()]))
            norm_scaler_sum = float(max([self.g.nodes[x]['sum'] for x in self.g.nodes()]))
            for x in self.g.nodes():
                self.g.nodes[x]['min'] /= norm_scaler
                self.g.nodes[x]['max'] /= norm_scaler
                self.g.nodes[x]['sum'] /= norm_scaler_sum
        for u, v in self.g.edges():
            self.g[u][v]['min'] = max(self.g.nodes[u]['min'], self.g.nodes[v]['min'])
            self.g[u][v]['max'] = max(self.g.nodes[u]['max'], self.g.nodes[v]['max'])
            self.g[u][v]['sum'] = max(self.g.nodes[u]['sum'], self.g.nodes[v]['sum'])
        return self.g

class graph2dgm():
    def __init__(self, g, **kwargs):
        self.graph = nx.convert_node_labels_to_integers(g)

    def get_simplices(self, gi, key='min'):
        """
        Used by get_diagram function
        :param key: 'min', 'max', 'root_1' or 'root_2'
        """

        assert str(
            type(gi)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        assert len(gi) > 0
        assert len(gi) == max(list(gi.nodes())) + 1
        simplices = list()
        for u, v, data in sorted(gi.edges(data=True), key=lambda x: x[2][key]):
            tup = ([u, v], data[key])
            simplices.append(tup)
        for v, data in sorted(gi.nodes(data=True), key=lambda x: x[1][key]):
            tup = ([v], data[key])
            simplices.append(tup)
        return simplices
    
    def get_desc_simplices(self, gi, key='min'):
        """
        Used by get_diagram function
        :param key: 'min', 'max', 'root_1' or 'root_2'
        """

        assert str(
            type(gi)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        assert len(gi) > 0
        assert len(gi) == max(list(gi.nodes())) + 1
        simplices = list()
        values = nx.get_node_attributes(gi, key)
        for u, v, data in sorted(gi.edges(data=True), key=lambda x: x[2][key]):
            tup = ([u, v], min(values[u],values[v]))
            simplices.append(tup)
        for v, data in sorted(gi.nodes(data=True), key=lambda x: x[1][key]):
            tup = ([v], data[key])
            simplices.append(tup)
        return simplices


    def del_inf(self, dgms):
        # remove inf
        dgms_list = [[], []]

        for i in range(min(2,len(dgms))):
            pt_list = list()
            for pt in dgms[i]:
                if (pt.birth == float('inf')) or (pt.death == float('inf')):
                    pass
                else:
                    pt_list.append(tuple([pt.birth, pt.death]))
            diagram = d.Diagram(pt_list)
            dgms_list[i] = diagram
        return dgms_list

    def compute_PD(self, simplices, sub=True, inf_flag='False'):
        def cmp(a, b):
            return (a > b) - (a < b)

        def compare(s1, s2, sub_flag=True):
            if sub_flag == True:
                if s1.dimension() > s2.dimension():
                    return 1
                elif s1.dimension() < s2.dimension():
                    return -1
                else:
                    return cmp(s1.data, s2.data)
            elif sub_flag == False:
                return -compare(s1, s2, sub_flag=True)

        f = d.Filtration()
        for simplex, time in simplices:
            f.append(d.Simplex(simplex, time))

        f.sort() if sub else f.sort(reverse=True)

        m = d.homology_persistence(f)
        dgms = d.init_diagrams(m, f)

        if inf_flag == 'False':
            dgms = self.del_inf(dgms)
        # for some degenerate case, return dgm(0,0)
        if (dgms == []) or (dgms == None):
            return d.Diagram([[0, 0]])
        return dgms

    def get_diagram(self, g, key='min', one_homology_flag=False):
        g = nx.convert_node_labels_to_integers(g)

        if one_homology_flag:
            epd_dgm = self.epd(g, key=key, pd_flag=False)[1]
            epd_dgm = self.post_process(epd_dgm)
            dgms = [[pt.birth, pt.death] for pt in epd_dgm]
            return dgms

        simplices = self.get_simplices(g, key=key)
        down_simplices = self.get_desc_simplices(g, key=key)
        super_dgms = self.compute_PD(down_simplices, sub=False)
        sub_dgms = self.compute_PD(simplices, sub=True)

        _min = min([g.nodes[n][key] for n in g.nodes()])
        _max = max([g.nodes[n][key] for n in g.nodes()]) + 1e-5  # avoid the extra node lies on diagonal
        p_min = d.Diagram([(_min, _max)])
        p_max = d.Diagram([(_max, _min)])

        sub_dgms[0].append(p_min[0])
        super_dgms[0].append(p_max[0])
        dgms = [[pt.birth, pt.death] for pt in sub_dgms[0]] + [[pt.birth, pt.death] for pt in super_dgms[0]]

        return dgms

    def epd(self, g__, key='min', pd_flag=False, debug_flag=False):
        w = -1

        values = nx.get_node_attributes(g__, key)
        simplices = [[x[0], x[1]] for x in list(g__.edges)] + [[x] for x in g__.nodes()]
        up_simplices = [d.Simplex(s, max(values[v] for v in s)) for s in simplices]
        down_simplices = [d.Simplex(s + [w], min(values[v] for v in s)) for s in simplices]
        if pd_flag:
            down_simplices = []  # mask the extended persistence here

        up_simplices.sort(key=lambda s1: (s1.dimension(), s1.data))
        down_simplices.sort(reverse=True, key=lambda s: (s.dimension(), s.data))
        f = d.Filtration([d.Simplex([w], -float('inf'))] + up_simplices + down_simplices)
        m = d.homology_persistence(f)
        dgms = d.init_diagrams(m, f)
        if debug_flag:
            print('Calling compute_EPD here with success. Print first dgm in dgms')
            print_dgm(dgms[0])
        return dgms

    def post_process(self, dgm, debug_flag=False):
        if len(dgm) == 0:
            return d.Diagram([(0, 0)])
        for p in dgm:
            if p.birth == np.float('-inf'):
                p.birth = 0
            if p.death == np.float('inf'):
                p.death = 0
        if debug_flag:
            print('Before flip:'),
            print_dgm(dgm)
        dgm = flip_dgm(dgm)
        if debug_flag:
            print('After:'),
            print_dgm(dgm)
        return dgm

class graph2pi():
    def __init__(self, g, ricci_curv):
        self.graph = nx.convert_node_labels_to_integers(g, label_attribute="old_label")
        self.dict_node = {}
        for new_label in self.graph._node:
            self.dict_node[self.graph._node[new_label]['old_label']] = new_label
        self.ricci_curv = {}
        for i in ricci_curv:
            self.ricci_curv[(self.dict_node[i[0]],self.dict_node[i[1]])] = i[2]
            self.ricci_curv[(self.dict_node[i[1]], self.dict_node[i[0]])] = i[2]
            self.graph[self.dict_node[i[0]]][self.dict_node[i[1]]]['weight'] = i[2] + 1
            self.graph[self.dict_node[i[1]]][self.dict_node[i[0]]]['weight'] = i[2] + 1

    def multi_wrapper(self, args):
        return self.sg2pimg(*args)

    def sg2pimg(self, u, v, hop, weight_graph=False, norm=True, extended_flag = False, range='intersection', resolution=5, descriptor = "min"):
        """
        :param u,v: edge (u,v) in the graph self.graph
        :param hop: the number of hops for constructing neighborhood subgraphs
        :param weight_graph: True if self.graph is a weighted graph, False otherwise.
        :param norm: True if normalizing the persistence diagrams within in (0,1), False otherwise
        :param extended_flag: True if computing the extended persistence diagrams, False otherwise
        :param range: 'union' or 'intersection' of two neighborhoods
        :param resolution: The persistence images size is resoltion * resolution
        :return:
        """
        if range == 'union':
            root = u
            nodes = [root] + [x for u, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            root = v
            nodes = nodes + [root] + [x for v, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            subgraph = self.graph.subgraph(nodes)
            fil = filtration(subgraph, u, v, hop, ricci_curv = self.ricci_curv)
            g = fil.build_fv(weight_graph=weight_graph, norm=norm)
            x = graph2dgm(g)

            diagram_zero = x.get_diagram(g, key=descriptor, one_homology_flag=False)
            if extended_flag:
                diagram_one = x.get_diagram(g, key=descriptor, one_homology_flag=True)
            else:
                diagram_one = []
            pers_imager = pimg.PersistenceImager(resolution=resolution)
            pers_img = pers_imager.transform(np.array(diagram_zero + diagram_one))
            return pers_img

        elif range == 'intersection':
            root = u
            nodes_u = [root] + [x for u, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            root = v
            nodes_v = [root] + [x for v, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            nodes = list(set(nodes_u) & set(nodes_v))
            subgraph = self.graph.subgraph(nodes)
            fil = filtration(subgraph, u, v, hop, ricci_curv = self.ricci_curv)
            g = fil.build_fv(weight_graph=weight_graph, norm=norm)
            x = graph2dgm(g)

            if descriptor != 'ricci':
                diagram_zero = x.get_diagram(g, key=descriptor, one_homology_flag=False)
                if extended_flag:
                    diagram_one = x.get_diagram(g, key=descriptor, one_homology_flag=True)
                else:
                    diagram_one = []
                pers_imager = pimg.PersistenceImager(resolution=resolution)
                pers_img = pers_imager.transform(np.array(diagram_zero + diagram_one))
                return pers_img

        elif range == 'removeinter':
            root = u
            nodes_u = [root] + [x for u, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            root = v
            nodes_v = [root] + [x for v, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            nodes_union = nodes_u + nodes_v
            nodes_intersec = set(nodes_u) & set(nodes_v)
            nodes = list(set(nodes_union).difference(nodes_intersec)) + [u, v]
            subgraph = self.graph.subgraph(nodes)
            fil = filtration(subgraph, u, v, hop, ricci_curv = self.ricci_curv)
            g = fil.build_fv(weight_graph=weight_graph, norm=norm)
            x = graph2dgm(g)


            diagram_zero = x.get_diagram(g, key=descriptor, one_homology_flag=False)
            if extended_flag:
                diagram_one = x.get_diagram(g, key=descriptor, one_homology_flag=True)
            else:
                diagram_one = []
            pers_imager = pimg.PersistenceImager(resolution=resolution)
            pers_img = pers_imager.transform(np.array(diagram_zero + diagram_one))
            return pers_img


        else:
            print("Error: 'range' should be 'union' or 'intersection'! ")
            sys.exit()

    def sg2dgm_accelerate(self, u, v, hop, extended_flag = False, descriptor = "seal", resolution = 5, norm = False, cnt = 0):
        root = u
        nodes_u = [root] + [x for u, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
        root = v
        nodes_v = [root] + [x for v, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
        nodes = list(set(nodes_u) & set(nodes_v))
        subgraph = self.graph.subgraph(nodes)

        assert len([cc for cc in nx.connected_components(subgraph)]) == 1
        fil = filtration(subgraph, u, v, hop, ricci_curv = self.ricci_curv)
        g = fil.build_fv(weight_graph=True, norm=norm)
        simplex_filter = perturb_filter_function(g, descriptor = descriptor)
        PD_zero, Pos_edges, Neg_edges = Union_find(simplex_filter)
        if extended_flag:
            PD_one = Accelerate_PD(Pos_edges,Neg_edges,simplex_filter)
        else:
            PD_one = []
        pers_imager = pimg.PersistenceImager(resolution=resolution)
        pers_img = pers_imager.transform(np.array(PD_zero + PD_one))
        return pers_img

    def get_pimg(self, cores, hop, weight_graph=False, norm=True, extended_flag = False, range='intersection', resolution=5):
        """
        :param cores: the number of cpu cores used for the parallel computing
        :param hop: the number of hops for constructing neighborhood subgraphs
        :param weight_graph: True if self.graph is a weighted graph, False otherwise.
        :param norm: True if normalizing the persistence diagrams within in (0,1), False otherwise
        :param extended_flag: True if computing the extended persistence diagrams, False otherwise
        :param range: 'union' or 'intersection' of two neighborhoods
        :param resolution: The persistence images size is resoltion * resolution
        :return:
        """
        params = [(u, v, hop, weight_graph, norm, extended_flag, range, resolution) for u,v in self.graph.edges()]
        pool = ThreadPool(cores)
        pool.map(self.multi_wrapper, params)
        pool.close()
        pool.join()

    def get_pimg_for_one_edge(self, u, v, hop=2, norm=True, extended_flag = False, resolution=5, descriptor = 'min', cnt = 0):
        if cnt % 1000 == 0:
           print("having computed: " + str(cnt) + " edges, really computed: " + str(self.cnt_compute) + " edges, cost " + str(time.time() - self.t1) + "s")

        try:
            self.pi_sg[cnt] = self.sg2dgm_accelerate(self.dict_node[u], self.dict_node[v], hop, norm = True, extended_flag = extended_flag, resolution=resolution, descriptor = descriptor).reshape(-1)
            self.cnt_compute += 1
            return self.pi_sg[cnt]
        except BaseException:
            return np.zeros([resolution*resolution])

    def multi_wrapper_all_edges(self, args):
        return self.get_pimg_for_one_edge(*args)

    def get_pimg_for_all_edges(self, total_edges, cores, hop=2, norm=True, extended_flag = False, resolution=5, descriptor = 'min'):
        self.pi_sg = np.zeros((len(total_edges), resolution * resolution))
        self.cnt_compute = 0
        self.t1 = time.time()
        params = [(edge[0], edge[1], hop, norm, extended_flag, resolution, descriptor, cnt) for cnt, edge in enumerate(total_edges)]
        pool = ThreadPool(cores)
        pool.map(self.multi_wrapper_all_edges, params)
        pool.close()
        pool.join()


