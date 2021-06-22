import os
import numpy as np
import networkx as nx


def perturb_filter_function(g, descriptor = 'seal'):
    simplex_filter = {}
    #e = 1e-4
    ee = 1e-6
    max_filter = 101
    for node in g.nodes():
        temp = {}
        temp['old'] = g.nodes[node][descriptor]
        temp['new'] = g.nodes[node][descriptor]
        simplex_filter[node] = temp
    for edge in g.edges():
        temp = {}
        max_node, min_node = max(simplex_filter[edge[0]]['old'], simplex_filter[edge[1]]['old']), min(
            simplex_filter[edge[0]]['old'], simplex_filter[edge[1]]['old'])
        temp['asc'] = max_node + (min_node + 1) * ee
        temp['desc'] = min_node - (max_filter - max_node) * ee
        simplex_filter[(edge[0],edge[1])] = temp
    return simplex_filter


def Union_find(simplex_filter):
    simplices = []
    min_value = 99999999
    max_value = -99999999
    for simplex in simplex_filter:
        if isinstance(simplex,tuple):
            simplices.append(([simplex[0],simplex[1]],simplex_filter[simplex]['asc']))
        else:
            simplices.append(([simplex], simplex_filter[simplex]['new']))
            if min_value > simplex_filter[simplex]['old']:
                min_value = simplex_filter[simplex]['old']
            if max_value < simplex_filter[simplex]['old']:
                max_value = simplex_filter[simplex]['old']

    op = lambda x: (x[1], len(x[0]))
    simplices.sort(key=op)
    #print(simplices)
    PD_up = []
    PD = []
    dict_component = {}
    for simplex in simplices:
        if len(simplex[0]) == 1:
            dict_component[simplex[0][0]] = simplex[0][0]
        else:
            u, v = simplex[0]
            #find parent
            parent_u, parent_v = u, v
            while  parent_u != dict_component[parent_u]:
                dict_component[parent_u] = dict_component[dict_component[parent_u]]
                parent_u = dict_component[parent_u]
            while  parent_v != dict_component[parent_v]:
                dict_component[parent_v] = dict_component[dict_component[parent_v]]
                parent_v = dict_component[parent_v]

            #union 2 connected components
            if parent_u != parent_v:
                u1, v1 = simplex_filter[parent_u]['new'], simplex_filter[parent_v]['new']
                small = parent_u if u1 <= v1 else parent_v
                large = parent_u + parent_v - small
                max_node = simplex[0][0] if simplex_filter[simplex[0][0]]['new'] > simplex_filter[simplex[0][1]]['new'] else simplex[0][1]
                if simplex_filter[large]['old'] < simplex_filter[max_node]['old']:
                    PD_up+=[[simplex_filter[large]['old'], simplex_filter[max_node]['old']]]
                dict_component[large] = small

    simplices = []
    for simplex in simplex_filter:
        if isinstance(simplex, tuple):
            simplices.append(([simplex[0], simplex[1]], simplex_filter[simplex]['desc']))
        else:
            simplices.append(([simplex], simplex_filter[simplex]['new']))
    op = lambda x: (x[1], -len(x[0]))
    simplices.sort(key=op, reverse=True)
    #print(simplices)
    PD_down = []
    Neg_edges = []
    Pos_edges = []
    dict_component = {}
    for simplex in simplices:
        if len(simplex[0]) == 1:
            dict_component[simplex[0][0]] = simplex[0][0]
        else:
            u, v = simplex[0]
            # find parent
            parent_u, parent_v = u, v
            while parent_u != dict_component[parent_u]:
                dict_component[parent_u] = dict_component[dict_component[parent_u]]
                parent_u = dict_component[parent_u]
            while parent_v != dict_component[parent_v]:
                dict_component[parent_v] = dict_component[dict_component[parent_v]]
                parent_v = dict_component[parent_v]

            #union 2 connected components
            if parent_u != parent_v:
                Neg_edges += [simplex[0]]
                u1, v1 = simplex_filter[parent_u]['new'], simplex_filter[parent_v]['new']
                small = parent_u if u1 <= v1 else parent_v
                large = parent_u + parent_v - small
                min_node = simplex[0][0] if simplex_filter[simplex[0][0]]['new'] < simplex_filter[simplex[0][1]][
                    'new'] else simplex[0][1]
                if simplex_filter[small]['old'] > simplex_filter[min_node]['old']:
                    PD_down += [[simplex_filter[small]['old'], simplex_filter[min_node]['old']]]
                dict_component[small] = large
            else:
                Pos_edges += [simplex[0]]
    PD = PD + PD_up + [[min_value,max_value]] + PD_down + [[max_value,min_value]]
    #print(Pos_edges)
    #print(Neg_edges)
    return PD, Pos_edges, Neg_edges

def Accelerate_PD(Pos_edges, Neg_edges, simplex_filter):

    # find parent
    Parent = {}
    g = nx.Graph()
    g.add_edges_from(Neg_edges)
    Nodes = g.nodes()
    root = list(Nodes)[0]
    for edge in nx.algorithms.bfs_tree(g, root).in_edges:
        Parent[edge[1]] = edge[0]
    Parent[root] = root


    PD_one = []
    #add positive edge
    for pos_edge in Pos_edges:
        if pos_edge[0] != root:
            path_0 = [(pos_edge[0],Parent[pos_edge[0]])]
        else:
            path_0 = []
        if pos_edge[1] != root:
            path_1 = [(pos_edge[1],Parent[pos_edge[1]])]
        else:
            path_1 = []

        #find the loop
        node = pos_edge[0]
        while Parent[node] != root:
            node = Parent[node]
            path_0 += [(node, Parent[node])]
        node = pos_edge[1]
        while Parent[node] != root:
            node = Parent[node]
            path_1 += [(node, Parent[node])]
        path_intersec = list(set(path_0) & set(path_1))
        path_union = path_0 + path_1
        Loop = list(set(path_union).difference(path_intersec))
        #print(Loop)

        #find persistence point
        Loop_value = []
        for i in Loop:
            tmp = simplex_filter[i]['asc'] if i in simplex_filter else simplex_filter[(i[1],i[0])]['asc']
            Loop_value += [tmp]
        large_edge = Loop[np.argmax(Loop_value)]
        large_value = max(simplex_filter[large_edge[0]]['old'],simplex_filter[large_edge[1]]['old'])
        #print(large_value)
        low_value = min(simplex_filter[pos_edge[0]]['old'],simplex_filter[pos_edge[1]]['old'])
        #print(low_value)
        if large_value > low_value:
            PD_one += [[low_value, large_value]]

        #change the parent
        if large_edge in path_0:
            node, nodec = pos_edge[0], pos_edge[1]
        else:
            node, nodec = pos_edge[1], pos_edge[0]
        while nodec != large_edge[0]:
            tmp_parent = Parent[node]
            Parent[node] = nodec
            nodec = node
            node = tmp_parent

    return PD_one








