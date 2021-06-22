#include<iostream>
#include<Python.h>
#include<map>
#include<vector>
#include<algorithm>
#include<queue>
#include<time.h>
using namespace std;

class Node_filter {
public:
    double filter_value;;

};

class Edge_filter {
public:
    double asc_value;
    double desc_value;

};

class c_Simplex {
public:
    int mark_node = 0;
    int node1 = 0;
    int node2 = 0;
    double asc_value = 0;
    double desc_value = 0;
};

class Simplex {
public:
    int mark_node = 0;
    long node = 0;
    long edge[2] = { 0 };
    double asc_value = 0;
    double desc_value = 0;
};
vector<Simplex> simplices;//用于之后的排序使用



//字典用于之后的查询filter value
map <long, Node_filter> Node_filtration;
map <pair<long, long>, Edge_filter> Edge_filtration;

//用于记录整个connected component的PD值
double max_value = -99999999999;
double min_value = 99999999999;

//用于记录Union Find算法的输出
vector <double> PD_first, PD_second;
vector <long*> Pos_edges;
vector <long*> Neg_edges;

//用于记录Accelerate PD的输出
vector <double> PD_one_first, PD_one_second;
map <long, long> Parent;
/*
void Read_Pydict_original(PyObject* pyValue)
{
    PyObject* key_dict = PyDict_Keys(pyValue); //return PyListObject
    Py_ssize_t len = PyDict_Size(pyValue);

    //for循环一个字典中的key和value值
    for (Py_ssize_t i = 0; i < len; ++i)
    {
        PyObject* key = PyList_GetItem(key_dict, i); // 返回元组
        size_t lent = PyTuple_Size(key);
        //一个key中有多个值的时候，把key看成Tuple，for循环一个key中的多个值
        for (size_t k = 0; k < lent; ++k)
        {
            PyObject* item = PyTuple_GetItem(key, k);
            // if(PyInt_Check(item))判断key中值是不是int类型的,返回值为“ture”
            if (PyLong_Check(item))
            {
                //PyInt_AsLong将PyObject中的Int类型转换成c++中的long类型
                long key = PyLong_AsLong(item);
                cout << key << endl;
            }
            // if(PyString_Check(item))判断key中值是不是string类型的,返回值为“ture”
            else if (PyUnicode_Check(item))
            {
                //PyString_AsString将PyObject中的String类型转换成c++中的String类型
                string key = PyUnicode_AsUTF8(item);
                cout << key << endl;
            }
        }
        //打印value的值
        PyObject* value = PyDict_GetItem(pyValue, key); //查询value
        string cval = PyUnicode_AsUTF8(value); //转换结果
        cout << cval << endl;
    }
}
*/

void Read_Pydict(PyObject* pyValue)
{
    PyObject* key_dict = PyDict_Keys(pyValue); //return PyListObject
    Py_ssize_t len = PyDict_Size(pyValue);

    //for循环一个字典中的key和value值
    for (Py_ssize_t i = 0; i < len; ++i)
    {
        // 如果key是node则直接使用
        int mark_node = 0;
        PyObject* key = PyList_GetItem(key_dict, i); // 返回元组
        long node = 0;
        pair<long, long> edge;
        if (PyLong_Check(key)) {
            node = PyLong_AsLong(key);
            mark_node = 1;
            //cout << node << endl;
        }
        //如果key是edge，则使用pair来转化
        else {
            PyObject* item = PyTuple_GetItem(key, 0);
            edge.first = PyLong_AsLong(item);
            PyObject* item_1 = PyTuple_GetItem(key, 1);
            edge.second = PyLong_AsLong(item_1);
            //cout << edge.first << " " << edge.second << endl;
        }

        //由于value是字典，需要再经过一次之前的操作
        PyObject* value = PyDict_GetItem(pyValue, key); //查询value
        PyObject* key_dict_value = PyDict_Keys(value); //return PyListObject
        PyObject* key_value = PyList_GetItem(key_dict_value, 0);
        PyObject* value_former = PyDict_GetItem(value, key_value);
        PyObject* key_value_1 = PyList_GetItem(key_dict_value, 1);
        PyObject* value_later = PyDict_GetItem(value, key_value_1);

        Simplex tmp_simplex;
        tmp_simplex.mark_node = mark_node;
        if (mark_node == 1) {
            Node_filter node_filter;
            node_filter.filter_value = PyFloat_AsDouble(value_former);
            Node_filtration[node] = node_filter;
            tmp_simplex.node = node;
            tmp_simplex.asc_value = node_filter.filter_value;
            tmp_simplex.desc_value = node_filter.filter_value;
            if (min_value > node_filter.filter_value) min_value = node_filter.filter_value;
            if (max_value < node_filter.filter_value) max_value = node_filter.filter_value;
        }
        else {
            Edge_filter edge_filter;
            edge_filter.asc_value = PyFloat_AsDouble(value_former);
            edge_filter.desc_value = PyFloat_AsDouble(value_later);
            Edge_filtration[edge] = edge_filter;
            tmp_simplex.edge[0] = edge.first;
            tmp_simplex.edge[1] = edge.second;
            tmp_simplex.asc_value = edge_filter.asc_value;
            tmp_simplex.desc_value = edge_filter.desc_value;
        }
        simplices.push_back(tmp_simplex);
    }
}


void get_python_dict() {
	Py_Initialize(); //初始Python解释器
    PyObject* pModule = NULL;
    PyObject* first = NULL;
    pModule = PyImport_ImportModule("cal");
    first = PyObject_GetAttrString(pModule, "test_PD");
    PyObject* pyValue = PyObject_CallFunction(first, NULL);
    Read_Pydict(pyValue);
    //Py_ssize_t len = PyTuple_Size(pyValue);
    //for循环py文件中有几个字典
    /*
    for (Py_ssize_t i = 0; i < len; ++i) {
        PyObject* pyDict = PyTuple_GetItem(pyValue, i);
        Read_Pydict(pyDict);
    }
    */
    Py_Finalize();
    //system("pause");
}

void transform_python_dict(c_Simplex *in, int n) {
    /*
    for (int i = 0; i < n; ++i) {
        
        cout << "Sample" << i << ":" << endl;
        cout << in[i].mark_node << endl;
        if (in[i].mark_node > 0) {
            cout << in[i].node1 << endl;
        }
        else {
            cout << in[i].node1 << " " << in[i].node2 << endl;
        }
        cout << in[i].asc_value << endl;
        cout << in[i].desc_value << endl;
        
        //simplices.push_back(in[i]);
    }*/
    
    for (int i = 0; i < n; ++i) {
        Simplex simplex;
        simplex.mark_node = in[i].mark_node;
        if (simplex.mark_node == 1) { 
            simplex.node = in[i].node1; 
            Node_filter tmp_node_filter; 
            tmp_node_filter.filter_value = in[i].asc_value; 
            Node_filtration[simplex.node] = tmp_node_filter; 
            if (min_value > tmp_node_filter.filter_value) min_value = tmp_node_filter.filter_value;
            if (max_value < tmp_node_filter.filter_value) max_value = tmp_node_filter.filter_value;
        }
        else { 
            simplex.edge[0] = in[i].node1; 
            simplex.edge[1] = in[i].node2; 
            Edge_filter tmp_edge_filter;
            tmp_edge_filter.asc_value = in[i].asc_value;
            tmp_edge_filter.desc_value = in[i].desc_value;
            pair <long, long> tmp_edge;
            tmp_edge.first = in[i].node1; tmp_edge.second = in[i].node2;
            Edge_filtration[tmp_edge] = tmp_edge_filter;
        }
        simplex.asc_value = in[i].asc_value;
        simplex.desc_value = in[i].desc_value;
        simplices.push_back(simplex);
    }
    /*
    for (int i = 0; i < n; ++i) {
        cout << simplices[i].mark_node << " " << simplices[i].node << " " << simplices[i].edge[0] << " " << simplices[i].edge[1] << " " << simplices[i].asc_value << " " << simplices[i].desc_value << endl;
    }*/
}

//used to sort simplices
bool compare_asc(Simplex sa, Simplex sb) {
    return sa.asc_value < sb.asc_value;
}

bool compare_desc(Simplex sa, Simplex sb) {
    return sa.desc_value > sb.desc_value;
}

void Union_find() {
    clock_t t1;
    vector <Simplex> asc_simplices(simplices);
    vector <Simplex> desc_simplices(simplices);
    sort(asc_simplices.begin(), asc_simplices.end(), compare_asc);
    sort(desc_simplices.begin(), desc_simplices.end(), compare_desc);

    /*
    for (int i = 0; i < simplices.size(); ++i) {
        cout << asc_simplices[i].mark_node << " " << asc_simplices[i].node << " " << asc_simplices[i].edge[0] << " " << asc_simplices[i].edge[1] << " " << asc_simplices[i].asc_value << " " << asc_simplices[i].desc_value << endl;
    }
    
    for (int i = 0; i < simplices.size(); ++i) {
        cout << desc_simplices[i].mark_node << " " << desc_simplices[i].node << " " << desc_simplices[i].edge[0] << " " << desc_simplices[i].edge[1] << " " << desc_simplices[i].asc_value << " " << desc_simplices[i].desc_value << endl;
    }
    */
    map <long, long > dict_component_asc;
    map <long, long> dict_component_desc;

    //ascending filtration
    for (int i = 0; i < simplices.size(); ++i) {
        Simplex tmp_simplex = asc_simplices[i];
        if (tmp_simplex.mark_node == 1) {
            dict_component_asc[tmp_simplex.node] = tmp_simplex.node;
        }
        else {
            long node_1 = tmp_simplex.edge[0]; long node_2 = tmp_simplex.edge[1];
            //find parent
            long parent_1 = node_1, parent_2 = node_2;
            while (parent_1 != dict_component_asc[parent_1]) {
                dict_component_asc[parent_1] = dict_component_asc[dict_component_asc[parent_1]];
                parent_1 = dict_component_asc[parent_1];
            }
            while (parent_2 != dict_component_asc[parent_2]) {
                dict_component_asc[parent_2] = dict_component_asc[dict_component_asc[parent_2]];
                parent_2 = dict_component_asc[parent_2];
            }

            //union 2 connected components
            if (parent_1 != parent_2) {
                double value_1 = Node_filtration[parent_1].filter_value;
                double value_2 = Node_filtration[parent_2].filter_value;
                long small = parent_1, large = parent_2;
                if (value_1 > value_2) {
                    small = parent_2; large = parent_1;
                }
                long max_node = node_2;
                if (Node_filtration[node_1].filter_value > Node_filtration[node_2].filter_value) max_node = node_1;
                if (Node_filtration[large].filter_value < Node_filtration[max_node].filter_value) {
                    PD_first.push_back(Node_filtration[large].filter_value);
                    PD_second.push_back(Node_filtration[max_node].filter_value);
                }
                dict_component_asc[large] = small;
            }
        }
    }

    PD_first.push_back(min_value);
    PD_second.push_back(max_value);


    //descending filtration
    for (int i = 0; i < simplices.size(); ++i) {
        Simplex tmp_simplex = desc_simplices[i];
        if (tmp_simplex.mark_node == 1) {
            dict_component_desc[tmp_simplex.node] = tmp_simplex.node;
        }
        else {
            long node_1 = tmp_simplex.edge[0]; long node_2 = tmp_simplex.edge[1];
            //find parent
            long parent_1 = node_1, parent_2 = node_2;
            while (parent_1 != dict_component_desc[parent_1]) {
                dict_component_desc[parent_1] = dict_component_desc[dict_component_desc[parent_1]];
                parent_1 = dict_component_desc[parent_1];
            }
            while (parent_2 != dict_component_desc[parent_2]) {
                dict_component_desc[parent_2] = dict_component_desc[dict_component_desc[parent_2]];
                parent_2 = dict_component_desc[parent_2];
            }

            //union 2 connected components
            if (parent_1 != parent_2) {
                long* tmp_edge = new long[2];
                tmp_edge[0] = node_1; tmp_edge[1] = node_2; Neg_edges.push_back(tmp_edge);

                double value_1 = Node_filtration[parent_1].filter_value;
                double value_2 = Node_filtration[parent_2].filter_value;
                long small = parent_1, large = parent_2;
                if (value_1 > value_2) {
                    small = parent_2; large = parent_1;
                }
                long min_node = node_2;
                if (Node_filtration[node_1].filter_value < Node_filtration[node_2].filter_value) min_node = node_1;
                if (Node_filtration[small].filter_value > Node_filtration[min_node].filter_value) {
                    PD_first.push_back(Node_filtration[small].filter_value);
                    PD_second.push_back(Node_filtration[min_node].filter_value);
                }
                dict_component_desc[small] = large;

            }
            else {
                long* tmp_edge = new long[2];
                tmp_edge[0] = node_1; tmp_edge[1] = node_2; Pos_edges.push_back(tmp_edge);
            }
        }
    }
    PD_first.push_back(max_value);
    PD_second.push_back(min_value);
    /*
    cout << "Neg Edges:" << endl;
    for (int ii = 0; ii < Neg_edges.size(); ++ii)
        cout << Neg_edges[ii][0] << "," << Neg_edges[ii][1] << endl;
    cout << "Pos Edges:" << endl;
    for (int ii = 0; ii < Pos_edges.size(); ++ii)
        cout << Pos_edges[ii][0] << "," << Pos_edges[ii][1] << endl;
    */
    /*
    for (int ii = 0; ii < PD.size(); ++ii)
        cout << PD[ii][0] << "," << PD[ii][1] << endl;
    */
}

/*
void Union_find_original() {
    clock_t t1;
    vector <Simplex> asc_simplices(simplices);
    vector <Simplex> desc_simplices(simplices);
    sort(asc_simplices.begin(), asc_simplices.end(), compare_asc);
    sort(desc_simplices.begin(), desc_simplices.end(), compare_desc);
    map <long, vector<long>> dict_component_asc;
    map <long, vector<long>> dict_component_desc;

    //ascending filtration
    for (int i = 0; i < simplices.size(); ++i) {
        Simplex tmp_simplex = asc_simplices[i];
        if (tmp_simplex.mark_node == 1) {
            vector <long> tmp_vector;
            tmp_vector.push_back(tmp_simplex.node);
            dict_component_asc[tmp_simplex.node] = tmp_vector;
        }
        else {
            long node_1 = tmp_simplex.edge[0]; long node_2 = tmp_simplex.edge[1];
            if (dict_component_asc[node_1][0] != dict_component_asc[node_2][0]) {
                double value_1 = Node_filtration[dict_component_asc[node_1][0]].filter_value;
                double value_2 = Node_filtration[dict_component_asc[node_2][0]].filter_value;
                long small = node_1, large = node_2;
                if (value_1 > value_2) {
                    small = node_2; large = node_1;
                }
                dict_component_asc[small].insert(dict_component_asc[small].end(), dict_component_asc[large].begin(), dict_component_asc[large].end());
                long max_node = node_2;
                if (Node_filtration[node_1].filter_value > Node_filtration[node_2].filter_value) max_node = node_1;
                if (Node_filtration[dict_component_asc[large][0]].filter_value < Node_filtration[max_node].filter_value) {
                    double* tmp_PD = new double[2];
                    tmp_PD[0] = Node_filtration[dict_component_asc[large][0]].filter_value;
                    tmp_PD[1] = Node_filtration[max_node].filter_value;
                    PD.push_back(tmp_PD);
                }
                for (int iterator_i = 0; iterator_i < dict_component_asc[small].size(); ++iterator_i) {
                    dict_component_asc[dict_component_asc[small][iterator_i]] = dict_component_asc[small];
                }
            }
        }
    }
    double* tmp_PD = new double[2];
    tmp_PD[0] = min_value;
    tmp_PD[1] = max_value;
    PD.push_back(tmp_PD);


    //descending filtration
    for (int i = 0; i < simplices.size(); ++i) {
        Simplex tmp_simplex = desc_simplices[i];
        if (tmp_simplex.mark_node == 1) {
            vector <long> tmp_vector;
            tmp_vector.push_back(tmp_simplex.node);
            dict_component_desc[tmp_simplex.node] = tmp_vector;
        }
        else {
            long node_1 = tmp_simplex.edge[0]; long node_2 = tmp_simplex.edge[1];
            if (dict_component_desc[node_1][0] != dict_component_desc[node_2][0]) {
                long* tmp_edge = new long[2];
                tmp_edge[0] = node_1; tmp_edge[1] = node_2; Neg_edges.push_back(tmp_edge);

                double value_1 = Node_filtration[dict_component_desc[node_1][0]].filter_value;
                double value_2 = Node_filtration[dict_component_desc[node_2][0]].filter_value;
                long small = node_1, large = node_2;
                if (value_1 > value_2) {
                    small = node_2; large = node_1;
                }
                dict_component_desc[large].insert(dict_component_desc[large].end(), dict_component_desc[small].begin(), dict_component_desc[small].end());
                long min_node = node_2;
                if (Node_filtration[node_1].filter_value < Node_filtration[node_2].filter_value) min_node = node_1;
                if (Node_filtration[dict_component_desc[small][0]].filter_value > Node_filtration[min_node].filter_value) {
                    double* tmp_PD = new double[2];
                    tmp_PD[0] = Node_filtration[dict_component_desc[small][0]].filter_value;
                    tmp_PD[1] = Node_filtration[min_node].filter_value;
                    PD.push_back(tmp_PD);
                }
                for (int iterator_i = 0; iterator_i < dict_component_desc[large].size(); ++iterator_i) {
                    dict_component_desc[dict_component_desc[large][iterator_i]] = dict_component_desc[large];
                }

            }
            else {
                long* tmp_edge = new long[2];
                tmp_edge[0] = node_1; tmp_edge[1] = node_2; Pos_edges.push_back(tmp_edge);
            }
        }
    }
    double* tmp_PD_ = new double[2];
    tmp_PD_[0] = max_value;
    tmp_PD_[1] = min_value;
    PD.push_back(tmp_PD_);
}
*/

const int sz = 1e5;
// Adjacency list representation 
// of the tree 
//vector <int> tree[sz + 1];
// Boolean array to mark all the 
// vertices which are visited 
bool vis[sz + 1];

// Function to create an 
// edge between two vertices 
/*
void addEdge(int a, int b)
{

    // Add a to b's list 
    tree[a].push_back(b);

    // Add b to a's list 
    tree[b].push_back(a);
}*/


// Modified Breadth-First Function 
/*
void bfs(int node)
{

    // Create a queue of {child, parent} 
    queue<pair<int, int> > qu;

    // Push root node in the front of 
    qu.push({ node, 0 });

    while (!qu.empty()) {
        pair<int, int> p = qu.front();

        // Dequeue a vertex from queue 
        qu.pop();
        Parent[p.first] = p.second;
        vis[p.first] = true;

        // Get all adjacent vertices of the dequeued 
        // vertex s. If any adjacent has not 
        // been visited then enqueue it 
        for (int child : tree[p.first]) {
            if (!vis[child]) {
                qu.push({ child, p.first });
            }
        }
    }

}*/

void Accelerate_PD() {
    //find parent
    // Adjacency list representation  of the tree 
    clock_t t1 = clock();
    vector <int> tree[sz + 1];
    for (int i = 0; i < Neg_edges.size(); ++i)
    {
        tree[Neg_edges[i][0]].push_back(Neg_edges[i][1]);
        tree[Neg_edges[i][1]].push_back(Neg_edges[i][0]);
    }
    long root = Neg_edges[0][0];

    // Create a queue of {child, parent} 
    queue<pair<int, int> > qu;


    // Push root node in the front of 
    qu.push({ root, 0 });

    //bfs to find parent
    while (!qu.empty()) {
        pair<int, int> p = qu.front();

        // Dequeue a vertex from queue 
        qu.pop();
        Parent[p.first] = p.second;
        vis[p.first] = true;

        // Get all adjacent vertices of the dequeued 
        // vertex s. If any adjacent has not 
        // been visited then enqueue it 
        for (int child : tree[p.first]) {
            if (!vis[child]) {
                qu.push({ child, p.first });
            }
        }
    }
    Parent[root] = root;
    //cout << root << endl;
    /*
    map<long, long>::iterator iter_parent;
    iter_parent = Parent.begin();
    while(iter_parent != Parent.end()) {
        cout << iter_parent->first << " : " << iter_parent->second << endl;
        iter_parent++;
    }
    */
    t1 = clock();

    //cout << "total positive edges number is" << Pos_edges.size() << endl;
    //add positive edges one by one


    for (int i = 0; i < Pos_edges.size(); ++i) {

        t1 = clock();
        vector<long> path_0, path_1;
        long* pos_edge = Pos_edges[i];
        path_0.push_back(pos_edge[0]);
        path_1.push_back(pos_edge[1]);

        //cout << "path push finished" << endl;
        //cout << "root node:" << root << endl;
        //find the loop
        long node = pos_edge[0];
        //cout << node << endl;
        while (node != root) {
            node = Parent[node];
            //cout << node << endl;
            path_0.push_back(node);
        }
        //cout << "path 0 find finished" << endl;
        node = pos_edge[1];
        //cout << node << endl;
        while (node != root) {
            node = Parent[node];
            path_1.push_back(node);
        }


        //cout << "find path 1 and path 0 cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();
        /*
        cout << "path 0:" << endl;
        for (int kkk = 0; kkk < path_0.size(); ++kkk) 
            cout << path_0[kkk] << " ";
        cout << endl;
        cout << "path 1:" << endl;
        for (int kkk = 0; kkk < path_1.size(); ++kkk)
            cout << path_1[kkk] << " ";
        cout << endl;
        */
        vector <long> path_loop;
        //vector <long> path_loop1;
        int j, k;
        if (path_0.size() < path_1.size()) {
            j = 0; k = path_1.size() - path_0.size();
        }
        else {
            k = 0; j = path_0.size() - path_1.size();
        }
        while (j < path_0.size() && k < path_1.size()) {
            if (path_0[j] == path_1[k]) break;
            j++; k++;
        }
        for (int j1 = 0; j1 <= j; ++j1) { path_loop.push_back(path_0[j1]); }
        for (int j1 = k - 1; j1 >= 0; j1--) { path_loop.push_back(path_1[j1]); }


        //cout << "find loop cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();

        //find persistence point
        int max_j = 0; double max_value = -999999999;
        for (j = 0; j < path_loop.size(); ++j) {
            if (max_value <= Node_filtration[path_loop[j]].filter_value) { max_value = Node_filtration[path_loop[j]].filter_value; max_j = j; }
        }
        long* large_edge = new long[2];
        if (max_j >= 1) {
            large_edge[0] = path_loop[max_j - 1]; large_edge[1] = path_loop[max_j];
        }
        else {
            large_edge[0] = path_loop[max_j]; large_edge[1] = path_loop[max_j + 1];
        }
        double large_value = Node_filtration[path_loop[max_j]].filter_value;
        double low_value = min(Node_filtration[pos_edge[0]].filter_value, Node_filtration[pos_edge[1]].filter_value);
        if (large_value > low_value) {
            PD_one_first.push_back(low_value);
            PD_one_second.push_back(large_value);
        }


        //cout << "find persistence point cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();

        //change the parent
        k = 0;
        int mark_k = 0;
        for (k = 0; k < path_0.size() - 1; ++k) {
            if (large_edge[0] == path_0[k] && large_edge[1] == path_0[k + 1]) { mark_k = 1; break; }
        }
        long nodec, nodecstop;
        node = pos_edge[1]; nodec = pos_edge[0]; nodecstop = large_edge[1];
        if (mark_k == 1) {
            node = pos_edge[0]; nodec = pos_edge[1];  nodecstop = large_edge[0];
        }
        while (nodec != nodecstop) {
            long tmp_parent = Parent[node];
            Parent[node] = nodec;
            nodec = node;
            node = tmp_parent;
        }

        //cout << "change parent cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();
        //cout << i << endl;
    }
    //cout << "total add positive edge cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
}



/*
//the new accelerate PD who uses finds the loop from the target nodes
void Accelerate_PD_path_as_pair_vector() {
    //find parent
    // Adjacency list representation  of the tree 
    clock_t t1 = clock();
    vector <int> tree[sz + 1];
    for (int i = 0; i < Neg_edges.size(); ++i)
    {
        tree[Neg_edges[i][0]].push_back(Neg_edges[i][1]);
        tree[Neg_edges[i][1]].push_back(Neg_edges[i][0]);
    }
    long root = Neg_edges[0][0];

    //cout << "add edge to tree cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
    //t1 = clock();
    // Create a queue of {child, parent} 
    queue<pair<int, int> > qu;


    // Push root node in the front of 
    qu.push({ root, 0 });

    //bfs to find parent
    while (!qu.empty()) {
        pair<int, int> p = qu.front();

        // Dequeue a vertex from queue 
        qu.pop();
        Parent[p.first] = p.second;
        vis[p.first] = true;

        // Get all adjacent vertices of the dequeued 
        // vertex s. If any adjacent has not 
        // been visited then enqueue it 
        for (int child : tree[p.first]) {
            if (!vis[child]) {
                qu.push({ child, p.first });
            }
        }
    }
    Parent[root] = root;

    //cout << "bfs cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
    t1 = clock();

    //cout << "total positive edges number is" << Pos_edges.size() << endl;
    //add positive edges one by one


    for (int i = 0; i < Pos_edges.size(); ++i) {
        t1 = clock();
        vector<long*> path_0, path_1;
        long* pos_edge = Pos_edges[i];
        if (pos_edge[0] != root) {
            long* tmp_path = new long[2];
            tmp_path[0] = pos_edge[0]; tmp_path[1] = Parent[pos_edge[0]]; 
            path_0.push_back(tmp_path);
        }
        if (pos_edge[1] != root) {
            long* tmp_path = new long[2];
            tmp_path[0] = pos_edge[1]; tmp_path[1] = Parent[pos_edge[1]]; 
            path_1.push_back(tmp_path);
        }

        //find the loop
        long node = pos_edge[0];
        while (Parent[node] != root) {
            node = Parent[node];
            long* tmp_path = new long[2];
            tmp_path[0] = node; tmp_path[1] = Parent[node]; 
            path_0.push_back(tmp_path);
        }
        node = pos_edge[1];
        while (Parent[node] != root) {
            node = Parent[node];
            long* tmp_path = new long[2];
            tmp_path[0] = node; tmp_path[1] = Parent[node]; 
            path_1.push_back(tmp_path);
        }

        cout << "find path 1 and path 0 cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();

        vector <long*> path_loop;
        vector <long*> path_loop1;
        int j, k;
        if (path_0.size() < path_1.size()) { 
            j = 0; k = path_1.size() - path_0.size(); 
            for (int k1 = 0; k1 < k; ++k1)
                path_loop.push_back(path_1[k1]);
        }
        else { 
            k = 0; j = path_0.size() - path_1.size(); 
            for (int j1 = 0; j1 < j; ++j1)
                path_loop.push_back(path_0[j1]);
        }
        while (j < path_0.size() && k < path_1.size()) {
            if (path_0[j][0] == path_1[k][0] && path_0[j][1] == path_1[k][1]) break;
            else { path_loop.push_back(path_0[j]); path_loop.push_back(path_1[k]); }
            j++; k++;
        }
        
        cout << "find loop cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        //t1 = clock();

        //find persistence point
        int max_j = 0; double max_value = -999999999;
        pair <long, long> tmp_edge;
        for (int j = 0; j < path_loop.size(); ++j) {
            tmp_edge.first = path_loop[j][0]; tmp_edge.second = path_loop[j][1];
            if (Edge_filtration.count(tmp_edge) == 0) { tmp_edge.first = path_loop[j][1]; tmp_edge.second = path_loop[j][0]; }
            if (max_value <= Edge_filtration[tmp_edge].asc_value) { max_value = Edge_filtration[tmp_edge].asc_value; max_j = j; }
        }
        long* large_edge = path_loop[max_j];
        double large_value = max(Node_filtration[large_edge[0]].filter_value, Node_filtration[large_edge[1]].filter_value);
        double low_value = min(Node_filtration[pos_edge[0]].filter_value, Node_filtration[pos_edge[1]].filter_value);
        if (large_value > low_value) {
            double* tmp_value = new double[2];
            tmp_value[0] = low_value; tmp_value[1] = large_value; PD_one.push_back(tmp_value);
        }

        //cout << "find persistence point cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        //t1 = clock();

        //change the parent
        k = 0;
        for (k = 0; k < path_1.size(); ++k) {
            if (large_edge[0] == path_1[k][0] && large_edge[1] == path_1[k][1]) break;
        }
        long nodec;
        node = pos_edge[0]; nodec = pos_edge[1];
        if (k != path_1.size()) {
            node = pos_edge[1]; nodec = pos_edge[0];
        }
        while (nodec != large_edge[0]) {
            long tmp_parent = Parent[node];
            Parent[node] = nodec;
            nodec = node;
            node = tmp_parent;
        }

        //cout << "change parent cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        t1 = clock();
        //cout << i << endl;
    }
    //cout << "total add positive edge cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
}


//the original accelerated PD, who finds the loop by counting from root to target nodes
void Accelerate_PD_original() {
    //find parent
    // Adjacency list representation  of the tree 
    vector <int> tree[sz + 1];
    for (int i = 0; i < Neg_edges.size(); ++i)
    {
        tree[Neg_edges[i][0]].push_back(Neg_edges[i][1]);
        tree[Neg_edges[i][1]].push_back(Neg_edges[i][0]);
    }
    long root = Neg_edges[0][0];
    // Create a queue of {child, parent} 
    queue<pair<int, int> > qu;

    // Push root node in the front of 
    qu.push({ root, 0 });

    //bfs to find parent
    while (!qu.empty()) {
        pair<int, int> p = qu.front();

        // Dequeue a vertex from queue 
        qu.pop();
        Parent[p.first] = p.second;
        vis[p.first] = true;

        // Get all adjacent vertices of the dequeued 
        // vertex s. If any adjacent has not 
        // been visited then enqueue it 
        for (int child : tree[p.first]) {
            if (!vis[child]) {
                qu.push({ child, p.first });
            }
        }
    }
    Parent[root] = root;
    cout << "mark bfs finished" << endl;

    //add positive edges one by one
    for (int i = 0; i < Pos_edges.size(); ++i) {
        vector<long*> path_0, path_1;
        long* pos_edge = Pos_edges[i];
        if (pos_edge[0] != root) {
            long* tmp_path = new long[2];
            tmp_path[0] = pos_edge[0]; tmp_path[1] = Parent[pos_edge[0]]; path_0.push_back(tmp_path);
        }
        if (pos_edge[1] != root) {
            long* tmp_path = new long[2];
            tmp_path[0] = pos_edge[1]; tmp_path[1] = Parent[pos_edge[1]]; path_1.push_back(tmp_path);
        }

        //find the loop
        long node = pos_edge[0];
        while (Parent[node] != root) {
            node = Parent[node];
            long* tmp_path = new long[2];
            tmp_path[0] = node; tmp_path[1] = Parent[node]; path_0.push_back(tmp_path);
        }
        node = pos_edge[1];
        while (Parent[node] != root) {
            node = Parent[node];
            long* tmp_path = new long[2];
            tmp_path[0] = node; tmp_path[1] = Parent[node]; path_1.push_back(tmp_path);
        }
        vector <long*> path_loop;

        int j, k;
        for (j = path_0.size() - 1, k = path_1.size() - 1; j >= 0 && k >= 0; j--, k--) {
            if (path_0[j][0] == path_1[k][0] && path_0[j][1] == path_1[k][1]) continue;
            else break;
        }
        for (int j1 = 0; j1 <= j; ++j1) 
            path_loop.push_back(path_0[j1]);
        for (int k1 = 0; k1 <= k; ++k1)
            path_loop.push_back(path_1[k1]);

        cout << "path 1:" << endl;
        for (int j1 = 0; j1 < path_1.size(); j1++)
            cout << path_1[j1][0] << " " << path_1[j1][1] << endl;
        cout << "path 0:" << endl;
        for (int j1 = 0; j1 < path_0.size(); j1++)
            cout << path_0[j1][0] << " " << path_0[j1][1] << endl;
        cout << "path loop:" << endl;
        for (int j1 = 0; j1 < path_loop.size(); j1++)
            cout << path_loop[j1][0] << " " << path_loop[j1][1] << endl;

        cout << "mark find loop finished" << endl;

        //find persistence point
        int max_j = 0; double max_value = -999999999;
        pair <long, long> tmp_edge;
        for (int j = 0; j < path_loop.size(); ++j) {
            tmp_edge.first = path_loop[j][0]; tmp_edge.second = path_loop[j][1];
            if (Edge_filtration.count(tmp_edge) == 0) { tmp_edge.first = path_loop[j][1]; tmp_edge.second = path_loop[j][0]; }
            if (max_value <= Edge_filtration[tmp_edge].asc_value) { max_value = Edge_filtration[tmp_edge].asc_value; max_j = j; }
        }
        long* large_edge = path_loop[max_j];
        double large_value = max(Node_filtration[large_edge[0]].filter_value, Node_filtration[large_edge[1]].filter_value);
        double low_value = min(Node_filtration[pos_edge[0]].filter_value, Node_filtration[pos_edge[1]].filter_value);
        if (large_value > low_value) {
            double* tmp_value = new double[2];
            tmp_value[0] = low_value; tmp_value[1] = large_value; PD_one.push_back(tmp_value);
        }

        cout << "mark find persistence point finished" << endl;

        //change the parent
        k = 0;
        for (k = 0; k < path_1.size(); ++k) {
            if (large_edge[0] == path_1[k][0] && large_edge[1] == path_1[k][1]) break;
        }
        long nodec;
        node = pos_edge[0]; nodec = pos_edge[1];
        if (k != path_1.size()) {
            node = pos_edge[1]; nodec = pos_edge[0];
        }
        while (nodec != large_edge[0]) {
            long tmp_parent = Parent[node];
            Parent[node] = nodec;
            nodec = node;
            node = tmp_parent;
        }
        cout << "mark change parent finished" << endl;
        //cout << i << endl;
    }

}
*/

void compute_extended_persistence_diagram(c_Simplex *In, int n, bool extended_flag = true) {
//void compute_extended_persistence_diagram(bool extended_flag = true) {
    clock_t t2, t3;
    //t2 = clock();
    //get_python_dict();
    transform_python_dict(In, n);
    //cout << "transform python dict cost " << (double)(clock() - t2) / CLOCKS_PER_SEC << endl;
    //t2 = clock();
    Union_find();
    //cout << "union find cost " << (double)(clock() - t2) / CLOCKS_PER_SEC << endl;
    //t3 = clock();
    if (extended_flag)
        Accelerate_PD();
        //Accelerate_PD_original();
        //cout << "accelerate algorithm cost " << (double)(clock() - t3) / CLOCKS_PER_SEC << endl;
    /*
    Output Final_output;
    Final_output.PD_zero = PD;
    Final_output.PD_one = PD_one;
    return Final_output;
    */
}

//需要将函数包裹起来用于给python使用
extern "C" {


    PyObject* c_compute_extended_persistence_diagram(c_Simplex* In, int n, bool extended_flag = true) {
        clock_t t1 = clock();

        simplices.clear();
        Node_filtration.clear();
        Edge_filtration.clear();
        max_value = -99999999999;
        min_value = 99999999999;
        PD_first.clear(); PD_second.clear();  Pos_edges.clear(); Neg_edges.clear(); PD_one_first.clear(); PD_one_second.clear();  Parent.clear();
        memset(vis, 0, sizeof(vis));
        //for (int i = 0; i <= sz; ++i) { tree[i].clear(); } 
        PyObject* result = PyList_New(0);
        //cout << "initialize cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        compute_extended_persistence_diagram(In, n, extended_flag);
        //t1 = clock();
        for (int i = 0; i < PD_first.size(); ++i) {
            PyList_Append(result, PyFloat_FromDouble(PD_first[i]));
            PyList_Append(result, PyFloat_FromDouble(PD_second[i]));
        }
        for (int i = 0; i < PD_one_first.size(); ++i) {
            PyList_Append(result, PyFloat_FromDouble(PD_one_first[i]));
            PyList_Append(result, PyFloat_FromDouble(PD_one_second[i]));
        }
        //cout << "pylist append dict cost " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        
        return result;
    }

}


int main()
{
    //compute_extended_persistence_diagram();
	return 0;
}