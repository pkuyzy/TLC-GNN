from ctypes import *
import numpy as np

def test_PD():
    #simplex_filter = {2118: {'old': 3, 'new': 3}, 2119: {'old': 1, 'new': 1}, 2120: {'old': 3, 'new': 3}, 2288: {'old': 2, 'new': 2}, 2289: {'old': 3, 'new': 3}, 2290: {'old': 1, 'new': 1}, 855: {'old': 4, 'new': 4}, (2118, 855): {'asc': 4.000000000003, 'desc': 2.999999999903}, (2118, 2119): {'asc': 3.000000000001, 'desc': 0.999999999902}, (2118, 2120): {'asc': 3.000000000003, 'desc': 2.999999999902}, (2119, 2288): {'asc': 2.000000000001, 'desc': 0.999999999901}, (2119, 2289): {'asc': 3.000000000001, 'desc': 0.999999999902}, (2119, 2290): {'asc': 1.000000000001, 'desc': 0.9999999999}, (2120, 2289): {'asc': 3.000000000003, 'desc': 2.999999999902}, (2120, 2290): {'asc': 3.000000000001, 'desc': 0.999999999902}, (2288, 855): {'asc': 4.000000000002, 'desc': 1.999999999903}, (2288, 2289): {'asc': 3.000000000002, 'desc': 1.999999999902}, (2288, 2290): {'asc': 2.000000000001, 'desc': 0.999999999901}}
    simplex_filter = {2118: {'old': 3, 'new': 3}, 2119: {'old': 1, 'new': 1}, 2120: {'old': 3, 'new': 3},
                      2288: {'old': 2, 'new': 2}, 2289: {'old': 3, 'new': 3}, 2290: {'old': 1, 'new': 1},
                      855: {'old': 4, 'new': 4}, (2118, 855): {'asc': 4.0003, 'desc': 2.9903},
                      (2118, 2119): {'asc': 3.0001, 'desc': 0.9902},
                      (2118, 2120): {'asc': 3.0003, 'desc': 2.9902},
                      (2119, 2288): {'asc': 2.0001, 'desc': 0.9901},
                      (2119, 2289): {'asc': 3.0001, 'desc': 0.9902},
                      (2119, 2290): {'asc': 1.0001, 'desc': 0.9999},
                      (2120, 2289): {'asc': 3.0003, 'desc': 2.9902},
                      (2120, 2290): {'asc': 3.0001, 'desc': 0.9902},
                      (2288, 855): {'asc': 4.0002, 'desc': 1.9903},
                      (2288, 2289): {'asc': 3.0002, 'desc': 1.9902},
                      (2288, 2290): {'asc': 2.0001, 'desc': 0.9901}}
    return simplex_filter

class Simplex_Dict(Structure):
    _fields_ = [('mark_Node', c_int),
                ('node1', c_int),
                ('node2', c_int),
                ('asc_value', c_double),
                ('desc_value', c_double)]



#repackage simplex_filter as a list of Simplex_Dict strictures

def c_compute_extended_persistence_diagram(simplex_filter, extended_flag = True):
    #print(simplex_filter)
    Simplex_list = []
    for key in simplex_filter:
        simplex_dict = Simplex_Dict()
        if isinstance(key, int):
            simplex_dict.mark_Node = 1
            simplex_dict.node1 = key
            simplex_dict.node2 = 0
            simplex_dict.asc_value = simplex_filter[key]['old'] if simplex_filter[key]['old'] > 1e-4 else 1e-4
            simplex_dict.desc_value = simplex_filter[key]['new'] if simplex_filter[key]['new'] > 1e-4 else 1e-4
        else:
            simplex_dict.mark_Node = 0
            simplex_dict.node1 = key[0]
            simplex_dict.node2 = key[1]
            simplex_dict.asc_value = simplex_filter[key]['asc'] if simplex_filter[key]['asc'] > 1e-5 else 1e-5
            simplex_dict.desc_value = simplex_filter[key]['desc'] if simplex_filter[key]['desc'] > 1e-5 else 1e-5
        Simplex_list.append(simplex_dict)

    # repackage python list of Smiplex_Dict to ctype array
    simplex_array = (Simplex_Dict * len(Simplex_list))(*Simplex_list)

    so = cdll.LoadLibrary
    lib = so("./sg2dgm/extended.so")

    my_compute = lib.c_compute_extended_persistence_diagram
    my_compute.argtypes = POINTER(Simplex_Dict), c_int, c_bool
    my_compute.restype = py_object

    x = my_compute(simplex_array, len(simplex_array), extended_flag)
    PD_zero = []
    PD_one = []
    mark_one = 0
    for i in range(int(len(x) / 2)):
        if mark_one == 0:
            PD_zero.append([x[2*i], x[2*i+1]])
            if 2*i+3 < len(x) and x[2*i] > x[2*i+1] and x[2*i+2] < x[2*i+3]:
                mark_one = 1
        else:
            PD_one.append([x[2*i], x[2*i+1]])
    #print(PD_zero)
    #print(PD_one)
    return PD_zero, PD_one

if __name__ == "__main__":
    simplex_filter = test_PD()
    print(c_compute_extended_persistence_diagram(simplex_filter, True))

