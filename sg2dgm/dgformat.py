import os
import dionysus as d
import numpy as np
from sklearn.preprocessing import normalize


def tuple2dgm(tup):
    return d.Diagram(tup)


def diag2array(diag):
    return np.array(diag)


def array2diag(array):
    res = []
    n = len(array)
    for i in range(n):
        p = [array[i, 0], array[i, 1]]
        res.append(p)
    return res


def dgm2diag(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag


def diag2dgm(diag):
    if type(diag) == list:
        diag = [tuple(i) for i in diag]
    elif type(diag) == np.ndarray:
        diag = [tuple(i) for i in diag]  # just help to tell diag might be an array
    dgm = d.Diagram(diag)
    return dgm


def assert_dgm_above(dgm):
    for p in dgm:
        try:
            assert p.birth <= p.death
        except AssertionError:
            raise Exception('birth is larger than death')


def assert_dgm_below(dgm):
    for p in dgm:
        try:
            assert p.birth >= p.death
        except AssertionError:
            raise Exception('birth is smaller than death')


def flip_dgm(dgm):
    # flip dgm from below to above, not vise versa
    for p in dgm:
        if np.float(p.birth) < np.float(p.death):
            assert_dgm_above(dgm)
            return dgm
        assert np.float(p.birth) >= np.float(p.death)
    data = [(np.float(p.death), np.float(p.birth)) for p in dgm]
    return d.Diagram(data)


def print_dgm(dgm):
    for p in dgm:
        print(p)


def precision_format(nbr, precision=1):
    # assert type(nbr)==float
    return round(nbr * (10 ** precision)) / (10 ** precision)


def normalize_(x, axis=0):
    return normalize(x, axis=axis)


def dgms_summary(dgms, debug='off'):
    n = len(dgms)
    total_pts = [-1] * n
    unique_total_pts = [-1] * n  # no duplicates
    for i in range(len(dgms)):
        total_pts[i] = len(dgms[i])
        unique_total_pts[i] = len(set(list(dgms[i])))
    if debug == 'on':
        print('Total number of points for all dgms')
        print(dgms)
    stat_with_multiplicity = (
        precision_format(np.mean(total_pts), precision=1), precision_format(np.std(total_pts), precision=1),
        np.min(total_pts), np.max(total_pts))
    stat_without_multiplicity = (
        precision_format(np.mean(unique_total_pts)), precision_format(np.std(unique_total_pts)), np.min(unique_total_pts),
        np.max(unique_total_pts))
    print('Dgms with multiplicity    Mean: %s, Std: %s, Min: %s, Max: %s' % (
        precision_format(np.mean(total_pts)), precision_format(np.std(total_pts)), precision_format(np.min(total_pts)),
        precision_format(np.max(total_pts))))
    print('Dgms without multiplicity Mean: %s, Std: %s, Min: %s, Max: %s' % (
        precision_format(np.mean(unique_total_pts)), precision_format(np.std(unique_total_pts)),
        precision_format(np.min(unique_total_pts)), precision_format(np.max(unique_total_pts))))
    return (stat_with_multiplicity, stat_without_multiplicity)