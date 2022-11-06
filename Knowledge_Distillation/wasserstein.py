# This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
# See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
# Author(s):       Theo Lacombe
#
# Copyright (C) 2019 Inria
#
# Modification(s):
#   - YYYY/MM Author: Description of the modification
# code of gudhi.wasserstein

import numpy as np
import scipy.spatial.distance as sc

try:
    import ot
except ImportError:
    print("POT (Python Optimal Transport) package is not installed. Try to run $ conda install -c conda-forge pot ; or $ pip install POT")


# Currently unused, but Théo says it is likely to be used again.
def _proj_on_diag(X):
    '''
    :param X: (n x 2) array encoding the points of a persistent diagram.
    :returns: (n x 2) array encoding the (respective orthogonal) projections of the points onto the diagonal
    '''
    Z = (X[:,0] + X[:,1]) / 2.
    return np.array([Z , Z]).T


def _dist_to_diag(X, internal_p):
    '''
    :param X: (n x 2) array encoding the points of a persistent diagram.
    :param internal_p: Ground metric (i.e. norm L^p).
    :returns: (n) array encoding the (respective orthogonal) distances of the points to the diagonal

    .. note::
        Assumes that the points are above the diagonal.
    '''
    if len(np.shape(X)) > 1:
        return (X[:, 1] - X[:, 0]) * 2 ** (1.0 / internal_p - 1)
    else:
        return (X[1] - X[0]) * 2 ** (1.0 / internal_p - 1)


def _build_dist_matrix(X, Y, order, internal_p):
    '''
    :param X: (n x 2) numpy.array encoding the (points of the) first diagram.
    :param Y: (m x 2) numpy.array encoding the second diagram.
    :param order: exponent for the Wasserstein metric.
    :param internal_p: Ground metric (i.e. norm L^p).
    :returns: (n+1) x (m+1) np.array encoding the cost matrix C.
                For 0 <= i < n, 0 <= j < m, C[i,j] encodes the distance between X[i] and Y[j],
                while C[i, m] (resp. C[n, j]) encodes the distance (to the p) between X[i] (resp Y[j])
                and its orthogonal projection onto the diagonal.
                note also that C[n, m] = 0  (it costs nothing to move from the diagonal to the diagonal).
    '''
    Cxd = _dist_to_diag(X, internal_p)**order
    Cdy = _dist_to_diag(Y, internal_p)**order
    if np.isinf(internal_p):
        C = sc.cdist(X,Y, metric='chebyshev')**order
    else:
        C = sc.cdist(X,Y, metric='minkowski', p=internal_p)**order
    Cf = np.hstack((C, Cxd[:,None]))
    Cdy = np.append(Cdy, 0)
    Cf = np.vstack((Cf, Cdy[None,:]))

    return Cf


def _perstot_autodiff(X, order, internal_p):
    '''
    Version of _perstot that works on eagerpy tensors.
    '''
    return _dist_to_diag(X, internal_p).norms.lp(order)

def _perstot(X, order, internal_p, enable_autodiff):
    '''
    :param X: (n x 2) numpy.array (points of a given diagram).
    :param order: exponent for Wasserstein.
    :param internal_p: Ground metric on the (upper-half) plane (i.e. norm L^p in R^2).
    :param enable_autodiff: If X is torch.tensor, tensorflow.Tensor or jax.numpy.ndarray, make the computation
        transparent to automatic differentiation.
    :type enable_autodiff: bool
    :returns: float, the total persistence of the diagram (that is, its distance to the empty diagram).
    '''
    if enable_autodiff:
        import eagerpy as ep

        return _perstot_autodiff(ep.astensor(X), order, internal_p).raw
    else:
        return np.linalg.norm(_dist_to_diag(X, internal_p), ord=order)

def wasserstein_distance_inference(X, Y, matching=False, order=1., internal_p=np.inf, enable_autodiff=False):
    n = len(X)
    m = len(Y)

    # handle empty diagrams
    if n == 0:
        if m == 0:
            if not matching:
                # What if enable_autodiff?
                return 0., np.array([]), 0, 0, 0
            else:
                return 0., np.array([]), 0, 0, 0
        else:
            if not matching:
                return _perstot(Y, order, internal_p, enable_autodiff), np.array([]), 0, 0, 0
            else:
                return _perstot(Y, order, internal_p, enable_autodiff), np.array([[-1, j] for j in range(m)]), 0, 0, 0
    elif m == 0:
        if not matching:
            return _perstot(X, order, internal_p, enable_autodiff), np.array([]), 0, 0, 0
        else:
            return _perstot(X, order, internal_p, enable_autodiff), np.array([[i, -1] for i in range(n)]), 0, 0, 0

    if enable_autodiff:
        import eagerpy as ep

        X_orig = ep.astensor(X)
        Y_orig = ep.astensor(Y)
        X = X_orig.numpy()
        Y = Y_orig.numpy()

    M = _build_dist_matrix(X, Y, order=order, internal_p=internal_p)

    # original
    a = np.ones(n+1) # weight vector of the input diagram. Uniform here.
    a[-1] = m
    b = np.ones(m+1) # weight vector of the input diagram. Uniform here.
    b[-1] = n



    if matching:
        assert not enable_autodiff, "matching and enable_autodiff are currently incompatible"
        P = ot.emd(a=a, b=b, M=M, numItermax=2000000)
        ot_cost = np.sum(np.multiply(P, M))
        P[-1, -1] = 0  # Remove matching corresponding to the diagonal
        match = np.argwhere(P)
        # Now we turn to -1 points encoding the diagonal
        match[:, 0][match[:, 0] >= n] = -1
        match[:, 1][match[:, 1] >= m] = -1
        return ot_cost ** (1. / order), match

    if enable_autodiff:
        P = ot.emd(a=a, b=b, M=M, numItermax=2000000)

        # original
        pairs_X_Y = np.argwhere(P[:-1, :-1])
        pairs_X_diag = np.nonzero(P[:-1, -1])
        pairs_Y_diag = np.nonzero(P[-1, :-1])

        new_dists = []
        tmp_dists = []
        wxy = 0; wxd = 0; wyd = 0
        # empty arrays are not handled properly by the helpers, so we avoid calling them
        if len(pairs_X_Y):
            for pXY in pairs_X_Y:
                new_dists.append((Y_orig[pXY[1]] - X_orig[pXY[0]]).norms.lp(internal_p, axis=-1).norms.lp(order))
                tmp_dists.append((Y_orig[pXY[1]] - X_orig[pXY[0]]).norms.lp(internal_p, axis=-1).norms.lp(order))
            tmp_dists = [dist.reshape(1) for dist in tmp_dists]
            wxy = ep.concatenate(tmp_dists).norms.lp(order).raw
            tmp_dists = []
        if len(pairs_X_diag[0]):
            for pX in pairs_X_diag[0]:
                new_dists.append(_perstot_autodiff(X_orig[pX], order, internal_p))
                tmp_dists.append(_perstot_autodiff(X_orig[pX], order, internal_p))
            tmp_dists = [dist.reshape(1) for dist in tmp_dists]
            wxd = ep.concatenate(tmp_dists).norms.lp(order).raw
            tmp_dists = []
        if len(pairs_Y_diag[0]):
            for pY in pairs_Y_diag[0]:
                new_dists.append(_perstot_autodiff(Y_orig[pY], order, internal_p))
                tmp_dists.append(_perstot_autodiff(Y_orig[pY], order, internal_p))
            tmp_dists = [dist.reshape(1) for dist in tmp_dists]
            wyd = ep.concatenate(tmp_dists).norms.lp(order).raw
            tmp_dists = []

        new_dists = [dist.reshape(1) for dist in new_dists]

        try:
            return ep.concatenate(new_dists).norms.lp(order).raw, None, wxy, wxd, wyd
        except BaseException:
            print(X)
            print(Y)
            print(new_dists)
            return ep.concatenate(new_dists).norms.lp(order).raw, None, wxy, wxd, wyd
        # We can also concatenate the 3 vectors to compute just one norm.

    # Comptuation of the otcost using the ot.emd2 library.
    # Note: it is the Wasserstein distance to the power q.
    # The default numItermax=100000 is not sufficient for some examples with 5000 points, what is a good value?
    ot_cost = ot.emd2(a, b, M, numItermax=2000000)

    return ot_cost ** (1. / order)


def wasserstein_distance(X, Y, matching=False, order=1., internal_p=np.inf, enable_autodiff=False, num_models = 1):
    '''
    :param X: (n x 2) numpy.array encoding the (finite points of the) first diagram. Must not contain essential points
                (i.e. with infinite coordinate).
    :param Y: (m x 2) numpy.array encoding the second diagram.
    :param matching: if True, computes and returns the optimal matching between X and Y, encoded as
                     a (n x 2) np.array  [...[i,j]...], meaning the i-th point in X is matched to
                     the j-th point in Y, with the convention (-1) represents the diagonal.
    :param order: exponent for Wasserstein; Default value is 1.
    :param internal_p: Ground metric on the (upper-half) plane (i.e. norm L^p in R^2);
                       Default value is `np.inf`.
    :param enable_autodiff: If X and Y are torch.tensor or tensorflow.Tensor, make the computation
        transparent to automatic differentiation. This requires the package EagerPy and is currently incompatible
        with `matching=True`.

        .. note:: This considers the function defined on the coordinates of the off-diagonal points of X and Y
            and lets the various frameworks compute its gradient. It never pulls new points from the diagonal.
    :type enable_autodiff: bool
    :num_models: number of SPT cycle bases
    :returns: the Wasserstein distance of order q (1 <= q < infinity) between persistence diagrams with
              respect to the internal_p-norm as ground metric.
              If matching is set to True, also returns the optimal matching between X and Y.
    '''
    n = len(X)
    m = len(Y)

    # handle empty diagrams
    if n == 0:
        if m == 0:
            if not matching:
                # What if enable_autodiff?
                return 0., np.array([]), 0, 0, 0
            else:
                return 0., np.array([]), 0, 0, 0
        else:
            if not matching:
                return _perstot(Y, order, internal_p, enable_autodiff), np.array([]), 0, 0, 0
            else:
                return _perstot(Y, order, internal_p, enable_autodiff), np.array([[-1, j] for j in range(m)]), 0, 0, 0
    elif m == 0:
        if not matching:
            return _perstot(X, order, internal_p, enable_autodiff), np.array([]), 0, 0, 0
        else:
            return _perstot(X, order, internal_p, enable_autodiff), np.array([[i, -1] for i in range(n)]), 0, 0, 0

    if enable_autodiff:
        import eagerpy as ep

        X_orig = ep.astensor(X)
        Y_orig = ep.astensor(Y)
        X = X_orig.numpy()
        Y = Y_orig.numpy()

    M = _build_dist_matrix(X, Y, order=order, internal_p=internal_p)
    '''
    # original
    a = np.ones(n+1) # weight vector of the input diagram. Uniform here.
    a[-1] = m
    b = np.ones(m+1) # weight vector of the input diagram. Uniform here.
    b[-1] = n
    '''

    # new
    a = np.ones(n)  # weight vector of the input diagram. Uniform here.
    b = np.ones(m + 1)  # weight vector of the input diagram. Uniform here.
    b[-1] = n - m


    if matching:
        assert not enable_autodiff, "matching and enable_autodiff are currently incompatible"
        P = ot.emd(a=a,b=b,M=M, numItermax=2000000)
        ot_cost = np.sum(np.multiply(P,M))
        P[-1, -1] = 0  # Remove matching corresponding to the diagonal
        match = np.argwhere(P)
        # Now we turn to -1 points encoding the diagonal
        match[:,0][match[:,0] >= n] = -1
        match[:,1][match[:,1] >= m] = -1
        return ot_cost ** (1./order) , match

    if enable_autodiff:
        # original
        #P = ot.emd(a=a, b=b, M=M, numItermax=2000000)

        # new
        P = ot.emd(a=a, b=b, M=M[:-1, :], numItermax=2000000)
        # original
        #pairs_X_Y = np.argwhere(P[:-1, :-1])
        #pairs_X_diag = np.nonzero(P[:-1, -1])
        # pairs_Y_diag = np.nonzero(P[-1, :-1])
        # dists = []

        # new
        pairs_X_Y = np.argwhere(P[: , :-1])
        pairs_X_diag = np.nonzero(P[: , -1])

        tmp_dists = []
        wxy = 0; wxd = 0; wyd = 0
        new_dists = []
        ind_tmp_dist = np.array([])
        # empty arrays are not handled properly by the helpers, so we avoid calling them
        if len(pairs_X_Y):
            # original
            #dists.append((Y_orig[pairs_X_Y[:, 1]] - X_orig[pairs_X_Y[:, 0]]).norms.lp(internal_p, axis=-1).norms.lp(order))
            # new
            for pXY in pairs_X_Y:
                new_dists.append((Y_orig[pXY[1]] - X_orig[pXY[0]]).norms.lp(internal_p, axis=-1).norms.lp(order))
                tmp_dists.append((Y_orig[pXY[1]] - X_orig[pXY[0]]).norms.lp(internal_p, axis=-1).norms.lp(order))
            tmp_dists = [dist.reshape(1) for dist in tmp_dists]
            wxy = ep.concatenate(tmp_dists).norms.lp(order).raw
            tmp_dists = []
        if len(pairs_X_diag[0]):
            # original
            #dists.append(_perstot_autodiff(X_orig[pairs_X_diag], order, internal_p))
            # new

            # delete cycles that are too far away
            new_tmp_dists = []
            try:
                assert n % num_models == 0
            except BaseException:
                print(n)
                assert n % num_models == 0
            save_PD = int(n / num_models) - len(pairs_X_Y)
            for pX in pairs_X_diag[0]:
                new_tmp_dists.append(_perstot_autodiff(X_orig[pX], order, internal_p))
            tmp_dists_array = np.array([td.numpy() for td in new_tmp_dists]).reshape(-1)
            if save_PD == 0:
                ind_tmp_dist = np.array([])
            else:
                # choose the save_PD points away from the diagonal
                ind_tmp_dist = np.argpartition(tmp_dists_array,-save_PD)[-save_PD:]
            for pX in ind_tmp_dist:
                new_dists.append(new_tmp_dists[pX])
                tmp_dists.append(new_tmp_dists[pX])
            tmp_dists = [dist.reshape(1) for dist in tmp_dists]
            if len(tmp_dists) > 0:
                wxd = ep.concatenate(tmp_dists).norms.lp(order).raw
            tmp_dists = []

        '''
        # original
        if len(pairs_Y_diag[0]):
            # original
        #    dists.append(_perstot_autodiff(Y_orig[pairs_Y_diag], order, internal_p))
            #new
            for pY in pairs_Y_diag[0]:
                new_dists.append(_perstot_autodiff(Y_orig[pY], order, internal_p))
        '''

        #dists = [dist.reshape(1) for dist in dists]
        new_dists = [dist.reshape(1) for dist in new_dists]


        #print(ep.concatenate(dists).norms.lp(order).raw)
        #print(X)
        #print(Y)
        #print(new_dists)
        #print(ep.concatenate(new_dists).norms.lp(order).raw)

        if len(ind_tmp_dist) > 0:
            final_ind = np.concatenate((pairs_X_Y[:, 0], pairs_X_diag[0][ind_tmp_dist]))
        else:
            final_ind = pairs_X_Y[:, 0]

        assert len(final_ind) == int(n / num_models)
        assert len(final_ind) == len(np.unique(final_ind))

        try:
            return ep.concatenate(new_dists).norms.lp(order).raw, final_ind, wxy, wxd, wyd
        except BaseException:
            print(X)
            print(Y)
            print(new_dists)
            return ep.concatenate(new_dists).norms.lp(order).raw, final_ind, wxy, wxd, wyd
        # We can also concatenate the 3 vectors to compute just one norm.

    # Comptuation of the otcost using the ot.emd2 library.
    # Note: it is the Wasserstein distance to the power q.
    # The default numItermax=100000 is not sufficient for some examples with 5000 points, what is a good value?
    ot_cost = ot.emd2(a, b, M, numItermax=2000000)

    return ot_cost ** (1./order)
