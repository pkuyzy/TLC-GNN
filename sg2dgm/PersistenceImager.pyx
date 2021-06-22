"""
Copyright: https://gitlab.com/csu-tda/PersistenceImages
"""

import numpy as np
from scipy.special import erfc


def linear_ramp(birth, pers, low=0.0, high=1.0, start=0.0, end=1.0):
    """
    continuous peicewise linear ramp function which is constant below and above specified input values
    :param birth: birth coordinates
    :param pers: persistence coordinates
    :param low: minimal weight
    :param high: maximal weight
    :param start: start persistence value of linear transition from low to high weight
    :param end: end persistence value of linear transition from low to high weight
    :return: weight at persistence pair
    """
    n = birth.shape[0]
    w = np.zeros((n,))
    for i in range(n):
        if pers[i] < start:
            w[i] = low
        elif pers[i] > end:
            w[i] = high
        else:
            w[i] = (pers[i] - start) * (high - low) / (end - start) + low

    return w

def bvncdf(birth, pers, mu=None, sigma=None):
    """
    Optimized bivariate normal cumulative distribution function for computing persistence images using a Gaussian kernel
    :param birth: birth-coordinate(s) of diagram pairs
    :param pers: persistence-coordinate(s) of diagram pairs
    :param mu: (2,)-numpy array specifying x and y coordinates of distribution mean
    :param sigma: (2,2)-numpy array specifying distribution covariance matrix or numeric if distribution is isotropic
    :return: P(X <= birth, Y <= pers)
    """
    if mu is None:
        mu = np.array([0.0, 0.0], dtype=np.float64)
    if sigma is None:
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    if sigma[0, 1] == 0.0:
        return _sbvn_cdf(birth, pers,
                         mu_x=mu[0], mu_y=mu[1], sigma_x=sigma[0, 0], sigma_y=sigma[1, 1])
    else:
        return _bvn_cdf(birth, pers,
                        mu_x=mu[0], mu_y=mu[1], sigma_xx=sigma[0, 0], sigma_yy=sigma[1, 1], sigma_xy=sigma[0, 1])


def _norm_cdf(x):
    """
    univariate normal cumulative distribution function with mean 0.0 and standard deviation 1.0
    :param x: value at which to evaluate the cdf (upper limit)
    :return: P(X <= x), for X ~ N[0,1]
    """
    return erfc(-x / np.sqrt(2.0)) / 2.0


def _sbvn_cdf(x, y, mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0):
    """
    standard bivariate normal cumulative distribution function with specified mean and variances
    :param x: x-coordinate(s) at which to evaluate the cdf (upper limit)
    :param y: y-coordinate(s) at which to evaluate the cdf (upper limit)
    :param mu_x: x-coordinate of mean of bivariate normal
    :param mu_y: y-coordinate of mean of bivariate normal
    :param sigma_x: variance in x
    :param sigma_y: variance in y
    :return: P(X <= x, Y <= y)
    """
    x = (x - mu_x) / np.sqrt(sigma_x)
    y = (y - mu_y) / np.sqrt(sigma_y)
    return _norm_cdf(x) * _norm_cdf(y)


def _bvn_cdf(x, y, mu_x=0.0, mu_y=0.0, sigma_xx=1.0, sigma_yy=1.0, sigma_xy=0.0):
    """
    bivariate normal cumulative distribution function with specified mean and covariance matrix based on the Matlab
    implementations by Thomas H. Jørgensen (http://www.tjeconomics.com/code/) and Alan Genz
    (http://www.math.wsu.edu/math/faculty/genz/software/matlab/bvnl.m) using the approach described by Drezner
    and Wesolowsky (https://doi.org/10.1080/00949659008811236)
    :param x: x-coordinate(s) at which to evaluate the cdf (upper limit)
    :param y: y-coordinate(s) at which to evaluate the cdf (upper limit)
    :param mu_x: x-coordinate of mean of bivariate normal
    :param mu_y: y-coordinate of mean of bivariate normal
    :param sigma_xx: variance in x
    :param sigma_yy: variance in y
    :param sigma_xy: covariance of x and y
    :return: P(X <= x, Y <= y)
    """
    dh = -(x - mu_x) / np.sqrt(sigma_xx)
    dk = -(y - mu_y) / np.sqrt(sigma_yy)

    hk = np.multiply(dh, dk)
    r = sigma_xy / np.sqrt(sigma_xx * sigma_yy)

    lg, w, x = _gauss_legendre_quad(r)

    dim1 = np.ones((len(dh),), dtype=np.float64)
    dim2 = np.ones((lg,), dtype=np.float64)
    bvn = np.zeros((len(dh),), dtype=np.float64)

    if abs(r) < 0.925:
        hs = (np.multiply(dh, dh) + np.multiply(dk, dk)) / 2.0
        asr = np.arcsin(r)
        sn1 = np.sin(asr * (1.0 - x) / 2.0)
        sn2 = np.sin(asr * (1.0 + x) / 2.0)
        dim1w = np.outer(dim1, w)
        hkdim2 = np.outer(hk, dim2)
        hsdim2 = np.outer(hs, dim2)
        dim1sn1 = np.outer(dim1, sn1)
        dim1sn2 = np.outer(dim1, sn2)
        sn12 = np.multiply(sn1, sn1)
        sn22 = np.multiply(sn2, sn2)
        bvn = asr * np.sum(np.multiply(dim1w, np.exp(np.divide(np.multiply(dim1sn1, hkdim2) - hsdim2,
                                                               (1 - np.outer(dim1, sn12))))) +
                           np.multiply(dim1w, np.exp(np.divide(np.multiply(dim1sn2, hkdim2) - hsdim2,
                                                               (1 - np.outer(dim1, sn22))))), axis=1) / (4 * np.pi) \
              + np.multiply(_norm_cdf(-dh), _norm_cdf(-dk))
    else:
        if r < 0:
            dk = -dk
            hk = -hk

        if abs(r) < 1:
            opmr = (1.0 - r) * (1.0 + r)
            sopmr = np.sqrt(opmr)
            xmy2 = np.multiply(dh - dk, dh - dk)
            xmy = np.sqrt(xmy2)
            rhk8 = (4.0 - hk) / 8.0
            rhk16 = (12.0 - hk) / 16.0
            asr = -1.0 * (np.divide(xmy2, opmr) + hk) / 2.0

            ind = asr > 100
            bvn[ind] = sopmr * np.multiply(np.exp(asr[ind]),
                                           1.0 - np.multiply(np.multiply(rhk8[ind], xmy2[ind] - opmr),
                                                             (1.0 - np.multiply(rhk16[ind], xmy2[ind]) / 5.0) / 3.0)
                                           + np.multiply(rhk8[ind], rhk16[ind]) * opmr * opmr / 5.0)

            ind = hk > -100
            ncdfxmyt = np.sqrt(2.0 * np.pi) * _norm_cdf(-xmy / sopmr)
            bvn[ind] = bvn[ind] - np.multiply(np.multiply(np.multiply(np.exp(-hk[ind] / 2.0), ncdfxmyt[ind]), xmy[ind]),
                                              1.0 - np.multiply(np.multiply(rhk8[ind], xmy2[ind]),
                                                                (1.0 - np.multiply(rhk16[ind], xmy2[ind]) / 5.0) / 3.0))
            sopmr = sopmr / 2
            for ix in [-1, 1]:
                xs = np.multiply(sopmr + sopmr * ix * x, sopmr + sopmr * ix * x)
                rs = np.sqrt(1 - xs)
                xmy2dim2 = np.outer(xmy2, dim2)
                dim1xs = np.outer(dim1, xs)
                dim1rs = np.outer(dim1, rs)
                dim1w = np.outer(dim1, w)
                rhk16dim2 = np.outer(rhk16, dim2)
                hkdim2 = np.outer(hk, dim2)
                asr1 = -1.0 * (np.divide(xmy2dim2, dim1xs) + hkdim2) / 2.0

                ind1 = asr1 > -100
                cdim2 = np.outer(rhk8, dim2)
                sp1 = 1.0 + np.multiply(np.multiply(cdim2, dim1xs), 1.0 + np.multiply(rhk16dim2, dim1xs))
                ep1 = np.divide(np.exp(np.divide(-np.multiply(hkdim2, (1.0 - dim1rs)),
                                                 2.0 * (1.0 + dim1rs))), dim1rs)
                bvn = bvn + np.sum(np.multiply(np.multiply(np.multiply(sopmr, dim1w), np.exp(np.multiply(asr1, ind1))),
                                               np.multiply(ep1, ind1) - np.multiply(sp1, ind1)), axis=1)
            bvn = -bvn / (2.0 * np.pi)

        if r > 0:
            bvn = bvn + _norm_cdf(-np.maximum(dh, dk))
        elif r < 0:
            bvn = -bvn + np.maximum(0, _norm_cdf(-dh) - _norm_cdf(-dk))

    return bvn


def _gauss_legendre_quad(r):
    """
    Return weights and abscissae for the Legendre-Gauss quadrature integral approximation
    :param r: correlation
    :return:
    """
    if np.abs(r) < 0.3:
        lg = 3
        w = np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904])
        x = np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970])
    elif np.abs(r) < 0.75:
        lg = 6
        w = np.array([.04717533638651177, 0.1069393259953183, 0.1600783285433464,
                      0.2031674267230659, 0.2334925365383547, 0.2491470458134029])
        x = np.array([0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
                      0.5873179542866171, 0.3678314989981802, 0.1252334085114692])
    else:
        lg = 10
        w = np.array([0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
                      0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
                      0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
                      0.1527533871307259])
        x = np.array([0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
                      0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
                      0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
                      0.07652652113349733])

    return lg, w, x

class PersistenceImager:
    def __init__(self, birth_range=None, pers_range=None, pixel_size=None, resolution = 5,
                 weight=linear_ramp, weight_params=None, kernel=bvncdf, kernel_params=None):
        """
        class for transforming persistence diagrams into persistence images
        :param birth_range: tuple specifying lower and upper birth value of the persistence image
        :param pers_range: tuple specifying lower and upper persistence value of the persistence image
        :param pixel_size: size of square pixel
        :param weight: function to weight the birth-persistence plane
        :param weight_params: arguments needed to specify the weight function
        :param kernel: cumulative distribution function of kernel
        :param kernel_params: arguments needed to specify the kernel (cumulative distribution) function
        """
        # set defaults
        if birth_range is None:
            birth_range = (0.0, 1.0)
        if pers_range is None:
            pers_range = (0.0, 1.0)
        self._resolution = (resolution, resolution)
        if pixel_size is None:
            pixel_size = np.min([pers_range[1] - pers_range[0], birth_range[1] - birth_range[0]]) / resolution
        if weight_params is None:
            weight_params = {}
        if kernel_params is None:
            kernel_params = {'sigma': np.array([[1.0, 0.0], [0.0, 1.0]])}

        self.weight = weight
        self.weight_params = weight_params
        self.kernel = kernel
        self.kernel_params = kernel_params
        self._pixel_size = pixel_size
        self._birth_range = birth_range
        self._pers_range = pers_range
        self._width = birth_range[1] - birth_range[0]
        self._height = pers_range[1] - pers_range[0]
        # self._resolution = (int(self._width / self._pixel_size), int(self._height / self._pixel_size))
        self._create_mesh()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def resolution(self):
        return self._resolution

    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, val):
        self._pixel_size = val
        self._width = int(np.ceil((self.birth_range[1] - self.birth_range[0]) / self.pixel_size)) * self.pixel_size
        self._height = int(np.ceil((self.pers_range[1] - self.pers_range[0]) / self.pixel_size)) * self.pixel_size
        self._resolution = (int(self.width / self.pixel_size), int(self.height / self.pixel_size))
        self._create_mesh()

    @property
    def birth_range(self):
        return self._birth_range

    @birth_range.setter
    def birth_range(self, val):
        self._birth_range = val
        self._width = int(np.ceil((self.birth_range[1] - self.birth_range[0]) / self.pixel_size)) * self._pixel_size
        self._resolution = (int(self.width / self.pixel_size), int(self.height / self.pixel_size))
        self._create_mesh()

    @property
    def pers_range(self):
        return self._pers_range

    @pers_range.setter
    def pers_range(self, val):
        self._pers_range = val
        self._height = int(np.ceil((self.pers_range[1] - self.pers_range[0]) / self.pixel_size)) * self._pixel_size
        self._resolution = (int(self.width / self.pixel_size), int(self.height / self.pixel_size))
        self._create_mesh()

    def __repr__(self):
        repr_str = 'PersistenceImager object: \n' +\
                   '  pixel size: %g \n' % self.pixel_size +\
                   '  resolution: (%d, %d) \n' % self.resolution +\
                   '  birth range: (%g, %g) \n' % self.birth_range +\
                   '  persistence range: (%g, %g) \n' % self.pers_range +\
                   '  weight: %s \n' % self.weight.__name__ +\
                   '  kernel: %s \n' % self.kernel.__name__ +\
                   '  weight parameters: %s \n' % dict_print(self.weight_params) +\
                   '  kernel parameters: %s' % dict_print(self.kernel_params)
        return repr_str

    def _create_mesh(self):
        # padding around specified image ranges as a result of incommensurable ranges and pixel width
        db = self._width - (self._birth_range[1] - self._birth_range[0])
        dp = self._height - (self._pers_range[1] - self._pers_range[0])

        # adjust image ranges to accommodate incommensurable ranges and pixel width
        self._birth_range = (self._birth_range[0] - db / 2, self._birth_range[1] + db / 2)
        self._pers_range = (self._pers_range[0] - dp / 2, self._pers_range[1] + dp / 2)
        # construct linear spaces defining pixel locations
        self._bpnts = np.array(np.linspace(self._birth_range[0], self._birth_range[1] + self._pixel_size,
                                           self._resolution[0] + 1, endpoint=False, dtype=np.float64))
        self._ppnts = np.array(np.linspace(self._pers_range[0], self._pers_range[1] + self._pixel_size,
                                           self._resolution[1] + 1, endpoint=False, dtype=np.float64))

    def fit(self, pers_dgms, skew=True):
        """
        automatically choose persistence images parameters based on a collection of persistence diagrams
        :param pers_dgms: An iterable of (N,2) numpy arrays encoding persistence diagrams
        :param skew: boolean flag indicating if diagram needs to be converted to birth-persistence coordinates
                     (default: True)
        """
        min_birth = np.Inf
        max_birth = -np.Inf
        min_pers = np.Inf
        max_pers = -np.Inf

        # loop over diagrams to determine the maximum extent of the pairs contained in the birth-persistence plane
        for pers_dgm in pers_dgms:
            pers_dgm = np.copy(pers_dgm)
            if skew:
                pers_dgm[:, 1] = pers_dgm[:, 1] - pers_dgm[:, 0]

            min_b, min_p = pers_dgm.min(axis=0)
            max_b, max_p = pers_dgm.max(axis=0)

            if min_b < min_birth:
                min_birth = min_b

            if min_p < min_pers:
                min_pers = min_p

            if max_b > max_birth:
                max_birth = max_b

            if max_p > max_pers:
                max_pers = max_p

        self.birth_range = (min_birth, max_birth)
        self.pers_range = (min_pers, max_pers)

    def transform(self, pers_dgm, skew=True ):
        """
        transform a persistence diagram to a persistence image using the parameters specified in the PersistenceImager
        object instance
        :param pers_dgm: (N,2) numpy array of persistence pairs encoding a persistence diagram
        :param skew: boolean flag indicating if diagram needs to be converted to birth-persistence coordinates
                     (default: True)
        :return: numpy array encoding the persistence image
        """
        pers_dgm = np.copy(pers_dgm)
        pers_img = np.zeros(self.resolution)
        n = pers_dgm.shape[0]
        general_flag = True

        # if necessary convert from birth-death coordinates to birth-persistence coordinates
        if skew:
            pers_dgm[:, 1] = pers_dgm[:, 1] - pers_dgm[:, 0]

        # compute weights at each persistence pair
        wts = self.weight(pers_dgm[:, 0], pers_dgm[:, 1], **self.weight_params)

        # handle the special case of a standard, isotropic Gaussian kernel
        if self.kernel == bvncdf:
            general_flag = False
            sigma = self.kernel_params['sigma']

            # sigma is specified by a single variance
            if isinstance(sigma, (int, float)):
                sigma = np.array([[sigma, 0.0], [0.0, sigma]], dtype=np.float64)

            if (sigma[0, 0] == sigma[1, 1] and sigma[0, 1] == 0.0):
                sigma = np.sqrt(sigma[0, 0])
                for i in range(n):
                    ncdf_b = _norm_cdf((self._bpnts - pers_dgm[i, 0]) / sigma)
                    ncdf_p = _norm_cdf((self._ppnts - pers_dgm[i, 1]) / sigma)
                    curr_img = ncdf_p[None, :] * ncdf_b[:, None]
                    pers_img += wts[i]*(curr_img[1:, 1:] - curr_img[:-1, 1:] - curr_img[1:, :-1] + curr_img[:-1, :-1])
            else:
                general_flag = True

        # handle the general case
        if general_flag:
            bb, pp = np.meshgrid(self._bpnts, self._ppnts)
            bb = bb.flatten(order='C')
            pp = pp.flatten(order='C')
            for i in range(n):
                self.kernel_params['mu'] = pers_dgm[i, :]
                curr_img = np.reshape(self.kernel(bb, pp, **self.kernel_params),
                                      (self.resolution[0]+1, self.resolution[1]+1), order='C')
                pers_img += wts[i] * (curr_img[1:, 1:] - curr_img[:-1, 1:] - curr_img[1:, :-1] + curr_img[:-1, :-1])

        return pers_img

    def fit_transform(self, pers_dgms, skew=True):
        """
        automatically choose persistence image parameters based on a collection of persistence diagrams and transform
        the collection of diagrams into images using the parameters specified in the PersistenceImager object instance
        :param pers_dgms: An iterable of (N,2) numpy arrays encoding persistence diagrams
        :param skew: boolean flag indicating if diagram needs to be converted to birth-persistence coordinates
                     (default: True)
        :return: Python list of numpy arrays encoding the persistence images
        """
        pers_dgms = np.copy(pers_dgms)

        # fit imager parameters
        self.fit(pers_dgms, skew=skew)

        # loop over each diagram and compute its image
        num_dgms = len(pers_dgms)
        pers_imgs = [None] * num_dgms
        for i in range(num_dgms):
            pers_imgs[i] = self.transform(pers_dgms[i], skew=skew)

        return pers_imgs

def dict_print(dict_in):
    # print dictionary contents in human-readable format
    if dict_in is None:
        str_out = 'None'
    else:
        str_out = []
        for key, val in dict_in.items():
            str_out.append('%s: %s' % (key, str(val)))
        str_out = '{' + ', '.join(str_out) + '}'

    return str_out