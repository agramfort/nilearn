"""
=================================================================
Run Chambolle-Pock primal-dual scheme for TV + L1 deconvolution
=================================================================

This example runs decoding using TV-L1 on simulated data.

See:
Identifying predictive regions from fMRI with TV-L1 prior
A. Gramfort, B. Thirion and G. Varoquaux, Proc. PRNI conf. 2013
available on IEEE explore.
"""

# author : alexandre gramfort <alexandre.gramfort@telecom-paristech.fr>
#          gael varoquaux <gael.varoquaux@normalesup.org>
#
# license : simplified BSD

print __doc__

import sys, itertools, operator
from time import time
import numpy as np
import pylab as pl
from scipy import linalg

from sklearn import metrics, linear_model, feature_selection
from sklearn.linear_model.base import LinearModel, center_data
from sklearn.linear_model.coordinate_descent import _path_residuals
from sklearn.base import RegressorMixin
from sklearn.utils import array2d, check_random_state, atleast2d_or_csc
from sklearn.cross_validation import check_cv
# from sklearn.externals.joblib import Parallel, delayed, Memory
from joblib import Parallel, delayed, Memory


from plot_simulated_data import create_simulation_data, plot_slices

###############################################################################
# Code to solve regression with l1 in synthesis

def estimate_lipschitz_constant(w0, D, DT, tol=1e-3, maxit=1000,
                                rand_state=None):
    """Compute approximate lipschitz constant
    of linear operator : x -> DT(D(x))
    using a power method"""
    # XXX: should be done using arpack
    rng = np.random.RandomState(rand_state)
    a = rng.randn(*w0.shape)
    a /= linalg.norm(a)
    lipschitz_constant_old = np.inf
    for i in range(100):
        b = DT.matvec(D.matvec(a))
        a = b / linalg.norm(b)
        lipschitz_constant = (b * a).sum()
        if abs(lipschitz_constant - lipschitz_constant_old) < tol:
            break
    return lipschitz_constant


def prox_l1(x, tau):
    """Prox tau*L1 inplace"""
    x_nz = x.nonzero()
    shrink = np.zeros_like(x)
    shrink[x_nz] = np.maximum(1 - tau / np.abs(x[x_nz]), 0)
    x *= shrink
    return x


def prox_l21(x, tau):
    """Prox tau*L21 inplace,

    where l21 is the l2 norm of the first 3 lines of x, and the l1 of
    all the rest (including the group defined by these, and the remaing
    last line of x).
    """
    shrink = np.zeros_like(x)
    # First deal with the last lines of x, which is a standard l1:
    x_nz = x[-1].nonzero()
    view = shrink[-1]
    view[x_nz] = np.maximum(1 - tau / np.abs(x[-1][x_nz]), 0)
    # Then deal with the first 2 lines, on which we do an l21 norm:
    norm = np.sqrt((x[:-1] ** 2).sum(axis=0))
    x_nz = norm.nonzero()
    if np.any(x_nz):
        scaling = np.maximum(1 - tau / norm[x_nz], 0)
        for view in shrink[:-1]:
            # Ugly: we need to take a view
            view[x_nz] = scaling
    x *= shrink
    return x


def ridge_svd(X_svd, y, alpha):
    U, s, V = X_svd
    return np.dot(V.T * (s / (alpha + s ** 2)), np.dot(U.T, y))


def _l1_objective(y, X, w, alpha, D, mask=None, mu=0):
    Dw = D.matvec(w)
    if mask is not None:
        w = w[mask]
    obj = 0.5 * linalg.norm(y - np.dot(X, w)) ** 2 + \
                    alpha * np.sum(np.abs(Dw))
    if mu != 0:
        obj += .5 * mu * (w**2).sum()
    return obj


def _l21_objective(y, X, w, alpha, D, mask=None, mu=0):
    Dw = D.matvec(w)
    if mask is not None:
        w = w[mask]
    obj = 0.5 * linalg.norm(y - np.dot(X, w)) ** 2 + \
                    alpha * (np.sum(np.abs(Dw[-1])) +
                             np.sum(np.sqrt((Dw[:-1]**2).sum(axis=0))))
    if mu != 0:
        obj += .5 * mu * (w**2).sum()
    return obj


def unmask(w, mask):
    out = np.zeros(mask.shape, dtype=w.dtype)
    out[mask] = w
    return out


def chambolle_pock_l1(X, y, alpha, D, DT, max_iter=300, verbose=False,
                      init=None, tol=1e-5, anisotropic=True, mask=None,
                      mu=0, store_path=False):
    """Run Chambolle-Pock primal-dual scheme for L1 synthesis pb

    minimize 0.5 || y - Xw ||^2 + alpha || D w ||_1 + 0.5 * mu ||w||^2
        x

    Where the norm on D w may either be an l_1 or l_21 depending on the
    choice of anisotropic

    D and DT can be scipy.sparse.linalg.LinearOperator objects
    """
    n_samples, n_features = X.shape

    if init is None:
        if mask is None:
            w = np.zeros(n_features, dtype=X.dtype)
        else:
            w = np.zeros(mask.shape, dtype=X.dtype)
        v = np.zeros_like(D.matvec(w))
    else:
        w, v = init
        w = w.copy()
        v = v.copy()

    z = w.copy()

    # Precompute SVD of X
    Ux, sx, Vx = X_svd = linalg.svd(X, full_matrices=False)
    # Careful: sx is the square of the singular values:
    sx **= 2
    if mu == 'auto':
        mu = sx[0] * 1e-4

    # Precompute np.dot(X.T, y), it's always useful
    Xy = np.dot(X.T, y)

    pobj = np.zeros(max_iter)
    prox_f = prox_l1
    objective = _l1_objective
    if not anisotropic:
        if isinstance(D, GradientId):
            prox_f = prox_l21
            objective = _l21_objective
        else:
            print 'Anistropic passed, but not TV+l1. Cannot solve'
    if store_path:
        path = list()
    else:
        path = None

    prox_f_star = lambda u, alpha: u - alpha * prox_f(u / alpha, 1. / alpha)
    if mask is None:
        prox_g = lambda w, alpha: w + ridge_svd(X_svd,
                                                y - np.dot(X, w), 1. / alpha)
        # A formula obtained by solving the normal equations
        def prox_g(w, alpha):
            w = w / alpha
            w += Xy
            # Trick to work on non full matrices in the SVD:
            # we us a different formula on the span of the matrices and
            # the orthogonal
            out = np.dot(Vx.T * (1. / (sx + (mu + 1. / alpha))
                                 - 1. / (mu + 1. / alpha)),
                         np.dot(Vx, w))
            out += w / (mu + 1. / alpha)
            return out
    else:
        def prox_g(w, alpha):
            res = y - np.dot(X, w[mask])
            return w + unmask(ridge_svd(X_svd, res, 1. / alpha),
                              mask)
        def prox_g(w, alpha):
            w_ = w[mask] / alpha
            w_ += Xy
            # Trick to work on non full matrices in the SVD:
            # we us a different formula on the span of the matrices and
            # the orthogonal
            out = np.dot(Vx.T * (1. / (sx + (mu + 1. / alpha))
                                 - 1. / (mu + 1. / alpha)),
                         np.dot(Vx, w_))
            out += w_ / (mu + 1. / alpha)
            #return unmask(out, mask)
            w = w.copy()
            w[mask] = out
            return w

    L = estimate_lipschitz_constant(w, D, DT, maxit=1000, tol=1e-3,
                                    rand_state=0)

    tau = 0.95 / L
    sigma = 0.95 / L

    cst = X_svd[1].max() / L
    tau /= cst
    sigma *= cst

    assert tau * sigma * L < 1

    theta = 0.5  # is 1 in paper
    times = list()

    for i in xrange(max_iter):
        v = prox_f_star(v + sigma * D.matvec(z), sigma)
        w_old = w.copy()
        if store_path:
            path.append(w_old)
        w = prox_g(w - tau * DT.matvec(v), tau / alpha)
        if mu != 0:
            # Apply algorithm 2 for accelerated convergence on uniformaly
            # convex problems
            theta = 1. / np.sqrt(1 + 2 * mu * tau)
            tau *= theta
            sigma /= theta
        z = w + theta * (w - w_old)
        pobj[i] = objective(y, X, w, alpha, D, mask=mask, mu=mu)
        times.append(time())
        dw = (linalg.norm(w - w_old, np.inf) / linalg.norm(w, np.inf))
        if verbose:
            print 'Chambolle-pock iteration %03i, cost %.2f, change % 5.2e' % (
                i, pobj[i], dw)
        # print dw
        # print gap(w, v)
        if dw < tol:
            print "Converged after %d iterations" % (i + 1)
            pobj = pobj[:i].copy()
            break
    else:
        print "Did not converge (%f > %f)" % (dw, tol)

    return w, v, pobj, path, np.array(times[:len(pobj)])


###############################################################################
# TV specific-code

def div(grad):
    """ Compute divergence of image gradient.

        This is an arbitrary-dimension divergence
    """
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] -= this_grad[:-1]
        this_res[1:-1] += this_grad[:-2]
        this_res[-1] += this_grad[-2]
    return res


def gradient(img):
    """Compute gradient of an arbitrary dimension image

    Parameters
    ===========
    img: ndarray
        N-dimensional image

    Returns
    =======
    gradient: ndarray
        Gradient of the image: the i-th component along the first
        axis is the gradient along the i-th axis of the original
        array img
    """
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    # 'Clever' code to have a view of the gradient with dimension i stop
    # at -1
    slice_all = [0, slice(None, -1), ]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


class Gradient(object):
    """ Gradient object that knows how to compute a 3D gradient on a flat
        array
    """
    def __init__(self, shape, precond=None):
        self.shape = tuple(shape)
        self.precond = precond

    def matvec(self, X):
        " From a flat array, return a (3, n_x, n_y, n_z) gradient"
        if self.precond is not None:
            X = np.dot(self.precond, X)
        X = np.reshape(X, self.shape)
        return gradient(X)


class Div(object):
    """ Div object that knows how to compute a 3D div on a 2D
        array: (3, n_voxels)
    """
    def __init__(self, precond=None):
        self.precond = precond

    def matvec(self, X):
        " From a (3, n_x, n_y, n_z) gradient return a flat divergence"
        divX = div(X).ravel()
        if self.precond is not None:
            divX = np.dot(self.precond.T, divX)
        return divX


def tv(X, y, shape, alpha, max_iter=300, verbose=False, init=None, tol=1e-5,
       precond=None, mask=None, mu=0, store_path=False):
    """Run Chambolle-Pock primal-dual scheme for TV deconvolution

    minimize 0.5 || y - Xw ||^2 + alpha || D w ||_1 + 0.5 * mu * ||w||^2
        x
    """
    D = Gradient(shape, precond=precond)
    DT = Div(precond=precond)
    return chambolle_pock_l1(X, y, alpha, D, DT, max_iter=max_iter,
                             verbose=verbose, init=init, tol=tol,
                             mask=mask, mu=mu, store_path=store_path)

###############################################################################
# TV-l1

class GradientId(object):
    """ Gradient + Identity object that knows how to compute a
        3D gradient on a flat array
    """
    def __init__(self, shape, l1_ratio, copy=True, precond=None):
        shape = tuple(shape)
        self.shape = shape
        self.l1_ratio = l1_ratio
        # Pre-allocate memory
        self.out = np.zeros((4, ) + shape)
        self.copy = copy
        self.precond = precond

    def matvec(self, X):
        "From a flat array, return a (4, n_x, n_y, n_z) gradient + identity"
        if self.precond is not None:
            X = np.dot(self.precond, X)
        X = np.reshape(X, self.shape)
        out = self.out
        l1_ratio = self.l1_ratio
        if self.copy:
            out = out.copy()
        # The gradient part
        out[0, :-1] = np.diff(X, axis=0)
        out[1, :, :-1] = np.diff(X, axis=1)
        out[2, ..., :-1] = np.diff(X, axis=2)
        out[:3] *= 1. - l1_ratio
        # The identity part:
        out[3] = X
        out[3] *= l1_ratio
        return out


class DivId(object):
    """The adjoint operator of GradientId
    """
    def __init__(self, l1_ratio, precond=None):
        self.l1_ratio = l1_ratio
        self.precond = precond

    def matvec(self, X):
        "From a (4, n_x, n_y, n_z) gradient + id return a flat adjoint"
        l1_ratio = self.l1_ratio
        divX = ((1. - l1_ratio) * div(X[:3]) + l1_ratio * X[3]).ravel()
        if self.precond is not None:
            divX = np.dot(self.precond.T, divX)
        return divX


class IdentityOperator(object):
    def __init__(self, l1_ratio=None, precond=None):
        pass

    def matvec(self, X):
        return X


def tv_l1(X, y, shape, alpha, l1_ratio, max_iter=300, verbose=False, init=None,
          tol=1e-5, precond=None, anisotropic=True, mask=None, mu=0,
          store_path=False):
    """Run Chambolle-Pock primal-dual scheme for TV+L1 deconvolution

    minimize 0.5 ||y - Xw||^2 + alpha ((1. - l1_ratio) * ||Dw||_1 + l1_ratio * ||w||_1) + 0.5 * mu * ||w||^2
        x
    """
    # The number of spatial features might be different than the number
    # of features in X, if we are using a mask
    n_spatial_features = np.prod(shape)
    if l1_ratio == 0.:  # pure TV
        D = Gradient(shape, precond=precond)
        DT = Div(precond=precond)
        _check_adjoint(n_spatial_features, D, DT, random_state=None)
        return chambolle_pock_l1(X, y, alpha, D, DT, max_iter=max_iter,
                                 verbose=verbose, init=init, tol=tol,
                                 mask=mask, mu=mu, store_path=store_path)
    elif l1_ratio == 1.:  # pure Lasso
        D = IdentityOperator()
        DT = IdentityOperator()
        _check_adjoint(n_spatial_features, D, DT, random_state=None)
        coef, dual_coef, pobj, path, times = chambolle_pock_l1(X, y, alpha, D,
                                        DT,
                                        max_iter=max_iter, verbose=verbose,
                                        init=init, tol=tol, mask=mask,
                                        mu=mu, store_path=store_path)
    else:
        D = GradientId(shape, l1_ratio, precond=precond)
        DT = DivId(l1_ratio, precond=precond)
        if precond is not None:
            X = np.dot(X, precond)
        _check_adjoint(n_spatial_features, D, DT, random_state=None)
        coef, dual_coef, pobj, path, times = chambolle_pock_l1(X, y, alpha, D,
                                        DT,
                                        max_iter=max_iter, verbose=verbose,
                                        init=init, tol=tol,
                                        anisotropic=anisotropic,
                                        mask=mask, mu=mu,
                                        store_path=store_path)
        if precond is not None:
            coef = np.dot(precond, coef)
    return coef, dual_coef, pobj, path, times


class TVL1Regression(LinearModel, RegressorMixin):
    """docstring for TVL1"""
    def __init__(self, shape=None, alpha=1., l1_ratio=0.5, max_iter=1000,
                 fit_intercept=True, normalize=False, copy_X=True,
                 verbose=False, warm_start=False, tol=1e-5,
                 anisotropic=False, mask=None, mu=0, scale_coef=False):
        self.shape = shape
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose
        self.warm_start = warm_start
        self.tol = tol
        self.anisotropic = anisotropic
        self.mask = mask
        self.mu = mu
        self.scale_coef = scale_coef

    def fit(self, X, y):
        X = array2d(X)
        y = np.asarray(y)
        shape = self.shape
        if shape is None:
            if self.mask is None:
                raise ValueError('[%s] Either a shape or a mask must be '
                                 'specified' % self.__class__.__name__)
            shape = self.mask.shape
        X, y, Xmean, ymean, Xstd = LinearModel._center_data(X, y,
                                                    self.fit_intercept,
                                                    self.normalize,
                                                    self.copy_X)
        if self.warm_start and hasattr(self, 'coef_'):
            init = self.coef_init_, self.dual_coef_
        else:
            init = None

        precond = None
        alpha = self.alpha

        mask = self.mask
        if mask is not None:
            mask = mask.ravel()

        alpha = alpha * X.shape[0]  # XXX : scale
        self.coef_init_, self.dual_coef_, self.pobj_, self.path_, \
            self.times_ = tv_l1(X, y, shape, alpha,
                               l1_ratio=self.l1_ratio, init=init,
                               max_iter=self.max_iter, verbose=self.verbose,
                               tol=self.tol, precond=precond,
                               anisotropic=self.anisotropic,
                               mask=mask, mu=self.mu, store_path=False)

        if self.scale_coef:
            if mask is None:
                y_pred = np.dot(X, self.coef_init_)
            else:
                y_pred = np.dot(X, self.coef_init_[mask])
            scaling = np.dot(y, y_pred) / linalg.norm(y_pred) ** 2
            self.coef_ = scaling * self.coef_init_
        else:
            self.coef_ = self.coef_init_

        self.coef_full_ = self.coef_.copy()
        if mask is not None:
            self.coef_ = self.coef_[mask]

        self._set_intercept(Xmean, ymean, Xstd)
        return self


def tvl1_path(X, y, shape=None, l1_ratio=0.5, eps=1e-3, alphas=None,
              fit_intercept=True, normalize=False, verbose=False,
              **params):
    X, y, X_mean, y_mean, X_std = center_data(X, y, fit_intercept,
                                              normalize, copy=False)
    n_samples, n_features = X.shape

    alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    params.pop('mode')
    params.pop('precompute')
    params.pop('X_std')
    params.pop('X_mean')
    params.pop('Xy')

    n_alphas = len(alphas)
    # models = []
    coefs = []
    for i, alpha in enumerate(alphas):
        model = TVL1Regression(shape=shape, alpha=alpha, l1_ratio=l1_ratio,
                               fit_intercept=False, warm_start=True)

        model.set_params(**params)
        model.fit(X, y)

        coefs.append(model.coef_)

        if verbose:
            if verbose > 2:
                print model
            elif verbose > 1:
                print 'Path: %03i out of %03i' % (i, n_alphas)
            else:
                sys.stderr.write('.')

        # if fit_intercept:
        #     model.fit_intercept = True
        #     model._set_intercept(X_mean, y_mean, X_std)
        # models.append(model)
    return alphas, np.array(coefs).T, None


class TVL1RegressionCV(LinearModel, RegressorMixin):
    def __init__(self, shape=None, alphas=None, l1_ratio=0.5,
            fit_intercept=True, normalize=False, max_iter=1000, tol=1e-4,
            cv=None, copy_X=True, n_jobs=1, verbose=False, debias=False,
            memory=Memory(None), anisotropic=False, mask=None,
            mu=0, scale_coef=False, mode='primal_dual'):
        self.shape = shape
        self.alphas = alphas
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.debias = debias
        self.memory = memory
        self.anisotropic = anisotropic
        self.mask = mask
        self.mu = mu
        self.scale_coef = scale_coef
        assert mode in ['primal_dual', 'fista']
        self.mode = mode

    def fit(self, X, y):
        X = atleast2d_or_csc(X, dtype=np.float64,
                             copy=self.copy_X and self.fit_intercept)
        # From now on X can be touched inplace
        y = np.asarray(y, dtype=np.float64)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (X.shape[0], y.shape[0]))

        # All LinearModelCV parameters except 'cv' are acceptable
        path_params = self.get_params()
        if 'l1_ratio' in path_params:
            l1_ratios = np.atleast_1d(path_params['l1_ratio'])
            # For the first path, we need to set l1_ratio
            path_params['l1_ratio'] = l1_ratios[0]
        else:
            l1_ratios = self.l1_ratios
        path_params.pop('cv', None)
        path_params.pop('n_jobs', None)
        path_params.pop('memory', None)
        path_params.pop('debias', None)
        # Useful for TVL1RegressionCVFast
        path_params.pop('cv_alpha', None)
        path_params['precompute'] = False

        if self.mode == 'fista':
            path_params.pop('mu', None)
            path_params.pop('anisotropic', None)

        # init cross-validation generator
        cv = check_cv(self.cv, X)

        # Compute path for all folds and compute MSE to get the best alpha
        folds = list(cv)
        best_mse = np.inf
        alphas = self.alphas
        all_mse_paths = list()
        path_func = self.memory.cache(tvl1_path)

        # We do a double for loop folded in one, in order to be able to
        # iterate in parallel on l1_ratio and folds
        for l1_ratio, mse_alphas in itertools.groupby(
                    Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                        delayed(_path_residuals)(X, y, train, test,
                                    path_func, path_params, l1_ratio=l1_ratio)
                            for l1_ratio in l1_ratios for train, test in folds
                    ), operator.itemgetter(1)):

            mse_alphas = [m[0] for m in mse_alphas]
            # alphas = [m[1] for m in mse_alphas]
            mse_alphas = np.array(mse_alphas)
            mse = np.mean(mse_alphas, axis=0)
            i_best_alpha = np.argmin(mse)
            this_best_mse = mse[i_best_alpha]

            all_mse_paths.append(mse_alphas.T)
            if this_best_mse < best_mse:
                best_mse = this_best_mse
                best_alpha = alphas[i_best_alpha]
                best_l1_ratio = l1_ratio

        path_params.update({'alphas': [best_alpha]})
        path_params.update({'l1_ratio': best_l1_ratio})
        self.l1_ratio_ = best_l1_ratio
        self.alpha_ = best_alpha

        # refit
        model = TVL1Regression()
        this_params = self.get_params()
        params = dict()
        for p in this_params:
            if p in model.get_params():
                params[p] = this_params[p]
        params['alpha'] = best_alpha
        params['l1_ratio'] = best_l1_ratio
        model.set_params(**params)
        model.fit(X, y)
        self.coef_full_ = model.coef_full_
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

        self.alphas_ = np.asarray(alphas)
        self.mse_path_ = all_mse_paths

        if self.debias:
            active_set = (np.abs(self.coef_)
                          > (1e-1 * np.max(np.abs(self.coef_))))
            ridge = linear_model.RidgeCV(alphas=alphas, cv=self.cv)
            ridge.fit(X[:, active_set], y)
            self.coef_[active_set] = ridge.coef_
            self.intercept_ = ridge.intercept_
        return self


class TVL1RegressionCVFast(TVL1RegressionCV):
    """ Decoupled cross_validation along l1-ratio and alphas
    """

    def __init__(self, shape=None, alphas=None, l1_ratio=0.5,
            fit_intercept=True, normalize=False, max_iter=1000, tol=1e-4,
            cv=None, cv_alpha=None, copy_X=True, n_jobs=1, verbose=False,
            debias=False, memory=Memory(None), anisotropic=False, mask=None,
            mu=0, scale_coef=False, mode='primal_dual', refit=True):
        self.shape = shape
        self.alphas = alphas
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.debias = debias
        self.memory = memory
        self.anisotropic = anisotropic
        self.mask = mask
        self.mu = mu
        self.scale_coef = scale_coef
        assert mode in ['primal_dual', 'fista']
        self.mode = mode
        self.cv_alpha = cv_alpha
        self.refit = refit


    def fit(self, X, y):
        # Completely hacky code

        # First select l1_ratio with alpha fixed
        alphas_orig = self.alphas
        # Take the geometric mean
        self.alphas = [np.exp(np.mean(np.log(self.alphas)))]
        refit_orig = self.refit
        self.refit = False
        TVL1RegressionCV.fit(self, X, y)
        self.alphas = alphas_orig
        self.refit = refit_orig

        # Now select alpha with l1_ratio fixed
        l1_ratios_orig = self.l1_ratio
        self.l1_ratio = self.l1_ratio_
        cv_orig = self.cv
        if self.cv_alpha is not None:
            self.cv = self.cv_alpha
        TVL1RegressionCV.fit(self, X, y)
        self.l1_ratio = l1_ratios_orig
        self.cv = cv_orig
        return self



###############################################################################
# Check that div is the adjoint of grad

def _check_adjoint(size, D, DT, random_state=None):
    # We need to check that <D x, y> = <x, DT y> for x and y random vectors
    random_state = check_random_state(random_state)

    x = np.random.normal(size=size)
    y = np.random.normal(size=D.matvec(x).shape)

    np.testing.assert_almost_equal(np.sum(D.matvec(x) * y),
                                np.sum(x * DT.matvec(y)))

if __name__ == '__main__':
    pl.close('all')
    results = dict()
    truth = list()

    grad = GradientId((10, 10, 10), .7)
    divergence = DivId(.7)
    _check_adjoint(size=10 * 10 * 10, D=grad, DT=divergence)

    ###########################################################################
    # Create data
    roi_size = 3
    if len(sys.argv) > 1:
        roi_size = int(sys.argv[1])

    n_samples, size, debias = 400, 12, False
    snr = 1

    X_train, X_test, y_train, y_test, snr, noise, coefs, size = \
            create_simulation_data(snr=5, n_samples=n_samples, size=size,
                                   roi_size=roi_size, random_state=42)

    coefs = np.reshape(coefs, [size, size, size])
    plot_slices(coefs, title="Ground truth")
    l1_ratios = np.linspace(0, 1, 11)
    alphas = np.logspace(-3, 0, 10)[::-1]

    mask = None

    l1_ratios = [0.05]
    alphas = [.05]

    # mem = Memory('/tmp')
    mem = Memory(None)

    if 1:
        clf = TVL1RegressionCV(shape=[size, size, size], alphas=alphas,
                    l1_ratio=l1_ratios, cv=3, n_jobs=1,
                    max_iter=1000, verbose=True, memory=mem,
                    debias=debias, anisotropic=False,
                    mask=mask, mu=0, scale_coef=True,
                    mode='primal_dual')
        t1 = time()
        clf.fit(X_train, y_train)
        elapsed_time = time() - t1

        pl.figure()
        pl.plot(np.log10(alphas), np.mean(clf.mse_path_, axis=2).T)
        pl.legend(["%1.2f" % l for l in l1_ratios], loc='upper left')
        pl.xlabel('log10(alpha)')
        pl.ylabel('MSE')
        print "alpha: %s -- l1_ratio: %s" % (clf.alpha_, clf.l1_ratio_)
    else:
        clf = TVL1Regression([size, size, size], alpha=.05, l1_ratio=0.05,
                            max_iter=5000, verbose=True, tol=1e-6,
                            anisotropic=False, mask=mask,
                            mu=0, scale_coef=True)

        t1 = time()
        if mask is None:
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train_masked, y_train)
        elapsed_time = time() - t1
        clf.alpha_ = clf.alpha  # HACK

        fig = pl.figure()
        pl.plot(clf.times_, clf.pobj_, label='Chambolle-Pock')
        pl.title('Cost')
        pl.xlabel('time (seconds)')
        pl.legend(loc='best')

        if clf.path_ is not None:
            distance = [np.sqrt(np.sum((clf.path_[-1] - w)**2))
                        for w in clf.path_[:-1]]
            distance = np.array(distance)
            pl.semilogy(distance + 1e-5)
            #pl.ylim(1e-2, 1e4)
            pl.title('Distance to minimizer')

    if mask is None:
        score = clf.score(X_test, y_test)
        mse = metrics.mean_squared_error(y_test, clf.predict(X_test))
        coef_ = np.reshape(clf.coef_, [size, size, size])
    else:
        score = clf.score(X_test[:, mask.ravel()], y_test)
        mse = metrics.mean_squared_error(y_test,
                        clf.predict(X_test[:, mask.ravel()]))
        coef_ = unmask(clf.coef_, mask)
        coef_ = np.reshape(coef_, [size, size, size])
    print "--- elapsed_time : %s" % elapsed_time
    print "--- score : %s" % score
    print "--- mse : %s" % mse

    #########################################################################
    # Plot results
    if hasattr(clf, 'l1_ratio_'):
        title = 'TV+L1: MSE %.3f, p=(%1.2f, %1.1f), time: %.1fs' % (
                                            mse, clf.alpha_,
                                            clf.l1_ratio_, elapsed_time)
    else:
        title = 'TV+L1: MSE %.3f, p=(%1.2f, %1.1f, time: %.1fs)' % (
                                            mse, clf.alpha_,
                                            clf.l1_ratio, elapsed_time)
    print title

    # We use the plot_slices function provided in the example to
    # plot the results
    plot_slices(coef_, title=title)

    ###########################################################################
    # Precision recall curves in identification
    f_score = feature_selection.f_regression(X_train, y_train)[0]

    title = 'F-score'
    plot_slices(np.reshape(f_score, [size, size, size]), title=title)
    pl.figure(-10, figsize=(6.2, 3.8))
    pl.clf()

    models = [('TVL1', np.abs(clf.coef_), 'g', '-'),
              ('F-test', f_score, 'k', '--')]

    for model_name, scores, color, linestyle in sorted(models):
        model_results = results.get(model_name, [])
        model_results.append(scores)
        results[model_name] = model_results
    truth.append((coefs != 0).ravel())

    coef_true = np.ravel(truth)
    for model_name, scores, color, linestyle in sorted(models):
        scores = np.ravel(results[model_name])
        precision, recall, thresholds = metrics.precision_recall_curve(
                                                (coef_true !=0), scores)
        roc_auc = metrics.auc(recall, precision)
        pl.plot(recall, precision,
                label='%s\n (area=%0.3f)' % (model_name, roc_auc),
                linewidth=3, color=color, linestyle=linestyle)

    pl.xlim([0.0, 1.03])
    pl.ylim([0.0, 1.03])
    pl.xlabel(r'Recall (i.e. sensitivity or TPR)', fontsize=16)
    pl.ylabel('Precision (i.e. PPV or 1 - FDR)', fontsize=16)
    pl.tight_layout(pad=.1)
    pl.subplots_adjust(right=.7)
    pl.legend(loc=(1, -.13), handletextpad=-.025,
              labelspacing=.78, frameon=False)
    pl.show()
