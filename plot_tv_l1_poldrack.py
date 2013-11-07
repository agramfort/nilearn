import os
from time import time
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
from joblib import Memory
from sklearn import cross_validation, linear_model
from sklearn.feature_selection import f_regression
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR

from load_poldrack import load_gain_poldrack
from plot_tv_l1 import TVL1Regression, TVL1RegressionCV

n_jobs = min(multiprocessing.cpu_count(), 16)

if n_jobs < 16:
    mem = Memory(cachedir=os.path.expanduser('.'), verbose=3)
else:
    mem = Memory(cachedir='/scratch/gramfort', verbose=3)

n_jobs = 1

plt.close('all')

X, y, subjects, mask, affine = mem.cache(load_gain_poldrack)(realigned=True,
            smooth=0)

if 0:
    F, pv = f_regression(X, y)
    data = np.zeros(mask.shape, dtype=np.float32)
    data[mask] = F.astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, 'anova_poldrack.nii')


nx, ny, nz = mask.shape
data = np.zeros((len(X), nx, ny, nz), dtype=np.float32)
data[:, mask] = X
ymin, ymax = 33, 35
# Full brain
ymin = ymax = None
ymin, ymax = 32, 36
# ymin, ymax = 30, 38
if ymin is not None:
    mask[:, :ymin] = False
if ymax is not None:
    mask[:, ymax:] = False
submask = mask[:, ymin:ymax, :]
X_full = data[:, :, ymin:ymax, :].copy()
del data

X_full = X_full.reshape(len(X), -1)

if 0:
    # Fit an SVR, just to show it gives crap
    svr = SVR(kernel='linear')
    clf = GridSearchCV(estimator=svr,
                       param_grid=dict(C=np.logspace(-6, 0, 7)),
                       cv=7, verbose=10, n_jobs=-1,
                       scoring='mse')
    clf.fit(X_full[:, submask.ravel()], y)

    coef_data = np.zeros((nx, ny, nz), dtype=np.float32)
    coef_data[submask] = clf.best_estimator_.coef_.ravel()
    img = nib.Nifti1Image(coef_data, affine)
    nib.save(img, 'svr_poldrack_coef.nii')

if 0:
    # SVR with Z-score under permutation, using the C param selected by CV
    svr = SVR(kernel='linear', C=0.001)
    from permute_clf import coef_variance_permutation
    coef_var = coef_variance_permutation(svr, X, y,
                                         n_jobs=-1, verbose=10)
    svr_z_score = svr.fit(X, y).coef_ / np.sqrt(coef_var)
    coef_data = np.zeros((nx, ny, nz), dtype=np.float32)
    coef_data[submask] = svr_z_score.ravel()
    img = nib.Nifti1Image(coef_data, affine)
    nib.save(img, 'svr_z_poldrack_coef.nii')

if 0:
    # Fit a Ridge
    ridge = linear_model.RidgeCV()
    ridge.fit(X_full[:, submask.ravel()], y)

    coef_data = np.zeros((nx, ny, nz), dtype=np.float32)
    coef_data[submask] = ridge.coef_.ravel()
    img = nib.Nifti1Image(coef_data, affine)
    nib.save(img, 'ridge_gcv_poldrack_coef.nii')

if 0:
    # An SVR C=1
    svr = SVR(kernel='linear')
    svr.fit(X_full[:, submask.ravel()], y)

    coef_data = np.zeros((nx, ny, nz), dtype=np.float32)
    coef_data[submask] = svr.coef_.ravel()
    img = nib.Nifti1Image(coef_data, affine)
    nib.save(img, 'svr_c=1_poldrack_coef.nii')

if 0:
    # Fit an SVR + Anova, as a reference method
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectPercentile
    anova = SelectPercentile(score_func=mem.cache(f_regression),
                             percentile=20)

    svr = SVR(kernel='linear')
    anova_svr = Pipeline([('anova', anova), ('svr', svr)])
    anova_svr_cv = GridSearchCV(estimator=anova_svr,
                       param_grid=dict(svr__C=np.logspace(-4, 4, 10)),
                       cv=7, verbose=10, n_jobs=-1,
                       scoring='mse')
    anova_svr_cv.fit(X_full[:, submask.ravel()], y)

if 0:
    # Compare to searchlight
    from nisl.searchlight import SearchLight

    # The radius is the one of the Searchlight sphere that will scan
    # the volume
    searchlight = SearchLight(
                        estimator=SVR(kernel='linear'),
                        mask=submask,
                        radius=2, n_jobs=-1, verbose=10, cv=7)

    t0 = time()
    searchlight.fit(X_full[:, submask.ravel()], y)
    searchlight_time = time() - t0
    print('Searchlight: time: %.1fs' % searchlight_time)
    coef_data = searchlight.scores_.copy()
    img = nib.Nifti1Image(coef_data, affine)
    nib.save(img, 'searchlight_poldrack_coef.nii')


# Scale
# X_full -= X_full.mean(0)
# X_full /= X_full.std(0)
# X_full[np.isnan(X_full)] = 0.0
anisotropic = False

if 0:
    clf = TVL1Regression(submask.shape, alpha=0.1, l1_ratio=0.3,
                                max_iter=1000, verbose=True,
                                warm_start=True, tol=1e-8,
                                anisotropic=anisotropic,
                                mask=submask, mu=0,
                                scale_coef=True)

else:
    l1_ratios = np.linspace(0.1, .9, 9)
    # l1_ratios = [0.3]
    # alphas = np.logspace(-3, 3, 10)[::-1]  # ok
    alphas = np.logspace(-2, 0, 10)[::-1]
    # alphas = [1, 0.2, 0.1]
    alphas = [0.1]
    clf = TVL1RegressionCV(shape=submask.shape, alphas=alphas,
                           l1_ratio=l1_ratios, cv=5, n_jobs=n_jobs,
                           max_iter=500, verbose=10, memory=mem,
                           anisotropic=anisotropic,
                           mask=submask, mu=0,
                           scale_coef=True)

t1 = time()
clf.fit(X_full[:, submask.ravel()], y)
elapsed_time = time() - t1
print "elapsed time: %03is " % elapsed_time
coef_slice = np.zeros(submask.shape)
coef_slice[submask] = clf.coef_

if hasattr(clf, 'pobj_'):
    plt.figure()
    plt.plot(clf.pobj_)
    plt.axhline(clf.pobj_[0], color='k')

if hasattr(clf, 'mse_path_'):
    plt.figure()
    for k in range(len(clf.mse_path_)):
        plt.errorbar(np.log10(alphas), np.mean(clf.mse_path_, axis=2)[k, :],
                    np.std(clf.mse_path_, axis=2)[k, :])

    plt.legend(["%1.2f" % l for l in l1_ratios], loc='upper left')
    plt.xlabel('log10(alpha)')
    plt.ylabel('MSE')
    print "alpha: %s -- l1_ratio: %s" % (clf.alpha_, clf.l1_ratio_)

vmin, vmax = np.min(coef_slice), np.max(coef_slice)
vmax = max(abs(vmin), abs(vmax))
vmin = -vmax

#for k in range(coef_slice.shape[1]):
for k in range(2):
    plt.figure()
    if ymin is None:
        k += 33
    plt.imshow(coef_slice[:, k, :].T, interpolation='nearest',
              origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    if ymin is None:
        plt.contour(mask[:, k, :].T, 1, colors='k', linewidth=3)
    else:
        plt.contour(submask[:, k, :].T, 1, colors='k', linewidth=3)

plt.show()
