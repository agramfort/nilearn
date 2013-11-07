import numpy as np
import os
from scipy import ndimage

import nibabel as nib
from nipy.labs import as_volume_img
from nipy.labs.datasets import VolumeImg

#### podrack data set
def down_sample(filename):
    nim = as_volume_img(filename)
    n_x, n_y, n_z, n_t = nim._data.shape
    v_x, v_y, v_z, _ = np.diag(nim.affine)
    this_mask = VolumeImg(np.all(nim._data != 0,
                          axis=-1).astype(np.float), nim.affine, None)
    nim2 = nim.as_volume_img(affine=np.diag((4, 4, 4)),
                             shape=[n_x*v_x/4, n_y*v_y/4, n_z*v_z/4])
    this_mask = this_mask.as_volume_img(affine=np.diag((4, 4, 4)),
                             shape=[n_x*v_x/4, n_y*v_y/4, n_z*v_z/4])
    return nim2, this_mask


def load_subject_poldrack(subject_id, realigned=False, smooth=0):
    if realigned:
        betas_fname = 'nilearn_data/Jimura_Poldrack_2012_zmaps/gain_realigned/sub0%02d_zmaps.nii.gz' % subject_id
        img = nib.load(betas_fname)
        X = img.get_data()
        affine = img.get_affine()
        finite_mask = np.all(np.isfinite(X), axis=-1)
        mask = np.logical_and(np.all(X != 0, axis=-1),
                              finite_mask)
        if smooth:
            for i in range(X.shape[-1]):
                X[..., i] = ndimage.gaussian_filter(X[..., i], smooth)
            X[np.logical_not(finite_mask)] = np.nan
    else:
        betas_fname = 'nilearn_data/Jimura_Poldrack_2012_zmaps/gain/sub0%02d_zmaps.nii.gz' % subject_id
        img, mask = down_sample(betas_fname)

        X = img._data
        affine = img.affine
        if smooth:
            for i in range(X.shape[-1]):
                X[..., i] = ndimage.gaussian_filter(X[..., i], smooth)
        mask = mask._data
    y = np.array([np.arange(1, 9)] * 6).ravel()

    assert len(y) == 48
    assert len(y) == X.shape[-1]
    return X, y, mask, affine


poldrack_subjects = np.arange(1, 17)
#poldrack_subjects = set(poldrack_subjects).difference((1, 3, 7, 11))

def load_gain_poldrack(realigned=True, smooth=0):
    X = []
    y = []
    subject = []
    mask = []
    for i in poldrack_subjects:
        X_, y_, this_mask, affine = load_subject_poldrack(i,
                            realigned=realigned, smooth=smooth)
        X_ -= X_.mean(axis=-1)[..., np.newaxis]
        std = X_.std(axis=-1)
        std[std==0] = 1
        X_ /= std[..., np.newaxis]
        X.append(X_)
        y.extend(y_)
        subject.extend(len(y_) * [i,])
        mask.append(this_mask)
    X = np.concatenate(X, axis=-1)
    mask = np.sum(mask, axis=0) > .5*len(poldrack_subjects)
    mask = np.logical_and(mask, np.all(np.isfinite(X), axis=-1))
    return X[mask, :].T, np.array(y), np.array(subject), mask, affine

# if __name__ == '__main__':
#     mask = None
#     for i in poldrack_subjects:
#         fname = 'Jimura_Poldrack_2012_zmaps/gain/sub0%02d_zmaps.nii.gz' % i
#         img = nib.load(fname)
#         affine = img.get_affine()
#         this_mask = np.all(img.get_data() != 0, axis=-1).astype(np.float)
#         img = nib.Nifti1Image(this_mask, affine)
#         nib.save(img, os.path.join('Jimura_Poldrack_2012_zmaps/masks',
#                         os.path.basename(fname)))
#         if mask is None:
#             mask = this_mask
#         else:
#             mask += this_mask
#     mask /= mask.max()
#     img = nib.Nifti1Image((mask > .5).astype(np.int), affine)
#     nib.save(img, 'Jimura_Poldrack_2012_zmaps/mask.nii')


if __name__ == '__main__':
    X, y, subject, mask, affine = load_gain_poldrack(realigned=True, smooth=0)

    from sklearn.feature_selection import f_regression

    F, pv = f_regression(X, y)
    data = np.zeros(mask.shape, dtype=np.float32)
    data[mask] = F.astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, 'anova_poldrack.nii')
