# emacs: at the end of the file
# ex: set sts=4 ts=4 sw=4 et:
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### #
import os

import numpy as np
import nibabel as nib
import scipy.io as sio

from pyale.dataset import import_sleuth
from pyale.meta import _compute_ale, _ale_to_p


def test_hist_bins():
    """
    """
    code_dir = os.path.dirname(__file__)
    raw_file = os.path.join(code_dir, 'data/nat.txt')
    true_file = os.path.join(code_dir, 'data/nat_histBins.mat')
    true = sio.loadmat(true_file)['histEdges'].squeeze()

    dataset = import_sleuth(raw_file)
    experiments = dataset.experiments

    max_poss_ale = 1.
    for exp in experiments:
        max_poss_ale *= (1. - np.max(exp.kernel))

    max_poss_ale = 1. - max_poss_ale
    hist_bins = np.round(np.arange(0, max_poss_ale+0.001, 0.0001), 4)
    assert np.array_equal(hist_bins, true)


def test_ma_hists():
    """
    """
    code_dir = os.path.dirname(__file__)
    par_dir = os.path.dirname(code_dir)
    raw_file = os.path.join(code_dir, 'data/nat.txt')
    template_file = os.path.join(par_dir, 'resources/Grey10.nii.gz')
    true_file = os.path.join(code_dir, 'data/nat_maHists.mat')
    true = sio.loadmat(true_file)['maHists']

    dataset = import_sleuth(raw_file)
    experiments = dataset.experiments

    info = nib.load(template_file)
    dims = info.shape

    max_poss_ale = 1
    for exp in experiments:
        max_poss_ale *= (1 - np.max(exp.kernel))

    max_poss_ale = 1 - max_poss_ale
    hist_bins = np.arange(0, max_poss_ale+0.001, 0.0001)

    # Compute ALE values
    template_arr = info.get_data().flatten()
    prior = np.where(template_arr!=0)[0]

    data = np.zeros(dims + np.array([30, 30, 30]))

    ma_hists = np.zeros((len(experiments), hist_bins.shape[0]))
    for i, exp in enumerate(experiments):
        ma_values = np.copy(data)
        for j_peak in range(exp.ijk.shape[0]):
            x = exp.ijk[j_peak, 0]
            y = exp.ijk[j_peak, 1]
            z = exp.ijk[j_peak, 2]
            ma_values[x:x+31, y:y+31, z:z+31] = np.maximum(ma_values[x:x+31,
                                                                     y:y+31,
                                                                     z:z+31],
                                                           exp.kernel)

        ma_values = ma_values[15:-15, 15:-15, 15:-15]
        ma_values = ma_values.flatten()
        brain_ma_values = ma_values[prior]
        n_zeros = len(np.where(brain_ma_values==0)[0])
        reduced_ma_values = brain_ma_values[brain_ma_values>0]

        # Remember that histogram uses bin edges (not centers), so it returns
        # a 1xhist_bins-1 array
        if hist_bins is not None:
            ma_hists[i, 0] = n_zeros
            ma_hists[i, 1:] = np.histogram(a=reduced_ma_values, bins=hist_bins,
                                           density=False)[0]

    assert np.array_equal(ma_hists, true)


def test_ale_values():
    """
    """
    code_dir = os.path.dirname(__file__)
    par_dir = os.path.dirname(code_dir)
    raw_file = os.path.join(code_dir, 'data/nat.txt')
    template_file = os.path.join(par_dir, 'resources/Grey10.nii.gz')
    true_file = os.path.join(code_dir, 'data/nat_ALE.nii')
    true_header = nib.load(true_file)
    true = np.array(true_header.get_data())
    true[np.isnan(true)] = 0

    dataset = import_sleuth(raw_file)
    experiments = dataset.experiments

    info = nib.load(template_file)
    dims = info.shape
    shape = dims + np.array([30, 30, 30])

    # Compute ALE values
    template_arr = info.get_data().flatten()
    prior = np.where(template_arr!=0)[0]

    ale_values, _ = _compute_ale(experiments, dims, shape, prior,
                                 hist_bins=None)

    ale_matrix = np.zeros(dims).flatten()
    ale_matrix[prior] = ale_values
    ale_matrix = ale_matrix.reshape(dims)

    assert np.array_equal(ale_matrix, true)


def test_null_distribution():
    """
    """
    code_dir = os.path.dirname(__file__)
    par_dir = os.path.dirname(code_dir)
    raw_file = os.path.join(code_dir, 'data/nat.txt')
    template_file = os.path.join(par_dir, 'resources/Grey10.nii.gz')
    true_file = os.path.join(code_dir, 'data/nat_nullDist.mat')
    true = np.squeeze(sio.loadmat(true_file)['cNULL'])

    dataset = import_sleuth(raw_file)
    experiments = dataset.experiments

    info = nib.load(template_file)
    dims = info.shape
    shape = dims + np.array([30, 30, 30])
    template_arr = info.get_data().flatten()
    prior = np.where(template_arr!=0)[0]

    max_poss_ale = 1.
    for exp in experiments:
        max_poss_ale *= (1. - np.max(exp.kernel))

    max_poss_ale = 1. - max_poss_ale
    hist_bins = np.round(np.arange(0, max_poss_ale+0.001, 0.0001), 4)

    _, null_distribution = _compute_ale(experiments, dims, shape, prior,
                                        hist_bins)

    # Truncate distributions to same length
    if len(null_distribution)<len(true):
        true = true[:len(null_distribution)]
    else:
        null_distribution = null_distribution[:len(true)]
    assert np.allclose(null_distribution, true)


def test__ale_to_p():
    """
    """
    code_dir = os.path.dirname(__file__)
    par_dir = os.path.dirname(code_dir)
    raw_file = os.path.join(code_dir, 'data/nat.txt')
    template_file = os.path.join(par_dir, 'resources/Grey10.nii.gz')
    true_file = os.path.join(code_dir, 'data/nat_z.nii')
    true_header = nib.load(true_file)
    true = true_header.get_data()
    true[np.isnan(true)] = 0

    dataset = import_sleuth(raw_file)
    experiments = dataset.experiments

    info = nib.load(template_file)
    dims = info.shape
    shape = dims + np.array([30, 30, 30])

    # Compute null distribution
    template_arr = info.get_data().flatten()
    prior = np.where(template_arr!=0)[0]

    max_poss_ale = 1.
    for exp in experiments:
        max_poss_ale *= (1. - np.max(exp.kernel))

    max_poss_ale = 1. - max_poss_ale
    hist_bins = np.round(np.arange(0, max_poss_ale+0.001, 0.0001), 4)

    ale_values, null_distribution = _compute_ale(experiments, dims, shape,
                                                 prior, hist_bins)

    p_values, z_values = _ale_to_p(ale_values, hist_bins, null_distribution)
    info = nib.load(template_file)
    dims = info.shape

    # Compute ALE values
    template_arr = info.get_data().flatten()
    prior = np.where(template_arr!=0)[0]

    z_matrix = np.zeros(dims).flatten()
    z_matrix[prior] = z_values
    z_matrix = z_matrix.reshape(dims)

    assert np.allclose(z_matrix, true)
