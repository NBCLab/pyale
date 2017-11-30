# emacs: at the end of the file
# ex: set sts=4 ts=4 sw=4 et:
"""
ALE-related meta-analysis tools.

TODO: Improve conventions.
A few in-function suffix conventions:
- _values: 1d array matching dimensions of prior (211590). Filled with
           meaningful values
- _vector: 1d array matching dimensions of prior. Not filled with values
           (probably boolean-like or labels)
- _matrix: 3d array matching dimensions of template image (91, 109, 91)
- _mat: nd array *not* matching dimensions of template image
- _arr: 1d array *not* matching dimensions of prior
"""
from __future__ import division, print_function
import os
import copy
from time import time
import multiprocessing as mp

import numpy as np
from scipy import ndimage
from scipy.special import ndtri

from .due import due, Doi
from .utils import round2, read_nifti, save_nifti, thresh_str, get_resource_path


@due.dcite(Doi('10.1002/hbm.20718'),
           description='Introduces the ALE algorithm.')
@due.dcite(Doi('10.1002/hbm.21186'),
           description='Modifies ALE algorithm to eliminate within-experiment '
                       'effects and generate MA maps based on subject group '
                       'instead of experiment.')
@due.dcite(Doi('10.1016/j.neuroimage.2011.09.017'),
           description='Modifies ALE algorithm to allow FWE correction and to '
                       'more quickly and accurately generate the null '
                       'distribution for significance testing.')
def ale(dataset, n_cores=1, voxel_thresh=0.001, clust_thresh=0.05,
        corr='FWE', n_iters=10000, verbose=True, plot_thresh=True, prefix='',
        template_file='Grey10.nii.gz'):
    """
    Perform activation likelihood estimation[1]_[2]_[3]_ meta-analysis on dataset.

    General steps:
    - Create ALE image
    - Create null distribution
    - Convert ALE to Z/p
    - Voxel-level threshold image
    - Perform iterations for FWE correction
    - Cluster-extent threshold image

    Parameters
    ----------
    dataset : ale.Dataset
        Dataset to analyze.

    voxel_thresh : float
        Uncorrected voxel-level threshold.

    clust_thresh : float
        Corrected threshold. Used for both voxel- and cluster-level FWE.

    corr : str
        Correction type. Currently supported: FWE.

    n_iters : int
        Number of iterations for correction. Default 10000

    verbose : bool
        If True, prints out status updates.

    prefix : str
        String prepended to default output filenames. May include path.

    Examples
    ----------


    References
    ----------
    .. [1] Eickhoff, S. B., Laird, A. R., Grefkes, C., Wang, L. E.,
           Zilles, K., & Fox, P. T. (2009). Coordinate-based activation likelihood
           estimation meta-analysis of neuroimaging data: A random-effects
           approach based on empirical estimates of spatial uncertainty.
           Human brain mapping, 30(9), 2907-2926.
    .. [2] Turkeltaub, P. E., Eickhoff, S. B., Laird, A. R., Fox, M.,
           Wiener, M., & Fox, P. (2012). Minimizing within-experiment and
           within-group effects in activation likelihood estimation
           meta-analyses. Human brain mapping, 33(1), 1-13.
    .. [3] Eickhoff, S. B., Bzdok, D., Laird, A. R., Kurth, F., & Fox, P. T.
           (2012). Activation likelihood estimation meta-analysis revisited.
           Neuroimage, 59(3), 2349-2361.

    """
    name = dataset.name
    experiments = dataset.experiments

    # Check path for template file
    if not os.path.dirname(template_file):
        template_file = os.path.join(get_resource_path(), template_file)

    max_cores = mp.cpu_count()
    if not 1 <= n_cores <= max_cores:
        print('Desired number of cores ({0}) outside range of acceptable values. '
              'Setting number of cores to max ({1}).'.format(n_cores, max_cores))
        n_cores = max_cores

    if prefix == '':
        prefix = name

    if os.path.basename(prefix) == '':
        prefix_sep = ''
    else:
        prefix_sep = '_'

    max_poss_ale = 1
    for exp in experiments:
        max_poss_ale *= (1 - np.max(exp.kernel))

    max_poss_ale = 1 - max_poss_ale
    hist_bins = np.round(np.arange(0, max_poss_ale+0.001, 0.0001), 4)

    # Compute ALE values
    template_data, affine = read_nifti(template_file)
    dims = template_data.shape
    template_arr = template_data.flatten()
    prior = np.where(template_arr != 0)[0]
    shape = dims + np.array([30, 30, 30])

    # Gray matter coordinates
    perm_ijk = np.where(template_data)
    perm_ijk = np.vstack(perm_ijk).transpose()

    ale_values, null_distribution = _compute_ale(experiments, dims, shape, prior, hist_bins)
    p_values, z_values = _ale_to_p(ale_values, hist_bins, null_distribution)

    # Write out unthresholded value images
    out_images = {'ale': ale_values,
                  'p': p_values,
                  'z': z_values,}

    for (f, vals) in out_images.items():
        mat = np.zeros(dims).ravel()
        mat[prior] = vals
        mat = mat.reshape(dims)

        thresh_suffix = '{0}.nii.gz'.format(f)
        filename = prefix + prefix_sep + thresh_suffix
        save_nifti(mat, filename, affine)
        if verbose:
            print('File {0} saved.'.format(filename))

    # Begin cluster-extent thresholding by thresholding matrix at cluster-
    # defining voxel-level threshold
    z_thresh = ndtri(1-voxel_thresh)
    sig_idx = np.where(z_values > z_thresh)[0]
    sig_vector = np.zeros(prior.shape)
    sig_vector[sig_idx] = 1

    sig_matrix = np.zeros(dims).ravel()
    sig_matrix[prior] = sig_vector
    sig_matrix = sig_matrix.reshape(dims)

    out_vector = np.zeros(prior.shape)
    out_vector[sig_idx] = z_values[sig_idx]

    out_matrix = np.zeros(dims).ravel()
    out_matrix[prior] = out_vector
    out_matrix = out_matrix.reshape(dims)

    thresh_suffix = 'thresh_{0}unc.nii.gz'.format(thresh_str(voxel_thresh))
    filename = prefix + prefix_sep + thresh_suffix
    save_nifti(out_matrix, filename, affine)
    if verbose:
        print('File {0} saved.'.format(filename))

    # Find number of voxels per cluster (includes 0, which is empty space in
    # the matrix)
    conn_mat = np.ones((3, 3, 3))  # 18 connectivity
    for i in [0, -1]:
        for j in [0, -1]:
            for k in [0, -1]:
                conn_mat[i, j, k] = 0
    labeled_matrix = ndimage.measurements.label(sig_matrix, conn_mat)[0]
    labeled_vector = labeled_matrix.flatten()
    labeled_vector = labeled_vector[prior]

    clust_sizes = [len(np.where(labeled_vector == val)[0]) for val in np.unique(labeled_vector)]
    clust_sizes = clust_sizes[1:]  # First is zeros

    ## Multiple comparisons correction
    if corr == 'FWE':
        if verbose:
            print('Performing FWE correction.')
        rd_experiments = copy.deepcopy(experiments)
        results = _FWE_correction(rd_experiments, perm_ijk, null_distribution, hist_bins,
                                  prior, dims, n_cores, voxel_thresh, n_iters)
        rd_max_ales, rd_clust_sizes = zip(*results)

        percentile = 100 * (1 - clust_thresh)

        # Generate plot of threshold convergence
        if plot_thresh:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style('whitegrid')

            plot_range = range(100, n_iters+1)
            cFWE_thresholds = np.empty(len(plot_range))
            vFWE_thresholds = np.empty(len(plot_range))
            for i, n in enumerate(plot_range):
                cFWE_thresholds[i] = np.percentile(rd_clust_sizes[:n], percentile)
                vFWE_thresholds[i] = np.percentile(rd_max_ales[:n], percentile)
            fig, axes = plt.subplots(2, sharex=True)
            axes[0].plot(plot_range, cFWE_thresholds, 'b')
            axes[0].set_ylabel('Minimum Cluster Size')
            axes[1].plot(plot_range, vFWE_thresholds, 'r')
            axes[1].set_ylabel('Minimum ALE Value')
            axes[1].set_xlabel('Number of FWE Iterations')
            fig.suptitle(os.path.basename(prefix), fontsize=20)
            fig.savefig(prefix+prefix_sep+'thresholds.png', dpi=400)

        ## Cluster-level FWE
        # Determine size of clusters in [1 - clust_thresh]th percentile (e.g. 95th)
        clust_size_thresh = np.percentile(rd_clust_sizes, percentile)

        sig_values = np.zeros(prior.shape)
        for i, clust_size in enumerate(clust_sizes):
            if clust_size >= clust_size_thresh:
                clust_idx = np.where(labeled_vector == i+1)[0]
                sig_values[clust_idx] = z_values[clust_idx]
        sig_matrix = np.zeros(dims).ravel()
        sig_matrix[prior] = sig_values
        sig_matrix = sig_matrix.reshape(dims)

        thresh_suffix = 'thresh_{0}cFWE_{1}unc.nii.gz'.format(thresh_str(clust_thresh),
                                                              thresh_str(voxel_thresh))
        filename = prefix + prefix_sep + thresh_suffix
        save_nifti(sig_matrix, filename, affine)
        if verbose:
            print('File {0} saved.'.format(filename))

        ## Voxel-level FWE
        # Determine ALE values in [1 - clust_thresh]th percentile (e.g. 95th)
        ale_value_thresh = np.percentile(rd_max_ales, percentile)

        sig_idx = ale_values >= ale_value_thresh
        sig_values = z_values * sig_idx
        sig_matrix = np.zeros(dims).ravel()
        sig_matrix[prior] = sig_values
        sig_matrix = sig_matrix.reshape(dims)

        thresh_suffix = 'thresh_{0}vFWE.nii.gz'.format(thresh_str(clust_thresh))
        filename = prefix + prefix_sep + thresh_suffix
        save_nifti(sig_matrix, filename, affine)
        if verbose:
            print('File {0} saved.'.format(filename))
    elif corr == 'FDR':
        print('Performing false discovery rate correction.')
        _FDR_correction()
    elif corr == 'TFCE':
        print('Performing threshold-free cluster enhancement.')
        _TFCE_correction()
    else:
        raise Exception('Unknown correction option: "{0}".'.format(corr))


@due.dcite(Doi('10.1016/j.neuroimage.2014.06.007'),
           description='Introduces the specific co-activation likelihood '
                       'estimation (SCALE) algorithm.')
def scale(dataset, n_cores=1, voxel_thresh=0.001, n_iters=2500, verbose=True,
          prefix='', database_file='grey_matter_ijk.txt',
          template_file='Grey10.nii.gz'):
    """
    Perform specific coactivation likelihood estimation[1]_ meta-analysis on dataset.

    Parameters
    ----------
    dataset : ale.Dataset
        Dataset to analyze.

    voxel_thresh : float
        Uncorrected voxel-level threshold.

    n_iters : int
        Number of iterations for correction. Default 2500

    verbose : bool
        If True, prints out status updates.

    prefix : str
        String prepended to default output filenames. May include path.

    database_file : str
        Tab-delimited file of coordinates from database. Voxels are rows and
        i, j, k (meaning matrix-space) values are the three columnns.

    Examples
    ----------


    References
    ----------
    .. [1] Langner, R., Rottschy, C., Laird, A. R., Fox, P. T., &
           Eickhoff, S. B. (2014). Meta-analytic connectivity modeling
           revisited: controlling for activation base rates.
           NeuroImage, 99, 559-570.

    """
    name = dataset.name
    experiments = dataset.experiments

    # Check paths for input files
    if not os.path.dirname(template_file):
        template_file = os.path.join(get_resource_path(), template_file)

    if not os.path.dirname(database_file):
        database_file = os.path.join(get_resource_path(), database_file)

    max_cores = mp.cpu_count()
    if not 1 <= n_cores <= max_cores:
        print('Desired number of cores ({0}) outside range of acceptable values. '
              'Setting number of cores to max ({1}).'.format(n_cores, max_cores))
        n_cores = max_cores

    if prefix == '':
        prefix = name

    if os.path.basename(prefix) == '':
        prefix_sep = ''
    else:
        prefix_sep = '_'

    max_poss_ale = 1
    for exp in experiments:
        max_poss_ale *= (1 - np.max(exp.kernel))

    max_poss_ale = 1 - max_poss_ale
    hist_bins = np.round(np.arange(0, max_poss_ale+0.001, 0.0001), 4)

    # Compute ALE values
    template_data, affine = read_nifti(template_file)
    dims = template_data.shape
    template_arr = template_data.flatten()
    prior = np.where(template_arr != 0)[0]
    shape = dims + np.array([30, 30, 30])

    database_ijk = np.loadtxt(database_file, dtype=int)
    if len(database_ijk.shape) != 2 or database_ijk.shape[-1] != 3:
        raise Exception('Database coordinates not in voxelsX3 shape.')
    elif np.any(database_ijk < 0):
        raise Exception('Negative value(s) detected. Database coordinates must '
                        'be in matrix space.')
    elif not np.all(np.equal(np.mod(database_ijk, 1), 0)):  # pylint: disable=no-member
        raise Exception('Float(s) detected. Database coordinates must all be '
                        'integers.')

    ale_values, _ = _compute_ale(experiments, dims, shape, prior)

    n_foci = np.sum([exp.ijk.shape[0] for exp in experiments])
    np.random.seed(0)  # pylint: disable=no-member
    rand_idx = np.random.choice(database_ijk.shape[0], size=(n_foci, n_iters))  # pylint: disable=no-member
    rand_ijk = database_ijk[rand_idx, :]

    # Define parameters
    iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
    shapes = [shape] * n_iters
    exp_list = [experiments] * n_iters
    priors = [prior] * n_iters
    dims_arrs = [dims] * n_iters
    iter_nums = range(1, n_iters+1)
    start = [time()] * n_iters

    params = zip(exp_list, iter_ijks, priors, dims_arrs, shapes, start, iter_nums)
    pool = mp.Pool(n_cores)
    scale_values = pool.map(_perm_scale, params)
    pool.close()
    scale_values = np.stack(scale_values)

    p_values, z_values = _scale_to_p(ale_values, scale_values, hist_bins)

    # Write out unthresholded value images
    out_images = {'ale': ale_values,
                  'p': p_values,
                  'z': z_values,}

    for (f, vals) in out_images.items():
        mat = np.zeros(dims).ravel()
        mat[prior] = vals
        mat = mat.reshape(dims)

        thresh_suffix = '{0}.nii.gz'.format(f)
        filename = prefix + prefix_sep + thresh_suffix
        save_nifti(mat, filename, affine)
        if verbose:
            print('File {0} saved.'.format(filename))

    # Perform voxel-level thresholding
    z_thresh = ndtri(1-voxel_thresh)
    sig_idx = np.where(z_values > z_thresh)[0]

    out_vector = np.zeros(prior.shape)
    out_vector[sig_idx] = z_values[sig_idx]

    out_matrix = np.zeros(dims).ravel()
    out_matrix[prior] = out_vector
    out_matrix = out_matrix.reshape(dims)

    thresh_suffix = 'thresh_{0}unc.nii.gz'.format(thresh_str(voxel_thresh))
    filename = prefix + prefix_sep + thresh_suffix
    save_nifti(out_matrix, filename, affine)
    if verbose:
        print('File {0} saved.'.format(filename))


def compute_ma(shape, ijk, kernel):
    """
    Generate modeled activation (MA) maps.
    """
    ma_values = np.zeros(shape)
    for j_peak in range(ijk.shape[0]):
        i = ijk[j_peak, 0]
        j = ijk[j_peak, 1]
        k = ijk[j_peak, 2]
        ma_values[i:i+31, j:j+31, k:k+31] = np.maximum(ma_values[i:i+31,  # pylint: disable=no-member
                                                                 j:j+31,
                                                                 k:k+31],
                                                       kernel)

    ma_values = ma_values[15:-15, 15:-15, 15:-15]
    ma_values = ma_values.ravel()
    return ma_values


def _compute_ale(experiments, dims, shape, prior, hist_bins=None):
    """
    Generate ALE-value array and null distribution from list of experiments.
    For ALEs on the original dataset, computes the null distribution.
    For permutation ALEs and all SCALEs, just computes ALE values.

    Returns masked array of ALE values and 1XnBins null distribution.
    """
    ale_values = np.ones(dims).ravel()
    if hist_bins is not None:
        ma_hists = np.zeros((len(experiments), hist_bins.shape[0]))
    else:
        ma_hists = None

    for i, exp in enumerate(experiments):
        ma_values = compute_ma(shape, exp.ijk, exp.kernel)

        # Remember that histogram uses bin edges (not centers), so it returns
        # a 1xhist_bins-1 array
        if hist_bins is not None:
            brain_ma_values = ma_values[prior]
            n_zeros = len(np.where(brain_ma_values == 0)[0])
            reduced_ma_values = brain_ma_values[brain_ma_values > 0]
            ma_hists[i, 0] = n_zeros
            ma_hists[i, 1:] = np.histogram(a=reduced_ma_values, bins=hist_bins,
                                           density=False)[0]
        ale_values *= (1. - ma_values)

    ale_values = 1 - ale_values
    ale_values = ale_values[prior]

    if hist_bins is not None:
        null_distribution = _get_null(hist_bins, ma_hists)
    else:
        null_distribution = None

    return ale_values, null_distribution


def _get_null(hist_bins, ma_hists):
    """
    Compute ALE null distribution.
    """
    # Inverse of step size in histBins (0.0001) = 10000
    step = 1 / np.mean(np.diff(hist_bins))

    # Null distribution to convert ALE to p-values.
    ale_hist = ma_hists[0, :]
    for i_exp in range(1, ma_hists.shape[0]):
        temp_hist = np.copy(ale_hist)
        ma_hist = np.copy(ma_hists[i_exp, :])

        # Find histogram bins with nonzero values for each histogram.
        ale_idx = np.where(temp_hist > 0)[0]
        exp_idx = np.where(ma_hist > 0)[0]

        # Normalize histograms.
        temp_hist /= np.sum(temp_hist)
        ma_hist /= np.sum(ma_hist)

        # Perform weighted convolution of histograms.
        ale_hist = np.zeros(hist_bins.shape[0])
        for j_idx in exp_idx:
            # Compute probabilities of observing each ALE value in histBins
            # by randomly combining maps represented by maHist and aleHist.
            # Add observed probabilities to corresponding bins in ALE
            # histogram.
            probabilities = ma_hist[j_idx] * temp_hist[ale_idx]
            ale_scores = 1 - (1 - hist_bins[j_idx]) * (1 - hist_bins[ale_idx])
            score_idx = np.floor(ale_scores * step).astype(int)
            np.add.at(ale_hist, score_idx, probabilities)

    # Convert aleHist into null distribution. The value in each bin
    # represents the probability of finding an ALE value (stored in
    # histBins) of that value or lower.
    last_used = np.where(ale_hist > 0)[0][-1]
    null_distribution = ale_hist[:last_used+1] / np.sum(ale_hist)
    null_distribution = np.cumsum(null_distribution[::-1])[::-1]
    null_distribution /= np.max(null_distribution)
    return null_distribution


def _ale_to_p(ale_values, hist_bins, null_distribution):
    """
    Compute p- and z-values.
    """
    eps = np.spacing(1)  # pylint: disable=no-member
    step = 1 / np.mean(np.diff(hist_bins))

    # Determine p- and z-values from ALE values and null distribution.
    p_values = np.ones(ale_values.shape)

    idx = np.where(ale_values > 0)[0]
    ale_bins = round2(ale_values[idx] * step)
    p_values[idx] = null_distribution[ale_bins]

    # NOTE: ndtri gives slightly different results from spm_invNcdf, so
    # z-values will differ between languages.
    z_values = ndtri(1-p_values)
    z_values[p_values < eps] = ndtri(1-eps) + (ale_values[p_values < eps] * 2)
    z_values[z_values < 0] = 0

    return p_values, z_values


def _perm_ale(params):
    """
    Run a single random permutation of a dataset. Does the shared work between
    vFWE and cFWE.
    """
    exps, ijk, null_dist, hist_bins, prior, dims, shape, voxel_thresh, start, iter_num = params
    ijk = np.squeeze(ijk)

    # Assign random GM coordinates to experiments
    start_idx = 0
    for exp in exps:
        n_peaks = exp.ijk.shape[0]
        end_idx = start_idx + n_peaks
        exp.ijk = ijk[start_idx:end_idx]
        start_idx = end_idx
    ale_values, _ = _compute_ale(exps, dims, shape, prior, hist_bins=None)
    _, z_values = _ale_to_p(ale_values, hist_bins, null_dist)
    rd_max_ale = np.max(ale_values)

    # Begin cluster-extent thresholding by thresholding matrix at cluster-
    # defining voxel-level threshold
    z_thresh = ndtri(1-voxel_thresh)

    sig_idx = np.where(z_values > z_thresh)[0]
    if sig_idx.size:
        sig_vector = np.zeros(prior.shape)
        sig_vector[sig_idx] = 1
        sig_matrix = np.zeros(dims).ravel()
        sig_matrix[prior] = sig_vector
        sig_matrix = sig_matrix.reshape(dims)

        # Find number of voxels per cluster (includes 0, which is empty space
        # in the matrix)
        conn_mat = np.ones((3, 3, 3))  # 18 connectivity
        for i in [0, -1]:
            for j in [0, -1]:
                for k in [0, -1]:
                    conn_mat[i, j, k] = 0
        labeled_matrix = ndimage.measurements.label(sig_matrix, conn_mat)[0]
        labeled_vector = labeled_matrix.ravel()
        labeled_vector = labeled_vector[prior]

        clust_sizes = [len(np.where(labeled_vector == val)[0]) for val in \
                       np.unique(labeled_vector)]
        clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
        rd_clust_size = np.max(clust_sizes)
    else:
        rd_clust_size = 0

    if iter_num % 1000 == 0:
        elapsed = (time() - start) / 60.
        print('Iteration {0} completed after {1} mins.'.format(iter_num,
                                                               elapsed))
    return rd_max_ale, rd_clust_size


def _FWE_correction(experiments, ijk, null_distribution, hist_bins, prior,
                    dims, n_cores, voxel_thresh=0.001, n_iters=100):
    """
    Performs both cluster-level (cFWE) and voxel-level (vFWE) correction.
    """
    n_foci = np.sum([exp.ijk.shape[0] for exp in experiments])
    np.random.seed(0)  # pylint: disable=no-member
    rand_idx = np.random.choice(ijk.shape[0], size=(n_foci, n_iters))  # pylint: disable=no-member
    rand_ijk = ijk[rand_idx, :]
    shape = dims + np.array([30, 30, 30])

    # Define parameters
    iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)
    shapes = [shape] * n_iters
    exp_list = [experiments] * n_iters
    null_distributions = [null_distribution] * n_iters
    priors = [prior] * n_iters
    dims_arrs = [dims] * n_iters
    hist_bin_arrs = [hist_bins] * n_iters
    voxel_thresholds = [voxel_thresh] * n_iters
    iter_nums = range(1, n_iters+1)
    start = [time()] * n_iters

    params = zip(exp_list, iter_ijks, null_distributions, hist_bin_arrs,
                 priors, dims_arrs, shapes, voxel_thresholds, start, iter_nums)
    pool = mp.Pool(n_cores)
    max_clusts = pool.map(_perm_ale, params)
    pool.close()

    return max_clusts


def _FDR_correction():
    """
    False discovery rate correction. Based on the findings of Eickhoff et al.
    (2016), this correction method should not be used.
    """
    raise Exception('Not currently implemented.')


def _TFCE_correction():
    """
    Threshold-free cluster enhancement multiple comparisons correction.
    """
    raise Exception('Not currently implemented.')


def _make_hist(oned_arr, hist_bins):
    hist_ = np.histogram(a=oned_arr, bins=hist_bins,
                         range=(np.min(hist_bins), np.max(hist_bins)),
                         density=False)[0]
    return hist_


def _scale_to_p(ale_values, scale_values, hist_bins):
    """
    Compute p- and z-values.
    """
    eps = np.spacing(1)  # pylint: disable=no-member
    step = 1 / np.mean(np.diff(hist_bins))

    scale_zeros = scale_values == 0
    n_zeros = np.sum(scale_zeros, axis=0)
    scale_values[scale_values == 0] = np.nan
    scale_hists = np.zeros(((len(hist_bins),) + n_zeros.shape))
    scale_hists[0, :] = n_zeros
    scale_hists[1:, :] = np.apply_along_axis(_make_hist, 0, scale_values,
                                             hist_bins=hist_bins)

    # Convert voxel-wise histograms to voxel-wise null distributions.
    null_distribution = scale_hists / np.sum(scale_hists, axis=0)
    null_distribution = np.cumsum(null_distribution[::-1, :], axis=0)[::-1, :]
    null_distribution /= np.max(null_distribution, axis=0)

    # Get the hist_bins associated with each voxel's ale value, in order to get
    # the p-value from the associated bin in the null distribution.
    n_bins = len(hist_bins)
    ale_bins = round2(ale_values * step).astype(int)
    ale_bins[ale_bins > n_bins] = n_bins

    # Get p-values by getting the ale_bin-th value in null_distribution
    # per voxel.
    p_values = np.empty_like(ale_bins).astype(float)
    for i,(x, y) in enumerate(zip(null_distribution.transpose(), ale_bins)):
        p_values[i] = x[y]

    z_values = ndtri(1-p_values)
    z_values[p_values < eps] = ndtri(1-eps) + (ale_values[p_values < eps] * 2)
    z_values[z_values < 0] = 0

    return p_values, z_values


def _perm_scale(params):
    """
    Run a single random SCALE permutation of a dataset.
    """
    exps, ijk, prior, dims, shape, start, iter_num = params
    ijk = np.squeeze(ijk)

    # Assign random GM coordinates to experiments
    start_idx = 0
    for exp in exps:
        n_peaks = exp.ijk.shape[0]
        end_idx = start_idx + n_peaks
        exp.ijk = ijk[start_idx:end_idx]
        start_idx = end_idx
    ale_values, _ = _compute_ale(exps, dims, shape, prior)

    if iter_num % 250 == 0:
        elapsed = (time() - start) / 60.
        print('Iteration {0} completed after {1} mins.'.format(iter_num, elapsed))
    return ale_values
