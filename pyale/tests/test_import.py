# emacs: at the end of the file
# ex: set sts=4 ts=4 sw=4 et:
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### #
import os

import numpy as np
import nibabel as nib
import scipy.io as sio

from pyale.utils import (round2, tal2mni, mni2tal, vox2mm, mm2vox, thresh_str,
                         read_nifti, save_nifti)
from pyale.utils import mem_smooth_64bit as smooth


def test_kernel():
    """
    Compare kernels generated with Python to those generated with SPM function.
    """
    code_dir = os.path.dirname(__file__)
    par_dir = os.path.dirname(code_dir)
    info = nib.load(os.path.join(par_dir, 'resources/ICBM152_2009c_2mm.nii.gz'))

    LV = sio.loadmat(os.path.join(code_dir, 'data/kernels.mat'))
    kernels = LV['kernels']

    # Assuming 5.7 mm ED between templates.
    uncertain_templates = (5.7/(2.*np.sqrt(2./np.pi)) * np.sqrt(8.*np.log(2.)))

    ns = range(1, 201)  # Test across sample sizes 1-50.
    test_kernels = np.zeros((31, 31, 31, len(ns)))
    for i, n in enumerate(ns):
        data = np.zeros((31, 31, 31))
        data[15, 15, 15] = 1.

        # Assuming 11.6 mm ED between matching points.
        uncertain_subjects = (11.6/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))) / np.sqrt(n)
        fwhm = np.sqrt(uncertain_subjects**2 + uncertain_templates**2)
        test_kernels[:, :, :, i] = smooth(data, fwhm, info)

    assert np.allclose(kernels, test_kernels)


def test_round2():
    """
    Check that round2 operates as expected.
    """
    test = np.array([1., 1.49, 1.5, 1.51, 2.])
    true = np.array([1., 1., 2., 2., 2.])
    assert np.all(round2(test)==true)


def test_tal2mni():
    """
    TODO: Get converted coords from official site.
    """
    test = np.array([[-44, 31, 27],
                     [20, -32, 14],
                     [28, -76, 28]])
    true = np.array([[-45.83997568,  35.97904559,  23.55194326],
                     [ 22.69248975, -31.34145016,  13.91284087],
                     [ 31.53113226, -76.61685748,  33.22105166]])
    assert np.allclose(tal2mni(test), true)


def test_mni2tal():
    """
    TODO: Get converted coords from official site.
    """
    test = np.array([[-44, 31, 27],
                     [20, -32, 14],
                     [28, -76, 28]])
    true = np.array([[-42.3176,  26.0594,  29.7364],
                     [ 17.4781, -32.6076,  14.0009],
                     [ 24.7353, -75.0184,  23.3283]])
    assert np.allclose(mni2tal(test), true)


def test_vox2mm():
    """
    TODO: Test w/ MATLAB.
    """
    code_dir = os.path.dirname(__file__)
    par_dir = os.path.dirname(code_dir)
    test = np.array([[20, 20, 20],
                     [0, 0, 0]])
    true = np.array([[-56.5, -92.5, -38.5],
                     [-96.5, -132.5, -78.5]])
    info = nib.load(os.path.join(par_dir, 'resources/ICBM152_2009c_2mm.nii.gz'))
    aff = info.affine
    print np.all(vox2mm(test, aff)==true)
    assert np.all(vox2mm(test, aff)==true)


def test_mm2vox():
    """
    TODO: Test w/ MATLAB.
    """
    code_dir = os.path.dirname(__file__)
    par_dir = os.path.dirname(code_dir)
    test = np.array([[20, 20, 20],
                     [0, 0, 0]])
    true = np.array([[58.25, 76.25, 49.25],
                     [48.25, 66.25, 39.25]])
    info = nib.load(os.path.join(par_dir, 'resources/ICBM152_2009c_2mm.nii.gz'))
    aff = info.affine
    print np.all(mm2vox(test, aff)==true)
    assert np.all(mm2vox(test, aff)==true)


def test_thresh_str():
    """
    """
    test = 0.05
    true = '05'
    assert thresh_str(test) == true


def test_read_nifti():
    """
    """
    code_dir = os.path.dirname(__file__)
    par_dir = os.path.dirname(code_dir)
    data, aff = read_nifti(os.path.join(par_dir, 'resources/ICBM152_2009c_2mm.nii.gz'))
    assert data.shape == (97, 115, 97)


def test_save_nifti():
    """
    """
    code_dir = os.path.dirname(__file__)
    par_dir = os.path.dirname(code_dir)
    data, aff = read_nifti(os.path.join(par_dir, 'resources/ICBM152_2009c_2mm.nii.gz'))
    out_file = os.path.join(code_dir, 'data/test_saved.nii')
    save_nifti(data, out_file, aff)
    assert os.path.isfile(out_file)
    os.remove(out_file)
