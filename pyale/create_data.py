# emacs: at the end of the file
# ex: set sts=4 ts=4 sw=4 et:
import os

import numpy as np
import nibabel as nib
from nipype.interfaces.fsl.maths import Threshold
from nipype.interfaces.freesurfer.preprocess import MRIConvert

from .due import due, Doi


@due.dcite(Doi('10.1016/j.neuroimage.2010.07.033'),
           description='Introduces the MNI152 template.')
def convert_template():
    """
    Reslice openly available MNI template in nii format to nii with desired
    voxel size. For MNI template license, see data/LICENSES file.
    """
    mc = MRIConvert(vox_size=(2., 2., 2.))
    mc.inputs.in_file = 'resources/mni_icbm152_gm_tal_nlin_sym_09c.nii.gz'
    mc.inputs.out_file = 'resources/ICBM152_2009c_2mm.nii.gz'
    mc.inputs.out_type = 'nii'
    mc.run()

    thr = Threshold()
    thr.inputs.in_file = 'resources/ICBM152_2009c_2mm.nii.gz'
    thr.inputs.terminal_output = 'none'
    thr.inputs.thresh = 0.10
    thr.inputs.args = '-bin'
    thr.inputs.out_file = 'resources/ICBM152_2009c_2mm_gm10.nii.gz'
    thr.run()

    thr = Threshold()
    thr.inputs.in_file = 'resources/mni_icbm152_gm_tal_nlin_sym_09c.nii.gz'
    thr.inputs.terminal_output = 'none'
    thr.inputs.thresh = 0.10
    thr.inputs.args = '-bin'
    thr.inputs.out_file = 'resources/ICBM152_2009c_1mm_gm10.nii.gz'
    thr.run()


def create_perm_space(template='Grey10.nii.gz'):
    """
    Generate file for containing relevant information for permutations from
    template.

    Parameters
    ----------
    template : str
        Template file determining voxel dimensions and permutation space for
        ALE. Should be binarized.

    Examples
    ----------

    """
    code_dir = os.path.dirname(__file__)
    resources_dir = os.path.join(code_dir, 'resources/')

    if not os.path.dirname(template):
        template = os.path.join(resources_dir, template)
    else:
        template = os.path.abspath(template)

    # Load template
    template_info = nib.load(template)
    voxel_sizes = np.array(template_info.header.get_zooms())  # pylint: disable=no-member
    if np.all(voxel_sizes == 1.):
        expand_val = 50
    elif np.all(voxel_sizes == 2.):
        expand_val = 30
    else:
        raise Exception('Voxel size not supported.')
    half_val = int(expand_val / 2.)
    template_data = template_info.get_data().astype(bool)
    idx = np.where(template_data != 0)
    n_xyz = idx[0].shape[0]
    all_xyz = np.vstack(idx)
    dims = np.array(template_info.shape)
    expanded_dims = dims + expand_val
    expanded_xyz = all_xyz + half_val
    expanded_idx = np.ravel_multi_index(expanded_xyz, expanded_dims)

    perm_space = {'n_xyz': n_xyz,
                  'expanded_xyz': expanded_xyz,
                  'expanded_dims': expanded_dims,
                  'expand_val': expand_val,
                  'half_val': half_val,
                  'template': template_info,
                  'expanded_idx': expanded_idx}

    return perm_space
