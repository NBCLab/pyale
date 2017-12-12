# emacs: at the end of the file
# ex: set sts=4 ts=4 sw=4 et:
"""
Functions:
- Read in text/Excel files as class
- For each experiment, generate kernel

"""
from __future__ import print_function
import os
import re

import numpy as np
import pandas as pd
import nibabel as nib

from .due import due, Doi
from .utils import (round2, mm2vox, tal2mni, get_kernel, get_resource_path,
                    cite_mni152)


class Dataset(object):
    """ Dataset class.
    """
    def __init__(self, name, experiments, space):
        """
        """
        self.name = name
        self.space = space
        self.experiments = experiments


class Experiment(object):
    """ Experiment class.
    """
    def __init__(self, name, n, xyz, template):
        """
        Args:
        - name: Name/identifier of experiment. String.
        - n: Sample size of experiment. Int.
        - xyz: Significant foci from experiment in MNI space. Numpy array.
        - template: nibabel.Nifti1Image

        NOTES:
            ijk: Foci in matrix space. Numpy array.
        """
        self.name = name
        self.n = n
        self.xyz = round2(xyz).astype(int)
        self.fwhm = None
        self.kernel = None

        ijk = round2(mm2vox(xyz, template.affine)).astype(int)

        # Smoosh foci outside template to the edge of the template.
        ijk[ijk[:, 0] >= template.shape[0], 0] = template.shape[0]-1
        ijk[ijk[:, 1] >= template.shape[1], 1] = template.shape[1]-1
        ijk[ijk[:, 2] >= template.shape[2], 2] = template.shape[2]-1
        ijk[ijk < 0] = 0
        self.ijk = ijk


def import_neurosynth(csv_file, sep='\t', template_file='Grey10.nii.gz'):
    """
    Read Neurosynth-style comma- or tab-separated values file into Dataset and
    Experiments.
    Required columns:
        - x, y, z: Coordinates in mm (not ijk matrix-space).
        - n: Sample size.
        - id: Unique experiment identifier. Standard is PMID-ExpID or BMID-ExpID.
        - space: Standard space of each experiment. Options: ['MNI', 'TAL']
                 Generated Experiments will be automatically transformed to MNI
                 space.
    """
    # Cite MNI152 paper if default template is used
    if template_file == 'Grey10.nii.gz':
        cite_mni152()

    # Check path for template file
    if not os.path.dirname(template_file):
        template_file = os.path.join(get_resource_path(), template_file)

    # Load template
    template_info = nib.load(template_file)

    filename = os.path.basename(csv_file)
    study_name, _ = os.path.splitext(filename)
    activations = pd.read_csv(csv_file, sep=sep)
    activations.columns = [col.lower() for col in list(activations.columns)]

    # Make sure all mandatory columns exist
    mand_cols = ['x', 'y', 'z', 'n', 'id', 'space']
    if set(mand_cols) - set(list(activations.columns)):
        diff = list(set(mand_cols) - set(list(activations.columns)))
        raise Exception('Missing required columns: {0}'.format(', '.join(diff)))

    # Check spaces of experiments in dataset
    spaces = activations['space'].unique()
    if not set(spaces).issubset(['MNI', 'TAL']):
        unk_spaces = list(set(spaces) - set(['MNI', 'TAL']))
        raise Exception('Space(s) {0} not recognized.\nOptions supported: '
                        'MNI or TAL.'.format(', '.join(unk_spaces)))

    # Convert Talairach coords to MNI
    xyz = activations[['x', 'y', 'z']].values
    tal_idx = activations['space'] == 'TAL'
    xyz[tal_idx] = tal2mni(xyz[tal_idx])
    activations[['x', 'y', 'z']] = xyz

    exp_list = [[] for exp in range(len(activations['id'].unique()))]

    grouped = activations.groupby('id')  # pylint: disable=no-member
    for i, exp in enumerate(grouped):
        id_ = exp[0]
        sample_size = int(exp[1]['n'].unique()[0])
        exp_xyz = exp[1][['x', 'y', 'z']].values.astype(float)

        exp_list[i] = Experiment(id_, sample_size, exp_xyz, template_info)
        exp_list[i].fwhm, exp_list[i].kernel = get_kernel(exp_list[i].n,
                                                          template_info)
    exp_list = [exp for exp in exp_list if exp]
    dataset = Dataset(study_name, exp_list, 'MNI')
    return dataset


@due.dcite(Doi('10.1038/nrn789'),
           description='Describes the BrainMap model.')
@due.dcite(Doi('10.1002/hbm.20141'),
           description='Describes the BrainMap taxonomy.')
def import_sleuth(text_file, template_file='Grey10.nii.gz'):
    """
    Read Sleuth output text file into Dataset and Experiments.
    """
    # Cite MNI152 paper if default template is used
    if template_file == 'Grey10.nii.gz':
        cite_mni152()

    # Check path for template file
    if not os.path.dirname(template_file):
        template_file = os.path.join(get_resource_path(), template_file)

    # Load template
    template_info = nib.load(template_file)

    filename = os.path.basename(text_file)
    study_name, _ = os.path.splitext(filename)
    with open(text_file, 'r') as file_object:
        data = file_object.read()
    data = [line.rstrip() for line in re.split('\n\r|\r\n|\n|\r', data)]
    data = [line for line in data if line]

    # First line indicates stereotactic space. The rest are studies, ns, and coords.
    space = data[0].replace(' ', '').replace('//Reference=', '')
    if space not in ['MNI', 'TAL']:
        raise Exception('Space {0} unknown. Options supported: '
                        'MNI or TAL.'.format(space))
    elif space == 'TAL':
        print('Converting coordinates from Talairach space to MNI.')

    # Split into experiments
    data = data[1:]
    metadata_idx = [i for i, line in enumerate(data) if line.startswith('//')]
    exp_idx = np.split(metadata_idx, np.where(np.diff(metadata_idx) != 1)[0]+1)
    start_idx = [tup[0] for tup in exp_idx]
    end_idx = start_idx[1:] + [len(data)+1]
    split_idx = zip(start_idx, end_idx)

    exp_list = [[] for exp in split_idx]
    for i_exp, exp_idx in enumerate(split_idx):
        exp_data = data[exp_idx[0]:exp_idx[1]]
        if exp_data:
            study_info = exp_data[0].replace('//', '').strip()
            sample_size = int(exp_data[1].replace(' ', '').replace('//Subjects=', ''))
            xyz = exp_data[2:]  # Coords are everything after study info and sample size
            xyz = [row.split('\t') for row in xyz]
            correct_shape = np.all([len(coord) == 3 for coord in xyz])
            if not correct_shape:
                all_shapes = np.unique([len(coord) for coord in xyz]).astype(str)  # pylint: disable=no-member
                raise Exception('Coordinates for study "{0}" are not all correct length. '
                                'Lengths detected: {1}.'.format(study_info,
                                                                ', '.join(all_shapes)))

            try:
                xyz = np.array(xyz, dtype=float)
            except:
                # Prettify xyz
                strs = [[str(e) for e in row] for row in xyz]
                lens = [max(map(len, col)) for col in zip(*strs)]
                fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
                table = '\n'.join([fmt.format(*row) for row in strs])
                raise Exception('Conversion to numpy array failed for study "{0}". '
                                'Coords:\n{1}'.format(study_info, table))

            if space == 'TAL':
                xyz = tal2mni(xyz)
            exp_list[i_exp] = Experiment(study_info, sample_size, xyz,
                                         template_info)
            exp_list[i_exp].fwhm, exp_list[i_exp].kernel = get_kernel(exp_list[i_exp].n,
                                                                      template_info)
        else:
            exp_list[i_exp] = []

    exp_list = [exp for exp in exp_list if exp]
    dataset = Dataset(study_name, exp_list, space)
    return dataset
