"""
Author: Anson Li
Created on: 25/7/2026
Purpose: script to read public dataset mentioned in devlog 0725
Location: project_dir/EEG/read_public_data.py
"""

import scipy.io
import os
import mne


# Readers for example public datasets. These functions are safe to import; they
# only perform file I/O when called explicitly.

def read_edf(path=None):
    """Read a single EDF file from the dataset/mi folder.

    Returns the raw numpy array from the EDF file.
    """
    if path is None:
        path = os.path.join('dataset', 'mi', 'S001R01.edf')
    data = mne.io.read_raw_edf(path)
    raw_data = data.get_data()
    return raw_data


def read_mat(path=None):
    """Read a MATLAB .mat file from the dataset/ssvep folder.

    Returns the loaded mat dict.
    """
    if path is None:
        path = os.path.join('dataset', 'ssvep', 'S1.mat')
    mat = scipy.io.loadmat(path)
    return mat


if __name__ == "__main__":
    # When run as a script, execute both readers and print summaries.
    edf_data = read_edf()
    print('EDF data shape:', getattr(edf_data, 'shape', type(edf_data)))
    mat = read_mat()
    print('MAT keys:', list(mat.keys()))
