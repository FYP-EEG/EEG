"""
Author: Anson Li
Created on: 25/7/2026
Purpose: script to read public dataset mentioned in devlog 0725
Location: project_dir/EEG/read_public_data.py
"""

import scipy.io
import os
import mne

#mi data uses edf files
"""
structure
S001-S109
S001R01.edf
so
SxxxRxx.edf
for sxx in dataset folder:
    for rxx in sxx:
        print(sxxrxx.edf)
"""
def read_edf():
    data_dir = os.path.join('dataset', 'mi', 'S001R01.edf')
    data = mne.io.read_raw_edf(data_dir)
    raw_data = data.get_data()
    print(raw_data)
    # you can get the metadata included in the file and a list of all channels:
    info = data.info
    channels = data.ch_names

#ssvep uses mat
"""
structure
sxx
"""
def read_mat():
    data_dir = os.path.join('dataset', 'ssvep', 'S1.mat')
    mat = scipy.io.loadmat(data_dir)
    print(mat["data"])

read_edf()
"""
(venv) D:\cu_hw\FYP>py eeg/read_public_data.py
Extracting EDF parameters from dataset\mi\S001R01.edf...
Setting channel info structure...
Creating raw.info structure...
[[-1.6e-05 -5.6e-05 -5.5e-05 ...  0.0e+00  0.0e+00  0.0e+00]
 [-2.9e-05 -5.4e-05 -5.5e-05 ...  0.0e+00  0.0e+00  0.0e+00]
 [ 2.0e-06 -2.7e-05 -2.9e-05 ...  0.0e+00  0.0e+00  0.0e+00]
 ...
 [-2.1e-05 -1.2e-05  2.0e-06 ...  0.0e+00  0.0e+00  0.0e+00]
 [-1.1e-05  1.0e-06  1.8e-05 ...  0.0e+00  0.0e+00  0.0e+00]
 [ 1.5e-05  2.1e-05  3.5e-05 ...  0.0e+00  0.0e+00  0.0e+00]]
"""
read_mat()
"""
(venv) D:\cu_hw\FYP>py eeg/read_mat.py
[[[[-3.03335762e+01  6.85251770e+01 -1.00500555e+01 -1.91660080e+01
     1.43911123e+01 -1.78026619e+01]
   [ 1.22116871e+01  5.23588257e+01 -1.83858089e+01  8.96278954e+00
    -2.86186810e+01 -2.06258774e+00]
   [-3.61557045e+01 -2.81652718e+01 -4.43965673e+00 -8.86730289e+00
    -1.76492023e+01  7.68885088e+00]
   ...
   [ 1.77202244e+01 -8.15160275e+00 -3.76124611e+01  3.32694740e+01
     1.09082437e+00  2.36016607e+00]
   [ 7.29934502e+00 -3.71400719e+01  8.85271301e+01 -1.19688206e+01
     1.13440094e+01 -7.07160187e+00]
   [-2.16404896e+01 -1.72674065e+01 -2.46735058e+01 -1.35547190e+01
     5.96535778e+00 -6.51751614e+00]]
"""