# -*- coding: utf-8 -*-
""" 
scipy.io.loadmat notes:
v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.
You will need an HDF5 python library to read matlab 7.3 format mat files. 
Because scipy does not supply one, we do not implement the HDF5 / 7.3 interface here.

h5py and hdf5 are both related to importing and reading matlab files
"""
import scipy.io as io
import h5py 
import hdf5
