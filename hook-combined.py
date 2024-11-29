# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:05:59 2024

@author: jacke
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = (
    collect_submodules('matplotlib') +
    collect_submodules('ants') +
    collect_submodules('screeninfo') +
    collect_submodules('scipy') +
    collect_submodules('numba') +
    collect_submodules('SimpleITK') +
    collect_submodules('nibabel') +
    collect_submodules('pydicom') +
    collect_submodules('pandas')
)

datas = (
    collect_data_files('matplotlib') +
    collect_data_files('scipy') +
    collect_data_files('PIL')
)
