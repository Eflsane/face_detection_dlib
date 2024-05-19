# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

# Function to collect data files
def collect_additional_files():
    data_files = '.\\my_content\\shape_predictor_68_face_landmarks.dat'
    additional_files = [('.\\shape_predictor_68_face_landmarks.dat', '.\\'), ('.\\example_video.mp4', '.\\'), ('.\\model4.pt', '.\\')]
    return additional_files


a = Analysis(
    ['main_test3_my_nn.py'],
    pathex=[],
    binaries=collect_additional_files(),
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    exclude_binaries=False,
    name='face_landmarks_detection',
    strip=False,
    upx=True,
)