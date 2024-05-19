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
    [],
    exclude_binaries=True,
    name='face_landmarks_detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)