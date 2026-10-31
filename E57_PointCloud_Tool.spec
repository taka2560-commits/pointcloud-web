# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

# 配布パッケージに同梱するドキュメント（EXEと同じ階層に配置）
datas = [
    ('ノイズフィルター設定ガイド_初心者向け完全マニュアル.html', '.'),
    ('ノイズフィルター設定ガイド_初心者向け完全マニュアル.md', '.'),
    ('ノイズフィルター設定ガイド_初心者向け完全マニュアル.txt', '.'),
    ('README.md', '.'),
    # 新UIのアイコン（コンボ/スピンの矢印・チェックマーク）。実行時は _internal/assets/icons 配下
    ('assets', 'assets'),
]
binaries = []
hiddenimports = []
hiddenimports += collect_submodules('OpenGL')
# pye57 / open3d はネイティブ DLL とデータを持つため明示的に全収集する
for _pkg in ('pye57', 'open3d'):
    tmp_ret = collect_all(_pkg)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
hiddenimports += ['pyquaternion', 'scipy.spatial', 'scipy.spatial._ckdtree']


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='E57_PointCloud_Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name='E57_PointCloud_Tool',
)
