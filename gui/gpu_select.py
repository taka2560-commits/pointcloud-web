# -*- coding: utf-8 -*-
"""
描画に使う GPU の選び方（内蔵 GPU と高性能 GPU の両方が載っているノート PC 向け）

なぜ要るか（2026-11-15、利用者の PC = AMD Radeon 890M（内蔵）+ NVIDIA GeForce RTX 5060 Laptop で確認）
    画面が内蔵 GPU につながっているノート PC では、何も頼まないアプリは内蔵 GPU で描かれる。
    このアプリもそうで、ログの「OpenGL描画環境」は毎回 Radeon 890M だった（RTX は使われていなかった）。
    5,490 万点の全点表示で 1 回描く時間は、内蔵 GPU 0.44 秒・RTX 0.15 秒。手作業のノイズ取りでは、
    消す・戻す・選ぶたびに 1 回描くので、この差がそのまま待ち時間になる。

どう頼むか
    OpenGL を使い始める前（最初のウィンドウを作る前）に、NVIDIA のドライバへ「高性能の GPU を使いたい」と伝える。
      1. 環境変数 SHIM_MCCOMPAT=0x800000001 を、このプロセスに入れる
      2. Windows のフォルダにある nvapi64.dll を読み込む（NVIDIA の資料にある、高性能の GPU が選ばれる条件の 1 つ）
    どちらも、このプロセスの中だけのこと。Windows の設定・レジストリ・ドライバの設定は変えない。
    NVIDIA の GPU が無い PC では、何も起きない（1 は読まれない環境変数、2 は DLL が無いので読まない）。
    実際にどの GPU で描かれたかは、起動のログ「OpenGL描画環境」と、設定 → システムの「いまの描画」で分かる。

選び方の優先順位（上が強い。gui/ui_mode.py と同じ形）
    1. 起動引数   --gpu=high / --gpu=system
    2. 環境変数   E57_GPU=high|system
    3. 保存設定   render/gpu（設定 → システム「描画の GPU」。次回の起動から）
    4. 既定       high（高性能の GPU を優先）
    system は「何も頼まない」= Windows とドライバに任せる（ノート PC では、ふつう内蔵 GPU。省電力）。
    高性能の GPU で表示がおかしい・起動しないときは、--gpu=system か E57_GPU=system で元の動きに戻せる。

決まり
    ・apply_gpu_choice() は main.py が、PySide6 の画面の部品を読み込む前に 1 回だけ呼ぶ。何が起きても例外を出さない。
    ・このファイルの先頭では、重いもの（PySide6 の画面の部品・OpenGL）を読み込まない。
"""

import os
import sys
from typing import Callable, List, Optional, Sequence

GPU_HIGH = "high"        # 高性能の GPU を優先（既定）
GPU_SYSTEM = "system"    # Windows に任せる（何も頼まない）
_VALID = (GPU_HIGH, GPU_SYSTEM)
GPU_LABELS = {GPU_HIGH: "高性能を優先", GPU_SYSTEM: "Windows に任せる"}

_SETTINGS_ORG = "E57PointCloudTool"
_SETTINGS_APP = "Settings"
SETTINGS_KEY = "render/gpu"
ENV_KEY = "E57_GPU"
ARG_PREFIX = "--gpu="

NVIDIA_ENV = "SHIM_MCCOMPAT"
NVIDIA_ENV_VALUE = "0x800000001"
NVIDIA_DLL = "nvapi64.dll"

_applied: Optional[str] = None      # このプロセスで実際に使った選び方（apply_gpu_choice が入れる）
_loaded_dll = None                  # 読み込んだ DLL（プロセスが終わるまで持っておく）


def _normalize(value) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("high", "nvidia", "rtx", "dgpu", "discrete", "performance", "高性能"):
        return GPU_HIGH
    if v in ("system", "auto", "windows", "integrated", "igpu", "powersave", "省電力", "内蔵"):
        return GPU_SYSTEM
    return None


def get_saved_gpu_choice() -> str:
    """保存されている選び方（未保存なら high）"""
    try:
        from PySide6 import QtCore

        return _normalize(QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP).value(SETTINGS_KEY, None)) or GPU_HIGH
    except Exception:
        return GPU_HIGH


def set_saved_gpu_choice(choice: str) -> None:
    """設定画面から選んだものを保存する（次回の起動から）"""
    from PySide6 import QtCore

    s = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.setValue(SETTINGS_KEY, _normalize(choice) or GPU_HIGH)
    s.sync()


def resolve_gpu_choice(argv: Optional[Sequence[str]] = None, env: Optional[dict] = None, use_settings: bool = True) -> str:
    """優先順位に従って選び方を決める（テストしやすいよう、引数で差し替えられる）"""
    args = list(sys.argv[1:] if argv is None else argv)
    for a in args:
        if a.startswith(ARG_PREFIX):
            choice = _normalize(a[len(ARG_PREFIX):])
            if choice:
                return choice
    environ = os.environ if env is None else env
    choice = _normalize(environ.get(ENV_KEY))
    if choice:
        return choice
    if use_settings:
        return get_saved_gpu_choice()
    return GPU_HIGH


def _nvidia_dll_path(environ) -> str:
    root = environ.get("SystemRoot") or environ.get("WINDIR") or r"C:\Windows"
    return os.path.join(root, "System32", NVIDIA_DLL)


def _load_dll(path: str):
    import ctypes

    return ctypes.WinDLL(path)


def request_high_performance_gpu(
    environ=None, platform: Optional[str] = None,
    dll_exists: Callable[[str], bool] = os.path.isfile, load_dll: Callable[[str], object] = _load_dll,
) -> List[str]:
    """
    NVIDIA のドライバへ、高性能の GPU を使いたいと伝える。したことを、短い言葉の並びで返す（ログ用）。
    OpenGL を使い始める前に呼ぶこと。Windows 以外では何もしない。
    """
    global _loaded_dll
    environ = os.environ if environ is None else environ
    done: List[str] = []
    if (sys.platform if platform is None else platform) != "win32":
        return done
    if environ.get(NVIDIA_ENV):
        done.append(f"{NVIDIA_ENV} は設定済み（そのまま使う）")       # 利用者が入れた値は変えない
    else:
        environ[NVIDIA_ENV] = NVIDIA_ENV_VALUE
        done.append(f"{NVIDIA_ENV} を入れた")
    path = _nvidia_dll_path(environ)
    if dll_exists(path):
        try:
            _loaded_dll = load_dll(path)
            done.append(f"{NVIDIA_DLL} を読み込んだ")
        except Exception as exc:       # 読めなくても、アプリは今までどおり動く
            done.append(f"{NVIDIA_DLL} は読み込めなかった（{exc.__class__.__name__}）")
    else:
        done.append("NVIDIA のドライバは見つからない")
    return done


def apply_gpu_choice(argv: Optional[Sequence[str]] = None, env: Optional[dict] = None) -> str:
    """
    選び方を決めて、高性能の GPU を頼む。main.py が、画面の部品を読み込む前に 1 回だけ呼ぶ。
    ログに書く 1 行を返す。調べる仕掛けのせいでアプリが起動しなくなってはいけないので、何が起きても例外を出さない。
    """
    global _applied
    try:
        choice = resolve_gpu_choice(argv, env)
        _applied = choice
        if choice != GPU_HIGH:
            return f"{GPU_LABELS[choice]}（何も頼まない）"
        done = request_high_performance_gpu(env)
        return f"{GPU_LABELS[choice]}（{'・'.join(done) if done else 'この OS では何もしない'}）"
    except Exception as exc:
        return f"選べませんでした（今までどおり Windows に任せます）: {exc!r}"


def applied_gpu_choice() -> Optional[str]:
    """このプロセスで使った選び方。main.py を通らない起動（テスト・品質ゲート）では None = 何も頼んでいない"""
    return _applied


def renderer_name(viewer) -> str:
    """いま点群を描いている GPU の名前（まだ描き始めていなければ空）"""
    try:
        if viewer is None or not viewer.isValid():
            return ""
        from OpenGL import GL as gl

        viewer.makeCurrent()
        try:
            name = gl.glGetString(gl.GL_RENDERER)
        finally:
            viewer.doneCurrent()
        text = name.decode("utf-8", errors="ignore") if isinstance(name, bytes) else str(name or "")
        return text.split("/")[0].strip()        # NVIDIA は「…/PCIe/SSE2」と続くので、名前だけにする
    except Exception:
        return ""


def describe_now(viewer, saved: Optional[str] = None) -> str:
    """設定 → システムに出す 1 行（いまの描画の GPU。保存した選び方が、いまの起動と違えばそのことも）"""
    name = renderer_name(viewer)
    text = f"いまの描画: {name}" if name else "いまの描画: 調べています…"
    saved = get_saved_gpu_choice() if saved is None else saved
    now = _applied or GPU_SYSTEM
    if saved != now:
        text += f"（「{GPU_LABELS[saved]}」は次回の起動から）"
    return text


# ----------------------------------------------------------------------------- 設定 → システムの行
def install_gpu_row(win, tab, lay, row: int) -> int:
    """
    設定 → システムに「描画の GPU」の行を足す（gui/system_tools.py: install_system_tools から呼ぶ）。次に空いている行を返す。
    """
    from PySide6.QtCore import QEvent, QObject, QTimer
    from PySide6.QtWidgets import QComboBox, QLabel

    lay.addWidget(QLabel("描画の GPU:", tab), row, 0)
    combo = QComboBox(tab)
    combo.setObjectName("combo_gpu_choice")
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(8)
    combo.addItem("高性能を優先 (標準)", GPU_HIGH)        # 左パネルの幅（約 14 文字）に収まる長さにする
    combo.addItem("Windows に任せる", GPU_SYSTEM)
    combo.setCurrentIndex(max(0, combo.findData(get_saved_gpu_choice())))
    combo.setToolTip(
        "内蔵 GPU と高性能の GPU（NVIDIA）の両方が載っている PC で、点群の描画にどちらを使うかを選びます（次回の起動から）。\n"
        "高性能の GPU では、大きな点群の回転・選択・削除が速くなります。\n"
        "バッテリーで長く使いたいときや、表示がおかしいときは「Windows に任せる」にしてください（省電力。今までの動き）。\n"
        "起動引数 --gpu=system、環境変数 E57_GPU=system でも指定できます。"
    )
    lay.addWidget(combo, row, 1)

    hint = QLabel(describe_now(None), tab)
    hint.setObjectName("lbl_gpu_now")
    hint.setWordWrap(True)
    hint.setToolTip("いま点群を描いている GPU です。起動のログの「OpenGL描画環境」にも同じ名前が出ます")
    lay.addWidget(hint, row + 1, 0, 1, 2)

    def refresh() -> None:
        try:
            hint.setText(describe_now(getattr(win, "viewer", None)))
        except RuntimeError:
            pass  # 待っている間に画面が閉じられた

    def on_changed(_index: int) -> None:
        choice = combo.currentData()
        if choice not in _VALID:
            return
        set_saved_gpu_choice(choice)
        refresh()
        try:
            win.show_toast(f"描画の GPU を「{GPU_LABELS[choice]}」に設定しました（次回の起動から）")
        except Exception:
            pass

    combo.currentIndexChanged.connect(on_changed)

    class _ShowWatcher(QObject):
        """タブが開かれたら、いまの描画の GPU の名前を読み直す"""

        def eventFilter(self, obj, event):  # type: ignore[override]
            if event.type() == QEvent.Show:
                QTimer.singleShot(0, hint, refresh)
            return False

    win.combo_gpu_choice = combo
    win.lbl_gpu_now = hint
    win._gpu_row_refresh = refresh
    win._gpu_row_watcher = _ShowWatcher(tab)
    tab.installEventFilter(win._gpu_row_watcher)
    return row + 2
