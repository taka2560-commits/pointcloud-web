"""
テストと品質ゲートのあいだ、アプリの保存設定（QSettings）を一時フォルダの INI ファイルへ向ける

MainWindow は閉じるときに設定を自動保存する。テストは `_reset_app_settings()` で初期値にしてから閉じるので、
そのままだと「テストを回すたびに、この PC の本番アプリの保存設定（点サイズ・描画点数・テーマ・最近使ったファイル
など）が初期値やテスト用の一時ファイルで上書きされる」。2026-11-11 の点検で見つけた。
これを防ぐため、MainWindow を作るテスト・品質ゲート・画面キャプチャの道具は、最初に use_sandbox_settings() を呼ぶ。

仕組み: アプリは `QSettings("E57PointCloudTool", "Settings")` と書く。この 2 引数の作り方は必ずレジストリ
（NativeFormat）を使い、`QSettings.setDefaultFormat()` では切り替えられない。そこで、テストのプロセスの中だけ
`PySide6.QtCore.QSettings` を「2 引数で作られたら一時フォルダの INI を使う」派生クラスに差し替える。
すでに import 済みのモジュールが持っている `QSettings` の名前も同じクラスへ付け替えるので、呼ぶ順番に依らない。
アプリ本体のコードは変えない（配布 EXE と通常の起動には影響しない）。
アプリ側の読み出しはすべて type= 指定か正規化関数を通すので、レジストリと INI で結果は変わらない。
"""

import atexit
import os
import shutil
import sys
import tempfile

_ENV_KEY = "E57_TEST_SETTINGS_DIR"
_MARK = "_e57_sandbox_settings"


def use_sandbox_settings() -> str:
    """
    QSettings を一時フォルダへ向ける。同じプロセスで何度呼んでも同じ場所を返す。
    MainWindow を作るより前に呼ぶこと（モジュールの先頭で呼んでおけばよい）。
    """
    from PySide6 import QtCore

    current = QtCore.QSettings
    if getattr(current, _MARK, False):
        return os.environ[_ENV_KEY]

    path = os.environ.get(_ENV_KEY, "")
    if not path or not os.path.isdir(path):
        path = tempfile.mkdtemp(prefix="e57_test_settings_")
        os.environ[_ENV_KEY] = path
        atexit.register(shutil.rmtree, path, True)

    original = current
    original.setPath(original.IniFormat, original.UserScope, path)
    original.setPath(original.IniFormat, original.SystemScope, path)

    class SandboxSettings(original):  # type: ignore[misc, valid-type]
        """組織名とアプリ名の 2 引数で作られたら、レジストリではなく一時フォルダの INI を使う"""

        def __init__(self, *args, **kwargs):
            if len(args) == 2 and not kwargs and isinstance(args[0], str) and isinstance(args[1], str):
                super().__init__(original.IniFormat, original.UserScope, args[0], args[1])
            else:
                super().__init__(*args, **kwargs)

    setattr(SandboxSettings, _MARK, True)
    SandboxSettings.__name__ = "QSettings"
    QtCore.QSettings = SandboxSettings
    # すでに `from PySide6.QtCore import QSettings` を済ませたモジュールの名前も付け替える
    for mod in list(sys.modules.values()):
        try:
            if mod is not None and getattr(mod, "QSettings", None) is original:
                setattr(mod, "QSettings", SandboxSettings)
        except Exception:
            continue
    return path


def is_sandboxed() -> bool:
    """いま、アプリと同じ作り方の QSettings が一時フォルダの INI を指しているか"""
    from PySide6 import QtCore

    path = os.environ.get(_ENV_KEY, "")
    if not path or not getattr(QtCore.QSettings, _MARK, False):
        return False
    probe = QtCore.QSettings("E57PointCloudTool", "Settings").fileName()
    return os.path.normcase(os.path.abspath(probe)).startswith(os.path.normcase(os.path.abspath(path)))
