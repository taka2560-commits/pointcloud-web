# -*- coding: utf-8 -*-
"""
描画に使う GPU の選び方（gui/gpu_select.py）

2026-11-15、利用者の指摘「内蔵 GPU ではなく RTX のほうは活用しないのか？」から:
- 利用者の PC（Radeon 890M + RTX 5060 Laptop）で、アプリは毎回 内蔵 GPU で描いていた（RTX は使われていなかった）
- OpenGL を使い始める前に NVIDIA のドライバへ頼むと、RTX で描かれる（5,490 万点を 1 回描く時間が 0.44 秒 → 0.15 秒）
ここで見張ること: 選び方の優先順位、頼み方が「このプロセスの中だけ」で安全なこと、頼むのが画面の部品より前であること、
設定 → システムの行。
"""
import ast
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtGui import QShowEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import gpu_select as G  # noqa: E402


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


class _Keep(unittest.TestCase):
    """テストごとに、保存した選び方と「このプロセスで使った選び方」を元へ戻す"""

    def setUp(self):
        saved, applied, dll = G.get_saved_gpu_choice(), G._applied, G._loaded_dll
        self.addCleanup(lambda: (G.set_saved_gpu_choice(saved), setattr(G, "_applied", applied), setattr(G, "_loaded_dll", dll)))


class TestChoice(_Keep):
    def test_default_is_high_performance(self):
        self.assertEqual(G.resolve_gpu_choice(argv=[], env={}, use_settings=False), G.GPU_HIGH)

    def test_argument_beats_environment_beats_setting(self):
        G.set_saved_gpu_choice(G.GPU_HIGH)
        self.assertEqual(G.resolve_gpu_choice(argv=[], env={}), G.GPU_HIGH)
        self.assertEqual(G.resolve_gpu_choice(argv=[], env={"E57_GPU": "system"}), G.GPU_SYSTEM)
        self.assertEqual(G.resolve_gpu_choice(argv=["--gpu=high"], env={"E57_GPU": "system"}), G.GPU_HIGH)
        G.set_saved_gpu_choice(G.GPU_SYSTEM)
        self.assertEqual(G.resolve_gpu_choice(argv=[], env={}), G.GPU_SYSTEM)
        self.assertEqual(G.resolve_gpu_choice(argv=[], env={"E57_GPU": "high"}), G.GPU_HIGH)
        self.assertEqual(G.resolve_gpu_choice(argv=["x.e57", "--gpu=system"], env={"E57_GPU": "high"}), G.GPU_SYSTEM)

    def test_unknown_values_are_ignored(self):
        G.set_saved_gpu_choice(G.GPU_SYSTEM)
        self.assertEqual(G.resolve_gpu_choice(argv=["--gpu=fastest"], env={"E57_GPU": "?"}), G.GPU_SYSTEM)
        G.set_saved_gpu_choice("なにか")          # 知らない値は、既定で保存する
        self.assertEqual(G.get_saved_gpu_choice(), G.GPU_HIGH)

    def test_words_people_would_type(self):
        for word in ("high", "HIGH", " nvidia ", "rtx", "高性能"):
            self.assertEqual(G._normalize(word), G.GPU_HIGH, word)
        for word in ("system", "integrated", "igpu", "省電力", "内蔵", "windows"):
            self.assertEqual(G._normalize(word), G.GPU_SYSTEM, word)
        self.assertIsNone(G._normalize(None))
        self.assertIsNone(G._normalize(""))

    def test_saved_choice_round_trip(self):
        for choice in (G.GPU_SYSTEM, G.GPU_HIGH):
            G.set_saved_gpu_choice(choice)
            self.assertEqual(G.get_saved_gpu_choice(), choice)


class TestRequest(_Keep):
    """頼み方: このプロセスの環境変数と、Windows のフォルダの DLL だけ。失敗してもアプリは動く"""

    def _request(self, env, exists=True, loader=None, platform="win32"):
        loaded = []

        def load(path):
            loaded.append(path)
            return object()

        done = G.request_high_performance_gpu(env, platform=platform, dll_exists=lambda p: exists, load_dll=loader or load)
        return done, loaded

    def test_sets_the_variable_and_loads_the_driver_library(self):
        env = {"SystemRoot": r"D:\Win"}
        done, loaded = self._request(env)
        self.assertEqual(env[G.NVIDIA_ENV], G.NVIDIA_ENV_VALUE)
        self.assertEqual(loaded, [os.path.join(r"D:\Win", "System32", "nvapi64.dll")])   # 作業フォルダなど、ほかの場所からは読まない
        self.assertEqual(len(done), 2)
        self.assertIsNotNone(G._loaded_dll)       # 読み込んだものは、持ったままにする

    def test_a_value_the_user_set_is_left_alone(self):
        env = {G.NVIDIA_ENV: "0x1", "SystemRoot": r"C:\Windows"}
        done, _ = self._request(env)
        self.assertEqual(env[G.NVIDIA_ENV], "0x1")
        self.assertIn("設定済み", done[0])

    def test_no_nvidia_driver_means_nothing_is_loaded(self):
        env = {}
        done, loaded = self._request(env, exists=False)
        self.assertEqual(loaded, [])
        self.assertEqual(env[G.NVIDIA_ENV], G.NVIDIA_ENV_VALUE)      # 読まれないだけの環境変数。害は無い
        self.assertIn("見つからない", done[-1])

    def test_a_library_that_fails_to_load_does_not_stop_the_app(self):
        def broken(path):
            raise OSError("読めない")

        done, _ = self._request({}, loader=broken)
        self.assertIn("読み込めなかった", done[-1])

    def test_nothing_happens_outside_windows(self):
        env = {}
        done, loaded = self._request(env, platform="linux")
        self.assertEqual((done, loaded, env), ([], [], {}))


class TestApply(_Keep):
    def _env(self, **extra):
        env = {"SystemRoot": tempfile.gettempdir()}      # ここには nvapi64.dll が無いので、本物の DLL は読まれない
        env.update(extra)
        return env

    def test_high_asks_the_driver(self):
        env = self._env()
        line = G.apply_gpu_choice(argv=["--gpu=high"], env=env)
        self.assertEqual(G.applied_gpu_choice(), G.GPU_HIGH)
        if sys.platform == "win32":
            self.assertEqual(env.get(G.NVIDIA_ENV), G.NVIDIA_ENV_VALUE)
        self.assertIn("高性能", line)

    def test_system_asks_for_nothing(self):
        env = self._env()
        line = G.apply_gpu_choice(argv=["--gpu=system"], env=env)
        self.assertEqual(G.applied_gpu_choice(), G.GPU_SYSTEM)
        self.assertNotIn(G.NVIDIA_ENV, env)
        self.assertIn("何も頼まない", line)

    def test_never_raises(self):
        original = G.resolve_gpu_choice
        self.addCleanup(setattr, G, "resolve_gpu_choice", original)

        def broken(*a, **k):
            raise RuntimeError("設定が読めない")

        G.resolve_gpu_choice = broken
        line = G.apply_gpu_choice(argv=[], env=self._env())
        self.assertIn("選べませんでした", line)


class TestStartupOrder(unittest.TestCase):
    """頼むのは、OpenGL を使い始める前でないと効かない。入口の順番と、このファイルの軽さを見張る"""

    def test_main_asks_before_loading_the_screen_parts(self):
        src = _read("main.py")
        ask = src.index("apply_gpu_choice()")
        self.assertLess(ask, src.index("from PySide6.QtWidgets import QApplication"))
        self.assertLess(ask, src.index("from gui.main_window import MainWindow"))
        self.assertLess(ask, src.index("QApplication(sys.argv)"))

    def test_module_top_level_imports_nothing_heavy(self):
        tree = ast.parse(_read("gui/gpu_select.py"))
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertEqual(names, {"os", "sys", "typing"})

    def test_nothing_outside_this_process_is_changed(self):
        src = _read("gui/gpu_select.py")
        for word in ("winreg", "UserGpuPreferences", "subprocess", "setx"):
            self.assertNotIn(word, src.split('"""', 2)[2], word)      # 説明の文は除く


class _FakeViewer:
    def __init__(self, valid=True):
        self.valid = valid
        self.calls = []

    def isValid(self):
        return self.valid

    def makeCurrent(self):
        self.calls.append("make")

    def doneCurrent(self):
        self.calls.append("done")


class TestRendererName(_Keep):
    def _patch_gl(self, value):
        from OpenGL import GL as gl

        original = gl.glGetString
        self.addCleanup(setattr, gl, "glGetString", original)
        if isinstance(value, Exception):
            def get(_what):
                raise value
        else:
            def get(_what):
                return value
        gl.glGetString = get

    def test_name_is_trimmed_to_the_gpu(self):
        self._patch_gl(b"NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2")
        v = _FakeViewer()
        self.assertEqual(G.renderer_name(v), "NVIDIA GeForce RTX 5060 Laptop GPU")
        self.assertEqual(v.calls, ["make", "done"])
        self._patch_gl("AMD Radeon(TM) 890M Graphics")
        self.assertEqual(G.renderer_name(_FakeViewer()), "AMD Radeon(TM) 890M Graphics")

    def test_not_ready_or_failing_gives_an_empty_name(self):
        self.assertEqual(G.renderer_name(None), "")
        v = _FakeViewer(valid=False)
        self.assertEqual(G.renderer_name(v), "")
        self.assertEqual(v.calls, [])
        self._patch_gl(RuntimeError("コンテキストが無い"))
        v = _FakeViewer()
        self.assertEqual(G.renderer_name(v), "")
        self.assertEqual(v.calls, ["make", "done"])       # 失敗しても、コンテキストは返す

    def test_line_tells_when_the_saved_choice_starts_next_time(self):
        self._patch_gl(b"AMD Radeon(TM) 890M Graphics")
        G._applied = G.GPU_SYSTEM
        self.assertEqual(G.describe_now(_FakeViewer(), saved=G.GPU_SYSTEM), "いまの描画: AMD Radeon(TM) 890M Graphics")
        line = G.describe_now(_FakeViewer(), saved=G.GPU_HIGH)
        self.assertIn("次回の起動から", line)
        self.assertIn("高性能を優先", line)
        G._applied = None                                  # main.py を通らない起動 = 何も頼んでいない
        self.assertIn("次回の起動から", G.describe_now(_FakeViewer(), saved=G.GPU_HIGH))
        self.assertNotIn("次回の起動から", G.describe_now(_FakeViewer(), saved=G.GPU_SYSTEM))
        self.assertIn("調べています", G.describe_now(None, saved=G.GPU_SYSTEM))


class TestSystemTabRow(_Keep):
    """設定 → システムの「描画の GPU」の行"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from gui.main_window import MainWindow

        cls.win = MainWindow()
        cls.win._reset_app_settings()
        cls.win.resize(1450, 900)
        cls.win.show()
        cls._pump()

    @classmethod
    def tearDownClass(cls):
        cls.win._set_dirty(False)
        cls.win.close()
        cls.win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()

    @staticmethod
    def _pump(n=5):
        for _ in range(n):
            QApplication.processEvents()

    def test_row_is_in_the_system_tab(self):
        win = self.win
        lay = win.tab_system.layout()
        self.assertGreaterEqual(lay.indexOf(win.combo_gpu_choice), 0)
        self.assertGreaterEqual(lay.indexOf(win.lbl_gpu_now), 0)
        self.assertEqual([win.combo_gpu_choice.itemData(i) for i in range(win.combo_gpu_choice.count())], [G.GPU_HIGH, G.GPU_SYSTEM])
        self.assertEqual(win.combo_gpu_choice.currentData(), G.get_saved_gpu_choice())
        # 行が重なっていない（ほかの行と同じマスに入れていない）
        cells = {}
        for i in range(lay.count()):
            r, c, rs, cs = lay.getItemPosition(i)
            for rr in range(r, r + rs):
                for cc in range(c, c + cs):
                    self.assertNotIn((rr, cc), cells, f"マス {(rr, cc)} に 2 つ入っている")
                    cells[(rr, cc)] = i

    def test_choosing_saves_for_the_next_start(self):
        win = self.win
        combo = win.combo_gpu_choice
        before = combo.currentIndex()
        self.addCleanup(lambda: combo.setCurrentIndex(before))      # ほかのテストへ、選んだままを残さない
        combo.setCurrentIndex(combo.findData(G.GPU_SYSTEM))
        self.assertEqual(G.get_saved_gpu_choice(), G.GPU_SYSTEM)
        combo.setCurrentIndex(combo.findData(G.GPU_HIGH))
        self.assertEqual(G.get_saved_gpu_choice(), G.GPU_HIGH)
        # テストは main.py を通らない = 何も頼んでいない起動。「高性能」は次回からだと分かる
        self.assertIn("次回の起動から", win.lbl_gpu_now.text())
        combo.setCurrentIndex(combo.findData(G.GPU_SYSTEM))
        self.assertNotIn("次回の起動から", win.lbl_gpu_now.text())

    def test_shows_the_gpu_that_is_drawing(self):
        win = self.win
        self._pump()
        QApplication.sendEvent(win.tab_system, QShowEvent())
        self._pump()
        name = G.renderer_name(win.viewer)
        self.assertTrue(name, "描いている GPU の名前が取れない")
        self.assertIn(name, win.lbl_gpu_now.text())


if __name__ == "__main__":
    unittest.main()
