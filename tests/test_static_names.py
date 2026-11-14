# -*- coding: utf-8 -*-
"""
読み込み漏れ（どこにも定義・import の無い名前）を、全ファイルで機械的に探す

2026-11-14 に 2 件続けて見つけた:
- gui/main_window.py が math を import しておらず、斜め壁の断面を視点追従でシークすると NameError になっていた（2026-10-27 から）
- 作業中に core/section_reload.py へ format_bytes の import を足し忘れた（作業用データを作ったときの記録で落ちる所だった）
どちらも「その行を通るまで分からない」不具合で、ふつうのテストでは通る道が限られていて見つからなかった。

調べ方: ファイルごとに、読まれている名前のうち、そのファイルのどこにも定義（代入・def・class・引数・import・例外の as など）が
無いものを挙げる。関数の中だけで定義された名前も「定義あり」と数えるので、見逃しはあるが、間違って挙げることは少ない。
"""
import ast
import builtins
import glob
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def python_files():
    out = []
    for pattern in ("core/*.py", "gui/*.py", "tools/*.py", "tests/*.py", "*.py"):
        out += glob.glob(os.path.join(ROOT, pattern))
    return sorted(set(out))


def undefined_names(source: str):
    """(名前, 行) の一覧。source のどこにも定義の無い名前を読んでいる所"""
    tree = ast.parse(source)
    defined = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__builtins__", "__spec__", "__package__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
                defined.add(arg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and getattr(node, "name", None):
            defined.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            defined.add(node.rest)
    return sorted({
        (n.id, n.lineno) for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined
    })


class TestNoUndefinedNames(unittest.TestCase):
    def test_checker_finds_a_missing_import(self):
        found = undefined_names("def f(x):\n    return math.sin(x) + helper(x)\n")
        self.assertEqual([name for name, _ in found], ["helper", "math"])
        self.assertEqual(undefined_names("import math\n\ndef helper(v):\n    return v\n\ndef f(x):\n    return math.sin(x) + helper(x)\n"), [])
        self.assertEqual(undefined_names("try:\n    pass\nexcept OSError as e:\n    print(e)\nfor i in range(3):\n    print(i)\n"), [])

    def test_every_source_file_is_clean(self):
        files = python_files()
        self.assertGreater(len(files), 60, "調べるファイルが見つかりません")
        problems = []
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                found = undefined_names(f.read())
            if found:
                rel = os.path.relpath(path, ROOT)
                problems.append(f"{rel}: " + ", ".join(f"{name} ({line} 行)" for name, line in found[:8]))
        self.assertEqual(problems, [], "どこにも定義・import の無い名前を使っています:\n" + "\n".join(problems))


_WIN_DELETE_RE = re.compile(r"\b((?:self\.|cls\.)?win\w*)\.deleteLater\(\)")
_WIN_CLOSE_RE = re.compile(r"^\s*((?:self\.|cls\.)?win\w*)\.close\(\)\s*$")


def window_cleanup_problems(source: str):
    """
    ウィンドウ（win… という名前）を、閉じただけ・破棄を予約しただけで終えている所を挙げる。
    認める書き方は 2 つ: tests/qt_cleanup.py の destroy_widget() を使う / close() → deleteLater() → DeferredDelete を流す。
    """
    lines = source.splitlines()
    problems = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        m = _WIN_DELETE_RE.search(line)
        if m:
            after = "\n".join(lines[i + 1:i + 5])
            if "DeferredDelete" not in after and "flush_deferred_deletes(" not in after:
                problems.append((i + 1, f"{m.group(1)}.deleteLater() のあとで、破棄を実行していません"))
        m = _WIN_CLOSE_RE.match(line)
        if m:
            after = "\n".join(lines[i + 1:i + 3])
            if f"{m.group(1)}.deleteLater()" not in after:
                problems.append((i + 1, f"{m.group(1)}.close() だけで、破棄していません"))
    return problems


class TestWindowsAreReallyDestroyed(unittest.TestCase):
    """
    テストと品質ゲートが、作ったウィンドウを閉じたあと本当に破棄していること。
    閉じただけのウィンドウが残ると、あとのテストが遅くなり、Qt の中でアクセス違反が起きてテストごと落ちることがある
    （2026-11-14 に見つけた。いきさつは tests/qt_cleanup.py）。
    """

    def _files(self):
        files = glob.glob(os.path.join(ROOT, "tests", "*.py")) + [os.path.join(ROOT, "quality_gate.py")]
        return sorted(p for p in files if os.path.basename(p) != os.path.basename(__file__))

    def test_checker_finds_the_two_mistakes(self):
        leak = "win.close()\nwin.deleteLater()\nQApplication.processEvents()\n"
        self.assertEqual([p[0] for p in window_cleanup_problems(leak)], [2])
        only_close = "    win_c.close()\n    os.environ.pop('X', None)\n"
        self.assertEqual([p[0] for p in window_cleanup_problems(only_close)], [1])
        good = (
            "self.win.close()\nself.win.deleteLater()\n"
            "QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)\nQApplication.processEvents()\n"
            "destroy_widget(win)\n"
        )
        self.assertEqual(window_cleanup_problems(good), [])

    def test_every_window_is_destroyed_after_use(self):
        problems = []
        for path in self._files():
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            rel = os.path.relpath(path, ROOT)
            for line, what in window_cleanup_problems(source):
                problems.append(f"{rel} ({line} 行): {what}")
            if "MainWindow()" in source and "destroy_widget(" not in source and "DeferredDelete" not in source:
                problems.append(f"{rel}: MainWindow を作っていますが、破棄していません")
        self.assertEqual(problems, [], "ウィンドウを破棄していない所があります（tests/qt_cleanup.py の destroy_widget を使う）:\n" + "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
