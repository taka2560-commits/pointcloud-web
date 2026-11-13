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


if __name__ == "__main__":
    unittest.main()
