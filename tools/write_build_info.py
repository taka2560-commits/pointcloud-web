# -*- coding: utf-8 -*-
"""
配布版を作る前に build_info.json（作った日時とコミット）を書く。EXE配布パッケージ作成.bat が呼ぶ。

    python tools/write_build_info.py

build_info.json は PyInstaller が同梱し（E57_PointCloud_Tool.spec）、配布版が起動時に読んで
タイトル・バージョン情報・ログに出す（core/build_info.py）。git の管理には入れない（.gitignore）。
画面への出力は英数字だけ（.bat から呼ばれるので、コンソールの文字コードに依らないようにする）。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.build_info import BUILD_INFO_FILE, make_build_record  # noqa: E402


def main(root: str = ROOT) -> int:
    record = make_build_record(root)
    path = os.path.join(root, BUILD_INFO_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    commit = str(record.get("commit") or "-")
    dirty = " (uncommitted changes)" if record.get("dirty") else ""
    print(f"  build_info.json: built {record['built_at']} / commit {commit}{dirty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
