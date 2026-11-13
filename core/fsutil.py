# -*- coding: utf-8 -*-
"""
ファイルの置き換えと、削除を邪魔しない読み方（Windows の事情への備え）。Qt に依存しない。

なぜ要るか（2026-11-14、テストで見つけた）
    Windows では、だれかが開いているファイルへ os.replace で上書きしようとすると「アクセスが拒否されました」になる。
    作業用データの一覧（core/workdata.py）が別スレッドで header.json を読んだ一瞬に、作業用データを作る側の
    header.json の置き換えが重なって失敗し、「読み直しキャッシュを作成できませんでした」で作成全体が止まった。
    同じことは、ウイルス対策や検索の索引作成がファイルを一瞬開いたときにも起きうる（5.4 億点では約 10 分の作業が無駄になる）。

確かめたこと（Windows 11、Python 3.14）
    - 置き換え先が開かれていると、開き方（共有の指定）に関係なく os.replace は断られる
    - 削除を許す開き方（FILE_SHARE_DELETE）で開かれているファイルは、開かれたままでも os.remove できる
      （Python のふつうの open は削除を許さない）

備えは 2 つ
    - replace_file(): 置き換える側は、断られたら少し待って繰り返す（読む側が開いているのは一瞬なので、次で通る）
    - open_shared(): 読む側は、削除を許す開き方をする（古い作業用データの作り直しや削除を、一覧の読み取りが止めない）
"""

import os
import time
from typing import BinaryIO

REPLACE_RETRIES = 40
REPLACE_WAIT_SEC = 0.05   # 合わせて最長 2 秒


def replace_file(src: str, dst: str, retries: int = REPLACE_RETRIES, wait_sec: float = REPLACE_WAIT_SEC) -> None:
    """
    os.replace(src, dst)。置き換え先をほかが開いていて断られたら（PermissionError）、少し待って繰り返す。
    繰り返しても駄目なら、最後の例外をそのまま出す。
    """
    for i in range(int(retries) + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i >= int(retries):
                raise
            time.sleep(wait_sec)


def _open_shared_windows(path: str) -> BinaryIO:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    GENERIC_READ = 0x80000000
    FILE_SHARE_ALL = 0x1 | 0x2 | 0x4          # 読み・書き・削除（置き換え）を、開いている間もほかに許す
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = k32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(os.path.abspath(path), GENERIC_READ, FILE_SHARE_ALL, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except Exception:
        close = k32.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close(handle)
        raise
    return os.fdopen(fd, "rb")


def open_shared(path: str) -> BinaryIO:
    """
    読み取り専用で開く。開いている間も、ほかからの削除・改名を邪魔しない
    （Windows 以外はふつうの open で同じ性質）。with で使う。
    このファイルへの置き換え（os.replace）は、開いている間は Windows が断る。置き換える側は replace_file() を使うこと。
    """
    if os.name == "nt":
        try:
            return _open_shared_windows(path)
        except FileNotFoundError:
            raise
        except OSError:
            pass  # 長すぎるパスなどで開けなかったら、ふつうの開き方へ
    return open(path, "rb")


def read_bytes_shared(path: str, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    """小さなファイル（header.json など）を、削除を邪魔しない開き方で読む"""
    with open_shared(path) as f:
        return f.read(max_bytes)
