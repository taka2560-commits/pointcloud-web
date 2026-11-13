# -*- coding: utf-8 -*-
"""
PC の状態（ディスクの空き・メモリ）を調べる小さな部品。Qt に依存しない。

なぜ要るか（2026-11-13、利用者の PC の確認から）
    C ドライブの空きが約 20GB、作業用データが 1 ファイルで約 17GB、全点で保存が約 16GB。
    空きが足りないまま始めると、数分〜十数分かかる処理が最後に失敗する。
    始める前に確かめるための、空きの取得と大きさの書き方をここに置く。

大きさの書き方は Windows のエクスプローラーに合わせる（1GB = 1024 の 3 乗バイト）。
アプリの表示とエクスプローラーの「空き領域」が同じ数字になる。
"""

import os
import shutil
from typing import Optional, Tuple

KB = 1024
MB = 1024 ** 2
GB = 1024 ** 3


def format_bytes(n: Optional[float]) -> str:
    """バイト数を「17.0GB」「850MB」の形にする（エクスプローラーと同じ数え方）。分からないときは「不明」"""
    if n is None:
        return "不明"
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= GB:
        v = n / GB
        return f"{sign}{v:.0f}GB" if v >= 100 else f"{sign}{v:.1f}GB"
    if n >= MB:
        return f"{sign}{n / MB:.0f}MB"
    if n >= KB:
        return f"{sign}{n / KB:.0f}KB"
    return f"{sign}{int(n)}B"


def nearest_existing_dir(path: str) -> Optional[str]:
    """path か、その親のうち、実在するいちばん近いフォルダ（これから作るファイルの保存先のドライブを調べるのに使う）"""
    p = os.path.abspath(path) if path else ""
    seen = set()
    while p and p not in seen:
        seen.add(p)
        if os.path.isdir(p):
            return p
        p = os.path.dirname(p)
    return None


def disk_status(path: str) -> Optional[Tuple[int, int]]:
    """path のあるドライブの (全体, 空き) バイト。調べられなければ None"""
    d = nearest_existing_dir(path)
    if d is None:
        return None
    try:
        usage = shutil.disk_usage(d)
    except OSError:
        return None
    return int(usage.total), int(usage.free)


def free_bytes(path: str) -> Optional[int]:
    st = disk_status(path)
    return None if st is None else st[1]


def drive_name(path: str) -> str:
    """表示用のドライブ名（「C:」）。ドライブ文字が無い環境ではフォルダのパス"""
    p = os.path.abspath(path) if path else ""
    drive = os.path.splitdrive(p)[0]
    return drive.upper() if drive else (nearest_existing_dir(p) or p)


def same_drive(a: str, b: str) -> bool:
    return drive_name(a) == drive_name(b)


def memory_status() -> Optional[Tuple[int, int]]:
    """物理メモリの (搭載量, 空き) バイト。Windows 以外や取得できないときは None"""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
    except Exception:
        pass
    return None


def can_write_dir(path: str) -> bool:
    """フォルダに実際に書き込めるか（小さなファイルを作って消して確かめる。フォルダは作らない）"""
    if not path or not os.path.isdir(path):
        return False
    probe = os.path.join(path, f".write_test_{os.getpid()}")
    try:
        with open(probe, "w", encoding="utf-8"):
            pass
        os.remove(probe)
        return True
    except OSError:
        try:
            if os.path.exists(probe):
                os.remove(probe)
        except OSError:
            pass
        return False
