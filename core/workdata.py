# -*- coding: utf-8 -*-
"""
作業用データ（読み直しキャッシュと LOD データ）の一覧・大きさ・削除と、空き容量の見積もり。Qt に依存しない。

なぜ要るか（2026-11-13、利用者の PC の確認から）
    作業用データは 5.4 億点のファイル 1 つで約 17GB。デスクトップに 2 つあり（約 34GB）、C ドライブの空きは約 20GB だった。
    アプリからは大きさが見えず、消せず、置き場所も変えられなかった。「全点で保存」は約 16GB 書くのに、
    始める前に空きを確かめていなかった（足りないと、約 10 分走ってから失敗する）。

ここにあるもの
    - find_workdata(): 作業用データのフォルダを探して、大きさと「今の設定で使われるか」を返す
    - remove_workdata(): 1 つ消す（消してよいのは、header.json でこのアプリのものと確かめられたフォルダだけ）
    - estimate_save_bytes() / check_free(): 保存の前に、書く大きさの見込みと空きを比べる
    - workdata_need_for_scans(): 高精細表示のデータを作る前に、要る空きを見積もる

作業用データは、元の E57 と保存した結果の写しにすぎない。消しても編集内容は失われず、必要になれば作り直される。
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.e57_io import file_signature
from core.lod_store import BUCKET_BYTES_PER_POINT, BYTES_PER_POINT as LOD_BYTES_PER_POINT, FREE_SPACE_MARGIN
from core.section_cache import (
    CACHE_SUFFIX, CACHE_VERSION, cache_dir_for, estimate_cache_bytes, get_default_cache_root, read_cache_header,
    remove_cache_directory,
)
from core.sysinfo import GB, MB, drive_name, free_bytes

# 元ファイルの大きさが分からないときの、保存 1 点あたりの大きさ（座標・色・強度・法線つきの実測は約 31 バイト）
DEFAULT_SAVE_BYTES_PER_POINT = 40
SAVE_MARGIN = 1.05
# 書き終えたあとに残しておく空き（ドライブを使い切ると、アプリも Windows も不安定になる）
KEEP_FREE_BYTES = 1 * GB

_RELOAD_FILE_RE = re.compile(r"^scan\d+_[a-z]+\.npy$")

REASON_OLD_FORMAT = "古い形式のため使われません"
REASON_NO_SOURCE = "元ファイルが見つかりません"
REASON_SOURCE_CHANGED = "元ファイルが変わったため使われません"
REASON_OTHER_PLACE = "今の置き場所の設定では使われません"


@dataclass
class WorkData:
    """作業用データのフォルダ 1 つ"""
    directory: str
    source_file: str        # 元ファイル（header.json の記録。一緒に移動されていたら、今の場所）
    size_bytes: int
    n_files: int
    modified: float         # いちばん新しいファイルの更新時刻（0 なら不明）
    has_reload: bool        # 断面の読み直し用のデータがある
    has_lod: bool           # 高精細表示用のデータがある
    source_exists: bool
    usable: bool            # 今の設定で、元ファイルの作業用データとして使われるか
    reason: str             # 使われない理由（使われるなら ""）

    @property
    def contents_text(self) -> str:
        parts = []
        if self.has_reload:
            parts.append("読み直し用")
        if self.has_lod:
            parts.append("高精細表示用")
        return " + ".join(parts) if parts else "空"

    @property
    def source_name(self) -> str:
        return os.path.basename(self.source_file) if self.source_file else os.path.basename(self.directory)


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def inspect_workdata_dir(directory: str) -> Optional[WorkData]:
    """作業用データのフォルダを調べる。このアプリのフォルダと確かめられなければ None（調べるだけで、何も変えない）"""
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return None
    header = read_cache_header(directory)
    if header is None:
        return None
    size = 0
    n_files = 0
    newest = 0.0
    has_reload = False
    has_lod = False
    try:
        with os.scandir(directory) as it:
            for e in it:
                try:
                    if not e.is_file(follow_symlinks=False):
                        continue
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                size += int(st.st_size)
                n_files += 1
                newest = max(newest, float(st.st_mtime))
                if _RELOAD_FILE_RE.match(e.name):
                    has_reload = True
                elif e.name == "lod.json":
                    has_lod = True
    except OSError:
        return None

    # 「使われるか」は、アプリが実際に使うときの決まりに合わせる: 元ファイル S について cache_dir_for(S) がこのフォルダで、
    # header の署名（サイズ・更新時刻）が S と合うこと（header に書いてあるパスは見ない）。
    # 元ファイルの候補は、header の記録と、「<元ファイル>.cache」の形ならすぐ隣のファイル
    # （元ファイルと作業用データをフォルダごと移動した場合、header の記録は古いパスのままだが、アプリは隣のファイルで使う）。
    recorded = str(header.get("source", "") or "")
    candidates = [recorded] if recorded else []
    if directory.lower().endswith(CACHE_SUFFIX):
        sibling = directory[: -len(CACHE_SUFFIX)]
        if sibling and all(not _same_path(sibling, c) for c in candidates):
            candidates.append(sibling)
    try:
        header_sig = tuple(int(v) for v in header.get("signature", ()))
    except (TypeError, ValueError):
        header_sig = None
    source = recorded
    exists_any = matched = usable = False
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        if not exists_any:
            source = cand
        exists_any = True
        try:
            sig_ok = header_sig is not None and header_sig == file_signature(cand)
        except OSError:
            sig_ok = False
        if not sig_ok:
            continue
        if not matched:
            source = cand
        matched = True
        if _same_path(cache_dir_for(cand), directory):
            usable, source = True, cand
            break
    if header.get("version") != CACHE_VERSION:
        usable, reason = False, REASON_OLD_FORMAT
    elif usable:
        reason = ""
    elif not exists_any:
        reason = REASON_NO_SOURCE
    elif not matched:
        reason = REASON_SOURCE_CHANGED
    else:
        reason = REASON_OTHER_PLACE
    source_exists = bool(source) and os.path.isfile(source)
    return WorkData(
        directory=directory, source_file=source, size_bytes=size, n_files=n_files, modified=newest,
        has_reload=has_reload, has_lod=has_lod, source_exists=source_exists, usable=usable, reason=reason,
    )


def find_workdata(source_files: Iterable[str] = (), extra_dirs: Iterable[str] = ()) -> List[WorkData]:
    """
    作業用データのフォルダを探す。探す場所は、source_files（今開いているファイルと最近使ったファイル）のあるフォルダ、
    置き場所の設定のフォルダ、extra_dirs。そこにある「….cache」という名前のフォルダのうち、このアプリのものだけを返す。
    大きい順。調べるだけで、何も変えない。
    """
    folders: Dict[str, str] = {}
    for src in source_files:
        if src:
            p = os.path.abspath(src)
            d = p if os.path.isdir(p) else os.path.dirname(p)  # フォルダごと開いた場合は、そのフォルダの中を探す
            folders.setdefault(os.path.normcase(d), d)
    root = get_default_cache_root()
    for d in ([root] if root else []) + [x for x in extra_dirs if x]:
        d = os.path.abspath(d)
        folders.setdefault(os.path.normcase(d), d)

    found: Dict[str, WorkData] = {}
    for folder in folders.values():
        try:
            with os.scandir(folder) as it:
                names = [e.name for e in it if e.name.lower().endswith(CACHE_SUFFIX) and e.is_dir(follow_symlinks=False)]
        except OSError:
            continue
        for name in names:
            path = os.path.join(folder, name)
            key = os.path.normcase(os.path.abspath(path))
            if key in found:
                continue
            entry = inspect_workdata_dir(path)
            if entry is not None:
                found[key] = entry
    return sorted(found.values(), key=lambda w: (-w.size_bytes, w.directory))


def summarize(entries: Sequence[WorkData]) -> Tuple[int, int, int, int]:
    """(件数, 合計バイト, 使われない件数, 使われない合計バイト)"""
    unused = [e for e in entries if not e.usable]
    return len(entries), sum(e.size_bytes for e in entries), len(unused), sum(e.size_bytes for e in unused)


def remove_workdata(directory: str) -> Tuple[bool, int, List[str]]:
    """
    作業用データのフォルダを 1 つ消す。戻り値: (フォルダごと消えたか, 空いたバイト数, 消さずに残したものの名前)。
    このアプリのフォルダと確かめられないもの、このアプリが作らないファイルは消さない（core.section_cache.remove_cache_directory）。
    """
    if inspect_workdata_dir(directory) is None:
        return False, 0, []
    return remove_cache_directory(directory)


# ----------------------------------------------------------------------------------------------------------------------
# 空きの確認
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class FreeCheck:
    ok: bool
    need_bytes: int
    free_bytes: Optional[int]   # 調べられなければ None（そのときは ok = True。止める根拠が無い）
    drive: str


def check_free(path: str, need_bytes: int, keep_free: int = KEEP_FREE_BYTES) -> FreeCheck:
    """path のドライブに need_bytes を書いても keep_free 以上の空きが残るか"""
    free = free_bytes(path)
    ok = True if free is None else (free >= int(need_bytes) + int(keep_free))
    return FreeCheck(ok=ok, need_bytes=int(need_bytes), free_bytes=free, drive=drive_name(path))


def _scan_points(scan) -> Tuple[int, int]:
    """(読み込んでいる点数, 元ファイルでの点数)"""
    try:
        loaded = int(scan.point_count)
    except Exception:
        loaded = int(len(getattr(scan, "points", ())))
    return loaded, int(getattr(scan, "original_point_count", loaded) or loaded)


def estimate_save_bytes(scans: Sequence, n_out_loaded: int, fullres: bool) -> int:
    """
    保存で書く大きさの見込み。
    - 1 点あたりの大きさは、元ファイルの大きさ ÷ 元ファイルの点数（保存は元ファイルと同じ項目・同じ形式で書くので近い値になる）。
      元ファイルが分からないときは DEFAULT_SAVE_BYTES_PER_POINT。
    - 出力する点数は、間引いたまま保存なら n_out_loaded。全点で保存なら、元ファイルの点数 ×（残す割合）。
    見込みなので、少し多めに見る（SAVE_MARGIN）。
    """
    loaded_total = 0
    orig_total = 0
    src_points = 0
    sources: Dict[str, int] = {}
    for s in scans:
        loaded, orig = _scan_points(s)
        loaded_total += loaded
        orig_total += orig
        src = getattr(s, "source_file", None)
        if src:
            key = os.path.normcase(os.path.abspath(src))
            if key not in sources:
                try:
                    sources[key] = int(os.path.getsize(src))
                except OSError:
                    sources[key] = 0
            if sources[key] > 0:
                src_points += orig
    src_bytes = sum(sources.values())
    per_point = (src_bytes / src_points) if (src_bytes > 0 and src_points > 0) else float(DEFAULT_SAVE_BYTES_PER_POINT)
    n_out = float(max(0, int(n_out_loaded)))
    if fullres and loaded_total > 0:
        n_out = orig_total * (n_out / loaded_total)
    return int(n_out * per_point * SAVE_MARGIN) + 16 * MB


@dataclass
class WorkDataNeed:
    """高精細表示のデータを作るのに要る大きさ"""
    n_points: int
    reload_bytes: int            # これから作る読み直し用（すでにあれば 0）
    lod_bytes: int               # 高精細表示用（出来上がりの大きさ）
    need_free_bytes: int         # 始める時点で要る空き（並べ替えの途中の一時ファイルを含む）
    directory: str               # 置き場所（ファイルが複数なら最初のもの）
    free_bytes: Optional[int]
    drive: str

    @property
    def ok(self) -> bool:
        return self.free_bytes is None or self.free_bytes >= self.need_free_bytes

    @property
    def total_bytes(self) -> int:
        return self.reload_bytes + self.lod_bytes


def workdata_need_for_scans(scans: Sequence) -> WorkDataNeed:
    """
    読み込んでいるスキャンの元ファイルについて、高精細表示のデータを作るのに要る大きさを見積もる。
    読み直し用がすでにあるスキャンは数えない。要る空きは core/lod_store.py の作成時の確認と同じ式
    （並べ替えの途中は 1 点 BUCKET_BYTES_PER_POINT バイトの一時ファイルがいちばん大きい）。
    座標を float64 で持つ元ファイルは読み直し用がこれより大きくなる（そのときは作成時の確認が止める）。
    """
    groups: Dict[str, List] = {}
    order: List[str] = []
    for s in scans:
        src = getattr(s, "source_file", None)
        if not src:
            continue
        key = os.path.abspath(src)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s)

    n_points = 0
    reload_bytes = 0
    lod_bytes = 0
    need = 0
    directory = ""
    for src in order:
        group = groups[src]
        cache_dir = cache_dir_for(src)
        directory = directory or cache_dir
        have = set()
        header = read_cache_header(cache_dir)
        if header is not None and header.get("version") == CACHE_VERSION:
            try:
                if tuple(int(v) for v in header.get("signature", ())) == file_signature(src):
                    have = {int(k) for k in (header.get("scans", {}) or {})}
            except (OSError, TypeError, ValueError):
                have = set()
        n_src = 0
        n_missing = 0
        for s in group:
            n = _scan_points(s)[1]
            n_src += n
            if int(getattr(s, "source_scan_index", -1)) not in have:
                n_missing += n
        missing_bytes = estimate_cache_bytes(n_missing) if n_missing > 0 else 0
        n_points += n_src
        reload_bytes += missing_bytes
        lod_bytes += n_src * LOD_BYTES_PER_POINT
        need += missing_bytes + int(n_src * max(LOD_BYTES_PER_POINT, BUCKET_BYTES_PER_POINT) * FREE_SPACE_MARGIN)
    return WorkDataNeed(
        n_points=n_points, reload_bytes=reload_bytes, lod_bytes=lod_bytes, need_free_bytes=need,
        directory=directory, free_bytes=free_bytes(directory) if directory else None,
        drive=drive_name(directory) if directory else "",
    )
