# -*- coding: utf-8 -*-
"""
作業用データの一覧・削除・置き場所と、書く前の空きの見積もり（core/workdata.py、core/sysinfo.py、core/section_cache.py）

2026-11-13、利用者の PC: 作業用データは 5.4 億点で 1 つ約 17GB（2 つで約 34GB）、C ドライブの空きは約 20GB。
アプリからは大きさが見えず、消せず、置き場所も変えられなかった。保存の前に空きを確かめていなかった。

削除は取り返しがつきにくいので、次を確かめる:
- このアプリのフォルダと確かめられないものには触らない
- このアプリが作らないファイルは残す
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import section_cache as SC  # noqa: E402
from core.e57_io import file_signature  # noqa: E402
from core import fsutil  # noqa: E402
from core.lod_store import BUCKET_BYTES_PER_POINT, BYTES_PER_POINT as LOD_BYTES_PER_POINT, FREE_SPACE_MARGIN  # noqa: E402
from core.sysinfo import (  # noqa: E402
    GB, MB, can_write_dir, disk_status, drive_name, format_bytes, free_bytes, memory_status, nearest_existing_dir,
)
from core.workdata import (  # noqa: E402
    DEFAULT_SAVE_BYTES_PER_POINT, KEEP_FREE_BYTES, REASON_NO_SOURCE, REASON_OLD_FORMAT, REASON_OTHER_PLACE,
    REASON_SOURCE_CHANGED, SAVE_MARGIN, check_free, estimate_save_bytes, find_workdata, inspect_workdata_dir,
    remove_workdata, summarize, workdata_need_for_scans,
)


def _write(path: str, n_bytes: int) -> None:
    with open(path, "wb") as f:
        f.write(b"\0" * n_bytes)


def make_source(folder: str, name: str = "現場.e57", n_bytes: int = 1000) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    _write(path, n_bytes)
    return path


def make_cache(directory: str, source: str, version=None, signature=None, scans=(0,), lod=True, xyz_bytes=6000) -> int:
    """作業用データのフォルダを手で作る（中身の配列は空でよい。大きさと名前だけを見る）。合計バイトを返す"""
    os.makedirs(directory)
    sig = list(signature) if signature is not None else list(file_signature(source))
    header = {
        "version": SC.CACHE_VERSION if version is None else version,
        "source": os.path.abspath(source),
        "signature": sig,
        "scans": {str(i): {"n_valid": 10, "has_rgb": True, "has_int": True, "xyz_kind": "int32"} for i in scans},
    }
    with open(os.path.join(directory, "header.json"), "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False)
    for i in scans:
        _write(os.path.join(directory, f"scan{i:02d}_xyz.npy"), xyz_bytes)
        _write(os.path.join(directory, f"scan{i:02d}_rgb.npy"), 1500)
        _write(os.path.join(directory, f"scan{i:02d}_int.npy"), 1000)
        _write(os.path.join(directory, f"scan{i:02d}_chunks.npy"), 64)
    if lod:
        with open(os.path.join(directory, "lod.json"), "w", encoding="utf-8") as f:
            json.dump({"version": 1}, f)
        for name, n in (("lod_xyz.npy", xyz_bytes), ("lod_rgb.npy", 1500), ("lod_int.npy", 1000), ("lod_cells.npy", 80), ("lod_starts.npy", 80)):
            _write(os.path.join(directory, name), n)
    return sum(os.path.getsize(os.path.join(directory, n)) for n in os.listdir(directory))


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="e57_workdata_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(SC.set_default_cache_root, None)
        SC.set_default_cache_root(None)


class TestSysInfo(_TmpCase):
    def test_sizes_read_like_explorer(self):
        self.assertEqual(format_bytes(18_304_833_558), "17.0GB", "5.4 億点の作業用データ（エクスプローラーの表示と同じ）")
        self.assertEqual(format_bytes(920 * GB), "920GB")
        self.assertEqual(format_bytes(99.94 * GB), "99.9GB")
        self.assertEqual(format_bytes(850 * MB), "850MB")
        self.assertEqual(format_bytes(1536), "2KB")
        self.assertEqual(format_bytes(0), "0B")
        self.assertEqual(format_bytes(None), "不明")
        self.assertEqual(format_bytes(-2 * GB), "-2.0GB")

    def test_free_space_of_a_file_that_does_not_exist_yet(self):
        deep = os.path.join(self.tmp, "これから作る", "出力.e57")
        self.assertEqual(os.path.normcase(nearest_existing_dir(deep)), os.path.normcase(self.tmp))
        total, free = disk_status(deep)
        self.assertGreater(total, 0)
        self.assertGreaterEqual(total, free)
        self.assertEqual(free_bytes(deep), disk_status(self.tmp)[1])
        self.assertTrue(drive_name(deep))
        self.assertIsNone(nearest_existing_dir(""))
        self.assertIsNone(disk_status(""))

    def test_can_write_dir(self):
        self.assertTrue(can_write_dir(self.tmp))
        self.assertEqual(os.listdir(self.tmp), [], "確かめるのに使ったファイルを残さない")
        self.assertFalse(can_write_dir(os.path.join(self.tmp, "無いフォルダ")))
        self.assertFalse(can_write_dir(""))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "無いフォルダ")), "フォルダは作らない")

    def test_memory_status(self):
        mem = memory_status()
        if sys.platform == "win32":
            self.assertIsNotNone(mem)
            self.assertGreater(mem[0], 0)
            self.assertGreaterEqual(mem[0], mem[1])


class TestPlace(_TmpCase):
    def test_default_is_beside_the_source(self):
        src = make_source(os.path.join(self.tmp, "a"))
        self.assertEqual(SC.cache_dir_for(src), os.path.abspath(src) + ".cache")
        self.assertIsNone(SC.get_default_cache_root())

    def test_root_setting_moves_every_cache_and_keeps_same_names_apart(self):
        root = os.path.join(self.tmp, "work")
        os.makedirs(root)
        src_a = make_source(os.path.join(self.tmp, "現場A"), "全体.e57")
        src_b = make_source(os.path.join(self.tmp, "現場B"), "全体.e57")
        SC.set_default_cache_root(root)
        da, db = SC.cache_dir_for(src_a), SC.cache_dir_for(src_b)
        for d in (da, db):
            self.assertEqual(os.path.normcase(os.path.dirname(d)), os.path.normcase(os.path.abspath(root)))
            self.assertTrue(os.path.basename(d).startswith("全体.e57."))
            self.assertTrue(d.endswith(".cache"))
        self.assertNotEqual(da, db, "別のフォルダにある同じ名前のファイルどうしで、作業用データを取り合わない")
        self.assertEqual(da, SC.cache_dir_for(src_a), "同じファイルなら毎回同じ場所")
        # 引数で場所を渡す呼び出しは、これまでどおり（設定より優先）
        other = os.path.join(self.tmp, "other")
        self.assertEqual(SC.cache_dir_for(src_a, other), os.path.join(other, "全体.e57.cache"))
        SC.set_default_cache_root("")
        self.assertEqual(SC.cache_dir_for(src_a), os.path.abspath(src_a) + ".cache")

    def test_real_cache_is_created_in_the_root_and_recognised(self):
        """本物の SectionCache が設定した置き場所に作られ、一覧で「使われる」と出る"""
        root = os.path.join(self.tmp, "work")
        os.makedirs(root)
        src = make_source(os.path.join(self.tmp, "a"))
        SC.set_default_cache_root(root)
        cache = SC.SectionCache.open(src, create=True)
        self.assertIsNotNone(cache)
        self.assertEqual(os.path.normcase(os.path.dirname(cache.directory)), os.path.normcase(os.path.abspath(root)))
        self.assertFalse(os.path.exists(src + ".cache"), "元ファイルの横には作らない")
        entries = find_workdata([src])
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].usable, entries[0].reason)
        self.assertEqual(entries[0].contents_text, "空")


class TestInspectAndFind(_TmpCase):
    def test_usable_cache_is_described(self):
        src = make_source(os.path.join(self.tmp, "a"))
        size = make_cache(src + ".cache", src)
        e = inspect_workdata_dir(src + ".cache")
        self.assertIsNotNone(e)
        self.assertEqual(e.size_bytes, size)
        self.assertEqual(e.n_files, 11, "5.4 億点の実物と同じ構成（header + 読み直し用 4 + LOD 6）")
        self.assertTrue(e.has_reload and e.has_lod)
        self.assertEqual(e.contents_text, "読み直し用 + 高精細表示用")
        self.assertTrue(e.source_exists and e.usable)
        self.assertEqual(e.reason, "")
        self.assertEqual(e.source_name, "現場.e57")
        self.assertGreater(e.modified, 0)

    def test_reasons_a_cache_is_not_used(self):
        folder = os.path.join(self.tmp, "a")
        src = make_source(folder)
        make_cache(src + ".cache", src)
        # 古い形式
        old = make_source(folder, "古い.e57")
        make_cache(old + ".cache", old, version=SC.CACHE_VERSION - 1, lod=False)
        self.assertEqual(inspect_workdata_dir(old + ".cache").reason, REASON_OLD_FORMAT)
        # 元ファイルが無い
        gone = make_source(folder, "消した.e57")
        make_cache(gone + ".cache", gone)
        os.remove(gone)
        e = inspect_workdata_dir(gone + ".cache")
        self.assertFalse(e.source_exists)
        self.assertEqual(e.reason, REASON_NO_SOURCE)
        # 元ファイルが変わった
        changed = make_source(folder, "直した.e57")
        make_cache(changed + ".cache", changed, signature=(1, 2))
        self.assertEqual(inspect_workdata_dir(changed + ".cache").reason, REASON_SOURCE_CHANGED)
        # 置き場所の設定を変えた（元ファイルの横のものは、もう使われない）
        root = os.path.join(self.tmp, "work")
        os.makedirs(root)
        SC.set_default_cache_root(root)
        self.assertEqual(inspect_workdata_dir(src + ".cache").reason, REASON_OTHER_PLACE)
        SC.set_default_cache_root(None)
        self.assertTrue(inspect_workdata_dir(src + ".cache").usable)
        for e in (inspect_workdata_dir(p + ".cache") for p in (old, gone, changed)):
            self.assertFalse(e.usable)

    def test_cache_moved_together_with_its_source_is_still_used(self):
        """
        元ファイルと作業用データをフォルダごと移動しても、アプリはそのまま使う（署名が合えばよく、header のパスは見ない）。
        一覧でも「使われる」と出す（「使われないものをまとめて削除」で、使える 17GB を消させない）。
        """
        folder_a = os.path.join(self.tmp, "a")
        src = make_source(folder_a)
        make_cache(src + ".cache", src)
        folder_b = os.path.join(self.tmp, "移動先")
        os.rename(folder_a, folder_b)
        new_src = os.path.join(folder_b, "現場.e57")
        e = inspect_workdata_dir(new_src + ".cache")
        self.assertTrue(e.usable, e.reason)
        self.assertTrue(e.source_exists)
        self.assertEqual(os.path.normcase(e.source_file), os.path.normcase(os.path.abspath(new_src)), "今の場所の元ファイルを出す")
        cache = SC.SectionCache.open(new_src, create=False)
        self.assertIsNotNone(cache, "アプリも実際にこのフォルダを使う")
        self.assertTrue(cache.has_scan(0))
        self.assertEqual(summarize(find_workdata([new_src]))[2], 0, "使われないものには数えない")
        # 隣のファイルが別物になっていたら、使われない
        with open(new_src, "ab") as f:
            f.write(b"x")
        self.assertEqual(inspect_workdata_dir(new_src + ".cache").reason, REASON_SOURCE_CHANGED)
        os.remove(new_src)
        self.assertEqual(inspect_workdata_dir(new_src + ".cache").reason, REASON_NO_SOURCE)

    def test_folders_that_are_not_ours_are_ignored(self):
        folder = os.path.join(self.tmp, "a")
        src = make_source(folder)
        make_cache(src + ".cache", src)
        os.makedirs(os.path.join(folder, "写真.cache"))                       # header.json が無い
        os.makedirs(os.path.join(folder, "壊れ.cache"))
        with open(os.path.join(folder, "壊れ.cache", "header.json"), "w", encoding="utf-8") as f:
            f.write("{ これは JSON ではない")
        os.makedirs(os.path.join(folder, "別物.cache"))
        with open(os.path.join(folder, "別物.cache", "header.json"), "w", encoding="utf-8") as f:
            json.dump({"name": "ほかのアプリの設定"}, f)
        _write(os.path.join(folder, "ファイル.cache"), 10)                     # フォルダではない
        for name in ("写真.cache", "壊れ.cache", "別物.cache", "ファイル.cache", "無い.cache"):
            self.assertIsNone(inspect_workdata_dir(os.path.join(folder, name)), name)
        found = find_workdata([src])
        self.assertEqual([os.path.basename(e.directory) for e in found], ["現場.e57.cache"])

    def test_find_looks_beside_sources_in_the_root_and_in_opened_folders(self):
        fa, fb, root = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b"), os.path.join(self.tmp, "work")
        os.makedirs(root)
        src_a = make_source(fa, "小.e57")
        src_b = make_source(fb, "大.e57")
        orphan_src = make_source(fa, "消した.e57")
        make_cache(src_a + ".cache", src_a, lod=False, xyz_bytes=1000)
        make_cache(src_b + ".cache", src_b, xyz_bytes=9000)
        make_cache(orphan_src + ".cache", orphan_src, xyz_bytes=3000)
        os.remove(orphan_src)
        # 開いたことのないファイルの作業用データも、同じフォルダにあれば見つかる（元ファイルを消したあとの残りを拾える）
        found = find_workdata([src_a, src_b, src_a.upper()])
        self.assertEqual([e.source_name for e in found], ["大.e57", "消した.e57", "小.e57"], "大きい順・重複なし")
        count, total, n_unused, unused_bytes = summarize(found)
        self.assertEqual((count, n_unused), (3, 1))
        self.assertEqual(total, sum(e.size_bytes for e in found))
        self.assertEqual(unused_bytes, found[1].size_bytes)
        # フォルダごと開いた場合は、そのフォルダの中を探す
        self.assertEqual(len(find_workdata([fa])), 2)
        # 置き場所の設定のフォルダも探す
        SC.set_default_cache_root(root)
        in_root = SC.cache_dir_for(src_a)
        make_cache(in_root, src_a, xyz_bytes=500)
        found = find_workdata([])
        self.assertEqual([os.path.normcase(e.directory) for e in found], [os.path.normcase(os.path.abspath(in_root))])
        self.assertTrue(found[0].usable)
        # 無いフォルダ・空の手がかりでは落ちない
        self.assertEqual(find_workdata([os.path.join(self.tmp, "無い", "x.e57"), ""], extra_dirs=["", os.path.join(self.tmp, "none")])[0].directory,
                         os.path.abspath(in_root))

    def test_inspecting_changes_nothing(self):
        """調べるだけで、古い作業用データを消したり作り直したりしない（SectionCache.open は合わないものを消す）"""
        src = make_source(os.path.join(self.tmp, "a"))
        make_cache(src + ".cache", src, signature=(1, 2))
        before = sorted(os.listdir(src + ".cache"))
        find_workdata([src])
        inspect_workdata_dir(src + ".cache")
        workdata_need_for_scans([SimpleNamespace(source_file=src, source_scan_index=0, point_count=5, original_point_count=10)])
        self.assertEqual(sorted(os.listdir(src + ".cache")), before)


class TestRemove(_TmpCase):
    def test_removes_the_whole_cache(self):
        src = make_source(os.path.join(self.tmp, "a"))
        size = make_cache(src + ".cache", src)
        _write(os.path.join(src + ".cache", "scan00_xyz.part.npy"), 77)       # 作りかけ
        _write(os.path.join(src + ".cache", "lod_bucket003.part.npy"), 33)
        ok, freed, kept = remove_workdata(src + ".cache")
        self.assertTrue(ok)
        self.assertEqual(kept, [])
        self.assertGreaterEqual(freed, size + 77 + 33 - 200, "空いた大きさを返す（header は書き換えるので少し変わる）")
        self.assertFalse(os.path.exists(src + ".cache"))
        self.assertTrue(os.path.isfile(src), "元ファイルには触らない")

    def test_keeps_files_that_are_not_ours(self):
        src = make_source(os.path.join(self.tmp, "a"))
        d = src + ".cache"
        make_cache(d, src)
        _write(os.path.join(d, "メモ.txt"), 5)
        _write(os.path.join(d, "別の点群.e57"), 50)
        os.makedirs(os.path.join(d, "下のフォルダ"))
        _write(os.path.join(d, "下のフォルダ", "scan00_xyz.npy"), 9)
        ok, freed, kept = remove_workdata(d)
        self.assertFalse(ok, "フォルダは残る")
        self.assertEqual(sorted(kept), sorted(["メモ.txt", "別の点群.e57", "下のフォルダ"]))
        self.assertGreater(freed, 0)
        self.assertEqual(sorted(os.listdir(d)), sorted(["header.json", "メモ.txt", "別の点群.e57", "下のフォルダ"]))
        self.assertTrue(os.path.isfile(os.path.join(d, "下のフォルダ", "scan00_xyz.npy")), "下のフォルダの中は見ない")
        # 残ったフォルダは「中身の無い作業用データ」になり、消したはずのデータを使おうとしない
        with open(os.path.join(d, "header.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["scans"], {})
        cache = SC.SectionCache.open(src, create=False)
        self.assertIsNotNone(cache)
        self.assertFalse(cache.has_scan(0))

    def test_refuses_folders_that_are_not_ours(self):
        folder = os.path.join(self.tmp, "a")
        photos = os.path.join(folder, "写真.cache")
        os.makedirs(photos)
        _write(os.path.join(photos, "scan00_xyz.npy"), 10)   # 名前が同じ形でも、header.json が無ければ触らない
        _write(os.path.join(photos, "lod.json"), 10)
        plain = os.path.join(folder, "ふつうのフォルダ")
        os.makedirs(plain)
        _write(os.path.join(plain, "header.json"), 10)
        for target in (photos, plain, folder, os.path.join(folder, "無い.cache")):
            self.assertEqual(remove_workdata(target), (False, 0, []), target)
        self.assertEqual(sorted(os.listdir(photos)), ["lod.json", "scan00_xyz.npy"])
        self.assertEqual(os.listdir(plain), ["header.json"])
        self.assertEqual(SC.remove_cache_directory(photos), (False, 0, []))

    def test_own_file_names(self):
        for name in ("header.json", "header.json.tmp", "lod.json", "lod.json.tmp", "scan00_xyz.npy", "scan12_chunks.npy",
                     "scan00_rgb.part.npy", "lod_xyz.npy", "lod_starts.npy", "lod_bucket003.part.npy"):
            self.assertTrue(SC.is_own_cache_file(name), name)
        for name in ("メモ.txt", "scan.e57", "lod_xyz.npy.bak", "x.npy", "header.json.bak", "Scan00_xyz.npy.txt", ""):
            self.assertFalse(SC.is_own_cache_file(name), name)


class TestSpaceChecks(_TmpCase):
    def _scan(self, src, loaded, orig, index=0):
        return SimpleNamespace(source_file=src, source_scan_index=index, point_count=loaded, original_point_count=orig)

    def test_save_size_follows_the_source_file(self):
        src = make_source(os.path.join(self.tmp, "a"), n_bytes=31_000)          # 1,000 点で 31,000 バイト = 1 点 31 バイト
        scans = [self._scan(src, 100, 1000)]
        # 全点で保存: 元ファイルの点数 ×（残す割合）
        self.assertEqual(estimate_save_bytes(scans, 100, True), int(1000 * 31 * SAVE_MARGIN) + 16 * MB)
        self.assertEqual(estimate_save_bytes(scans, 50, True), int(500 * 31 * SAVE_MARGIN) + 16 * MB)
        # 間引いたまま保存: 読み込んでいる点のうち、残す点だけ
        self.assertEqual(estimate_save_bytes(scans, 50, False), int(50 * 31 * SAVE_MARGIN) + 16 * MB)
        # 同じファイルの 2 スキャンは、ファイルの大きさを 1 回だけ数える
        two = [self._scan(src, 50, 500, 0), self._scan(src, 50, 500, 1)]
        self.assertEqual(estimate_save_bytes(two, 100, True), int(1000 * 31 * SAVE_MARGIN) + 16 * MB)

    def test_save_size_without_a_source_uses_a_safe_default(self):
        scans = [SimpleNamespace(source_file=None, point_count=200, original_point_count=200)]
        self.assertEqual(estimate_save_bytes(scans, 200, False), int(200 * DEFAULT_SAVE_BYTES_PER_POINT * SAVE_MARGIN) + 16 * MB)
        missing = [self._scan(os.path.join(self.tmp, "無い.e57"), 10, 10)]
        self.assertEqual(estimate_save_bytes(missing, 10, True), int(10 * DEFAULT_SAVE_BYTES_PER_POINT * SAVE_MARGIN) + 16 * MB)
        self.assertEqual(estimate_save_bytes([], 0, True), 16 * MB)
        # point_count を持たない入れ物でも落ちない
        odd = [SimpleNamespace(source_file=None, points=[1, 2, 3])]
        self.assertGreater(estimate_save_bytes(odd, 3, False), 0)

    def test_real_file_size_is_predicted_closely(self):
        """利用者の実物: 538,377,435 点・16,764,611,584 バイト。全点を残す保存の見込みは、実物より少し多い"""
        src = os.path.join(self.tmp, "big.e57")
        scan = self._scan(src, 30_000_000, 538_377_435)
        real = 16_764_611_584
        orig_getsize = os.path.getsize
        os.path.getsize = lambda p: real if os.path.normcase(str(p)) == os.path.normcase(src) else orig_getsize(p)
        try:
            est = estimate_save_bytes([scan], 30_000_000, True)
        finally:
            os.path.getsize = orig_getsize
        self.assertGreater(est, real)
        self.assertLess(est, real * 1.07)

    def test_check_free(self):
        dest = os.path.join(self.tmp, "出力", "結果.e57")
        free = free_bytes(dest)
        small = check_free(dest, 1024)
        self.assertEqual(small.free_bytes, free)
        self.assertEqual(small.ok, free >= 1024 + KEEP_FREE_BYTES)
        self.assertFalse(check_free(dest, free + 1).ok, "空きより大きければ足りない")
        self.assertFalse(check_free(dest, max(0, free - KEEP_FREE_BYTES + 1)).ok, "書いたあとに残す空きも見る")
        self.assertTrue(check_free(dest, 0, keep_free=0).ok)
        self.assertTrue(small.drive)
        # 空きを調べられないときは、止めない（止める根拠が無い）
        unknown = check_free("", 10 ** 18)
        self.assertIsNone(unknown.free_bytes)
        self.assertTrue(unknown.ok)

    def test_workdata_need_counts_only_what_is_missing(self):
        src = make_source(os.path.join(self.tmp, "a"))
        n = 2_000_000
        scans = [self._scan(src, 1000, n)]
        need = workdata_need_for_scans(scans)
        reload_bytes = SC.estimate_cache_bytes(n)
        lod_peak = int(n * max(LOD_BYTES_PER_POINT, BUCKET_BYTES_PER_POINT) * FREE_SPACE_MARGIN)
        self.assertEqual(need.n_points, n)
        self.assertEqual(need.reload_bytes, reload_bytes)
        self.assertEqual(need.lod_bytes, n * LOD_BYTES_PER_POINT)
        self.assertEqual(need.total_bytes, reload_bytes + n * LOD_BYTES_PER_POINT)
        self.assertEqual(need.need_free_bytes, reload_bytes + lod_peak, "並べ替えの途中の一時ファイルまで見る（core/lod_store.py と同じ式）")
        self.assertEqual(need.directory, os.path.abspath(src) + ".cache")
        self.assertEqual(need.free_bytes, free_bytes(src))
        self.assertTrue(need.drive)
        # 読み直し用がすでにあれば、そのぶんは数えない
        make_cache(src + ".cache", src, lod=False)
        need2 = workdata_need_for_scans(scans)
        self.assertEqual(need2.reload_bytes, 0)
        self.assertEqual(need2.need_free_bytes, lod_peak)
        # 元ファイルが変わっていたら作り直しになる
        with open(src, "ab") as f:
            f.write(b"x")
        self.assertEqual(workdata_need_for_scans(scans).reload_bytes, reload_bytes)

    def test_workdata_need_for_the_real_file_fits_the_real_cache(self):
        """5.4 億点: 出来上がりは実物（18,304,833,558 バイト = 17.0GB）とほぼ同じ。始めるには約 20GB の空きが要る"""
        src = make_source(os.path.join(self.tmp, "a"))
        need = workdata_need_for_scans([self._scan(src, 30_000_000, 538_377_435)])
        self.assertAlmostEqual(need.total_bytes / 18_304_833_558, 1.0, delta=0.002)
        self.assertEqual(format_bytes(need.total_bytes), "17.0GB")
        self.assertEqual(format_bytes(need.need_free_bytes), "20.1GB")

    def test_workdata_need_ok_flag(self):
        src = make_source(os.path.join(self.tmp, "a"))
        self.assertTrue(workdata_need_for_scans([self._scan(src, 10, 1000)]).ok)
        huge = workdata_need_for_scans([self._scan(src, 10, 10 ** 13)])
        self.assertFalse(huge.ok)
        nothing = workdata_need_for_scans([SimpleNamespace(source_file=None, point_count=5, original_point_count=5)])
        self.assertEqual((nothing.n_points, nothing.need_free_bytes, nothing.directory), (0, 0, ""))
        self.assertTrue(nothing.ok)


class TestFsUtil(_TmpCase):
    """
    2026-11-14 にテストで見つけた競合: 作業用データの一覧が header.json を読んだ一瞬に、作る側の置き換えが重なると
    Windows が断り、作業用データの作成全体が止まった。置き換える側が少し待って繰り返す。
    """

    def _pair(self):
        dst = os.path.join(self.tmp, "header.json")
        src = dst + ".tmp"
        with open(dst, "w", encoding="utf-8") as f:
            f.write("old")
        with open(src, "w", encoding="utf-8") as f:
            f.write("new")
        return src, dst

    def test_replace_retries_when_refused(self):
        src, dst = self._pair()
        real = os.replace
        calls, sleeps = [], []

        def flaky(a, b):
            calls.append((a, b))
            if len(calls) <= 3:
                raise PermissionError(13, "アクセスが拒否されました")
            return real(a, b)

        with mock.patch.object(os, "replace", flaky), mock.patch.object(fsutil.time, "sleep", sleeps.append):
            fsutil.replace_file(src, dst)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(sleeps), 3, "断られるたびに少し待つ")
        with open(dst, encoding="utf-8") as f:
            self.assertEqual(f.read(), "new")
        self.assertFalse(os.path.exists(src))

    def test_replace_gives_up_with_the_real_error(self):
        src, dst = self._pair()
        calls = []

        def refuse(a, b):
            calls.append(1)
            raise PermissionError(13, "アクセスが拒否されました")

        with mock.patch.object(os, "replace", refuse), mock.patch.object(fsutil.time, "sleep", lambda s: None):
            with self.assertRaises(PermissionError):
                fsutil.replace_file(src, dst, retries=2)
        self.assertEqual(len(calls), 3, "最初の 1 回 + 繰り返し 2 回")
        # 断られた以外の失敗（元が無いなど）は、繰り返さずにすぐ出す
        with mock.patch.object(fsutil.time, "sleep", lambda s: self.fail("待たない")):
            with self.assertRaises(FileNotFoundError):
                fsutil.replace_file(os.path.join(self.tmp, "無い.tmp"), dst)
        with open(dst, encoding="utf-8") as f:
            self.assertEqual(f.read(), "old", "失敗したら置き換え先はそのまま")

    def test_replace_succeeds_while_something_briefly_holds_the_target(self):
        """置き換え先をほかが 0.25 秒開いていても、待って置き換える（Windows では最初の何回かは断られる）"""
        src, dst = self._pair()
        opened, release = threading.Event(), threading.Event()

        def holder():
            with open(dst, "rb"):
                opened.set()
                release.wait(0.25)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        self.assertTrue(opened.wait(2.0))
        t0 = time.perf_counter()
        fsutil.replace_file(src, dst)
        elapsed = time.perf_counter() - t0
        release.set()
        t.join(2.0)
        with open(dst, encoding="utf-8") as f:
            self.assertEqual(f.read(), "new")
        if os.name == "nt":
            self.assertGreater(elapsed, 0.1, "開かれている間は断られ、閉じられてから通る")
        self.assertLess(elapsed, 2.0)

    def test_shared_open_does_not_block_delete(self):
        """一覧の読み取りが header.json を開いている間も、作り直しや削除は止まらない"""
        _, dst = self._pair()
        with fsutil.open_shared(dst) as f:
            os.remove(dst)
            self.assertEqual(f.read(), b"old", "開いていた側は読み終えられる")
        self.assertFalse(os.path.exists(dst))
        with self.assertRaises(FileNotFoundError):
            fsutil.read_bytes_shared(dst)
        _, dst = self._pair()
        self.assertEqual(fsutil.read_bytes_shared(dst), b"old")
        self.assertEqual(fsutil.read_bytes_shared(dst, max_bytes=2), b"ol")

    def test_listing_does_not_break_header_updates(self):
        """一覧を別スレッドで繰り返し読んでいる最中に header.json を書き換え続けても、失敗しない"""
        src = make_source(os.path.join(self.tmp, "a"))
        cache = SC.SectionCache.open(src, create=True)
        stop = threading.Event()
        reads = []

        def reader():
            while not stop.is_set():
                reads.append(len(find_workdata([src])))
                time.sleep(0.001)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            for i in range(40):
                cache._register_scan(i % 4, 10, True, True, "int32", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        finally:
            stop.set()
            t.join(3.0)
        self.assertGreater(len(reads), 5)
        self.assertTrue(all(n == 1 for n in reads), "読むたびに 1 件見つかる（書き換えの途中でも読める）")
        self.assertEqual(sorted(SC.read_cache_header(cache.directory)["scans"]), ["0", "1", "2", "3"])


if __name__ == "__main__":
    unittest.main()
