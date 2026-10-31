"""
計測ツールの座標読み取りテキスト（gui/measure_text.py）のテスト

- 1 点目で座標行、2 点目で距離行と ΔX/ΔY/ΔZ 行が出ること（小数 3 桁）
- 座標系変換（測量 X/Y 入替など）で読み込んだときは、ファイル系の座標が逆変換で正しく併記されること
- クリップボード文字列がタブ区切りで、ヘッダー・各点・距離の行を持つこと
"""

import unittest

import numpy as np

from gui.measure_text import (
    file_coords,
    measure_clipboard_text,
    measure_distance_text,
    measure_hud_lines,
    measure_status_text,
)


P1 = np.array([100.1234, 200.5678, 10.0], dtype=np.float64)
P2 = np.array([103.1234, 204.5678, 11.5], dtype=np.float64)


class TestFileCoords(unittest.TestCase):
    def test_identity_modes_return_same_point(self):
        for mode in ("cartesian", "", None, "unknown"):
            np.testing.assert_allclose(file_coords(P1, mode), P1)

    def test_survey_swap_inverse(self):
        # survey_swap: 表示 X = ファイル Y, 表示 Y = ファイル X → ファイル系は X/Y を入れ替えた値
        f = file_coords(P1, "survey_swap")
        np.testing.assert_allclose(f, [200.5678, 100.1234, 10.0])

    def test_survey_swap_inv_inverse(self):
        # survey_swap_inv: new = (old_y, -old_x, old_z) → old = (-new_y, new_x, new_z)
        f = file_coords(np.array([1.0, 2.0, 3.0]), "survey_swap_inv")
        np.testing.assert_allclose(f, [-2.0, 1.0, 3.0])

    def test_z_invert_inverse(self):
        f = file_coords(np.array([1.0, 2.0, 3.0]), "z_invert")
        np.testing.assert_allclose(f, [1.0, 2.0, -3.0])


class TestHudLines(unittest.TestCase):
    def test_one_point_shows_coordinates(self):
        lines = measure_hud_lines([P1], "cartesian")
        self.assertEqual(lines, ["P1 X=100.123  Y=200.568  Z=10.000"])

    def test_two_points_add_distance_and_delta(self):
        lines = measure_hud_lines([P1, P2], "cartesian")
        self.assertEqual(lines[0], "P1 X=100.123  Y=200.568  Z=10.000")
        self.assertEqual(lines[1], "P2 X=103.123  Y=204.568  Z=11.500")
        self.assertEqual(lines[2], "直線距離: 5.220m | 水平: 5.000m | 高低差: 1.500m")
        self.assertEqual(lines[3], "ΔX=+3.000  ΔY=+4.000  ΔZ=+1.500")
        self.assertEqual(measure_distance_text([P1, P2]), lines[2])

    def test_transformed_load_adds_file_coordinate_lines(self):
        lines = measure_hud_lines([P1], "survey_swap")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith("   ファイル系 "))
        self.assertIn("X=200.568  Y=100.123  Z=10.000", lines[1])

    def test_third_point_is_ignored_in_lines(self):
        lines = measure_hud_lines([P1, P2, P1], "cartesian")
        self.assertEqual(len(lines), 4)

    def test_status_text(self):
        self.assertIn("座標を表示", measure_status_text([], "cartesian"))
        self.assertTrue(measure_status_text([P1], "cartesian").startswith("P1 X=100.123"))
        self.assertTrue(measure_status_text([P1, P2], "cartesian").startswith("直線距離: 5.220m"))


class TestClipboardText(unittest.TestCase):
    def test_tab_separated_rows(self):
        txt = measure_clipboard_text([P1, P2], "cartesian")
        rows = [r.split("\t") for r in txt.strip("\n").split("\n")]
        self.assertEqual(rows[0], ["点", "X", "Y", "Z"])
        self.assertEqual(rows[1], ["P1", "100.123", "200.568", "10.000"])
        self.assertEqual(rows[2], ["P2", "103.123", "204.568", "11.500"])
        self.assertEqual(rows[3][:2], ["直線距離", "5.220"])
        self.assertEqual(rows[3][2:4], ["水平距離", "5.000"])
        self.assertEqual(rows[3][4:6], ["高低差", "1.500"])
        self.assertEqual(rows[4], ["ΔX", "3.000", "ΔY", "4.000", "ΔZ", "1.500"])
        self.assertTrue(txt.endswith("\n"))

    def test_single_point_has_no_distance_row(self):
        txt = measure_clipboard_text([P1], "cartesian")
        self.assertEqual(len(txt.strip("\n").split("\n")), 2)

    def test_file_columns_when_transformed(self):
        txt = measure_clipboard_text([P1], "survey_swap")
        rows = [r.split("\t") for r in txt.strip("\n").split("\n")]
        self.assertEqual(rows[0], ["点", "X", "Y", "Z", "ファイルX", "ファイルY", "ファイルZ"])
        self.assertEqual(rows[1], ["P1", "100.123", "200.568", "10.000", "200.568", "100.123", "10.000"])


if __name__ == "__main__":
    unittest.main()
