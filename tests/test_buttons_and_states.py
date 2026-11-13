"""
ボタンの役割と状態（gui/button_kit.py）・状態表示（gui/status_feedback.py）・ヒント（gui/hints.py）のテスト
- アイコン: 全ての SVG が指定した色で描ける
- QSS: 役割（soft / ghost / pill）と状態（kbfocus / busy / done / 分割 / 通知 / 未保存 / 入力の誤り）の定義がある
- ボタン: 役割の割り当て、アイコン+文字、無効の理由、処理中 → 完了、オン/オフの印、キーボードの現在位置、分割ボタン
- 通知: 印と種類、点を動かした直後の「元に戻す」、3D 画面の中に収まる
- 未保存の印、入力の誤り（赤い枠と理由）
- ヒント: 状態ごとの提案、カードの実行・閉じる・今後表示しない・戻す、ヒント集、設定の切り替え
- 説明書「ボタンと表示の見方」、従来UIには入らないこと
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import QApplication, QPushButton, QTabWidget

from gui.button_kit import BUTTON_ICONS, DISABLED_REASONS, ROLE_GUIDE, KeyboardFocusRing, disabled_reason
from gui.design_tokens import THEMES, build_modern_qss
from gui.hints import TIPS, HintState, _suggestions, next_suggestion, tip_ids
from gui.icons import ICON_NAMES, icon_path, icon_pixmap
from gui.manual import TOPICS, render_topic, topic_ids
from gui.modern_ui import has_emoji
from gui.status_feedback import (
    UNDO_ACTION_TEXT, classify_message, is_edit_report, validate_input_path, validate_output_path,
)

_SETTING_KEYS = ("launcher_recent", "mode_hints_enabled", "hints_enabled", "hints_never")


def _points(n=4000):
    rng = np.random.default_rng(7)
    return rng.random((n, 3)) * [10.0, 8.0, 3.0]


# ----------------------------------------------------------------------------- 画面を作らないテスト
class TestMessagesAndValidation(unittest.TestCase):
    def test_classify_message(self):
        self.assertEqual(classify_message("5,000 点をノイズレイヤーへ移動しました"), "success")
        self.assertEqual(classify_message("保存を中止しました"), "warning")
        self.assertEqual(classify_message("スクリーンショットの取得に失敗しました"), "error")
        self.assertEqual(classify_message("LOD データを開けませんでした"), "error")
        self.assertEqual(classify_message("視点回転モード (V)"), "info")
        self.assertEqual(classify_message(""), "info")
        # 「…はありませんでした」は良い結果。注意（揺れる通知）にしない
        self.assertEqual(classify_message("周囲に浮遊塊ノイズはありませんでした"), "info")
        self.assertEqual(classify_message("選択範囲内に微細ノイズはありませんでした"), "info")
        self.assertEqual(classify_message("読み込み元ファイルの記録が無いため高精細表示は使えません"), "warning")

    def test_edit_report(self):
        self.assertTrue(is_edit_report("5,000 点をノイズレイヤーへ移動しました (Ctrl+Zで取消可)"))
        self.assertTrue(is_edit_report("自動除去: 1,234 点を高速分離完了"))
        self.assertFalse(is_edit_report("操作を元に戻しました (Undo)"))
        self.assertFalse(is_edit_report("連結塊 300 点を一括選択しました"))
        self.assertFalse(is_edit_report("3 スキャン (9,000 点) を読み込みました"))
        self.assertFalse(is_edit_report("視点回転モード (V)"))
        for msg in ("地面 1,234 点を自動抽出しました", "主平面 900 点を自動検出しました", "5 点を即消去しました",
                    "浮遊塊 77 点を一括除去しました", "全ノイズ 40 点を通常レイヤーへ復元しました"):
            self.assertTrue(is_edit_report(msg), msg)
        # 点数が出てくるだけの文は、編集の報告ではない（「元に戻す」を付けない）
        for msg in ("断面ボックス内の全点表示: OFF（表示 768 点）", "元ファイルから 8,705,925 点を重ね描き",
                    "検出ノイズ 0 点（除去対象なし）", "大容量保護: 5,000,000 点を高速表示 (メモリ圧迫防止)",
                    "高精細表示 (LOD): ON（元ファイル 538,377,435 点。視点が止まると読み込みます）"):
            self.assertFalse(is_edit_report(msg), msg)

    def test_path_validation(self):
        from gui.status_feedback import clean_path_text

        self.assertIsNone(validate_input_path(""))
        self.assertIn("見つかりません", validate_input_path(os.path.join(tempfile.gettempdir(), "no_such_dir_e57", "a.e57")))
        with tempfile.TemporaryDirectory() as d:
            e57 = os.path.join(d, "a.e57")
            txt = os.path.join(d, "a.txt")
            for p in (e57, txt):
                with open(p, "wb") as f:
                    f.write(b"x")
            self.assertIsNone(validate_input_path(e57))
            self.assertIsNone(validate_input_path(d), "フォルダの指定は誤りではない")
            self.assertIn(".e57", validate_input_path(txt))
            self.assertIsNone(validate_output_path(os.path.join(d, "out.e57")))
            # まだ無いフォルダは保存のときに作られるので、誤りにしない（何段あっても同じ）
            self.assertIsNone(validate_output_path(os.path.join(d, "none", "out.e57")))
            self.assertIsNone(validate_output_path(os.path.join(d, "a", "b", "c", "out.e57")))
            # 作れない場所だけを誤りにする: 途中がファイル
            self.assertIn("ファイルです", validate_output_path(os.path.join(txt, "out.e57")))
            self.assertIn("ファイル名", validate_output_path(d))
            # エクスプローラーの「パスのコピー」で付く引用符は外して調べる
            self.assertEqual(clean_path_text(f'  "{e57}"  '), e57)
            self.assertEqual(clean_path_text('"'), '"')
            self.assertIsNone(validate_input_path(f'"{e57}"'))
        free = [c for c in "ZYXWVUTSRQ" if not os.path.exists(f"{c}:\\")]
        if free and os.name == "nt":
            self.assertIn("見つかりません", validate_output_path(f"{free[0]}:\\none\\out.e57"))
        self.assertIsNone(validate_output_path(""))


class TestQssRolesAndStates(unittest.TestCase):
    def test_every_role_and_state_is_defined_in_both_themes(self):
        for t in THEMES.values():
            qss = build_modern_qss(t)
            for role, _look, _usage in ROLE_GUIDE:
                self.assertIn(f'QPushButton[role="{role}"]', qss, f"[{t.name}] 役割 {role} の QSS がありません")
            for role in ("soft", "ghost", "pill", "outline", "segment"):
                self.assertIn(f'QPushButton[role="{role}"]:pressed', qss, f"[{t.name}] {role} の押下がありません")
            for role in ("soft", "ghost", "pill", "outline", "danger", "segment", "primary"):
                self.assertIn(f'QPushButton[role="{role}"]:disabled', qss, f"[{t.name}] {role} の無効がありません")
            for selector in (
                'QPushButton[kbfocus="true"]',
                'QPushButton[busy="true"]',
                'QPushButton[done="true"]',
                'QPushButton[split="arrow"]',
                'QFrame#toast_host[kind="error"]',
                "QPushButton#toast_action",
                "QPushButton#unsaved_pill",
                'QLineEdit[invalid="true"]',
                "QLabel#field_error",
                'QWidget#bottom_bar QPushButton[role="outline"]',
                "QWidget#bottom_bar QPushButton:disabled",
                "QFrame#hint_card",
                'QPushButton[role="segment"]:checked:pressed',
                'QWidget#nav_bar QPushButton[role="segment"]:checked:pressed',
                'QWidget#nav_bar QPushButton[role="danger-toggle"]:hover',
                'QPushButton[split="arrow"][kbfocus="true"]',
            ):
                self.assertIn(selector, qss, f"[{t.name}] QSS に {selector} がありません")

    def test_role_guide_is_documented_in_manual(self):
        self.assertIn("legend", topic_ids())
        legend = next(t for t in TOPICS if t.id == "legend")
        html_text = render_topic(legend, [])
        for _role, look, usage in ROLE_GUIDE:
            self.assertIn(look, html_text)
            self.assertIn(usage.split("。")[0][:8], html_text)
        for word in ("保存中…", "保存済み", "元に戻す", "未保存の変更あり", "理由"):
            self.assertIn(word, html_text)


class TestHintRules(unittest.TestCase):
    def test_no_hint_without_points_or_while_working(self):
        self.assertIsNone(next_suggestion(HintState()))
        s = HintState(has_points=True, n_points=1000)
        self.assertEqual(next_suggestion(s).id, "auto_first")
        self.assertIsNone(next_suggestion(HintState(has_points=True, default_mode=False)), "ツールを使っている間は出さない")
        self.assertIsNone(next_suggestion(HintState(has_points=True, n_selected=10)), "選択中は出さない")

    def test_priority_and_hidden(self):
        s = HintState(has_points=True, n_points=1000, noise_count=500, auto_done=True, dirty=True)
        self.assertEqual(next_suggestion(s).id, "check_noise")
        self.assertIn("500", next_suggestion(s).text)
        self.assertEqual(next_suggestion(s, {"check_noise"}).id, "save")
        s.noise_visible = True
        self.assertEqual(next_suggestion(s).id, "manual_fix")
        self.assertEqual(next_suggestion(s, {"manual_fix"}).id, "save")
        self.assertIsNone(next_suggestion(s, {"manual_fix", "save"}))

    def test_detail_hints_for_subsampled_files(self):
        base = dict(has_points=True, n_points=5_000_000, auto_done=True, subsampled=True, subsample_ratio=4)
        s = HintState(section_active=True, **base)
        self.assertEqual(next_suggestion(s).id, "section_reload")
        self.assertIn("4 点に 1 点", next_suggestion(s).text)
        self.assertIsNone(next_suggestion(HintState(section_active=True, has_overlay=True, **base)))
        self.assertEqual(next_suggestion(HintState(zoomed_in=True, **base)).id, "detail_lod")
        self.assertIsNone(next_suggestion(HintState(zoomed_in=True, lod_on=True, **base)))
        heavy = HintState(has_points=True, n_points=20_000_000, auto_done=True, budget_unlimited=True)
        self.assertEqual(next_suggestion(heavy).id, "heavy_unlimited")

    def test_tips_are_consistent(self):
        self.assertEqual(len(tip_ids()), len(set(tip_ids())), "ヒントの ID が重複しています")
        self.assertGreaterEqual(len(TIPS), 15)
        for tip in TIPS:
            self.assertTrue(tip.text.strip(), tip.id)
            self.assertFalse(has_emoji(tip.text), tip.id)
            self.assertIn(tip.topic_id, topic_ids(), f"ヒント {tip.id} が存在しない説明書の項目を指しています")


# ----------------------------------------------------------------------------- 画面を作るテスト
class _WinCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        from gui.main_window import MainWindow

        s = QSettings("E57PointCloudTool", "Settings")
        self._backup = {k: s.value(k, None) for k in _SETTING_KEYS}
        for k in _SETTING_KEYS:
            s.remove(k)
        self.win = MainWindow()
        self.win._reset_app_settings()
        self.win.resize(1400, 900)
        self.win.show()
        QApplication.processEvents()
        self.ex = self.win.ui_extras
        self.kit = self.win.button_kit

    def tearDown(self):
        self.win._set_dirty(False)
        self.win.close()
        self.win.deleteLater()
        # processEvents() だけでは deleteLater() が実行されず、閉じたウィンドウが残り続けて後のテストが遅くなる。
        # ここで本当に破棄する（破棄後にタイマーなどが例外を出さないことの確認にもなる）
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()
        s = QSettings("E57PointCloudTool", "Settings")
        for k, v in self._backup.items():
            if v is None:
                s.remove(k)
            else:
                s.setValue(k, v)

    def _load(self):
        self.win.viewer.set_point_cloud(_points())
        QApplication.processEvents()

    def _select(self, n):
        win = self.win
        win.viewer.selected_indices = np.arange(n, dtype=np.int64)
        win._update_floating_selection_bar(n)


class TestIcons(_WinCase):
    def test_all_icons_render_in_the_requested_color(self):
        for name in ICON_NAMES:
            self.assertTrue(os.path.isfile(icon_path(name)), f"アイコンのファイルがありません: {name}")
            img = icon_pixmap(name, "#ff0000", 24).toImage()
            painted = [
                img.pixelColor(x, y)
                for x in range(img.width()) for y in range(img.height())
                if img.pixelColor(x, y).alpha() > 200
            ]
            self.assertGreater(len(painted), 8, f"アイコンが描けていません: {name}")
            self.assertTrue(all(c.red() > 200 and c.green() < 60 and c.blue() < 60 for c in painted), f"色が違います: {name}")
        blank = icon_pixmap("no-such-icon", "#ff0000", 16).toImage()
        self.assertEqual(blank.pixelColor(8, 8).alpha(), 0, "無いアイコンは空（文字だけで表示される）")
        from gui.icons import failed_icons

        self.assertIn("no-such-icon", failed_icons(), "描けなかったことを黙って流さない（警告をログに残す）")
        self.assertFalse(set(failed_icons()) & set(ICON_NAMES), "本物のアイコンは全部描ける")


class TestButtonRolesAndIcons(_WinCase):
    def test_roles_follow_the_guide(self):
        win = self.win
        role = lambda name: str(getattr(win, name).property("role") or "")  # noqa: E731
        self.assertEqual(role("save_manual_btn"), "primary")
        self.assertEqual(role("btn_save_settings"), "primary")
        for name in ("delete_pts_btn", "floating_del_btn", "btn_crop_section"):
            self.assertEqual(role(name), "danger", name)
            self.assertEqual(getattr(win, name).property("icon_name"), "trash", f"削除の操作はごみ箱の印: {name}")
        self.assertEqual(role("floating_move_btn"), "outline", "下部バーの塗りつぶしは「ノイズ削除」だけ")
        self.assertEqual(role("btn_expand_all"), "ghost")
        self.assertEqual(role("btn_collapse_all"), "ghost")
        pills = [b.text() for b in win.section_group.findChildren(QPushButton) if b.property("role") == "pill"]
        self.assertEqual(sorted(pills), ["0.1m", "0.5m", "1m", "2m", "5m"])
        soft = [b for b in win.findChildren(QPushButton) if b.property("role") == "soft"]
        # 淡い塗りは、使う回数の少ない操作だけ（ログのフォルダを開く・作業用データの管理）
        self.assertTrue(soft and all("ログフォルダ" in b.text() or b is win.btn_workdata_manage for b in soft))
        self.assertIn(win.btn_workdata_manage, soft)
        self.assertEqual(win.btn_toggle_sec_guides.text(), "切替", "幅に収まらない文字を残さない")

    def test_icon_buttons_keep_text_and_have_no_emoji(self):
        win = self.win
        for name, icon in BUTTON_ICONS.items():
            b = getattr(win, name, None)
            if b is None:
                continue
            self.assertFalse(b.icon().isNull(), f"アイコンがありません: {name}")
            self.assertEqual(b.property("icon_name"), icon)
            self.assertTrue(b.text().strip(), f"アイコンだけにしない（文字も出す）: {name}")
        self.assertEqual(win.auto_undo_btn.text(), "1つ戻る")
        self.assertEqual(win.undo_btn.text(), "戻す")
        self.assertIn("Ctrl+Z", win.undo_btn._disabled_reason_helper._normal_tip)
        self.assertEqual(win.auto_redo_btn.text(), "1つ進む")
        self.assertFalse(self.ex.nav_launcher_btn.icon().isNull())
        self.assertFalse(win.empty_state_card.btn_open.icon().isNull())
        for b in win.findChildren(QPushButton):
            self.assertFalse(has_emoji(b.text()), b.text())

    def test_disabled_buttons_explain_why(self):
        win = self.win
        self._load()
        for name, reason in DISABLED_REASONS.items():
            self.assertEqual(disabled_reason(getattr(win, name)), reason, name)
        self.assertFalse(win.floating_del_btn.isEnabled())
        self.assertEqual(win.floating_del_btn.toolTip(), DISABLED_REASONS["floating_del_btn"])
        self.assertIn("点を囲む", win.floating_sel_badge.text(), "押せない理由をバーにも書く")
        wand_tip = "選択した点と3Dで繋がっている塊全体を一発選択します (G)"
        self.assertEqual(win.floating_wand_btn.toolTip(), DISABLED_REASONS["floating_wand_btn"])
        self._select(40)
        self.assertTrue(win.floating_del_btn.isEnabled())
        self.assertNotEqual(win.floating_del_btn.toolTip(), DISABLED_REASONS["floating_del_btn"])
        self.assertEqual(win.floating_wand_btn.toolTip(), wand_tip, "使えるときは元の説明に戻る")
        self.assertFalse(win.auto_undo_btn.isEnabled())
        self.assertEqual(win.auto_undo_btn.toolTip(), DISABLED_REASONS["auto_undo_btn"])

    def test_toggle_marks(self):
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QIcon

        from gui.button_kit import ToggleMark, install_toggle_mark

        win = self.win
        btn = win.btn_instant_del
        mark = btn._toggle_mark
        self.assertEqual(mark.mark(), "off")
        icon = btn.icon()
        on_img = icon.pixmap(QSize(14, 14), QIcon.Normal, QIcon.On).toImage()
        off_img = icon.pixmap(QSize(14, 14), QIcon.Normal, QIcon.Off).toImage()
        self.assertNotEqual(on_img, off_img, "オン用とオフ用の絵を 1 つのアイコンに登録する")

        # 本体は blockSignals の中で setChecked する（N キーのノイズ専用、Esc でのピック終了など）。印が取り残されないこと
        win._toggle_clean_focus_mode(True)
        self.assertTrue(win.btn_clean_mode.isChecked() and win.btn_instant_del.isChecked())
        self.assertEqual(win.btn_clean_mode._toggle_mark.mark(), "on")
        self.assertEqual(mark.mark(), "on")
        win._toggle_clean_focus_mode(False)
        self.assertEqual(win.btn_clean_mode._toggle_mark.mark(), "off")

        # 実際に描かれる絵で確かめる: シグナルを止めて切り替えても、オンならオンの色の印が描かれる
        class _ColorMark(ToggleMark):
            def _colors(self, t):
                return "#ff00ff", "#ffff00"

        plain = QPushButton("ON/OFF", win)
        plain.setCheckable(True)
        plain._toggle_mark = _ColorMark(plain, self.kit.tokens)
        plain.resize(plain.sizeHint())
        plain.show()
        QApplication.processEvents()

        def count(img, pred):
            m = int(4 * img.devicePixelRatio())  # 枠と角の境目は数えない
            return sum(
                1 for x in range(m, img.width() - m) for y in range(m, img.height() - m) if pred(img.pixelColor(x, y))
            )

        magenta = lambda c: c.red() > 180 and c.blue() > 180 and c.green() < 90  # noqa: E731
        yellow = lambda c: c.red() > 180 and c.green() > 180 and c.blue() < 90  # noqa: E731
        w0 = plain.sizeHint().width()
        img = plain.grab().toImage()
        self.assertGreater(count(img, yellow), 3)
        self.assertEqual(count(img, magenta), 0)
        plain.blockSignals(True)
        plain.setChecked(True)
        plain.blockSignals(False)
        QApplication.processEvents()
        img = plain.grab().toImage()
        self.assertGreater(count(img, magenta), 3, "シグナルを止めた切り替えでも、オンの印に変わる")
        self.assertEqual(count(img, yellow), 0)
        self.assertEqual(plain.sizeHint().width(), w0, "印の切り替えで幅が変わらない")
        self.assertIs(install_toggle_mark(plain, self.kit.tokens), plain._toggle_mark)
        plain.deleteLater()

    def test_keyboard_focus_ring_only_for_keyboard(self):
        ring = self.kit.focus_ring
        self.assertIsInstance(ring, KeyboardFocusRing)
        btn = self.win.btn_fit
        ring.eventFilter(btn, QFocusEvent(QEvent.FocusIn, Qt.MouseFocusReason))
        self.assertFalse(bool(btn.property("kbfocus")), "マウスで押しただけでは枠を出さない")
        ring.eventFilter(btn, QFocusEvent(QEvent.FocusIn, Qt.TabFocusReason))
        self.assertTrue(bool(btn.property("kbfocus")))
        ring.eventFilter(btn, QFocusEvent(QEvent.FocusOut, Qt.TabFocusReason))
        self.assertFalse(bool(btn.property("kbfocus")))

    def test_split_save_button(self):
        win = self.win
        split = win.save_split
        self.assertIs(split.main_btn, win.save_manual_btn)
        self.assertEqual(win.save_manual_btn.property("split"), "main")
        self.assertEqual(split.arrow_btn.property("split"), "arrow")
        self.assertTrue(split.arrow_btn.toolTip())
        titles = [a.text() for a in split.menu.actions() if not a.isSeparator()]
        self.assertEqual(len(titles), 3)
        self.assertTrue(any("保存先を選んで保存" in t for t in titles))
        self.assertTrue(any(".npz" in t for t in titles))
        self.assertTrue(win.save_group.isVisible() and win.save_manual_btn.isVisible())
        self.assertGreater(win.save_manual_btn.width(), win.export_mode_combo.width() * 0.6, "保存は全幅の主導線")


class TestBusyAndDone(_WinCase):
    def test_save_button_busy_then_done(self):
        win = self.win
        btn = win.save_manual_btn
        binding = self.kit.busy["save_manual_btn"]
        arrow = win.save_split.arrow_btn
        normal = btn.text()
        self.assertEqual(normal, "保存 (Ctrl+S)")
        self.assertEqual(binding.state(), "idle")

        btn.setEnabled(False)  # 本体が保存を始めるときの動き
        self.assertEqual(binding.state(), "busy")
        self.assertEqual(btn.text(), "保存中…")
        self.assertTrue(bool(btn.property("busy")))
        self.assertFalse(arrow.isEnabled(), "処理中は関連操作のメニューも押せない")
        self.assertEqual(btn.toolTip(), DISABLED_REASONS["save_manual_btn"])

        btn.setEnabled(True)   # 終わった
        self.assertEqual(binding.state(), "idle")
        self.assertEqual(btn.text(), normal)
        self.assertFalse(bool(btn.property("busy")))
        self.assertTrue(arrow.isEnabled())

        win._set_dirty(False)  # 保存成功の直後に本体が呼ぶ
        self.assertEqual(binding.state(), "done")
        self.assertEqual(btn.text(), "保存済み")
        self.assertTrue(bool(btn.property("done")))
        binding._end_done()
        self.assertEqual(binding.state(), "idle")
        self.assertEqual(btn.text(), normal)
        self.assertFalse(bool(btn.property("done")))
        self.assertEqual(btn.property("icon_name"), "save")
        self.assertFalse(btn.icon().isNull())

    def test_failed_save_does_not_show_done(self):
        win = self.win
        binding = self.kit.busy["save_manual_btn"]
        win.save_manual_btn.setEnabled(False)
        win.save_manual_btn.setEnabled(True)  # 失敗のときは未保存フラグを下ろさない
        self.assertEqual(binding.state(), "idle")
        binding._left_busy_at -= 10.0         # 時間がたってから別の理由でフラグが下りても「保存済み」にしない
        win._set_dirty(False)
        self.assertEqual(binding.state(), "idle")

    def test_execute_and_settings_buttons(self):
        win = self.win
        exec_binding = self.kit.busy["auto_exec_btn"]
        text = win.auto_exec_btn.text()
        win.auto_exec_btn.setEnabled(False)
        self.assertEqual(win.auto_exec_btn.text(), "実行中…")
        win.auto_exec_btn.setEnabled(True)
        self.assertEqual(win.auto_exec_btn.text(), text)
        self.assertEqual(exec_binding.state(), "idle")

        set_binding = self.kit.busy["btn_save_settings"]
        label = win.btn_save_settings.text()
        set_binding.flash_done()
        self.assertEqual(win.btn_save_settings.text(), "保存済み")
        set_binding._end_done()
        self.assertEqual(win.btn_save_settings.text(), label)


class TestToastAndUnsaved(_WinCase):
    def test_toast_kinds_and_placement(self):
        win = self.win
        host = win.toast_host
        self.assertIs(host.parentWidget(), win.viewer, "通知は 3D 画面の上に浮かせる（下部バーを押し広げない）")
        bar_h = win.bottom_overlay_bar.height()
        win.show_toast("テスト: 保存しました")
        QApplication.processEvents()
        self.assertTrue(win.toast_label.isVisible())
        self.assertEqual(host.kind(), "success")
        self.assertFalse(host.has_action())
        self.assertIsNotNone(host.icon_label.pixmap())
        self.assertTrue(win.viewer.rect().contains(host.geometry()), "通知が 3D 画面からはみ出しています")
        self.assertEqual(win.bottom_overlay_bar.height(), bar_h, "通知で下部バーの高さが変わらない")
        win.show_toast("スクリーンショットの取得に失敗しました")
        self.assertEqual(host.kind(), "error")
        win.show_toast("読み込みを中止しました")
        self.assertEqual(host.kind(), "warning")
        win.show_toast("とても長いお知らせです。" * 30)
        QApplication.processEvents()
        self.assertTrue(win.viewer.rect().contains(host.geometry()), "長い文は折り返して画面に収める")
        self.assertFalse(has_emoji(win.toast_label.text()))

    def test_undo_action_after_edit(self):
        win = self.win
        host = win.toast_host
        self._load()
        win.show_toast("視点回転モード (V)")
        self.assertFalse(host.has_action())
        self._select(120)
        win._delete_selected_points()
        QApplication.processEvents()
        self.assertTrue(host.has_action(), "点を動かした直後の通知には「元に戻す」を付ける")
        self.assertEqual(host.action_btn.text(), UNDO_ACTION_TEXT)
        self.assertGreaterEqual(win._toast_timer.remainingTime(), 4000, "押す時間が取れるように長めに出す")
        self.assertTrue(win.layer_history.can_undo())
        self.assertEqual(int(np.sum(~win.viewer.layer_mask)), 120)
        host.action_btn.click()
        QApplication.processEvents()
        self.assertEqual(int(np.sum(~win.viewer.layer_mask)), 0, "「元に戻す」でノイズへ移した点が戻る")
        self.assertFalse(host.has_action(), "取り消しの通知には付けない")
        win.show_toast("別の 10 点のお知らせ")
        self.assertFalse(host.has_action(), "編集のあとでない通知には付けない")

    def test_unsaved_pill(self):
        win = self.win
        pill = win.unsaved_pill
        self.assertTrue(pill.isHidden())
        self._load()
        win._push_layer_undo_state()
        self.assertTrue(win._dirty)
        self.assertFalse(pill.isHidden(), "未保存の編集があるときに出す")
        self.assertIn("未保存", pill.text())
        self.assertFalse(pill.icon().isNull(), "色だけでなく印も付ける")
        self.assertTrue(win.windowTitle().startswith("* "))
        win._set_dirty(False)
        self.assertTrue(pill.isHidden())

    def test_field_errors(self):
        win = self.win
        errs = win.field_errors
        self.assertTrue(errs.row.isHidden())
        missing = os.path.join(tempfile.gettempdir(), "no_such_dir_e57", "missing.e57")
        win.input_edit.setText(missing)
        self.assertTrue(errs.row.isHidden(), "打っている途中では誤りを出さない")
        win.input_edit.editingFinished.emit()
        self.assertFalse(errs.row.isHidden())
        self.assertTrue(bool(win.input_edit.property("invalid")))
        self.assertIn("入力 E57", errs.label.text())
        with tempfile.TemporaryDirectory() as d:
            blocker = os.path.join(d, "file.txt")
            with open(blocker, "wb") as f:
                f.write(b"x")
            # まだ無いフォルダへの保存は成功する（保存のときに作られる）ので、誤りにしない
            win.output_edit.setText(os.path.join(d, "new_folder", "out.e57"))
            win.output_edit.editingFinished.emit()
            self.assertEqual(len(errs.messages()), 1)
            self.assertFalse(bool(win.output_edit.property("invalid")))
            # 途中がファイルで、フォルダを作れない場所は誤り
            win.output_edit.setText(os.path.join(blocker, "out.e57"))
            win.output_edit.editingFinished.emit()
            self.assertEqual(len(errs.messages()), 2)
            self.assertTrue(bool(win.output_edit.property("invalid")))
            win.input_edit.setText("")
            self.assertFalse(bool(win.input_edit.property("invalid")), "直したらすぐ消す")
            self.assertEqual(len(errs.messages()), 1)
            # 引用符付きで貼られたパスは、確定したときに欄の文字を直す（本体は引用符を外さないため）
            good = os.path.join(d, "out.e57")
            win.output_edit.setText(f'"{good}"')
            win.output_edit.editingFinished.emit()
            self.assertEqual(win.output_edit.text(), good)
            self.assertTrue(errs.row.isHidden())
            win.input_edit.setText(f'"{missing}"')
            win.input_edit.editingFinished.emit()
            self.assertEqual(win.input_edit.text(), missing)
            self.assertIn("入力 E57", errs.label.text())
            win.input_edit.setText("")
        win.output_edit.setText("")
        self.assertTrue(errs.row.isHidden())

    def test_theme_switch_repaints_icons(self):
        win = self.win
        other = "dark" if getattr(win, "ui_theme", "dark") != "dark" else "kiseki"
        key_before = win.save_manual_btn.icon().cacheKey()
        win.apply_ui_theme(other)
        QApplication.processEvents()
        self.assertFalse(win.save_manual_btn.icon().isNull())
        self.assertNotEqual(win.save_manual_btn.icon().cacheKey(), key_before, "テーマの色で描き直す")
        self.assertEqual(win.btn_instant_del._toggle_mark.mark(), "off")
        win.show_toast("テーマを切り替えました")
        self.assertTrue(win.toast_label.isVisible())


class TestHintsOnWindow(_WinCase):
    def test_commands_referenced_by_hints_exist(self):
        ids = {c.id for c in self.ex.commands()}
        for tip in TIPS:
            if tip.command_id:
                self.assertIn(tip.command_id, ids, f"ヒント {tip.id} が存在しないコマンドを指しています")
        states = (
            HintState(has_points=True),
            HintState(has_points=True, noise_count=5, auto_done=True, dirty=True),
            HintState(has_points=True, noise_count=5, noise_visible=True, auto_done=True, dirty=True),
            HintState(has_points=True, auto_done=True, subsampled=True, section_active=True),
            HintState(has_points=True, auto_done=True, subsampled=True, zoomed_in=True),
            HintState(has_points=True, n_points=20_000_000, auto_done=True, budget_unlimited=True),
        )
        seen = set()
        for st in states:
            for sug in _suggestions(st):
                seen.add(sug.id)
                self.assertIn(sug.command_id, ids, f"ヒント {sug.id} が存在しないコマンドを指しています")
                self.assertFalse(has_emoji(sug.text + sug.action_label))
        self.assertEqual(seen, {"auto_first", "check_noise", "manual_fix", "save", "section_reload", "detail_lod", "heavy_unlimited"})
        for cid in ("help.tips", "help.hints"):
            self.assertIn(cid, ids)

    def test_card_shows_runs_and_hides(self):
        win, engine = self.win, self.ex.hint_engine
        card = engine.card
        self.assertTrue(engine.is_enabled())
        engine.refresh(force=True)
        self.assertTrue(card.isHidden(), "点群が無いときは出さない")
        self._load()
        engine.refresh(force=True)
        self.assertFalse(card.isHidden())
        self.assertEqual(engine.current.id, "auto_first")
        self.assertEqual(card.btn_action.text(), "①自動 を開く")
        self.assertTrue(win.viewer.rect().contains(card.geometry()))
        win._set_sidebar_workflow_tab("view")
        card.btn_action.click()
        self.assertTrue(card.isHidden(), "実行したら閉じる")
        self.assertTrue(win.auto_group.isVisible(), "ヒントのボタンでその機能が動く")
        engine.refresh(force=True)
        self.assertTrue(card.isHidden(), "閉じたヒントは同じ起動中には出さない")

        win._set_interaction_mode("rect")
        engine.reset_hidden()
        self.assertTrue(card.isHidden(), "ツールを使っている間は出さない")
        win._set_interaction_mode("orbit")
        engine.refresh(force=True)
        self.assertFalse(card.isHidden())

    def test_never_show_again_and_reset(self):
        engine = self.ex.hint_engine
        self._load()
        engine.refresh(force=True)
        self.assertEqual(engine.current.id, "auto_first")
        engine.card.btn_never.click()
        self.assertIn("auto_first", engine.hidden_ids())
        stored = QSettings("E57PointCloudTool", "Settings").value("hints_never", [])
        stored = [stored] if isinstance(stored, str) else list(stored or [])
        self.assertIn("auto_first", stored, "今後表示しないは次の起動にも引き継ぐ")
        engine.refresh(force=True)
        self.assertTrue(engine.card.isHidden())
        self.ex.manual_tab.btn_reset_hints.click()
        self.assertEqual(engine.hidden_ids(), set())
        self.assertFalse(engine.card.isHidden(), "戻したらまた出る")

    def test_setting_toggle_and_tips_dialog(self):
        ex, engine = self.ex, self.ex.hint_engine
        self._load()
        engine.refresh(force=True)
        self.assertFalse(engine.card.isHidden())
        ex.manual_tab.chk_tips.setChecked(False)
        self.assertFalse(engine.is_enabled())
        self.assertTrue(engine.card.isHidden())
        self.assertTrue(ex.run_command("help.hints"))
        self.assertTrue(engine.is_enabled())
        self.assertTrue(ex.manual_tab.chk_tips.isChecked(), "コマンドで切り替えても設定のチェックが合う")

        self.assertTrue(ex.run_command("help.tips"))
        dlg = ex.tips_dialog()
        self.assertTrue(dlg.isVisible())
        ex.open_tips("fit")
        self.assertEqual(dlg.current_tip().id, "fit")
        self.assertFalse(dlg.btn_try.isHidden())
        self.assertIn("全体", dlg.btn_try.text())
        n = len(TIPS)
        self.assertIn(f"/ {n}", dlg.counter_label.text())
        idx = tip_ids().index("fit")
        dlg.step(+1)
        self.assertEqual(dlg.current_tip().id, tip_ids()[(idx + 1) % n])
        dlg.step(-1)
        ex.open_tips("search")
        self.assertTrue(dlg.btn_try.isHidden(), "試す機能が無いヒントにはボタンを出さない")
        dlg.btn_manual.click()
        self.assertEqual(ex.manual_window().browser.current_topic_id(), "quick")
        ex.manual_window().close()
        dlg.close()
        self.assertTrue(ex.startup_tip_label.text().startswith("ヒント: "))


class TestBugCheck20261111(_WinCase):
    """2026-11-11 のエラー・バグ点検で見つけた不具合の再発防止"""

    def test_settings_are_sandboxed(self):
        from tests.settings_sandbox import is_sandboxed

        self.assertTrue(is_sandboxed(), "テストがこの PC の本番アプリの保存設定を書き換えています")
        self.assertTrue(QSettings("E57PointCloudTool", "Settings").fileName().lower().endswith(".ini"))

    def test_undo_action_only_for_the_edit_just_made(self):
        win, host = self.win, self.win.toast_host
        self._load()
        self._select(50)
        win._delete_selected_points()   # 取り消せる編集を 1 つ作っておく
        QApplication.processEvents()
        win.show_toast("視点回転モード (V)")
        self.assertFalse(host.has_action())

        # 通知を出さない編集（断面外のクロップなど）のあと、別の操作の通知に「元に戻す」が付いてはいけない
        win._push_layer_undo_state()
        QApplication.processEvents()     # その編集の処理が終わり、イベントループへ戻った
        win.show_toast("12 点をノイズレイヤーへ移動しました")
        self.assertFalse(host.has_action(), "前の編集の印が残って、あとの通知に「元に戻す」が付いています")

        # 編集と同じ処理の中でも、点数が出るだけの文には付けない
        win._push_layer_undo_state()
        win.show_toast("断面ボックス内の全点表示: OFF（表示 768 点）")
        self.assertFalse(host.has_action())

        # 編集と同じ処理の中で出た編集の報告には付く
        win._push_layer_undo_state()
        win.show_toast("12 点をノイズレイヤーへ移動しました")
        self.assertTrue(host.has_action())

    def test_real_crop_does_not_leak_undo_to_later_toast(self):
        win, host = self.win, self.win.toast_host
        self._load()
        win.chk_clip_x.setChecked(True) if hasattr(win, "chk_clip_x") else None
        if not win.viewer.is_section_active():
            self.ex.run_command("section.x")
        QApplication.processEvents()
        if not win.viewer.is_section_active():
            self.skipTest("断面を有効にできない構成")
        from PySide6.QtWidgets import QMessageBox

        orig_q, orig_i = QMessageBox.question, QMessageBox.information
        QMessageBox.question = lambda *a, **k: QMessageBox.Yes
        QMessageBox.information = lambda *a, **k: QMessageBox.Ok
        try:
            n_before = int(np.sum(~win.viewer.layer_mask))
            win._crop_outside_section()
        finally:
            QMessageBox.question, QMessageBox.information = orig_q, orig_i
        moved = int(np.sum(~win.viewer.layer_mask)) - n_before
        self.assertGreater(moved, 0, "断面の外に点がある配置で検査すること")
        # クロップそのものの通知には「元に戻す」が付き、押すとクロップだけが戻る
        self.assertIn("断面ボックス外", win.toast_label.text())
        self.assertTrue(host.has_action(), "クロップの結果の通知に「元に戻す」が付いていません")
        QApplication.processEvents()
        win.show_toast("断面ボックス内の全点表示: OFF（表示 768 点）")
        self.assertFalse(host.has_action(), "クロップのあとの無関係な通知に「元に戻す」が付いています")
        win.show_toast(f"{moved:,} 点をノイズレイヤーへ移動しました")  # 同じ文でも、編集の直後でなければ付けない
        self.assertFalse(host.has_action())

    def test_action_toast_survives_a_modal_dialog(self):
        from PySide6.QtWidgets import QDialog

        from gui.status_feedback import ACTION_TOAST_MS

        win, host = self.win, self.win.toast_host
        self._load()
        self._select(30)
        win._delete_selected_points()
        self.assertTrue(host.has_action())
        dlg = QDialog(win)
        dlg.setModal(True)
        dlg.show()
        QApplication.processEvents()
        self.assertIsNotNone(QApplication.activeModalWidget())
        win._toast_timer.stop()
        win._toast_timer.timeout.emit()   # 完了ダイアログの裏で時間が来た
        self.assertFalse(host.isHidden(), "ダイアログが開いているあいだは「元に戻す」を消さない")
        self.assertTrue(win._toast_timer.isActive())
        dlg.close()
        dlg.deleteLater()
        QApplication.processEvents()
        win._toast_timer.stop()
        win._toast_timer.timeout.emit()   # 閉じたあと最初の確認
        self.assertFalse(host.isHidden())
        self.assertGreaterEqual(win._toast_timer.remainingTime(), ACTION_TOAST_MS - 500, "閉じてから押す時間を取り直す")
        self.assertTrue(host.has_action())
        # 「元に戻す」の無い通知は、ダイアログが開いていても時間どおりに消してよい
        win.show_toast("視点回転モード (V)")
        self.assertFalse(host.has_action())

    def test_done_is_cleared_when_edited_again(self):
        win = self.win
        binding = self.kit.busy["save_manual_btn"]
        self._load()
        binding.flash_done("保存済み")
        self.assertEqual(win.save_manual_btn.text(), "保存済み")
        win._push_layer_undo_state()      # 保存の直後にまた編集した
        self.assertTrue(win._dirty)
        self.assertEqual(binding.state(), "idle", "「保存済み」と「未保存の変更あり」を同時に出さない")
        self.assertEqual(win.save_manual_btn.text(), "保存 (Ctrl+S)")
        self.assertFalse(win.unsaved_pill.isHidden())

    def test_icon_follows_theme_changed_while_busy_or_done(self):
        from gui.button_kit import icon_colors_for, tokens_for
        from gui.icons import themed_icon

        win = self.win
        btn = win.save_manual_btn
        binding = self.kit.busy["save_manual_btn"]

        def expected():
            color, dis = icon_colors_for(btn, tokens_for(win))
            return themed_icon("save", color, dis, 14).pixmap(14, 14).toImage()

        for enter in ("busy", "done"):
            other = "dark" if getattr(win, "ui_theme", "dark") != "dark" else "kiseki"
            if enter == "busy":
                btn.setEnabled(False)
            else:
                binding.flash_done("保存済み")
            win.apply_ui_theme(other)
            QApplication.processEvents()
            if enter == "busy":
                btn.setEnabled(True)
            else:
                binding._end_done()
            self.assertEqual(binding.state(), "idle")
            self.assertEqual(btn.icon().pixmap(14, 14).toImage(), expected(), f"{enter} 中にテーマを替えると、戻ったアイコンが古い色のまま")

    def test_split_arrow_keeps_size_and_releases_after_menu(self):
        from gui.button_kit import repolish

        arrow = self.win.save_split.arrow_btn
        QApplication.processEvents()
        hint0 = arrow.sizeHint()
        arrow.setProperty("kbfocus", True)
        repolish(arrow)
        self.assertEqual(arrow.sizeHint(), hint0, "Tab で枠が太くなっても大きさは変わらない")
        arrow.setProperty("kbfocus", False)
        repolish(arrow)
        arrow.setDown(True)
        self.win.save_split.menu.aboutToHide.emit()
        self.assertFalse(arrow.isDown(), "メニューを閉じたら押下の見た目を戻す")

    def test_no_deprecated_mouse_event_calls(self):
        """QMouseEvent.pos() は Qt 6 で非推奨（警告がテストのたびに出ていた）。position().toPoint() を使う"""
        import gui.gl_viewer as glv

        with open(glv.__file__, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("event.pos()", src)
        self.assertNotIn("event.globalPos()", src)

    def test_sidebar_fits_in_both_themes(self):
        win = self.win
        for theme in ("dark", "kiseki"):
            win.apply_ui_theme(theme)
            win._set_sidebar_workflow_tab("all")
            for _ in range(5):
                QApplication.processEvents()
            min_w = win.sidebar_widget.minimumSizeHint().width()
            vp_w = win.sidebar_scroll.viewport().width()
            self.assertLessEqual(min_w, vp_w, f"[{theme}] 「全て」タブで左パネルの最小幅({min_w}px)が表示領域幅({vp_w}px)を超えています")


class TestDensity20261113(_WinCase):
    """2026-11-13 の利用者の指摘（表示の余白が多い・ログ枠が小さい）の再発防止"""

    def _tabs(self):
        return self.win.settings_group.findChild(QTabWidget)

    def test_settings_tab_height_follows_current_tab(self):
        win = self.win
        win._set_sidebar_workflow_tab("settings")
        tabs = self._tabs()
        names = [tabs.tabText(i) for i in range(tabs.count())]
        self.assertIn("説明書", names)
        heights = {}
        for i, name in enumerate(names):
            tabs.setCurrentIndex(i)
            for _ in range(5):
                QApplication.processEvents()
            heights[name] = (tabs.height(), tabs.widget(i).sizeHint().height())
        display_h, display_hint = heights["表示"]
        # 以前は、いちばん背の高い「説明書」に全タブの高さが合わされ、「表示」の行が引き伸ばされて余白だらけだった
        self.assertLess(display_h, heights["説明書"][0] - 150, f"「表示」タブが説明書の高さに引き伸ばされています: {heights}")
        self.assertLessEqual(display_h, display_hint + 60, f"「表示」タブに余白があります: {heights}")
        self.assertGreaterEqual(heights["説明書"][0], 400, "説明書のタブは読める高さを保つ")
        tabs.setCurrentIndex(0)

    def test_log_box_is_large_and_grows(self):
        win, ex = self.win, self.ex
        log = win.log_text
        self.assertGreaterEqual(log.minimumHeight(), 150, "ログ枠が小さいままです（以前は高さ 70px = 3 行の固定）")
        self.assertGreater(log.maximumHeight(), 2000)
        for _ in range(5):
            QApplication.processEvents()
        self.assertGreaterEqual(log.height(), 150)
        lay = win.sidebar_widget.layout()
        self.assertEqual(lay.stretch(lay.indexOf(win.log_group)), 1, "左パネルに空きがあれば、ログ枠が空きいっぱいまで広がる")
        win.btn_toggle_log.setChecked(False)
        self.assertEqual(lay.stretch(lay.indexOf(win.log_group)), 0, "ログを隠しているときは広げない")
        win.btn_toggle_log.setChecked(True)
        self.assertEqual(lay.stretch(lay.indexOf(win.log_group)), 1)

        # ログも小窓に出せる。窓を大きくすれば、そのぶんログ枠が広がる
        self.assertIn("panel.log", {c.id for c in ex.commands()})
        w = ex.popouts.pop_out("log")
        self.assertIsNotNone(w)
        w.resize(720, 700)
        for _ in range(8):
            QApplication.processEvents()
        self.assertGreater(log.height(), 380, "小窓を大きくしてもログ枠が広がりません")
        ex.popouts.restore("log")
        for _ in range(5):
            QApplication.processEvents()
        self.assertLess(lay.indexOf(win.save_group), lay.indexOf(win.log_group), "ログは保存の箱の次（いちばん下）へ戻る")
        self.assertGreaterEqual(lay.indexOf(win.log_group), 0)
        self.assertEqual(lay.stretch(lay.indexOf(win.log_group)), 1, "戻した後も、空きいっぱいまで広がる")

    def test_bars_fit_in_one_row_at_1450px_wide_screen(self):
        """横 1,450px の画面（3D 画面の幅は約 1,000px）で、上下のバーが 1 段に収まる。明るいテーマのほうが文字が広い"""
        win = self.win
        win.apply_ui_theme("kiseki")
        win.resize(1450, 900)
        for _ in range(10):
            QApplication.processEvents()
        self.assertLess(win.viewer.width(), 1030)

        def rows(bar):
            return len({b.geometry().y() for b in bar.findChildren(QPushButton) if b.isVisible() and b.parentWidget() is bar})

        self.assertEqual(rows(win._nav_bar_widget), 1, "上のバーが 2 段になっています")
        self.assertEqual(rows(win.bottom_overlay_bar), 1, "下のバーが 2 段になっています（未選択）")
        self._load()
        self._select(123456 if len(win.viewer.points_raw) > 123456 else 1000)
        for _ in range(5):
            QApplication.processEvents()
        self.assertEqual(rows(win.bottom_overlay_bar), 1, "下のバーが 2 段になっています（選択中）")

    def test_hud_banner_wraps_instead_of_being_cut(self):
        from gui.gl_viewer import HUD_SEPARATOR, wrap_hud_segments

        text = HUD_SEPARATOR.join(["断面シーク中 (枠非表示 / Shift+B:再表示)", "Z基点:10.84m [10.34〜11.34m / 厚1.00m]",
                                   "表示: 30,246,033点", "元ファイルから8,506,840点を重ね描き (ボックス内272,214,414点を上限で間引き)"])
        measure = lambda t: len(t) * 10  # noqa: E731
        lines = wrap_hud_segments(text, 600, measure)
        self.assertGreater(len(lines), 1)
        self.assertEqual(HUD_SEPARATOR.join(lines), text, "折り返しても文は欠けない")
        self.assertTrue(all(measure(ln) <= 600 or HUD_SEPARATOR not in ln for ln in lines))
        self.assertEqual(wrap_hud_segments("短い案内", 600, measure), ["短い案内"])
        self.assertEqual(len(wrap_hud_segments(text, 100000, measure)), 1)


class TestClassicUiHasNoButtonKit(unittest.TestCase):
    def test_classic_ui_is_untouched(self):
        from gui.main_window import MainWindow
        from gui.ui_mode import reset_ui_mode_cache

        QApplication.instance() or QApplication(sys.argv)
        os.environ["E57_UI_MODE"] = "classic"
        reset_ui_mode_cache()
        try:
            win = MainWindow()
            win.show()
            QApplication.processEvents()
            self.assertEqual(win.ui_mode, "classic")
            for attr in ("button_kit", "status_feedback", "toast_host", "unsaved_pill", "save_split"):
                self.assertFalse(hasattr(win, attr), attr)
            self.assertIs(win.toast_label.parentWidget(), win.bottom_overlay_bar)
            self.assertTrue(win.save_manual_btn.icon().isNull())
            win.close()
            win.deleteLater()
        finally:
            os.environ.pop("E57_UI_MODE", None)
            reset_ui_mode_cache()
        QApplication.processEvents()


if __name__ == "__main__":
    unittest.main()
