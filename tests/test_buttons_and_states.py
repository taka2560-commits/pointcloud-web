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

import numpy as np
from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import QApplication, QPushButton

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

    def test_edit_report(self):
        self.assertTrue(is_edit_report("5,000 点をノイズレイヤーへ移動しました (Ctrl+Zで取消可)"))
        self.assertTrue(is_edit_report("自動除去: 1,234 点を高速分離完了"))
        self.assertFalse(is_edit_report("操作を元に戻しました (Undo)"))
        self.assertFalse(is_edit_report("連結塊 300 点を一括選択しました"))
        self.assertFalse(is_edit_report("3 スキャン (9,000 点) を読み込みました"))
        self.assertFalse(is_edit_report("視点回転モード (V)"))

    def test_path_validation(self):
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
            self.assertIn("フォルダがありません", validate_output_path(os.path.join(d, "none", "out.e57")))
            self.assertIn("ファイル名", validate_output_path(d))
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
        self.assertTrue(soft and all("ログフォルダ" in b.text() for b in soft))
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
        from gui.button_kit import install_toggle_mark

        win = self.win
        btn = win.btn_instant_del
        self.assertEqual(btn.property("toggle_mark"), "off")
        self.assertFalse(btn.icon().isNull())
        btn.setChecked(True)
        self.assertEqual(btn.property("toggle_mark"), "on")
        self.assertFalse(btn.icon().isNull())
        btn.setChecked(False)
        self.assertEqual(btn.property("toggle_mark"), "off")
        plain = QPushButton("ON/OFF", win)
        plain.setCheckable(True)
        install_toggle_mark(plain, self.kit.tokens)
        w0 = plain.sizeHint().width()
        plain.setChecked(True)
        self.assertEqual(plain.sizeHint().width(), w0)
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
        win.output_edit.setText(os.path.join(tempfile.gettempdir(), "no_such_dir_e57", "out.e57"))
        win.output_edit.editingFinished.emit()
        self.assertEqual(len(errs.messages()), 2)
        win.input_edit.setText("")
        self.assertFalse(bool(win.input_edit.property("invalid")), "直したらすぐ消す")
        self.assertEqual(len(errs.messages()), 1)
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
        self.assertEqual(win.btn_instant_del.property("toggle_mark"), "off")
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
