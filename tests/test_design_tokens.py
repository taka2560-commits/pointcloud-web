"""
デザイントークン（新UI）のテスト

- 文字色/背景色の全ペアが WCAG のコントラスト基準を満たすこと（通常文字 4.5、無効/大きな文字 3.0）
- 生成 QSS に 12px 未満の文字サイズが含まれないこと（可読性ルール）
- 絵文字除去ユーティリティが UI 記号（◀ ▼ ━ ①）を残し、絵文字だけを取り除くこと
"""

import re
import unittest

from gui.design_tokens import TOKENS, build_modern_qss, contrast_ratio, text_background_pairs
from gui.modern_ui import has_emoji, strip_emoji


class TestTokensContrast(unittest.TestCase):
    def test_all_text_pairs_meet_wcag(self):
        failures = []
        for name, (fg, bg, need) in text_background_pairs(TOKENS).items():
            ratio = contrast_ratio(fg, bg)
            if ratio < need:
                failures.append(f"{name}: {fg} on {bg} = {ratio:.2f} < {need}")
        self.assertFalse(failures, "コントラスト不足:\n" + "\n".join(failures))

    def test_disabled_text_is_still_readable(self):
        self.assertGreaterEqual(contrast_ratio(TOKENS.text_disabled, TOKENS.bg_surface), 3.0)
        self.assertGreaterEqual(contrast_ratio(TOKENS.text_disabled, TOKENS.bg_elevated), 3.0)


class TestModernQss(unittest.TestCase):
    def test_qss_builds_and_has_no_tiny_fonts(self):
        qss = build_modern_qss()
        self.assertGreater(len(qss), 2000)
        tiny = re.findall(r"font-size:\s*(\d+)px", qss)
        self.assertTrue(tiny)
        self.assertTrue(all(int(px) >= 12 for px in tiny), f"12px 未満の文字サイズがあります: {sorted(set(tiny))}")

    def test_qss_covers_key_components(self):
        qss = build_modern_qss()
        for selector in (
            "QPushButton[role=\"primary\"]",
            "QPushButton[role=\"danger\"]",
            "QPushButton[role=\"segment\"]:checked",
            "QPushButton#execute_button",
            "QComboBox::down-arrow",
            "QSlider::handle:horizontal",
            "QToolTip",
            "QMenuBar",
            "QStatusBar",
            "QLabel#toast_label[kind=\"warning\"]",
            "QFrame#empty_state_card",
            "QWidget#progressContainer",
        ):
            self.assertIn(selector, qss, f"QSS に {selector} の定義がありません")

    def test_no_selectorless_declarations(self):
        """セレクタ無しの宣言（子孫へ伝播する事故の元）が QSS 先頭に無いこと"""
        qss = build_modern_qss().strip()
        self.assertTrue(qss.startswith("/*") or qss.startswith("Q"), qss[:40])


class TestStripEmoji(unittest.TestCase):
    def test_removes_emoji_keeps_ui_symbols(self):
        self.assertEqual(strip_emoji("🪄 自動ノイズ処理"), "自動ノイズ処理")
        self.assertEqual(strip_emoji("⚡ 即削除: ON"), "即削除: ON")
        self.assertEqual(strip_emoji("◀ パネル格納"), "◀ パネル格納")
        self.assertEqual(strip_emoji("▼ ログ"), "▼ ログ")
        self.assertEqual(strip_emoji("━ 横(上面)"), "━ 横(上面)")
        self.assertEqual(strip_emoji("①自動"), "①自動")
        self.assertEqual(strip_emoji("透視 ⇄ 正射影 (P)"), "透視 ⇄ 正射影 (P)")
        self.assertEqual(strip_emoji("🗺️ 測量座標系 → 数学系 (X/Y入替)"), "測量座標系 → 数学系 (X/Y入替)")
        self.assertEqual(strip_emoji("✨ 1,640 点を選択中:"), "1,640 点を選択中:")

    def test_has_emoji(self):
        self.assertTrue(has_emoji("📁 ファイル"))
        self.assertFalse(has_emoji("ファイル入出力"))
        self.assertFalse(has_emoji("◀ パネル格納"))


if __name__ == "__main__":
    unittest.main()
