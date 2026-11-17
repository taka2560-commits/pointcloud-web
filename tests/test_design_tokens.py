"""
デザイントークン（新UI）のテスト

- 全テーマ（dark / kiseki）で文字色/背景色の全ペアが WCAG のコントラスト基準を満たすこと（通常文字 4.5、無効/大きな文字 3.0）
- 生成 QSS に 12px 未満の文字サイズが含まれないこと（可読性ルール）
- 「軌跡（明）」テーマは明るい地・黒い帯・深緑の主ボタン・信号赤の選択中であること（デザインシステムの写し）
- 絵文字除去ユーティリティが UI 記号（◀ ▼ ━ ①）を残し、絵文字だけを取り除くこと
"""

import re
import unittest

from gui.design_tokens import (
    KISEKI_TOKENS,
    THEMES,
    TOKENS,
    build_modern_qss,
    contrast_ratio,
    get_tokens,
    luminance,
    text_background_pairs,
)
from gui.modern_ui import has_emoji, strip_emoji


class TestTokensContrast(unittest.TestCase):
    def test_all_text_pairs_meet_wcag_in_every_theme(self):
        failures = []
        for theme, t in THEMES.items():
            for name, (fg, bg, need) in text_background_pairs(t).items():
                ratio = contrast_ratio(fg, bg)
                if ratio < need:
                    failures.append(f"[{theme}] {name}: {fg} on {bg} = {ratio:.2f} < {need}")
        self.assertFalse(failures, "コントラスト不足:\n" + "\n".join(failures))

    def test_disabled_text_is_still_readable(self):
        for t in THEMES.values():
            self.assertGreaterEqual(contrast_ratio(t.text_disabled, t.bg_surface), 3.0, t.name)
            self.assertGreaterEqual(contrast_ratio(t.text_disabled, t.bg_elevated), 3.0, t.name)

    def test_tone_pairs_exist_for_every_theme(self):
        for t in THEMES.values():
            self.assertEqual(set(t.tones.keys()), {"violet", "teal", "amber", "green", "blue", "rose"}, t.name)


class TestKisekiTheme(unittest.TestCase):
    def test_get_tokens(self):
        self.assertIs(get_tokens("dark"), TOKENS)
        self.assertIs(get_tokens("kiseki"), KISEKI_TOKENS)
        self.assertIs(get_tokens("unknown"), TOKENS)
        self.assertIs(get_tokens(""), TOKENS)

    def test_kiseki_is_light_with_black_band(self):
        t = KISEKI_TOKENS
        self.assertTrue(t.is_light)
        self.assertGreater(luminance(t.bg_canvas), 0.5, "地は明るい")
        self.assertLess(luminance(t.band_bg), 0.05, "帯は黒に近い")
        self.assertEqual(t.group_header_bg, t.band_bg, "箱の頭は黒い帯")
        self.assertEqual(t.primary_bg, "#004c2c", "主ボタンは深緑 (cta)")
        self.assertEqual(t.on_bg, "#c0392b", "選択中は信号赤 (signal)")
        self.assertEqual(t.radius_sm, 3)
        self.assertEqual(t.radius_band, 0)
        self.assertFalse(TOKENS.is_light)

    def test_kiseki_qss_uses_light_icons(self):
        qss = build_modern_qss(KISEKI_TOKENS)
        self.assertIn("chevron-down-light.svg", qss)
        self.assertIn("chevron-up-light.svg", qss)
        self.assertNotIn('chevron-down.svg"', qss)
        dark = build_modern_qss(TOKENS)
        self.assertIn('chevron-down.svg"', dark)
        self.assertNotIn("chevron-down-light.svg", dark)


class TestAbyssAndForestThemes(unittest.TestCase):
    """2026-11-17、利用者提供のスタイル資料 2 つを写したテーマ（深海 abyss・若葉 forest）"""

    def test_themes_are_registered_and_selectable(self):
        from gui.design_tokens import ABYSS_TOKENS, FOREST_TOKENS, THEME_CHOICES, THEME_LABELS
        from gui.ui_mode import _VALID_THEMES

        self.assertIs(get_tokens("abyss"), ABYSS_TOKENS)
        self.assertIs(get_tokens("forest"), FOREST_TOKENS)
        self.assertEqual([k for k, _ in THEME_CHOICES], list(THEMES.keys()), "設定の選択肢は、テーマの全部")
        self.assertEqual(set(THEME_LABELS), set(THEMES))
        self.assertEqual(set(_VALID_THEMES), set(THEMES), "保存した設定・起動引数から選べる")
        self.assertEqual(THEME_CHOICES[0][0], "dark", "標準は今までどおりダーク")

    def test_abyss_is_a_teal_stack_with_one_glowing_button(self):
        from gui.design_tokens import ABYSS_TOKENS as t

        self.assertFalse(t.is_light)
        # 沈んだ面 < 地 < 箱 < 浮いた面 の順に明るい（影でなく、面の色で奥行きを出す）
        steps = [luminance(c) for c in (t.bg_input, t.bg_canvas, t.bg_surface, t.bg_elevated)]
        self.assertEqual(steps, sorted(steps))
        self.assertEqual(len(set(steps)), 4)
        qss = build_modern_qss(t)
        self.assertEqual(qss.count("#fad1ff"), 2, "光の帯は主ボタンだけ（通常の 1 か所。止めの色が 2 回）")
        tail = qss[qss.index("深海: 主ボタンだけが光の帯"):]
        self.assertIn('QPushButton[role="primary"]', tail)
        self.assertIn("qlineargradient", tail)
        self.assertNotIn("box-shadow", qss)
        self.assertIn('chevron-down.svg"', qss, "暗いテーマの矢印")

    def test_forest_is_light_with_forest_bands_and_one_green(self):
        from gui.design_tokens import FOREST_TOKENS as t

        self.assertTrue(t.is_light)
        self.assertGreater(luminance(t.bg_canvas), 0.8)
        self.assertLess(luminance(t.band_bg), 0.03)
        self.assertEqual(t.group_header_bg, t.band_bg)
        self.assertEqual(t.primary_bg, t.on_bg, "塗りの色は若葉の緑ひとつ")
        self.assertEqual(t.primary_bg, t.success_base)
        self.assertIn("chevron-down-light.svg", build_modern_qss(t))

    def test_other_themes_get_no_extras(self):
        from gui.design_tokens import _theme_extras_qss

        self.assertEqual(_theme_extras_qss(TOKENS), "")
        self.assertEqual(_theme_extras_qss(KISEKI_TOKENS), "")


class TestModernQss(unittest.TestCase):
    def test_qss_builds_and_has_no_tiny_fonts(self):
        for t in THEMES.values():
            qss = build_modern_qss(t)
            self.assertGreater(len(qss), 2000)
            tiny = re.findall(r"font-size:\s*(\d+)px", qss)
            self.assertTrue(tiny)
            self.assertTrue(all(int(px) >= 12 for px in tiny), f"[{t.name}] 12px 未満の文字サイズがあります: {sorted(set(tiny))}")

    def test_qss_covers_key_components(self):
        qss = build_modern_qss()
        for selector in (
            "QPushButton[role=\"primary\"]",
            "QPushButton[role=\"danger\"]",
            "QPushButton[role=\"segment\"]:checked",
            "QPushButton[tone=\"teal\"]",
            "QPushButton#execute_button",
            "QComboBox::down-arrow",
            "QSlider::handle:horizontal",
            "QToolTip",
            "QMenuBar",
            "QStatusBar",
            "QLabel#toast_label[kind=\"warning\"]",
            "QFrame#empty_state_card",
            "QWidget#progressContainer",
            "QWidget#collapsible_header",
        ):
            self.assertIn(selector, qss, f"QSS に {selector} の定義がありません")

    def test_no_selectorless_declarations(self):
        """セレクタ無しの宣言（子孫へ伝播する事故の元）が QSS 先頭に無いこと"""
        for t in THEMES.values():
            qss = build_modern_qss(t).strip()
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
