"""
UI モード解決（新UI / 従来UI）とテーマ解決の優先順位テスト: 起動引数 > 環境変数 > 保存設定 > 既定
"""

import unittest

from gui.ui_mode import UI_MODE_CLASSIC, UI_MODE_MODERN, resolve_ui_mode, resolve_ui_theme, _to_bool


class TestResolveUiMode(unittest.TestCase):
    def test_default_is_modern(self):
        self.assertEqual(resolve_ui_mode(argv=[], env={}, use_settings=False), UI_MODE_MODERN)

    def test_env_classic(self):
        self.assertEqual(resolve_ui_mode(argv=[], env={"E57_UI_MODE": "classic"}, use_settings=False), UI_MODE_CLASSIC)
        self.assertEqual(resolve_ui_mode(argv=[], env={"E57_UI_MODE": "LEGACY"}, use_settings=False), UI_MODE_CLASSIC)

    def test_env_invalid_falls_back(self):
        self.assertEqual(resolve_ui_mode(argv=[], env={"E57_UI_MODE": "purple"}, use_settings=False), UI_MODE_MODERN)

    def test_argv_overrides_env(self):
        self.assertEqual(
            resolve_ui_mode(argv=["--classic-ui"], env={"E57_UI_MODE": "modern"}, use_settings=False), UI_MODE_CLASSIC
        )
        self.assertEqual(
            resolve_ui_mode(argv=["--modern-ui"], env={"E57_UI_MODE": "classic"}, use_settings=False), UI_MODE_MODERN
        )


class TestResolveUiTheme(unittest.TestCase):
    def test_default_is_dark(self):
        self.assertEqual(resolve_ui_theme(argv=[], env={}, use_settings=False), "dark")

    def test_env_kiseki_and_aliases(self):
        for v in ("kiseki", "KISEKI", "軌跡", "light", "明"):
            self.assertEqual(resolve_ui_theme(argv=[], env={"E57_UI_THEME": v}, use_settings=False), "kiseki", v)
        for v in ("dark", "ダーク", "default", "標準"):
            self.assertEqual(resolve_ui_theme(argv=[], env={"E57_UI_THEME": v}, use_settings=False), "dark", v)

    def test_env_invalid_falls_back(self):
        self.assertEqual(resolve_ui_theme(argv=[], env={"E57_UI_THEME": "neon"}, use_settings=False), "dark")

    def test_argv_overrides_env(self):
        self.assertEqual(resolve_ui_theme(argv=["--theme=kiseki"], env={"E57_UI_THEME": "dark"}, use_settings=False), "kiseki")
        self.assertEqual(resolve_ui_theme(argv=["--theme=dark"], env={"E57_UI_THEME": "kiseki"}, use_settings=False), "dark")
        # 不正な引数は無視して次の優先順位へ
        self.assertEqual(resolve_ui_theme(argv=["--theme=neon"], env={"E57_UI_THEME": "kiseki"}, use_settings=False), "kiseki")


class TestMotionSettingParse(unittest.TestCase):
    def test_to_bool(self):
        self.assertTrue(_to_bool(None, True))
        self.assertFalse(_to_bool(None, False))
        self.assertTrue(_to_bool(True))
        self.assertFalse(_to_bool(False))
        self.assertTrue(_to_bool("true"))
        self.assertFalse(_to_bool("false"))
        self.assertFalse(_to_bool("0"))
        self.assertTrue(_to_bool("1"))


if __name__ == "__main__":
    unittest.main()
