"""
UI モード解決（新UI / 従来UI）の優先順位テスト: 起動引数 > 環境変数 > 保存設定 > 既定(modern)
"""

import unittest

from gui.ui_mode import UI_MODE_CLASSIC, UI_MODE_MODERN, resolve_ui_mode


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


if __name__ == "__main__":
    unittest.main()
