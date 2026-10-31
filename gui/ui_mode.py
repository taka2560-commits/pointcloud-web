# -*- coding: utf-8 -*-
"""
UI モード（新UI / 従来UI）の解決と保存

デザイン改修を「いつでも元に戻せる」ようにするための安全弁。
優先順位（上が強い）:
  1. 起動引数   --classic-ui / --modern-ui
  2. 環境変数   E57_UI_MODE=classic|modern
  3. 保存設定   QSettings("E57PointCloudTool","Settings") の ui/mode（設定画面「UIスタイル」で変更）
  4. 既定       modern（新UI）

新UIのテーマ（dark / kiseki）も同じ優先順位で決まる: --theme=kiseki > E57_UI_THEME > ui/theme > dark。
新UIのアニメーションは ui/motion_enabled（既定 ON）。

従来UIのコードは削除せず、classic を選ぶと改修前とほぼ同じ見た目・配置で起動する。
"""

import os
import sys
from typing import Optional, Sequence

UI_MODE_MODERN = "modern"
UI_MODE_CLASSIC = "classic"
_VALID = (UI_MODE_MODERN, UI_MODE_CLASSIC)

_SETTINGS_ORG = "E57PointCloudTool"
_SETTINGS_APP = "Settings"
SETTINGS_KEY = "ui/mode"
ENV_KEY = "E57_UI_MODE"

_cached_mode: Optional[str] = None


def _normalize(value) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("classic", "legacy", "old", "従来"):
        return UI_MODE_CLASSIC
    if v in ("modern", "new", "新"):
        return UI_MODE_MODERN
    return None


def resolve_ui_mode(
    argv: Optional[Sequence[str]] = None,
    env: Optional[dict] = None,
    use_settings: bool = True,
) -> str:
    """優先順位に従って UI モードを決定する（テスト容易性のため引数で差し替え可能）"""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--classic-ui" in args:
        return UI_MODE_CLASSIC
    if "--modern-ui" in args:
        return UI_MODE_MODERN

    environ = os.environ if env is None else env
    from_env = _normalize(environ.get(ENV_KEY))
    if from_env:
        return from_env

    if use_settings:
        try:
            from PySide6.QtCore import QSettings

            s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
            from_settings = _normalize(s.value(SETTINGS_KEY, None))
            if from_settings:
                return from_settings
        except Exception:
            pass
    return UI_MODE_MODERN


def get_ui_mode() -> str:
    """プロセス内で一度だけ解決してキャッシュする（起動中にモードが変わることはない）"""
    global _cached_mode
    if _cached_mode is None:
        _cached_mode = resolve_ui_mode()
    return _cached_mode


def reset_ui_mode_cache() -> None:
    """テスト・品質ゲート用: 次回 get_ui_mode() で再解決させる"""
    global _cached_mode
    _cached_mode = None


def is_modern() -> bool:
    return get_ui_mode() == UI_MODE_MODERN


# ----------------------------------------------------------------------------- テーマ（新UIの見た目）
THEME_SETTINGS_KEY = "ui/theme"
THEME_ENV_KEY = "E57_UI_THEME"
_VALID_THEMES = ("dark", "kiseki")
_cached_theme: Optional[str] = None


def _normalize_theme(value) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("dark", "ダーク", "default", "標準"):
        return "dark"
    if v in ("kiseki", "軌跡", "light", "明"):
        return "kiseki"
    return None


def resolve_ui_theme(argv: Optional[Sequence[str]] = None, env: Optional[dict] = None, use_settings: bool = True) -> str:
    """新UIのテーマ: 起動引数 --theme=<name> > 環境変数 E57_UI_THEME > 設定 ui/theme > 既定 dark"""
    args = list(sys.argv[1:] if argv is None else argv)
    for a in args:
        if a.startswith("--theme="):
            t = _normalize_theme(a.split("=", 1)[1])
            if t:
                return t
    environ = os.environ if env is None else env
    t = _normalize_theme(environ.get(THEME_ENV_KEY))
    if t:
        return t
    if use_settings:
        try:
            from PySide6.QtCore import QSettings

            s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
            t = _normalize_theme(s.value(THEME_SETTINGS_KEY, None))
            if t:
                return t
        except Exception:
            pass
    return "dark"


def get_ui_theme() -> str:
    global _cached_theme
    if _cached_theme is None:
        _cached_theme = resolve_ui_theme()
    return _cached_theme


def reset_ui_theme_cache() -> None:
    global _cached_theme
    _cached_theme = None


def set_saved_ui_theme(theme: str) -> None:
    norm = _normalize_theme(theme) or "dark"
    from PySide6.QtCore import QSettings

    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.setValue(THEME_SETTINGS_KEY, norm)
    s.sync()


def get_saved_ui_theme() -> str:
    try:
        from PySide6.QtCore import QSettings

        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        return _normalize_theme(s.value(THEME_SETTINGS_KEY, None)) or "dark"
    except Exception:
        return "dark"


# ----------------------------------------------------------------------------- モーション（新UIのアニメーション ON/OFF）
MOTION_SETTINGS_KEY = "ui/motion_enabled"


def _to_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def get_saved_motion_enabled() -> bool:
    """設定「アニメーション」（未保存なら ON）"""
    try:
        from PySide6.QtCore import QSettings

        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        return _to_bool(s.value(MOTION_SETTINGS_KEY, None), True)
    except Exception:
        return True


def set_saved_motion_enabled(enabled: bool) -> None:
    from PySide6.QtCore import QSettings

    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.setValue(MOTION_SETTINGS_KEY, bool(enabled))
    s.sync()


def set_saved_ui_mode(mode: str) -> None:
    """設定画面から選んだ UI モードを保存する（次回起動時に適用）"""
    norm = _normalize(mode) or UI_MODE_MODERN
    from PySide6.QtCore import QSettings

    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.setValue(SETTINGS_KEY, norm)
    s.sync()


def get_saved_ui_mode() -> str:
    """保存されている UI モード（未保存なら modern）"""
    try:
        from PySide6.QtCore import QSettings

        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        return _normalize(s.value(SETTINGS_KEY, None)) or UI_MODE_MODERN
    except Exception:
        return UI_MODE_MODERN
