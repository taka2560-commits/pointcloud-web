# -*- coding: utf-8 -*-
"""
UI モード（新UI / 従来UI）の解決と保存

デザイン改修を「いつでも元に戻せる」ようにするための安全弁。
優先順位（上が強い）:
  1. 起動引数   --classic-ui / --modern-ui
  2. 環境変数   E57_UI_MODE=classic|modern
  3. 保存設定   QSettings("E57PointCloudTool","Settings") の ui/mode（設定画面「UIスタイル」で変更）
  4. 既定       modern（新UI）

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
