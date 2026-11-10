# -*- coding: utf-8 -*-
"""
新UIのアイコン（assets/icons/ui/*.svg）を、テーマの色で塗って QIcon / QPixmap にする

SVG は線画で stroke="currentColor"。読み込むときに色を差し替えて描くので、テーマごとの色違いファイルは持たない。
ボタンの文字と同じ色で描くこと（役割ごとの色は gui/button_kit.py の icon_color_for が決める）。
絵文字は環境で見た目が変わるためボタンには使わない。意味を素早く伝えたい所はここのアイコン＋文字にする。
"""

import os
from typing import Dict, Optional, Tuple

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QPainter, QPixmap

from core.logger import log_warning
from gui.design_tokens import assets_dir

ICON_NAMES = (
    "save", "trash", "undo", "redo", "search", "folder", "open-file", "check", "check-circle", "circle",
    "alert", "error", "info", "close", "camera", "popout", "spinner", "chevron-down", "play", "hint", "restore",
)

_svg_cache: Dict[str, str] = {}
_failed: set = set()
_pixmap_cache: Dict[Tuple[str, str, int, int, int], QPixmap] = {}


def icon_path(name: str) -> str:
    return os.path.join(assets_dir(), "icons", "ui", f"{name}.svg")


def _svg_text(name: str) -> str:
    if name not in _svg_cache:
        with open(icon_path(name), "r", encoding="utf-8") as f:
            _svg_cache[name] = f.read()
    return _svg_cache[name]


def _device_ratio() -> float:
    screen = QGuiApplication.primaryScreen()
    try:
        return max(1.0, float(screen.devicePixelRatio())) if screen is not None else 1.0
    except Exception:
        return 1.0


def icon_pixmap(name: str, color: str, size: int = 16, rotate_deg: int = 0) -> QPixmap:
    """名前のアイコンを color（#rrggbb）で描いた QPixmap。描けないときは空の QPixmap（文字だけで表示される）"""
    ratio = _device_ratio()
    key = (name, color, int(size), int(rotate_deg) % 360, int(round(ratio * 100)))
    cached = _pixmap_cache.get(key)
    if cached is not None:
        return cached
    px = int(round(size * ratio))
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)
    try:
        from PySide6.QtSvg import QSvgRenderer

        data = _svg_text(name).replace("currentColor", color)
        renderer = QSvgRenderer(QByteArray(data.encode("utf-8")))
        if not renderer.isValid():
            raise ValueError(name)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if rotate_deg:
            painter.translate(px / 2.0, px / 2.0)
            painter.rotate(float(rotate_deg))
            painter.translate(-px / 2.0, -px / 2.0)
        renderer.render(painter, QRectF(0, 0, px, px))
        painter.end()
    except Exception as exc:
        pm = QPixmap(px, px)
        pm.fill(Qt.transparent)
        if name not in _failed:
            # 黙って消えると気付けないので、アイコンごとに 1 回だけ記録する（配布 EXE に SVG の部品が入っていない等）
            _failed.add(name)
            log_warning(f"アイコン「{name}」を描けませんでした（文字だけで表示します）: {exc}")
    pm.setDevicePixelRatio(ratio)
    _pixmap_cache[key] = pm
    return pm


def failed_icons() -> Tuple[str, ...]:
    """描けなかったアイコンの名前（起動後の点検用）"""
    return tuple(sorted(_failed))


def themed_icon(name: str, color: str, disabled_color: Optional[str] = None, size: int = 16) -> QIcon:
    """通常時は color、無効時は disabled_color で描く QIcon"""
    icon = QIcon()
    icon.addPixmap(icon_pixmap(name, color, size), QIcon.Normal, QIcon.Off)
    icon.addPixmap(icon_pixmap(name, color, size), QIcon.Active, QIcon.Off)
    icon.addPixmap(icon_pixmap(name, disabled_color or color, size), QIcon.Disabled, QIcon.Off)
    return icon


def icon_size(size: int = 16) -> QSize:
    return QSize(size, size)


def clear_icon_cache() -> None:
    _pixmap_cache.clear()
