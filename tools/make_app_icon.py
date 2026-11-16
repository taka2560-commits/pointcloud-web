# -*- coding: utf-8 -*-
"""
アプリのアイコン（assets/app/）を描いて書き出す

    python -X utf8 tools/make_app_icon.py            assets/app/ へ書き出す
    python -X utf8 tools/make_app_icon.py <フォルダ>   別のフォルダへ書き出す（見た目を試すとき）

絵: 点でできた箱（= 測った物の点群）と、そのまわりの空中に浮いた水色の点（= 取り除くノイズ）。
    水色は、反射強度の配色「ノイズ探索」で、空中のノイズが光って見える色と同じ（gui/noise_palette.py）。
書き出す物:
    app_icon_<大きさ>.png   画面（ウィンドウ・タスクバー・起動時のカード）で使う。gui/app_identity.py が読む
    app_icon.ico            配布版の EXE に埋め込む（E57_PointCloud_Tool.spec の icon=）
大きさごとに描き直す（大きな絵を縮めると、16px では点がつぶれて読めない）。小さいほど点の数を減らす。
絵を変えたら、このスクリプトを実行して、出てきたファイルをそのままコミットする（アプリは、起動のたびには描かない）。
"""

import math
import os
import struct
import sys
from typing import List, Tuple

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

BG_TOP = "#1f7a55"      # 地の上（深い緑。明るいテーマの主ボタンの色に寄せる）
BG_BOTTOM = "#0a3626"   # 地の下（ほぼ黒。明るいテーマの帯の色に寄せる）
FACE_TOP = "#ffffff"    # 箱の上の面（いちばん明るい）
FACE_LEFT = "#a9e6cb"   # 箱の左の面
FACE_RIGHT = "#4fb58c"  # 箱の右の面（いちばん暗い）
NOISE = "#7fe0ff"       # 空中のノイズ（明るい水色）

# 空中のノイズの点: (横, 縦, 大きさ, 濃さ)。横と縦は絵の一辺を 1 とした位置、大きさは箱の点 1 つを 1 とした比
NOISE_DOTS = ((0.80, 0.20, 1.25, 1.0), (0.62, 0.115, 0.85, 0.9), (0.885, 0.40, 0.8, 0.8), (0.17, 0.17, 0.9, 0.85))
NOISE_DOTS_SMALL = ((0.80, 0.20, 1.2, 1.0), (0.17, 0.19, 1.0, 0.9))   # 24px 以下は 2 つだけ


def _dots_per_edge(size: int) -> int:
    """箱の 1 辺に並べる点の数（小さい絵ほど減らす）"""
    if size <= 24:
        return 2
    if size <= 48:
        return 3
    return 4


def _face_points(a, b, c, d, n: int) -> List[Tuple[float, float]]:
    """四辺形 a-b-c-d（a→b が横、a→d が縦）の中に、n×n の点を等間隔に置く"""
    out = []
    for i in range(n):
        for j in range(n):
            u, v = (i + 0.5) / n, (j + 0.5) / n
            x = a[0] + (b[0] - a[0]) * u + (d[0] - a[0]) * v
            y = a[1] + (b[1] - a[1]) * u + (d[1] - a[1]) * v
            out.append((x, y))
    return out


def draw_icon(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(Qt.NoPen)

    # 地（角の丸い四角）
    pad = size * 0.03
    rect = QRectF(pad, pad, size - 2 * pad, size - 2 * pad)
    grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
    grad.setColorAt(0.0, QColor(BG_TOP))
    grad.setColorAt(1.0, QColor(BG_BOTTOM))
    path = QPainterPath()
    path.addRoundedRect(rect, size * 0.21, size * 0.21)
    p.fillPath(path, grad)
    if size >= 32:  # ふち（暗い地の上でも形が分かるように、細い明るい線）
        edge = QColor("#ffffff")
        edge.setAlphaF(0.22)
        w = max(1.0, size / 96.0)
        inner = QPainterPath()
        inner.addRoundedRect(rect.adjusted(w / 2, w / 2, -w / 2, -w / 2), size * 0.21 - w / 2, size * 0.21 - w / 2)
        p.strokePath(inner, QPen(edge, w))
        p.setPen(Qt.NoPen)

    # 点でできた箱（斜め上から見た立方体の 3 つの面）
    n = _dots_per_edge(size)
    cx, cy = size * 0.47, size * 0.57
    s = size * (0.33 if size <= 24 else 0.32)                     # 箱の辺の長さ
    hx, hy = s * math.cos(math.radians(30)), s * 0.5
    top, ctr, bottom = (cx, cy - s), (cx, cy), (cx, cy + s)
    ul, ur = (cx - hx, cy - hy), (cx + hx, cy - hy)
    ll, lr = (cx - hx, cy + hy), (cx + hx, cy + hy)
    r = s / n * (0.30 if n >= 3 else 0.33)  # 点の半径
    for color, pts in (
        (FACE_TOP, _face_points(ul, top, ur, ctr, n)),
        (FACE_LEFT, _face_points(ul, ctr, bottom, ll, n)),
        (FACE_RIGHT, _face_points(ctr, ur, lr, bottom, n)),
    ):
        p.setBrush(QColor(color))
        for x, y in pts:
            p.drawEllipse(QPointF(x, y), r, r)

    # 空中のノイズ
    for fx, fy, scale, alpha in (NOISE_DOTS_SMALL if size <= 24 else NOISE_DOTS):
        col = QColor(NOISE)
        col.setAlphaF(alpha)
        p.setBrush(col)
        p.drawEllipse(QPointF(size * fx, size * fy), r * scale, r * scale)
    p.end()
    return img


def _png_bytes(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def _dib_bytes(img: QImage) -> bytes:
    """ICO の中の 1 枚（32bit の BMP。下の行から並べ、透明の印の 1bit の面を後ろに付ける）"""
    w, h = img.width(), img.height()
    src = img.convertToFormat(QImage.Format_ARGB32)
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, 0, 0, 0, 0, 0)
    rows = []
    mask_rows = []
    row_bytes = ((w + 31) // 32) * 4
    for y in range(h - 1, -1, -1):
        line = bytearray()
        mask = bytearray(row_bytes)
        for x in range(w):
            c = src.pixelColor(x, y)
            line += bytes((c.blue(), c.green(), c.red(), c.alpha()))
            if c.alpha() == 0:
                mask[x // 8] |= 0x80 >> (x % 8)
        rows.append(bytes(line))
        mask_rows.append(bytes(mask))
    return header + b"".join(rows) + b"".join(mask_rows)


def write_ico(path: str, images: List[QImage]) -> None:
    """ICO を書く。256px は PNG のまま、それより小さい絵は BMP で入れる（古い読み手でも読める形）"""
    blobs = [_png_bytes(im) if im.width() >= 256 else _dib_bytes(im) for im in images]
    out = bytearray(struct.pack("<HHH", 0, 1, len(images)))
    offset = 6 + 16 * len(images)
    for im, blob in zip(images, blobs):
        w = im.width()
        out += struct.pack("<BBBBHHII", w % 256, w % 256, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    for blob in blobs:
        out += blob
    with open(path, "wb") as f:
        f.write(bytes(out))


def main(argv: List[str]) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = argv[1] if len(argv) > 1 else os.path.join(root, "assets", "app")
    os.makedirs(out_dir, exist_ok=True)
    app = QGuiApplication.instance() or QGuiApplication(argv[:1])  # 絵を描くのに要る（画面は出さない）
    images = [draw_icon(s) for s in SIZES]
    for s, im in zip(SIZES, images):
        im.save(os.path.join(out_dir, f"app_icon_{s}.png"), "PNG")
    write_ico(os.path.join(out_dir, "app_icon.ico"), images)
    print(f"書き出しました: {out_dir}（PNG {len(SIZES)} 枚 + app_icon.ico）")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
