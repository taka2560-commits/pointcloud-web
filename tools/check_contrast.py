# -*- coding: utf-8 -*-
"""
現行テーマ（gui/styles.py）の主要な文字色／背景色の組み合わせについて WCAG 2.1 コントラスト比を算出する。
UI/UX レビューとデザイン改修（トークン設計）の根拠資料として使う。

使い方: python -X utf8 tools/check_contrast.py
"""

from typing import Tuple


def _lin(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(fg: str, bg: str) -> float:
    l1, l2 = luminance(fg), luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


# (用途, 文字色, 背景色, 文字サイズの目安)  ※ gui/styles.py / main_window.py の実値
PAIRS: Tuple[Tuple[str, str, str, str], ...] = (
    ("本文ラベル QLabel", "#cccccc", "#1e1e24", "13px"),
    ("グループ内ラベル (CollapsibleGroupBox 背景)", "#cccccc", "#172033", "13px"),
    ("グループタイトル", "#38bdf8", "#172033", "11px bold"),
    ("通常ボタン文字", "#ffffff", "#2e6296", "13px bold"),
    ("無効ボタン文字", "#777788", "#333340", "13px"),
    ("segment ボタン (非選択)", "#94a3b8", "#1e293b", "11px"),
    ("segment ボタン (選択)", "#ffffff", "#0284c7", "11px bold"),
    ("outline ボタン", "#cbd5e1", "#1e293b", "13px"),
    ("outline ボタン (無効)", "#555566", "#181820", "13px"),
    ("タブ (非選択)", "#aaaaaa", "#1e1e24", "12px"),
    ("タブ (選択)", "#4da6ff", "#252530", "12px bold"),
    ("ログ (QTextEdit)", "#b0ffb0", "#16161c", "12px mono"),
    ("入力欄文字", "#ffffff", "#16161c", "13px"),
    ("注記テキスト (例: ※点サイズ…)", "#64748b", "#172033", "10-11px"),
    ("プライマリ実行ボタン", "#ffffff", "#28a745", "14px bold"),
    ("danger ボタン", "#ffffff", "#b91c1c", "13px bold"),
    ("進捗ダイアログ サブテキスト", "#94a3b8", "#0f172a", "11px"),
)


def grade(ratio: float, small_text: bool) -> str:
    aa = 4.5 if small_text else 3.0
    aaa = 7.0 if small_text else 4.5
    if ratio >= aaa:
        return "AAA"
    if ratio >= aa:
        return "AA"
    return "NG"


def main() -> None:
    print(f"{'用途':<36} {'文字':<8} {'背景':<8} {'比':>6}  判定")
    print("-" * 72)
    for name, fg, bg, size in PAIRS:
        r = contrast(fg, bg)
        small = not ("14px bold" in size or "18px" in size)
        print(f"{name:<36} {fg:<8} {bg:<8} {r:>6.2f}  {grade(r, small)}  ({size})")


if __name__ == "__main__":
    main()
