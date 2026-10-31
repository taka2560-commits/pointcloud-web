# -*- coding: utf-8 -*-
"""
UI モーション（新UI用）

参考: モーションリファレンス（https://motion-reference.takayustudio.jp/）の全モーションから、
      業務ツール向けに「意味のある状態変化だけに、短く、位置と透明度だけで」を基準に選定した。
      選定一覧と理由は docs/DESIGN_STYLE_PLAN.md §9 を参照。

採用（実用）:
  - プレスフィードバック   押下 80ms で沈み（不透明度↓・1px↓）、離して 160ms で復帰      → 全ボタン
  - フェード＋スライド     出現は ease-out で下から 8px 上がりながら現れる（240ms）      → トースト・空状態カード・進捗ダイアログ
  - 離散プロパティ退場     退出は ease-in で短く薄れ、消えきってから hide（160ms）        → トースト・空状態カード
  - カスケード表示         タブ切替で表示される箱が上から 25ms 刻みで順に現れる（≤310ms） → サイドバー
  - アコーディオン展開     折りたたみ箱の展開は高さと透明度が同時に増える（180ms）        → CollapsibleGroupBox
  - 数値ティッカー         点数の表示は前の値から新しい値へ桁幅を固定して数え上げる（420ms）→ 点数ピル

採用（遊び心・1 回きり・≤ 420ms）:
  - オーバーシュート       トーストは定位置を少し行き過ぎて戻る（easeOutBack）
  - エラーシェイク         警告トーストは左右に小さく震えて減衰（320ms）
  - アテンションバウンス   選択バッジは選択が始まった瞬間に 1 回だけ跳ねる（360ms）
  - フォーカスパルス       読み込み完了後、次に押す「実行」ボタンの外周にリングが 1 回広がる（420ms）
  - ハイライト（なし）     常時動くもの（呼吸・グロー・シマー）は採用しない

不採用の理由: 拡大縮小・回転・ぼかし・3D は QWidget では描画コストが高くレイアウトも乱れる。
             常時アニメーションは注意を奪い、2,000 万点の描画と GPU を取り合う。

QSS はトランジションを持たないため、QVariantAnimation と QGraphicsOpacityEffect・move() で実装する。
従来UI（classic）では使わない。設定「アニメーション」を OFF にすると全て即時反映になる（set_motion_enabled）。
"""

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPoint, QRect, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QApplication, QGraphicsOpacityEffect, QLabel, QWidget

try:
    import shiboken6

    def _alive(obj) -> bool:
        try:
            return obj is not None and shiboken6.isValid(obj)
        except Exception:
            return False
except Exception:  # pragma: no cover

    def _alive(obj) -> bool:
        return obj is not None


@dataclass(frozen=True)
class MotionTokens:
    duration_fast_ms: int = 80      # 押下の反応
    duration_base_ms: int = 160     # 押下からの復帰、小さな出入り
    duration_slow_ms: int = 240     # カード・トーストの出入り
    duration_expand_ms: int = 180   # 折りたたみ展開
    duration_shake_ms: int = 320    # エラーシェイク
    duration_bounce_ms: int = 360   # アテンションバウンス
    duration_pulse_ms: int = 420    # フォーカスパルス
    duration_ticker_ms: int = 420   # 数値ティッカー
    press_opacity: float = 0.62     # 押下中の不透明度（「沈む」感じ）
    press_offset_px: int = 1        # 押下中にわずかに下がる量
    slide_px: int = 8               # フェード＋スライドの移動量
    cascade_step_ms: int = 25       # カスケード表示の時間差
    cascade_max_steps: int = 6      # 時間差の上限（6 段 = 150ms）
    shake_px: int = 6               # シェイクの最大振幅
    bounce_px: int = 6              # バウンスの高さ
    pulse_spread_px: int = 10       # パルスリングの広がり


MOTION = MotionTokens()

_EASE_OUT = QEasingCurve(QEasingCurve.OutCubic)
_EASE_IN = QEasingCurve(QEasingCurve.InCubic)
_EASE_OUT_FAST = QEasingCurve(QEasingCurve.OutQuad)
_EASE_IN_OUT = QEasingCurve(QEasingCurve.InOutCubic)
_EASE_OUT_BACK = QEasingCurve(QEasingCurve.OutBack)
_EASE_LINEAR = QEasingCurve(QEasingCurve.Linear)

_enabled = True


def set_motion_enabled(enabled: bool) -> None:
    """False にすると全モーションが即時反映（アクセシビリティ・低スペック機向け）"""
    global _enabled
    _enabled = bool(enabled)


def motion_enabled() -> bool:
    return _enabled


# ----------------------------------------------------------------------------- 低レベル
def _get_or_make_opacity_effect(w: QWidget) -> QGraphicsOpacityEffect:
    fx = w.graphicsEffect()
    if isinstance(fx, QGraphicsOpacityEffect):
        return fx
    fx = QGraphicsOpacityEffect(w)
    fx.setOpacity(1.0)
    w.setGraphicsEffect(fx)
    return fx


def _stop_anim(w: QWidget, attr: str) -> None:
    anim = getattr(w, attr, None)
    if anim is not None:
        try:
            anim.stop()
        except Exception:
            pass
        try:
            setattr(w, attr, None)
        except Exception:
            pass


def _run_anim(w: QWidget, attr: str, duration_ms: int, easing: QEasingCurve, on_value: Callable[[float], None],
              on_finished: Optional[Callable[[], None]] = None) -> QVariantAnimation:
    """0.0→1.0 の進行率を on_value に流す汎用アニメーション。同じ attr の前回分は止める。"""
    _stop_anim(w, attr)
    anim = QVariantAnimation(w)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setDuration(int(duration_ms))
    anim.setEasingCurve(easing)

    def _tick(v):
        if not _alive(w):
            return
        try:
            on_value(float(v))
        except RuntimeError:
            pass

    def _done():
        try:
            setattr(w, attr, None)
        except Exception:
            pass
        if on_finished is not None and _alive(w):
            try:
                on_finished()
            except RuntimeError:
                pass

    anim.valueChanged.connect(_tick)
    anim.finished.connect(_done)
    setattr(w, attr, anim)
    anim.start()
    return anim


def animate_opacity(
    w: QWidget,
    start: float,
    end: float,
    duration_ms: int,
    easing: QEasingCurve = _EASE_OUT,
    on_finished=None,
    remove_effect_when_done: bool = False,
) -> Optional[QVariantAnimation]:
    """不透明度を start→end へアニメーション。終了後に効果を外すと通常描画に戻る（描画コストゼロ）"""
    w._motion_fading_out = False  # type: ignore[attr-defined]  # 新しい不透明度アニメーションは退場を上書きする
    if not _enabled:
        if remove_effect_when_done or abs(float(end) - 1.0) < 1e-6:
            try:
                w.setGraphicsEffect(None)
            except Exception:
                pass
        if on_finished is not None:
            on_finished()
        return None
    fx = _get_or_make_opacity_effect(w)
    fx.setOpacity(float(start))
    s, e = float(start), float(end)

    def _set(v: float):
        if _alive(fx):
            fx.setOpacity(s + (e - s) * v)

    def _done():
        if remove_effect_when_done and abs(e - 1.0) < 1e-6:
            try:
                w.setGraphicsEffect(None)
            except Exception:
                pass
        if on_finished is not None:
            on_finished()

    return _run_anim(w, "_motion_opacity_anim", duration_ms, easing, _set, _done)


def _animate_offset(w: QWidget, duration_ms: int, easing: QEasingCurve,
                    offset_fn: Callable[[float], QPoint], on_finished=None) -> Optional[QVariantAnimation]:
    """
    基準位置からの相対オフセットを進行率に応じて与え、終了時に基準位置へ戻す（位置のみ・レイアウトは不変）。
    - 途中でレイアウトがウィジェットを動かしたら、その位置を新しい基準にする（最終位置がずれない）
    - 同じウィジェットの位置アニメーションは常に 1 本だけ。新しく始めるとき前のものは位置を戻してから止める
    """
    if not _enabled:
        if on_finished is not None:
            on_finished()
        return None
    prev_restore = getattr(w, "_motion_offset_restore", None)
    if prev_restore is not None:
        try:
            prev_restore()
        except Exception:
            pass
    state = {"base": w.pos(), "off": QPoint(0, 0)}

    def _rebase():
        cur = w.pos()
        if cur != state["base"] + state["off"]:
            state["base"] = cur
            state["off"] = QPoint(0, 0)

    def _set(v: float):
        _rebase()
        off = offset_fn(v)
        state["off"] = off
        w.move(state["base"] + off)

    def _restore():
        _stop_anim(w, "_motion_offset_anim")
        if _alive(w):
            _rebase()
            w.move(state["base"])
            state["off"] = QPoint(0, 0)
            w._motion_offset_restore = None  # type: ignore[attr-defined]

    def _done():
        if _alive(w):
            _rebase()
            w.move(state["base"])
            w._motion_offset_restore = None  # type: ignore[attr-defined]
        if on_finished is not None:
            on_finished()

    w._motion_offset_restore = _restore  # type: ignore[attr-defined]
    return _run_anim(w, "_motion_offset_anim", duration_ms, easing, _set, _done)


# ----------------------------------------------------------------------------- 出現・退場
def fade_in(w: QWidget, duration_ms: Optional[int] = None) -> None:
    """表示しながらフェードイン（出現 = ease-out）"""
    w.show()
    animate_opacity(w, 0.0, 1.0, duration_ms or MOTION.duration_slow_ms, _EASE_OUT, remove_effect_when_done=True)


def fade_slide_in(w: QWidget, dy: Optional[int] = None, duration_ms: Optional[int] = None,
                  overshoot: bool = False) -> None:
    """
    フェード＋スライド: 下から dy px 上がりながら現れる（easeOutCubic）。
    overshoot=True で「オーバーシュート」（定位置を少し行き過ぎて戻る easeOutBack）。
    """
    w.show()
    if not _enabled:
        try:
            w.setGraphicsEffect(None)
        except Exception:
            pass
        return
    d = duration_ms or MOTION.duration_slow_ms
    dist = MOTION.slide_px if dy is None else int(dy)
    animate_opacity(w, 0.0, 1.0, d, _EASE_OUT, remove_effect_when_done=True)
    ease = _EASE_OUT_BACK if overshoot else _EASE_OUT
    _animate_offset(w, d, ease, lambda v: QPoint(0, int(round(dist * (1.0 - v)))))


def fade_out_then_hide(w: QWidget, duration_ms: Optional[int] = None) -> None:
    """フェードアウトしてから隠す（退出 = ease-in、消えきるまで描画し、その後に hide）"""
    if not w.isVisible():
        return

    def _hide():
        w._motion_fading_out = False  # type: ignore[attr-defined]
        w.hide()
        try:
            w.setGraphicsEffect(None)
        except Exception:
            pass

    anim = animate_opacity(w, 1.0, 0.0, duration_ms or MOTION.duration_base_ms, _EASE_IN, on_finished=_hide)
    w._motion_fading_out = anim is not None  # type: ignore[attr-defined]


def cancel_fade_out(w: QWidget) -> bool:
    """進行中の fade_out_then_hide を取り消し、不透明のまま表示を保つ。取り消したら True"""
    if not getattr(w, "_motion_fading_out", False):
        return False
    _stop_anim(w, "_motion_opacity_anim")
    w._motion_fading_out = False  # type: ignore[attr-defined]
    try:
        w.setGraphicsEffect(None)
    except Exception:
        pass
    return True


def cascade_in(widgets: Sequence[QWidget], each_ms: Optional[int] = None) -> None:
    """カスケード表示: 表示中のウィジェットを上から順に時間差（25ms 刻み・最大 6 段）でフェード＋スライド"""
    if not _enabled:
        return
    d = each_ms or MOTION.duration_base_ms
    idx = 0
    for w in widgets:
        if w is None or not w.isVisible():
            continue
        delay = MOTION.cascade_step_ms * min(idx, MOTION.cascade_max_steps)
        idx += 1
        if delay == 0:
            fade_slide_in(w, dy=MOTION.slide_px // 2, duration_ms=d)
        else:
            fx = _get_or_make_opacity_effect(w)
            fx.setOpacity(0.0)
            def _later(_w=w, _d=d):
                if not _alive(_w):
                    return
                if _w.isVisible():
                    fade_slide_in(_w, dy=MOTION.slide_px // 2, duration_ms=_d)
                else:
                    try:
                        _w.setGraphicsEffect(None)
                    except Exception:
                        pass

            QTimer.singleShot(delay, _later)


def expand_animated(content: QWidget, duration_ms: Optional[int] = None) -> None:
    """アコーディオン展開: 高さ 0→自然高さ と 透明度 0→1 を同時に（easeInOutCubic）。終了後は高さ制限を外す。"""
    if not _enabled or content is None:
        return
    target_h = max(1, content.sizeHint().height())
    d = duration_ms or MOTION.duration_expand_ms
    content.setMaximumHeight(0)
    animate_opacity(content, 0.0, 1.0, d, _EASE_IN_OUT, remove_effect_when_done=True)

    def _set(v: float):
        content.setMaximumHeight(max(1, int(round(target_h * v))))

    def _done():
        content.setMaximumHeight(16777215)

    _run_anim(content, "_motion_expand_anim", d, _EASE_IN_OUT, _set, _done)


# ----------------------------------------------------------------------------- 注目・強調（遊び心）
def shake(w: QWidget, amplitude_px: Optional[int] = None, duration_ms: Optional[int] = None) -> None:
    """エラーシェイク: 左右に震えて急速に減衰（位置のみ。終了時に元の位置へ必ず戻す）"""
    amp = MOTION.shake_px if amplitude_px is None else int(amplitude_px)
    d = duration_ms or MOTION.duration_shake_ms

    def _off(v: float) -> QPoint:
        decay = (1.0 - v) ** 2
        return QPoint(int(round(amp * decay * math.sin(v * math.pi * 5.0))), 0)

    _animate_offset(w, d, _EASE_LINEAR, _off)


def bounce(w: QWidget, height_px: Optional[int] = None, duration_ms: Optional[int] = None) -> None:
    """アテンションバウンス: 上へ 1 回跳ねて着地し、小さな 2 回目で止まる（位置のみ）"""
    h = MOTION.bounce_px if height_px is None else int(height_px)
    d = duration_ms or MOTION.duration_bounce_ms

    def _off(v: float) -> QPoint:
        if v < 0.6:
            y = -h * math.sin(math.pi * v / 0.6)
        else:
            y = -h * 0.25 * math.sin(math.pi * (v - 0.6) / 0.4)
        return QPoint(0, int(round(y)))

    _animate_offset(w, d, _EASE_LINEAR, _off)


class _PulseOverlay(QWidget):
    """フォーカスパルス用の透明オーバーレイ（対象の外周にリングを描き、広がりながら薄れる）"""

    def __init__(self, host: QWidget, target_rect: QRect, color: QColor, spread: int, radius: int):
        super().__init__(host)
        self._target = target_rect
        self._color = color
        self._spread = spread
        self._radius = radius
        self._progress = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setGeometry(target_rect.adjusted(-spread - 2, -spread - 2, spread + 2, spread + 2))
        self.show()
        self.raise_()

    def set_progress(self, v: float) -> None:
        self._progress = v
        self.update()

    def paintEvent(self, _event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        grow = self._spread * self._progress
        c = QColor(self._color)
        c.setAlphaF(max(0.0, 1.0 - self._progress) * 0.9)
        pen = QPen(c, 2.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        inner = QRect(self._spread + 2, self._spread + 2, self._target.width(), self._target.height())
        r = inner.adjusted(int(-grow), int(-grow), int(grow), int(grow))
        p.drawRoundedRect(r, self._radius + grow, self._radius + grow)
        p.end()


def focus_pulse(w: QWidget, color="#38bdf8", duration_ms: Optional[int] = None, radius: int = 6) -> Optional[QWidget]:
    """フォーカスパルス: 対象の輪郭から外へリングが 1 回広がって消える（対象自体は動かさない）"""
    if not _enabled or w is None or not w.isVisible():
        return None
    host = w.window()
    if host is None:
        return None
    top_left = w.mapTo(host, QPoint(0, 0))
    rect = QRect(top_left, w.size())
    overlay = _PulseOverlay(host, rect, QColor(color), MOTION.pulse_spread_px, radius)
    d = duration_ms or MOTION.duration_pulse_ms

    def _done():
        try:
            overlay.hide()
            overlay.deleteLater()
        except RuntimeError:
            pass

    _run_anim(overlay, "_motion_pulse_anim", d, _EASE_OUT, overlay.set_progress, _done)
    return overlay


# ----------------------------------------------------------------------------- 数値ティッカー
_NUM_RE = re.compile(r"\d[\d,]*")


def _numbers_in(text: str) -> List[int]:
    return [int(m.group(0).replace(",", "")) for m in _NUM_RE.finditer(text or "")]


def _format_with_numbers(template: str, values: Sequence[int]) -> str:
    it = iter(values)

    def _rep(m):
        try:
            return f"{int(next(it)):,}"
        except StopIteration:
            return m.group(0)

    return _NUM_RE.sub(_rep, template)


def install_number_ticker(label: QLabel, duration_ms: Optional[int] = None) -> None:
    """
    ラベルの setText() を包み、数値部分だけを前の値から新しい値へ数え上げる（数値ティッカー / カウントアップ）。
    桁幅は最終文字列の幅に先に固定して周辺を動かさない。文字列の形が変わった（数値の個数が違う）ときは即時反映。
    """
    base_set = label.setText  # 既に別の包み（絵文字除去）があればそれを継承する
    d = duration_ms or MOTION.duration_ticker_ms
    orig_min_w = label.minimumWidth()

    def set_text(text: str, _lbl=label, _base=base_set):
        new_text = str(text)
        old_vals = _numbers_in(_lbl.text())
        new_vals = _numbers_in(new_text)
        _stop_anim(_lbl, "_motion_ticker_anim")
        if (not _enabled) or (not new_vals) or len(old_vals) != len(new_vals) or old_vals == new_vals:
            _base(new_text)
            _lbl.setMinimumWidth(orig_min_w)
            return
        # 桁幅を固定（最終文字列の幅）
        try:
            fm = _lbl.fontMetrics()
            pad = _lbl.contentsMargins().left() + _lbl.contentsMargins().right() + 24
            _lbl.setMinimumWidth(fm.horizontalAdvance(new_text) + pad)
        except Exception:
            pass

        def _set(v: float):
            vals = [int(round(o + (n - o) * v)) for o, n in zip(old_vals, new_vals)]
            _base(_format_with_numbers(new_text, vals))

        def _done():
            _base(new_text)
            _lbl.setMinimumWidth(orig_min_w)

        _run_anim(_lbl, "_motion_ticker_anim", d, _EASE_OUT, _set, _done)

    label.setText = set_text  # type: ignore[assignment]


# ----------------------------------------------------------------------------- 押下フィードバック
class ButtonPressMotion(QObject):
    """
    すべての QAbstractButton に押下フィードバックを与えるアプリ全体のイベントフィルタ。
    押した瞬間: 不透明度を下げ 1px 沈む（80ms）。離した瞬間: 元へ戻る（160ms）。
    アニメーション終了後は効果を外すので、普段の描画コストは増えない。
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._pressed: dict = {}

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if not isinstance(obj, QAbstractButton):
            return False
        et = event.type()
        if et == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton and obj.isEnabled() and _enabled:
                self._press(obj)
        elif et in (QEvent.MouseButtonRelease, QEvent.Leave):
            if id(obj) in self._pressed:
                self._release(obj)
        return False

    def _press(self, btn: QAbstractButton) -> None:
        self._pressed[id(btn)] = True
        animate_opacity(btn, 1.0, MOTION.press_opacity, MOTION.duration_fast_ms, _EASE_OUT_FAST)
        if MOTION.press_offset_px:
            # 1px だけ沈める（レイアウト管理下でも次の再配置で元に戻るため安全）
            try:
                btn._motion_orig_pos = btn.pos()  # type: ignore[attr-defined]
                btn.move(btn.pos() + QPoint(0, MOTION.press_offset_px))
            except Exception:
                pass

    def _release(self, btn: QAbstractButton) -> None:
        self._pressed.pop(id(btn), None)
        orig = getattr(btn, "_motion_orig_pos", None)
        if orig is not None:
            try:
                btn.move(orig)
            except Exception:
                pass
            btn._motion_orig_pos = None  # type: ignore[attr-defined]
        fx = btn.graphicsEffect()
        start = fx.opacity() if isinstance(fx, QGraphicsOpacityEffect) else MOTION.press_opacity
        animate_opacity(btn, start, 1.0, MOTION.duration_base_ms, _EASE_OUT, remove_effect_when_done=True)


_installed_filter: Optional[ButtonPressMotion] = None


def install_button_motion(app: Optional[QApplication] = None) -> ButtonPressMotion:
    """アプリ全体に押下フィードバックを 1 回だけ取り付ける"""
    global _installed_filter
    app = app or QApplication.instance()
    if _installed_filter is None and app is not None:
        _installed_filter = ButtonPressMotion(app)
        app.installEventFilter(_installed_filter)
    return _installed_filter  # type: ignore[return-value]
