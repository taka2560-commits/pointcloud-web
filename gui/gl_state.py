# -*- coding: utf-8 -*-
"""
点群を描く間だけ、OpenGL の状態を決まった形にそろえる。

なぜ要るか:
    paintGL は、最後に QPainter で文字や枠（2D の重ね描き）を描く。QPainter は描き終わったあと、OpenGL の状態を
    「ブレンド ON（ONE, ONE_MINUS_SRC_ALPHA）・奥行きの判定 OFF」にして返す。initializeGL が入れた
    「奥行きの判定 ON」は 1 回目の描画でしか効かず、2 回目からの点群は次のように描かれていた。
      - 奥行きの判定が無いので、手前・奥に関係なく、配列であとにある点が上に描かれる
      - 丸い点（GL_POINT_SMOOTH）は、縁の画素の色が下の色へ足されていく。点が重なる所は白く飛ぶ
        （全点表示の 5,490 万点では、建物がほぼ白一色になる）。描くのも遅い（RTX で 0.14 秒 → そろえると 0.06 秒）

やること:
    begin_point_pass() … 点群を描く直前に呼ぶ。奥行きの判定 ON（LEQUAL・書き込みあり）・ブレンド OFF にして、
                          奥行きのバッファを消す（paintGL の頭の glClear は、前の描画が残した状態によっては奥行きを消せない）。
                          色の書き込みは RGB だけにする（丸い点は縁の画素の「かぶり具合」を透明度として書くので、
                          そのままだと、スクリーンショットの PNG に半透明の画素が混ざる）
    end_point_pass(saved) … 点群を描き終わったら呼ぶ。begin の前の状態へ戻す。
                          格子・断面の枠・計測の線・軸の印は、今までと同じ状態で描かれる（見た目を変えない）

約束:
    - ここでは「点群を描く間の状態」だけを決める。ほかの部品の状態は変えない（戻す）
    - 何が起きても例外を出さない（描画を止めない）
"""
from typing import Optional, Tuple

from OpenGL import GL as gl

Saved = Tuple[bool, bool, int, bool, Tuple[bool, bool, bool, bool]]


def _first_int(value) -> int:
    try:
        return int(value)
    except TypeError:
        return int(list(value)[0])


def _bools(value, count: int) -> Tuple[bool, ...]:
    try:
        items = [bool(x) for x in value]
    except TypeError:
        items = [bool(value)]
    if len(items) < count:
        items = items + [True] * (count - len(items))
    return tuple(items[:count])


def read_point_pass_state() -> Saved:
    """いまの状態（ブレンド・奥行きの判定・奥行きの比べ方・奥行きの書き込み・色の書き込み）を読む"""
    return (
        bool(gl.glIsEnabled(gl.GL_BLEND)),
        bool(gl.glIsEnabled(gl.GL_DEPTH_TEST)),
        _first_int(gl.glGetIntegerv(gl.GL_DEPTH_FUNC)),
        _bools(gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK), 1)[0],
        _bools(gl.glGetBooleanv(gl.GL_COLOR_WRITEMASK), 4),
    )


def begin_point_pass() -> Optional[Saved]:
    """点群を描く直前に呼ぶ。前の状態を返す（end_point_pass へ渡す）"""
    saved: Optional[Saved] = None
    try:
        saved = read_point_pass_state()
    except Exception:
        saved = None
    try:
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)
        # 奥行きのバッファは、ここで消す。paintGL の頭の glClear は、前の描画が「奥行きの書き込み OFF」を残していると
        # 奥行きを消せない（RTX で、文字を描かない描画が続くと起きた。前の視点の奥行きが残り、点が消える）。
        # 点群より前に描く格子は、奥行きを書かないので、ここで消しても絵は変わらない
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_FALSE)
    except Exception:
        pass
    return saved


def end_point_pass(saved: Optional[Saved]) -> None:
    """点群を描き終わったら呼ぶ。begin_point_pass の前の状態へ戻す"""
    if saved is None:
        try:
            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
        except Exception:
            pass
        return
    blend, depth, depth_func, depth_mask, color_mask = saved
    try:
        (gl.glEnable if blend else gl.glDisable)(gl.GL_BLEND)
        (gl.glEnable if depth else gl.glDisable)(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(depth_func)
        gl.glDepthMask(gl.GL_TRUE if depth_mask else gl.GL_FALSE)
        gl.glColorMask(*[gl.GL_TRUE if c else gl.GL_FALSE for c in color_mask])
    except Exception:
        pass
