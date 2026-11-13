"""
テストと品質ゲートで作ったウィンドウ・部品を、閉じたあと本当に破棄する

なぜ要るか（2026-11-14 の点検で見つけた）
    deleteLater() は「イベントループへ戻ったら破棄する」という予約で、テストのように app.exec() を回していない所では、
    QApplication.processEvents() を呼んでも実行されない。閉じただけのウィンドウが、見えないまま生き続ける。
    - 残ったウィンドウの部品も、アプリ全体の見た目の切り替え（QApplication.setStyleSheet）のたびに塗り直されるので、
      ウィンドウを作るテストが、後になるほど遅くなる。
    - 従来 UI のウィンドウを残したまま、新 UI のウィンドウがテーマを切り替え、次のウィンドウが作られると、
      Qt の中でアクセス違反が起き、テストのプロセスごと落ちた。
      実測: tests.test_buttons_and_states を 14 回走らせて 3 回。条件を絞った再現では、残すと 3 回中 2 回、破棄すると 0 回。
    実際のアプリはウィンドウが 1 つで、イベントループの中で動くので、この組み合わせは起きない。

使い方: MainWindow や PointCloudViewer を作ったテストは、tearDown の最後で destroy_widget() を呼ぶ。
tests/test_static_names.py が、呼び忘れを見張る。
"""

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication


def flush_deferred_deletes() -> None:
    """予約されている破棄（deleteLater）を、その場で実行する"""
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()


def destroy_widget(widget, close: bool = True) -> None:
    """ウィンドウや部品を（close=True なら閉じてから）その場で破棄する"""
    if widget is None:
        return
    try:
        if close:
            widget.close()
        widget.deleteLater()
    except RuntimeError:
        pass  # すでに破棄されている
    flush_deferred_deletes()
