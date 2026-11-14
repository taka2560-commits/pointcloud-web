"""
E57点群ノイズ処理ツール GUIアプリケーション エントリーポイント

起動の順番（2026-11-14 から）
    1. ログを用意する
    2. 異常終了の記録を始める（core/crashlog.py。前回までに正常に終了しなかった起動があれば、ログへ書く）
    3. 描画に使う GPU の希望を出す（gui/gpu_select.py。OpenGL を使い始める前でないと効かない。2026-11-15 から）
       そのあと、画面の部品と重いライブラリ（PySide6・OpenGL・open3d・pye57）を読み込む
    4. ウィンドウを作って、イベントループを回す
3 を 1・2 のあとに置くのは、ライブラリを読み込む途中で落ちたり例外が出たりしたときにも、記録が残るようにするため
（以前はファイルの先頭で読み込んでいて、そこで起きたことはどこにも残らなかった）。
"""

import atexit
import sys

from core import crashlog
from core.logger import (
    get_logger, get_logs_dir, log_error, log_exception, log_info, log_system_environment, log_warning, setup_logging,
)

SESSION_SIZE_CHECK_MS = 60_000  # 「起動中の印」が大きくなりすぎていないかを見る間隔


def _begin_crash_record() -> crashlog.Session:
    """
    異常終了の記録を始める。スレッドの中の例外も、ここからログへ残るようにする。
    調べる仕掛けのせいでアプリが起動しなくなってはいけないので、何が起きても先へ進む。
    """
    try:
        try:
            from core.build_info import get_build_info

            build = get_build_info().describe()
        except Exception:
            build = ""
        session = crashlog.begin_session(get_logs_dir(), build=build)
        logger = get_logger()
        crashlog.install_exception_hooks(logger.critical, logger.error)
    except Exception as e:
        log_warning(f"異常終了の記録を始められませんでした（アプリはそのまま使えます）: {e!r}")
        session = crashlog.Session(logs_dir="")
    # Python が終わるときに印を消す。ウィンドウや OpenGL の後始末の途中で落ちたときも記録に残るよう、ここまで消さない
    atexit.register(session.end)
    return session


def main():
    # 1. システム履歴ロギングの初期化と起動環境の記録
    setup_logging()
    session = _begin_crash_record()
    log_system_environment()
    log_info(f"・異常終了の記録: {'有効' if session.active else '無効（ログのフォルダに書けません）'}")
    for line in crashlog.describe_for_log(session):
        log_warning(line)

    try:
        # OpenGL を使い始める前に、描画に使う GPU の希望を出す（内蔵 GPU と高性能の GPU の両方が載っている PC 向け。gui/gpu_select.py）
        from gui.gpu_select import apply_gpu_choice

        log_info(f"・描画の GPU の選び方: {apply_gpu_choice()}")

        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import QApplication

        from gui.diagnostics import install_qt_message_log, show_previous_crash_notice, uninstall_qt_message_log
        from gui.main_window import MainWindow
    except Exception as e:
        log_exception(f"起動に必要な部品を読み込めませんでした: {e}")
        raise

    # 高DPIスケーリングの有効化
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    install_qt_message_log()
    app = QApplication(sys.argv)
    app.setApplicationName("E57点群ノイズ処理ツール")
    # ウィンドウを閉じたときも、Windows の終了・ログオフのときも、ここを通る。通れば「終了処理が始まった」と印に書く
    app.aboutToQuit.connect(session.mark_clean_exit)

    try:
        log_info("メインウィンドウを初期化中...")
        window = MainWindow()
        window.show()
        window.raise_()
        window.activateWindow()
        log_info("メインウィンドウの表示完了 (イベントループ開始)")
        try:
            show_previous_crash_notice(window, session.abandoned)
        except Exception as e:
            log_warning(f"前回の異常終了の知らせを出せませんでした: {e!r}")

        size_timer = QTimer()
        size_timer.setInterval(SESSION_SIZE_CHECK_MS)

        def _check_session_size():
            if session.check_size() and session.trimmed == 1:
                log_warning("続行できる低レベルの例外が大量に記録されています（異常終了の記録の古い分は捨てました。アプリは動き続けています）")

        size_timer.timeout.connect(_check_session_size)
        size_timer.start()

        exit_code = app.exec()
        size_timer.stop()
        noted = session.fault_bytes()
        if noted:
            log_info(f"終了時: 続行できた低レベルの例外の記録が {noted:,} バイトありました（正常終了なので、印と一緒に消します）")
        log_info(f"アプリケーションが正常終了しました (終了コード: {exit_code})")
        uninstall_qt_message_log()
        sys.exit(exit_code)
    except Exception as e:
        log_error(f"アプリケーション実行中に予期しない例外が発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
