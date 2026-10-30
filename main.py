"""
E57点群ノイズ処理ツール GUIアプリケーション エントリーポイント
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from core.logger import setup_logging, log_system_environment, log_info, log_error
from gui.main_window import MainWindow


def main():
    # 1. システム履歴ロギングの初期化と起動環境の記録
    setup_logging()
    log_system_environment()

    # 高DPIスケーリングの有効化
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("E57点群ノイズ処理ツール")

    try:
        log_info("メインウィンドウを初期化中...")
        window = MainWindow()
        window.show()
        window.raise_()
        window.activateWindow()
        log_info("メインウィンドウの表示完了 (イベントループ開始)")

        exit_code = app.exec()
        log_info(f"アプリケーションが正常終了しました (終了コード: {exit_code})")
        sys.exit(exit_code)
    except Exception as e:
        log_error(f"アプリケーション実行中に予期しない例外が発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
