# CLAUDE.md — Claude がこのリポジトリで作業するときの指針

- まず `AGENTS.md`（複数AI並行開発の共通ルール）と `AI_HANDOVER.md`（設計憲章・アーキテクチャ）を読むこと。
- 会話・ドキュメント・コメント・コミットメッセージは**すべて日本語**。
- Claude の作業場所は worktree `C:\Users\taka2\Documents\点群マイツール-claude`（ブランチ `claude/*`）。
  元AIのフォルダ `C:\Users\taka2\Documents\点群マイツール`（`main`）は読み取り専用。取り込みは git マージのみ。
- `gui/main_window.py` / `gui/gl_viewer.py` は元AIのホットファイル。変更は最小・まとめて・main 取り込み直後に。
- コード変更後は `python -X utf8 quality_gate.py --quick`、コミット前・マージ前は全ゲート `python -X utf8 quality_gate.py`。
- pre-commit フックが全ゲートを実行する。`--no-verify` は使わない。
- 依存を追加したら `requirements-desktop.txt`・`起動.bat` の import 検査・`AI_HANDOVER.md` §3 を同時に更新。
- 品質状況・既知の問題・backlog は `QA_REPORT_2026-10-31.md` に記録し、修正したら同じコミットで更新する。
