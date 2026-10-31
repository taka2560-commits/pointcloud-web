# AGENTS.md — 複数AI・複数開発者で本リポジトリを並行開発するための共通ルール

本プロジェクト（E57点群ノイズ処理・断面スライサー ツール）は、**複数のAIエージェント（元AI と Claude）と人間が同時に開発**しています。
互いの作業を上書き・消失させないために、すべてのエージェントは以下を守ってください。
設計思想・アーキテクチャ・厳守ルールは `AI_HANDOVER.md` を、品質状況は `QA_REPORT_2026-10-31.md` を参照してください。

## 1. 作業場所（フォルダとブランチ）

| 担当 | フォルダ | ブランチ |
| :--- | :--- | :--- |
| 元AI（従来からの開発AI） | `C:\Users\taka2\Documents\点群マイツール` | `main` |
| Claude | `C:\Users\taka2\Documents\点群マイツール-claude`（git worktree） | `claude/qa-handover` など `claude/*` |

- 2つのフォルダは**同じ `.git` を共有する worktree** です。履歴・ブランチ・フック・stash は共通です。
- **他方のフォルダのファイルを直接編集しない**。相手の成果の取り込みは必ず **git マージ**で行う。
- 相手のフォルダで `git stash` / `git reset --hard` / `git checkout -- <file>` を実行しない（未コミットの作業が消える）。
- worktree 側には `dist/`（配布EXE）が無いので `起動.bat` は自動的にソース起動になる。元フォルダも、`main.py` と `core/` がある開発チェックアウトでは常にソース起動する。

## 2. 言語

- 会話・ドキュメント・実装計画・コード内コメント・Gitコミットメッセージは**すべて日本語**。

## 3. コミット

- 作業開始時に `git status` と `git log -1` を確認する。
- 作業終了時は**必ずコミット**する（`git add -A` 相当。「一部のファイルだけコミット」は禁止。未コミット差分が溜まると
  相手が基準点を取れず、2026-10-31 のように +4,700 行の未コミット作業が発生する）。
- コミットメッセージは `feat:` / `fix:` / `docs:` / `chore:` / `test:` の接頭辞＋日本語の要約。
- `.git/hooks/pre-commit` に全5ゲートが登録されています（約10秒、GUIウィンドウが一瞬開く）。
  ゲート NG のコミットはできません。`--no-verify` でのバイパスは禁止。

## 4. ファイルの担当（衝突を避けるため）

| ファイル | 原則の担当 | 備考 |
| :--- | :--- | :--- |
| `gui/main_window.py`, `gui/gl_viewer.py` | 元AI | Claude が触る場合は「小さく・まとめて・main 取り込み直後に」。着手前に `git -C <元フォルダ> diff gui/main_window.py` で元AIが同じ関数を編集中でないか確認する |
| `core/*`, `gui/worker_thread.py`, `gui/progress_dialog.py` | 共有（小さな変更単位で） | 変更したら必ず該当テストを追加 |
| `tests/*`, `quality_gate.py`, `*.spec`, `*.bat`, `requirements*.txt`, `.gitattributes` | Claude 優先（品質・ビルド基盤） | 元AIが変更しても構わないが、同じコミットで AI_HANDOVER.md の記述も更新する |
| `AI_HANDOVER.md`, `AGENTS.md`, `QA_REPORT_*.md` | 共有 | **実装と記述がずれたら同じコミットで直す**（資料と実体の乖離が過去の主要リスク） |

## 5. 品質ゲート

```bash
python -X utf8 quality_gate.py          # 全5ゲート（マージ前・配布前は必ずこちら）
python -X utf8 quality_gate.py --quick  # Gate 1/3/5 のみ（GUIを開かない。作業中の確認用）
python -X utf8 quality_gate.py --fix-eol  # .bat の改行コードを CRLF へ修復（既定では検査のみ）
```

- `quality_gate.py --install-hook` は実行しない（既に登録済み。hooks は全 worktree 共有）。
- テストは `unittest` 形式（`tests/test_*.py`）。新機能・バグ修正には必ずテストを追加する。

## 6. マージ手順（main ⇄ claude/*）

1. 元フォルダで `git status` を確認し、差分があれば元AI側でコミットする（スナップショットでよい）。
2. Claude worktree で `git merge main`（**rebase は使わない**）。衝突は Claude 側で解決し、main 側（元AI）を正として Claude の変更を再適用する。
3. 全5ゲート合格を確認してから main へ `git merge --no-ff claude/qa-handover`（または GitHub の Pull Request 経由）。
4. 元フォルダで全5ゲートを再実行し、両フォルダが同じコミットを指していることを確認する。
5. `origin` への push はユーザーの確認を得てから行う。

## 7. 依存関係とビルド

- 依存パッケージを追加・変更したら **`requirements-desktop.txt`・`起動.bat` の import 検査行・`AI_HANDOVER.md` §3** を同じコミットで更新する。
- 配布EXEは `EXE配布パッケージ作成.bat`（内部で `E57_PointCloud_Tool.spec` を使用）でビルドする。
  `pyinstaller main.py` を直接実行しない（spec が上書きされ、マニュアル同梱・open3d 収集設定が消える）。

## 8. 守るべき設計憲章（詳細は AI_HANDOVER.md §2）

1. E57 はロスレス（倍精度 / ScaledInteger の生整数・スキーマ・ポーズ・メタデータを保持、`.tmp` 経由のアトミック置換）。
2. 読み込み・フィルタ・座標変換・保存などの重い処理は**メインスレッドで実行しない**（`gui/worker_thread.py` のワーカー、汎用 `FunctionWorker` を使う）。
3. 取り込み時の座標系変換は表示・編集用。**保存は元の座標系・元のスケールに逆変換して書き戻す**（`FileSaveWorker` が自動で行う）。
4. 大規模点群のクラスタリングは代表点安全リミッター（`REPRESENTATIVE_POINT_LIMIT = 200,000`）を介す。
5. フィルタが「全点を除去」する結果になる場合は安全側（全点保持＋警告）に倒す。
