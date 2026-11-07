# E57点群ノイズ処理・断面スライサー ツール 開発引継ぎ資料 (AI Handover Document)

本資料は、今後**別のAI（Claude, ChatGPT, Cursor, Gemini, GitHub Copilot等）や新しい開発者**が本プロジェクトの開発・保守を引き継ぐ際に、設計思想、アーキテクチャ、厳守すべきルール、過去のトラブルと対策、テスト・ビルド手順を正確に把握し、事故なく迅速に開発を進めるための完全引継ぎドキュメントです。

---

## 1. プロジェクト概要

- **プロダクト名**: E57点群ノイズ処理・断面スライサー ツール (`E57_PointCloud_Tool`)
- **主要機能**:
  - 3Dレーザースキャナーから取得された大容量E57形式（`.e57`）点群データの読み込み・高速可視化。
  - 数千〜数千万点の点群に対するフリーズなし・高速ノイズ自動除去（SOR、ROR、孤立クラスタ、反射強度、距離、大まかな孤立塊除去）。
  - レイヤー管理システム（通常点群／ノイズ点群の二層構造、ノイズのピンポイント復元）。
  - 多彩な手動編集ツール（矩形、円形、多角形ラッソ、ブラシ選択、消しゴム）。
  - XYZ 3軸断面スライサー、および2点ピックによる斜め壁アラインメント断面スライス。
  - E57形式の完全ロスレス（倍精度座標、ScaledInteger、反射強度、色、スキャナーポーズ、メタデータ完全保持）書き出し。
  - 現場シチュエーション別実践レシピセット（8種類）のワンクリック適用。
  - 取り込み座標系変換（数学座標系、測量座標系X/Y入替、Y-up変換、Z軸反転）。
- **主要対象ユーザー**: 測量士、土木・建築技術者、点群データを扱う現場担当者（点群初心者からプロフェッショナルまで）。

---

## 2. 厳守すべき絶対ルール・設計憲章

今後のAIが開発を行う際、**以下のルールは絶対に遵守**してください。

### ① 【言語規則】日本語出力の厳守
- **会話、ドキュメント、実装計画書、コード内のコメント、Gitコミットメッセージは、必ずすべて日本語**で記述してください。
- 専門用語（英語の関数名など）を扱う場合でも、解説やユーザーへの返答は日本語で行ってください。

### ② 【データ保全】E57完全ロスレス保持の原則
- 点群データを単精度Float32に安易にダウンキャストして保存してはなりません（ミリ単位の測量精度が破壊されます）。
- 元のE57が持つ `FloatNode`（倍精度 Float64）または `ScaledIntegerNode`（スケール・オフセット）、カラー（RGB）、反射強度（Intensity）、法線、タイムスタンプ、スキャナーポーズ（Pose行列）を**1ビットたりとも損失させずに再保存**してください。
- 保存時は必ず一時ファイル（`.tmp`）に書き込んでからアトミックに置換し、書き込み途中のPC停止等によるデータ破損を防止してください（`core/e57_io.py` 参照）。

### ③ 【パフォーマンス】数千万点処理でのUIフリーズ・メモリ枯渇防止
- 点群データは2,000万点〜5,000万点規模になることがあります。単純な $O(N^2)$ ループや、全点に対する素朴なDBSCAN・KDTree構築は**PCフリーズ（応答なし）の直接原因**になります。
- 巨大点群のノイズ除去では、**「適応型ボクセルグリッド化」「1Dビットパック高速ハッシュ」「代表点安全リミッター（`REPRESENTATIVE_POINT_LIMIT` = 20万点。ボクセル化後に超過していればボクセルを粗くして再量子化）」**を必ず介してください（`core/filters.py` 参照。※旧版資料の「50万点」は誤記）。
- ファイル読み込み、ノイズ処理、座標変換、ファイル書き出しなどの重い処理は、**絶対にメインスレッドで直接実行してはなりません**。必ず `gui/worker_thread.py`（QThread）に逃がし、キャンセル可能な進捗ダイアログ（`gui/progress_dialog.py`）を介してください。
  - 専用ワーカーが無い処理は汎用 `FunctionWorker` と `MainWindow._run_in_background()` を使ってください（地面抽出・主平面・マジックワンド・エッジ飛散除去がこの方式）。
  - ワーカーの停止は `cancel()` → `wait()` の協調的中断が原則です。`QThread.terminate()` は最後の手段に限ります。
- フィルタの結果が「全点除去」になる場合は、パラメータ設定ミスとみなして**全点保持＋警告ログ**に倒してください（`filter_isolated_clusters` の `allow_remove_all=False` が既定）。

### ④ 【座標系】数学座標系準拠と測量座標系の違い
- 内部の3Dビューア（`gui/gl_viewer.py`）は、**厳密な「数学座標系（デカルト右手系・Z-up）」**で実装されています：
  - **画面右**: $+X$ 軸（横）
  - **画面上**: $+Y$ 軸（縦・奥）
  - **画面手前**: $+Z$ 軸（高さ・上空）
- 一方、日本の公共測量（平面直角座標系）は **「$X$＝北（縦）、$Y$＝東（横）」** のため、数学系とは縦横が逆です。
- ビューアの軸方向を勝手に改造してはなりません。公共測量データに対応するため、インポート時に座標を入れ替える **「取り込み座標系セレクター（`transform_coordinate_system`）」** が実装されています。

### ⑤ 【座標系変換と保存】保存は常に元の座標系・元のスケールで
- 取り込み時の座標系変換（測量X/Y入替・Y-up・Z反転）は**表示・編集のための変換**です。`E57ScanData.coord_mode` に適用モードが記録されます。
- 保存時は `FileSaveWorker` が `restore_scan_original_coordinates()` で元の座標系へ逆変換してから書き出し、終了後に表示用座標系へ再適用します。これにより ScaledInteger の生整数・スケール定義がそのまま使われ、**ビット完全一致**で書き戻せます（`tests/test_coord_roundtrip.py`）。
- `core.processor.transform_scan_coordinate_system()` は points だけでなく `raw_unscaled_fields` と `field_schemas` も軸ごとに付け替えるため、変換後も `points == raw × scale + offset` が厳密に成り立ちます。この整合を壊す変更をしないでください。

### ⑧ 【pose】表示は pose 適用後の座標、保存は生座標と pose をそのまま使う
- E57 の点はスキャンのローカル座標で記録され、`world = R @ local + t` で現場の座標になります。
- `E57ScanData.points` と `raw_fields` は保存用の生座標です。pose を適用して書き換えないでください（ロスレス保存が壊れます）。
- `PointCloudViewer.points_raw`（名前に反して pose 適用後の表示座標）と `scan.points` は点の順番だけが共通です。両者の座標を混ぜず、インデックスで対応させてください。表示座標は `core.processor.scan_display_points()` で作ります。
- 距離フィルタの基準はスキャン原点です。表示座標で実行するときは `MainWindow._distance_reference_points()` を `distance_points` に渡してください。
- 座標系変換（測量 X/Y 入替など）は pose の後に適用します。pose が恒等ならコピーせず `scan.points` を共有します。回転を含むファイルでは全点ぶん（2,200 万点で約 500MB）メモリが増えます。

### ⑨ 【1 億点超】間引き読込で判定し、保存は元ファイルの全点へ転写する
- 実務の点群は 1 億点を超えます。読込上限（設定「読込上限」。自動は 16GB で 3,000 万点、32GB で 6,000 万点）を超えるファイルは `read_e57_scans(point_budget=…)` が **k 点に 1 点の間引き**で読み込み（`E57ScanData.is_subsampled`、`original_point_count`）、ユーザーはその間引き点群でノイズ判定をします。
- 保存時に間引き点群をそのまま書くと納品ファイルの点密度が落ちます。保存ダイアログの既定は **「全点で保存」**（`FullResolutionSaveWorker`）で、スキャンごとに元ファイルを全点で読み直し（`read_e57_scans(scan_indices=[…])`）、判定点のラベルを最近傍の判定点から全点へ写し（`core/fullres_transfer.py`）、`E57StreamWriter` で 1 スキャンずつ追記します。メモリは「間引き点群 + 1 スキャンの全点」で済みます。
- 転写の規則: 判定点 q ごとに他の判定点までの最短距離 s(q) を持ち、全点の各点は最近傍の判定点との距離が `max(factor × s(q), 1cm)` 以内ならそのラベルを引き継ぎ、遠ければ**孤立点として除外**します。factor は間引き率 k に応じて `max(3, k+1)`（格子状の点群では同じ面の未判定点が必ず (k/2)×点間隔以内に判定点を持つため、面の点は孤立点になりません）。
- 照合はスキャンのローカル座標で行います（pose と取り込み時の座標系変換は両側に掛かっていない。判定点は `inverse_transform_coordinate_system()` で座標系変換を外す）。
- 各スキャンは読み込み元（`source_file` / `source_scan_index` / `source_signature` = サイズと更新時刻）を持ちます。保存前に照合し、元ファイルが変更されていれば保存しません。合成データ（`source_file=None`）では全点転写は使えず、「間引いたまま保存」のみです。
- 検証は `tests/test_fullres_save.py`（間引き k=3 で浮遊塊が全点除外・面が残る・手動削除範囲が全点で消える・ロスレス・分割保存・変更検知・中止）。

### ⑦ 【UI モード】新UIと従来UIを両立させ、いつでも戻せるようにする
- 2026-10-31 のデザイン改修は **「後付けアダプタ」方式**です。`MainWindow._init_ui()` が組み立てた従来UIに対し、`gui/modern_ui.py` の `apply_modern_ui()` が「作業順の並べ替え・メニュー/ステータスバー・重複ボタンの非表示・絵文字除去・インライン QSS 無効化・折り返しバー・空状態カード」を起動後に適用します。従来UIのコードは削除していません。
- UI モードは `gui/ui_mode.py` が決めます（優先順: 起動引数 `--classic-ui`/`--modern-ui` ＞ 環境変数 `E57_UI_MODE` ＞ 設定「システム → UIスタイル」 ＞ 既定 modern）。**不具合時は `起動.bat` に `--classic-ui` を付けるか、設定で「従来UI」を選べば改修前に戻ります。**
- 新UIの見た目は `gui/design_tokens.py` のトークンから生成した QSS（`get_theme_qss("modern")`、アプリ全体に適用）で描きます。色・文字サイズ・角丸はここだけで変更してください。従来UIの `DARK_THEME_QSS` は変更しないでください。
- 新UIで新しいウィジェットを追加するときは、インライン `setStyleSheet` を書かず `setProperty("role", "primary|outline|ghost|danger|segment|danger-toggle")` と objectName で表現し、絵文字をラベルに使わないでください（Gate4 が検査します）。両モードで動くコードにする場合は `self.ui_mode == UI_MODE_MODERN` で分岐します（例: レイヤー行のロックボタン）。
- 品質ゲート Gate4 は modern で全シナリオを検証したあと、classic でも起動・タブ・モード切替のスモークを行います。
- 新UIには **テーマ**（`dark` 既定 / `kiseki` = 「Stand 軌跡」準拠のライト）があります。`gui/design_tokens.py: THEMES` のトークンだけが違い、QSS 生成・部品・ロジックは共通です（`get_theme_qss("modern", theme)`）。解決順は UI モードと同じ（`--theme=kiseki` ＞ `E57_UI_THEME` ＞ 設定「テーマ (新UI)」 ＞ dark）。設定画面で選ぶと `MainWindow.apply_ui_theme()` がその場で切り替えます（再起動不要）。新しい色を足すときは両テーマに値を入れ、`tests/test_design_tokens.py` のコントラスト検査を通してください。
- 新UIの **モーション**は `gui/motion.py` に集約しています（押下・出入り・カスケード・展開・数値ティッカー・シェイク・バウンス・パルス。選定理由は `docs/DESIGN_STYLE_PLAN.md` §9）。ルール: 1 回きり・420ms 以内・位置と透明度のみ・終了後は描画効果を外す。設定「アニメーション」OFF（`set_motion_enabled(False)`）で全て即時反映になるよう `motion_enabled()` を見てください。Gate4 は dark・classic・kiseki の 3 通りで起動し、モーション後に描画効果が残らないことも検査します。

### ⑥ 【UI スタイル】セレクタ無しのインライン QSS を書かない
- `widget.setStyleSheet("background: transparent;")` のようにセレクタを付けない宣言は**その子孫すべてに伝播**し、グローバル QSS（`QPushButton#execute_button` 等）を打ち消します（2026-10-31 に主要ボタンが文字だけに見える不具合として顕在化）。
- 必ず `QWidget#name { ... }` のように objectName / クラスで限定するか、`setProperty("role", ...)` ＋ `gui/styles.py` のグローバル QSS で表現してください。デザイン改修の方針は `docs/DESIGN_STYLE_PLAN.md` を参照。

---

## 3. 技術スタック & 実行環境

| カテゴリ | 技術・ライブラリ | 用途・特記事項 |
| :--- | :--- | :--- |
| **言語** | Python 3.10+ (3.14動作実績あり) | Windows 64bit環境 |
| **GUI** | PySide6 (Qt for Python) | メインUI、リボンバー、マルチスレッド (`QThread`) |
| **3D描画** | PyOpenGL, PyOpenGL_accelerate | モダンOpenGL (VBO GPUキャッシュ、Shader、深度フォグ、視点ギズモ) |
| **点群入出力** | pye57 | E57フォーマットの低レベル読み書き（AST構造保持） |
| **数値・幾何計算** | NumPy, SciPy (`cKDTree`) | ベクトル演算、超高速近傍探索、ボクセル化 |
| **クラスタ解析・SOR/ROR** | Open3D（**必須**） | `core/filters.py` が無条件 import。DBSCAN・統計/半径外れ値除去・RANSAC平面検出。scikit-learn は使用していない |
| **ポーズ計算** | pyquaternion（**必須**） | `core/e57_io.py` のスキャナーポーズ（クォータニオン）処理 |
| **パッケージング** | PyInstaller | Python不要のスタンドアロンEXE (`dist/E57_PointCloud_Tool/`)。`E57_PointCloud_Tool.spec` でマニュアル同梱・open3d 収集を定義 |

依存の正確な一覧は `requirements-desktop.txt`（最小バージョン）と `requirements-lock.txt`（動作確認済みの固定バージョン）を参照。**依存を追加したら `requirements-desktop.txt`・`起動.bat` の import 検査行・本表を同じコミットで更新すること。**
| **Web版** | HTML5, JavaScript, Three.js | ブラウザ簡易ビューア (`index.html`, `app.js`) |

---

## 4. ディレクトリ構成 & モジュール責務

```
点群マイツール/
├── main.py                     # アプリ起動エントリーポイント（高DPI設定、ロガー初期化、例外フック）
├── quality_gate.py             # 全5ゲート自動回帰テストスクリプト（最重要CI検証。--quick / --fix-eol あり）
├── E57_PointCloud_Tool.spec    # PyInstaller ビルド設定ファイル（マニュアル同梱・open3d/pye57 収集を定義）
├── 起動.bat                    # ワンクリック起動（開発チェックアウトではソース起動、配布ZIPではEXE起動）
├── 品質チェック実行.bat        # テスト一括実行スクリプト
├── EXE配布パッケージ作成.bat   # spec からEXEビルド → 旧ZIPを dist/archive へ退避 → 新ZIP生成
├── requirements-desktop.txt    # 依存パッケージ（最小バージョン）
├── requirements-lock.txt       # 動作確認済みの固定バージョン
├── AGENTS.md / CLAUDE.md       # 複数AI並行開発の共通ルール（作業場所・担当ファイル・マージ手順）
├── QA_REPORT_2026-10-31.md     # 総合品質管理チェック報告書（欠陥一覧・修正状況・backlog）
├── assets/icons/               # 新UIの SVG アイコン（コンボ/スピンの矢印・チェック）。spec の datas で同梱
├── docs/                       # UI/UX レビュー（UI_UX_REVIEW_*.md）、デザイン改修計画（DESIGN_STYLE_PLAN.md）、画面証跡 ui_review/（新UI）, ui_review/before/（改修前）, ui_review/classic/（従来UIモード）
├── tools/                      # 開発補助: capture_ui_screens.py（実画面スクリーンショット）、check_contrast.py（コントラスト比）
├── .gitattributes              # 改行コード固定（.py/.md 等は LF、.bat は CRLF、.e57 はバイナリ）
│
├── core/                       # コアロジック（UI非依存）
│   ├── e57_io.py               # E57入出力エンジン（ScaledInteger・ポーズ・全属性の完全ロスレス保持）
│   ├── filters.py              # ノイズ除去・間引きアルゴリズム（SOR, ROR, DBSCAN, 孤立塊, ボクセル）
│   ├── processor.py            # 点群処理マネージャー、FilterConfig、座標系変換関数
│   └── logger.py               # システム履歴ロガー（日別ログ、例外トレース、クラッシュダンプ）
│
├── gui/                        # デスクトップGUI層（PySide6）
│   ├── main_window.py          # メインウィンドウ、UIリボン、レイヤー管理、レシピ、ステータス
│   ├── gl_viewer.py            # OpenGL 3D点群ビューア（カメラ、VBO、色分け、ギズモ、手動選択）
│   ├── worker_thread.py        # バックグラウンドワーカスレッド（非同期読み込み・処理・保存、汎用 FunctionWorker）
│   ├── layer_history.py        # 差分ベースのレイヤー Undo/Redo 履歴（メモリは変化点数に比例）
│   ├── ui_mode.py              # UI モード（modern / classic）の解決と保存（起動引数・環境変数・設定）
│   ├── design_tokens.py        # 新UIのデザイントークンと QSS 生成（色・文字サイズ・角丸・間隔の唯一の定義）
│   ├── modern_ui.py            # 新UIアダプタ（従来UIに後付け: 作業順・メニュー/ステータスバー・空状態カード・絵文字除去）
│   ├── flow_layout.py          # 折り返しレイアウト（狭い画面で上部/下部バーのボタンが切れないようにする）
│   ├── progress_dialog.py      # キャンセル対応モダン進捗ダイアログ（ApplicationModal）
│   ├── styles.py               # モダンダークテーマQSSスタイルシート
│   └── preview_dialog.py       # プレビュー用補助ダイアログ
│
├── tests/                      # テストスイート（unittest 形式。quality_gate.py の Gate 5 が discover で一括実行）
│   ├── test_io_integrity.py    # E57入出力の完全ロスレス性・属性保持テスト
│   ├── test_processor.py       # フィルタパラメータ・処理パイプラインテスト
│   ├── test_layer_integration.py # 通常/ノイズ二層レイヤー連携テスト
│   ├── test_ui_and_settings_integrity.py # UIスモーク・設定永続化テスト
│   ├── test_filters_safety.py  # 全点消失ガード・代表点安全リミッター
│   ├── test_coord_roundtrip.py # 座標系変換のロスレス往復（ScaledInteger 軸別スケール・FileSaveWorker 経由）
│   ├── test_e57_read_consistency.py # 読み込み点数とヘッダー宣言点数の照合
│   └── test_undo_snapshot.py   # 差分 Undo 履歴（全点コピー方式との等価性・メモリ比例・外部 clear）
│
├── dist/                       # ビルド成果物（配布用。Git 管理外）
│   ├── E57_PointCloud_Tool/    # スタンドアロンEXE実行フォルダ
│   ├── E57_PointCloud_Tool_Dist.zip # 最新の配布用ZIPパッケージ
│   └── archive/                # 過去ビルドの退避フォルダ
│
├── samples/                    # テスト用点群データ (.e57)。サンプル点群.e57 のみ Git 管理
└── logs/                       # 実行時のシステム履歴ログフォルダ（日別UTF-8出力。Git 管理外）
```

※ Claude 用の git worktree `../点群マイツール-claude`（ブランチ `claude/*`）が同じ `.git` を共有しています。詳細は `AGENTS.md`。

---

## 5. 主要モジュールの深層解説

### (1) `core/e57_io.py`（E57入出力）
- `pye57` ライブラリのラッパーですが、標準の `read_points()` では欠落しやすい **ScaledInteger（スケール・オフセット値）やカスタム拡張ノード** を保護します。
- `read_e57_scans()` / `iter_e57_scans()`: 点群データ（XYZ, Intensity, RGB, ScanIndex）とともに、E57のXMLツリー構造、スキャナーポーズ行列、各フィールドのデータ型情報を完全抽出します。`iter_e57_scans()` はスキャンを 1 つずつ返すジェネレータ、`scan_indices=[…]` で特定のスキャンだけを読めます（全点転写保存が使う）。
- `write_e57_scans()` / `E57StreamWriter`: 元のデータ型が `ScaledInteger` であった場合はスケール値を復元して書き込み、ファイルサイズ肥大化や桁落ちを防止します。`E57StreamWriter` は `add_scan()` で 1 スキャンずつ追記でき、`close()` で再オープン検証に合格したときだけ出力先へアトミック置換します。

### (2) `core/filters.py`（ノイズ除去アルゴリズム）
- **統計的外れ値除去 (SOR)**: `scipy.spatial.cKDTree` を用いた $k$ 近傍平均距離の正規分布外れ値判定。
- **半径外れ値除去 (ROR)**: 半径 $r$ 内の点数閾値判定。
- **超高速ボクセルクラスタリング**: 巨大点群に対して全点DBSCANを行うとクラッシュするため、一度ボクセル代表点に縮約してDBSCANを行い、ノイズクラスタに属するボクセル内の全点をノイズと判定する適応型手法を採用。
- **大まかな孤立塊除去**: 1Dビットパック超高速ボクセルハッシュにより、巨大点群から数秒でメイン構造（建屋や地形）のみを残し、周囲の浮遊塊を一括隔離。
- **代表点リミッター**: ボクセル代表点の数が `REPRESENTATIVE_POINT_LIMIT`（20万点）を超える場合はボクセルを段階的に粗くして再量子化し（`_voxelize_with_limit`）、PCの完全フリーズを物理的に防ぎます。
- **全点消失ガード**: `filter_isolated_clusters` は条件を満たす塊が1つも無い場合、全点を保持して警告ログを残します（旧実装は空集合を返し、点群が丸ごとノイズ化していました）。

### (3) `gui/gl_viewer.py`（OpenGL 3Dビューア）
- **描画パフォーマンス**: GPU VBO（頂点バッファオブジェクト）に点群座標とカラー配列をキャッシュ。数千万点でも60FPS近い滑らかなオービット操作を実現。
- **カメラモデル**:
  - `pitch`（仰角）と `yaw`（方位角）による球面カメラ。
  - 数学座標系に準拠（上面視点 `pitch=89.9°, yaw=0.0°` で右がX、上がY、高さZ）。
- **左下XYZ軸ギズモ**:
  - 赤(X), 緑(Y), 青(Z) のバッジをクリックすることで、上面(Top)、正面(Front)、側面(Side)、等角(Iso)にワンクリック切り替え可能。
- **リアルタイムカラー変更**: 高さスライダー操作中はカラーマップLUTによる超高速プレビュー、マウスを離した時にVBOを確定更新する遅延ゼロ設計。
- **断面スライサー**: XYZ 3軸のクリッピングプレーンに加え、2点ピックによる任意の斜め壁に沿った断面スライスをサポート。
- **表示予算と断面ボックス内の全点表示**: 描画するのは `point_budget`（設定「表示点数」、既定 200 万点、0 で無制限）でストライド間引きした `display_points`（float32、VBO）。`_update_display_arrays()` は断面マスク → レイヤー/スキャン可視 → 予算の順に掛けるので、断面が有効な間はボックス内の点だけが予算の対象になる。さらに `section_full_detail`（断面パネル「🔬 ボックス内は間引かず全点表示」、設定 `sec_full_detail`）が ON なら予算を `SECTION_DETAIL_MAX_POINTS`（1,000 万点）まで引き上げ、超えた分だけ間引いて HUD に「上限で間引き」と出す（`section_box_count` / `section_detail_capped`）。スライダードラッグ中の高速プレビューは確定表示の部分集合で、離した時に全点から再構築する。
  - 描画の実測（2026-11-07、内蔵 Radeon 890M、1400×900、合成点群）: 200 万点 31ms（32fps）、1,000 万点 131ms（7.6fps）、2,000 万点 248ms、4,000 万点 456ms/フレーム。初回の VBO 転送は 4,000 万点で 11.6 秒。1 億点の全点表示は VBO 2.4GB・約 1.1 秒/フレームで実用外 → §9 の LOD が本命。

### (4) `gui/worker_thread.py`（非同期処理ワーカー）
- `FileLoadWorker`: ファイル読み込み＋座標系変換＋ダウンサンプリング。
- `FilterWorker`: バックグラウンドでの各種ノイズフィルタリング実行。
- `FileSaveWorker`: 元のメタデータを引き継いだロスレスアトミック保存。保存前に座標系変換を元に戻し、保存後に再適用する。
- `FullResolutionSaveWorker`: 全点転写保存（憲章⑨）。間引き読込のデータを元ファイルの全点へ写して保存する。1 スキャンずつ「読み直し → 転写 → `E57StreamWriter` で追記」を繰り返す。
- `FunctionWorker`: 任意の重い関数をバックグラウンド実行する汎用ワーカー。`MainWindow._run_in_background(title, sub, fn, *args, on_success=...)` から使う。
- すべてのワーカーは `progress_changed(int, str, str)` シグナルと `cancel()` を持ち、進捗ダイアログの「中止」から協調的に中断できます（読込はファイル境界/進捗コールバック、保存はファイル境界で反映）。
- `MainWindow._is_background_busy()` が True の間は新しい重い処理を開始しない（再入防止）。結果適用時は開始時点の点数と一致するか確認し、点群が入れ替わっていれば破棄する。

### (5) `gui/layer_history.py`（差分 Undo / Redo 履歴）
- レイヤー編集の Undo は **全点コピーではなく差分**で保持します。履歴が持つのは「直前の `_push_layer_undo_state()` 時点の `layer_ids` の写し（shadow）1 本」と、ステップごとの「変化した点のインデックス＋変化前の layer_id＋レイヤー定義辞書の複製」だけです。`layer_mask` は `layer_ids` とレイヤー定義（`is_noise`）から `_sync_mask_from_ids()` で再計算するため保存しません。
- `push()` の時点では何が変わるか分からないため、差分は次の `push()` / `undo()` のときに shadow と現在値を比較して確定します。変更が無かった push（0 点移動）はステップとして数えません。
- `MainWindow.layer_undo_stack` / `layer_redo_stack` は履歴オブジェクト内の list をそのまま公開しており、`len()` / `bool()` / `clear()` が従来どおり使えます（`clear()` は shadow もリセットします）。
- **注意**: Undo/Redo で `layer_ids` を戻した直後は `layer_mask` が古いままです。`_ensure_layer_sync()` は古い mask を正として ids を上書きするため、必ず `_apply_layer_state()`（mask 同期）を先に呼んでから選択解除などの他処理を行ってください（この順序を誤ると Redo が復元できません）。

---

## 6. これまでに解決された重要バグ・トラブルの教訓

| 事象 | 原因 | 解決策（再発防止策） |
| :--- | :--- | :--- |
| **クラスタフィルタでPCがフリーズ** | 2,000万点に対して直接DBSCANを実行し、メモリ不足とCPU100%飽和が発生した。 | `core/filters.py` に対称ボクセル代表点縮約＋代表点安全リミッター（最大50万点）を導入。必ず事前チェックを介す設計にした。 |
| **大まかな処理の解析が途中で止まる** | 巨大点群のボクセル化処理でメモリ確保が追いつかずUIスレッドが応答不能になった。 | 1Dビットパック超高速ボクセル化を実装し、さらに処理を完全に `QThread` に分離して中止ボタンを設置。 |
| **E57保存時に精度が劣化・ファイル破損** | 単精度Floatへの変換による丸め誤差、および保存中のプロセス中断によるファイル破壊。 | `core/e57_io.py` で倍精度/ScaledIntegerの完全復元を実装し、一時ファイル書き出し＋アトミック置換に変更。 |
| **上面ビューの方位が測量図面と合わない** | 数学座標系（X=横）と測量座標系（X=北/縦）の定義の不一致。 | インポート時の座標系セレクターに「🗺️ 測量座標系 → 数学系 (X/Y入替)」を追加し、取り込み段階で整合させた。 |
| **高さスライダー操作時に画面が重い** | スライダーを動かすたびに数百万点のVBO全体を再生成していた。 | ドラッグ中はカラーマップLUTの高速参照のみを行い、ドラッグ終了時にVBOを確定同期する2段階更新に変更。 |
| **新規PCで起動不能（2026-10-31 QA）** | `open3d` / `pyquaternion` を無条件 import しているのに requirements に無く、`起動.bat` の import 検査も見ていなかった。 | requirements-desktop.txt と 起動.bat の検査行に追加。依存追加時は3箇所同時更新をルール化（AGENTS.md）。 |
| **孤立クラスタ除去で点群が丸ごと消える** | 条件を満たす塊が無いとき空集合を返していた。 | 全点保持＋警告ログに変更（`allow_remove_all` で旧挙動）。`tests/test_filters_safety.py`。 |
| **測量座標系で取り込んだ点群の保存で丸め誤差・bounds 不整合** | X/Y入替がシグネチャで検知されず cartesianBounds が古いまま、ScaledInteger が他軸のスケールで再量子化されていた。 | 変換時に生整数・スキーマも軸ごとに付け替え、保存時は元の座標系へ逆変換してビット完全一致で書き戻す。`tests/test_coord_roundtrip.py`。 |
| **ソース修正が起動に反映されない** | `起動.bat` が dist の旧EXEを優先起動していた。 | 開発チェックアウト（main.py と core/ あり）ではソース起動に変更。 |
| **ビルドでマニュアルが消える・spec が上書きされる** | ビルドbatが `pyinstaller main.py` を実行し spec を無視していた。 | spec 経由のビルドに変更し、datas にマニュアル3点・README を追加、旧ZIPを dist/archive へ退避。 |

---

## 7. テスト & 品質管理（Quality Gate）

コードを変更した際は、**必ず以下の自動回帰テストを実行し、全ゲートが合格（PASS）することを確認**してください。

### テスト実行方法
```bash
python -X utf8 quality_gate.py           # 全5ゲート（約10秒。GUIウィンドウが一瞬開く）
python -X utf8 quality_gate.py --quick   # Gate 1/3/5 のみ（GUIを開かない。作業中の確認用）
python -X utf8 quality_gate.py --fix-eol # .bat の改行コードを CRLF に修復（既定では検査のみで書き換えない）
python -m unittest discover tests -v     # 単体テストのみ
```
（または `品質チェック実行.bat` をダブルクリック）

- `.git/hooks/pre-commit` に全ゲートが登録済みで、コミット時に自動実行されます（NG ならコミット不可）。hooks は全 worktree で共有されるため `--install-hook` を再実行する必要はありません。
- ゲートはリポジトリ内にファイルを残しません（一時ファイルは OS の一時ディレクトリ）。

### テスト内容（全5段階・単体テスト 149 件）
1. **Gate 1: 静的構文・クロスリファレンス整合性**（全 .py のコンパイル/AST、Web版 HTML⇄JS の DOM ID 照合、.bat の純粋ASCII/CRLF 検査）
2. **Gate 2: ビューア状態不変条件**（`validate_invariants`、点数の異なる点群の連続ロードで古いマスクが残らないこと）
3. **Gate 3: 境界値・カオス入力ファジング**（0点/1点/2点/分散ゼロ/平面点群に対する全フィルタ・全配色の耐性）
4. **Gate 4: 実機 OpenGL & GUI シナリオ**（MainWindow 起動、VBO 描画、レイヤー操作、Undo/Redo、保存、レシピ、座標系、断面、壁アライン。バックグラウンド処理は `_wait_for_background()` で完了待ち）
5. **Gate 5: unittest スイート**（`tests/` を discover。E57 ロスレス、フィルタ安全性、座標系往復、読込点数照合、レイヤー統合、UI/設定、全点転写保存、断面ボックス内の全点表示）

※ 配布EXE・ZIP の存在確認はゲートに含まれません。配布前に `EXE配布パッケージ作成.bat` を実行し、`dist/E57_PointCloud_Tool/` にマニュアル3点が同梱されていることを目視確認してください。

---

## 8. ビルド & 配布手順

本ツールはPython環境のない現場PCでも動作するよう、スタンドアロンEXEとして配布します。

### ビルドの実行
```bash
EXE配布パッケージ作成.bat
```
- このバッチファイルを実行すると、自動的に以下の処理が行われます：
  1. `python -m PyInstaller --noconfirm E57_PointCloud_Tool.spec` によるEXEビルド（spec の `datas` でマニュアル3点と README を同梱、`collect_all` で pye57 / open3d のネイティブDLLを収集）。
  2. 既存の `dist/E57_PointCloud_Tool_Dist.zip` を `dist/archive/E57_PointCloud_Tool_Dist_<日付>_<時刻>.zip` へ退避。
  3. 最新の配布用ZIPファイル（`dist/E57_PointCloud_Tool_Dist.zip`）の自動生成。
- **`pyinstaller main.py` を直接実行しないこと**（spec が再生成・上書きされ、同梱設定が失われます）。
- 配布ZIP内には `main.py` / `core/` が無いため、`起動.bat` を同梱しても EXE 起動になります。

---

## 9. 今後の開発課題・ロードマップ（Backlog）

今後のAIや開発者が機能追加を検討する際の推奨ロードマップです：

1. **超大規模点群（5,000万点〜1億点超）のLODストリーミング**
   - 八分木（Octree）構造による視点連動LOD（Level of Detail）読み込みの導入。
   - ※ 保存側は全点転写保存（憲章⑨）で 1 億点超に対応済み。残る課題は「判定も全点で行う」こと（面から数 cm 以内のヒゲは判定点のラベルを引き継ぐため、全点で SOR をかけ直す再フィルタ方式を全点転写に追加する案）。
2. **CAD連携・断面線エクスポート**
   - スライスした断面の輪郭線・エッジ点群を DXF または SVG 形式で出力する機能。
3. **3D寸法・角度計測ツール**
   - ビューア上での2点間距離測定、ポリライン面積測定、法線傾斜角度測定機能。
4. **他フォーマット（LAS/LAZ/PCD/PLY/XYZ）への対応**
   - E57以外の点群フォーマットの読み込みおよび相互変換機能。
5. **複数器械点（マルチスキャン）の個別整準・レジストレーション**
   - スキャンごとの位置合わせ微調整機能。

品質面の backlog（Undo メモリ上限、`_init_ui` 分割、Web版 SOR のアルゴリズム一致、ロガーのローテーション、pytest/CI 導入など）は `QA_REPORT_2026-10-31.md` §6 を参照。

---

## 10. 複数AI・複数開発者での並行開発ルール

2026年10月31日以降、本リポジトリは **元AI（`main` / 元フォルダ）と Claude（`claude/*` / worktree `../点群マイツール-claude`）が並行して開発**しています。
作業場所・担当ファイル・コミット規約・マージ手順・品質ゲートの運用は **`AGENTS.md`** に定めています。すべてのAI・開発者は作業前に必ず読んでください。要点:

- 他方のフォルダのファイルを直接編集しない。取り込みは git マージのみ。相手のフォルダで `stash` / `reset --hard` / `checkout --` を実行しない。
- 作業終了時は必ず全ファイルをコミットする（選択的コミット禁止）。
- `gui/main_window.py` / `gui/gl_viewer.py` は元AIのホットファイル。Claude の変更は最小・まとめて・main 取り込み直後に。
- マージ前は全5ゲート合格。`origin` への push はユーザー確認後。
- 資料（本書・AGENTS.md・QA報告書）と実装がずれたら、同じコミットで資料も直す。

---

## 11. 品質管理チェック（2026-10-31）の要約

総合品質管理チェックの詳細は `QA_REPORT_2026-10-31.md` を参照。修正済みの主な欠陥: 新規PCでの起動不能（依存欠落）、孤立クラスタ除去での全点消失、座標系変換点群の保存でのスケール取り違え・bounds 不整合、GUIスレッドでの重処理実行、中断ボタン未配線、`QThread.terminate()` 乱用、ビルドbatの spec 無視・マニュアル未同梱、品質ゲートの副作用、Web版 XSS / LAN 公開。

---
*初版作成日: 2026年10月31日*
*改訂: 2026年10月31日（Claude による品質管理チェック・並行開発体制の反映）*
*プロジェクト主管: 点群マイツール開発プロジェクト*
