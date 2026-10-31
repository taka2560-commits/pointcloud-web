# デザインスタイル改修 計画書

- 作成日: 2026年10月31日（同日に実装。§8 に実施結果）
- 前提資料: `docs/UI_UX_REVIEW_2026-10-31.md`（所見と優先順位）、`docs/ui_review/before/*.png`（改修前の証跡）
- §7 の決定事項はユーザーから「操作性重視で一任、元に戻せる安全策を必須」との回答を得て、以下のとおり決定した:
  テーマはダークのみ（ライトは backlog）／ 絵文字は新UIで除去し、部品アイコンのみ SVG（機能アイコンは backlog）／
  情報設計は §4.2 の「作業順サイドバー＋メニュー/ステータスバー＋空状態」まで実施し、操作モードは上部バーへ一本化（縦ツールバー化は見送り）／
  用語は上部バーの表記ゆれのみ統一（円形・多角形・ノイズ専用）／ 最小解像度 1366×768 @125%（実効約1090px）で切れないことを Gate4 で検査／
  元AIとの分担は「後付けアダプタ方式」により main_window.py の大改修を回避して解決

---

## 1. 目的と原則

### 目的
1. 視覚言語を 1 系統に統一し、「何が主で何が従か」が一目で分かるようにする
2. 読みやすさ（文字サイズ・コントラスト・クリック目標）を現場 PC の拡大率で保証する
3. 以後の機能追加が「トークンと役割を選ぶだけ」で済む仕組み（デザインシステム）にする

### 原則
- **点群が主役**: UI は低彩度・低コントラストの面で構成し、彩度の高い色は「アクセント（操作可能）」「危険」「状態」にだけ使う
- **色は役割で決める**: 機能ごとの色分け（紫=ワンド、茶=地面…）は廃止。色は primary / danger / warning / success / info の意味だけを持つ
- **絵文字ゼロ・アイコン＋日本語ラベル**: アイコンは単色 SVG（Lucide 系）、ラベルは必ず日本語を併記
- **トークンを唯一の情報源にする**: 色・サイズ・角丸・間隔は `gui/design_tokens.py` にのみ定義し、QSS も GL の HUD もそこから生成する
- **セレクタ無しのインライン QSS を禁止**: `setStyleSheet("background: ...")` は子孫に伝播して事故を起こす（レビュー V4）。スタイルは `role` / `objectName` プロパティ＋グローバル QSS で表現する

---

## 2. デザイントークン案（slate / sky 系に統一。Web 版 `style.css` と同系統）

### 2.1 色

| トークン | 値 | 用途 |
| :--- | :--- | :--- |
| `bg.canvas` | #0f172a | ウィンドウ最背面、スクロール溝 |
| `bg.surface` | #172033 | サイドバー、パネル |
| `bg.elevated` | #1e293b | カード、メニュー、ツールチップ、入力欄の枠内 |
| `bg.input` | #0b1220 | 入力欄・スピンボックスの地 |
| `border.subtle` | #263247 | パネル枠、区切り |
| `border.strong` | #3b4a63 | 入力欄枠、ボタン枠 |
| `text.primary` | #e2e8f0 | 本文 |
| `text.secondary` | #a3b1c6 | 補足、ラベル（#94a3b8 より明るくして 11px でも AA） |
| `text.disabled` | #6b7a90 | 無効状態（背景 #172033 に対し 3.3:1） |
| `text.on-accent` | #ffffff | アクセント上の文字 |
| `accent.base` | #0ea5e9 | 主アクセント（selected / checked / primary hover） |
| `accent.strong` | #0284c7 | primary ボタン地（白文字 4.6:1） |
| `accent.soft` | #38bdf8 | 見出し、フォーカス枠、リンク |
| `accent.tint` | rgba(14,165,233,0.14) | 選択行、checked の薄い背景 |
| `danger.base` / `danger.strong` | #ef4444 / #b91c1c | 削除、即削除 ON、エラー |
| `warning.base` | #f59e0b | 注意、ロック中、断面 HUD |
| `success.base` | #22c55e | 完了トースト、実行ボタン（白文字は 13px では不足するため黒文字 #052e16 を使う） |
| `info.base` | #38bdf8 | 情報トースト |
| `layer.normal` | #38bdf8 | 通常点（配色「レイヤー」時） |
| `layer.noise` | #f472b6（マゼンタ） | ノイズ点。高さ配色の赤と区別する |
| `layer.ground` / `layer.plane` | #a16207 / #16a34a | 自動分類レイヤー |
| `gl.background.dark` / `.black` / `.light` | #14141a / #000000 / #f5f5f7 | ビューア背景（ライト時は HUD/グリッドも `gl.*.light` に切替） |
| `gl.grid` | rgba(148,163,184,0.25) | グリッド |
| `gl.selection` | #0ea5e9 | 選択枠・選択点（= accent.base） |

### 2.2 タイポグラフィ

| トークン | 値 | 用途 |
| :--- | :--- | :--- |
| `font.ui` | "Segoe UI", "Yu Gothic UI", "Meiryo UI", sans-serif | UI 全般（ダイアログ・HUD も同じ） |
| `font.mono` | "Cascadia Mono", "Consolas", monospace | ログ、数値表示 |
| `size.xs` | 12px | 注記（最小。10〜11px は廃止） |
| `size.sm` | 13px | 本文・ボタン |
| `size.md` | 15px | グループ見出し、主ボタン |
| `size.lg` | 18px | 数値バッジ、進捗 % |
| `weight` | 400 / 600 | normal / 強調（bold 700 は見出しのみ） |

### 2.3 形状・間隔・サイズ

| トークン | 値 |
| :--- | :--- |
| `radius.sm` / `md` / `lg` | 4 / 6 / 10 px（3 種のみ。ピルは `lg`） |
| `space.1〜6` | 4 / 8 / 12 / 16 / 24 / 32 px（4 の倍数） |
| `control.h.sm` / `md` / `lg` | 28 / 32 / 36 px（ボタン・入力欄の高さ） |
| `icon.sm` / `md` | 16 / 20 px |
| `sidebar.min` / `default` / `max` | 320 / 380 / 640 px |
| `touch.min` | 24×24 px（色見本・ロック・開閉ボタンの下限） |

### 2.4 状態（全部品共通）

hover: 背景を `accent.tint` 重ね / pressed: 明度 −8% / checked: `accent.strong` 地＋白文字 / focus: 2px `accent.soft` 外枠 / disabled: `text.disabled`・枠 `border.subtle`・背景は変えない

---

## 3. 部品仕様（QSS の `role` プロパティで表現）

| role | 見た目 | 使いどころ | 現状からの置換 |
| :--- | :--- | :--- | :--- |
| `primary` | `accent.strong` 地、白文字、h=36 | 画面に1つ。「保存」「自動ノイズ処理を実行」 | #execute_button, role=primary, #0369a1 系 |
| `secondary` | `bg.elevated` 地、`text.primary`、枠 `border.strong` | 通常の操作 | 既定 QPushButton(#2e6296)、色付きボタン全般 |
| `outline` | 透明地、枠 `border.strong` | 補助操作 | role=outline |
| `ghost` | 透明地、枠なし | ツールバー、アイコンボタン | 上部バー・下部バーの各ボタン |
| `danger` | `danger.strong` 地、白文字 | 削除・即削除 | role=danger, #9a3412 系 |
| `segment` | `secondary` と同じ。checked で `accent.strong` | 排他切替（モード、視点、タブ） | role=segment, 状態をインライン QSS で切替している箇所 |
| 入力系 | `bg.input` 地、枠 `border.strong`、focus で `accent.soft` | QLineEdit / QSpinBox / QComboBox（独自矢印） | 未スタイルの QComboBox を含めて統一 |
| CollapsibleGroupBox | `bg.surface` 地、見出し `size.md` + アイコン、開閉はシェブロン | 全グループ | 絵文字見出しを置換 |
| トースト | `info / success / warning / error` の4種、左にアイコン | 一時通知 | 成功色1種のみを拡張 |
| ステータスバー（新設） | 左: 状態文、右: 点数 / 座標系 / モード | 常設 | サイドバー最下部の status_label を移設 |
| HUD（GL） | ピル `radius.lg`、`bg.elevated` 0.85、文字 `size.sm` | モード・断面・計測 | gl_viewer.py のハードコードを `design_tokens` 参照に |

---

## 4. 情報設計の整理案（段階的に実施。§7-3 の判断に依存）

### 4.1 低リスク（見た目のみ）
- 絵文字 → アイコン（見出し・ボタン・タブ）。ラベル文言は維持。
- 上部バーの重複は残したまま、`ghost` 役割で軽く見せる。

### 4.2 中リスク（配置の変更。main_window.py の大改修）
- **操作モードは縦ツールバー**（ビューア左端、アイコン 20px、ツールチップにキー）に集約し、サイドバーの `side_btn_*`・形状コンボ・「手動選択モード」を廃止。
- **サイドバーを作業ステップ順に**: ① ファイル（開く・座標系） → ② 自動除去（レシピ・パラメータ・実行） → ③ 手動修正（選択ツール設定・ブラシ径） → ④ 確認（レイヤー・表示切替・断面） → ⑤ 保存。「表示・カメラ」「設定」はツールバー／メニューへ。
- **メニューバー新設**: ファイル（開く Ctrl+O、保存 Ctrl+S、最近使ったファイル）／編集（Undo/Redo）／表示／ヘルプ（マニュアル、ショートカット一覧 `?`）。
- **空状態カード**（ビューア中央）: ドロップ領域＋「ファイルを開く」＋最近使ったファイル。
- **ステータスバー新設**、サイドバー最下部の進捗・状態表示を移設。ログは折りたたみパネルのまま。

### 4.3 用語の統一（§7-4）
| 現状の揺れ | 統一案 |
| :--- | :--- |
| 削除 / 消去 / 除去 / ノイズへ移動 | **ノイズへ移動**（ボタン名は「ノイズへ」） |
| 有効点群 / 通常点群 / 通常レイヤー | **通常レイヤー** |
| 円(C) / 円形(C)、多角 / 多角形、専用(N) / ノイズ専用(N) | 円形 / 多角形 / ノイズ専用 |
| 「爆速化」「完全網羅」「絶対に削らない」等 | 事実ベースの表現に（例: 「高速化」「主要な設定を網羅」「構造物を保護」） |
| 進捗ダイアログ "LOADING" | 読み込み中 / 保存中 / 解析中 |

---

## 5. 実装の進め方（フェーズ）

| フェーズ | 内容 | 主な変更ファイル | ホットファイル影響 |
| :--- | :--- | :--- | :--- |
| P0 準備（本書） | 所見・トークン案・検証ツール | docs/, tools/ | なし |
| P1 トークン基盤 | `gui/design_tokens.py` 新設（dataclass でトークン定義、`build_qss(tokens)` で QSS 生成）。`gui/styles.py` は生成結果を返すだけにする。コントラスト検査を `tests/test_design_tokens.py` で自動化（全ペア AA 以上） | gui/design_tokens.py（新）, gui/styles.py, tests/ | なし |
| P2 インライン QSS 撤去 | main_window.py の約150箇所の `setStyleSheet` を `setProperty("role", ...)` ＋ `objectName` に置換。状態切替はインライン差し替えではなく `setProperty`＋`style().unpolish/polish` | gui/main_window.py, gui/progress_dialog.py | **大**（機械的だが広範囲。元AIと日程調整し、main 取り込み直後に 1 コミットで） |
| P3 アイコン化・文字サイズ | `assets/icons/*.svg` を追加し `QIcon` で設定（PyInstaller spec の datas に追加）。10〜11px を 12px 以上へ。クリック目標 24px 以上 | gui/main_window.py, gui/gl_viewer.py（HUD）, E57_PointCloud_Tool.spec | 中 |
| P4 情報設計 | §4.2（縦ツールバー・ステップ順サイドバー・メニューバー・ステータスバー・空状態） | gui/main_window.py（分割を併せて実施: `gui/panels/*.py` へ切り出し） | **特大**（元AIの開発と競合しやすい。単独フェーズとして合意の上で実施） |
| P5 安全性・応答性 | 変更フラグ、読込トリガー、破壊操作のプレビュー、ツールバー溢れメニュー、下部バー折り返し | gui/main_window.py, gui/gl_viewer.py | 中 |
| P6 ライト／ハイコントラスト（任意） | トークンセットを追加し設定から切替。GL 背景・HUD・グリッドも追従 | gui/design_tokens.py, gui/gl_viewer.py | 小 |

各フェーズの完了条件: 全5ゲート合格、`tools/capture_ui_screens.py` で撮り直したスクリーンショットを `docs/ui_review/` に更新、レビュー文書の該当項目を「対応済み」に更新。

---

## 6. 検証方法

```powershell
# 現状・改修後のスクリーンショット（実画面を撮影。ウィンドウを前面にしておく）
python -X utf8 tools/capture_ui_screens.py docs/ui_review 1600 900
# コントラスト比（P1 以降は tests/test_design_tokens.py が全ペアを検査）
python -X utf8 tools/check_contrast.py
# 品質ゲート（Gate4 に左パネルの見切れ検査あり）
python -X utf8 quality_gate.py
```

追加予定の自動検査（Gate4 へ）:
- 1100×700 でも上部ツールバー・下部アクションバーのボタンが切れない（各ボタンの `visibleRegion` が `rect` と一致）
- QSS に `font-size: 1[01]px` / `9px` が含まれない
- `setStyleSheet(` の呼び出しが許可リスト以外に存在しない（インライン QSS の再発防止）
- ボタン・見出しテキストに絵文字（U+1F300〜U+1FAFF、U+2600〜U+27BF）が含まれない

---

## 7. 決定をお願いしたい事項（回答後に P1 着手）

1. テーマ: ダークのみ / ダーク＋ライト / ダーク＋ライト＋ハイコントラスト
2. アイコン: 絵文字を全廃して SVG アイコンへ置換してよいか（Lucide（ISC ライセンス）を想定）
3. 情報設計の規模: §4.1 のみ / §4.2 まで（main_window.py の分割を伴う）
4. 用語統一案（§4.3）の採否
5. 最小対応解像度: 1366×768 @125% を保証対象にするか
6. 元AIとの分担: P2（インライン QSS 撤去）と P4（情報設計）は main_window.py の大改修になるため、着手期間中は元AIに当該ファイルの編集を控えてもらう必要がある。その調整方法（期間の指定、または元AIに実施してもらう）

---

## 8. 実施結果（2026-10-31）

| フェーズ | 状況 | 実装 |
| :--- | :---: | :--- |
| P1 トークン基盤 | ✅ | `gui/design_tokens.py`（Tokens / build_modern_qss / contrast_ratio）、`gui/styles.py: get_theme_qss()`、`tests/test_design_tokens.py` |
| P2 インライン QSS 撤去 | ✅（方式変更） | 150 箇所を書き換える代わりに、新UIでは `gui/modern_ui.py` が起動後にインライン QSS を外し以後の差し替えを無効化。状態は `role` ＋ :checked。従来UIはそのまま |
| P3 アイコン化・文字サイズ | ✅（一部） | 絵文字は新UIで全除去（ボタン 0 件を Gate4 で検査）。SVG は矢印・チェックのみ（`assets/icons/`）。最小 12px |
| P4 情報設計 | ✅ | 作業順サイドバー、保存グループ、タブ名（全て / ①自動 / ②手動 / ③確認 / 断面 / 表示 / 設定）、メニューバー、ステータスバー、空状態カード、重複モードボタンの非表示 |
| P5 安全性・応答性 | ✅ | 未保存フラグと確認、読込トリガーの明示化、破壊操作の件数確認、Shift+B、折り返しバー |
| P6 ライト／ハイコントラスト | — | backlog |

安全策（元に戻す手段）:
1. **UIモード切替**: 設定 → システム → UIスタイル「従来UI」／ ヘルプ → 従来UIに戻す／ 起動引数 `--classic-ui`／ 環境変数 `E57_UI_MODE=classic`。従来UIのコードと QSS は無変更で残している。
2. **Git タグ** `before-ui-redesign-2026-10-31`（改修直前の main）。`git checkout before-ui-redesign-2026-10-31` で改修前のコード一式に戻れる。
3. **マージコミット** は `--no-ff`。`git revert -m 1 <merge>` で一括取り消し可能。

残 backlog: 機能アイコン（SVG）の導入、ライトテーマ、HUD/グリッド色のトークン化、移動先レイヤーのポップアップ化、ウィンドウ位置・座標系・フィルタ値の保存、ノイズ色の凡例。

## 資料編 A: 現状トークン棚卸し（抜粋）

- 背景 15 色（ニュートラル系 #1e1e24 / #252530 / #16161c / #17171c / #141419 / #181820 / #333340、slate 系 #0b1120 / #0f172a / #172033 / #1e293b、他）
- 枠線 9 色、文字 13 色、アクセント 11 系統（#38bdf8 / #0284c7 / #4da6ff / #0078d4 / #2e6296 / #2563eb / #0369a1 / #22bdd6 / #60a5fa / #1e40af / #00d7ff）
- 成功系 4 系統、危険系 4 系統、警告系 3 系統、紫系 4 系統（`#preview_button` は未使用のデッドコード）
- 角丸 3 / 4 / 5 / 6 / 8 / 10 / 12 / 13 / 14、ボタン padding 約 19 種、spacing 0〜10 で 3 と 5 が混在
- フォント: Segoe UI / Meiryo UI（QMainWindow のみ指定）、サイズ px 7 種（9〜18）＋ GL pt 4 種、絵文字 57 種・約 350 箇所、QIcon 0 件
- インライン `setStyleSheet`: main_window.py 150 箇所、progress_dialog.py 7 箇所。最大ブロックは中止ボタン・ログフォルダボタン・折りたたみトグル
- 未スタイル: QComboBox（14）、QSlider（17）、QToolTip（101 件の設定あり）、QMessageBox、QSplitter ハンドル、横スクロールバー
- GL: 背景 #14141a / 黒 / 白の 3 択だが HUD・グリッドは追従しない。選択色 #00d7ff はどの QSS 色とも不一致

## 資料編 B: 情報設計の棚卸し（抜粋）

- サイドバー 9 グループ（ファイル、タブバー、3D表示、断面、レイヤー、選択ツール、フィルタ、設定、ログ）。「全て」タブで約 145 部品＋上部 14＋下部 7＋ギズモ 4 ≒ 170、サブタブ・断面詳細を含めると約 210
- 重複操作: 操作モード 5 経路、即削除 3、ノイズ専用 4、計測 3、2点壁ピック 3 ボタン＋キー、フィット／投影／スクショ各 2、削除 5、Undo 4、連結塊 3、ログフォルダ 3、点数表示 2、プリセット 3 ボタン＝レシピの先頭 3 項目
- ショートカット 23 件（Esc, Del, BS, Return, Enter, Ctrl+Z, Ctrl+Y, Ctrl+Shift+Z, I, F, P, M, [, ], N, V, H, B, R, C, L, G, U）＋ WASD/QE/Space。B の二重割当、V の表記と動作の不一致、Ctrl+O/S なし
- 通知: status_label 75 箇所、トースト 67、ログ 39、QMessageBox 40（information 19 / warning 12 / critical 7 / question 2）
- ツールチップ: 101 件（約 55%）。未設定: 実行ボタン、断面外クロップ、削除、Undo、視点ボタン、下部バー主要ボタン
- 設定の保存: 17 キー。未保存: ウィンドウ位置、最終フォルダ、座標系、フィルタ値、レシピ、配色、ブラシ径
