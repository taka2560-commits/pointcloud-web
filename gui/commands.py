# -*- coding: utf-8 -*-
"""
コマンド表（新UIの操作の入口をまとめた単一の元データ）

コマンドランチャー（Ctrl+K）・右クリックのクイックパレット・操作説明書のリンク・ショートカット一覧は、
すべてこの表から作る。機能を足したらここに 1 行足せば、4 か所に反映される。

- 実行は既存のボタンのクリックや既存メソッドの呼び出しで行う（処理を二重に持たない）。
- available(win) が文字列を返したら「今は使えない理由」。ランチャーでは薄く表示し、実行すると理由を通知する。
- keywords はひらがな・別名（検索用）。検索は NFKC・小文字・カタカナ→ひらがなに揃えてから部分一致で見る。
"""

import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

GROUP_FILE = "開く・保存"
GROUP_AUTO = "① 自動ノイズ除去"
GROUP_EDIT = "② 手動修正"
GROUP_LAYER = "③ 確認・レイヤー"
GROUP_SECTION = "断面"
GROUP_VIEW = "表示・視点"
GROUP_HELP = "設定・ヘルプ"
GROUP_ORDER = [GROUP_FILE, GROUP_AUTO, GROUP_EDIT, GROUP_LAYER, GROUP_SECTION, GROUP_VIEW, GROUP_HELP]


@dataclass
class Command:
    id: str
    group: str
    title: str
    desc: str = ""
    shortcut: str = ""
    keywords: str = ""
    run: Optional[Callable[[Any], None]] = None
    available: Optional[Callable[[Any], Optional[str]]] = None

    def unavailable_reason(self, win) -> Optional[str]:
        """使えるなら None、使えないなら理由"""
        if self.available is None:
            return None
        try:
            return self.available(win) or None
        except Exception:
            return "今は使えません"

    def execute(self, win) -> bool:
        """実行する。使えないときは理由をトーストで知らせて False"""
        reason = self.unavailable_reason(win)
        if reason:
            if hasattr(win, "show_toast"):
                win.show_toast(f"「{self.title}」は今は使えません: {reason}")
            return False
        if self.run is not None:
            self.run(win)
        return True


def normalize(text: str) -> str:
    """検索用の正規化: NFKC、小文字、カタカナ→ひらがな、空白除去"""
    s = unicodedata.normalize("NFKC", text or "").lower()
    out = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:  # カタカナ → ひらがな
            ch = chr(code - 0x60)
        if not ch.isspace():
            out.append(ch)
    return "".join(out)


# グループ名の読み（ひらがなで「だんめん」と打てば、断面グループの機能がすべて出るようにする）
GROUP_READINGS: Dict[str, str] = {
    GROUP_FILE: "ひらく ほぞん ふぁいる",
    GROUP_AUTO: "じどう のいず じょきょ",
    GROUP_EDIT: "しゅどう しゅうせい へんしゅう",
    GROUP_LAYER: "かくにん れいやー",
    GROUP_SECTION: "だんめん",
    GROUP_VIEW: "ひょうじ してん",
    GROUP_HELP: "せってい へるぷ",
}


def search_text(cmd: Command) -> str:
    return normalize(" ".join((cmd.title, cmd.desc, cmd.keywords, cmd.group, GROUP_READINGS.get(cmd.group, ""), cmd.shortcut)))


def match_score(cmd: Command, query: str) -> int:
    """
    検索語との一致度（0 = 一致しない）。空白区切りの語をすべて含むこと。
    題名の先頭 > 題名の途中 > 説明・別名 > グループ名 の順に高くする（打った言葉そのものの機能が上に来る）。
    """
    tokens = [normalize(t) for t in (query or "").replace("　", " ").split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 1
    title = normalize(cmd.title)
    body = normalize(" ".join((cmd.desc, cmd.keywords, cmd.shortcut)))
    group = normalize(cmd.group + " " + GROUP_READINGS.get(cmd.group, ""))
    score = 0
    for t in tokens:
        if title.startswith(t):
            score += 30
        elif t in title:
            score += 20
        elif t in body:
            score += 12
        elif t in group:
            score += 4
        else:
            return 0
    return score


# ----------------------------------------------------------------------------- 使えるかどうかの判定
def _need_points(win) -> Optional[str]:
    v = getattr(win, "viewer", None)
    if v is None or v.points_raw is None or len(v.points_raw) == 0:
        return "先に点群を開いてください"
    return None


def _need_selection(win) -> Optional[str]:
    r = _need_points(win)
    if r:
        return r
    sel = win.viewer.selected_indices
    if sel is None or len(sel) == 0:
        return "先に点を選択してください（ブラシ・矩形・円形・多角形）"
    return None


def _need_section(win) -> Optional[str]:
    r = _need_points(win)
    if r:
        return r
    if not win.viewer.is_section_active():
        return "先に断面（X / Y / Z か斜め壁）を有効にしてください"
    return None


def _button_state(attr: str, base: Optional[Callable[[Any], Optional[str]]] = None):
    """既存ボタンの有無と有効状態を見る判定を作る"""

    def check(win) -> Optional[str]:
        if base is not None:
            r = base(win)
            if r:
                return r
        b = getattr(win, attr, None)
        if b is None:
            return "この画面にはありません"
        if not b.isEnabled():
            return "今の状態では押せません"
        return None

    return check


# ----------------------------------------------------------------------------- 実行の作り方
def _click(attr: str) -> Callable[[Any], None]:
    def run(win):
        b = getattr(win, attr, None)
        if b is not None:
            b.click()

    return run


def _toggle(attr: str) -> Callable[[Any], None]:
    def run(win):
        w = getattr(win, attr, None)
        if w is not None:
            w.toggle()

    return run


def _mode(mode: str) -> Callable[[Any], None]:
    return lambda win: win._set_interaction_mode(mode)


def _tab(key: str) -> Callable[[Any], None]:
    def run(win):
        scroll = getattr(win, "sidebar_scroll", None)
        if scroll is not None and scroll.isHidden():
            win._toggle_sidebar()
        win._set_sidebar_workflow_tab(key)

    return run


def _color(prefix: str) -> Callable[[Any], None]:
    def run(win):
        combo = getattr(win, "color_combo", None)
        if combo is None:
            return
        for i in range(combo.count()):
            if combo.itemText(i).strip().startswith(prefix):
                combo.setCurrentIndex(i)
                return

    return run


def _thickness(value: float) -> Callable[[Any], None]:
    return lambda win: win._apply_section_thickness_preset(value)


def build_commands(win) -> List[Command]:
    """コマンドの一覧（作業順）。win に無い部品のコマンドは available が理由を返す"""
    C = Command
    cmds: List[Command] = [
        # --- 開く・保存 ---
        C("file.open", GROUP_FILE, "E57 ファイルを開く", "点群ファイルを選んで読み込む", "Ctrl+O", "ひらく よみこみ ふぁいる open",
          lambda w: w._browse_input_file()),
        C("file.open_folder", GROUP_FILE, "フォルダ内の E57 を一括で開く", "フォルダの全 E57 を 1 つにまとめて読み込む", "Ctrl+Shift+O",
          "ふぉるだ いっかつ まとめて", lambda w: w._browse_input_folder()),
        C("file.save", GROUP_FILE, "E57 を保存", "ノイズを除いた点群を書き出す（間引き読込なら全点で保存を選べる）", "Ctrl+S",
          "ほぞん かきだし しゅつりょく save", lambda w: w._save_current_scans(), _need_points),
        C("file.screenshot", GROUP_FILE, "3D 画面を撮影", "今の 3D 画面を画像として保存する", "", "さつえい すくしょ がぞう screenshot",
          lambda w: w._on_capture_screenshot(), _need_points),
        # --- ① 自動 ---
        C("auto.panel", GROUP_AUTO, "自動ノイズ除去のパネルを開く", "左パネルを「①自動」に切り替える", "", "じどう ふぃるた ぱねる",
          _tab("filter")),
        C("auto.preset_rough", GROUP_AUTO, "プリセット: 大まか", "大きな浮遊ノイズだけを粗く取る設定にする", "", "おおまか あらどり ぷりせっと",
          _click("btn_preset_rough"), _button_state("btn_preset_rough")),
        C("auto.preset_std", GROUP_AUTO, "プリセット: 標準", "迷ったらこれ。標準的な強さの設定にする", "", "ひょうじゅん ぷりせっと",
          _click("btn_preset_std"), _button_state("btn_preset_std")),
        C("auto.preset_fine", GROUP_AUTO, "プリセット: 細かい", "細かいノイズまで取る設定にする（削りすぎに注意）", "", "こまかい せいみつ ぷりせっと",
          _click("btn_preset_fine"), _button_state("btn_preset_fine")),
        C("auto.run", GROUP_AUTO, "自動ノイズ処理を一括実行", "今の設定でノイズを検出し、ノイズレイヤーへ移す", "", "じっこう じどう のいず じょきょ run",
          _click("auto_exec_btn"), _button_state("auto_exec_btn", _need_points)),
        C("auto.undo", GROUP_AUTO, "自動処理を 1 つ戻す", "直前の自動処理を取り消す", "", "もどす とりけし",
          _click("auto_undo_btn"), _button_state("auto_undo_btn")),
        C("auto.keep_main", GROUP_AUTO, "メイン塊のみ残す", "いちばん大きな塊以外（離れた浮遊塊）をまとめてノイズへ", "", "めいん かたまり ふゆう",
          _click("btn_keep_main_cluster"), _button_state("btn_keep_main_cluster", _need_points)),
        C("auto.edge", GROUP_AUTO, "エッジ・ヒゲ除去", "面の縁から飛び出したヒゲ状のノイズを取る", "", "えっじ ひげ とびちり",
          _click("btn_quick_edge_clean"), _button_state("btn_quick_edge_clean", _need_points)),
        # --- ② 手動 ---
        C("mode.orbit", GROUP_EDIT, "回転モード", "左ドラッグで視点を回す（基本の状態）", "V", "かいてん してん orbit", _mode("orbit")),
        C("mode.pan", GROUP_EDIT, "パンモード", "左ドラッグで画面を平行移動する", "H", "ぱん へいこういどう pan", _mode("pan")),
        C("mode.brush", GROUP_EDIT, "ブラシ選択", "なぞった所の点を選ぶ", "B", "ぶらし なぞる brush", _mode("brush"), _need_points),
        C("mode.rect", GROUP_EDIT, "矩形選択", "ドラッグした四角の中の点を選ぶ", "R", "くけい しかく rect", _mode("rect"), _need_points),
        C("mode.circle", GROUP_EDIT, "円形選択", "ドラッグした円の中の点を選ぶ", "C", "えんけい まる circle", _mode("circle"), _need_points),
        C("mode.polygon", GROUP_EDIT, "多角形選択", "クリックで囲んだ形の中の点を選ぶ（Enter で確定）", "L", "たかっけい なげなわ らっそ polygon",
          _mode("polygon"), _need_points),
        C("mode.instant", GROUP_EDIT, "即削除の入/切", "囲んだ・なぞった瞬間にノイズへ移す", "", "そくさくじょ すぐけす",
          lambda w: w._toggle_instant_delete_mode(not w.viewer.instant_delete_mode), _need_points),
        C("mode.clean", GROUP_EDIT, "ノイズ専用モードの入/切", "ノイズ処理に必要な物だけを表示する", "N", "のいず せんよう",
          lambda w: w._toggle_clean_focus_mode(not w.viewer.clean_focus_mode)),
        C("edit.delete", GROUP_EDIT, "選択点をノイズへ移す", "選んだ点をノイズレイヤーへ移す（消えずに残るので戻せる）", "Del", "さくじょ けす のいず delete",
          lambda w: w._delete_selected_points(), _need_selection),
        C("edit.wand", GROUP_EDIT, "連結塊を選択", "選んだ点とつながっている塊全体を選ぶ", "G", "れんけつ かたまり まじっくわんど wand",
          lambda w: w._on_magic_wand_cluster_select(), _need_selection),
        C("edit.local_clean", GROUP_EDIT, "選択範囲内だけ精密除去", "選んだ範囲だけに細かいノイズ除去をかける", "", "せいみつ はんい きょくしょ",
          lambda w: w._on_local_fine_clean_selected(), _need_selection),
        C("edit.invert", GROUP_EDIT, "選択を反転", "選んでいる点と選んでいない点を入れ替える", "I", "はんてん invert",
          lambda w: w._invert_selection(), _need_points),
        C("edit.clear", GROUP_EDIT, "選択を解除", "", "Esc", "かいじょ くりあ", lambda w: w._clear_selection(), _need_points),
        C("edit.undo", GROUP_EDIT, "元に戻す", "直前の編集を取り消す", "Ctrl+Z", "もどす あんどぅ undo", lambda w: w._undo_action()),
        C("edit.redo", GROUP_EDIT, "やり直し", "取り消した編集をもう一度行う", "Ctrl+Y", "やりなおし りどぅ redo", lambda w: w._redo_action()),
        C("mode.measure", GROUP_EDIT, "計測（座標・距離）", "点をクリックして座標を読む。2 点目で距離と高低差", "M", "けいそく ざひょう きょり すんぽう measure",
          lambda w: w._toggle_measure_mode(not w.viewer.measure_mode), _need_points),
        # --- ③ 確認 ---
        C("layer.panel", GROUP_LAYER, "確認・レイヤーのパネルを開く", "左パネルを「③確認」に切り替える", "", "かくにん れいやー ぱねる",
          _tab("layer")),
        C("layer.toggle_noise", GROUP_LAYER, "ノイズレイヤーの表示 入/切", "消した点（赤）を表示して、消しすぎを確認する", "", "のいず ひょうじ あか",
          _toggle("chk_layer_noise"), _need_points),
        C("layer.toggle_normal", GROUP_LAYER, "有効点群の表示 入/切", "残す点を隠して、ノイズだけを見る", "", "ゆうこう ひょうじ",
          _toggle("chk_layer_normal"), _need_points),
        C("layer.restore_selected", GROUP_LAYER, "選択したノイズを復元", "選んだ範囲のノイズ点を通常レイヤーへ戻す", "", "ふくげん もどす",
          lambda w: w._restore_selected_noise(), _need_selection),
        C("layer.restore_all", GROUP_LAYER, "全ノイズを復元", "ノイズレイヤーの点をすべて通常レイヤーへ戻す", "", "ぜんぶ ふくげん りせっと",
          lambda w: w._restore_all_noise(), _need_points),
        C("layer.move", GROUP_LAYER, "選択点をレイヤーへ移動", "選んだ点を、レイヤー一覧で選んだレイヤーへ移す", "", "れいやー いどう",
          lambda w: w._on_move_selected_to_target_layer(), _need_selection),
        C("layer.ground", GROUP_LAYER, "地面を自動抽出", "地面の点を見つけて地面レイヤーへ分ける", "", "じめん ぐらうんど",
          _click("btn_auto_ground"), _button_state("btn_auto_ground", _need_points)),
        C("layer.plane", GROUP_LAYER, "主平面を自動検出", "壁や床などの大きな平面を見つけてレイヤーへ分ける", "", "へいめん かべ ゆか",
          _click("btn_auto_plane"), _button_state("btn_auto_plane", _need_points)),
        # --- 断面 ---
        C("section.panel", GROUP_SECTION, "断面のパネルを開く", "左パネルを「断面」に切り替える", "", "だんめん ぱねる", _tab("section")),
        C("section.z", GROUP_SECTION, "Z 断面（高さ）の入/切", "ある高さの範囲だけを表示する（床・天井の確認に）", "", "だんめん たかさ ぜっと すらいす",
          _toggle("chk_sec_z"), _need_points),
        C("section.x", GROUP_SECTION, "X 断面（左右）の入/切", "X 方向のある範囲だけを表示する", "", "だんめん さゆう えっくす", _toggle("chk_sec_x"), _need_points),
        C("section.y", GROUP_SECTION, "Y 断面（奥行）の入/切", "Y 方向のある範囲だけを表示する", "", "だんめん おくゆき わい", _toggle("chk_sec_y"), _need_points),
        C("section.thick_0.1", GROUP_SECTION, "断面の厚みを 0.1m にする", "", "", "あつみ うすい", _thickness(0.1), _need_points),
        C("section.thick_0.5", GROUP_SECTION, "断面の厚みを 0.5m にする", "", "", "あつみ", _thickness(0.5), _need_points),
        C("section.thick_1", GROUP_SECTION, "断面の厚みを 1m にする", "", "", "あつみ", _thickness(1.0), _need_points),
        C("section.thick_2", GROUP_SECTION, "断面の厚みを 2m にする", "", "", "あつみ", _thickness(2.0), _need_points),
        C("section.from_current", GROUP_SECTION, "今見ている所を断面の基点にする", "画面中央（選択があれば選択点）へ断面を移す", "", "きてん げんざいち",
          _click("btn_sec_from_current"), _button_state("btn_sec_from_current", _need_points)),
        C("section.wall_pick", GROUP_SECTION, "斜め壁を 2 点で指定", "壁の上の 2 点をクリックし、壁に沿った断面と視点を作る", "U", "ななめ かべ にてん あらいん",
          _click("nav_btn_wall_align"), _button_state("nav_btn_wall_align", _need_points)),
        C("section.crop", GROUP_SECTION, "断面ボックス外をノイズへ移す", "箱の外側の点をまとめてノイズレイヤーへ移す（周囲の切り落とし）", "", "くろっぷ そとがわ きりおとし",
          _click("btn_crop_section"), _button_state("btn_crop_section", _need_section)),
        C("section.full_detail", GROUP_SECTION, "ボックス内の全点表示 入/切", "断面の中は間引かず、読み込んだ点をすべて描く", "", "ぜんてん まびかない",
          _toggle("chk_sec_full_detail")),
        C("section.reload", GROUP_SECTION, "元ファイルから全点で読み直す", "断面の箱の中を、元ファイルの密度で重ね描きする", "", "よみなおし もとふぁいる みつど",
          _click("btn_sec_reload"), _button_state("btn_sec_reload", _need_section)),
        C("section.reload_clear", GROUP_SECTION, "読み直した重ね描きを消す", "", "", "かさねがき けす", _click("btn_sec_reload_clear"),
          _button_state("btn_sec_reload_clear")),
        C("section.lod", GROUP_SECTION, "高精細表示 (LOD) の入/切", "視点が止まると、見えている所を元ファイルの密度で描く", "", "こうせいさい えるおーでぃー みつど lod",
          _toggle("chk_lod")),
        C("section.guides", GROUP_SECTION, "断面の枠線 表示/非表示", "黄色（基点）と水色（厚み）の枠を出したり消したりする", "Shift+B", "わくせん がいど",
          _click("btn_toggle_sec_guides"), _button_state("btn_toggle_sec_guides")),
        C("section.reset", GROUP_SECTION, "断面をすべて解除", "X / Y / Z と斜め壁の断面を外して全体表示に戻す", "", "かいじょ りせっと ぜんたい",
          lambda w: w._reset_all_sections(), _need_points),
        # --- 表示 ---
        C("view.panel", GROUP_VIEW, "表示・カメラのパネルを開く", "左パネルを「表示」に切り替える", "", "ひょうじ かめら ぱねる", _tab("view")),
        C("view.fit", GROUP_VIEW, "全体フィット", "点群全体が画面に収まるようにする", "F", "ふぃっと ぜんたい fit",
          lambda w: w.viewer.fit_to_screen(), _need_points),
        C("view.top", GROUP_VIEW, "上面ビュー", "真上から見る", "", "じょうめん うえ とっぷ top", _click("btn_view_top")),
        C("view.front", GROUP_VIEW, "正面ビュー", "正面から見る", "", "しょうめん ふろんと front", _click("btn_view_front")),
        C("view.side", GROUP_VIEW, "側面ビュー", "横から見る", "", "そくめん よこ さいど side", _click("btn_view_side")),
        C("view.iso", GROUP_VIEW, "等角ビュー", "斜め上から見る（最初の向き）", "", "とうかく ななめ あいそ iso", _click("btn_view_iso")),
        C("view.projection", GROUP_VIEW, "透視 / 正射影の切替", "遠近感あり（透視）と、寸法が歪まない表示（正射影）を切り替える", "P", "とうし せいしゃえい おるそ",
          lambda w: w.viewer.toggle_projection()),
        C("view.color_rgb", GROUP_VIEW, "配色: RGB（写真の色）", "", "", "はいしょく いろ かめら rgb", _color("RGB"), _need_points),
        C("view.color_height", GROUP_VIEW, "配色: 高さ", "高さで色分けする（段差・たわみの確認に）", "", "はいしょく たかさ ひょうこう", _color("高さ (Z)"), _need_points),
        C("view.color_intensity", GROUP_VIEW, "配色: 反射強度", "反射の強さで色分けする（材質の違いの確認に）", "", "はいしょく はんしゃ きょうど", _color("反射強度"),
          _need_points),
        C("view.color_noise", GROUP_VIEW, "配色: ノイズ探索（弱い反射を光らせる）",
          "空中に浮いた点を探しやすくする。弱い反射ほど明るい水色〜白になり、強い反射（面）は暗い灰になる", "",
          "はいしょく のいず ふゆう くうちゅう たんさく はんしゃ きょうど よわい", _color("反射強度 (ノイズ探索 /"), _need_points),
        C("view.color_noise_soft", GROUP_VIEW, "配色: ノイズ探索・淡色（弱い 10% だけ着色）",
          "灰色の濃淡を基本に、いちばん弱い 10% の反射だけを水色にする。画面が落ち着くので、長い作業向け", "",
          "はいしょく のいず ふゆう くうちゅう たんさく はんしゃ きょうど たんしょく", _color("反射強度 (ノイズ探索・淡色"), _need_points),
        C("view.color_layer", GROUP_VIEW, "配色: レイヤー別", "レイヤーごとの色で表示する", "", "はいしょく れいやー", _color("レイヤー別"), _need_points),
        C("view.color_scan", GROUP_VIEW, "配色: スキャン別", "器械点（スキャン）ごとの色で表示する", "", "はいしょく すきゃん きかいてん", _color("スキャン別"),
          _need_points),
        C("view.sidebar", GROUP_VIEW, "左パネルの表示 / 格納", "左パネルを隠して 3D 画面を広く使う", "", "ぱねる かくのう ひろく",
          lambda w: w._toggle_sidebar()),
        # --- 設定・ヘルプ ---
        C("help.settings", GROUP_HELP, "設定を開く", "点サイズ・描画点数・操作感度・テーマなど", "", "せってい おぷしょん", _tab("settings")),
        C("help.log_folder", GROUP_HELP, "ログフォルダを開く", "不具合の調査に使う履歴ログの場所を開く", "", "ろぐ りれき",
          lambda w: __import__("core.logger", fromlist=["open_log_folder"]).open_log_folder()),
        C("help.workdata", GROUP_HELP, "作業用データの管理", "断面の読み直しと高精細表示のためのデータの大きさを見る・消す・置き場所を変える", "",
          "さぎょうよう でーた きゃっしゅ でぃすく あき ようりょう さくじょ おきばしょ cache",
          lambda w: __import__("gui.system_tools", fromlist=["open_workdata_dialog"]).open_workdata_dialog(w)),
        C("help.about", GROUP_HELP, "バージョン情報", "今動いている版（作った日時）とログの場所を見る", "", "ばーじょん はん びるど version",
          lambda w: __import__("gui.system_tools", fromlist=["show_about_dialog"]).show_about_dialog(w)),
    ]
    return cmds


def commands_by_id(cmds: List[Command]) -> Dict[str, Command]:
    return {c.id: c for c in cmds}
