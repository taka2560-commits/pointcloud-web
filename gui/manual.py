# -*- coding: utf-8 -*-
"""
操作説明書（新UI）: 作業順に並べた使い方を、設定タブの「説明書」と別ウィンドウ（F1）で読めるようにする

- 本文はこのファイルの TOPICS に持つ（目的 → 手順 → コツ）。機能名とショートカットはコマンド表（gui/commands.py）から
  埋め込むので、名前が食い違わない。本文中の記法:
    [[cmd:ID]]        … その機能へのリンク（押すと実行）。題名とショートカットを自動で入れる
    [[cmd:ID|文字]]   … 表示する文字を指定したリンク
    [[key:Ctrl+Z]]    … キーの表示
    [[topic:ID|文字]] … 別の項目へのリンク
- 「ショートカット一覧」の項目はコマンド表から自動で作る。
- 色はテーマのトークンから作る（テーマを切り替えたら refresh_theme()）。
"""

import html
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui.commands import GROUP_ORDER, Command, normalize
from gui.design_tokens import get_tokens


@dataclass
class Topic:
    id: str
    title: str
    summary: str
    body: str


TOPICS: List[Topic] = [
    Topic(
        "start",
        "はじめに（画面の見方と基本の流れ）",
        "何をするツールか、画面のどこに何があるか、作業の順番",
        """
<p>E57 点群から浮遊ノイズや不要な点を取り除き、きれいな E57 として書き出すツールです。</p>
<h3>画面の見方</h3>
<ul>
<li><b>左パネル</b>: 作業順のタブ（①自動 → ②手動 → ③確認、断面・表示・設定）。上に入力ファイル、下に保存とログがあります。</li>
<li><b>上のバー</b>: マウスの役割を切り替えるツール（回転・パン・ブラシ・矩形・円形・多角形・計測・2点壁）と、フィット・投影・撮影。</li>
<li><b>下のバー</b>: 点を選ぶと、その場でできる操作（ノイズ削除・連結塊・レイヤー移動など）が押せるようになります。</li>
<li><b>右下</b>: 残す点（有効）とノイズの点数。</li>
</ul>
<h3>基本の流れ</h3>
<ol>
<li>[[cmd:file.open]] で点群を開く。</li>
<li>[[cmd:auto.run]] で大きなノイズをまとめて取る。</li>
<li>残ったノイズを選択ツールで囲み、[[key:Del]] でノイズへ移す。</li>
<li>ノイズレイヤーを表示して消しすぎを確かめ、必要なら復元する。</li>
<li>[[cmd:file.save]] で書き出す。</li>
</ol>
<p class="tip">消した点は「ノイズレイヤー」へ移るだけで、保存するまで元のデータは変わりません。[[key:Ctrl+Z]] でいつでも戻せます。</p>
<p>機能の場所が分からないときは [[key:Ctrl+K]] を押して名前を打ってください（[[topic:quick|素早く操作する]]）。</p>
""",
    ),
    Topic(
        "quick",
        "素早く操作する（検索・右クリック・小窓）",
        "Ctrl+K の検索窓、右クリックのパレット、パネルの小窓、操作ガイド",
        """
<h3>機能を検索して実行する（Ctrl+K）</h3>
<ol>
<li>[[key:Ctrl+K]] を押す（上のバーの「機能を検索」でも開きます）。</li>
<li>やりたいことを打つ。ひらがなで構いません（例: だんめん、けいそく、ほぞん）。</li>
<li>[[key:↑]] [[key:↓]] で選び、[[key:Enter]] で実行。[[key:Esc]] で閉じます。</li>
</ol>
<p>何も打たなければ、最近使った機能と、作業順に並んだ全機能が出ます。薄く表示される機能は今は使えないもので、理由が横に出ます。</p>
<h3>右クリックのパレット</h3>
<p>3D 画面で右クリックすると、カーソルの横に操作のタイルが出ます。</p>
<ul>
<li><b>点を選んでいるとき</b>: ノイズへ・連結塊を選択・範囲内を精密除去・レイヤーへ移動・ノイズを復元・反転・解除。</li>
<li><b>選んでいないとき</b>: ツールの切替（回転・ブラシ・矩形など）と視点（上面・正面・側面・フィット）。</li>
</ul>
<p>以前の右クリックメニューは、パレットの下の「その他のメニュー」から開けます。</p>
<h3>パネルを小窓にする</h3>
<ul>
<li>左パネルの各グループの見出しにある「別窓」を押すと、そのグループが小窓として浮きます。閉じると左パネルへ戻ります。</li>
<li>[[cmd:view.sidebar]] で左パネルを格納すると、3D 画面の左端にボタンの列が出ます。押すと必要なグループだけを小窓で開けます。広い画面で作業したいときに使います。</li>
</ul>
<h3>操作ガイド</h3>
<p>ブラシや計測などのツールを選ぶと、3D 画面の右上に「次に何をすればよいか」が出ます。慣れたら、この説明書の下にあるチェックで消せます。</p>
<h3>ヒント</h3>
<ul>
<li><b>状況に応じたヒント</b>: 画面の状態を見て、次にやるとよいことを 3D 画面の右上にカードで出します（例: 自動処理の後に「消した点を赤で表示して確認」）。カードのボタンを押すとその場で実行します。「閉じる」はこの起動中だけ、「今後表示しない」は以後ずっと出しません。設定の「説明書」タブの「閉じたヒントを戻す」で元に戻せます。</li>
<li><b>ヒント集</b>: 知っていると便利な操作をまとめています。[[cmd:help.tips]] で開き、「試す」でその機能が動きます。</li>
</ul>
""",
    ),
    Topic(
        "legend",
        "ボタンと表示の見方",
        "ボタンの形の意味、処理中・完了・押せないときの表示、通知と「元に戻す」",
        """
<p>ボタンは「役割」で形を、「状態」で表示を変えています。色だけに頼らず、文字と印（アイコン）でも分かるようにしています。</p>
<h3>ボタンの形（役割）</h3>
[[roles]]
<p class="tip">迷ったら、その場でいちばん目立つ塗りつぶしのボタンが「次に押すもの」です。赤いボタンは点を消す操作なので、押す前に範囲を確かめてください（消した点はノイズレイヤーに残り、[[cmd:edit.undo]] で戻せます）。</p>
<h3>状態の表示</h3>
<table>
<tr><td><b>処理中</b></td><td>ボタンの文字が「保存中…」「実行中…」に変わり、回る印が出ます。終わるまで押せません。</td></tr>
<tr><td><b>完了</b></td><td>保存が終わると、ボタンがしばらく「保存済み」とチェックの印になります。</td></tr>
<tr><td><b>押せない</b></td><td>薄い表示のボタンは今は使えません。マウスを置くと「点を選ぶと使えます」のように理由が出ます。</td></tr>
<tr><td><b>選択中</b></td><td>選んでいる道具やタブは塗りつぶしと太字になります。オン/オフのボタンは、オンでチェック、オフで丸の印が付きます。</td></tr>
<tr><td><b>キーボードの現在位置</b></td><td>Tab キーで移動すると、今いるボタンに太い枠が出ます。Space か Enter で押せます。</td></tr>
</table>
<h3>通知と「元に戻す」</h3>
<p>操作の結果は 3D 画面の右下に出ます。印で種類が分かります（チェック = できた、三角 = 注意、丸に ! = 失敗、丸に i = お知らせ）。</p>
<p>点を動かした直後の通知には<b>「元に戻す」</b>が付きます。押すとその操作だけを取り消します（[[key:Ctrl+Z]] と同じ）。数秒で消えますが、消えたあとも [[cmd:edit.undo]] で戻せます。</p>
<h3>未保存の変更</h3>
<p>保存していない編集があると、画面の右下（状態の行）に<b>「未保存の変更あり」</b>が出て、ウィンドウの題名に「*」が付きます。押すと保存します（[[cmd:file.save]]）。</p>
<h3>入力の誤り</h3>
<p>入力 E57 や出力 E57 の欄に、無い場所を書いて確定すると、欄が赤い枠になり、すぐ下に理由が出ます。直すと消えます。</p>
<h3>保存ボタンの右の矢印</h3>
<p>「保存」の右の小さな矢印を押すと、保存に関連する操作（保存先を選んで保存・作業の状態を保存）を選べます。</p>
""",
    ),
    Topic(
        "open",
        "点群を開く",
        "ファイル・フォルダの開き方、座標系、最近使ったファイル",
        """
<ol>
<li>[[cmd:file.open]] を押してファイルを選ぶ。ウィンドウへファイルをドラッグ＆ドロップしても開けます。</li>
<li>複数の E57 をまとめて扱うときは [[cmd:file.open_folder]]。フォルダ内の全ファイルを 1 つの点群として読み込みます。</li>
</ol>
<ul>
<li><b>座標系</b>: 測量座標（X と Y が入れ替わっている）などのファイルは、開く前に左パネル上部の「座標系」で選びます。画面の向きが変わるだけで、保存するときは元の座標のまま書き出します。</li>
<li><b>最近使ったファイル</b>: メニューの「ファイル → 最近使ったファイル」、または起動直後の画面中央から開けます。</li>
<li><b>大きなファイル</b>: 点数が読込上限を超えると、自動で間引いて読み込みます（[[topic:big|1 億点を超えるファイル]]）。</li>
</ul>
""",
    ),
    Topic(
        "nav",
        "視点の動かし方",
        "回転・移動・拡大、視点の切替、透視と正射影",
        """
<table>
<tr><td><b>回転</b></td><td>左ドラッグ（回転モードのとき）</td></tr>
<tr><td><b>平行移動</b></td><td>右ドラッグ、または中ボタンのドラッグ。[[key:Space]] を押している間は左ドラッグでも移動</td></tr>
<tr><td><b>拡大・縮小</b></td><td>マウスホイール</td></tr>
<tr><td><b>歩くように移動</b></td><td>[[key:W]] [[key:A]] [[key:S]] [[key:D]]（前後左右）、[[key:Q]] [[key:E]]（上下）</td></tr>
</table>
<ul>
<li>[[cmd:view.fit]]: 見失ったらこれ。全体が画面に収まります。</li>
<li>[[cmd:view.top]] / [[cmd:view.front]] / [[cmd:view.side]] / [[cmd:view.iso]]: 決まった向きへ切り替えます。左下の X・Y・Z の丸印を押しても切り替わります。</li>
<li>[[cmd:view.projection]]: 寸法や通りを見るときは正射影、立体感が欲しいときは透視。</li>
<li>選択ツールを使っている最中に視点を回したいときは、[[key:Alt]] を押しながら左ドラッグします。</li>
</ul>
""",
    ),
    Topic(
        "auto",
        "① 自動ノイズ除去",
        "プリセットを選んで一括実行。結果の確認と取り消し",
        """
<ol>
<li>左パネルの「①自動」を開く（[[cmd:auto.panel]]）。</li>
<li>強さを選ぶ: [[cmd:auto.preset_rough]] → [[cmd:auto.preset_std]] → [[cmd:auto.preset_fine]] の順に強くなります。現場に合う「現場レシピ」があればそれを選びます。</li>
<li>[[cmd:auto.run]] を押す。見つかったノイズはノイズレイヤーへ移ります。</li>
<li>取りすぎていたら [[cmd:auto.undo]]。足りなければ一段強いプリセットでもう一度実行します。</li>
</ol>
<p class="tip">最初は「大まか」で実行し、残り具合を見てから強くするのが安全です。「細かい」は面の端を削ることがあります。</p>
<h3>フィルタの種類（タブ）</h3>
<ul>
<li><b>SOR</b>: 周りの点から離れた孤立点を取る。まずはこれ。倍率を下げるほど強くなります。</li>
<li><b>ROR</b>: 決めた半径の中に点が少ない所を取る。</li>
<li><b>クラスタ</b>: 小さな塊（人影の残り、粉じんの塊）をまとめて取る。</li>
<li><b>エッジ・ヒゲ</b>: 面の縁から飛び出したヒゲ状の点を取る。</li>
<li><b>距離・強度</b>: スキャナからの距離や、反射強度の範囲で取る。</li>
<li><b>間引き</b>: データを軽くしたいときに、一定間隔で点を減らす。</li>
</ul>
<p>各数値の意味は「ヘルプ → ノイズフィルター設定ガイド」に詳しくあります。</p>
<h3>その他の一括処理</h3>
<ul>
<li>[[cmd:auto.keep_main]]: 本体から離れた浮遊塊をまとめて取る。</li>
<li>[[cmd:auto.edge]]: ヒゲだけを手早く取る。</li>
</ul>
""",
    ),
    Topic(
        "select",
        "② 手動修正: 選んで消す",
        "ブラシ・矩形・円形・多角形で囲み、Del でノイズへ",
        """
<ol>
<li>上のバーで選択ツールを選ぶ: [[cmd:mode.brush]] / [[cmd:mode.rect]] / [[cmd:mode.circle]] / [[cmd:mode.polygon]]。</li>
<li>3D 画面で消したい所を囲む（ブラシはなぞる、多角形はクリックで頂点を置いて [[key:Enter]]）。選ばれた点は黄色になります。</li>
<li>[[key:Del]]（[[cmd:edit.delete|ノイズへ移す]]）。下のバーや右クリックのパレットからでも実行できます。</li>
</ol>
<table>
<tr><td>[[key:Shift]] を押しながら囲む</td><td>今の選択に足す</td></tr>
<tr><td>[[key:Ctrl]] を押しながら囲む</td><td>今の選択から外す</td></tr>
<tr><td>[[key:I]]</td><td>選択を反転（囲んだ所だけ残したいとき）</td></tr>
<tr><td>[[key:Esc]]</td><td>選択を解除</td></tr>
<tr><td>[[key:[]] [[key:]]]</td><td>ブラシを細く / 太く</td></tr>
</table>
<ul>
<li>[[cmd:mode.instant]]: 囲んだ瞬間にノイズへ移します。たくさん消すときに速い方法です（[[key:Ctrl+Z]] で戻せます）。</li>
<li>[[cmd:mode.clean]]: ノイズ処理に関係する物だけを画面に残します。</li>
<li>選択は画面に見えている点が対象です。奥の点まで選びたくないときは、[[topic:section|断面]] で範囲を絞ってから選びます。</li>
<li>終わったら [[cmd:mode.orbit]] に戻します。</li>
</ul>
""",
    ),
    Topic(
        "tools",
        "② 手動修正: 塊・ヒゲ・範囲を指定して取る",
        "連結塊の選択、範囲内だけの精密除去",
        """
<h3>塊ごと選ぶ</h3>
<ol>
<li>消したい塊の一部を選択ツールで少しだけ選ぶ。</li>
<li>[[cmd:edit.wand]] を押す。選んだ点とつながっている塊全体が選ばれます。</li>
<li>[[key:Del]] でノイズへ移す。</li>
</ol>
<h3>ここだけ細かく取る</h3>
<ol>
<li>ノイズが残っている範囲を大きめに囲む。</li>
<li>[[cmd:edit.local_clean]] を押す。囲んだ範囲だけに細かいノイズ除去がかかり、範囲の外は変わりません。</li>
</ol>
<p class="tip">全体に強いフィルタをかけると面まで削れることがあります。全体は「大まか」で済ませ、残った所だけをこの機能で取るときれいに仕上がります。</p>
""",
    ),
    Topic(
        "layer",
        "③ 確認とレイヤー",
        "ノイズの表示と復元、レイヤー分け、地面・平面の自動抽出",
        """
<h3>消しすぎを確認して戻す</h3>
<ol>
<li>[[cmd:layer.toggle_noise]] でノイズ（赤）を表示する。</li>
<li>必要な点が赤くなっていたら、選択ツールで囲んで [[cmd:layer.restore_selected]]。</li>
<li>最初からやり直したいときは [[cmd:layer.restore_all]]。</li>
</ol>
<ul>
<li>[[cmd:layer.toggle_normal]] で残す点を隠すと、ノイズだけを見られます。</li>
<li>[[cmd:view.color_layer]] にすると、どの点がどのレイヤーかが色で分かります。</li>
</ul>
<h3>レイヤーに分ける</h3>
<ul>
<li>「③確認」のレイヤー一覧で、レイヤーの追加・名前の変更・表示の切替・ロック（保護）ができます。ロックしたレイヤーの点は、選択や自動処理で動きません。</li>
<li>点を選んでから [[cmd:layer.move]] で、一覧で選んだレイヤーへ移せます。</li>
<li>[[cmd:layer.ground]] / [[cmd:layer.plane]]: 地面や大きな平面を自動で見つけて、別のレイヤーに分けます。</li>
<li>「状態保存 (.npz)」で、レイヤー分けの途中経過をファイルに保存し、後で読み込めます。</li>
</ul>
""",
    ),
    Topic(
        "section",
        "断面で切って見る",
        "ある範囲だけを表示して、中を確認・編集する",
        """
<ol>
<li>左パネルの「断面」を開く（[[cmd:section.panel]]）。</li>
<li>切る向きを選ぶ: [[cmd:section.z]]（床・天井）、[[cmd:section.x]]、[[cmd:section.y]]。</li>
<li>「シーク」のスライダーで位置を動かし、「厚み幅」で厚さを決める。厚みは [[cmd:section.thick_0.1|0.1m]] / [[cmd:section.thick_0.5|0.5m]] / [[cmd:section.thick_1|1m]] / [[cmd:section.thick_2|2m]] のボタンでも変えられます。</li>
<li>終わったら [[cmd:section.reset]]。</li>
</ol>
<ul>
<li>[[cmd:section.from_current]]: 今見ている所（点を選んでいればその点）へ断面を移します。</li>
<li>「正対」: 断面に対してまっすぐ向く視点にします。「視点追従」を入れると、位置を動かすたびに視点も付いてきます。</li>
<li>「基点」: 位置を厚みの中心にするか、片側の端にするかを選びます。</li>
<li>スライダーはどちらも真ん中が起点です。「シーク」は基点が真ん中で、左右どちらへも動かせます（点群の端から端まで届きます）。「厚み幅」は真ん中が 1m で、左へ動かすと薄く（1cm まで）、右へ動かすと厚く（100m まで）なります。細かい値は右の数値欄に打てます。</li>
<li>[[cmd:section.guides]]: 黄色（基点）と水色（厚み）の枠が邪魔なときに消します。</li>
<li>[[cmd:section.crop]]: 箱の外をまとめてノイズへ移します。周囲の不要な範囲を切り落とすのに使います。</li>
</ul>
<h3>斜めの壁に沿って切る</h3>
<ol>
<li>[[cmd:section.wall_pick]] を押し、壁の上の 2 点を順にクリックする。</li>
<li>「横」「縦」「正面」「真横」で、壁に合わせた視点へ切り替える。</li>
<li>「斜め壁スライス」を入れると、壁からの距離で切った断面になります。</li>
</ol>
<p class="tip">断面を入れている間は、選択も断面の中の点だけが対象になります。壁の手前のノイズだけを消したいときに便利です。</p>
""",
    ),
    Topic(
        "detail",
        "細部を元の密度で見る",
        "断面内の全点表示、元ファイルからの読み直し、高精細表示 (LOD)",
        """
<p>大きなファイルは点を間引いて読み込み、画面にはさらに間引いて描いています。細部を確かめたいときは次の 3 つを使います。</p>
<h3>1. 断面の中は全点を描く</h3>
<p>[[cmd:section.full_detail]] が入っていると、断面を入れている間は、箱の中の点を間引かずに描きます（上限 1,000 万点）。回転が重いときは切ってください。</p>
<h3>2. 断面の中を元ファイルから読み直す</h3>
<ol>
<li>断面で見たい範囲を絞る（箱の中が 1,000 万点以下になるくらい）。</li>
<li>[[cmd:section.reload]] を押す。箱の中が、元ファイルにある全部の点で重ね描きされます。</li>
</ol>
<p>初回は作業用データ（元ファイルの横の「.e57.cache」フォルダ。1 点 17 バイト）を作ります。2 回目からは速くなります。</p>
<h3>3. 高精細表示 (LOD)</h3>
<ol>
<li>[[cmd:section.lod]] を入れる。初回だけ LOD データを作るか聞かれます（1 億点あたり 1〜2 分）。</li>
<li>見たい所へ寄って手を止める。約 0.5 秒後に、画面に入っている範囲が元ファイルの密度で描かれます。</li>
</ol>
<ul>
<li>2 と 3 で描かれる点は表示専用です。選択やノイズへの移動は、これまでどおり読み込んだ点に対して行います。保存するときに、その判定が元ファイルの全点へ写されます。</li>
<li>ノイズへ移した所は、重ね描きからも隠れます。</li>
<li>読み込んでから保存するまで、元ファイルを移動したり上書きしたりしないでください。</li>
</ul>
""",
    ),
    Topic(
        "measure",
        "計測（座標と距離）",
        "点の座標を読む、2 点間の距離・高低差を測る",
        """
<ol>
<li>[[cmd:mode.measure]] を押す。</li>
<li>点をクリックする。その点の X・Y・Z が画面に出ます。</li>
<li>2 点目をクリックすると、直線距離・水平距離・高低差と、X・Y・Z それぞれの差が出ます。</li>
<li>[[key:Ctrl+C]] で値をコピー（Excel や CAD にそのまま貼れます）。右クリックで測り直し、[[key:M]] で終了。</li>
</ol>
<ul>
<li>クリックした位置の近く（12 ピクセル以内）に点が無いときは拾いません。寄ってからクリックしてください。</li>
<li>座標系を変えて読み込んだときは、元ファイルの座標も一緒に表示します。</li>
<li>水平・垂直を正確に見たいときは、[[cmd:view.projection|正射影]] にして [[cmd:view.top]] や [[cmd:view.front]] から測ります。</li>
</ul>
""",
    ),
    Topic(
        "save",
        "保存",
        "ノイズを除いた点群を E57 で書き出す",
        """
<ol>
<li>左パネル下の「保存対象」を選ぶ（通常は「有効レイヤー統合（ノイズ除外）」）。</li>
<li>[[cmd:file.save]] を押し、保存先を決める。</li>
<li>間引いて読み込んだファイルでは保存方法を聞かれます。「全点で保存（推奨）」を選ぶと、元ファイルの全部の点に判定を写して、元と同じ密度で書き出します。</li>
</ol>
<ul>
<li>座標の精度、色、反射強度、スキャンの構成は、元ファイルのまま保たれます。</li>
<li>「全点で保存」は元ファイルを読み直すので、1 億点で数分かかります。読み込んでから保存が終わるまで、元ファイルを動かさないでください。</li>
<li>保存していない編集があると、ウィンドウの題名の先頭に「*」が付きます。</li>
</ul>
""",
    ),
    Topic(
        "big",
        "1 億点を超えるファイル",
        "間引き読込の仕組みと、密度を保つための使い方",
        """
<p>点数が読込上限を超えるファイルは、メモリを守るために「何点かに 1 点」だけを読み込みます。作業はその間引いた点で行い、保存のときに元ファイルの全点へ結果を写します。</p>
<table>
<tr><td><b>読込上限</b></td><td>設定 → システム →「大容量点群読込上限」。自動のままが安全です（32GB の PC で 6,000 万点）。</td></tr>
<tr><td><b>描画点数</b></td><td>設定 → 表示 →「描画点数」。画面に描く点の上限。重いときは下げます。</td></tr>
<tr><td><b>細部の確認</b></td><td>[[topic:detail|細部を元の密度で見る]] の 3 つの方法を使います。</td></tr>
<tr><td><b>保存</b></td><td>「全点で保存（推奨）」を選ぶと、元と同じ密度で書き出せます（[[topic:save|保存]]）。</td></tr>
</table>
<ul>
<li>読み直し用と高精細表示用の作業用データは、元ファイルの横の「.e57.cache」フォルダに入ります。5 億点で、合わせて約 17GB です。[[cmd:help.workdata]] で、大きさの確認・削除・置き場所の変更ができます。削除しても、元の E57 と保存した結果は変わりません。</li>
<li>元ファイルを書き換えたり移動したりすると、作業用データは使えなくなり、作り直しになります（使われなくなったものは、管理の画面でまとめて消せます）。</li>
<li>作業用データの作成と保存の前に、ドライブの空きを確かめます。足りない見込みのときは、書き始める前に知らせます。</li>
</ul>
""",
    ),
    Topic(
        "settings",
        "設定の項目",
        "表示・操作感度・システムの各設定",
        """
<h3>表示</h3>
<ul>
<li><b>点サイズ / 点形状</b>: 点の大きさと形。密な点群は 1〜2px の四角がくっきり見えます。</li>
<li><b>描画点数</b>: 画面に描く点の上限。少ないほど軽くなります。</li>
<li><b>環境設定 / グリッド</b>: 背景の色と、床のマス目の表示。</li>
<li><b>初期配色</b>: 開いた直後の色の付け方。</li>
<li><b>視野角 (FOV) / 深度フォグ</b>: 遠近感の強さと、奥を霞ませる表示。</li>
</ul>
<h3>操作感度</h3>
<p>回転・移動・拡大の速さと、上下の回転方向を変えられます。</p>
<h3>システム</h3>
<ul>
<li><b>大容量点群読込上限</b>: 読み込む点数の上限（[[topic:big|1 億点を超えるファイル]]）。</li>
<li><b>Undo 最大保持数</b>: 元に戻せる回数。点数が多いと 1 回ぶんのメモリも大きくなります。</li>
<li><b>UI スタイル / テーマ / アニメーション</b>: 画面の見た目。従来の画面に戻すこともできます（次回の起動から）。</li>
<li><b>作業用データ</b>: 断面の読み直しと高精細表示のためのデータの大きさと、ドライブの空き。「管理...」で削除と置き場所の変更ができます（[[cmd:help.workdata]]）。</li>
<li><b>版</b>: 今動いているアプリの版（配布版は作った日時）。ウィンドウのタイトルと [[cmd:help.about]] にも出ます。</li>
</ul>
<p>「設定を保存」で次回も同じ設定になります。「初期設定」で最初の状態に戻ります。</p>
""",
    ),
    Topic("keys", "ショートカット一覧", "キーボードとマウスの操作の一覧", ""),
    Topic(
        "trouble",
        "困ったとき",
        "重い、選べない、消しすぎた、保存できない",
        """
<table>
<tr><td><b>動きが重い</b></td><td>設定 → 表示の「描画点数」を下げる。[[cmd:section.lod|高精細表示]] と [[cmd:section.full_detail|ボックス内の全点表示]] を切る。点サイズを小さくする。</td></tr>
<tr><td><b>点が選べない</b></td><td>上のバーが「回転」になっていないか確かめ、選択ツールを選ぶ。レイヤーがロックされていないか、「③確認」で確かめる。</td></tr>
<tr><td><b>消しすぎた</b></td><td>[[key:Ctrl+Z]] で戻す。または [[cmd:layer.toggle_noise|ノイズを表示]] して、囲んで [[cmd:layer.restore_selected|復元]]。</td></tr>
<tr><td><b>点群を見失った</b></td><td>[[cmd:view.fit]]。断面が入ったままなら [[cmd:section.reset]]。</td></tr>
<tr><td><b>保存や読み直しでエラー</b></td><td>元ファイルを、読み込んだ後に移動・上書きしていないか確かめる。開き直してからやり直す。</td></tr>
<tr><td><b>ディスクが足りない</b></td><td>[[cmd:help.workdata]] で、使わない作業用データを削除する（次に使うときに作り直されます）。置き場所を、空きの多いドライブへ変えることもできます。</td></tr>
<tr><td><b>画面の表示がおかしい</b></td><td>ヘルプ →「従来UIに戻す」で以前の画面にする。</td></tr>
<tr><td><b>原因を調べたい</b></td><td>[[cmd:help.log_folder]] で履歴ログを開く。問い合わせるときは、[[cmd:help.about]] の内容（版とログの場所）を添える。</td></tr>
</table>
""",
    ),
]

# 検索用の読み（本文は漢字が多いので、ひらがなでも探せるようにする）
TOPIC_KEYWORDS: Dict[str, str] = {
    "start": "はじめに がめん みかた ながれ きほん てじゅん",
    "quick": "けんさく みぎくりっく こまど べつまど らんちゃー ぱれっと がいど すばやく",
    "legend": "ぼたん みかた ひょうじ じょうたい しょりちゅう かんりょう おせない むこう つうち もとにもどす みほぞん あいこん しるし",
    "open": "ひらく よみこみ ふぁいる ふぉるだ ざひょうけい さいきん",
    "nav": "してん かいてん いどう かくだい しゅくしょう とうし せいしゃえい ふぃっと",
    "auto": "じどう のいず じょきょ ふぃるた ぷりせっと れしぴ じっこう",
    "select": "せんたく えらぶ けす さくじょ ぶらし くけい えんけい たかっけい そくさくじょ",
    "tools": "かたまり れんけつ ひげ せいみつ はんい きょくしょ",
    "layer": "れいやー かくにん ふくげん もどす じめん へいめん ろっく",
    "section": "だんめん きる すらいす あつみ きてん ななめ かべ くろっぷ",
    "detail": "みつど ぜんてん よみなおし こうせいさい えるおーでぃー lod きゃっしゅ さいぶ",
    "measure": "けいそく ざひょう きょり こうていさ すんぽう はかる",
    "save": "ほぞん かきだし しゅつりょく ぜんてん うわがき",
    "big": "おくてん おおきい きょだい まびき めもり じょうげん でぃすく",
    "settings": "せってい てんさいず びょうがてんすう かんど てーま あんどぅ",
    "keys": "しょーとかっと きー いちらん まうす",
    "trouble": "こまった おもい えらべない けしすぎ えらー ほぞんできない とらぶる",
}

MOUSE_ROWS = [
    ("左ドラッグ", "視点を回す（選択ツールのときは囲む・なぞる）"),
    ("右ドラッグ / 中ドラッグ", "画面を平行移動する"),
    ("ホイール", "拡大・縮小"),
    ("右クリック", "操作のパレットを開く（計測中は測り直し）"),
    ("Alt + 左ドラッグ", "選択ツールのまま視点を回す"),
    ("Shift / Ctrl + 囲む", "選択に足す / 選択から外す"),
    ("Space 長押し", "押している間だけ平行移動"),
    ("W A S D / Q E", "前後左右 / 上下に移動"),
    ("[ / ]", "ブラシを細く / 太く"),
    ("Enter", "多角形選択を確定"),
    ("Ctrl+K", "機能を検索して実行"),
    ("F1", "この説明書を開く"),
]

_TOKEN_RE = re.compile(r"\[\[(cmd|key|topic):(.+?)\]\]")


def topic_ids() -> List[str]:
    return [t.id for t in TOPICS]


def command_ids_in_manual() -> List[str]:
    """本文から参照しているコマンド ID（表に無い ID を書いていないかの検査用）"""
    ids: List[str] = []
    for t in TOPICS:
        for kind, rest in _TOKEN_RE.findall(t.body):
            if kind == "cmd":
                ids.append(rest.split("|", 1)[0].strip())
    return ids


def _kbd(text: str) -> str:
    return f'<span class="kbd">&nbsp;{html.escape(text)}&nbsp;</span>'


def _keys_body(commands: List[Command]) -> str:
    parts = ["<h3>マウスとキーの基本</h3><table>"]
    for key, what in MOUSE_ROWS:
        parts.append(f"<tr><td>{_kbd(key)}</td><td>{html.escape(what)}</td></tr>")
    parts.append("</table>")
    for group in GROUP_ORDER:
        rows = [c for c in commands if c.group == group and c.shortcut]
        if not rows:
            continue
        parts.append(f"<h3>{html.escape(group)}</h3><table>")
        for c in rows:
            parts.append(f'<tr><td>{_kbd(c.shortcut)}</td><td><a href="cmd:{c.id}">{html.escape(c.title)}</a></td></tr>')
        parts.append("</table>")
    return "".join(parts)


def _expand(body: str, by_id: Dict[str, Command]) -> str:
    titles = {t.id: t.title for t in TOPICS}

    def repl(m: "re.Match") -> str:
        kind, rest = m.group(1), m.group(2)
        if kind == "key":
            return _kbd(rest)
        target, _, label = rest.partition("|")
        target = target.strip()
        if kind == "topic":
            return f'<a href="topic:{target}">{html.escape(label or titles.get(target, target))}</a>'
        cmd = by_id.get(target)
        if cmd is None:
            return html.escape(label or target)
        text = f'<a href="cmd:{cmd.id}">{html.escape(label or cmd.title)}</a>'
        if cmd.shortcut and not label:
            text += " " + _kbd(cmd.shortcut)
        return text

    return _TOKEN_RE.sub(repl, body)


def manual_css(theme: str) -> str:
    t = get_tokens(theme)
    link = t.accent_soft
    return f"""
body {{ color: {t.text_primary}; font-size: {t.size_sm}px; }}
h2 {{ color: {t.text_primary}; font-size: {t.size_lg}px; margin-top: 4px; margin-bottom: 6px; }}
h3 {{ color: {t.text_primary}; font-size: {t.size_md}px; margin-top: 14px; margin-bottom: 4px; }}
p {{ margin-top: 4px; margin-bottom: 6px; }}
li {{ margin-bottom: 4px; }}
a {{ color: {link}; text-decoration: none; font-weight: 600; }}
td {{ padding: 3px 10px 3px 0px; vertical-align: top; }}
.summary {{ color: {t.text_secondary}; }}
.tip {{ background-color: {t.bg_elevated}; color: {t.text_secondary}; padding: 6px; }}
.kbd {{ background-color: {t.bg_elevated}; color: {t.text_primary}; font-family: {t.font_mono}; font-size: {t.size_xs}px; }}
"""


def _roles_table() -> str:
    """ボタンの役割の一覧（gui/button_kit.py の ROLE_GUIDE をそのまま表にする。説明書と実装がずれないように）"""
    from gui.button_kit import ROLE_GUIDE

    rows = "".join(
        f"<tr><td><b>{html.escape(look)}</b></td><td>{html.escape(usage)}</td></tr>" for _role, look, usage in ROLE_GUIDE
    )
    return f"<table>{rows}</table>"


def render_topic(topic: Topic, commands: List[Command]) -> str:
    by_id = {c.id: c for c in commands}
    body = _keys_body(commands) if topic.id == "keys" else _expand(topic.body, by_id)
    if "[[roles]]" in body:
        body = body.replace("[[roles]]", _roles_table())
    return f'<h2>{html.escape(topic.title)}</h2><p class="summary">{html.escape(topic.summary)}</p>{body}'


def topic_search_text(topic: Topic, commands: List[Command]) -> str:
    raw = render_topic(topic, commands)
    return normalize(re.sub(r"<[^>]+>", " ", raw) + " " + TOPIC_KEYWORDS.get(topic.id, ""))


class ManualBrowser(QWidget):
    """説明書の表示部品。compact=True は左パネル用（項目はコンボ）、False は別ウィンドウ用（項目は左の一覧）"""

    command_requested = Signal(str)

    def __init__(self, win, commands_provider: Callable[[], List[Command]], compact: bool = False, parent=None):
        super().__init__(parent)
        self._win = win
        self._provider = commands_provider
        self._compact = compact
        self._current = TOPICS[0].id
        self._visible_ids: List[str] = topic_ids()

        self.search = QLineEdit(self)
        self.search.setObjectName("manual_search")
        self.search.setPlaceholderText("説明書の中を検索（例: 断面、保存、重い）")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)

        self.view = QTextBrowser(self)
        self.view.setObjectName("manual_view")
        self.view.setOpenLinks(False)
        self.view.setOpenExternalLinks(False)
        self.view.anchorClicked.connect(self._on_anchor)

        if compact:
            self.topic_combo: Optional[QComboBox] = QComboBox(self)
            # 長い項目名で左パネルの最小幅を押し広げない
            self.topic_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self.topic_combo.setMinimumContentsLength(10)
            self.topic_list: Optional[QListWidget] = None
            self.topic_combo.currentIndexChanged.connect(self._on_combo_changed)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(4)
            lay.addWidget(self.search)
            lay.addWidget(self.topic_combo)
            lay.addWidget(self.view, 1)
            self.view.setMinimumHeight(300)
        else:
            self.topic_combo = None
            self.topic_list = QListWidget(self)
            self.topic_list.setObjectName("manual_topics")
            self.topic_list.setFixedWidth(330)
            self.topic_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.topic_list.setWordWrap(True)
            self.topic_list.currentRowChanged.connect(self._on_list_changed)
            left = QVBoxLayout()
            left.setSpacing(6)
            left.addWidget(self.search)
            left.addWidget(self.topic_list, 1)
            lay = QHBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(10)
            lay.addLayout(left)
            lay.addWidget(self.view, 1)
        self._fill_topics()
        self.refresh_theme()

    # --- 項目 ---
    def _topics_visible(self) -> List[Topic]:
        return [t for t in TOPICS if t.id in self._visible_ids]

    def _fill_topics(self) -> None:
        topics = self._topics_visible()
        if self.topic_combo is not None:
            self.topic_combo.blockSignals(True)
            self.topic_combo.clear()
            for t in topics:
                self.topic_combo.addItem(t.title, t.id)
            self.topic_combo.blockSignals(False)
        if self.topic_list is not None:
            self.topic_list.blockSignals(True)
            self.topic_list.clear()
            for t in topics:
                it = QListWidgetItem(t.title)
                it.setData(Qt.UserRole, t.id)
                it.setToolTip(t.summary)
                self.topic_list.addItem(it)
            self.topic_list.blockSignals(False)
        ids = [t.id for t in topics]
        if ids:
            self.show_topic(self._current if self._current in ids else ids[0])
        else:
            self.view.setHtml("<p>見つかりませんでした。別の言葉で検索してください。</p>")

    def _apply_filter(self, text: str) -> None:
        tokens = [normalize(x) for x in (text or "").replace("　", " ").split()]
        tokens = [x for x in tokens if x]
        if not tokens:
            self._visible_ids = topic_ids()
        else:
            cmds = self._provider()
            self._visible_ids = [t.id for t in TOPICS if all(tok in topic_search_text(t, cmds) for tok in tokens)]
        self._fill_topics()

    def visible_topic_ids(self) -> List[str]:
        return list(self._visible_ids)

    def current_topic_id(self) -> str:
        return self._current

    def show_topic(self, topic_id: str) -> None:
        topic = next((t for t in TOPICS if t.id == topic_id), None)
        if topic is None:
            return
        self._current = topic.id
        self.view.setHtml(render_topic(topic, self._provider()))
        if self.topic_combo is not None:
            idx = self.topic_combo.findData(topic.id)
            if idx >= 0 and idx != self.topic_combo.currentIndex():
                self.topic_combo.blockSignals(True)
                self.topic_combo.setCurrentIndex(idx)
                self.topic_combo.blockSignals(False)
        if self.topic_list is not None:
            for r in range(self.topic_list.count()):
                if self.topic_list.item(r).data(Qt.UserRole) == topic.id and r != self.topic_list.currentRow():
                    self.topic_list.blockSignals(True)
                    self.topic_list.setCurrentRow(r)
                    self.topic_list.blockSignals(False)

    def _on_combo_changed(self, idx: int) -> None:
        if self.topic_combo is not None and idx >= 0:
            self.show_topic(str(self.topic_combo.itemData(idx)))

    def _on_list_changed(self, row: int) -> None:
        if self.topic_list is not None and row >= 0:
            self.show_topic(str(self.topic_list.item(row).data(Qt.UserRole)))

    # --- リンク ---
    def _on_anchor(self, url: QUrl) -> None:
        self.follow_link(url.toString())

    def follow_link(self, link: str) -> None:
        kind, _, target = link.partition(":")
        if kind == "topic":
            if target not in self._visible_ids:
                self.search.clear()
            self.show_topic(target)
        elif kind == "cmd":
            self.command_requested.emit(target)

    # --- テーマ ---
    def refresh_theme(self) -> None:
        self.view.document().setDefaultStyleSheet(manual_css(getattr(self._win, "ui_theme", "dark")))
        self.show_topic(self._current)


class ManualWindow(QDialog):
    """操作説明書の別ウィンドウ（開いたまま作業できる）"""

    def __init__(self, win, commands_provider: Callable[[], List[Command]]):
        super().__init__(win)
        self.setObjectName("manual_window")
        self.setWindowTitle("操作説明書")
        self.setModal(False)
        self.resize(1000, 720)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 10)
        lay.setSpacing(8)
        self.browser = ManualBrowser(win, commands_provider, compact=False, parent=self)
        lay.addWidget(self.browser, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton("閉じる", self)
        btn.setProperty("role", "outline")
        btn.clicked.connect(self.close)
        row.addWidget(btn)
        lay.addLayout(row)


class ManualTab(QWidget):
    """設定パネルに足す「説明書」タブ（小さな説明書 ＋ 別ウィンドウで開く ＋ 操作ガイドの表示切替）"""

    open_window_requested = Signal()
    hints_toggled = Signal(bool)          # モード別の操作ガイド
    tips_toggled = Signal(bool)           # 状況に応じたヒント
    open_tips_requested = Signal()        # ヒント集を開く
    reset_hints_requested = Signal()      # 閉じたヒントをもう一度出す

    def __init__(self, win, commands_provider: Callable[[], List[Command]], hints_enabled: bool = True,
                 tips_enabled: bool = True, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 6, 4, 6)
        lay.setSpacing(6)
        self.browser = ManualBrowser(win, commands_provider, compact=True, parent=self)
        lay.addWidget(self.browser, 1)
        self.btn_open_window = QPushButton("大きな画面で開く (F1)", self)
        self.btn_open_window.setProperty("role", "outline")
        self.btn_open_window.clicked.connect(self.open_window_requested.emit)
        lay.addWidget(self.btn_open_window)
        self.chk_hints = QCheckBox("モード別の操作ガイドを 3D 画面に表示する", self)
        self.chk_hints.setChecked(bool(hints_enabled))
        self.chk_hints.toggled.connect(self.hints_toggled.emit)
        lay.addWidget(self.chk_hints)
        self.chk_tips = QCheckBox("状況に応じたヒントを 3D 画面に表示する", self)
        self.chk_tips.setToolTip("「次にやるとよいこと」を 3D 画面の右上にカードで出します（ツールを使っている間は出しません）")
        self.chk_tips.setChecked(bool(tips_enabled))
        self.chk_tips.toggled.connect(self.tips_toggled.emit)
        lay.addWidget(self.chk_tips)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_tips = QPushButton("ヒント集を開く", self)
        self.btn_tips.setProperty("role", "outline")
        self.btn_tips.clicked.connect(lambda _=False: self.open_tips_requested.emit())
        row.addWidget(self.btn_tips, 1)
        self.btn_reset_hints = QPushButton("閉じたヒントを戻す", self)
        self.btn_reset_hints.setProperty("role", "outline")
        self.btn_reset_hints.setToolTip("「閉じる」「今後表示しない」にしたヒントを、もう一度出るようにします")
        self.btn_reset_hints.clicked.connect(lambda _=False: self.reset_hints_requested.emit())
        row.addWidget(self.btn_reset_hints, 1)
        lay.addLayout(row)
