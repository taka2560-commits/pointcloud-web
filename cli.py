"""
E57点群ノイズ処理のコマンドラインインターフェース (CLI)
"""

import argparse
import os
import sys

from core.processor import FilterConfig, PointCloudProcessor


def main():
    parser = argparse.ArgumentParser(
        description="E57点群データからノイズ（浮遊点・外れ値）を除去するCLIツール"
    )
    parser.add_argument("input", help="入力E57ファイルパス")
    parser.add_argument("-o", "--output", help="出力先E57ファイルパス (省略時は自動命名)")

    # 統計的外れ値除去 (SOR)
    parser.add_argument(
        "--no-sor",
        action="store_true",
        help="統計的外れ値除去 (SOR) を無効化する",
    )
    parser.add_argument(
        "--sor-neighbors",
        type=int,
        default=20,
        help="SOR近傍探索点数 (デフォルト: 20)",
    )
    parser.add_argument(
        "--sor-ratio",
        type=float,
        default=2.0,
        help="SOR標準偏差倍率閾値 (デフォルト: 2.0)",
    )

    # 半径ベース外れ値除去 (ROR)
    parser.add_argument(
        "--ror",
        action="store_true",
        help="半径ベース外れ値除去 (ROR) を有効化する",
    )
    parser.add_argument(
        "--ror-radius",
        type=float,
        default=0.05,
        help="ROR探索半径 [m] (デフォルト: 0.05)",
    )
    parser.add_argument(
        "--ror-min-points",
        type=int,
        default=16,
        help="ROR必要最小近傍点数 (デフォルト: 16)",
    )

    # ボクセル間引き
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="ボクセル間引きサイズ [m] (0の場合は間引きなし)",
    )

    # 距離フィルタ
    parser.add_argument(
        "--min-dist",
        type=float,
        default=None,
        help="原点からの最小許容距離 [m]",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=None,
        help="原点からの最大許容距離 [m]",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"エラー: 入力ファイルが見つかりません: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_denoised{ext}"

    config = FilterConfig(
        use_sor=not args.no_sor,
        sor_neighbors=args.sor_neighbors,
        sor_std_ratio=args.sor_ratio,
        use_ror=args.ror,
        ror_radius=args.ror_radius,
        ror_min_points=args.ror_min_points,
        use_voxel_downsample=(args.voxel_size > 0.0),
        voxel_size=args.voxel_size,
        use_distance_filter=(args.min_dist is not None or args.max_dist is not None),
        min_distance=args.min_dist,
        max_distance=args.max_dist,
    )

    processor = PointCloudProcessor(config=config)

    def print_progress(ratio: float, msg: str):
        print(f"[{int(ratio * 100):3d}%] {msg}")

    print("=== E57点群ノイズ処理開始 ===")
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")

    try:
        summary = processor.process_file(input_path, output_path, progress_callback=print_progress)
        print("\n=== 処理結果サマリー ===")
        print(f"・元点数:   {summary.total_initial_points:,} 点")
        print(f"・残存点数: {summary.total_final_points:,} 点")
        print(f"・除去点数: {summary.total_removed_points:,} 点 ({summary.total_removal_ratio:.2f}% 削減)")
        print(f"・所要時間: {summary.total_time_taken_sec:.2f} 秒")
        print(f"・保存先:   {summary.output_file}")
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
