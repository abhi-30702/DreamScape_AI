import argparse
import sys
from pathlib import Path

from eval.stats import correlations, formatters, kappa, loader, mos


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.stats",
        description="Compute Phase 3 rater study statistics and write thesis tables",
    )
    parser.add_argument("--responses", type=Path, required=True,
                        help="Directory containing responses/<rater_id>/<video_id>.json")
    parser.add_argument("--manifest", type=Path,
                        default=Path("study_videos/manifest.json"),
                        help="Path to study_videos/manifest.json")
    parser.add_argument("--metrics", type=Path, default=None,
                        help="Directory of Phase 1 metric JSONs (eval_results/<run_id>.json)")
    parser.add_argument("--out", type=Path, default=Path("thesis_tables"),
                        help="Output directory for tables and provenance")
    parser.add_argument("--min-raters-per-video", type=int, default=2,
                        help="Minimum raters per video for kappa (default 2)")
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        ratings_df, auto_df, overall_comments = loader.load_all(
            args.responses, args.manifest, args.metrics,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.out.exists() and any(args.out.iterdir()):
        print(f"overwriting existing files in {args.out}")

    per_video = mos.compute_mos_per_video(ratings_df)
    by_sentiment = mos.compute_mos_by_sentiment(ratings_df)
    by_style = mos.compute_mos_by_style(ratings_df)
    summary = mos.compute_summary(ratings_df)

    kappa_df = kappa.compute_fleiss_kappa_per_dimension(
        ratings_df, min_raters_per_video=args.min_raters_per_video,
    )
    valid_kappas = kappa_df["kappa"].dropna()
    summary["mean_kappa"] = float(valid_kappas.mean()) if not valid_kappas.empty else float("nan")

    formatters.write_summary(summary, args.out)
    formatters.write_table(per_video, args.out, stem="mos_per_video")
    formatters.write_table(by_sentiment, args.out, stem="mos_by_sentiment")
    formatters.write_table(by_style, args.out, stem="mos_by_style")
    formatters.write_table(kappa_df, args.out, stem="kappa_per_dimension")

    if args.metrics is not None:
        corr_df = correlations.compute_correlations(per_video, auto_df)
        formatters.write_table(corr_df, args.out, stem="auto_vs_human_correlations")
    else:
        print("auto metrics not provided; skipping correlations")

    formatters.write_provenance(
        args.out,
        args={
            "responses": str(args.responses),
            "manifest": str(args.manifest),
            "metrics": str(args.metrics) if args.metrics else "(none)",
            "out": str(args.out),
            "min_raters_per_video": args.min_raters_per_video,
        },
        input_counts={
            "n_submissions": int(ratings_df[["rater_id", "video_id"]].drop_duplicates().shape[0]),
            "n_overall_comments": len(overall_comments),
            "n_videos_in_manifest": int(auto_df.shape[0]),
        },
    )

    for stem in ["summary", "mos_per_video", "mos_by_sentiment",
                 "mos_by_style", "kappa_per_dimension"]:
        n_rows = sum(1 for _ in (args.out / f"{stem}.csv").open(encoding="utf-8")) - 1
        print(f"{stem}.csv  ({n_rows} rows)")
    if args.metrics is not None:
        n_rows = sum(1 for _ in (args.out / "auto_vs_human_correlations.csv").open(encoding="utf-8")) - 1
        print(f"auto_vs_human_correlations.csv  ({n_rows} rows)")

    return 0


def main() -> int:
    args = _build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
