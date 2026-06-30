from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _df_to_markdown(df: pd.DataFrame, decimals: int = 2) -> str:
    rounded = df.copy()
    for col in rounded.select_dtypes(include="number").columns:
        rounded[col] = rounded[col].round(decimals)
    return rounded.to_markdown(index=False, floatfmt=f".{decimals}f")


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, decimals: int = 2) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_df_to_markdown(df, decimals=decimals) + "\n", encoding="utf-8")


def write_summary(summary: dict, out_dir: Path) -> None:
    df = pd.DataFrame([summary])
    write_table(df, out_dir, stem="summary")


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("pandas", "scipy", "statsmodels"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def write_provenance(out_dir: Path, args: dict, input_counts: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = _package_versions()
    lines = [
        f"# DreamScapeAI Phase 3 stats provenance",
        f"# generated_at_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "[arguments]",
    ]
    for key, value in args.items():
        lines.append(f"{key} = {value}")
    lines.append("")
    lines.append("[input_counts]")
    for key, value in input_counts.items():
        lines.append(f"{key} = {value}")
    lines.append("")
    lines.append("[package_versions]")
    for name, version in versions.items():
        lines.append(f"{name} = {version}")
    (out_dir / "provenance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
