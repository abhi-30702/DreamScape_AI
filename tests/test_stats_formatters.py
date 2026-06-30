import pandas as pd
import pytest

from eval.stats import formatters


def test_write_table_creates_both_files(tmp_path):
    df = pd.DataFrame({"video_id": ["v01", "v02"], "mos": [4.0, 3.5]})
    formatters.write_table(df, tmp_path, stem="mos_test")
    assert (tmp_path / "mos_test.csv").is_file()
    assert (tmp_path / "mos_test.md").is_file()


def test_write_table_csv_roundtrips(tmp_path):
    df = pd.DataFrame({"video_id": ["v01"], "mos": [4.0]})
    formatters.write_table(df, tmp_path, stem="t")
    loaded = pd.read_csv(tmp_path / "t.csv")
    assert loaded["video_id"].iloc[0] == "v01"
    assert loaded["mos"].iloc[0] == pytest.approx(4.0)


def test_write_table_markdown_contains_pipe_table(tmp_path):
    df = pd.DataFrame({"video_id": ["v01"], "mos": [4.0]})
    formatters.write_table(df, tmp_path, stem="t")
    md = (tmp_path / "t.md").read_text(encoding="utf-8")
    assert "| video_id" in md
    assert "v01" in md
    assert "4.00" in md  # rounded to 2 dp


def test_write_summary_writes_csv_and_md(tmp_path):
    summary = {"n_raters": 20, "n_videos": 20, "mos_visual_quality": 3.85}
    formatters.write_summary(summary, tmp_path)
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "summary.md").is_file()
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "n_raters" in md
    assert "20" in md


def test_write_provenance_writes_text_file(tmp_path):
    formatters.write_provenance(tmp_path, args={"responses": "responses_dump/"}, input_counts={"n_submissions": 350})
    prov_path = tmp_path / "provenance.txt"
    assert prov_path.is_file()
    text = prov_path.read_text(encoding="utf-8")
    assert "pandas" in text
    assert "scipy" in text
    assert "statsmodels" in text
    assert "responses_dump/" in text
    assert "350" in text
