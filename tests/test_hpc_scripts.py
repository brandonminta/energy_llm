"""hpc/pilot_bundle.sh on fake logs: it must bundle the run directory printed
by THE given jobs (not another run), and work without sacct."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "hpc" / "pilot_bundle.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_pilot_bundle_uses_the_runs_of_the_given_jobs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    runs = {}
    for kind, report in (("verify", "verify_report.json"), ("smoke", "smoke_report.json")):
        for tag in ("old", "this"):
            d = tmp_path / "runs" / f"{kind}-{tag}"
            d.mkdir(parents=True)
            (d / report).write_text(json.dumps({"marker": f"{kind}-{tag}-report"}))
            (d / "manifest.json").write_text(json.dumps({"marker": f"{kind}-{tag}-manifest"}))
            runs[kind, tag] = d
    (logs / "verify_11.out").write_text(f"[verify] job=11\n  => ALL PASS   report: {runs['verify', 'this']}/verify_report.json\n")
    (logs / "verify_11.err").write_text("warn\n")
    (logs / "smoke_22.out").write_text(f"smoke x: 1/1\n  report: {runs['smoke', 'this']}/smoke_report.json\n")
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home), LOG_DIR=str(logs), PATH="/usr/bin:/bin")
    r = subprocess.run(["bash", str(SCRIPT), "11", "22"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = list(home.glob("pilot_bundle_*.txt"))
    assert len(out) == 1 and str(out[0]) in r.stdout
    text = out[0].read_text()
    for kind in ("verify", "smoke"):
        assert f"{kind}-this-report" in text and f"{kind}-this-manifest" in text
        assert f"{kind}-old" not in text
    assert "(missing: " in text and "smoke_22.err" in text      # absent .err reported, not fatal
    assert "sacct" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_pilot_bundle_usage_and_unfinished_job(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path), LOG_DIR=str(tmp_path))
    r = subprocess.run(["bash", str(SCRIPT), "1"], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "usage" in r.stderr
    (tmp_path / "verify_1.out").write_text("crashed\n")
    r = subprocess.run(["bash", str(SCRIPT), "1", "2"], env=env, capture_output=True, text=True)
    assert r.returncode == 0
    text = next(tmp_path.glob("pilot_bundle_*.txt")).read_text()
    assert "did not finish its report" in text
