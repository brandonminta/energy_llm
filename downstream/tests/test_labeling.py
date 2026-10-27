"""Labelling tests (subsec:stage3_labeling): locked prompt, verdict parsing,
resume, raw-response archiving, retry, and Cohen's kappa."""

import json

import pytest

from energy_llm.labeling import (
    JUDGE_PROMPT_SHA256,
    JUDGE_PROMPT_TEMPLATE,
    judge_human_kappa,
    label_samples,
    load_labels,
    make_human_subset_csv,
    parse_verdict,
    query_judge,
    render_judge_prompt,
)


class FakeJudge:
    """Minimal OpenAI-compatible client; verdict alternates per call."""

    def __init__(self, responses=None, fail_first_n=0):
        self.calls = []
        self.responses = responses
        self.fail_first_n = fail_first_n
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self.chat.completions.create = self._create

    def _create(self, model, messages, temperature, max_tokens):
        if len(self.calls) < self.fail_first_n:
            self.calls.append(None)
            raise ConnectionError("transient")
        self.calls.append(messages[0]["content"])
        i = len(self.calls) - 1
        text = (self.responses[i % len(self.responses)] if self.responses
                else ("SUPPORTED" if i % 2 == 0 else "UNSUPPORTED"))
        message = type("Msg", (), {"content": text})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


def write_trajectory_stub(traj_dir, sid, question="q", answer="a", failed=False):
    traj_dir.mkdir(parents=True, exist_ok=True)
    (traj_dir / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid, "question": question, "gold_answers": ["gold"],
        "generated_text": None if failed else answer, "failed": failed,
    }))


class TestVerdictParsing:
    def test_unsupported_checked_before_supported(self):
        # 'UNSUPPORTED' contains 'SUPPORTED' as a substring
        assert parse_verdict("UNSUPPORTED") == 1
        assert parse_verdict("SUPPORTED") == 0
        assert parse_verdict("  unsupported.") == 1
        assert parse_verdict("The answer is SUPPORTED") == 0
        assert parse_verdict("no idea") is None

    def test_prompt_is_locked(self):
        import hashlib
        assert hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode()).hexdigest() == JUDGE_PROMPT_SHA256

    def test_render_includes_all_fields(self):
        prompt = render_judge_prompt("Q?", "A.", ["g1", "g2"])
        assert "Q?" in prompt and "A." in prompt and "- g1" in prompt and "- g2" in prompt


class TestLabelSamples:
    def test_labels_written_with_raw_response(self, tmp_path):
        traj = tmp_path / "traj"
        labels = tmp_path / "labels"
        for i in range(4):
            write_trajectory_stub(traj, f"s{i}")
        judge = FakeJudge()
        counts = label_samples(traj, labels, judge, judge_model="fake")
        assert counts["labelled"] == 4
        record = json.loads((labels / "s0_label.json").read_text())
        assert record["judge_raw_response"] in ("SUPPORTED", "UNSUPPORTED")
        assert record["judge_prompt_sha256"] == JUDGE_PROMPT_SHA256
        assert record["is_hallucination"] in (0, 1)

    def test_resume_skips_labelled(self, tmp_path):
        traj = tmp_path / "traj"
        labels = tmp_path / "labels"
        for i in range(4):
            write_trajectory_stub(traj, f"s{i}")
        judge = FakeJudge()
        label_samples(traj, labels, judge)
        n_calls = len(judge.calls)
        counts = label_samples(traj, labels, judge)  # rerun: no-op
        assert counts["labelled"] == 0
        assert len(judge.calls) == n_calls

    def test_failed_trajectories_skipped(self, tmp_path):
        traj = tmp_path / "traj"
        labels = tmp_path / "labels"
        write_trajectory_stub(traj, "ok")
        write_trajectory_stub(traj, "bad", failed=True)
        counts = label_samples(traj, labels, FakeJudge())
        assert counts["labelled"] == 1
        assert counts["skipped_failed"] == 1

    def test_retry_with_backoff(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda s: None)
        judge = FakeJudge(fail_first_n=2)
        raw = query_judge(judge, "fake", "prompt", max_retries=5)
        assert raw in ("SUPPORTED", "UNSUPPORTED")
        assert len(judge.calls) == 3  # 2 failures + 1 success

    def test_unparseable_recorded_as_none(self, tmp_path):
        traj = tmp_path / "traj"
        labels = tmp_path / "labels"
        write_trajectory_stub(traj, "s0")
        counts = label_samples(traj, labels, FakeJudge(responses=["gibberish"]))
        assert counts["unparseable"] == 1
        assert load_labels(labels) == {}  # None labels excluded


class TestKappa:
    def test_perfect_agreement(self, tmp_path):
        traj = tmp_path / "traj"
        labels = tmp_path / "labels"
        for i in range(10):
            write_trajectory_stub(traj, f"s{i}")
        label_samples(traj, labels, FakeJudge())
        judge_labels = load_labels(labels)

        csv_path = tmp_path / "human.csv"
        lines = ["sample_id,human_label"] + [
            f"{sid},{lab}" for sid, lab in sorted(judge_labels.items())
        ]
        csv_path.write_text("\n".join(lines))
        out = judge_human_kappa(labels, csv_path)
        assert out["kappa"] == pytest.approx(1.0)
        assert out["passes_threshold"]
        assert out["n_subset"] == 10

    def test_human_subset_csv_template(self, tmp_path):
        traj = tmp_path / "traj"
        labels = tmp_path / "labels"
        for i in range(10):
            write_trajectory_stub(traj, f"s{i}")
        label_samples(traj, labels, FakeJudge())
        csv_path = make_human_subset_csv(labels, tmp_path / "human.csv", n=6)
        lines = csv_path.read_text().strip().splitlines()
        assert lines[0] == "sample_id,human_label"
        assert len(lines) == 7
