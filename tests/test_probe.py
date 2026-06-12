"""Probe protocol and leakage tests (subsec:probe, subsec:stage4).

Leakage: the imputer/scaler must be refit inside every CV fold (the sklearn
pipeline is cloned per fold) and the locked test split indices must never
appear in ANY fit call."""

import numpy as np
import pytest
from sklearn.impute import SimpleImputer

import energy_llm.probe as probe_mod
from energy_llm.probe import (
    Splits,
    make_locked_splits,
    make_probe_pipeline,
    run_base_and_full_probes,
    run_probe_protocol,
    select_c,
)

RNG = np.random.default_rng(42)


def make_dataset(n=120, m=6, seed=42):
    """Column 0 is a row-id tracer; the rest carry a weak signal."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    X = rng.normal(size=(n, m))
    X[:, 1] += y * 1.0  # signal
    X[:, 0] = np.arange(n)  # row-id tracer for leakage tracking
    ids = [f"s{i:04d}" for i in range(n)]
    return ids, X, y


class RecordingImputer(SimpleImputer):
    """Records the row-id tracer (column 0) of every X it is fit on."""
    fit_row_ids: list[set] = []

    def fit(self, X, y=None):
        RecordingImputer.fit_row_ids.append(set(np.asarray(X)[:, 0].astype(int)))
        return super().fit(X, y)


class TestSplits:
    def test_ratios_and_stratification(self, tmp_path):
        ids, X, y = make_dataset(n=200)
        splits = make_locked_splits(ids, y, tmp_path / "split.json", seed=42)
        assert len(splits.test_ids) == round(0.15 * 200)
        assert len(splits.val_ids) == round(0.15 * 200)
        assert len(splits.train_ids) == 200 - 2 * round(0.15 * 200)
        # disjoint cover
        all_ids = splits.train_ids + splits.val_ids + splits.test_ids
        assert sorted(all_ids) == sorted(ids)
        # approximate stratification on the test split
        pos = {sid: y[int(sid[1:])] for sid in ids}
        test_rate = np.mean([pos[s] for s in splits.test_ids])
        assert abs(test_rate - y.mean()) < 0.1

    def test_split_is_locked_and_hash_checked(self, tmp_path):
        ids, X, y = make_dataset(n=100)
        path = tmp_path / "split.json"
        s1 = make_locked_splits(ids, y, path, seed=42)
        s2 = make_locked_splits(ids, y, path, seed=99)  # seed ignored: loaded
        assert s1.test_ids == s2.test_ids
        # tampering with the file trips the hash check
        import json
        raw = json.loads(path.read_text())
        raw["test_ids"][0] = raw["train_ids"][0]
        path.write_text(json.dumps(raw))
        with pytest.raises(RuntimeError, match="hash check"):
            make_locked_splits(ids, y, path, seed=42)

    def test_calibration_ids_forced_into_train(self, tmp_path):
        ids, X, y = make_dataset(n=100)
        forced = ids[:10]
        splits = make_locked_splits(ids, y, tmp_path / "split.json",
                                    force_train_ids=forced)
        assert set(forced) <= set(splits.train_ids)
        assert not set(forced) & set(splits.val_ids)
        assert not set(forced) & set(splits.test_ids)


class TestLeakage:
    @pytest.fixture(autouse=True)
    def patched_pipeline(self, monkeypatch):
        RecordingImputer.fit_row_ids = []

        def recording_pipeline(C=1.0, seed=42):
            pipe = make_probe_pipeline(C=C, seed=seed)
            pipe.steps[0] = ("imputer", RecordingImputer(strategy="median"))
            return pipe

        monkeypatch.setattr(probe_mod, "make_probe_pipeline", recording_pipeline)
        yield

    def test_imputer_refit_per_fold_and_test_never_fit(self, tmp_path):
        ids, X, y = make_dataset(n=120)
        splits = make_locked_splits(ids, y, tmp_path / "split.json")
        idx = splits.index_of(ids)
        result = run_probe_protocol(X, y, idx, c_grid=[0.1, 1.0],
                                    inner_cv_folds=3, outer_cv_folds=5)

        fits = RecordingImputer.fit_row_ids
        # 2 C values x 3 inner folds + 5 outer folds + 1 final refit
        assert len(fits) == 2 * 3 + 5 + 1

        test_rows = set(idx["test"].tolist())
        train_rows = set(idx["train"].tolist())
        for fit_rows in fits:
            assert not fit_rows & test_rows, "test indices leaked into a fit call"
        # inner-CV fits are strict subsets of train (refit per fold, not once)
        inner_fits = fits[: 2 * 3]
        for fit_rows in inner_fits:
            assert fit_rows < train_rows
        assert result["test_auroc"] > 0.5  # signal present


class TestProtocol:
    def test_select_c_uses_train_only(self):
        ids, X, y = make_dataset(n=90)
        best_c, table = select_c(X, y, c_grid=[0.01, 1.0], n_folds=3)
        assert best_c in (0.01, 1.0)
        assert set(table) == {0.01, 1.0}

    def test_protocol_outputs(self, tmp_path):
        ids, X, y = make_dataset(n=120)
        splits = make_locked_splits(ids, y, tmp_path / "split.json")
        idx = splits.index_of(ids)
        result = run_probe_protocol(X, y, idx, c_grid=[0.1, 1.0],
                                    inner_cv_folds=3, outer_cv_folds=5)
        assert 0.0 <= result["test_auroc"] <= 1.0
        assert len(result["outer_cv_aurocs"]) == 5
        assert len(result["test_scores"]) == len(idx["test"])

    def test_nan_features_handled_by_imputer(self, tmp_path):
        ids, X, y = make_dataset(n=120)
        X = X.copy()
        X[::7, 2] = np.nan  # failed-sample layers -> NaN (median imputed)
        splits = make_locked_splits(ids, y, tmp_path / "split.json")
        result = run_probe_protocol(X, y, splits.index_of(ids), c_grid=[1.0])
        assert np.isfinite(result["test_auroc"])

    def test_base_and_full_share_protocol(self, tmp_path):
        ids, X, y = make_dataset(n=120)
        splits = make_locked_splits(ids, y, tmp_path / "split.json")
        idx = splits.index_of(ids)
        out = run_base_and_full_probes(X[:, :2], X[:, 2:], y, idx, c_grid=[1.0])
        assert set(out) == {"base", "full"}
        np.testing.assert_array_equal(out["base"]["y_test"], out["full"]["y_test"])
