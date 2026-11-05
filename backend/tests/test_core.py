"""分割・指標・停止状態・予測入力のコアロジックを確かめる小さなテスト。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

from analysis import api
from analysis.engine import (
    aligned_prob,
    metrics,
    normalize_features,
    prepare,
    profile,
    read_json,
    resolve_metric,
    run_experiment,
    write_json,
)


def classification_frame(n=100, seed=0):
    x, y = make_classification(
        n_samples=n, n_features=4, n_informative=3, n_redundant=0, random_state=seed
    )
    frame = pd.DataFrame(x, columns=list("abcd"))
    frame["target"] = y
    return frame


def cfg(**kw):
    return {
        "task": "classification",
        "target": "target",
        "models": ["logistic"],
        "cv": 3,
        "seed": 42,
        "test_size": 0.2,
        "epochs": 2,
        "split": "random",
        **kw,
    }


class SplitTest(unittest.TestCase):
    def assert_disjoint_and_complete(self, n, dev, hold, cv):
        self.assertEqual(len(np.intersect1d(dev, hold)), 0)
        self.assertEqual(sorted(np.r_[dev, hold].tolist()), list(range(n)))
        for tr, va in cv:
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            # CVの添字は開発データ内の位置で、最終テストを含まない。
            self.assertLess(max(tr.max(), va.max()), len(dev))

    def test_random_split_keeps_holdout_out_of_cv(self):
        frame = classification_frame()
        df, X, y, labels, dev, hold, cv = prepare(frame, cfg())
        self.assert_disjoint_and_complete(len(df), dev, hold, cv)
        self.assertEqual(len(hold), 20)
        self.assertEqual(labels, ["0", "1"])
        self.assertNotIn("target", X.columns)
        # 層化分割なので最終テストにも両クラスが残る。
        self.assertEqual(set(y[hold]), {0, 1})

    def test_group_split_never_shares_groups(self):
        frame = classification_frame(120)
        frame["person"] = np.repeat(np.arange(12), 10)
        df, X, y, labels, dev, hold, cv = prepare(frame, cfg(group="person"))
        self.assertNotIn("person", X.columns)
        groups = df["person"].to_numpy()
        self.assertFalse(set(groups[dev]) & set(groups[hold]))
        dev_groups = groups[dev]
        for tr, va in cv:
            self.assertFalse(set(dev_groups[tr]) & set(dev_groups[va]))

    def test_time_split_puts_holdout_after_development(self):
        frame = classification_frame()
        frame["date"] = pd.date_range("2021-01-01", periods=len(frame))
        frame = frame.sample(frac=1, random_state=1).reset_index(drop=True)
        frame["target"] = np.arange(len(frame)) % 2
        df, X, y, labels, dev, hold, cv = prepare(frame, cfg(split="time", date="date"))
        dates = pd.to_datetime(df["date"]).to_numpy()
        self.assertTrue((np.diff(dates).astype(np.int64) > 0).all())
        self.assertLess(dates[dev].max(), dates[hold].min())
        dev_dates = dates[dev]
        for tr, va in cv:
            self.assertLess(dev_dates[tr].max(), dev_dates[va].min())
        # 並べ替え後も行とラベルの対応が崩れない。
        original = frame.set_index("date")["target"]
        self.assertTrue((original.loc[df["date"]].to_numpy() == y).all())

    def test_rejects_too_few_rows_and_bad_test_size(self):
        with self.assertRaisesRegex(ValueError, "20行以上"):
            prepare(classification_frame(19), cfg())
        with self.assertRaisesRegex(ValueError, "最終テスト割合"):
            prepare(classification_frame(), cfg(test_size=0.5))

    def test_rejects_target_in_features(self):
        with self.assertRaisesRegex(ValueError, "説明変数"):
            prepare(classification_frame(), cfg(features=["a", "target"]))


class MetricTest(unittest.TestCase):
    def test_regression_metrics_match_sklearn(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        p = np.array([1.5, 2.0, 2.0, 5.0])
        out = metrics("regression", y, p)
        self.assertAlmostEqual(out["mae"], mean_absolute_error(y, p))
        self.assertAlmostEqual(out["rmse"], mean_squared_error(y, p) ** 0.5)
        self.assertAlmostEqual(out["r2"], r2_score(y, p))

    def test_classification_metrics_and_auc(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        p = np.array([0, 1, 1, 1, 2, 0])
        prob = np.eye(3)[[0, 1, 1, 1, 2, 0]] * 0.7 + 0.1
        out = metrics("classification", y, p, prob)
        self.assertAlmostEqual(out["f1_macro"], f1_score(y, p, average="macro"))
        self.assertIn("roc_auc", out)
        binary = metrics("classification", np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]))
        self.assertAlmostEqual(binary["roc_auc"], 1.0)

    def test_aligned_prob_places_columns_by_class(self):
        class Model:
            classes_ = np.array([0, 2])

            def predict_proba(self, X):
                return np.array([[0.25, 0.75]] * len(X))

        out = aligned_prob(Model(), np.zeros((2, 1)), 3)
        np.testing.assert_allclose(out, [[0.25, 0, 0.75]] * 2)

    def test_resolve_metric(self):
        self.assertEqual(resolve_metric({"task": "classification"}), "f1_macro")
        self.assertEqual(resolve_metric({"task": "timeseries", "metric": "rmse"}), "rmse")
        # 教師なしでは画面に残った教師あり指標を無視する（向きを誤らないため）。
        self.assertEqual(resolve_metric({"task": "clustering", "metric": "mae"}), "silhouette")
        self.assertEqual(resolve_metric({"task": "anomaly", "metric": "f1_macro"}), "detected")
        with self.assertRaisesRegex(ValueError, "分類用"):
            resolve_metric({"task": "classification", "metric": "mae"})
        with self.assertRaisesRegex(ValueError, "回帰用"):
            resolve_metric({"task": "regression", "metric": "accuracy"})


class ProfileTest(unittest.TestCase):
    def test_suggests_target_and_task(self):
        frame = classification_frame()
        p = profile(frame)
        self.assertEqual(p["suggested_target"], "target")
        self.assertEqual(p["suggested_task"], "classification")
        frame["target"] = np.linspace(0, 1, len(frame))
        self.assertEqual(profile(frame)["suggested_task"], "regression")

    def test_normalize_features_unifies_missing_values(self):
        frame = pd.DataFrame({"n": [1.0, np.inf, None], "c": ["x", None, pd.NA]})
        out = normalize_features(frame)
        self.assertTrue(np.isnan(out.loc[1, "n"]))
        self.assertIsInstance(out.loc[1, "c"], float)
        self.assertIsInstance(out.loc[2, "c"], float)


class ApiStateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="analysis_core_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        saved = api.ROOT
        api.ROOT = self.root
        self.addCleanup(setattr, api, "ROOT", saved)
        app = FastAPI()
        app.include_router(api.router)
        self.client = TestClient(app)

    def run_dir(self, frame, **kw):
        d = self.root / "runs" / ("b" * 32)
        d.mkdir(parents=True)
        frame.to_parquet(d / "data.parquet", index=False)
        write_json(d / "config.json", cfg(**kw))
        return d

    def upload(self, frame):
        r = self.client.post(
            "/api/analysis/upload",
            files={"file": ("data.csv", frame.to_csv(index=False).encode(), "text/csv")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["id"]

    def test_invalid_metric_is_rejected_before_run_is_created(self):
        data_id = self.upload(classification_frame())
        r = self.client.post("/api/analysis/runs", json=dict(cfg(metric="mae"), data_id=data_id))
        self.assertEqual(r.status_code, 400)
        self.assertIn("分類用", r.json()["detail"])
        self.assertFalse((self.root / "runs").exists())

    def test_clustering_ignores_stale_supervised_metric(self):
        frame = classification_frame().drop(columns="target")
        d = self.run_dir(frame, task="clustering", target=None, metric="mae", models=["kmeans", "gmm"])
        run_experiment(d)
        result = read_json(d / "result.json")
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual((result["metric"], result["direction"]), ("silhouette", "maximize"))
        best = max((r for r in result["rows"] if r["mean"] is not None), key=lambda r: r["mean"])
        self.assertEqual(result["suggested_model"], best["model"])

    def test_cancelled_run_leaves_no_running_rows(self):
        d = self.run_dir(classification_frame(), models=["logistic", "forest"])
        (d / "cancel").touch()
        run_experiment(d)
        result = read_json(d / "result.json")
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse(any(r["status"] == "running" for r in result["rows"]))

    def test_cancel_marks_running_rows_and_keeps_finished_runs(self):
        d = self.run_dir(classification_frame())
        write_json(d / "result.json", {"id": d.name, "status": "running", "rows": [{"model": "dummy", "status": "completed"}, {"model": "logistic", "status": "running"}]})
        r = self.client.post(f"/api/analysis/runs/{d.name}/cancel")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "cancelled")
        self.assertEqual([row["status"] for row in r.json()["rows"]], ["completed", "cancelled"])

        write_json(d / "result.json", {"id": d.name, "status": "completed", "rows": []})
        (d / "cancel").unlink()
        r = self.client.post(f"/api/analysis/runs/{d.name}/cancel")
        self.assertEqual(r.json()["status"], "completed")
        self.assertFalse((d / "cancel").exists())

    def test_predict_accepts_missing_category_like_training(self):
        frame = classification_frame()
        frame["color"] = np.where(frame["a"] > 0, "red", "blue")
        frame.loc[::7, "color"] = None
        d = self.run_dir(frame, models=["logistic"])
        run_experiment(d)
        self.assertEqual(read_json(d / "result.json")["status"], "completed")
        from analysis.engine import finalize

        finalize(d, "logistic", "テスト")
        head = frame.drop(columns="target").head(3)
        rows = head.astype(object).where(head.notna(), None).to_dict("records")
        rows[0]["color"] = None
        r = self.client.post(f"/api/analysis/runs/{d.name}/predict", json={"rows": rows})
        self.assertEqual(r.status_code, 200, r.text)
        # JSONのnullは学習時の欠測（np.nan）と同じく補完される。未知カテゴリ扱い（全0）にならない。
        import joblib

        model = joblib.load(d / "adopted.joblib")
        features = read_json(d / "metadata.json")["features"]
        as_nan = head[features].copy()
        as_nan.loc[as_nan.index[0], "color"] = np.nan
        as_none = pd.DataFrame(rows)[features]
        pre = model.named_steps["pre"]
        np.testing.assert_allclose(pre.transform(normalize_features(as_none)), pre.transform(as_nan))
        self.assertEqual(r.json()["predictions"], [str(v) for v in model.predict(as_nan)])


if __name__ == "__main__":
    unittest.main()
