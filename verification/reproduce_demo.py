"""READMEの合成データ結果をAPI経由で再現する。

examples/make_demo.py と同じ120行を取り込み、おすすめ5モデルの比較、判断の保存、
最終テスト、別の合成3行の予測までを一時フォルダで実行する。実データは使わない。

    python verification/reproduce_demo.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "backend"))
COLUMNS = ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"]


def demo_frame():
    features, target = make_classification(
        n_samples=120, n_features=5, n_informative=3, random_state=42
    )
    frame = pd.DataFrame(features, columns=COLUMNS)
    frame["target"] = target
    return frame


def new_rows():
    features, _ = make_classification(
        n_samples=3, n_features=5, n_informative=3, random_state=7
    )
    return pd.DataFrame(features, columns=COLUMNS).to_dict("records")


def wait_for_run(client, run_id):
    for _ in range(600):
        run = client.get(f"/api/analysis/runs/{run_id}").json()
        if run["status"] not in ("queued", "running"):
            return run
        time.sleep(0.5)
    raise TimeoutError("比較が300秒以内に終わりませんでした")


def wait_for_workers(api):
    for _ in range(100):
        with api.job_lock:
            if not api.jobs:
                return
        time.sleep(0.1)


def main():
    with tempfile.TemporaryDirectory(prefix="sklearn_map_lab_demo_") as directory:
        os.environ["SKLEARN_ANALYSIS_DIR"] = directory
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from analysis import api

        app = FastAPI()
        app.include_router(api.router)
        client = TestClient(app)

        csv = demo_frame().to_csv(index=False).encode()
        data = client.post(
            "/api/analysis/upload", files={"file": ("demo.csv", csv, "text/csv")}
        ).json()
        config = {
            "data_id": data["id"],
            "task": "classification",
            "target": "target",
            "features": COLUMNS,
            "metric": "f1_macro",
            "cv": 3,
            "seed": 42,
            "test_size": 0.2,
        }
        models = client.post("/api/analysis/recommend", json=config).json()
        config["models"] = [model["id"] for model in models if model["recommended"]]
        run_id = client.post("/api/analysis/runs", json=config).json()["id"]
        run = wait_for_run(client, run_id)
        if run["status"] != "completed":
            raise RuntimeError(f"比較に失敗しました: {run.get('error')}")

        print(f"rows: {data['rows']}")
        for row in run["rows"]:
            print(f"cv_f1_macro {row['model']}: {row['mean']:.4f} ({row['status']})")

        client.post(
            f"/api/analysis/runs/{run_id}/decision",
            json={
                "model": "logistic",
                "status": "adopted",
                "reason": "共通の分割で評価し、交差検証の結果を確認して採用。",
            },
        ).raise_for_status()
        final = client.post(f"/api/analysis/runs/{run_id}/final", json={})
        final.raise_for_status()
        final = final.json()
        print(f"final_test_rows: {final['test_rows']}")
        print(f"final_f1_macro logistic: {final['metrics']['f1_macro']:.4f}")

        changed = client.post(
            f"/api/analysis/runs/{run_id}/decision",
            json={"model": "dummy", "status": "adopted", "reason": "評価後の変更"},
        )
        print(f"decision_after_final_status: {changed.status_code}")

        predicted = client.post(
            f"/api/analysis/runs/{run_id}/predict", json={"rows": new_rows()}
        )
        predicted.raise_for_status()
        print(f"predictions: {predicted.json()['predictions']}")
        wait_for_workers(api)


if __name__ == "__main__":
    main()
