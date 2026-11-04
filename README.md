# sklearn-map-lab

手元の表データで、機械学習モデルを同じ条件で比較し、採用した理由と最終評価を残すローカルWebアプリです。

**English summary.** A local web app that profiles tabular data, compares machine-learning models on shared splits, and records the selection rationale before holdout evaluation.

## 背景と担当

モデル比較のスコアに加えて、何を予測するか、どの列を使うか、どのモデルをなぜ採用したかを同じ画面で確認できるよう、Codexを活用して個人開発しました。この公開版は、既存アプリからデータ取り込みと自動分析の機能を切り出したものです。

## できること

| 工程 | 実装 |
| --- | --- |
| データ診断 | CSV、TSV、Excel、JSON、Parquetを取り込み、列の型・欠損・分布を表示 |
| 候補選定 | 分類、回帰、クラスタリング、異常検知、次元削減、等間隔時系列に応じて候補と除外理由を表示 |
| 比較 | 分類・回帰でDummy基準モデルを使い、共通の分割で交差検証。前処理は各foldの訓練部分で学習 |
| 判断の記録 | 採用・保留・却下と理由を実験に保存。分類・回帰・時系列の最終テストは採用確定後に評価 |
| 振り返り | 適用条件に合う診断グラフを生成。設定・結果をZIPに書き出し、対応モデルでは新しい行を予測 |

Reactが設定入力と結果表示を担当し、FastAPIがデータ取り込みと実験管理を担当します。学習・描画は別プロセスで実行し、実験記録はローカルの `analysis_data/` に保存します。

| ソース | 担当 |
| --- | --- |
| [frontend/src/AutoAnalysis.jsx](frontend/src/AutoAnalysis.jsx) | 取り込みから判断の保存までの操作画面 |
| [backend/analysis/api.py](backend/analysis/api.py) | データ取り込み、学習ワーカー、実験管理API |
| [backend/analysis/engine.py](backend/analysis/engine.py) | データ分割、モデル比較、採用後の最終評価 |
| [backend/analysis/graphs.py](backend/analysis/graphs.py) | 診断グラフの条件判定と描画 |

## 動作画面

合成データ120行を使った5モデルの比較と、採用理由を保存した画面です。

![合成データでのモデル比較と採用理由](docs/model-comparison.png)

## セットアップ

Python 3.12とNode.js 24で検証しました。リポジトリのルートから実行します。

```sh
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux の場合は source .venv/bin/activate
python -m pip install -r backend/requirements-analysis.txt
npm --prefix frontend ci
npm --prefix frontend run build
python examples/make_demo.py examples/demo.csv
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

ブラウザで `http://127.0.0.1:8000/` を開き、`examples/demo.csv` を取り込みます。目的変数に `target`、タスクに分類を設定し、「おすすめで比較を実行」を選びます。比較結果からモデルを選んで理由を保存し、最終テストを実行します。起動にAPIキーは不要です。保存先は環境変数 `SKLEARN_ANALYSIS_DIR` で指定できます。

## 検証

- 合成データ120行のブラウザ試験で、5モデルの比較、判断の保存、最終テスト24行の評価、別途生成した3行の予測まで確認しました。3分割交差検証の平均マクロF1はDummyが0.3333、採用したロジスティック回帰が0.9472でした。ロジスティック回帰の最終テストでのマクロF1は0.9161でした。すべてこの合成データに限った動作確認の値です。
- 公開用コピーでフロントエンドの本番ビルドと、バックエンド既存14テストの通過を確認しました。テストはCSV・Excel取り込み、学習、描画、採用後の評価、予測などをカバーします。
- バックエンドのテストを実行する場合は `python -m pip install httpx` の後、`backend/` から `python -m unittest discover -s tests -v` を実行します。

## 制約

ローカルの単一ユーザー向けです。外部公開サーバーに必要な認証やユーザー分離は備えていません。実験ZIPには取り込んだデータが含まれます。比較スコアはデータと分割条件に依存します。予測時に使えない列は利用者が除外し、グループや時系列の構造に応じて分割条件を設定します。今後の課題は、実データでの比較と分割条件ごとの検証範囲を増やすことです。
