"""95グラフ・30系統と、実際に学習する推定器の対応。"""
GRAPH_GROUPS = {
"EDA": ["Histogram","KDE / Density Plot","ECDF","Box Plot","Violin Plot","Strip / Swarm Plot","Count / Bar Plot","Scatter Plot","Bubble Plot","Hexbin / 2D Histogram","Pair Plot","Joint Plot","Correlation Heatmap","Covariance Heatmap","Missing Value Matrix / Heatmap","Q-Q Plot","P-P Plot","Parallel Coordinates","Target Distribution"],
"回帰": ["Residual Plot","Residual Histogram / KDE","Residual Q-Q Plot","Scale-Location Plot","Leverage / Cook's Distance","Actual vs Predicted","Predicted vs Residual","Prediction Interval Plot","Residual ACF","Residual vs Feature"],
"分類": ["ROC Curve","Precision-Recall Curve","Confusion Matrix","Calibration / Reliability Curve","Lift Curve","Cumulative Gains Curve","KS Curve","Decision Boundary","Decision Score / Margin Distribution","Threshold Metrics Curve","DET Curve","Class Probability Distribution"],
"説明": ["Feature Importance","Permutation Importance","SHAP Summary Plot","SHAP Bar Plot","SHAP Dependence Plot","SHAP Waterfall / Force","Partial Dependence Plot (PDP)","ICE Plot","ALE Plot","Counterfactual / What-if Plot"],
"クラスタリング": ["Elbow Plot","Silhouette Plot","Dendrogram","Cluster Scatter Plot","k-distance Plot","Cluster Size Bar Plot"],
"次元削減": ["Scree Plot","Cumulative Explained Variance","PCA Biplot","Loadings Plot","t-SNE Scatter","UMAP Scatter"],
"異常検知": ["Anomaly Score Distribution","Outlier Scatter / Timeline","Reconstruction Error Distribution","Precision@k / Recall@k"],
"学習過程": ["Loss Curve","Metric Curve","Train vs Validation Curve","Learning Rate Curve / Schedule","Gradient Norm Plot","Weight Distribution","Activation Distribution"],
"深層モデル説明": ["Attention Heatmap","Grad-CAM / Saliency Map","Embedding Visualization"],
"時系列": ["Time Series Plot","Rolling Mean / Std","Seasonal Decomposition / STL","ACF Plot","PACF Plot","Lag Plot","Forecast vs Actual","Forecast Residual Plot","Cross-Correlation / CCF","Periodogram / FFT Spectrum","Seasonal Box Plot","Backtest / Rolling-origin Plot"],
"モデル選択": ["Learning Curve","Validation Curve","CV Score Distribution","Hyperparameter Heatmap","Model Comparison Bar / Box Plot","Regularization Path / Coefficient Path"],
}
GRAPHS = [{"id": f"g{i+1:02}", "category": category, "name": name} for i,(category,name) in enumerate((c,n) for c,ns in GRAPH_GROUPS.items() for n in ns)]
FAMILIES = ["Linear Regression","Ridge / Lasso / ElasticNet","Logistic Regression","GLM (Poisson / Gamma)","Decision Tree","Random Forest","Extra Trees","XGBoost / LightGBM / CatBoost","SVM / SVR","KNN","Naive Bayes","LDA / QDA","Gaussian Process","MLP","CNN","RNN / LSTM / GRU","Transformer","k-means","Hierarchical Clustering","DBSCAN / HDBSCAN","Gaussian Mixture Model","PCA","t-SNE / UMAP","Isolation Forest","One-Class SVM","Autoencoder / VAE","ARIMA / SARIMA","ETS / Exponential Smoothing","Prophet","Ensemble / Stacking"]
SPECS = []
def add(ids, family, tasks, reason):
    for name in ids.split(','):
        SPECS.append(dict(id=name,family=FAMILIES[family-1],tasks=tasks.split(','),reason=reason))
add('dummy',1,'classification,regression','単純な基準。モデルが改善したかを比較')
add('linear',1,'regression','線形関係・説明性を重視')
add('ridge,lasso,elasticnet',2,'regression','正則化。相関した特徴や不要な特徴に対応')
add('logistic',3,'classification','確率を出す線形分類の基準')
add('poisson,gamma',4,'regression','非負の件数／正の連続値。目的変数の条件を検査')
for names,fam,reason in [('tree',5,'非線形な分岐。解釈しやすい'),('forest',6,'表データの非線形性・相互作用'),('extra',7,'ランダム化した木の集合'),('xgboost,lightgbm,catboost',8,'勾配ブースティングで表の非線形性を比較'),('svm',9,'マージン・カーネル。大規模データは時間制限'),('knn',10,'近いデータに基づく予測。スケーリングが必要'),('gp',13,'小規模データと不確実性'),('mlp',14,'多層ニューラルネット'),('cnn',15,'列順が意味を持つ1D信号／画像ピクセル'),('rnn,lstm,gru',16,'順序を持つ系列を処理。表では列順の解釈に注意'),('transformer',17,'特徴／時点間のattention。学習量を要する'),('stacking',30,'複数モデルの検証予測を組み合わせる')]:
    add(names,fam,'classification,regression',reason)
add('gaussiannb,complementnb',11,'classification','分布／非負のテキスト頻度に基づく分類')
add('lda,qda',12,'classification','クラスごとの分布を仮定。件数と次元に注意')
add('kmeans',18,'clustering','球状のクラスタ。k候補を比較')
add('hierarchical',19,'clustering','階層構造と結合距離を表示')
add('dbscan,hdbscan',20,'clustering','密度によるクラスタとノイズ')
add('gmm',21,'clustering','確率的なクラスタ割当て')
add('pca',22,'reduction','分散を保持する線形次元削減')
add('tsne,umap',23,'reduction','局所構造の可視化。予測精度のランキングとは別')
add('isolation',24,'anomaly','孤立しやすさに基づく異常検知')
add('oneclass',25,'anomaly','正常領域の境界を推定')
add('autoencoder,vae',26,'anomaly','実際の再構成誤差を異常スコアに使用')
add('arima,sarima',27,'timeseries','時間順に学習。ARIMA／季節ARIMA')
add('ets',28,'timeseries','指数平滑による水準・傾向・季節')
add('prophet',29,'timeseries','日付とトレンド・季節性を使う予測')
add('rnn,lstm,gru',16,'timeseries','過去の値だけを使うラグ系列予測')
assert len(GRAPHS) == 95
assert len(FAMILIES) == 30

add('transformer',17,'timeseries','過去の値だけを使うラグ系列予測')
