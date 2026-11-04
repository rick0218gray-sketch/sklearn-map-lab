import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, PoissonRegressor, GammaRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, StackingClassifier, StackingRegressor, IsolationForest
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .neural import NeuralClassifier, NeuralRegressor, Reconstruction

def dense(x): return x.toarray() if hasattr(x,'toarray') else np.asarray(x)
def strings(x): return pd.Series(x).fillna('').astype(str).tolist()
def fitted_preprocessor(frame, nonnegative=False):
    num=frame.select_dtypes(include='number').columns.tolist()
    txt=[c for c in frame.columns if c not in num and frame[c].dropna().astype(str).str.len().median()>40 and frame[c].nunique()>20]
    cat=[c for c in frame.columns if c not in num+txt]
    parts=[]
    if num: parts.append(('numeric',Pipeline([('impute',SimpleImputer(keep_empty_features=True)),('scale',MinMaxScaler(clip=True) if nonnegative else StandardScaler())]),num))
    if cat: parts.append(('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False,max_categories=40))]),cat))
    for i,c in enumerate(txt): parts.append((f'text{i}',Pipeline([('str',FunctionTransformer(strings)),('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(2,3),max_features=128)),('dense',FunctionTransformer(dense))]),c))
    return ColumnTransformer(parts,verbose_feature_names_out=False)
class AdaptivePreprocessor(TransformerMixin,BaseEstimator):
    def __init__(self,nonnegative=False):self.nonnegative=nonnegative
    def fit(self,X,y=None):
        self.transformer_=fitted_preprocessor(X,self.nonnegative);self.transformer_.fit(X,y);self.n_features_in_=X.shape[1]
        self.feature_names_in_=np.asarray(X.columns,dtype=object);return self
    def transform(self,X):return self.transformer_.transform(X)
    def get_feature_names_out(self,input_features=None):return self.transformer_.get_feature_names_out(input_features)
def preprocessor(frame,nonnegative=False):return AdaptivePreprocessor(nonnegative)

def estimator(name, task, seed=42, epochs=25, period=7, image_shape=None):
    c=task=='classification'
    pair=lambda a,b: a if c else b
    factories={
    'dummy':lambda:pair(DummyClassifier(),DummyRegressor()),'linear':LinearRegression,
    'ridge':lambda:Ridge(alpha=1),'lasso':lambda:Lasso(alpha=.01,max_iter=3000),'elasticnet':lambda:ElasticNet(alpha=.01,max_iter=3000),
    'logistic':lambda:LogisticRegression(max_iter=1000,random_state=seed),'poisson':lambda:PoissonRegressor(max_iter=500),'gamma':lambda:GammaRegressor(max_iter=500),
    'tree':lambda:pair(DecisionTreeClassifier(max_depth=8,random_state=seed),DecisionTreeRegressor(max_depth=8,random_state=seed)),
    'forest':lambda:pair(RandomForestClassifier(n_estimators=60,n_jobs=1,random_state=seed),RandomForestRegressor(n_estimators=60,n_jobs=1,random_state=seed)),
    'extra':lambda:pair(ExtraTreesClassifier(n_estimators=60,n_jobs=1,random_state=seed),ExtraTreesRegressor(n_estimators=60,n_jobs=1,random_state=seed)),
    'svm':lambda:pair(SVC(probability=True,random_state=seed),SVR()),'knn':lambda:pair(KNeighborsClassifier(3),KNeighborsRegressor(3)),
    'gaussiannb':GaussianNB,'complementnb':ComplementNB,'lda':lambda:LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'),'qda':lambda:QuadraticDiscriminantAnalysis(reg_param=.1),
    'gp':lambda:pair(GaussianProcessClassifier(random_state=seed,max_iter_predict=40),GaussianProcessRegressor(alpha=.01,random_state=seed)),
    'kmeans':lambda:KMeans(3,n_init=10,random_state=seed),'hierarchical':lambda:AgglomerativeClustering(3),'dbscan':lambda:DBSCAN(eps=1,min_samples=5),'hdbscan':lambda:HDBSCAN(min_cluster_size=5),
    'gmm':lambda:GaussianMixture(3,random_state=seed),'pca':lambda:PCA(2,random_state=seed),
    'tsne':lambda:TSNE(2,perplexity=15,random_state=seed,max_iter=300),
    'isolation':lambda:IsolationForest(n_estimators=60,contamination=.05,random_state=seed,n_jobs=1),'oneclass':lambda:OneClassSVM(nu=.05),
    'autoencoder':lambda:Reconstruction(False,epochs,random_state=seed),'vae':lambda:Reconstruction(True,epochs,random_state=seed),
    }
    if name in ['mlp','cnn','rnn','lstm','gru','transformer']:
        return (NeuralClassifier if c else NeuralRegressor)(name,epochs,random_state=seed,image_shape=image_shape if name=='cnn' else None)
    if name=='xgboost':
        from xgboost import XGBClassifier,XGBRegressor
        return (XGBClassifier if c else XGBRegressor)(n_estimators=50,max_depth=4,n_jobs=1,random_state=seed,verbosity=0)
    if name=='lightgbm':
        from lightgbm import LGBMClassifier,LGBMRegressor
        return (LGBMClassifier if c else LGBMRegressor)(n_estimators=50,max_depth=5,n_jobs=1,random_state=seed,verbosity=-1)
    if name=='catboost':
        from catboost import CatBoostClassifier,CatBoostRegressor
        return (CatBoostClassifier if c else CatBoostRegressor)(iterations=50,depth=4,random_seed=seed,verbose=False,thread_count=1)
    if name=='umap':
        from umap import UMAP
        return UMAP(n_components=2,n_neighbors=10,random_state=seed,n_jobs=1)
    if name=='stacking':
        base=[('linear',LogisticRegression(max_iter=500) if c else Ridge()),('tree',DecisionTreeClassifier(max_depth=4,random_state=seed) if c else DecisionTreeRegressor(max_depth=4,random_state=seed))]
        return (StackingClassifier if c else StackingRegressor)(base,cv=3,n_jobs=1)
    return factories[name]()
def pipeline(name,task,X,seed=42,epochs=25,image_shape=None):
    return Pipeline([('pre',preprocessor(X,name=='complementnb')),('model',estimator(name,task,seed,epochs,image_shape=image_shape))])

class ForecastModel:
    def __init__(self,kind,period=7,epochs=25,seed=42): self.kind=kind;self.period=period;self.epochs=epochs;self.seed=seed
    def fit(self,y,dates):
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.y_=np.asarray(y,float);self.dates_=pd.DatetimeIndex(dates)
        if self.kind=='arima': self.model_=ARIMA(self.y_,order=(1,1,1)).fit()
        elif self.kind=='sarima': self.model_=SARIMAX(self.y_,order=(1,1,1),seasonal_order=(1,0,0,self.period),enforce_stationarity=False).fit(disp=False,maxiter=30)
        elif self.kind=='ets': self.model_=ExponentialSmoothing(self.y_,trend='add',seasonal='add' if len(y)>=2*self.period else None,seasonal_periods=self.period).fit()
        elif self.kind=='prophet':
            from prophet import Prophet
            self.model_=Prophet(stan_backend='CMDSTANPY',yearly_seasonality=False,daily_seasonality=False,weekly_seasonality=True)
            self.model_.fit(pd.DataFrame({'ds':self.dates_,'y':self.y_}))
        else:
            self.lag_=min(self.period,max(2,len(y)//4));self.scaler_=StandardScaler().fit(self.y_.reshape(-1,1))
            z=self.scaler_.transform(self.y_.reshape(-1,1)).ravel()
            X=np.array([z[i-self.lag_:i] for i in range(self.lag_,len(z))]);target=z[self.lag_:]
            self.model_=NeuralRegressor(self.kind,self.epochs,random_state=self.seed).fit(X,target)
        return self
    def predict(self,dates):
        dates=pd.DatetimeIndex(dates);n=len(dates)
        if n<1:raise ValueError('未来の日付を1行以上指定してください')
        step=self.dates_[-1]-self.dates_[-2]
        expected=pd.date_range(self.dates_[-1]+step,periods=n,freq=step)
        if not dates.equals(expected):raise ValueError('未来日付は学習終了直後から、学習データと同じ間隔で指定してください')
        if self.kind=='prophet': return self.model_.predict(pd.DataFrame({'ds':dates}))['yhat'].values
        if self.kind in ('arima','sarima','ets'): return np.asarray(self.model_.forecast(n))
        values=self.scaler_.transform(self.y_.reshape(-1,1)).ravel().tolist();out=[]
        for _ in range(n):
            p=float(self.model_.predict(np.array([values[-self.lag_:]]))[0]);values.append(p);out.append(p)
        return self.scaler_.inverse_transform(np.array(out).reshape(-1,1)).ravel()
