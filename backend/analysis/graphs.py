"""添付表95項目の実計算・描画。条件不適合を代理グラフで埋めない。"""
from pathlib import Path
import json, threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_names={f.name for f in font_manager.fontManager.ttflist}
for preferred in ("Meiryo","Yu Gothic","MS Gothic","Noto Sans CJK JP"):
    if preferred in font_names:plt.rcParams["font.family"]=preferred;break
plt.rcParams["axes.unicode_minus"]=False
from scipy import stats, signal
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import *
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.model_selection import learning_curve, validation_curve, ParameterGrid, train_test_split
from .catalog import GRAPHS
from .engine import clean, write_json
LOCK=threading.RLock()
class NotApplicable(ValueError): pass

def need(condition,reason):
    if not condition:raise NotApplicable(reason)
def native(context):
    model=context['model'];return model.named_steps['model'] if hasattr(model,'named_steps') else getattr(model,'model_',model)
def projection(z):
    z=np.asarray(z,float);need(z.shape[1]>=2 and len(z)>=3,'数値化後2特徴以上・3行以上必要')
    return PCA(2,random_state=42).fit_transform(z)

def draw(context,graph_id,directory):
    graph=next((g for g in GRAPHS if g['id']==graph_id),None)
    if graph is None:raise ValueError('未登録グラフ')
    directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
    with LOCK:
        try:
            plt.close('all');fig,ax=plt.subplots(figsize=(9,5));note=_draw(context,graph['name'],ax)
            fig=plt.gcf();fig.suptitle(graph['name'],fontsize=14);fig.tight_layout();p=directory/f'{graph_id}.png';fig.savefig(p,dpi=120,bbox_inches='tight');plt.close('all')
            output=dict(graph,status='rendered',file=p.name,note=note or '',scope='development / out-of-fold; final holdout excluded')
        except NotApplicable as e:plt.close('all');output=dict(graph,status='not_applicable',reason=str(e))
        except Exception as e:plt.close('all');output=dict(graph,status='failed',reason=f'{type(e).__name__}: {e}')
        write_json(directory/f'{graph_id}.json',output);return output

def _draw(c,n,ax):
    import seaborn as sns
    from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
    X=c['X'];task=c['task'];model=c['model'];m=native(c);cfg=c['config'];z=c.get('Z');y=c.get('y');p=c.get('pred');prob=c.get('prob')
    num=X.select_dtypes('number').replace([np.inf,-np.inf],np.nan);cols=num.columns.tolist()
    vals=num.iloc[:,0].dropna().to_numpy() if cols else np.array([])
    rawz=np.asarray(z,float) if z is not None else None
    if n in [g['name'] for g in GRAPHS if g['category']=='EDA']:
        if n in ('Count / Bar Plot','Target Distribution'):
            if n=='Target Distribution':need(y is not None,'目的変数が必要')
            s=pd.Series(y) if n=='Target Distribution' else X.iloc[:,0]
            need(s is not None,'目的変数が必要');vc=s.value_counts().head(20);ax.bar(vc.index.astype(str),vc.values);ax.tick_params(axis='x',rotation=45);return '開発データのみ'
        if n=='Missing Value Matrix / Heatmap':sns.heatmap(X.isna().iloc[:200,:30],cbar=False,ax=ax);return
        need(len(vals)>2,'数値列に有効な値が3件以上必要')
        if n=='Histogram':ax.hist(vals,bins='auto')
        elif n=='KDE / Density Plot':need(np.std(vals)>0,'定数列では密度を推定できません');sns.kdeplot(vals,ax=ax)
        elif n=='ECDF':a=np.sort(vals);ax.step(a,np.arange(1,len(a)+1)/len(a),where='post')
        elif n=='Box Plot':sns.boxplot(data=num.iloc[:,:12],ax=ax);ax.tick_params(axis='x',rotation=45)
        elif n=='Violin Plot':sns.violinplot(data=num.iloc[:,:8],ax=ax);ax.tick_params(axis='x',rotation=45)
        elif n=='Strip / Swarm Plot':sns.swarmplot(data=num.iloc[:min(200,len(num)),:5],size=3,ax=ax)
        elif n in ('Scatter Plot','Bubble Plot','Hexbin / 2D Histogram','Pair Plot','Joint Plot','Correlation Heatmap','Covariance Heatmap','Parallel Coordinates'):
            need(len(cols)>=2,'数値列が2列以上必要');a=num.iloc[:,0];b=num.iloc[:,1]
            if n=='Scatter Plot':ax.scatter(a,b,s=12,alpha=.6)
            elif n=='Bubble Plot':need(len(cols)>=3,'サイズに使う第3数値列が必要');size=num.iloc[:,2].abs().fillna(0);ax.scatter(a,b,s=10+100*size/(size.max()+1e-9),alpha=.5)
            elif n=='Hexbin / 2D Histogram':ax.hexbin(a,b,gridsize=25,mincnt=1);ax.set(xlabel=cols[0],ylabel=cols[1])
            elif n=='Pair Plot':plt.close(ax.figure);sns.pairplot(num.iloc[:500,:4].dropna())
            elif n=='Joint Plot':plt.close(ax.figure);sns.jointplot(x=a,y=b,kind='scatter')
            elif n=='Correlation Heatmap':sns.heatmap(num.iloc[:,:15].corr(),cmap='coolwarm',center=0,ax=ax)
            elif n=='Covariance Heatmap':sns.heatmap(num.iloc[:,:15].cov(),cmap='coolwarm',center=0,ax=ax)
            else:
                df=num.iloc[:100,:8].fillna(0).copy();df=(df-df.mean())/df.std().replace(0,1);df['group']=pd.Series(y[:len(df)]).astype(str).values if y is not None else 'all';pd.plotting.parallel_coordinates(df,'group',ax=ax,alpha=.3)
        elif n=='Q-Q Plot':stats.probplot(vals,plot=ax)
        elif n=='P-P Plot':need(np.std(vals)>0,'定数列では分布比較不可');a=np.sort(vals);ax.plot(stats.norm.cdf(a,loc=np.mean(a),scale=np.std(a)),np.arange(1,len(a)+1)/len(a));ax.plot([0,1],[0,1],'--')
        return '数値列は先頭から表示。大規模データの一部グラフはサンプル表示'
    if n in [g['name'] for g in GRAPHS if g['category']=='回帰']:
        need(task in ('regression','timeseries'),'回帰／時系列の実測値と評価予測が必要')
        mask=np.isfinite(p);actual=y[mask];pred=p[mask];res=actual-pred
        if n=='Residual Plot':ax.scatter(pred,res,s=12);ax.axhline(0,color='red')
        elif n=='Residual Histogram / KDE':sns.histplot(res,kde=np.std(res)>0,ax=ax)
        elif n=='Residual Q-Q Plot':stats.probplot(res,plot=ax)
        elif n=='Scale-Location Plot':ax.scatter(pred,np.sqrt(np.abs((res-res.mean())/(res.std()+1e-12))),s=12)
        elif n=='Leverage / Cook\'s Distance':
            need(cfg.get('selected_model')=='linear','Cook距離はこの実装ではOLS Linear Regressionに限定')
            zz=np.c_[np.ones(len(rawz)),rawz];hat=np.einsum('ij,ji->i',zz,np.linalg.pinv(zz));rr=y-model.predict(X);mse=np.sum(rr**2)/max(1,len(rr)-zz.shape[1]);cook=rr**2/(max(1,zz.shape[1])*mse)*(hat/(1-hat+1e-12)**2);ax.scatter(hat,cook,s=12);ax.set(xlabel='Leverage',ylabel="Cook's distance")
        elif n=='Actual vs Predicted':ax.scatter(actual,pred,s=12);lo=min(actual.min(),pred.min());hi=max(actual.max(),pred.max());ax.plot([lo,hi],[lo,hi],'--')
        elif n=='Predicted vs Residual':ax.scatter(pred,res,s=12);ax.axhline(0,color='red')
        elif n=='Prediction Interval Plot':
            need(task=='regression' and not cfg.get('group') and cfg.get('split')!='time','split conformalは独立した回帰データに限定。時系列・グループには別方式が必要')
            ids=np.arange(len(X));tr,other=train_test_split(ids,test_size=.4,random_state=43);cal,ev=train_test_split(other,test_size=.5,random_state=43)
            mm=clone(model).fit(X.iloc[tr],y[tr]);errors=np.abs(y[cal]-mm.predict(X.iloc[cal]));q=np.quantile(errors,min(1,np.ceil((len(cal)+1)*.9)/len(cal)),method='higher');pp=mm.predict(X.iloc[ev]);ii=np.argsort(pp);ax.errorbar(np.arange(len(ev)),pp[ii],yerr=q,fmt='.',alpha=.7);ax.scatter(np.arange(len(ev)),y[ev][ii],s=8);return f'Split conformal nominal 90%; calibration={len(cal)}, evaluation coverage={np.mean(abs(y[ev]-pp)<=q):.3f}'
        elif n=='Residual ACF':plot_acf(res,lags=min(20,len(res)//2-1),ax=ax)
        elif n=='Residual vs Feature':need(cols,'数値特徴が必要');ax.scatter(num.loc[mask,cols[0]],res,s=12);ax.set_xlabel(cols[0])
        return '残差＝実測－予測。交差検証で未学習の行に出した予測を使用'
    if n in [g['name'] for g in GRAPHS if g['category']=='分類']:
        need(task=='classification','分類の評価予測が必要');mask=np.isfinite(p);yy=y[mask].astype(int);pp=p[mask].astype(int);binary=len(np.unique(y))==2
        if n=='Confusion Matrix':ConfusionMatrixDisplay.from_predictions(yy,pp,ax=ax);return '交差検証予測の混同行列'
        if n=='Decision Boundary':
            need(len(cols)>=2,'2数値特徴が必要');first,second=cols[:2];a=np.linspace(num[first].quantile(.01),num[first].quantile(.99),35);b=np.linspace(num[second].quantile(.01),num[second].quantile(.99),35);xx,yygrid=np.meshgrid(a,b)
            base=X.iloc[[0]*xx.size].reset_index(drop=True).copy()
            for col in X:
                base[col]=X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else X[col].mode().iloc[0]
            base[first]=xx.ravel();base[second]=yygrid.ravel();zz=model.predict(base).reshape(xx.shape);ax.contourf(xx,yygrid,zz,alpha=.3);ax.scatter(num[first],num[second],c=y,s=12);return '先頭2数値特徴を変化。他特徴は中央値／最頻値に固定した断面'
        need(prob is not None,'確率／連続スコアを提供するモデルが必要');pr=prob[mask]
        if n=='Class Probability Distribution':
            for j in range(pr.shape[1]):sns.histplot(pr[:,j],label=str(j),element='step',fill=False,ax=ax)
            ax.legend();return '交差検証で算出したクラス確率'
        if not binary and n in ('ROC Curve','Precision-Recall Curve'):
            for j in range(pr.shape[1]):
                curve=roc_curve((yy==j).astype(int),pr[:,j]) if n=='ROC Curve' else precision_recall_curve((yy==j).astype(int),pr[:,j]);ax.plot(curve[0] if n=='ROC Curve' else curve[1],curve[1] if n=='ROC Curve' else curve[0],label=str(j))
            ax.legend();return 'One-vs-rest'
        need(binary,'このグラフは2クラス分類用。多クラスにはROC／PRのOne-vs-restを使用');positive=int(cfg.get('positive_class',1));yy=(yy==positive).astype(int);s=pr[:,positive]
        if n=='ROC Curve':a,b,_=roc_curve(yy,s);ax.plot(a,b);ax.set(xlabel='FPR',ylabel='TPR')
        elif n=='Precision-Recall Curve':a,b,_=precision_recall_curve(yy,s);ax.plot(b,a);ax.set(xlabel='Recall',ylabel='Precision')
        elif n=='Calibration / Reliability Curve':a,b=calibration_curve(yy,s,n_bins=8);ax.plot(b,a,'o-');ax.plot([0,1],[0,1],'--')
        elif n in ('Lift Curve','Cumulative Gains Curve','KS Curve'):
            order=np.argsort(-s);ys=yy[order];fraction=np.arange(1,len(ys)+1)/len(ys);gains=np.cumsum(ys)/max(1,ys.sum());neg=np.cumsum(1-ys)/max(1,(1-ys).sum())
            if n=='Lift Curve':ax.plot(fraction,gains/fraction);ax.axhline(1,color='red')
            elif n=='Cumulative Gains Curve':ax.plot(fraction,gains);ax.plot([0,1],[0,1],'--')
            else:ax.plot(fraction,gains,label='positive');ax.plot(fraction,neg,label='negative');ax.set_title(f'KS={max(abs(gains-neg)):.3f}');ax.legend()
        elif n=='Decision Score / Margin Distribution':
            margin=np.log(np.clip(s,1e-8,1-1e-8)/np.clip(1-s,1e-8,1));sns.histplot(x=margin,hue=yy,element='step',ax=ax);return '交差検証確率のlog-odds。SVMの生マージンではない'
        elif n=='Threshold Metrics Curve':
            t=np.linspace(0,1,51)
            for label,fun in [('Precision',precision_score),('Recall',recall_score),('F1',f1_score)]:ax.plot(t,[fun(yy,s>=v,zero_division=0) for v in t],label=label)
            ax.legend()
        elif n=='DET Curve':a,b,_=det_curve(yy,s);ax.plot(stats.norm.ppf(np.clip(a,.001,.999)),stats.norm.ppf(np.clip(b,.001,.999)));ax.set(xlabel='normal quantile(FPR)',ylabel='normal quantile(FNR)')
        return f'交差検証予測。positive class={positive}'
    if n in [g['name'] for g in GRAPHS if g['category']=='説明']:
        need(task in ('classification','regression'),'教師ありモデルの説明用');need(rawz is not None,'変換済み特徴が必要')
        features=[str(i) for i in range(rawz.shape[1])]
        try:features=model.named_steps['pre'].get_feature_names_out().tolist()
        except Exception:pass
        if n=='Feature Importance':
            importance=getattr(m,'feature_importances_',None)
            if importance is None and hasattr(m,'coef_'):importance=np.abs(m.coef_).reshape(-1,rawz.shape[1]).mean(0)
            need(importance is not None,'このモデルにはネイティブ重要度／係数がありません。Permutation Importanceを使用');idx=np.argsort(importance)[-15:];ax.barh(np.array(features)[idx],np.asarray(importance)[idx]);return '係数は絶対値で集約。因果効果を意味しない'
        if n=='Permutation Importance':
            need(cfg.get('split')!='time' and not cfg.get('group'),'この版のPermutation評価は独立行用');tr,va=c['cv'][0];mm=clone(model).fit(X.iloc[tr],y[tr]);imp=permutation_importance(mm,X.iloc[va],y[va],scoring='f1_macro' if task=='classification' else 'neg_mean_absolute_error',n_repeats=3,random_state=42);idx=np.argsort(imp.importances_mean)[-15:];ax.barh(X.columns[idx],imp.importances_mean[idx],xerr=imp.importances_std[idx]);return '第1検証fold上で計算'
        if n.startswith('SHAP'):
            import shap
            background=rawz[:min(30,len(rawz))];sample=rawz[:min(20,len(rawz))]
            fn=(lambda a:m.predict_proba(a)[:,int(cfg.get('positive_class',1))]) if task=='classification' and hasattr(m,'predict_proba') else m.predict
            explainer=shap.Explainer(fn,shap.maskers.Independent(background),algorithm='permutation',feature_names=features)
            explanation=explainer(sample,max_evals=max(2*rawz.shape[1]+1,101));v=np.asarray(explanation.values);plt.close(ax.figure)
            if n=='SHAP Summary Plot':shap.plots.beeswarm(explanation,max_display=12,show=False)
            elif n=='SHAP Bar Plot':shap.plots.bar(explanation,max_display=12,show=False)
            elif n=='SHAP Dependence Plot':shap.dependence_plot(int(np.argmax(abs(v).mean(0))),v,sample,feature_names=features,show=False)
            else:shap.plots.waterfall(explanation[0],max_display=12,show=False)
            return '実際のSHAP値。変換後特徴、開発データの背景とサンプル。Permutation SHAPでありPermutation Importanceとは別'
        need(cols,'数値特徴を1列以上選択してください');feature=cols[0]
        if n in ('Partial Dependence Plot (PDP)','ICE Plot'):
            from sklearn.inspection import PartialDependenceDisplay
            PartialDependenceDisplay.from_estimator(model,X.iloc[:100],[feature],kind='average' if n.startswith('Partial') else 'individual',ax=ax,grid_resolution=15);return '対象＝先頭数値列。ほかの特徴との独立性に注意'
        if n=='ALE Plot':
            q=np.unique(np.quantile(num[feature].dropna(),np.linspace(0,1,11)));need(len(q)>=3,'ALEには複数の異なる値が必要');increments=[];counts=[]
            for lo,hi in zip(q[:-1],q[1:]):
                chunk=X.loc[(num[feature]>=lo)&(num[feature]<=hi)].copy();a=chunk.copy();b=chunk.copy();a[feature]=lo;b[feature]=hi
                fn=(lambda d:model.predict_proba(d)[:,int(cfg.get('positive_class',1))]) if task=='classification' else model.predict
                increments.append(float(np.mean(fn(b)-fn(a))) if len(chunk) else 0);counts.append(len(chunk))
            ale=np.cumsum(increments);ale-=np.average(ale,weights=np.maximum(counts,1));ax.plot((q[:-1]+q[1:])/2,ale,'o-');ax.set_xlabel(feature);return '一次ALE。区間内の局所予測差分を累積・重み付き中心化'
        if n=='Counterfactual / What-if Plot':
            base=X.iloc[[0]].copy();grid=np.linspace(num[feature].quantile(.01),num[feature].quantile(.99),30);data=pd.concat([base]*len(grid),ignore_index=True);data[feature]=grid
            fn=(lambda d:model.predict_proba(d)[:,int(cfg.get('positive_class',1))]) if task=='classification' else model.predict;out=fn(data);ax.plot(grid,out);ax.axvline(float(base[feature].iloc[0]),color='red');ax.set_xlabel(feature);return 'What-if：先頭サンプルの先頭数値列だけ変更。実行可能性・因果性の保証なし'
    if n in [g['name'] for g in GRAPHS if g['category']=='クラスタリング']:
        need(rawz is not None and len(rawz)>6,'数値化済みデータが7行以上必要');a=rawz[:min(1500,len(rawz))]
        if n=='Elbow Plot':ks=range(2,min(9,len(a)));ax.plot(list(ks),[KMeans(k,n_init=5,random_state=42).fit(a).inertia_ for k in ks],'o-');return 'k-meansのSSE。すべてのモデルに共通のk最適化ではない'
        if n=='Dendrogram':dendrogram(linkage(a[:150],method='ward'),truncate_mode='lastp',p=20,ax=ax);return 'Ward linkage、先頭150件以内'
        if n=='k-distance Plot':d=NearestNeighbors(n_neighbors=5).fit(a).kneighbors(a)[0][:,-1];ax.plot(np.sort(d));return 'k=5の近傍距離。DBSCAN epsの検討用'
        need(task=='clustering','選択したクラスタモデルの割当てが必要');labels=np.asarray(p);mask=labels!=-1
        if n=='Silhouette Plot':
            need(1<len(np.unique(labels[mask]))<mask.sum(),'有効なクラスタが2個以上必要。ノイズは除外');s=silhouette_samples(rawz[mask],labels[mask]);start=0
            for k in np.unique(labels[mask]):v=np.sort(s[labels[mask]==k]);ax.fill_betweenx(np.arange(start,start+len(v)),0,v);start+=len(v)+5
        elif n=='Cluster Scatter Plot':q=projection(rawz);ax.scatter(q[:,0],q[:,1],c=labels,s=12);return '表示のみPCAで2次元化。クラスタリングは変換済み全特徴で実行'
        elif n=='Cluster Size Bar Plot':lab,cnt=np.unique(labels,return_counts=True);ax.bar(lab.astype(str),cnt)
        return '開発データ。ラベル-1はノイズ'
    if n in [g['name'] for g in GRAPHS if g['category']=='次元削減']:
        need(rawz is not None,'数値化済み特徴が必要');zz=rawz[:min(1500,len(rawz))];pc=PCA(min(10,*zz.shape),random_state=42).fit(zz)
        if n=='Scree Plot':ax.plot(np.arange(1,len(pc.explained_variance_)+1),pc.explained_variance_,'o-')
        elif n=='Cumulative Explained Variance':ax.plot(np.arange(1,len(pc.explained_variance_ratio_)+1),np.cumsum(pc.explained_variance_ratio_),'o-')
        elif n in ('PCA Biplot','Loadings Plot'):
            need(pc.n_components_>=2,'2成分以上必要')
            if n=='PCA Biplot':
                q=pc.transform(zz);ax.scatter(q[:,0],q[:,1],s=10,alpha=.3)
                for j in range(min(12,zz.shape[1])):ax.arrow(0,0,pc.components_[0,j]*3,pc.components_[1,j]*3,color='red');ax.text(pc.components_[0,j]*3,pc.components_[1,j]*3,str(j))
            else:sns.heatmap(pc.components_[:2,:20],center=0,cmap='coolwarm',ax=ax)
        elif n=='t-SNE Scatter':
            from sklearn.manifold import TSNE
            q=TSNE(2,perplexity=min(30,len(zz)-1),random_state=42,max_iter=300).fit_transform(zz);ax.scatter(q[:,0],q[:,1],s=12)
        else:
            from umap import UMAP
            q=UMAP(n_components=2,n_neighbors=min(15,len(zz)-1),random_state=42,n_jobs=1).fit_transform(zz);ax.scatter(q[:,0],q[:,1],s=12)
        return '開発データの実アルゴリズムによる次元削減。軸の意味は手法に依存'
    if n in [g['name'] for g in GRAPHS if g['category']=='異常検知']:
        need(task=='anomaly','異常検知モデルを選択してください');scores=-m.decision_function(rawz)
        if n=='Anomaly Score Distribution':sns.histplot(scores,ax=ax);ax.set_xlabel('higher = more anomalous')
        elif n=='Outlier Scatter / Timeline':
            if c.get('dates') is not None:ax.scatter(c['dates'],scores,c=p==-1,s=12)
            else:q=projection(rawz);ax.scatter(q[:,0],q[:,1],c=p==-1,s=12)
        elif n=='Reconstruction Error Distribution':need(hasattr(m,'errors'),'Autoencoder／VAEなど再構成を行うモデルが必要');sns.histplot(m.errors(rawz),ax=ax)
        else:
            need(y is not None and len(np.unique(y))==2,'Precision@k/Recall@kには異常の正解ラベル（2値）が必要');yy=(np.asarray(y)==int(cfg.get('positive_class',1)));a=yy[np.argsort(-scores)];ks=np.arange(1,len(a)+1);ax.plot(ks,np.cumsum(a)/ks,label='Precision@k');ax.plot(ks,np.cumsum(a)/max(1,a.sum()),label='Recall@k');ax.legend()
        return '異常スコアの大きい側が異常。正解ラベルなしでは精度を推定しない'
    if n in [g['name'] for g in GRAPHS if g['category']=='学習過程']:
        history=getattr(m,'history_',None);need(history is not None,'学習履歴を取得するニューラルモデルが必要')
        if n=='Loss Curve':ax.plot(history['loss'],label='train');ax.plot(history['val_loss'],label='validation');ax.legend()
        elif n=='Metric Curve':ax.plot(history['metric'],label='train');ax.plot(history['val_metric'],label='validation');ax.legend()
        elif n=='Train vs Validation Curve':ax.plot(history['loss'],label='train loss');ax.plot(history['val_loss'],label='validation loss');ax.legend()
        elif n=='Learning Rate Curve / Schedule':ax.plot(history['lr'])
        elif n=='Gradient Norm Plot':ax.plot(history['grad'])
        else:
            diag=m.diagnostics(rawz[:32]);a=diag['weights' if n=='Weight Distribution' else 'activation'];ax.hist(np.asarray(a).ravel(),bins=40)
        return '実学習中の履歴。validationは学習fold内部の独立部分'
    if n in ('Attention Heatmap','Grad-CAM / Saliency Map','Embedding Visualization'):
        need(rawz is not None,'モデル入力の数値特徴が必要')
        if n=='Embedding Visualization':
            need(hasattr(m,'embedding'),'内部表現を提供するニューラルモデルが必要');q=projection(m.embedding(rawz));ax.scatter(q[:,0],q[:,1],s=12,c=y if task=='classification' else None);return '実際の内部表現をPCAで表示'
        need(hasattr(m,'diagnostics'),'勾配／attentionを取得するニューラルモデルが必要');d=m.diagnostics(rawz[:8],target_class=int(cfg.get('positive_class',1))) if hasattr(m,'classes_') else m.diagnostics(rawz[:8])
        if n=='Attention Heatmap':need('attention' in d,'Transformerなどattentionを持つモデルが必要');sns.heatmap(d['attention'][0],ax=ax);return '実際の特徴トークン間attention。因果的説明ではない'
        sal=d.get('gradcam',abs(d['saliency']))[0];shape=cfg.get('image_shape')
        if np.asarray(sal).ndim==2:
            ax.imshow(sal,cmap='inferno');return f'実Conv2D Grad-CAM。説明対象クラス={cfg.get("positive_class",1)}' if task=='classification' else '実Conv2D Grad-CAM。回帰予測値への正の寄与'
        if shape:
            need(np.prod(shape)==len(sal),'画像shapeと特徴数が一致しません');ax.imshow(sal.reshape(shape),cmap='inferno');return '入力勾配／Conv1D Grad-CAMを画像shapeに対応付け。2D画像CNN用Grad-CAMとは異なる'
        ax.plot(sal);return '表／1D系列入力の実勾配Saliency。CNNでは実Conv1D Grad-CAM。画像ではなく特徴列の寄与'
    if n in [g['name'] for g in GRAPHS if g['category']=='時系列']:
        dates=c.get('dates');need(dates is not None,'日付列が必要');values=np.asarray(y if y is not None else vals,float);need(len(values)==len(dates),'日付と値の長さが一致する数値列が必要');period=int(cfg.get('period',7));series=pd.Series(values,index=pd.DatetimeIndex(dates))
        if n=='Time Series Plot':ax.plot(dates,values)
        elif n=='Rolling Mean / Std':ax.plot(series.rolling(period).mean(),label='mean');ax.plot(series.rolling(period).std(),label='std');ax.legend()
        elif n=='Seasonal Decomposition / STL':
            from statsmodels.tsa.seasonal import STL
            need(len(values)>=2*period,'季節周期の2倍以上の件数が必要');plt.close(ax.figure);STL(series,period=period).fit().plot()
        elif n=='ACF Plot':plot_acf(values,lags=min(20,len(values)//2-1),ax=ax)
        elif n=='PACF Plot':plot_pacf(values,lags=min(20,len(values)//2-1),ax=ax)
        elif n=='Lag Plot':pd.plotting.lag_plot(series,lag=1,ax=ax)
        elif n in ('Forecast vs Actual','Forecast Residual Plot','Backtest / Rolling-origin Plot'):
            need(task=='timeseries','時間順の実予測が必要');mask=np.isfinite(p)
            if n=='Forecast Residual Plot':ax.plot(np.asarray(dates)[mask],values[mask]-p[mask]);ax.axhline(0,color='red')
            elif n=='Backtest / Rolling-origin Plot':
                for i,(tr,va) in enumerate(c['cv']):
                    mm=clone_forecast(model);mm.fit(values[tr],dates.iloc[tr]);pp=mm.predict(dates.iloc[va]);ax.plot(dates.iloc[va],pp,label=f'origin {i+1}')
                ax.plot(dates,values,color='black',alpha=.4,label='actual');ax.legend()
            else:ax.plot(dates,values,label='actual');ax.plot(np.asarray(dates)[mask],p[mask],label='forecast');ax.legend()
        elif n=='Cross-Correlation / CCF':
            from statsmodels.tsa.stattools import ccf
            need(cols and (len(cols)>1 or cfg.get('target') not in cols),'目的変数以外の数値系列が必要');other=num.iloc[:,0].interpolate().bfill().to_numpy();a=ccf(values,other,nlags=min(20,len(values)//2));ax.stem(np.arange(len(a)),a)
        elif n=='Periodogram / FFT Spectrum':
            gaps=pd.DatetimeIndex(dates).to_series().diff().dropna().dt.total_seconds();need(len(gaps)>0 and np.allclose(gaps,gaps.iloc[0]),'周波数計算は等間隔の日時が必要');freq,power=signal.periodogram(values,fs=1/gaps.iloc[0]);ax.plot(freq,power);ax.set_xlabel('Hz')
        elif n=='Seasonal Box Plot':groups=[series.iloc[i::period].values for i in range(period)];ax.boxplot(groups);ax.set_xlabel('seasonal phase')
        return '時間順の開発データ。最終テストは含まない'
    if n in [g['name'] for g in GRAPHS if g['category']=='モデル選択']:
        if n=='Model Comparison Bar / Box Plot':
            rows=[r for r in c['rows'] if r.get('cv_scores')];need(rows,'共通CVスコアが必要');ax.boxplot([r['cv_scores'] for r in rows],tick_labels=[r['model'] for r in rows]);ax.tick_params(axis='x',rotation=45);return '同じ分割・主指標のfoldスコア分布'
        need(task in ('classification','regression'),'このモデル選択グラフは教師あり表データ用')
        metric=cfg.get('metric') or ('f1_macro' if task=='classification' else 'mae');scoring=metric if task=='classification' else {'mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','r2':'r2'}[metric]
        if n=='CV Score Distribution':
            row=next(r for r in c['rows'] if r['model']==cfg['selected_model']);ax.boxplot(row['cv_scores']);ax.scatter(np.ones(len(row['cv_scores'])),row['cv_scores']);return
        if n in ('Learning Curve','Validation Curve'):ax.set_ylabel(scoring)
        if n=='Learning Curve':
            sizes,tr,va=learning_curve(model,X,y,cv=c['cv'],train_sizes=[.5,.75,1.],scoring=scoring,n_jobs=1,error_score='raise');ax.plot(sizes,tr.mean(1),label='train');ax.plot(sizes,va.mean(1),label='validation');ax.legend();return f'実際に再学習した学習曲線。CV採点={scoring}。neg_指標は負の誤差で、0に近いほど誤差が小さい'
        params=model.get_params();choice=next((p for p in ['model__alpha','model__C','model__max_depth','model__n_neighbors'] if p in params),None)
        if n=='Regularization Path / Coefficient Path':
            need(cfg['selected_model'] in ('ridge','lasso','elasticnet','logistic'),'正則化を持つ線形モデルが必要');parameter='model__C' if cfg['selected_model']=='logistic' else 'model__alpha';grid=np.logspace(-3,2,12);coeff=[]
            for v in grid:mm=clone(model).set_params(**{parameter:v}).fit(X,y);coeff.append(mm.named_steps['model'].coef_.ravel())
            ax.semilogx(grid,np.array(coeff));ax.set_xlabel(parameter);return '開発データで実学習した係数。Cは正則化強度の逆数'
        need(choice is not None,'変更可能な比較パラメータを持つモデルが必要');grid=[3,5,8] if choice.endswith(('max_depth','n_neighbors')) else [.01,.1,1.]
        if n=='Validation Curve':
            tr,va=validation_curve(model,X,y,param_name=choice,param_range=grid,cv=c['cv'],scoring=scoring,n_jobs=1,error_score='raise');ax.plot(grid,tr.mean(1),label='train');ax.plot(grid,va.mean(1),label='validation');ax.legend();ax.set_xlabel(choice)
        elif n=='Hyperparameter Heatmap':
            second=next((p for p in ['model__min_samples_leaf','model__gamma','model__l1_ratio','model__fit_intercept','model__weights'] if p in params and p!=choice),None);need(second is not None,'探索する第2パラメータが必要');g2=[1,3] if second.endswith('min_samples_leaf') else [.1,1.] if second.endswith(('gamma','l1_ratio')) else [True,False] if second.endswith('fit_intercept') else ['uniform','distance'];matrix=[]
            from sklearn.model_selection import cross_val_score
            for b in g2:
                matrix.append([np.mean(cross_val_score(clone(model).set_params(**{choice:a,second:b}),X,y,cv=c['cv'],scoring=scoring,n_jobs=1,error_score='raise')) for a in grid])
            sns.heatmap(matrix,annot=True,xticklabels=grid,yticklabels=g2,ax=ax);ax.set(xlabel=choice,ylabel=second)
        return f'実再学習／CV。採点={scoring}。neg_指標は負の誤差。探索結果は最終テストの代替ではない'
    raise RuntimeError('描画処理未接続')

def clone_forecast(model):
    from .estimators import ForecastModel
    return ForecastModel(model.kind,model.period,model.epochs,model.seed)
