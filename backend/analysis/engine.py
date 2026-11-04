"""同一分割で比較し、最終テストを選択まで封印する分析エンジン。"""
import json, os, sys, time, uuid, hashlib, platform, importlib.metadata, warnings
import datetime, threading
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold, GroupShuffleSplit, TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from .catalog import SPECS
from .estimators import pipeline, ForecastModel
warnings.filterwarnings('ignore',category=FutureWarning)

def clean(x):
    if x is pd.NA or x is pd.NaT:return None
    if isinstance(x,(datetime.datetime,datetime.date,pd.Timestamp,np.datetime64)):return pd.Timestamp(x).isoformat()
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,Path): return str(x)
    return x

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('w',encoding='utf-8') as out:json.dump(clean(value),out,ensure_ascii=False,indent=2);out.flush();os.fsync(out.fileno())
        for attempt in range(8):
            try:os.replace(tmp,p);break
            except PermissionError:
                if attempt==7:raise
                time.sleep(.04*(attempt+1))
    finally:
        tmp.unlink(missing_ok=True)
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def profile(frame):
    columns=[]
    for c in frame:
        s=frame[c]; columns.append({'name':c,'dtype':str(s.dtype),'missing':int(s.isna().sum()),'unique':int(s.nunique()),'sample':s.dropna().astype(str).head(4).tolist(),'class_values':sorted(s.dropna().astype(str).unique().tolist()) if s.nunique()<=50 else []})
    targets=[c for c in frame if c.lower() in ('target','label','class','y','目的変数','売上','結果')]
    suggested=targets[0] if len(targets)==1 else (frame.columns[-1] if len(frame.columns)>1 else None)
    dates=[c for c in frame if any(t in c.lower() for t in ['date','time','日付','日時'])]
    task='clustering'
    if suggested:
        s=frame[suggested];task='classification' if not pd.api.types.is_numeric_dtype(s) or s.nunique()<=min(20,max(2,len(s)//10)) else 'regression'
    return dict(rows=len(frame),columns=columns,duplicates=int(frame.duplicated().sum()),preview=frame.head(8).astype(object).where(pd.notna(frame.head(8)),None).to_dict('records'),suggested_target=suggested,suggested_task=task,date_candidates=dates,warnings=['目的変数と分析種類は候補です。実行前に確認してください。'])

def recommendations(df,config):
    task=config.get('task','classification');target=config.get('target');y=df[target] if target in df else None
    specs={}
    for s in SPECS:
        if s['id'] not in specs:specs[s['id']]=dict(s,tasks=list(s['tasks']))
        else: specs[s['id']]['tasks']+=s['tasks']
    out=[]
    fast={'classification':['dummy','logistic','forest','extra','lightgbm'],'regression':['dummy','ridge','forest','extra','lightgbm'],'clustering':['kmeans','hierarchical','gmm'],'anomaly':['isolation','oneclass'],'reduction':['pca','tsne'],'timeseries':['arima','ets','prophet']}.get(task,[])
    if config.get('image_shape') and task in ('classification','regression'):fast=fast+['cnn']
    for s in specs.values():
        reason=s['reason'];available=task in s['tasks']
        if not available:reason='この分析目的には対応しません'
        if s['id']=='poisson' and (y is None or not pd.api.types.is_numeric_dtype(y) or (y.dropna()<0).any()):available=False;reason='Poissonには非負の数値目的変数が必要'
        if s['id']=='gamma' and (y is None or not pd.api.types.is_numeric_dtype(y) or (y.dropna()<=0).any()):available=False;reason='Gammaには正の数値目的変数が必要'
        if s['id']=='gp' and len(df)>2000:available=False;reason='Gaussian Processはこの版では2000行を上限に設定'
        if s['id'] in ('svm','oneclass','tsne','hierarchical') and len(df)>10000:available=False;reason='計算量のため10000行を上限に設定'
        if s['id']=='stacking' and config.get('group'):available=False;reason='グループ指定時の内部stacking分割には未対応。この条件では除外'
        out.append(dict(s,available=available,recommended=available and s['id'] in fast,reason=reason))
    return out

SUPERVISED_METRICS={'classification':('accuracy','precision_macro','recall_macro','f1_macro'),'regression':('mae','rmse','r2'),'timeseries':('mae','rmse','r2')}
UNSUPERVISED_METRICS={'clustering':'silhouette','anomaly':'detected','reduction':'dimensions'}
def resolve_metric(cfg):
    """主指標を決める。教師なしは画面で指標を選べないため、残っていた教師あり指標を無視して固定値を使う。"""
    task=cfg['task']
    if task in UNSUPERVISED_METRICS:return UNSUPERVISED_METRICS[task]
    metric=cfg.get('metric') or {'classification':'f1_macro'}.get(task,'mae')
    if metric not in SUPERVISED_METRICS[task]:raise ValueError('主指標を分類用から選んでください' if task=='classification' else '主指標を回帰用から選んでください')
    return metric

def prepare(df,cfg):
    task=cfg['task'];target=cfg.get('target');date=cfg.get('date');group=cfg.get('group')
    if task in ('classification','regression','timeseries') and target not in df:raise ValueError('目的変数を選択してください')
    if task=='timeseries':
        if date not in df:raise ValueError('時系列には日付列が必要です')
        if df[target].isna().any():raise ValueError('時系列の目的変数に欠測があります。日時を保持して補完方法を決めてから取り込んでください')
        df=df.copy();df[date]=pd.to_datetime(df[date],errors='raise')
        if df[date].isna().any():raise ValueError('時系列の日付に欠測があります。日付を補ってから取り込んでください')
        df=df.sort_values(date).reset_index(drop=True)
        if df[date].duplicated().any():raise ValueError('時系列の日付が重複しています。集計してから取り込んでください')
        gaps=df[date].diff().dropna()
        if len(gaps) and not (gaps==gaps.iloc[0]).all():raise ValueError('この版の時系列比較は等間隔の日付が必要です。欠測日時を補ってください')
    if target in df:df=df.loc[df[target].notna()].reset_index(drop=True)
    if len(df)<20:raise ValueError('公平な分割と比較のため20行以上必要です')
    features=cfg.get('features') or [c for c in df if c not in (target,date,group)]
    if any(c in features for c in (target,date,group) if c) or any(c not in df for c in features):raise ValueError('説明変数に目的変数・存在しない列が含まれています')
    shape=cfg.get('image_shape')
    if shape and (len(shape)!=2 or min(shape)<2):raise ValueError('画像の縦・横は2以上の整数2個で指定してください')
    if shape:
        features=[c for c in df if c in features]
        if len(features)!=int(np.prod(shape)):raise ValueError('2D画像の縦×横と選択したピクセル列数が一致しません')
        if any(not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_complex_dtype(df[c]) for c in features):raise ValueError('2D画像では数値ピクセル列だけを選択してください')
    if not features and task!='timeseries':raise ValueError('説明変数を1列以上選択してください')
    X=df[features].copy().replace([np.inf,-np.inf],np.nan)
    if shape:X=X.astype(float)
    for c in X.select_dtypes(exclude='number'):X[c]=X[c].astype(object).where(X[c].notna(),np.nan)
    labels=None;y=None
    if target in df:
        if task=='classification':
            enc=LabelEncoder();y=enc.fit_transform(df[target].astype(str));labels=enc.classes_.tolist()
            if len(labels)<2 or pd.Series(y).value_counts().min()<4:raise ValueError('分類には2クラス以上・各クラス4行以上必要です')
        else:y=pd.to_numeric(df[target],errors='raise').to_numpy(float)
    seed=int(cfg.get('seed',42));ids=np.arange(len(df));test=float(cfg.get('test_size',.2))
    if not .1<=test<=.4:raise ValueError('最終テスト割合は0.1～0.4にしてください')
    if task=='timeseries' or cfg.get('split')=='time':
        if date not in df:raise ValueError('時間順分割には日付列が必要です')
        if task!='timeseries':
            parsed_dates=pd.to_datetime(df[date],errors='raise')
            if parsed_dates.isna().any():raise ValueError('時間順分割の日付に欠測があります')
            order=np.argsort(parsed_dates.to_numpy());df=df.iloc[order].reset_index(drop=True);X=X.iloc[order].reset_index(drop=True);y=y[order] if y is not None else None
        cut=int(len(df)*(1-test));dev,hold=ids[:cut],ids[cut:]
    elif group:
        if group not in df or df[group].nunique()<4:raise ValueError('グループ分割には4グループ以上必要です')
        dev,hold=next(GroupShuffleSplit(1,test_size=test,random_state=seed).split(X,y,df[group]))
    else:dev,hold=train_test_split(ids,test_size=test,random_state=seed,stratify=y if task=='classification' else None)
    k=int(cfg.get('cv',3));k=max(2,min(k,5));Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev] if y is not None else None
    if task=='timeseries' or cfg.get('split')=='time':cv=list(TimeSeriesSplit(k).split(Xd))
    elif group:cv=list(GroupKFold(min(k,df.iloc[dev][group].nunique())).split(Xd,yd,df.iloc[dev][group]))
    elif task=='classification':
        counts=pd.Series(yd).value_counts();k=min(k,int(counts.min()))
        if k<2:raise ValueError('学習側の各クラス件数が交差検証に不足しています')
        cv=list(StratifiedKFold(k,shuffle=True,random_state=seed).split(Xd,yd))
    else:cv=list(KFold(k,shuffle=True,random_state=seed).split(Xd))
    if task=='classification':
        for tr,va in cv:
            if len(np.unique(yd[tr]))!=len(labels):raise ValueError('学習foldに含まれないクラスがあります。時間範囲・グループ・分割を見直してください')
    if task in ('regression','timeseries') and not np.isfinite(y).all():raise ValueError('目的変数に無限大など無効な数値があります')
    return df,X,y,labels,dev,hold,cv

def metrics(task,y,p,prob=None):
    if task=='classification':
        result=dict(accuracy=accuracy_score(y,p),precision_macro=precision_score(y,p,average='macro',zero_division=0),recall_macro=recall_score(y,p,average='macro',zero_division=0),f1_macro=f1_score(y,p,average='macro',zero_division=0))
        if prob is not None:
            try:result['roc_auc']=roc_auc_score(y,prob[:,1] if prob.shape[1]==2 else prob,multi_class='ovr')
            except ValueError:pass
        return result
    return dict(mae=mean_absolute_error(y,p),rmse=mean_squared_error(y,p)**.5,r2=r2_score(y,p))

def aligned_prob(model,X,n_classes):
    value=model.predict_proba(X);out=np.zeros((len(X),n_classes))
    out[:,np.asarray(model.classes_,int)]=value
    return out

def score(task,y,p,metric):
    vals=metrics(task,y,p);val=vals.get(metric)
    if val is None:raise ValueError('この分析種類では指定された主指標を計算できません')
    return float(val)

def run_experiment(directory):
    directory=Path(directory);cfg=read_json(directory/'config.json');df=pd.read_parquet(directory/'data.parquet')
    result={'id':directory.name,'status':'running','config':cfg,'rows':[],'warnings':[],'started_at':pd.Timestamp.now(tz='Asia/Tokyo').isoformat()}
    write_json(directory/'result.json',result)
    try:
        df,X,y,labels,dev,hold,cv=prepare(df,cfg);task=cfg['task'];Xd=X.iloc[dev].reset_index(drop=True);yd=y[dev] if y is not None else None
        meta={'features':X.columns.tolist(),'labels':labels,'development_indices':dev.tolist(),'holdout_indices':hold.tolist(),'folds':[(tr.tolist(),va.tolist()) for tr,va in cv], 'data_sha256':hashlib.sha256((directory/'data.parquet').read_bytes()).hexdigest(),'environment':{'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['scikit-learn','pandas','numpy','torch','statsmodels','scipy','matplotlib','seaborn','shap','xgboost','lightgbm','catboost','prophet','umap-learn','openpyxl','pyarrow','joblib']}}}
        write_json(directory/'metadata.json',meta)
        joblib.dump({'df':df,'X':X,'y':y,'dev':dev,'hold':hold,'cv':cv,'labels':labels},directory/'split.joblib')
        rec=recommendations(df,cfg);allowed={s['id'] for s in rec if s['available']};requested=cfg.get('models') or [s['id'] for s in rec if s['recommended']]
        if task in ('classification','regression') and 'dummy' not in requested:requested=['dummy']+requested
        metric=resolve_metric(cfg)
        result.update(metric=metric,direction='minimize' if metric in ('mae','rmse') else 'maximize',recommendations=rec)
        for name in dict.fromkeys(requested):
            if (directory/'cancel').exists():result['status']='cancelled';break
            row={'model':name,'status':'running','cv_scores':[]};result['rows'].append(row);write_json(directory/'result.json',result)
            if name not in allowed:row.update(status='not_applicable',reason=next((s['reason'] for s in rec if s['id']==name),'未登録モデル'));continue
            try:
                start=time.perf_counter();pred=np.full(len(dev),np.nan);prob=None;foldmetrics=[];search=[]
                if task in ('classification','regression','timeseries'):
                    for fold,(tr,va) in enumerate(cv):
                        if task=='timeseries':
                            model=ForecastModel(name,int(cfg.get('period',7)),int(cfg.get('epochs',25)),int(cfg.get('seed',42)))
                            dd=df.iloc[dev];model.fit(yd[tr],dd.iloc[tr][cfg['date']]);p=model.predict(dd.iloc[va][cfg['date']]);pb=None
                        else:
                            model=pipeline(name,task,Xd.iloc[tr],int(cfg.get('seed',42)),int(cfg.get('epochs',25)),image_shape=cfg.get('image_shape'))
                            model.fit(Xd.iloc[tr],yd[tr]);p=model.predict(Xd.iloc[va]);pb=aligned_prob(model,Xd.iloc[va],len(labels)) if hasattr(model,'predict_proba') else None
                            if pb is not None:
                                if prob is None:prob=np.full((len(dev),len(labels)),np.nan)
                                prob[va]=pb
                        pred[va]=p;foldmetrics.append(metrics('regression' if task=='timeseries' else task,yd[va],p,pb))
                        row['cv_scores'].append(score('regression' if task=='timeseries' else task,yd[va],p,metric))
                    if task=='timeseries':model.fit(yd,df.iloc[dev][cfg['date']])
                    else:
                        # 必要なら主指標で開発データ内のパラメータを探索し、探索の外側に最終テストを保持する。
                        if cfg.get('search') and name in ('ridge','lasso','elasticnet','logistic','tree','forest','extra','svm','knn'):
                            grid={'model__alpha':[.01,.1,1.]} if name in ('ridge','lasso','elasticnet') else {'model__C':[.1,1.,10.]} if name in ('logistic','svm') else {'model__n_neighbors':[3,5,9]} if name=='knn' else {'model__max_depth':[3,6,None]}
                            best=None
                            for params in ParameterGrid(grid):
                                vals=[]
                                for tr,va in cv:
                                    m=clone(model).set_params(**params).fit(Xd.iloc[tr],yd[tr]);vals.append(score(task,yd[va],m.predict(Xd.iloc[va]),metric))
                                val=float(np.mean(vals));search.append({'params':params,'score':val,'folds':vals})
                                if best is None or (val<best[0] if result['direction']=='minimize' else val>best[0]):best=(val,params)
                            model.set_params(**best[1]);row['search_selected']=best[1]
                            # 選ばれた設定のCVを再計算。探索後スコアであり外部検証ではないと明記する。
                            row['cv_scores']=[];foldmetrics=[]
                            for tr,va in cv:
                                m=clone(model).fit(Xd.iloc[tr],yd[tr]);p=m.predict(Xd.iloc[va]);pred[va]=p
                                pb=aligned_prob(m,Xd.iloc[va],len(labels)) if hasattr(m,'predict_proba') else None
                                if pb is not None:prob[va]=pb
                                foldmetrics.append(metrics(task,yd[va],p,pb));row['cv_scores'].append(score(task,yd[va],p,metric))
                            row['score_note']='探索に使ったCVでのスコア。最終テストで別途確認する'
                        model.fit(Xd,yd)
                    valid=np.isfinite(pred);vals=metrics('regression' if task=='timeseries' else task,yd[valid],pred[valid],prob[valid] if prob is not None else None)
                    row.update(metrics=vals,mean=float(np.mean(row['cv_scores'])),std=float(np.std(row['cv_scores'])),search=search)
                else:
                    model=pipeline(name,task,Xd,int(cfg.get('seed',42)),int(cfg.get('epochs',25)),image_shape=cfg.get('image_shape'))
                    z=model.named_steps['pre'].fit_transform(Xd);m=model.named_steps['model'];m.fit(z)
                    if task=='clustering':
                        pred=m.predict(z) if hasattr(m,'predict') else m.labels_;mask=pred!=-1
                        ncl=len(np.unique(pred[mask]));vals={}
                        if 1<ncl<mask.sum():vals={'silhouette':silhouette_score(z[mask],pred[mask],sample_size=min(1500,int(mask.sum())),random_state=42),'calinski':calinski_harabasz_score(z[mask],pred[mask]),'davies_bouldin':davies_bouldin_score(z[mask],pred[mask])}
                        row.update(metrics=vals,mean=vals.get('silhouette'),std=None,score_note='教師なしの開発データ指標。予測精度や交差検証とは別',noise_ratio=float((pred==-1).mean()))
                    elif task=='anomaly':
                        pred=m.predict(z);row.update(mean=None,std=None,score_note='異常の正解ラベルなしでは優劣を自動順位付けしない',metrics={'detected':int((pred==-1).sum()),'rate':float((pred==-1).mean())})
                    else:pred=m.embedding_ if hasattr(m,'embedding_') else m.transform(z);row.update(mean=None,std=None,score_note='埋め込みは目的と構造を見て評価。精度ランキング対象外',metrics={'dimensions':int(pred.shape[1])})
                elapsed=time.perf_counter()-start
                transformed=model.named_steps['pre'].transform(Xd) if task!='timeseries' else None
                context={'model':model,'X':Xd,'y':yd,'pred':pred,'prob':prob,'task':task,'config':cfg,'cv':cv,'rows':result['rows'],'dates':df.iloc[dev][cfg['date']].reset_index(drop=True) if cfg.get('date') else None,'Z':transformed,'labels':labels}
                joblib.dump(context,directory/f'{name}.joblib')
                row.update(status='completed',seconds=elapsed,parameters={k:clean(v) if isinstance(v,(str,int,float,bool,type(None))) else repr(v) for k,v in model.named_steps['model'].get_params(deep=False).items()} if hasattr(model,'named_steps') else {'kind':name,'period':cfg.get('period',7)})
                threshold=cfg.get('success_threshold');time_limit=cfg.get('max_fit_seconds')
                row['meets_conditions']=(row.get('mean') is not None and (threshold is None or (row['mean']<=threshold if result['direction']=='minimize' else row['mean']>=threshold)) and (time_limit is None or elapsed<=time_limit))
            except Exception as e:row.update(status='failed',reason=f'{type(e).__name__}: {e}')
            write_json(directory/'result.json',result)
        if result['status']!='cancelled':result['status']='completed'
        eligible=[r for r in result['rows'] if r.get('meets_conditions') and r['model']!='dummy']
        result['suggested_model']=sorted(eligible,key=lambda r:r['mean'],reverse=result['direction']=='maximize')[0]['model'] if eligible else None
        result['completed_at']=pd.Timestamp.now(tz='Asia/Tokyo').isoformat();write_json(directory/'result.json',result)
    except Exception as e:result.update(status='failed',error=f'{type(e).__name__}: {e}');write_json(directory/'result.json',result)

def _finalize(directory,name,reason):
    directory=Path(directory)
    if not reason.strip():raise ValueError('採用理由を書いてください')
    if (directory/'final.json').exists():return read_json(directory/'final.json')
    result=read_json(directory/'result.json');row=next((r for r in result['rows'] if r['model']==name and r['status']=='completed'),None)
    if row is None:raise ValueError('学習が完了したモデルを選択してください')
    ctx=joblib.load(directory/f'{name}.joblib');split=joblib.load(directory/'split.joblib');model=ctx['model'];hold=split['hold'];task=ctx['task'];cfg=ctx['config']
    if task not in ('classification','regression','timeseries'):
        final={'model':name,'reason':reason,'metrics':None,'test_rows':0,'evaluated_at':pd.Timestamp.now(tz='Asia/Tokyo').isoformat(),'note':'教師なし分析の採用を確定。未使用テストの予測精度は定義しない'}
        atomic_model(model,directory/'adopted.joblib');write_json(directory/'final.json',final);return final
    marker=directory/'finalizing.json'
    if marker.exists():
        seal=read_json(marker)
        if seal.get('model')!=name:raise ValueError('最終評価開始時に選択したモデルと一致しません')
        if seal.get('result'):
            deployed=production_model(model,split,task,cfg)
            atomic_model(deployed,directory/'adopted.joblib');write_json(directory/'final.json',seal['result']);return seal['result']
        raise ValueError('最終評価は開始済みです。二重評価を防ぐため再実行できません。記録を確認してください')
    write_json(marker,{'model':name,'reason':reason,'started_at':pd.Timestamp.now(tz='Asia/Tokyo').isoformat()})
    atomic_model(model,directory/'adopted.joblib')
    pred=model.predict(split['df'].iloc[hold][cfg['date']]) if task=='timeseries' else model.predict(split['X'].iloc[hold])
    pb=aligned_prob(model,split['X'].iloc[hold],len(split['labels'])) if hasattr(model,'predict_proba') else None
    final={'model':name,'reason':reason,'metrics':metrics('regression' if task=='timeseries' else task,split['y'][hold],pred,pb),'test_rows':len(hold),'evaluated_at':pd.Timestamp.now(tz='Asia/Tokyo').isoformat(),'note':'最終テストは選択確定後に一度評価。この実験では採用モデルを変更しない'}
    final['production_rows']=len(split['df']);final['production_note']='最終指標は未使用テスト上の値。評価後に設定を固定して全データで再学習したモデルを予測用に保存'
    write_json(marker,{'model':name,'reason':reason,'result':final})
    deployed=production_model(model,split,task,cfg);atomic_model(deployed,directory/'adopted.joblib')
    write_json(directory/'final.json',final);return final

FINAL_LOCKS={}; FINAL_LOCK_GUARD=threading.Lock()
def run_lock(directory):
    with FINAL_LOCK_GUARD:return FINAL_LOCKS.setdefault(str(directory),threading.RLock())

def atomic_model(model,path):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:joblib.dump(model,tmp);os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)

def finalize(directory,name,reason):
    with run_lock(directory):return _finalize(directory,name,reason)

def production_model(model,split,task,cfg):
    if task=='timeseries':return ForecastModel(model.kind,model.period,model.epochs,model.seed).fit(split['y'],split['df'][cfg['date']])
    return clone(model).fit(split['X'],split['y'])
