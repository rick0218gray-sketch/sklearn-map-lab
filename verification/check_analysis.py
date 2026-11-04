"""全推定器と95描画の実行検証。生成データを使いユーザー記録に触れない。"""
import sys,time,json,tempfile,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import numpy as np,pandas as pd,joblib
from sklearn.datasets import make_classification,make_regression,make_blobs
from analysis.catalog import SPECS,GRAPHS
from analysis.engine import run_experiment,write_json,read_json,finalize
from analysis.graphs import draw
root=Path(tempfile.mkdtemp(prefix='sklearn_analysis_verify_')); output={'directory':str(root),'models':{},'graphs':{}}
rng=np.random.default_rng(42)
x,y=make_classification(n_samples=120,n_features=5,n_informative=3,random_state=42)
cls=pd.DataFrame(x,columns=list('abcde'));cls['category']=np.where(x[:,0]>0,'A','B');cls['target']=y
x,y=make_regression(n_samples=120,n_features=5,noise=4,random_state=42)
reg=pd.DataFrame(x,columns=list('abcde'));reg['target']=np.exp((y-y.mean())/y.std()*.3)+1
x,y=make_blobs(n_samples=100,n_features=4,centers=3,cluster_std=.5,random_state=42)
clu=pd.DataFrame(x,columns=list('abcd'))
ano=clu.copy();ano['target']=(np.arange(len(ano))%15==0).astype(int);ano.loc[ano.target==1,'a']+=15
n=100;t=np.arange(n);ts=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=n),'a':np.sin(t/7)+rng.normal(0,.1,n),'target':10+.05*t+np.sin(t*2*np.pi/7)+rng.normal(0,.1,n)})
fixtures={'classification':cls,'regression':reg,'clustering':clu,'anomaly':ano,'reduction':clu,'timeseries':ts}; contexts={}
for task,frame in fixtures.items():
    ids=list(dict.fromkeys(s['id'] for s in SPECS if task in s['tasks']))
    d=root/task;d.mkdir();frame.to_parquet(d/'data.parquet',index=False)
    cfg={'task':task,'target':'target' if 'target' in frame else None,'date':'date' if task=='timeseries' else None,'models':ids,'cv':2,'seed':42,'epochs':2,'period':7,'test_size':.2,'split':'time' if task=='timeseries' else 'random','metric':'f1_macro' if task=='classification' else 'mae' if task in ('regression','timeseries') else None}
    write_json(d/'config.json',cfg);run_experiment(d);res=read_json(d/'result.json')
    for row in res['rows']:
        output['models'][task+':'+row['model']]=row;print(task,row['model'],row['status'],row.get('reason',''),flush=True)
        if row['status']=='completed':
            c=joblib.load(d/(row['model']+'.joblib'));c['config']['selected_model']=row['model'];c['rows']=res['rows'];contexts[(task,row['model'])]=c
    if task in ('classification','regression','timeseries'):
        chosen='logistic' if task=='classification' else 'ridge' if task=='regression' else 'arima'
        if (task,chosen) in contexts:
            output.setdefault('final',{})[task]=finalize(d,chosen,'検証用の採用理由')
    write_json(Path(__file__).parent/'analysis_check.json',output)
for g in GRAPHS:
    cat=g['category'];name=g['name']
    key=('classification','logistic')
    if cat=='回帰':key=('regression','linear')
    elif cat=='説明':key=('classification','logistic')
    elif cat=='クラスタリング':key=('clustering','kmeans')
    elif cat=='次元削減':key=('reduction','pca')
    elif cat=='異常検知':key=('anomaly','autoencoder')
    elif cat=='学習過程':key=('classification','mlp')
    elif cat=='深層モデル説明':key=('classification','transformer' if name=='Attention Heatmap' else 'cnn')
    elif cat=='時系列':key=('timeseries','arima')
    elif cat=='モデル選択':key=('regression','ridge')
    c=contexts.get(key)
    if c is None:out=dict(g,status='failed',reason='model fixture unavailable')
    else:out=draw(c,g['id'],root/'graphs')
    output['graphs'][g['id']]=out;print(g['id'],name,out['status'],out.get('reason',''),flush=True)
    write_json(Path(__file__).parent/'analysis_check.json',output)
print('VERIFIED',sum(v['status']=='completed' for v in output['models'].values()),'/',len(output['models']),'models;',sum(v['status']=='rendered' for v in output['graphs'].values()),'/95 graphs',flush=True)
