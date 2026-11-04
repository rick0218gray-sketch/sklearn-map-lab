import os,io,json,time,tempfile,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np,pandas as pd,joblib
from fastapi import FastAPI
from fastapi.testclient import TestClient
from analysis import api
from analysis.engine import clean,write_json,read_json,prepare,run_experiment,finalize,profile
from analysis.estimators import preprocessor
from sklearn.datasets import make_classification

class AnalysisTest(unittest.TestCase):
 def setUp(self):
    self.temp=tempfile.TemporaryDirectory(prefix='analysis_api_test_');self.addCleanup(self.temp.cleanup)
    self.root=Path(self.temp.name);self.savedRoot=api.ROOT;api.ROOT=self.root;self.addCleanup(setattr,api,'ROOT',self.savedRoot)
    self.addCleanup(self.wait_workers)
    app=FastAPI();app.include_router(api.router);self.client=TestClient(app)
    x,y=make_classification(n_samples=80,n_features=4,n_informative=3,n_redundant=0,random_state=42)
    self.df=pd.DataFrame(x,columns=list('abcd'));self.df['target']=y
 def wait_workers(self):
    for _ in range(100):
        with api.job_lock:active=any(str(proc.args[3]).startswith(str(self.root)) for proc in api.jobs.values())
        if not active:return
        time.sleep(.1)
 def cfg(self,**kw):return {'data_id':'a'*32,'task':'classification','target':'target','models':['logistic'],'cv':2,'seed':42,'test_size':.2,'epochs':2,'split':'random',**kw}
 def fixture(self,**kw):
    d=self.root/'runs'/('a'*32);d.mkdir(parents=True);self.df.to_parquet(d/'data.parquet',index=False);write_json(d/'config.json',self.cfg(**kw));run_experiment(d);return d
 def test_excel_dates_and_missing_values_serialize(self):
    frame=self.df.copy();frame['date']=pd.date_range('2020-01-01',periods=len(frame));frame.loc[2,'a']=np.nan
    data=io.BytesIO();frame.to_excel(data,index=False)
    r=self.client.post('/api/analysis/upload',files={'file':('日付付き.xlsx',data.getvalue(),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
    self.assertEqual(r.status_code,200,r.text);self.assertTrue(r.json()['date_candidates']);self.assertEqual(r.json()['preview'][0]['date'],'2020-01-01T00:00:00')
 def test_cp932_csv_and_candidates(self):
    frame=self.df.rename(columns={'a':'数値'})
    r=self.client.post('/api/analysis/upload',files={'file':('日本語.csv',frame.to_csv(index=False).encode('cp932'),'text/csv')})
    self.assertEqual(r.status_code,200,r.text);id=r.json()['id']
    result=self.client.post('/api/analysis/recommend',json=self.cfg(data_id=id));self.assertEqual(result.status_code,200);self.assertTrue(any(s['id']=='logistic' and s['recommended'] for s in result.json()))
 def test_time_missing_target_rejected(self):
    frame=self.df.copy();frame['date']=pd.date_range('2020-01-01',periods=len(frame));frame.loc[5,'target']=np.nan
    with self.assertRaisesRegex(ValueError,'欠測'):prepare(frame,self.cfg(task='timeseries',date='date',models=['arima']))
 def test_missing_time_date_rejected_before_seal(self):
    frame=self.df.copy();frame['date']=pd.date_range('2020-01-01',periods=len(frame));frame.loc[len(frame)-1,'date']=pd.NaT
    with self.assertRaisesRegex(ValueError,'日付に欠測'):prepare(frame,self.cfg(task='timeseries',date='date',models=['arima']))
 def test_missing_classes_in_time_fold_rejected(self):
    frame=self.df.copy();frame['date']=pd.date_range('2020-01-01',periods=len(frame));frame.target=np.r_[np.zeros(40),np.ones(40)]
    with self.assertRaisesRegex(ValueError,'クラス'):prepare(frame,self.cfg(split='time',date='date'))
 def test_transformer_selects_only_fit_statistics(self):
    frame=pd.DataFrame({'text':['x']*10+['long story '+str(i)+'x'*45 for i in range(30)]})
    tr=preprocessor(frame).fit(frame.iloc[:10]);self.assertEqual(tr.transformer_.transformers_[0][0],'category')
    full=preprocessor(frame).fit(frame);self.assertEqual(full.transformer_.transformers_[0][0],'text0')
 def test_final_once_and_new_prediction(self):
    d=self.fixture();self.assertTrue(all(r['status']=='completed' for r in read_json(d/'result.json')['rows']))
    calls=[]
    from analysis import engine
    original=engine.metrics
    def count(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    with patch('analysis.engine.metrics',side_effect=count):
        with ThreadPoolExecutor(2) as pool:results=list(pool.map(lambda _:finalize(d,'logistic','理由'),range(2)))
    self.assertEqual(len(calls),1);self.assertEqual(results[0],results[1]);self.assertTrue((d/'adopted.joblib').exists())
    r=self.client.post('/api/analysis/runs/'+d.name+'/predict',json={'rows':self.df.drop(columns='target').head(3).to_dict('records')})
    self.assertEqual(r.status_code,200,r.text);self.assertEqual(len(r.json()['predictions']),3)
    r=self.client.post('/api/analysis/runs/'+d.name+'/decision',json={'model':'dummy','status':'adopted','reason':'変更'})
    self.assertEqual(r.status_code,400)
 def test_model_save_failure_preserves_seal_and_recovers(self):
    d=self.fixture();from analysis import engine
    original=engine.atomic_model;calls=[]
    def fail_on_second(model,path):
        calls.append(1)
        if len(calls)==2:raise OSError('保存失敗')
        return original(model,path)
    with patch('analysis.engine.atomic_model',side_effect=fail_on_second):
        with self.assertRaises(OSError):finalize(d,'logistic','理由')
    self.assertFalse((d/'final.json').exists());self.assertIn('result',read_json(d/'finalizing.json'))
    with patch('analysis.engine.metrics',side_effect=AssertionError('二度評価した')):finalize(d,'logistic','理由')
    self.assertTrue((d/'adopted.joblib').exists());self.assertTrue((d/'final.json').exists())
 def test_unsupervised_adoption_and_predict(self):
    d=self.fixture(task='clustering',target=None,models=['kmeans']);final=finalize(d,'kmeans','クラスタを確認した')
    self.assertIsNone(final['metrics']);r=self.client.post('/api/analysis/runs/'+d.name+'/predict',json={'rows':self.df.head(2).to_dict('records')});self.assertEqual(r.status_code,200,r.text)
 def test_timeseries_adoption_future_dates_and_bad_gap(self):
    self.df=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=60),'a':np.sin(np.arange(60)),'target':10+np.arange(60)*.1+np.sin(np.arange(60))})
    d=self.fixture(task='timeseries',date='date',models=['arima'],split='time');finalize(d,'arima','時間順の誤差で採用')
    r=self.client.post('/api/analysis/runs/'+d.name+'/predict',json={'rows':[{'date':'2020-03-01'},{'date':'2020-03-02'}]});self.assertEqual(r.status_code,200,r.text)
    r=self.client.post('/api/analysis/runs/'+d.name+'/predict',json={'rows':[{'date':'2020-03-03'}]});self.assertEqual(r.status_code,400)
 def test_atomic_json_failure_retains_original(self):
    p=self.root/'state.json';write_json(p,{'reason':'元'})
    with patch('analysis.engine.os.replace',side_effect=PermissionError('locked')):
        with self.assertRaises(PermissionError):write_json(p,{'reason':'更新'})
    self.assertEqual(read_json(p)['reason'],'元')
 def test_real_2d_cnn_and_gradcam_from_pixel_table(self):
    rng=np.random.default_rng(42);x=rng.normal(0,.15,(80,4,4));y=np.arange(80)%2;x[y==0,:,1]+=1;x[y==1,1,:]+=1
    self.df=pd.DataFrame(x.reshape(80,16),columns=[f'pixel_{i}' for i in range(16)]);self.df['target']=y
    names=self.df.drop(columns='target').columns.tolist()
    d=self.fixture(models=['cnn'],image_shape=[4,4],epochs=3,features=names[1:]+names[:1])
    self.assertEqual(read_json(d/'metadata.json')['features'],names)
    self.assertEqual(next(r for r in read_json(d/'result.json')['rows'] if r['model']=='cnn')['status'],'completed')
    context=joblib.load(d/'cnn.joblib');model=context['model'].named_steps['model'];from torch import nn
    self.assertIsInstance(model.net_.conv,nn.Conv2d);z=context['Z'];self.assertEqual(model.diagnostics(z[:3])['gradcam'].shape,(3,4,4))
    from analysis.graphs import draw
    context['config']['selected_model']='cnn';graph=draw(context,'g76',d/'graphs'/'cnn');self.assertEqual(graph['status'],'rendered',graph);self.assertIn('Conv2D',graph['note'])
    finalize(d,'cnn','画像ピクセルで確認した');r=self.client.post('/api/analysis/runs/'+d.name+'/predict',json={'rows':self.df.head(3).drop(columns='target').to_dict('records')});self.assertEqual(r.status_code,200,r.text)
 def test_legacy_profile_lists_class_values_without_mutating_file(self):
    r=self.client.post('/api/analysis/upload',files={'file':('data.csv',self.df.to_csv(index=False).encode(),'text/csv')});id=r.json()['id']
    file=self.root/'data'/id/'profile.json';value=read_json(file)
    for c in value['columns']:c.pop('class_values',None)
    write_json(file,value)
    returned=self.client.get('/api/analysis/data').json()[0]
    target=next(c for c in returned['columns'] if c['name']=='target');self.assertEqual(target['class_values'],['0','1'])
    self.assertNotIn('class_values',read_json(file)['columns'][0])
 def test_real_worker_from_upload_to_graph(self):
    r=self.client.post('/api/analysis/upload',files={'file':('data.csv',self.df.to_csv(index=False).encode(),'text/csv')});self.assertEqual(r.status_code,200,r.text)
    body=self.cfg(data_id=r.json()['id']);response=self.client.post('/api/analysis/runs',json=body);self.assertEqual(response.status_code,200,response.text);id=response.json()['id']
    for _ in range(100):
        r=self.client.get('/api/analysis/runs/'+id).json()
        if r['status'] not in ('queued','running'):break
        time.sleep(.2)
    self.assertEqual(r['status'],'completed',r);self.assertTrue(all(row['status']=='completed' for row in r['rows']))
    r=self.client.post(f'/api/analysis/runs/{id}/graphs/logistic/g30',json={});self.assertEqual(r.status_code,200,r.text)
    for _ in range(100):
        g=self.client.get(f'/api/analysis/runs/{id}/graphs/logistic').json()[29]
        if g['status']!='queued':break
        time.sleep(.2)
    self.assertEqual(g['status'],'rendered',g);self.assertEqual(self.client.get(f'/api/analysis/runs/{id}/graphs/logistic/g30.png').status_code,200)
    r=self.client.post('/api/analysis/runs/'+id+'/decision',json={'model':'logistic','status':'adopted','reason':'比較した'});self.assertEqual(r.status_code,200)
    self.assertEqual(len(self.client.get('/api/analysis/runs/'+id).json()['decision_history']),1)
    self.assertEqual(self.client.post('/api/analysis/runs/'+id+'/final',json={}).status_code,200)
    self.assertEqual(self.client.get('/api/analysis/runs/'+id+'/export').status_code,200)

if __name__=='__main__':unittest.main()
