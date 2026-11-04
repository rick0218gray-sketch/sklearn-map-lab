import io, json, os, re, sys, uuid, subprocess, threading, time, hashlib, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
from .engine import profile,recommendations,write_json,read_json,clean,finalize,run_lock,normalize_features,resolve_metric
from .catalog import GRAPHS,FAMILIES,SPECS
ROOT=Path(os.environ.get('SKLEARN_ANALYSIS_DIR',str(Path(__file__).resolve().parents[2]/'analysis_data')));ROOT.mkdir(exist_ok=True,parents=True)
router=APIRouter(prefix='/api/analysis');jobs={};job_lock=threading.Lock();slots=threading.Semaphore(2)
class Config(BaseModel):
    data_id:str
    task:Literal['classification','regression','clustering','anomaly','reduction','timeseries']='classification'
    target:str|None=None
    features:list[str]=[]
    date:str|None=None
    group:str|None=None
    split:Literal['random','time']='random'
    metric:str|None=None
    models:list[str]=[]
    cv:int=Field(3,ge=2,le=5)
    seed:int=42
    test_size:float=Field(.2,ge=.1,le=.4)
    epochs:int=Field(25,ge=2,le=100)
    period:int=Field(7,ge=2,le=365)
    search:bool=False
    success_threshold:float|None=None
    max_fit_seconds:float|None=Field(None,gt=0)
    timeout_seconds:int=Field(300,ge=10,le=3600)
    purpose:str=''
    prediction:str=''
    positive_class:int=1
    image_shape:list[int]|None=None

def path(kind,id):
    if not re.fullmatch('[0-9a-f]{32}',id):raise HTTPException(400,'不正なID')
    p=ROOT/kind/id
    if not p.exists():raise HTTPException(404,'記録がありません')
    return p

def stop_run(directory,status,error=None):
    """実験全体と計算途中の行を同じ停止状態にそろえる。完了済みの行はそのまま残す。"""
    result=read_json(directory/'result.json')
    for row in result.get('rows',[]):
        if row.get('status')=='running':row['status']=status
    result['status']=status
    if error:result['error']=error
    write_json(directory/'result.json',result);return result

def start_worker(directory,mode='run',args=(),timeout=300):
    key=f'{directory.name}:{mode}:'+':'.join(args)
    def work():
        with slots:
            logpath=directory/('worker.log' if mode=='run' else f'graph_{args[-1]}.log')
            if mode=='run' and (directory/'cancel').exists():return
            with logpath.open('w',encoding='utf-8') as output:
                env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
                process=subprocess.Popen([sys.executable,str(Path(__file__).parent/'worker.py'),mode,str(directory),*args],stdout=output,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                with job_lock:jobs[key]=process
                try:
                    code=process.wait(timeout=timeout)
                    if mode=='run' and (directory/'cancel').exists():
                        # 停止要求後にワーカーが書いた途中状態を上書きし、画面が計算中のまま残らないようにする。
                        if read_json(directory/'result.json').get('status')!='completed':stop_run(directory,'cancelled')
                    elif code!=0:
                        if mode=='run':stop_run(directory,'failed','分析プロセスが失敗しました。worker.logを確認')
                        elif mode=='graph':write_json(directory/'graphs'/args[0]/f'{args[1]}.json',{'id':args[1],'status':'failed','reason':'描画プロセスが終了しました。ログを確認してください'})
                except subprocess.TimeoutExpired:
                    process.terminate();process.wait(timeout=10)
                    if mode=='run':stop_run(directory,'timed_out',f'上限{timeout}秒で停止。完了した結果は保持')
                    else:
                        write_json(directory/'graphs'/args[0]/f'{args[1]}.json',{'id':args[1],'status':'timed_out','reason':f'{timeout}秒で停止'})
                finally:
                    with job_lock:jobs.pop(key,None)
    threading.Thread(target=work,daemon=True).start()

@router.get('/catalog')
def catalog():return {'graphs':GRAPHS,'families':FAMILIES,'models':SPECS}
@router.get('/data')
def list_data():
    base=ROOT/'data';profiles=[]
    if not base.exists():return profiles
    for file in sorted(base.glob('*/profile.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        value=read_json(file)
        old=[c['name'] for c in value['columns'] if 'class_values' not in c and c['unique']<=50]
        if old:
            frame=pd.read_parquet(file.parent/'data.parquet',columns=old)
            for column in value['columns']:
                if column['name'] in frame:column['class_values']=sorted(frame[column['name']].dropna().astype(str).unique().tolist())
        profiles.append(value)
    return profiles
@router.post('/upload')
async def upload(file:UploadFile=File(...)):
    content=await file.read(50*1024*1024+1)
    if len(content)>50*1024*1024:raise HTTPException(400,'ファイル上限は50MB')
    suffix=Path(file.filename or '').suffix.lower()
    try:
        if suffix in ('.csv','.tsv'):
            frame=None
            for encoding in ('utf-8-sig','cp932'):
                try:frame=pd.read_csv(io.BytesIO(content),encoding=encoding,sep='\t' if suffix=='.tsv' else ',',nrows=100001);break
                except UnicodeDecodeError:continue
            if frame is None:raise ValueError('UTF-8/CP932のCSVとして読み込めません')
        elif suffix=='.xlsx':
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                if sum(info.file_size for info in z.infolist())>150*1024*1024:raise ValueError('展開後Excelが大きすぎます')
            book=pd.ExcelFile(io.BytesIO(content));frame=pd.read_excel(book,sheet_name=0,nrows=100001)
        elif suffix=='.parquet':frame=pd.read_parquet(io.BytesIO(content))
        elif suffix=='.json':frame=pd.read_json(io.BytesIO(content))
        else:raise ValueError('CSV / TSV / XLSX / JSON / Parquetを選んでください')
        if len(frame)>100000 or len(frame.columns)>500:raise ValueError('上限100000行・500列')
        if len(frame)==0:raise ValueError('データ行がありません')
        frame.columns=frame.columns.astype(str)
        if frame.columns.duplicated().any():raise ValueError('列名が重複しています')
        id=uuid.uuid4().hex;d=ROOT/'data'/id;d.mkdir(parents=True)
        frame.to_parquet(d/'data.parquet',index=False);p=profile(frame);p.update(id=id,filename=Path(file.filename).name,sha256=hashlib.sha256(content).hexdigest())
        if suffix=='.xlsx':p['warnings'].append(f'Excelの先頭シート「{book.sheet_names[0]}」を読込。全{len(book.sheet_names)}シート')
        write_json(d/'profile.json',p)
        from .graphs import draw
        target=p['suggested_target'];features=frame.drop(columns=[target]) if target else frame
        ctx={'X':features,'model':None,'task':'data','config':{},'y':frame[target].to_numpy() if target else None,'pred':None,'prob':None,'Z':None}
        p['eda']=[draw(ctx,g,d/'eda') for g in ('g01','g04','g13','g15','g19')];write_json(d/'profile.json',p)
        return clean(p)
    except Exception as e:raise HTTPException(400,f'読込み失敗：{e}') from e
@router.get('/data/{id}/eda/{file}')
def eda_file(id:str,file:str):
    if not re.fullmatch(r'g[0-9]{2}\.png',file):raise HTTPException(400,'不正な画像名')
    p=path('data',id)/'eda'/file
    if not p.exists():raise HTTPException(404,'画像なし')
    return FileResponse(p)
@router.post('/recommend')
def recommend(config:Config):return recommendations(pd.read_parquet(path('data',config.data_id)/'data.parquet'),config.model_dump())
@router.post('/runs')
def run(config:Config):
    source=path('data',config.data_id);df=pd.read_parquet(source/'data.parquet')
    from .engine import prepare
    try:prepare(df,config.model_dump());resolve_metric(config.model_dump())
    except Exception as e:raise HTTPException(400,str(e)) from e
    id=uuid.uuid4().hex;d=ROOT/'runs'/id;d.mkdir(parents=True)
    df.to_parquet(d/'data.parquet',index=False);write_json(d/'config.json',config.model_dump());write_json(d/'result.json',{'id':id,'status':'queued','rows':[],'config':config.model_dump()})
    start_worker(d,timeout=config.timeout_seconds);return {'id':id,'status':'queued'}
@router.get('/runs')
def list_runs():
    base=ROOT/'runs';return [read_json(p) for p in sorted(base.glob('*/result.json'),key=lambda p:p.stat().st_mtime,reverse=True)] if base.exists() else []
@router.get('/runs/{id}')
def get_run(id:str):
    d=path('runs',id);result=read_json(d/'result.json')
    if (d/'decision.json').exists():result['decision']=read_json(d/'decision.json')
    if (d/'decision_history.json').exists():result['decision_history']=read_json(d/'decision_history.json')
    if (d/'finalizing.json').exists():result['finalizing']=read_json(d/'finalizing.json')
    if (d/'final.json').exists():result['final']=read_json(d/'final.json')
    return result
@router.post('/runs/{id}/cancel')
def cancel(id:str):
    d=path('runs',id);result=read_json(d/'result.json')
    if result.get('status') not in ('queued','running'):return result
    (d/'cancel').touch()
    with job_lock:
        for key,process in list(jobs.items()):
            if key.startswith(id+':run:'):process.terminate()
    return stop_run(d,'cancelled')
class Decision(BaseModel):
    model:str
    status:Literal['adopted','pending','rejected']='adopted'
    reason:str
def _decision(id:str,body:Decision):
    d=path('runs',id)
    if not body.reason.strip():raise HTTPException(400,'理由が必要です')
    if (d/'final.json').exists() or (d/'finalizing.json').exists():raise HTTPException(400,'最終テスト済み。この実験の採用モデルは変更できません')
    if not any(r['model']==body.model and r['status']=='completed' for r in read_json(d/'result.json')['rows']):raise HTTPException(400,'学習完了モデルを選択してください')
    event=dict(body.model_dump(),saved_at=pd.Timestamp.now(tz='Asia/Tokyo').isoformat())
    history=read_json(d/'decision_history.json') if (d/'decision_history.json').exists() else []
    history.append(event);write_json(d/'decision_history.json',history);write_json(d/'decision.json',event);return event
@router.post('/runs/{id}/decision')
def decision(id:str,body:Decision):
    with run_lock(path('runs',id)):return _decision(id,body)

def _final(id:str):
    d=path('runs',id)
    if not (d/'decision.json').exists():raise HTTPException(400,'モデルを採用してから実行してください')
    dec=read_json(d/'decision.json')
    if dec['status']!='adopted':raise HTTPException(400,'採用状態のモデルが必要です')
    try:return finalize(d,dec['model'],dec['reason'])
    except Exception as e:raise HTTPException(400,str(e)) from e
@router.post('/runs/{id}/final')
def final(id:str):
    with run_lock(path('runs',id)):return _final(id)

@router.get('/runs/{id}/graphs/{model}')
def graph_list(id:str,model:str):
    d=path('runs',id);validate_model(model);base=d/'graphs'/model
    return [dict(g,**read_json(base/f"{g['id']}.json")) if (base/f"{g['id']}.json").exists() else dict(g,status='not_generated',implemented=True) for g in GRAPHS]
def validate_model(model):
    if model not in {s['id'] for s in SPECS}:raise HTTPException(400,'未登録モデル')
@router.post('/runs/{id}/graphs/{model}/{graph}')
def generate_graph(id:str,model:str,graph:str):
    d=path('runs',id);validate_model(model)
    if graph not in {g['id'] for g in GRAPHS}:raise HTTPException(400,'未登録グラフ')
    if not (d/f'{model}.joblib').exists():raise HTTPException(400,'学習済みモデルが必要')
    base=d/'graphs'/model;base.mkdir(parents=True,exist_ok=True);state=base/f'{graph}.json'
    if state.exists() and read_json(state).get('status') in ('queued','rendered'):return read_json(state)
    value=dict(next(g for g in GRAPHS if g['id']==graph),status='queued');write_json(state,value)
    start_worker(d,'graph',(model,graph),timeout=120);return value
@router.get('/runs/{id}/graphs/{model}/{graph}.png')
def graph_image(id:str,model:str,graph:str):
    validate_model(model)
    if graph not in {g['id'] for g in GRAPHS}:raise HTTPException(400,'未登録グラフ')
    p=path('runs',id)/'graphs'/model/f'{graph}.png'
    if not p.exists():raise HTTPException(404,'描画未完了')
    return FileResponse(p)
@router.get('/runs/{id}/export')
def export(id:str):
    d=path('runs',id);archive=d/'experiment.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in d.rglob('*'):
            if p.is_file() and p!=archive and p.suffix not in ('.tmp','.log'):z.write(p,p.relative_to(d))
    return FileResponse(archive,filename=f'experiment_{id}.zip')
class Predict(BaseModel):rows:list[dict]
@router.post('/runs/{id}/predict')
def predict(id:str,body:Predict):
    d=path('runs',id)
    if not (d/'adopted.joblib').exists() or not (d/'final.json').exists():raise HTTPException(400,'最終テストとモデル保存を先に完了してください')
    meta=read_json(d/'metadata.json');model=joblib.load(d/'adopted.joblib');frame=pd.DataFrame(body.rows)
    config=read_json(d/'config.json')
    if not 0<len(frame)<=10000:raise HTTPException(400,'予測は1～10000行で指定してください')
    if config['task']=='timeseries':
        if config['date'] not in frame:raise HTTPException(400,'未来の日付列が必要です')
        try:return {'predictions':clean(model.predict(pd.to_datetime(frame[config['date']]))),'dates':frame[config['date']].astype(str).tolist()}
        except Exception as e:raise HTTPException(400,str(e)) from e
    missing=set(meta['features'])-set(frame.columns)
    if missing:raise HTTPException(400,f'不足列：{sorted(missing)}')
    try:
        features=normalize_features(frame[meta['features']].copy())
        if config['task']=='reduction':
            if not hasattr(model.named_steps['model'],'transform'):raise ValueError('t-SNEは新しい行への変換を定義しません。新しいデータで別の実験を作成してください')
            return {'embedding':clean(model.transform(features))}
        if config['task']=='clustering' and not hasattr(model.named_steps['model'],'predict'):raise ValueError('階層／密度クラスタは新しい行へのpredictを提供しません。新データで別の実験を作成してください')
        pred=model.predict(features)
    except Exception as e:raise HTTPException(400,str(e)) from e
    labels=meta.get('labels')
    if labels:pred=[labels[int(x)] for x in pred]
    return {'predictions':clean(pred)}

@router.post('/runs/{id}/predict-file')
async def predict_file(id:str,file:UploadFile=File(...)):
    data=await file.read(20*1024*1024+1)
    if len(data)>20*1024*1024:raise HTTPException(400,'予測用ファイルは20MBまで')
    ext=Path(file.filename or '').suffix.lower()
    try:
        if ext=='.xlsx':
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if sum(i.file_size for i in z.infolist())>150*1024*1024:raise ValueError('展開後Excelが大きすぎます')
            frame=pd.read_excel(io.BytesIO(data),nrows=10001)
        elif ext in ('.csv','.tsv'):
            try:frame=pd.read_csv(io.BytesIO(data),encoding='utf-8-sig',sep='\t' if ext=='.tsv' else ',',nrows=10001)
            except UnicodeDecodeError:frame=pd.read_csv(io.BytesIO(data),encoding='cp932',sep='\t' if ext=='.tsv' else ',',nrows=10001)
        else:raise ValueError('CSV / TSV / XLSXを選んでください')
        frame.columns=frame.columns.astype(str)
        return predict(id,Predict(rows=clean(frame.astype(object).where(frame.notna(),None).to_dict('records'))))
    except HTTPException:raise
    except Exception as e:raise HTTPException(400,str(e)) from e
