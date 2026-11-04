import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.engine import run_experiment,read_json
if __name__=='__main__':
    mode,directory=sys.argv[1:3]
    if mode=='run':run_experiment(directory)
    else:
        import joblib
        from analysis.graphs import draw
        model,graph=sys.argv[3:5];d=Path(directory);ctx=joblib.load(d/f'{model}.joblib');ctx['config']['selected_model']=model;ctx['rows']=read_json(d/'result.json')['rows'];draw(ctx,graph,d/'graphs'/model)
