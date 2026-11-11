"""FastAPI backend serving an HTML/CSS/JS interface and real simulation results."""
import csv
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from neural_network import PulsePredictionNetwork
from quantum_simulator import target_rx, simulate_pulse, gate_fidelity
from grape_optimizer import optimize_pulse

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models'/'pulse_model.pt'
LOG=ROOT/'results'/'inference_log.csv'
LOCK=threading.Lock()
app=FastAPI(title='QuantumPulseAI API',version='1.0.0')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
_cached_model=None
_cached_mtime=None

class AngleRequest(BaseModel):
    angle: float=Field(ge=0,le=180)

class BenchmarkRequest(AngleRequest):
    random_trials: int=Field(default=10,ge=1,le=30)
    max_iterations: int=Field(default=150,ge=10,le=300)

def load_model():
    global _cached_model,_cached_mtime
    if not MODEL.exists():
        raise HTTPException(status_code=409,detail='Model not trained. Run python train.py first.')
    mtime=MODEL.stat().st_mtime_ns
    with LOCK:
        if _cached_model is None or _cached_mtime!=mtime:
            model=PulsePredictionNetwork()
            try:
                model.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=True))
            except Exception as exc:
                raise HTTPException(status_code=500,detail=f'Unable to load model: {exc}') from exc
            model.eval()
            _cached_model=model;_cached_mtime=mtime
        return _cached_model

def predict(theta):
    model=load_model()
    with torch.no_grad():
        return model(torch.tensor([[theta]],dtype=torch.float32)).numpy().flatten()

def log_prediction(row):
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOCK:
        exists=LOG.exists() and LOG.stat().st_size>0
        with LOG.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=row.keys())
            if not exists:writer.writeheader()
            writer.writerow(row)

@app.get('/')
def home():return FileResponse(ROOT/'templates'/'index.html')

@app.get('/api/status')
def status():
    return {'model_ready':MODEL.exists(),'model_version':'pulse_model_v1' if MODEL.exists() else None,
            'mlflow_tracking':'local SQLite','simulator':'single qubit / X rotation',
            'mode':'real computation — no mock metrics'}

@app.post('/api/predict')
def prediction(request:AngleRequest):
    start=time.perf_counter()
    theta=np.deg2rad(request.angle)
    pulse=predict(theta)
    fidelity=gate_fidelity(simulate_pulse(pulse),target_rx(theta))
    elapsed=(time.perf_counter()-start)*1000
    log_prediction({'timestamp':datetime.now(timezone.utc).isoformat(),
                    'angle_degrees':request.angle,'fidelity':fidelity,
                    'prediction_pipeline_ms':elapsed,'model_version':'pulse_model_v1'})
    return {'angle':request.angle,'pulse':pulse.tolist(),'fidelity':fidelity,
            'time_ms':elapsed,'model_version':'pulse_model_v1'}

@app.post('/api/benchmark')
def benchmark(request:BenchmarkRequest):
    theta=np.deg2rad(request.angle)
    ml_start=time.perf_counter();pulse=predict(theta)
    inference_ms=(time.perf_counter()-ml_start)*1000
    generator=np.random.default_rng()
    seeds=generator.integers(0,2**32-1,size=request.random_trials)
    random_runs=[]
    for seed in seeds:
        r=optimize_pulse(theta,seed=int(seed),maxiter=request.max_iterations)
        random_runs.append({'seed':int(seed),**r})
    constant=optimize_pulse(theta,initial_pulse=np.full(20,np.clip(theta,-4,4)),maxiter=request.max_iterations)
    ml=optimize_pulse(theta,initial_pulse=pulse,maxiter=request.max_iterations)
    def aggregate(runs):
        return {'initial_fidelity':float(np.mean([r['initial_fidelity'] for r in runs])),
                'fidelity':float(np.mean([r['fidelity'] for r in runs])),
                'iterations':float(np.mean([r['iterations'] for r in runs])),
                'seconds':float(np.mean([r['seconds'] for r in runs]))}
    return {'angle':request.angle,'random_trials':random_runs,
            'random_mean':aggregate(random_runs),'constant':constant,'deep_learning':ml,
            'inference_ms':inference_ms,'note':'Random values are per-trial means; DL time excludes model loading. Compare final fidelities before iteration counts.'}

@app.get('/api/training')
def training():
    path=ROOT/'results'/'training_history.json'
    summary=ROOT/'results'/'training_summary.json'
    return {'history':json.loads(path.read_text()) if path.exists() else None,
            'summary':json.loads(summary.read_text()) if summary.exists() else None}

@app.get('/api/monitoring')
def monitoring():
    if not LOG.exists():return {'predictions':[],'count':0}
    with LOCK:
        with LOG.open(newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('angle_degrees','fidelity','prediction_pipeline_ms'):
            r[k]=float(r[k])
    return {'predictions':rows[-200:],'count':len(rows),
            'average_fidelity':float(np.mean([r['fidelity'] for r in rows])) if rows else None,
            'average_latency_ms':float(np.mean([r['prediction_pipeline_ms'] for r in rows])) if rows else None}
