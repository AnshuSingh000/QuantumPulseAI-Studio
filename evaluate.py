"""Reproducible multi-angle evaluation. Run: python evaluate.py"""
import csv
from pathlib import Path
import numpy as np
import torch
from neural_network import PulsePredictionNetwork
from grape_optimizer import optimize_pulse
ROOT=Path(__file__).resolve().parent

def main():
    checkpoint=ROOT/'models'/'pulse_model.pt'
    if not checkpoint.exists():
        raise SystemExit('Train first: python train.py')
    model=PulsePredictionNetwork()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.eval()
    rows=[]
    for i,angle in enumerate(np.linspace(0,np.pi,25)):
        with torch.no_grad():
            pulse=model(torch.tensor([[angle]],dtype=torch.float32)).numpy().flatten()
        methods={'random':dict(seed=42+i),
                 'constant':dict(initial_pulse=np.full(20,np.clip(angle,-4,4))),
                 'deep_learning':dict(initial_pulse=pulse)}
        for name,kwargs in methods.items():
            r=optimize_pulse(angle,**kwargs)
            rows.append(dict(angle_degrees=round(np.degrees(angle),3),method=name,
                             initial_fidelity=r['initial_fidelity'],final_fidelity=r['fidelity'],
                             iterations=r['iterations'],seconds=r['seconds'],success=r['success']))
    path=ROOT/'results'/'evaluation.csv';path.parent.mkdir(exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print('Saved',path)
    for method in ('random','constant','deep_learning'):
        subset=[r for r in rows if r['method']==method]
        print(method,'mean final fidelity',np.mean([r['final_fidelity'] for r in subset]),
              'mean iterations',np.mean([r['iterations'] for r in subset]))
if __name__=='__main__':main()
