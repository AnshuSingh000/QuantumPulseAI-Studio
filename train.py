"""Train a physics-informed pulse predictor. Run: python train.py"""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from neural_network import PulsePredictionNetwork

ROOT = Path(__file__).resolve().parent

def batch_fidelity(pulses, angles, drift=1.0, total_time=1.0):
    """Differentiable 2x2 complex matrix evolution, one matrix per time slice."""
    b, n = pulses.shape
    device = pulses.device
    x = torch.tensor([[0,1],[1,0]], device=device, dtype=torch.complex64)
    z = torch.tensor([[1,0],[0,-1]], device=device, dtype=torch.complex64)
    eye = torch.eye(2, device=device, dtype=torch.complex64).expand(b,-1,-1).clone()
    u = eye
    dt = total_time / n
    for k in range(n):
        h = 0.5 * (drift*z[None,:,:] + pulses[:,k,None,None]*x[None,:,:])
        step = torch.matrix_exp(-1j * dt * h)
        u = step @ u
    target = torch.matrix_exp(-0.5j * angles[:,None,None] * x[None,:,:])
    overlap = torch.diagonal(target.conj().transpose(-1,-2) @ u, dim1=-2,dim2=-1).sum(-1)
    return overlap.abs().square() / 4

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    angles = np.random.default_rng(42).uniform(0,np.pi,700).astype('float32')
    train_a, val_a, test_a = angles[:500], angles[500:600], angles[600:]
    np.savez(ROOT/'results'/'angle_splits.npz', train=train_a,val=val_a,test=test_a)
    model = PulsePredictionNetwork()
    opt = torch.optim.Adam(model.parameters(),lr=0.002)
    train_tensor = torch.from_numpy(train_a).unsqueeze(-1)
    val_tensor = torch.from_numpy(val_a).unsqueeze(-1)
    history = {'train_loss':[], 'val_loss':[]}
    best = float('inf')
    try:
        import mlflow
        mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}')
        mlflow.set_experiment('QuantumPulseAI Training')
        run = mlflow.start_run()
        mlflow.log_params({'epochs':120,'learning_rate':0.002,'train_size':500,'val_size':100,'test_size':100,'n_steps':20,'seed':42,'objective':'gate_infidelity'})
    except Exception as exc:
        mlflow = None
        print('MLflow unavailable; training still proceeds:', exc)
    for epoch in range(120):
        model.train()
        order = torch.randperm(len(train_tensor))
        for ids in order.split(32):
            x = train_tensor[ids]
            loss = (1-batch_fidelity(model(x),x.squeeze(-1))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            train_loss = float((1-batch_fidelity(model(train_tensor),train_tensor.squeeze(-1))).mean())
            val_loss = float((1-batch_fidelity(model(val_tensor),val_tensor.squeeze(-1))).mean())
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(),ROOT/'models'/'pulse_model.pt')
        if mlflow is not None:
            mlflow.log_metrics({'train_infidelity':train_loss,'val_infidelity':val_loss},step=epoch+1)
        if (epoch+1)%10 == 0:
            print(f'Epoch {epoch+1:3d}/120 | train {train_loss:.5f} | val {val_loss:.5f}')
    (ROOT/'results'/'training_history.json').write_text(json.dumps(history,indent=2))
    model.load_state_dict(torch.load(ROOT/'models'/'pulse_model.pt',map_location='cpu',weights_only=True))
    model.eval()
    with torch.no_grad():
        test_inputs = torch.from_numpy(test_a).unsqueeze(-1)
        test_fidelity = float(batch_fidelity(model(test_inputs), test_inputs.squeeze(-1)).mean())
    summary = {'best_validation_infidelity':best,'held_out_test_mean_fidelity':test_fidelity,'epochs':120}
    (ROOT/'results'/'training_summary.json').write_text(json.dumps(summary,indent=2))
    if mlflow is not None:
        mlflow.log_metrics(summary)
        mlflow.log_artifact(str(ROOT/'models'/'pulse_model.pt'))
        mlflow.log_artifact(str(ROOT/'results'/'training_history.json'))
        mlflow.end_run()
    print('Training complete:',summary)

if __name__=='__main__':
    main()
