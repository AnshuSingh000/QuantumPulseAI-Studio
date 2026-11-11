# QuantumPulseAI Studio

**Deep Learning-Assisted Quantum Pulse Optimization Using GRAPE**  
Anshu Singh · Semester VII · CS3246 MLOps in Practise · RV University

A full-stack quantum control demonstration: **PyTorch physics-informed neural network + real SciPy GRAPE + FastAPI + responsive HTML/CSS/JavaScript dashboard + MLflow + automated tests + GitHub Actions CI**.

## Quick start (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python -m uvicorn app:app --reload
```

Open **http://127.0.0.1:8000**. Training is required the first time: no trained checkpoint or synthetic result is shipped as a fake demonstration. The training run takes time because it differentiates through quantum matrix exponentials.

Windows activation: `.venv\\Scripts\\activate`.

## MLflow

Training logs metrics, parameters and model artifacts to a local SQLite store. In a separate terminal:

```bash
source .venv/bin/activate
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open **http://127.0.0.1:5001**.

## Reproducible evaluation

```bash
python evaluate.py
python -m pytest tests/ -q
```

`evaluate.py` uses 25 angles and fixed random seeds, and saves `results/evaluation.csv`. The live dashboard uses *fresh* random seeds on every benchmark click, and reports random means over the selected number of trials. Optimizer convergence (`success`) does not itself imply reaching a particular fidelity threshold.

## Architecture

- `quantum_simulator.py` — 2×2 quantum time evolution and gate fidelity
- `grape_optimizer.py` — bounded L-BFGS-B with analytic Fréchet gradients
- `neural_network.py` — MLP (1 → 64 → 128 → 128 → 20), bounded pulse output
- `train.py` — 500/100/100 angle split, 120 epochs, differentiable physics loss, MLflow tracking
- `evaluate.py` — reproducible GRAPE baseline comparison
- `app.py` — FastAPI prediction, benchmarking, training and monitoring endpoints
- `templates/index.html`, `static/style.css`, `static/app.js` — responsive web dashboard
- `tests/`, `.github/workflows/tests.yml` — automated unit tests and CI workflow

## MLOps coverage

| Concept | Implementation |
|---|---|
| Experiment tracking | MLflow local SQLite metrics, parameters, artifacts |
| Model artifact | Best-validation PyTorch state dict in `models/pulse_model.pt` |
| Evaluation | Fixed-seed multi-angle benchmark and independent random trials |
| Inference logging | Timestamp, angle, fidelity, latency, model label in CSV |
| Monitoring | Fidelity history, mean latency, threshold-based warnings |
| Input validation | Pydantic bounds, pulse shape and finiteness checks |
| Automated tests | Pytest |
| Continuous integration | GitHub Actions runs tests on push / PR |
| Model versioning | Simple identifier only; **not** an MLflow Model Registry |
| Deployment | Local FastAPI server; **not** a production deployment |

## Research limitations

Single qubit, X rotations only, fixed drift, normalized duration and amplitude constraints. ML predicts a high-fidelity starting point, not guaranteed faster GRAPE convergence. Hardware noise and calibration are not modeled. Training/evaluation outcomes should be generated and reported from actual runs; no performance advantage is guaranteed for every angle.

## GitHub

The `.gitignore` excludes virtual environments, MLflow local databases, runtime inference logs and model checkpoints. To share a model checkpoint, use a GitHub Release or an approved artifact store rather than committing a large binary. Readers can reproduce the model with `python train.py`.
