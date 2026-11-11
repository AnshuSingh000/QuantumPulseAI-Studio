import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm_frechet
from quantum_simulator import SIGMA_X, SIGMA_Z, IDENTITY, target_rx, simulate_pulse, gate_fidelity

def optimize_pulse(theta, n_steps=20, total_time=1.0, max_amplitude=4.0,
                   drift=1.0, seed=42, initial_pulse=None, maxiter=150):
    rng = np.random.default_rng(seed)
    dt = total_time / n_steps
    target = target_rx(theta)
    if initial_pulse is None:
        initial = rng.uniform(-max_amplitude, max_amplitude, n_steps)
    else:
        initial = np.asarray(initial_pulse, dtype=float).copy()
        if initial.shape != (n_steps,) or not np.all(np.isfinite(initial)):
            raise ValueError(f'initial_pulse must be a finite vector of length {n_steps}')
        initial = np.clip(initial, -max_amplitude, max_amplitude)
    def objective(pulse):
        steps, derivatives = [], []
        for amp in pulse:
            a = -0.5j * dt * (drift * SIGMA_Z + amp * SIGMA_X)
            e = -0.5j * dt * SIGMA_X
            step, derivative = expm_frechet(a, e, compute_expm=True)
            steps.append(step)
            derivatives.append(derivative)
        forward = [IDENTITY.copy()]
        for step in steps:
            forward.append(step @ forward[-1])
        backward = [IDENTITY.copy() for _ in range(n_steps + 1)]
        for k in range(n_steps - 1, -1, -1):
            backward[k] = backward[k+1] @ steps[k]
        overlap = np.trace(target.conj().T @ forward[-1])
        fidelity = np.abs(overlap)**2 / 4
        grad = np.empty(n_steps)
        for k in range(n_steps):
            du = backward[k+1] @ derivatives[k] @ forward[k]
            d_overlap = np.trace(target.conj().T @ du)
            grad[k] = -np.real(np.conj(overlap) * d_overlap) / 2
        return float(1-fidelity), grad
    start = time.perf_counter()
    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      bounds=[(-max_amplitude, max_amplitude)] * n_steps,
                      options={'maxiter': maxiter, 'ftol': 1e-11})
    elapsed = time.perf_counter() - start
    return {'pulse': result.x.tolist(),
            'initial_fidelity': gate_fidelity(simulate_pulse(initial, total_time, drift), target),
            'fidelity': gate_fidelity(simulate_pulse(result.x, total_time, drift), target),
            'iterations': int(result.nit), 'evaluations': int(result.nfev),
            'success': bool(result.success), 'seconds': float(elapsed)}
