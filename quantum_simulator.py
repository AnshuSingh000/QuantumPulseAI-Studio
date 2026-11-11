import numpy as np
from scipy.linalg import expm
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)

def target_rx(theta):
    return expm(-0.5j * float(theta) * SIGMA_X)

def simulate_pulse(pulse, total_time=1.0, drift=1.0):
    pulse = np.asarray(pulse, dtype=float)
    if pulse.ndim != 1 or len(pulse) == 0 or not np.all(np.isfinite(pulse)):
        raise ValueError('pulse must be a nonempty finite 1D array')
    dt = total_time / len(pulse)
    u = IDENTITY.copy()
    for amplitude in pulse:
        u = expm(-0.5j * dt * (drift * SIGMA_Z + amplitude * SIGMA_X)) @ u
    return u

def gate_fidelity(actual, target):
    return float(np.abs(np.trace(target.conj().T @ actual))**2 / 4)
