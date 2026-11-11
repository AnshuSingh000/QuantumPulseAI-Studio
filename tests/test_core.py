import numpy as np
import pytest
from quantum_simulator import target_rx,simulate_pulse,gate_fidelity
from grape_optimizer import optimize_pulse
from neural_network import PulsePredictionNetwork

def test_target_identity():
    assert np.allclose(target_rx(0),np.eye(2))

def test_gate_fidelity_is_one_for_identical_unitaries():
    u=target_rx(np.pi/3)
    assert gate_fidelity(u,u)==pytest.approx(1,abs=1e-10)

def test_simulator_unitary():
    u=simulate_pulse(np.zeros(20))
    assert np.allclose(u.conj().T@u,np.eye(2),atol=1e-10)

def test_pulse_validation():
    with pytest.raises(ValueError):simulate_pulse([])
    with pytest.raises(ValueError):simulate_pulse([float('nan')])

def test_grape_amplitude_constraints():
    r=optimize_pulse(np.pi/2,n_steps=5,maxiter=3)
    assert len(r['pulse'])==5
    assert max(np.abs(r['pulse']))<=4+1e-9
    assert 0<=r['fidelity']<=1+1e-9

def test_network_output():
    import torch
    net=PulsePredictionNetwork()
    result=net(torch.tensor([[0.0],[1.0]],dtype=torch.float32))
    assert result.shape==(2,20)
    assert torch.all(result.abs()<=4.0)
