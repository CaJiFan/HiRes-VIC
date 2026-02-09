import torch
from geometry import so3_exp_map, so3_log_map


# tests/test_geometry.py
def test_so3_consistency():
    tau = torch.randn(10, 3)
    R = so3_exp_map(tau)
    tau_recon = so3_log_map(R)
    assert torch.allclose(tau, tau_recon, atol=1e-5)