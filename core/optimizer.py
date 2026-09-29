import numpy as np

def optimize_spm(viscosity: np.ndarray) -> np.ndarray:
    """M6: Fast loop MPC surrogate to adjust SPM based on predicted viscosity"""
    # SPM slows down as viscosity rises to avoid rod float
    spm = 8.0 - (viscosity / 12000) * 4.0
    return np.clip(spm, 1.0, 12.0)
