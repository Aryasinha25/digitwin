import numpy as np

def calculate_float_risk(viscosity: np.ndarray, spm: np.ndarray) -> np.ndarray:
    """M2: Surrogate model for rod float risk (%)"""
    risk = (viscosity / 12000) * (spm / 8.0) * 100
    return np.clip(risk, 0, 100)

def calculate_fatigue_damage(float_risk: np.ndarray, spm: np.ndarray) -> np.ndarray:
    """M7: Cumulative rod fatigue damage"""
    damage_rate = (float_risk / 100) * (spm / 5.0)
    return np.cumsum(damage_rate) * 0.1

def calculate_energy(spm: np.ndarray, stroke_length: float, viscosity: np.ndarray) -> np.ndarray:
    """Energy consumption in kWh/bbl"""
    return 15 + (spm * stroke_length * (viscosity / 1000) * 0.2)

def calculate_production(temp: np.ndarray, t_res: float, spm: np.ndarray, float_risk: np.ndarray) -> np.ndarray:
    """Daily production heuristics based on mobility and pump efficiency"""
    efficiency = 1.0 - (float_risk / 200)
    daily_prod = (temp / t_res) * 50 * (spm / 6.0) * efficiency
    return daily_prod
