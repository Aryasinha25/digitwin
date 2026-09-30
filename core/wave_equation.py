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

def diagnose_failure_mode(viscosity: np.ndarray, spm: np.ndarray, float_risk: np.ndarray) -> np.ndarray:
    """
    Multi-class failure diagnosis heuristics.
    Categorizes the primary risk based on physics interactions:
    - Normal (0): Operations optimal
    - Rod Float (1): Viscosity high, preventing rod fall
    - Fluid Pound (2): Fast pumping, low viscosity (incomplete barrel fill)
    - Gas Locking (3): Very hot/low pressure, gas breakout (simplified proxy via extreme low visc)
    """
    diagnosis = np.full_like(spm, "Normal", dtype=object)
    
    # Heuristics
    diagnosis[float_risk > 60] = "Rod Float"
    
    # Fluid pound often happens if we pump too fast when fluid is very mobile (not enough inflow)
    pound_condition = (spm > 9.0) & (viscosity < 2000) & (float_risk <= 60)
    diagnosis[pound_condition] = "Fluid Pound"
    
    # Gas interference at very low viscosity (highly mobilized, gas expansion)
    gas_condition = (viscosity < 500) & (spm > 5.0) & ~pound_condition
    diagnosis[gas_condition] = "Gas Interference"
    
    return diagnosis

def generate_dynacard(viscosity: float, spm: float, base_load: float = 5000.0):
    """
    Physics-informed visual approximation of a surface dynamometer card.
    NOTE: Generates parameterized shape for demo UI, actual risk/load metrics
    come from the physics solvers in this module.
    
    Returns standard position and load arrays that form a sealed polygon.
    As viscosity rises (rod float), the lower left corner rounds/cuts upwards.
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    position = 0.5 * (1 - np.cos(theta)) 
    
    upstroke = (theta <= np.pi)
    load = np.zeros_like(theta)
    
    fric_factor = (viscosity / 12000.0) 
    vel_factor = (spm / 8.0)
    
    peak_load = base_load + (2000 * fric_factor * vel_factor)
    min_load = base_load * 0.4 - (500 * fric_factor)
    
    load[upstroke] = base_load + (peak_load - base_load) * np.sin(theta[upstroke])**0.2
    
    float_severity = max(0, fric_factor * vel_factor - 0.5)
    down_curve = np.abs(np.sin(theta[~upstroke]))**(0.2 + float_severity * 2)
    load[~upstroke] = min_load + (base_load * 0.6) * down_curve
    
    noise = np.random.normal(0, peak_load * 0.01, size=len(theta))
    return position, load + noise
