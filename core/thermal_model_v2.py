import numpy as np

def calculate_temperature_decay_with_uncertainty(days: np.ndarray, steam_vol: float, soak_time: int, t_res: float = 47.0, error_multiplier: float = 1.0):
    """M1: Thermal decay with a confidence band (uncertainty) and drift injection"""
    t_max = t_res + (steam_vol * 0.05)
    tau = 25 + soak_time * 1.5
    
    # Base predicted decay
    base_temp = t_res + (t_max - t_res) * np.exp(-days / tau)
    
    # Introduce uncertainty (confidence interval grows over time)
    uncertainty_band = (days * 0.05) + 2.0 
    
    # The "Actual" well behavior might deviate (simulate error)
    actual_temp = base_temp - (days * 0.15 * error_multiplier) # cools faster in reality
    
    return base_temp, actual_temp, uncertainty_band

def calculate_viscosity(temp_array: np.ndarray, t_res: float = 47.0) -> np.ndarray:
    """Walther-type approximation for viscosity"""
    return 12000 * np.exp(-0.06 * (temp_array - t_res))
