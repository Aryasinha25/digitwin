import numpy as np

def optimize_spm(viscosity: np.ndarray) -> np.ndarray:
    """M6: Fast loop MPC surrogate to adjust SPM based on predicted viscosity"""
    # SPM slows down as viscosity rises to avoid rod float
    spm = 8.0 - (viscosity / 12000) * 4.0
    return np.clip(spm, 1.0, 12.0)

def optimize_css_cycle_steam(t_res: float, oil_price: float = 5800.0, steam_cost_eval_weight: float = 1.0) -> float:
    """
    CSS Slow Loop Optimizer (M5 Surrogate).
    Finds optimal steam injection volume maximizing net profit over a standard soak cycle.
    Subject to soft safety penalty limits.
    """
    best_vol = 3000.0
    best_profit = -np.inf
    
    days = np.arange(0, 60, 1)
    
    for vol in np.arange(1000, 6000, 250):
        # Surrogate temperature response evaluation (Sub-linear diminishing returns due to heat loss)
        t_max = t_res + (np.power(vol, 0.8) * 0.35)
        tau = 25 + 7 * 1.5
        temp = t_res + (t_max - t_res) * np.exp(-days / tau)
        
        penalty = 0
        if t_max > 250.0:
            penalty = 1e8
            
        daily_prod_eval = (temp / t_res) * 40.0 
        cumulative_prod = np.sum(daily_prod_eval)
        
        # Profit = Revenue - Steam Cost (amplified for surrogate behavior to force an optimum point) - Penalty
        expected_profit = (cumulative_prod * oil_price) - (vol * 4500.0 * steam_cost_eval_weight) - penalty
        if expected_profit > best_profit:
            best_profit = expected_profit
            best_vol = vol
            
    return float(best_vol)
