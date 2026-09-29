import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from core.thermal_model_v2 import calculate_temperature_decay_with_uncertainty, calculate_viscosity
from core.wave_equation import calculate_float_risk, calculate_energy
from core.optimizer import optimize_spm

def generate_synthetic_dataset(output_path="synthetic_data.csv", days_to_simulate=365):
    """Generates a synthetic dataset for ML model training"""
    days = np.arange(0, days_to_simulate, 1)
    
    # We generate a single idealized decay to show the structure.
    t_res = 47.0
    temp, _, _ = calculate_temperature_decay_with_uncertainty(days, steam_vol=3000, soak_time=7, t_res=t_res)
    visc = calculate_viscosity(temp, t_res)
    
    spm_fixed = np.full_like(days, 6.0)
    spm_optimized = optimize_spm(visc)
    
    df = pd.DataFrame({
        "Day": days,
        "Temperature_C": temp,
        "Viscosity_cP": visc,
        "SPM_Manual": spm_fixed,
        "SPM_Optimized": spm_optimized,
        "FloatRisk_Manual": calculate_float_risk(visc, spm_fixed),
        "FloatRisk_Optimized": calculate_float_risk(visc, spm_optimized),
        "Energy_Manual_kWh": calculate_energy(spm_fixed, 3.0, visc),
        "Energy_Optimized_kWh": calculate_energy(spm_optimized, 3.0, visc)
    })
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path}")

if __name__ == "__main__":
    generate_synthetic_dataset()
