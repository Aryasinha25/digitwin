import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

def generate_training_data(output_path="synthetic_training_data.csv", samples=10000):
    """
    Generates a massive dataset representing thousands of random days of operation.
    We inject physical equations but add randomized noise so the ML model has to 
    learn the patterns rather than just memorizing a clean equation.
    """
    np.random.seed(42)
    
    # Random operational parameters
    t_res = 47.0
    temperatures = np.random.uniform(47.0, 200.0, samples) # Well temps from 47C to 200C
    spm = np.random.uniform(1.0, 12.0, samples)
    stroke_length = np.random.uniform(1.0, 5.0, samples)
    
    # Base Physics
    viscosities = 12000 * np.exp(-0.06 * (temperatures - t_res))
    
    # Base float risk (equation) + Random Noise to simulate geological chaos
    base_float_risk = (viscosities / 12000) * (spm / 8.0) * 100
    noise = np.random.normal(0, 10, samples) # Add +/- 10% random Gaussian noise
    float_risk_actual = np.clip(base_float_risk + noise, 0, 100)
    
    # Classification logic: Did the rod actually snap/fail today?
    # If float risk > 80%, there's an exponentially high chance of failure.
    failure_probability = np.where(float_risk_actual > 80, 0.8, float_risk_actual / 200)
    failure_occurred = np.random.binomial(1, failure_probability)
    
    # Production calculation
    efficiency = 1.0 - (float_risk_actual / 200)
    daily_prod = (temperatures / t_res) * 50 * (spm / 6.0) * efficiency + np.random.normal(0, 5, samples)
    daily_prod = np.clip(daily_prod, 0, None)
    
    df = pd.DataFrame({
        "Temperature_C": temperatures,
        "Viscosity_cP": viscosities,
        "SPM": spm,
        "Stroke_Length_m": stroke_length,
        "Float_Risk_Pct": float_risk_actual,
        "Daily_Production_BBL": daily_prod,
        "Failure_Occurred": failure_occurred
    })
    
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {samples} rows of training data at {output_path}")

if __name__ == "__main__":
    generate_training_data(output_path=os.path.join(os.path.dirname(__file__), "synthetic_training_data.csv"))
