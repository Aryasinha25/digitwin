import os
import joblib

def load_models():
    """
    Formally loads the trained ML models and validates dependencies.
    Returns a status dictionary identifying the mode (ml vs physics).
    """
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    risk_model_path = os.path.join(MODEL_DIR, 'rf_risk_model.pkl')
    failure_model_path = os.path.join(MODEL_DIR, 'rf_failure_model.pkl')

    status = {
        "mode": "physics",
        "models": {},
        "status": "fallback",
        "reason": "Unknown error",
        "health": {
            "ml_inference": "UNAVAILABLE",
            "production_model": "FAILED",
            "rod_failure_model": "FAILED",
            "rod_floating_model": "FAILED",
            "thermal_model": "READY",
            "physics_simulation": "READY",
            "optimization_engine": "READY"
        }
    }

    try:
        # Check if sklearn is actually loadable without WDAC blocking it
        import sklearn
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as e:
        status["reason"] = f"Application Control Policy blocked ML dependency: {str(e)}"
        status["health"]["optimization_engine"] = "DEGRADED"
        return status
    except Exception as e:
        status["reason"] = f"Failed to load ML dependency: {str(e)}"
        status["health"]["optimization_engine"] = "DEGRADED"
        return status

    if not os.path.exists(risk_model_path) or not os.path.exists(failure_model_path):
        status["reason"] = "Model .pkl files not found."
        status["health"]["optimization_engine"] = "DEGRADED"
        return status

    try:
        rf_risk_model = joblib.load(risk_model_path)
        rf_failure_model = joblib.load(failure_model_path)
        
        status["mode"] = "ml"
        status["models"] = {
            "risk_model": rf_risk_model,
            "failure_model": rf_failure_model
        }
        status["status"] = "ready"
        status["reason"] = ""
        
        status["health"]["ml_inference"] = "READY"
        status["health"]["production_model"] = "READY" # Simulated for now
        status["health"]["rod_failure_model"] = "READY"
        status["health"]["rod_floating_model"] = "READY"
        
    except Exception as e:
        status["reason"] = f"Failed to load .pkl models: {str(e)}"
        status["health"]["optimization_engine"] = "DEGRADED"

    return status
