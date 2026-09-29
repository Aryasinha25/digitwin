import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os

def train_models():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_training_data.csv')
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Features (What the AI sees)
    X = df[['Temperature_C', 'Viscosity_cP', 'SPM', 'Stroke_Length_m']]
    
    # Targets (What the AI must predict)
    y_risk = df['Float_Risk_Pct']
    y_failure = df['Failure_Occurred']
    
    # Split data
    X_train, X_test, yr_train, yr_test, yf_train, yf_test = train_test_split(X, y_risk, y_failure, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor for Float Risk...")
    rf_risk = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_risk.fit(X_train, yr_train)
    risk_preds = rf_risk.predict(X_test)
    print(f"Float Risk RMSE: {mean_squared_error(yr_test, risk_preds, squared=False):.2f}%")
    
    print("Training Random Forest Classifier for Failure Prediction...")
    rf_failure = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_failure.fit(X_train, yf_train)
    fail_preds = rf_failure.predict(X_test)
    print(f"Failure Prediction Accuracy: {accuracy_score(yf_test, fail_preds) * 100:.2f}%")
    
    # Save the models
    risk_model_path = os.path.join(model_dir, 'rf_risk_model.pkl')
    failure_model_path = os.path.join(model_dir, 'rf_failure_model.pkl')
    
    joblib.dump(rf_risk, risk_model_path)
    joblib.dump(rf_failure, failure_model_path)
    print(f"Models saved to {model_dir}")

if __name__ == "__main__":
    train_models()
