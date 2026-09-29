# Baghewala Adaptive Twin - Environment & Reproducibility Guide

## 1. System Diagnosis & Resolution
* **Operating System:** Windows (Managed by Security Policy)
* **Python Runtime:** Python 3.11.x (via `.venv311`)
* **Security Constraints:** Windows Defender Application Control (WDAC) was aggressively blocking the execution of native `.pyd` extensions in Python 3.14 (system global). By isolating the environment into Python 3.11 and cleanly reinstalling binary wheels, the ML extensions are now loading successfully and bypassing the previous WDAC conflict.

## 2. Environment Setup (Local Inference Restored)
The application is fully restored to **ML Inference Mode**.

To reproduce this environment on any machine, follow these exact steps to use the target 3.11 environment:

### Venv Creation and Activation (PowerShell)
```bash
# Locate a Python 3.11 executable on your machine and create the venv
C:\Users\<user>\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv311
.venv311\Scripts\Activate.ps1
```

### Installation Command
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy scipy pandas scikit-learn joblib streamlit plotly
```

### Dependency Versions (Tested & Compatible in Python 3.11)
- `python == 3.11.x`
- `numpy == 2.4.6`
- `scipy == 1.17.1`
- `scikit-learn == 1.9.1`
- `pandas == 3.0.6`
- `joblib == 1.6.0`
- `streamlit == 1.64.0`
- `plotly == 7.1.0`

## 3. Model Loading & Compatibility
The models were trained using `scikit-learn==1.6.1` (in Colab) and are being loaded here using `scikit-learn==1.9.1`. 
**Warning:** You may see an `InconsistentVersionWarning` in the console during boot. This is documented and expected. The models load and perform inference successfully without retraining.

## 4. Starting the Application
**CRITICAL:** Do NOT use a global `streamlit run` command. Ensure you are using the `.venv311` executable.

```bash
.venv311\Scripts\python.exe -m streamlit run app.py
```
