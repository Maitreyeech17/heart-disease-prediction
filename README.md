# Intelligent Network Congestion Detection using Explainable ML

This project uses machine learning to detect and predict congestion in simulated network traffic data. It integrates SHAP for explainability and provides interactive visualizations via a dashboard.

## Features
- Simulated network traffic data with features like packet loss, jitter, latency, and throughput
- ML model training for congestion detection (Random Forest/XGBoost)
- SHAP-based feature importance and explainability
- Interactive dashboard for model insights

## Setup
1. Clone this repository or download the files.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
### Run the main script (training and SHAP analysis):
```
python network_congestion_detection.py
```

### Launch the interactive dashboard:
```
streamlit run dashboard.py
```

The dashboard will open in your browser, showing model predictions and SHAP explanations. 