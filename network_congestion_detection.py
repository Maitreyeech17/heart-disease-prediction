import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import shap
import joblib

# 1. Simulate network traffic data
def simulate_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    data = pd.DataFrame({
        'packet_loss': np.random.beta(2, 10, n_samples) * 10,  # %
        'jitter': np.random.gamma(2, 2, n_samples),           # ms
        'latency': np.random.normal(50, 15, n_samples),       # ms
        'throughput': np.random.normal(100, 20, n_samples),   # Mbps
        'connections': np.random.poisson(20, n_samples),      # count
        'bandwidth_util': np.random.uniform(30, 100, n_samples), # %
    })
    # Congestion label: if packet_loss, jitter, latency, or bandwidth_util are high, or throughput is low
    congestion = (
        (data['packet_loss'] > 3) |
        (data['jitter'] > 6) |
        (data['latency'] > 70) |
        (data['throughput'] < 80) |
        (data['bandwidth_util'] > 85)
    ).astype(int)
    data['congestion'] = congestion
    return data

data = simulate_data()

# 2. EDA
print('First 5 rows:')
print(data.head())
print('\nClass balance:')
print(data['congestion'].value_counts())

sns.countplot(x='congestion', data=data)
plt.title('Congestion Class Distribution')
plt.show()

sns.pairplot(data, hue='congestion', diag_kind='kde')
plt.show()

# 3. Train/test split
X = data.drop('congestion', axis=1)
y = data['congestion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model training
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print(f'\nAccuracy: {acc*100:.2f}%')
print('\nConfusion Matrix:')
print(cm)
print('\nClassification Report:')
print(cr)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 6. SHAP Explainability
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, show=True)
shap.summary_plot(shap_values, X_test, plot_type='bar', show=True)

# Save model and SHAP values for dashboard
joblib.dump(model, 'congestion_model.pkl')
X_test.to_csv('X_test.csv', index=False)
pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_csv('y_test_pred.csv', index=False)
shap_values_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
shap_values_df.to_csv('shap_values.csv', index=False)

print('Model, test data, and SHAP values saved for dashboard use.') 