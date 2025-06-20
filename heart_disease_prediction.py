import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download dataset if not present
DATA_URL='https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv'
DATA_PATH = 'framingham.csv'

if not os.path.exists(DATA_PATH):
    print('Downloading dataset...')
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

print('Dataset loaded. Shape:', df.shape)

# Exploratory Data Analysis (EDA)
print('\nFirst 5 rows:')
print(df.head())

print('\nDataset Info:')
df.info()

print('\nMissing values per column:')
print(df.isnull().sum())

# Visualize target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='TenYearCHD', data=df)
plt.title('Distribution of Heart Disease (TenYearCHD)')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Data Preprocessing
# Fill missing values with median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Features and target
y = df['TenYearCHD']
X = df.drop('TenYearCHD', axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print(f'\nModel Accuracy: {acc*100:.2f}%')
print('\nConfusion Matrix:')
print(cm)
print('\nClassification Report:')
print(cr)

# Visualize confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show() 