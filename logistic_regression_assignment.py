import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, 
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import statsmodels.api as sm

# Load and preprocess data
titanic_df = pd.read_csv(r"tested.csv")

# Fill missing values
titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())
titanic_df["Fare"] = titanic_df["Fare"].fillna(titanic_df["Fare"].median())

# Drop cabin and identity columns
titanic_df = titanic_df.drop(columns=["Cabin", "PassengerId", "Name", "Ticket"], axis=1)

# Encode categorical variables
le = LabelEncoder()
titanic_df["Sex"] = le.fit_transform(titanic_df["Sex"])
titanic_df = pd.get_dummies(titanic_df, columns=["Pclass", "Embarked"], drop_first=True, dtype=int)

# Separate features and target
X = titanic_df.drop(columns=["Survived"], axis=1)
y = titanic_df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print("MODEL 1: BASELINE LOGISTIC REGRESSION (No Scaling)")
lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train, y_train)
y_pred_baseline = lr_baseline.predict(X_test)

print("\nPerformance Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_baseline):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_baseline):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_baseline))

print("MODEL 2: LOGISTIC REGRESSION WITH STANDARD SCALING")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_scaled = LogisticRegression(max_iter=1000, random_state=42)
lr_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = lr_scaled.predict(X_test_scaled)

print("\nPerformance Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_scaled):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_scaled):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_scaled):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_scaled):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_scaled))

print("MODEL 3: LOGISTIC REGRESSION WITH SCALING + CLASS WEIGHTS")

lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_balanced.fit(X_train_scaled, y_train)
y_pred_balanced = lr_balanced.predict(X_test_scaled)

print("\nPerformance Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_balanced):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_balanced):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_balanced):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_balanced):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_balanced))


print("MODEL 4: STATSMODELS LOGIT (WITH SCALING)")

# Add constant for statsmodels
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)

logit_model = sm.Logit(y_train, X_train_sm)
logit_result = logit_model.fit(disp=0)

print("\nModel Summary:")
print(logit_result.summary())

# Predictions
y_pred_proba_sm = logit_result.predict(X_test_sm)
y_pred_sm = (y_pred_proba_sm >= 0.5).astype(int)

print("\nPerformance Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_sm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_sm):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_sm):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_sm):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_sm))