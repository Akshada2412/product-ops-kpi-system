import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import os

print("=" * 55)
print("CHURN PREDICTION MODEL")
print("=" * 55)

# ---- LOAD DATA ----
df = pd.read_csv('data/raw/product_activity.csv')
users = pd.read_csv('data/raw/users.csv')

# ---- BUILD CHURN LABELS ----
# A user is "churned" if they disappear before month 12
all_months = df['month'].unique()
last_month = pd.Series(all_months).max()

user_last_month = (
    df.groupby('user_id')['month'].max().reset_index()
)
user_last_month.columns = ['user_id', 'last_active_month']
user_last_month['churned'] = (
    user_last_month['last_active_month'] < last_month
).astype(int)

print(f"Total users: {len(user_last_month)}")
print(f"Churned: {user_last_month['churned'].sum()}")
print(f"Active: {(user_last_month['churned']==0).sum()}")
print()

# ---- BUILD FEATURES ----
# Aggregate each user's behavior into one row
user_features = df.groupby('user_id').agg(
    avg_sessions=('sessions', 'mean'),
    avg_features_used=('features_used', 'mean'),
    total_tickets=('support_tickets', 'sum'),
    avg_nps=('nps_score', 'mean'),
    total_mrr=('mrr', 'sum'),
    months_active=('month', 'count'),
    plan=('plan', 'first'),
).reset_index()

# Merge with churn labels and user info
user_features = user_features.merge(
    user_last_month[['user_id', 'churned']],
    on='user_id'
)
user_features = user_features.merge(
    users[['user_id', 'industry',
           'company_size', 'acquisition_channel']],
    on='user_id'
)

# Save original plan text before encoding
plan_text = user_features['plan'].copy()

# ---- ENCODE CATEGORICAL COLUMNS ----
le = LabelEncoder()
for col in ['plan', 'industry',
            'company_size', 'acquisition_channel']:
    user_features[col] = le.fit_transform(
        user_features[col]
    )

# Fill NaN NPS with median
user_features['avg_nps'] = (
    user_features['avg_nps']
    .fillna(user_features['avg_nps'].median())
)

# ---- TRAIN MODEL ----
feature_cols = [
    'avg_sessions', 'avg_features_used',
    'total_tickets', 'avg_nps',
    'plan', 'industry',
    'company_size', 'acquisition_channel'
]

X = user_features[feature_cols]
y = user_features['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# ---- EVALUATE ----
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("MODEL PERFORMANCE:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
print()

# ---- FEATURE IMPORTANCE ----
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("TOP CHURN PREDICTORS:")
print(importance.to_string(index=False))
print()

# ---- SCORE ALL USERS ----
user_features['churn_probability'] = (
    model.predict_proba(X)[:, 1]
)

# Risk segments
def risk_segment(prob):
    if prob >= 0.7:
        return 'HIGH RISK'
    elif prob >= 0.4:
        return 'MEDIUM RISK'
    else:
        return 'LOW RISK'

user_features['risk_segment'] = (
    user_features['churn_probability'].apply(risk_segment)
)

print("CHURN RISK DISTRIBUTION:")
print(user_features['risk_segment'].value_counts())
print()

# Save results
os.makedirs('data/processed', exist_ok=True)
# Save with readable plan names
user_features['plan_name'] = plan_text
user_features.to_csv(
    'data/processed/churn_scores.csv', index=False
)
print("✅ Churn scores saved!")