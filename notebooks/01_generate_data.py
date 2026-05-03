import pandas as pd
import numpy as np
import os

np.random.seed(42)

print("Generating SaaS product dataset...")

# ---- 1000 USERS ----
n_users = 1000

users = pd.DataFrame({
    'user_id': [f'U{str(i).zfill(4)}' for i in range(1, n_users+1)],
    'signup_date': pd.date_range(
        start='2023-01-01', periods=n_users, freq='8H'
    ),
    'plan': np.random.choice(
        ['free', 'starter', 'pro', 'enterprise'],
        n_users,
        p=[0.40, 0.30, 0.20, 0.10]
    ),
    'industry': np.random.choice(
        ['SaaS', 'Ecommerce', 'Healthcare',
         'Finance', 'Education', 'Retail'],
        n_users
    ),
    'company_size': np.random.choice(
        ['1-10', '11-50', '51-200', '201-1000', '1000+'],
        n_users,
        p=[0.30, 0.30, 0.20, 0.15, 0.05]
    ),
    'acquisition_channel': np.random.choice(
        ['organic', 'paid_search', 'referral',
         'social', 'direct'],
        n_users,
        p=[0.35, 0.25, 0.20, 0.12, 0.08]
    ),
    'country': np.random.choice(
        ['USA', 'UK', 'Canada', 'Germany',
         'India', 'Australia', 'France', 'Other'],
        n_users,
        p=[0.40, 0.12, 0.10, 0.08,
           0.10, 0.07, 0.06, 0.07]
    ),
})

# ---- MONTHLY ACTIVITY (12 months) ----
months = pd.date_range(start='2023-01-01', periods=12, freq='MS')
rows = []

plan_base_sessions = {
    'free': 3, 'starter': 8,
    'pro': 18, 'enterprise': 35
}

plan_churn_rate = {
    'free': 0.12, 'starter': 0.07,
    'pro': 0.04, 'enterprise': 0.02
}

plan_revenue = {
    'free': 0, 'starter': 29,
    'pro': 99, 'enterprise': 399
}

for _, user in users.iterrows():
    churned = False
    for month in months:
        if churned:
            break

        # Churn check
        if np.random.random() < plan_churn_rate[user['plan']]:
            churned = True
            break

        base = plan_base_sessions[user['plan']]
        sessions = max(1, int(np.random.normal(base, base*0.3)))
        features_used = min(10, max(1, int(
            np.random.normal(base/3, 1.5)
        )))
        support_tickets = np.random.poisson(
            0.3 if user['plan'] == 'enterprise' else 0.8
        )
        nps_score = None
        if np.random.random() < 0.15:
            nps_score = int(np.random.normal(
                7.5 if user['plan'] in ['pro', 'enterprise'] else 6,
                1.5
            ))
            nps_score = max(0, min(10, nps_score))

        rows.append({
            'user_id': user['user_id'],
            'month': month,
            'plan': user['plan'],
            'industry': user['industry'],
            'company_size': user['company_size'],
            'acquisition_channel': user['acquisition_channel'],
            'country': user['country'],
            'sessions': sessions,
            'features_used': features_used,
            'support_tickets': support_tickets,
            'nps_score': nps_score,
            'mrr': plan_revenue[user['plan']],
            'is_churned': 0,
        })

df = pd.DataFrame(rows)

os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/product_activity.csv', index=False)
users.to_csv('data/raw/users.csv', index=False)

print("Dataset generated!")
print("Activity rows:", len(df))
print("Unique users:", df['user_id'].nunique())
print("Months:", df['month'].nunique())
print("\nPlan distribution:")
print(df.groupby('plan')['mrr'].agg(['count', 'sum']))
print("\nSample:")
print(df.head(5))