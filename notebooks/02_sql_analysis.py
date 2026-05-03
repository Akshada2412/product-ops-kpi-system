import pandas as pd
import sqlite3

# ---- SETUP ----
df = pd.read_csv('data/raw/product_activity.csv')
users = pd.read_csv('data/raw/users.csv')

conn = sqlite3.connect('data/processed/product_ops.db')
df.to_sql('activity', conn, if_exists='replace', index=False)
users.to_sql('users', conn, if_exists='replace', index=False)

print("✅ Database ready!")
print(f"   Activity rows: {len(df)}")
print(f"   Users: {len(users)}")
print("---")

# ============================================================
# QUERY 1 — Monthly Revenue Trend (MRR)
# LESSON: GROUP BY month, SUM revenue
# ============================================================
q1 = """
SELECT
    month,
    SUM(mrr) AS total_mrr,
    COUNT(DISTINCT user_id) AS active_users,
    ROUND(SUM(mrr) * 1.0 / COUNT(DISTINCT user_id), 2) 
        AS arpu
FROM activity
GROUP BY month
ORDER BY month
"""
print("💰 MONTHLY REVENUE TREND (MRR):")
print(pd.read_sql_query(q1, conn).to_string(index=False))
print("---")

# ============================================================
# QUERY 2 — Revenue by Plan
# LESSON: GROUP BY plan, understand revenue concentration
# ============================================================
q2 = """
SELECT
    plan,
    COUNT(DISTINCT user_id) AS users,
    SUM(mrr) AS total_mrr,
    ROUND(SUM(mrr) * 100.0 / 
        (SELECT SUM(mrr) FROM activity), 2) 
        AS revenue_share_pct,
    ROUND(AVG(sessions), 1) AS avg_sessions,
    ROUND(AVG(features_used), 1) AS avg_features_used
FROM activity
GROUP BY plan
ORDER BY total_mrr DESC
"""
print("📊 REVENUE AND ENGAGEMENT BY PLAN:")
print(pd.read_sql_query(q2, conn).to_string(index=False))
print("---")

# ============================================================
# QUERY 3 — Feature Adoption Rate
# LESSON: Identify which user segments use product most
# ============================================================
q3 = """
SELECT
    plan,
    industry,
    ROUND(AVG(features_used), 2) AS avg_features_used,
    ROUND(AVG(sessions), 2) AS avg_sessions,
    COUNT(DISTINCT user_id) AS users
FROM activity
GROUP BY plan, industry
ORDER BY avg_features_used DESC
LIMIT 12
"""
print("🎯 FEATURE ADOPTION BY PLAN AND INDUSTRY:")
print(pd.read_sql_query(q3, conn).to_string(index=False))
print("---")

# ============================================================
# QUERY 4 — Support Ticket Burden by Plan
# LESSON: WHERE, GROUP BY, business insight on support cost
# ============================================================
q4 = """
SELECT
    plan,
    COUNT(DISTINCT user_id) AS users,
    SUM(support_tickets) AS total_tickets,
    ROUND(SUM(support_tickets) * 1.0 / 
        COUNT(DISTINCT user_id), 2) AS tickets_per_user,
    ROUND(AVG(sessions), 1) AS avg_sessions
FROM activity
GROUP BY plan
ORDER BY tickets_per_user DESC
"""
print("🎫 SUPPORT TICKET BURDEN BY PLAN:")
print(pd.read_sql_query(q4, conn).to_string(index=False))
print("---")

# ============================================================
# QUERY 5 — NPS Score by Plan (satisfaction signal)
# LESSON: AVG on nullable column, HAVING clause
# ============================================================
q5 = """
SELECT
    plan,
    ROUND(AVG(nps_score), 2) AS avg_nps,
    COUNT(nps_score) AS responses,
    COUNT(DISTINCT user_id) AS total_users,
    ROUND(COUNT(nps_score) * 100.0 / 
        COUNT(DISTINCT user_id), 1) AS response_rate_pct
FROM activity
GROUP BY plan
HAVING COUNT(nps_score) > 0
ORDER BY avg_nps DESC
"""
print("⭐ NPS SATISFACTION BY PLAN:")
print(pd.read_sql_query(q5, conn).to_string(index=False))
print("---")

print("✅ All queries complete!")