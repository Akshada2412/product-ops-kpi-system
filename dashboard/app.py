import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Product Ops KPI System",
    page_icon="📊",
    layout="wide"
)

# ---- LOAD DATA ----
@st.cache_data
@st.cache_data
def load_data():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    activity = pd.read_csv('data/raw/product_activity.csv')
    users = pd.read_csv('data/raw/users.csv')

    # Build churn scores on the fly
    all_months = activity['month'].unique()
    last_month = pd.Series(all_months).max()

    user_last = activity.groupby('user_id')['month'].max().reset_index()
    user_last.columns = ['user_id', 'last_active_month']
    user_last['churned'] = (
        user_last['last_active_month'] < last_month
    ).astype(int)

    user_features = activity.groupby('user_id').agg(
        avg_sessions=('sessions', 'mean'),
        avg_features_used=('features_used', 'mean'),
        total_tickets=('support_tickets', 'sum'),
        avg_nps=('nps_score', 'mean'),
        plan=('plan', 'first'),
    ).reset_index()

    user_features = user_features.merge(
        user_last[['user_id', 'churned']], on='user_id'
    )
    user_features = user_features.merge(
        users[['user_id', 'industry',
               'company_size', 'acquisition_channel']],
        on='user_id'
    )

    plan_text = user_features['plan'].copy()

    le = LabelEncoder()
    for col in ['plan', 'industry',
                'company_size', 'acquisition_channel']:
        user_features[col] = le.fit_transform(
            user_features[col]
        )

    user_features['avg_nps'] = (
        user_features['avg_nps']
        .fillna(user_features['avg_nps'].median())
    )

    feature_cols = [
        'avg_sessions', 'avg_features_used',
        'total_tickets', 'avg_nps', 'plan',
        'industry', 'company_size',
        'acquisition_channel'
    ]

    X = user_features[feature_cols]
    y = user_features['churned']

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X, y)

    user_features['churn_probability'] = (
        model.predict_proba(X)[:, 1]
    )
    user_features['plan_name'] = plan_text

    def risk_segment(prob):
        if prob >= 0.7:
            return 'HIGH RISK'
        elif prob >= 0.4:
            return 'MEDIUM RISK'
        else:
            return 'LOW RISK'

    user_features['risk_segment'] = (
        user_features['churn_probability']
        .apply(risk_segment)
    )

    return activity, users, user_features

activity, users, churn = load_data()

activity, users, churn = load_data()

def get_conn():
    activity = pd.read_csv('data/raw/product_activity.csv')
    conn = sqlite3.connect(':memory:')
    activity.to_sql('activity', conn,
                    if_exists='replace', index=False)
    return conn

conn = get_conn()

# ---- HEADER ----
st.title("📊 Product Ops KPI Intelligence System")
st.markdown(
    "Real-time SaaS product analytics — "
    "revenue trends, feature adoption, "
    "and **ML-powered churn prediction** "
    "across 1,000 users"
)
st.divider()

# ---- SIDEBAR ----
st.sidebar.header("🔧 Filters")

selected_plan = st.sidebar.multiselect(
    "Plan",
    options=activity['plan'].unique().tolist(),
    default=activity['plan'].unique().tolist()
)

selected_industry = st.sidebar.multiselect(
    "Industry",
    options=activity['industry'].unique().tolist(),
    default=activity['industry'].unique().tolist()
)

# ---- FILTER ----
filtered = activity[
    (activity['plan'].isin(selected_plan)) &
    (activity['industry'].isin(selected_industry))
]

# ---- KPI METRICS ----
col1, col2, col3, col4, col5 = st.columns(5)

total_mrr = filtered['mrr'].sum()
active_users = filtered['user_id'].nunique()
avg_sessions = filtered['sessions'].mean()
avg_nps = filtered['nps_score'].mean()
filtered_churn = churn[churn['plan_name'].isin(selected_plan)]
high_risk = len(filtered_churn[
    filtered_churn['risk_segment'] == 'HIGH RISK'
])

col1.metric("Total MRR", f"${total_mrr:,.0f}")
col2.metric("Active Users", f"{active_users:,}")
col3.metric("Avg Sessions", f"{avg_sessions:.1f}")
col4.metric("Avg NPS", f"{avg_nps:.1f}/10")
col5.metric("High Churn Risk", f"{high_risk} users",
            delta=f"{high_risk} need attention",
            delta_color="inverse")

st.divider()

# ---- ROW 1: MRR TREND + REVENUE BY PLAN ----
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("💰 MRR Trend (2023)")
    mrr_trend = (
        filtered.groupby('month')['mrr']
        .sum()
        .reset_index()
    )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=mrr_trend['month'],
        y=mrr_trend['mrr'],
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=7)
    ))
    fig1.update_layout(
        height=380,
        xaxis_title='Month',
        yaxis_title='MRR (USD)',
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_r:
    st.subheader("📦 Revenue Share by Plan")
    rev_plan = (
        filtered.groupby('plan')['mrr']
        .sum()
        .reset_index()
    )
    fig2 = px.pie(
        rev_plan,
        values='mrr',
        names='plan',
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4
    )
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---- ROW 2: FEATURE ADOPTION + SUPPORT TICKETS ----
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("🎯 Feature Adoption by Plan")
    adoption = (
        filtered.groupby('plan')
        .agg(
            avg_features=('features_used', 'mean'),
            avg_sessions=('sessions', 'mean')
        )
        .reset_index()
    )
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='Avg Features Used',
        x=adoption['plan'],
        y=adoption['avg_features'],
        marker_color='#2ecc71'
    ))
    fig3.add_trace(go.Bar(
        name='Avg Sessions',
        x=adoption['plan'],
        y=adoption['avg_sessions'],
        marker_color='#3498db'
    ))
    fig3.update_layout(
        height=380,
        barmode='group',
        xaxis_title='Plan',
        yaxis_title='Average Count'
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_r2:
    st.subheader("🎫 Support Tickets by Plan")
    tickets = (
        filtered.groupby('plan')
        .agg(
            total_tickets=('support_tickets', 'sum'),
            users=('user_id', 'nunique')
        )
        .reset_index()
    )
    tickets['tickets_per_user'] = (
        tickets['total_tickets'] / tickets['users']
    ).round(2)

    fig4 = px.bar(
        tickets,
        x='plan',
        y='tickets_per_user',
        color='tickets_per_user',
        color_continuous_scale='Reds',
        text='tickets_per_user',
        labels={
            'tickets_per_user': 'Tickets per User',
            'plan': 'Plan'
        }
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ---- ROW 3: NPS + USER GROWTH ----
col_l3, col_r3 = st.columns(2)

with col_l3:
    st.subheader("⭐ NPS Score by Plan")
    nps = (
        filtered.groupby('plan')['nps_score']
        .mean()
        .reset_index()
    )
    nps.columns = ['plan', 'avg_nps']
    nps['avg_nps'] = nps['avg_nps'].round(2)

    fig5 = px.bar(
        nps.sort_values('avg_nps', ascending=False),
        x='plan',
        y='avg_nps',
        color='avg_nps',
        color_continuous_scale='Greens',
        text='avg_nps',
        labels={
            'avg_nps': 'Average NPS Score',
            'plan': 'Plan'
        }
    )
    fig5.update_traces(textposition='outside')
    fig5.update_layout(
        height=380,
        showlegend=False,
        yaxis_range=[0, 10]
    )
    st.plotly_chart(fig5, use_container_width=True)

with col_r3:
    st.subheader("📈 Active Users by Month")
    user_trend = (
        filtered.groupby('month')['user_id']
        .nunique()
        .reset_index()
    )
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=user_trend['month'],
        y=user_trend['user_id'],
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=7)
    ))
    fig6.update_layout(
        height=380,
        xaxis_title='Month',
        yaxis_title='Active Users',
        showlegend=False
    )
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ---- CHURN RISK PANEL ----
st.subheader("🚨 ML-Powered Churn Risk Panel")
st.markdown(
    "Users scored by **Random Forest model** "
    "(ROC-AUC: 0.883) — "
    "identify at-risk accounts before they leave"
)

# Connect churn to sidebar filters
filtered_churn = churn[churn['plan_name'].isin(selected_plan)]

risk_col1, risk_col2, risk_col3 = st.columns(3)

high = filtered_churn[
    filtered_churn['risk_segment'] == 'HIGH RISK']
med = filtered_churn[
    filtered_churn['risk_segment'] == 'MEDIUM RISK']
low = filtered_churn[
    filtered_churn['risk_segment'] == 'LOW RISK']

risk_col1.metric(
    "🔴 High Risk Users",
    len(high),
    "Immediate action needed"
)
risk_col2.metric(
    "🟡 Medium Risk Users",
    len(med),
    "Monitor closely"
)
risk_col3.metric(
    "🟢 Low Risk Users",
    len(low),
    "Healthy"
)

# Risk distribution chart
fig7 = px.pie(
    filtered_churn,
    names='risk_segment',
    color='risk_segment',
    color_discrete_map={
        'HIGH RISK': '#e74c3c',
        'MEDIUM RISK': '#f39c12',
        'LOW RISK': '#2ecc71'
    },
    hole=0.4
)
fig7.update_layout(height=300)
st.plotly_chart(fig7, use_container_width=True)

# High risk user table
st.markdown("**🔴 Top High Risk Users — Act Now**")
high_risk_display = (
    filtered_churn[
        filtered_churn['risk_segment'] == 'HIGH RISK'
    ]
    [[
        'user_id', 'churn_probability',
        'avg_sessions', 'avg_features_used',
        'total_tickets'
    ]]
    .sort_values('churn_probability', ascending=False)
    .head(10)
    .round(3)
)
high_risk_display.columns = [
    'User ID', 'Churn Probability',
    'Avg Sessions', 'Avg Features Used',
    'Total Tickets'
]
st.dataframe(
    high_risk_display.reset_index(drop=True),
    use_container_width=True
)

# Risk distribution chart
fig7 = px.pie(
    churn,
    names='risk_segment',
    color='risk_segment',
    color_discrete_map={
        'HIGH RISK': '#e74c3c',
        'MEDIUM RISK': '#f39c12',
        'LOW RISK': '#2ecc71'
    },
    hole=0.4
)
fig7.update_layout(height=300)
st.plotly_chart(fig7, use_container_width=True)

# High risk user table
st.markdown("**🔴 Top High Risk Users — Act Now**")
high_risk_display = (
    churn[churn['risk_segment'] == 'HIGH RISK']
    [[
        'user_id', 'churn_probability',
        'avg_sessions', 'avg_features_used',
        'total_tickets'
    ]]
    .sort_values('churn_probability', ascending=False)
    .head(10)
    .round(3)
)
high_risk_display.columns = [
    'User ID', 'Churn Probability',
    'Avg Sessions', 'Avg Features Used',
    'Total Tickets'
]
st.dataframe(
    high_risk_display.reset_index(drop=True),
    use_container_width=True
)

st.divider()

# ---- FOOTER ----
st.markdown(
    "**Built by Akshada Karade** | "
    "MS Engineering Management, UMass Amherst | "
    "Stack: Python, SQL, Streamlit, Scikit-learn, Plotly"
)