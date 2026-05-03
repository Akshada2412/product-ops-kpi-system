import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Product Operations Intelligence",
    page_icon=None,
    layout="wide"
)

# ---- CUSTOM CSS ----
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .block-container { padding-top: 1rem; }
    h1 { font-size: 1.6rem; font-weight: 600; 
         color: #1a1a2e; letter-spacing: -0.3px; }
    h2 { font-size: 1rem; font-weight: 600; 
         color: #2d2d2d; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6c757d;
        margin-bottom: 0.75rem;
    }
    .alert-high {
        background: #fff5f5;
        border-left: 3px solid #e53e3e;
        padding: 0.5rem 1rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 1.5rem 0;
    }
    [data-testid="stSidebar"] {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    activity = pd.read_csv('data/raw/product_activity.csv')
    users = pd.read_csv('data/raw/users.csv')

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
    industry_text = user_features['industry'].copy()

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
    user_features['industry_name'] = industry_text

    def risk_segment(prob):
        if prob >= 0.7:
            return 'High Risk'
        elif prob >= 0.4:
            return 'Medium Risk'
        else:
            return 'Low Risk'

    user_features['risk_segment'] = (
        user_features['churn_probability']
        .apply(risk_segment)
    )

    return activity, users, user_features

activity, users, churn = load_data()

# ---- SIDEBAR ----
st.sidebar.markdown("### Filters")
st.sidebar.markdown("---")

selected_plan = st.sidebar.multiselect(
    "Plan tier",
    options=sorted(activity['plan'].unique().tolist()),
    default=sorted(activity['plan'].unique().tolist())
)

selected_industry = st.sidebar.multiselect(
    "Industry",
    options=sorted(activity['industry'].unique().tolist()),
    default=sorted(activity['industry'].unique().tolist())
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Data: Simulated SaaS cohort, 2023<br>"
    "Model: Random Forest (ROC-AUC 0.883)</small>",
    unsafe_allow_html=True
)

# ---- FILTER ----
filtered = activity[
    (activity['plan'].isin(selected_plan)) &
    (activity['industry'].isin(selected_industry))
]

filtered_churn = churn[
    churn['plan_name'].isin(selected_plan) &
    churn['industry_name'].isin(selected_industry)
]

# ---- HEADER ----
st.markdown("## Product Operations Intelligence")
st.markdown(
    "<p style='color:#6c757d;font-size:0.9rem;margin-top:-0.5rem;'>"
    "SaaS cohort analysis — revenue, engagement, "
    "and ML-powered churn risk across 1,000 users | "
    "January–December 2023"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ---- KPI ROW ----
col1, col2, col3, col4, col5 = st.columns(5)

total_mrr = filtered['mrr'].sum()
active_users = filtered['user_id'].nunique()
avg_sessions = filtered['sessions'].mean()
avg_nps = filtered['nps_score'].mean()
high_risk_count = len(filtered_churn[
    filtered_churn['risk_segment'] == 'High Risk'
])

col1.metric("Total MRR", f"${total_mrr:,.0f}")
col2.metric("Active Users", f"{active_users:,}")
col3.metric("Avg Sessions / User", f"{avg_sessions:.1f}")
col4.metric("Avg NPS Score", f"{avg_nps:.1f} / 10")
col5.metric("High Churn Risk", f"{high_risk_count}",
            delta="users need attention",
            delta_color="inverse")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ---- REVENUE SECTION ----
st.markdown(
    "<p class='section-label'>Revenue</p>",
    unsafe_allow_html=True
)

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("**MRR Trend**")
    mrr_trend = (
        filtered.groupby('month')['mrr']
        .sum().reset_index()
    )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=mrr_trend['month'],
        y=mrr_trend['mrr'],
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(66, 133, 244, 0.1)',
        line=dict(color='#4285F4', width=2.5),
        marker=dict(size=6, color='#4285F4')
    ))
    fig1.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title='Month',
        yaxis_title='MRR (USD)',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_r:
    st.markdown("**Revenue Concentration by Plan**")
    rev_plan = (
        filtered.groupby('plan')['mrr']
        .sum().reset_index()
    )
    fig2 = px.pie(
        rev_plan,
        values='mrr',
        names='plan',
        color_discrete_sequence=[
            '#4285F4', '#34A853',
            '#FBBC04', '#EA4335'
        ],
        hole=0.55
    )
    fig2.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='white',
        legend=dict(orientation='h', y=-0.1)
    )
    fig2.update_traces(
        textposition='outside',
        textinfo='percent+label'
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ---- ENGAGEMENT SECTION ----
st.markdown(
    "<p class='section-label'>User Engagement</p>",
    unsafe_allow_html=True
)

col_l2, col_r2 = st.columns(2)

with col_l2:
    st.markdown("**Feature Adoption vs Session Depth by Plan**")
    adoption = (
        filtered.groupby('plan')
        .agg(
            avg_features=('features_used', 'mean'),
            avg_sessions=('sessions', 'mean')
        ).reset_index()
    )
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='Avg Features Used',
        x=adoption['plan'],
        y=adoption['avg_features'].round(1),
        marker_color='#34A853',
        text=adoption['avg_features'].round(1),
        textposition='outside'
    ))
    fig3.add_trace(go.Bar(
        name='Avg Sessions',
        x=adoption['plan'],
        y=adoption['avg_sessions'].round(1),
        marker_color='#4285F4',
        text=adoption['avg_sessions'].round(1),
        textposition='outside'
    ))
    fig3.update_layout(
        height=320,
        barmode='group',
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#f0f0f0'),
        legend=dict(orientation='h', y=1.1)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_r2:
    st.markdown("**Support Ticket Load per User by Plan**")
    tickets = (
        filtered.groupby('plan')
        .agg(
            total_tickets=('support_tickets', 'sum'),
            users=('user_id', 'nunique')
        ).reset_index()
    )
    tickets['tickets_per_user'] = (
        tickets['total_tickets'] / tickets['users']
    ).round(2)

    fig4 = px.bar(
        tickets.sort_values(
            'tickets_per_user', ascending=True
        ),
        x='tickets_per_user',
        y='plan',
        orientation='h',
        color='tickets_per_user',
        color_continuous_scale=[
            '#34A853', '#FBBC04', '#EA4335'
        ],
        text='tickets_per_user',
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title='Tickets per User',
            showgrid=False
        ),
        yaxis=dict(title='')
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ---- SATISFACTION + RETENTION ----
st.markdown(
    "<p class='section-label'>Satisfaction & Retention</p>",
    unsafe_allow_html=True
)

col_l3, col_r3 = st.columns(2)

with col_l3:
    st.markdown("**NPS Score by Plan**")
    nps = (
        filtered.groupby('plan')['nps_score']
        .mean().reset_index()
    )
    nps.columns = ['plan', 'avg_nps']
    nps['avg_nps'] = nps['avg_nps'].round(2)

    fig5 = px.bar(
        nps.sort_values('avg_nps', ascending=True),
        x='avg_nps',
        y='plan',
        orientation='h',
        color='avg_nps',
        color_continuous_scale=[
            '#EA4335', '#FBBC04', '#34A853'
        ],
        text='avg_nps',
    )
    fig5.update_traces(textposition='outside')
    fig5.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title='Average NPS Score',
            range=[0, 10],
            showgrid=False
        ),
        yaxis=dict(title='')
    )
    st.plotly_chart(fig5, use_container_width=True)

with col_r3:
    st.markdown("**Monthly Active Users**")
    user_trend = (
        filtered.groupby('month')['user_id']
        .nunique().reset_index()
    )
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=user_trend['month'],
        y=user_trend['user_id'],
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(234, 67, 53, 0.1)',
        line=dict(color='#EA4335', width=2.5),
        marker=dict(size=6, color='#EA4335')
    ))
    fig6.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title='Month',
        yaxis_title='Active Users',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ---- CHURN RISK ----
st.markdown(
    "<p class='section-label'>Churn Risk Intelligence"
    " — Random Forest Model (ROC-AUC: 0.883)</p>",
    unsafe_allow_html=True
)

risk_col1, risk_col2, risk_col3 = st.columns(3)

high = filtered_churn[
    filtered_churn['risk_segment'] == 'High Risk']
med = filtered_churn[
    filtered_churn['risk_segment'] == 'Medium Risk']
low = filtered_churn[
    filtered_churn['risk_segment'] == 'Low Risk']

risk_col1.metric("High Risk Users", len(high),
                 "Immediate action needed",
                 delta_color="inverse")
risk_col2.metric("Medium Risk Users", len(med),
                 "Monitor closely",
                 delta_color="off")
risk_col3.metric("Low Risk Users", len(low),
                 "Healthy",
                 delta_color="normal")

col_pie, col_table = st.columns([1, 2])

with col_pie:
    st.markdown("**Risk Distribution**")
    fig7 = px.pie(
        filtered_churn,
        names='risk_segment',
        color='risk_segment',
        color_discrete_map={
            'High Risk': '#EA4335',
            'Medium Risk': '#FBBC04',
            'Low Risk': '#34A853'
        },
        hole=0.5
    )
    fig7.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='white',
        legend=dict(orientation='h', y=-0.15)
    )
    st.plotly_chart(fig7, use_container_width=True)

with col_table:
    st.markdown("**Top High Risk Accounts — Intervention Required**")
    high_risk_display = (
        filtered_churn[
            filtered_churn['risk_segment'] == 'High Risk'
        ][[
            'user_id', 'churn_probability',
            'avg_sessions', 'avg_features_used',
            'total_tickets', 'plan_name'
        ]]
        .sort_values('churn_probability', ascending=False)
        .head(8)
        .round(3)
    )
    high_risk_display.columns = [
        'User ID', 'Churn Prob.',
        'Avg Sessions', 'Features Used',
        'Support Tickets', 'Plan'
    ]
    st.dataframe(
        high_risk_display.reset_index(drop=True),
        use_container_width=True,
        height=250
    )

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown(
    "<p style='color:#6c757d;font-size:0.8rem;'>"
    "Akshada Karade &nbsp;|&nbsp; "
    "MS Engineering Management, UMass Amherst &nbsp;|&nbsp; "
    "Stack: Python · SQL · Streamlit · Scikit-learn · Plotly"
    "</p>",
    unsafe_allow_html=True
)