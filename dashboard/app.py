import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Product Operations Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fix title cutoff */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Header strip */
    .dashboard-header {
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4285F4;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            rgba(66,133,244,0.08) 0%,
            rgba(52,168,83,0.04) 100%
        );
    }
    .dashboard-title {
        font-size: 8.00rem;
        font-weight: 1000;
        letter-spacing: -0.5px;
        margin: 0 0 0.25rem 0;
    }
    .dashboard-subtitle {
        font-size: 0.85rem;
        opacity: 0.65;
        margin: 0;
    }
    /* Section labels */
    .section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.5;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(128,128,128,0.2);
    }
    /* Risk badges */
    .badge-high {
        background: rgba(234,67,53,0.12);
        color: #EA4335;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-med {
        background: rgba(251,188,4,0.15);
        color: #F9A825;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-low {
        background: rgba(52,168,83,0.12);
        color: #34A853;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    /* Remove plotly modebar clutter */
    .modebar { display: none !important; }
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

    user_last = (
        activity.groupby('user_id')['month']
        .max().reset_index()
    )
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
with st.sidebar:
    st.markdown("### Filters")
    st.markdown("---")

    selected_plan = st.multiselect(
        "Plan Tier",
        options=sorted(activity['plan'].unique()),
        default=sorted(activity['plan'].unique())
    )

    selected_industry = st.multiselect(
        "Industry",
        options=sorted(activity['industry'].unique()),
        default=sorted(activity['industry'].unique())
    )

    st.markdown("---")

    # Mini stats in sidebar
    total_users = activity['user_id'].nunique()
    st.markdown(f"""
    <div style='font-size:0.78rem;opacity:0.6;line-height:1.8;'>
    <b>Dataset</b><br>
    {total_users:,} users &nbsp;·&nbsp; 12 months<br>
    5 industries &nbsp;·&nbsp; 4 plan tiers<br><br>
    <b>Model</b><br>
    Random Forest Classifier<br>
    ROC-AUC: 0.883 &nbsp;·&nbsp; Accuracy: 80%
    </div>
    """, unsafe_allow_html=True)

# ---- FILTERS ----
filtered = activity[
    activity['plan'].isin(selected_plan) &
    activity['industry'].isin(selected_industry)
]

filtered_churn = churn[
    churn['plan_name'].isin(selected_plan) &
    churn['industry_name'].isin(selected_industry)
]

# ---- HEADER ----
st.markdown("""
<div class='dashboard-header'>
    <p class='dashboard-title'>
        Product Operations Intelligence
    </p>
    <p class='dashboard-subtitle'>
        SaaS cohort analysis &nbsp;·&nbsp;
        Revenue, engagement & ML-powered churn risk
        &nbsp;·&nbsp; January – December 2023
    </p>
</div>
""", unsafe_allow_html=True)

# ---- KPI ROW ----
k1, k2, k3, k4, k5 = st.columns(5)

total_mrr = filtered['mrr'].sum()
active_users = filtered['user_id'].nunique()
avg_sessions = filtered['sessions'].mean()
avg_nps = filtered['nps_score'].mean()
high_risk_n = len(
    filtered_churn[
        filtered_churn['risk_segment'] == 'High Risk'
    ]
)
churn_rate = round(
    len(churn[churn['churned'] == 1]) /
    len(churn) * 100, 1
)

k1.metric("Monthly Recurring Revenue",
          f"${total_mrr:,.0f}")
k2.metric("Active Users", f"{active_users:,}")
k3.metric("Avg Sessions / User",
          f"{avg_sessions:.1f}")
k4.metric("Avg NPS Score", f"{avg_nps:.1f} / 10")
k5.metric("High Churn Risk",
          f"{high_risk_n} users",
          delta=f"{churn_rate}% overall churn rate",
          delta_color="inverse")

st.markdown("---")

# ---- REVENUE ----
st.markdown(
    "<p class='section-label'>Revenue Analysis</p>",
    unsafe_allow_html=True
)

r1, r2 = st.columns([3, 2])

with r1:
    st.markdown("**Monthly Recurring Revenue Trend**")
    mrr_by_plan = (
        filtered.groupby(['month', 'plan'])['mrr']
        .sum().reset_index()
    )
    fig1 = px.area(
        mrr_by_plan,
        x='month',
        y='mrr',
        color='plan',
        color_discrete_sequence=[
            '#4285F4', '#34A853', '#FBBC04', '#EA4335'
        ],
    )
    fig1.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=5, b=0),
        xaxis_title='',
        yaxis_title='MRR (USD)',
        legend=dict(
            orientation='h', y=1.15,
            title=None
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='rgba(128,128,128,0.15)')
    )
    st.plotly_chart(fig1, use_container_width=True)

with r2:
    st.markdown("**Revenue by Plan**")
    rev_plan = (
        filtered.groupby('plan')['mrr']
        .sum().reset_index()
    )
    fig2 = px.pie(
        rev_plan,
        values='mrr',
        names='plan',
        color_discrete_sequence=[
            '#4285F4', '#34A853', '#FBBC04', '#EA4335'
        ],
        hole=0.6
    )
    fig2.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=5, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', y=-0.15,
                    title=None)
    )
    fig2.update_traces(
        textinfo='percent',
        textposition='outside'
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---- ENGAGEMENT ----
st.markdown(
    "<p class='section-label'>User Engagement</p>",
    unsafe_allow_html=True
)

e1, e2, e3 = st.columns(3)

with e1:
    st.markdown("**Sessions by Plan**")
    sess = (
        filtered.groupby('plan')['sessions']
        .mean().reset_index()
        .sort_values('sessions', ascending=True)
    )
    fig3 = px.bar(
        sess,
        x='sessions', y='plan',
        orientation='h',
        color='sessions',
        color_continuous_scale=[
            '#4285F4', '#34A853'
        ],
        text=sess['sessions'].round(1)
    )
    fig3.update_traces(textposition='outside')
    fig3.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=5, b=0),
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Avg Sessions',
                   showgrid=False),
        yaxis=dict(title='')
    )
    st.plotly_chart(fig3, use_container_width=True)

with e2:
    st.markdown("**Feature Adoption by Plan**")
    feat = (
        filtered.groupby('plan')['features_used']
        .mean().reset_index()
        .sort_values('features_used', ascending=True)
    )
    fig4 = px.bar(
        feat,
        x='features_used', y='plan',
        orientation='h',
        color='features_used',
        color_continuous_scale=[
            '#FBBC04', '#34A853'
        ],
        text=feat['features_used'].round(1)
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=5, b=0),
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Avg Features Used',
                   showgrid=False),
        yaxis=dict(title='')
    )
    st.plotly_chart(fig4, use_container_width=True)

with e3:
    st.markdown("**Support Tickets per User**")
    tix = (
        filtered.groupby('plan')
        .agg(
            total=('support_tickets', 'sum'),
            users=('user_id', 'nunique')
        ).reset_index()
    )
    tix['per_user'] = (
        tix['total'] / tix['users']
    ).round(2)
    tix = tix.sort_values('per_user', ascending=True)

    fig5 = px.bar(
        tix,
        x='per_user', y='plan',
        orientation='h',
        color='per_user',
        color_continuous_scale=[
            '#34A853', '#FBBC04', '#EA4335'
        ],
        text='per_user'
    )
    fig5.update_traces(textposition='outside')
    fig5.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=5, b=0),
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Tickets per User',
                   showgrid=False),
        yaxis=dict(title='')
    )
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ---- SATISFACTION ----
st.markdown(
    "<p class='section-label'>"
    "Satisfaction & Retention</p>",
    unsafe_allow_html=True
)

s1, s2 = st.columns(2)

with s1:
    st.markdown("**NPS Score by Plan**")
    nps = (
        filtered.groupby('plan')['nps_score']
        .mean().reset_index()
    )
    nps.columns = ['plan', 'avg_nps']
    nps = nps.sort_values('avg_nps', ascending=True)

    fig6 = px.bar(
        nps,
        x='avg_nps', y='plan',
        orientation='h',
        color='avg_nps',
        color_continuous_scale=[
            '#EA4335', '#FBBC04', '#34A853'
        ],
        text=nps['avg_nps'].round(2),
        range_color=[0, 10]
    )
    fig6.update_traces(textposition='outside')
    fig6.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=5, b=0),
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Avg NPS',
                   range=[0, 10],
                   showgrid=False),
        yaxis=dict(title='')
    )
    st.plotly_chart(fig6, use_container_width=True)

with s2:
    st.markdown("**Monthly Active Users**")
    mau = (
        filtered.groupby('month')['user_id']
        .nunique().reset_index()
    )
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=mau['month'],
        y=mau['user_id'],
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(234,67,53,0.08)',
        line=dict(color='#EA4335', width=2.5),
        marker=dict(size=6)
    ))
    fig7.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=5, b=0),
        xaxis_title='',
        yaxis_title='Active Users',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.15)'
        )
    )
    st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")

# ---- CHURN RISK ----
st.markdown(
    "<p class='section-label'>"
    "Churn Risk Intelligence &nbsp;·&nbsp; "
    "Random Forest Model &nbsp;·&nbsp; "
    "ROC-AUC 0.883 &nbsp;·&nbsp; "
    "80% Accuracy</p>",
    unsafe_allow_html=True
)

high = filtered_churn[
    filtered_churn['risk_segment'] == 'High Risk']
med = filtered_churn[
    filtered_churn['risk_segment'] == 'Medium Risk']
low = filtered_churn[
    filtered_churn['risk_segment'] == 'Low Risk']

c1, c2, c3 = st.columns(3)
c1.metric("High Risk", len(high),
          "Needs immediate action",
          delta_color="inverse")
c2.metric("Medium Risk", len(med),
          "Monitor this week",
          delta_color="off")
c3.metric("Low Risk", len(low),
          "Healthy & stable",
          delta_color="normal")

ch1, ch2 = st.columns([1, 2])

with ch1:
    st.markdown("**Risk Distribution**")
    fig8 = px.pie(
        filtered_churn,
        names='risk_segment',
        color='risk_segment',
        color_discrete_map={
            'High Risk': '#EA4335',
            'Medium Risk': '#FBBC04',
            'Low Risk': '#34A853'
        },
        hole=0.58
    )
    fig8.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=5, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h', y=-0.2, title=None
        )
    )
    fig8.update_traces(textinfo='percent')
    st.plotly_chart(fig8, use_container_width=True)

with ch2:
    st.markdown(
        "**High Risk Accounts — Intervention Required**"
    )
    display = (
        filtered_churn[
            filtered_churn['risk_segment'] == 'High Risk'
        ][[
            'user_id', 'churn_probability',
            'plan_name', 'avg_sessions',
            'avg_features_used', 'total_tickets'
        ]]
        .sort_values(
            'churn_probability', ascending=False
        )
        .head(8)
    )
    display['churn_probability'] = (
        display['churn_probability']
        .round(3)
    )
    display['avg_sessions'] = (
        display['avg_sessions'].round(1)
    )
    display['avg_features_used'] = (
        display['avg_features_used'].round(1)
    )
    display.columns = [
        'User ID', 'Churn Prob.',
        'Plan', 'Avg Sessions',
        'Features Used', 'Tickets'
    ]
    st.dataframe(
        display.reset_index(drop=True),
        use_container_width=True,
        height=260
    )

st.markdown("---")
st.markdown(
    "<p style='font-size:0.78rem;opacity:0.45;'>"
    "Akshada Karade &nbsp;·&nbsp; "
    "MS Engineering Management, UMass Amherst "
    "&nbsp;·&nbsp; "
    "Python · SQL · Streamlit · Scikit-learn · Plotly"
    "</p>",
    unsafe_allow_html=True
)