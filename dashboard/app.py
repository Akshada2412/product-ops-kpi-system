import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from user_analytics_tab import render_user_analytics_tab

st.set_page_config(
    page_title="Product Operations Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Hero header */
    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        margin: -1rem -2rem 1.5rem -2rem;
        background: linear-gradient(
            135deg,
            #0f1b2d 0%,
            #1a2f4e 50%,
            #0d2137 100%
        );
        border-bottom: 1px solid rgba(66,133,244,0.3);
    }
    .hero-tag {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4285F4;
        background: rgba(66,133,244,0.12);
        padding: 3px 12px;
        border-radius: 20px;
        border: 1px solid rgba(66,133,244,0.25);
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-size: 6rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1px;
        line-height: 1.05;
        margin: 0 0 0.5rem 0;
    }
    .hero-title span {
        color: #4285F4;
    }
    .hero-sub {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.5);
        margin: 0;
        line-height: 1.6;
    }
    .hero-pills {
        display: flex;
        gap: 8px;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .pill {
        font-size: 0.72rem;
        font-weight: 500;
        color: rgba(255,255,255,0.6);
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 3px 12px;
        border-radius: 20px;
    }
    /* Section labels */
    .section-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.4;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(128,128,128,0.15);
    }
    /* KPI cards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.15);
        background: rgba(128,128,128,0.04);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        opacity: 0.5;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        line-height: 1;
    }
    .kpi-sub {
        font-size: 0.72rem;
        opacity: 0.45;
        margin-top: 0.3rem;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15,27,45,0.4);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    .sidebar-stat {
        background: rgba(66,133,244,0.08);
        border: 1px solid rgba(66,133,244,0.15);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 8px;
    }
    .sidebar-stat-label {
        font-size: 0.68rem;
        opacity: 0.5;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .sidebar-stat-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #4285F4;
        line-height: 1.2;
    }
    .sidebar-stat-sub {
        font-size: 0.7rem;
        opacity: 0.45;
    }
    /* Chart containers */
    .chart-title {
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.85;
        margin-bottom: 0.25rem;
    }
    /* Modebar */
    .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
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

    uf = activity.groupby('user_id').agg(
        avg_sessions=('sessions', 'mean'),
        avg_features_used=('features_used', 'mean'),
        total_tickets=('support_tickets', 'sum'),
        avg_nps=('nps_score', 'mean'),
        plan=('plan', 'first'),
    ).reset_index()

    uf = uf.merge(
        user_last[['user_id', 'churned']], on='user_id'
    )
    uf = uf.merge(
        users[['user_id', 'industry',
               'company_size', 'acquisition_channel']],
        on='user_id'
    )

    plan_text = uf['plan'].copy()
    industry_text = uf['industry'].copy()

    le = LabelEncoder()
    for col in ['plan', 'industry',
                'company_size', 'acquisition_channel']:
        uf[col] = le.fit_transform(uf[col])

    uf['avg_nps'] = (
        uf['avg_nps'].fillna(uf['avg_nps'].median())
    )

    feature_cols = [
        'avg_sessions', 'avg_features_used',
        'total_tickets', 'avg_nps', 'plan',
        'industry', 'company_size', 'acquisition_channel'
    ]

    X, y = uf[feature_cols], uf['churned']
    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X, y)

    uf['churn_probability'] = model.predict_proba(X)[:, 1]
    uf['plan_name'] = plan_text
    uf['industry_name'] = industry_text

    def risk_segment(p):
        if p >= 0.7: return 'High Risk'
        elif p >= 0.4: return 'Medium Risk'
        return 'Low Risk'

    uf['risk_segment'] = (
        uf['churn_probability'].apply(risk_segment)
    )
    return activity, users, uf

activity, users, churn = load_data()

# ---- HERO HEADER ----
total_users = activity['user_id'].nunique()
churn_rate = round(
    len(churn[churn['churned'] == 1]) /
    len(churn) * 100, 1
)

st.markdown(f"""
<div class='hero'>
    <div class='hero-tag'>Live Analytics Dashboard</div>
    <p style='font-size:3rem;font-weight:800;
              color:#ffffff;letter-spacing:-1px;
              line-height:1.05;margin:0 0 0.5rem 0;'>
        Product Operations<br>
        <span style='color:#4285F4;'>Intelligence</span>
    </p>
    <p class='hero-sub'>
        End-to-end SaaS cohort analysis —
        revenue trends, feature adoption metrics,
        and ML-powered churn risk scoring
        across {total_users:,} users
    </p>
    <div class='hero-pills'>
        <span class='pill'>January – December 2023</span>
        <span class='pill'>5 Industries</span>
        <span class='pill'>4 Plan Tiers</span>
        <span class='pill'>
            Random Forest · ROC-AUC 0.883
        </span>
        <span class='pill'>
            {churn_rate}% Observed Churn Rate
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown(
        "<p style='font-size:0.75rem;font-weight:700;"
        "letter-spacing:0.1em;text-transform:uppercase;"
        "opacity:0.4;margin-bottom:1rem;'>Dashboard Filters</p>",
        unsafe_allow_html=True
    )

    selected_plan = st.multiselect(
        "Plan Tier",
        options=sorted(activity['plan'].unique()),
        default=sorted(activity['plan'].unique())
    )

    industry_options = sorted(
        activity['industry'].unique().tolist()
    )
    selected_industry = st.multiselect(
        "Industry",
        options=industry_options,
        default=industry_options
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.75rem;font-weight:700;"
        "letter-spacing:0.1em;text-transform:uppercase;"
        "opacity:0.4;margin-bottom:0.75rem;'>"
        "Key Stats</p>",
        unsafe_allow_html=True
    )

    # Sidebar stat cards
    filtered_temp = activity[
        activity['plan'].isin(selected_plan) &
        activity['industry'].isin(selected_industry)
    ]
    mrr_temp = filtered_temp['mrr'].sum()
    users_temp = filtered_temp['user_id'].nunique()

    st.markdown(f"""
    <div class='sidebar-stat'>
        <div class='sidebar-stat-label'>Total MRR</div>
        <div class='sidebar-stat-value'>
            ${mrr_temp:,.0f}
        </div>
        <div class='sidebar-stat-sub'>
            filtered selection
        </div>
    </div>
    <div class='sidebar-stat'>
        <div class='sidebar-stat-label'>Active Users</div>
        <div class='sidebar-stat-value'>{users_temp:,}</div>
        <div class='sidebar-stat-sub'>
            in selected cohort
        </div>
    </div>
    """, unsafe_allow_html=True)

    high_temp = len(churn[
        churn['plan_name'].isin(selected_plan) &
        churn['industry_name'].isin(selected_industry) &
        (churn['risk_segment'] == 'High Risk')
    ])

    st.markdown(f"""
    <div class='sidebar-stat' style='border-color:
        rgba(234,67,53,0.25);
        background:rgba(234,67,53,0.06);'>
        <div class='sidebar-stat-label'>High Risk Users</div>
        <div class='sidebar-stat-value'
             style='color:#EA4335;'>{high_temp}</div>
        <div class='sidebar-stat-sub'>
            need immediate attention
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.68rem;opacity:0.3;"
        "line-height:1.7;'>"
        "Built by Akshada Karade<br>"
        "MS Eng. Mgmt · UMass Amherst<br>"
        "Python · SQL · Streamlit · sklearn</p>",
        unsafe_allow_html=True
    )

# ---- FILTER ----
filtered = activity[
    activity['plan'].isin(selected_plan) &
    activity['industry'].isin(selected_industry)
]

filtered_churn = churn[
    churn['plan_name'].isin(selected_plan) &
    churn['industry_name'].isin(selected_industry)
]

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

k1.metric("Monthly Recurring Revenue",
          f"${total_mrr:,.0f}")
k2.metric("Active Users", f"{active_users:,}")
k3.metric("Avg Sessions / User", f"{avg_sessions:.1f}")
k4.metric("Avg NPS Score", f"{avg_nps:.1f} / 10")
k5.metric("High Churn Risk", f"{high_risk_n} users",
          delta=f"{churn_rate}% overall churn",
          delta_color="inverse")

st.markdown("---")

# ---- REVENUE ----
st.markdown(
    "<p class='section-label'>Revenue</p>",
    unsafe_allow_html=True
)

r1, r2 = st.columns([3, 2])

with r1:
    st.markdown(
        "<p class='chart-title'>"
        "Monthly Recurring Revenue by Plan</p>",
        unsafe_allow_html=True
    )
    mrr_by_plan = (
        filtered.groupby(['month', 'plan'])['mrr']
        .sum().reset_index()
    )
    fig1 = px.area(
        mrr_by_plan, x='month', y='mrr', color='plan',
        color_discrete_sequence=[
            '#4285F4', '#34A853', '#FBBC04', '#EA4335'
        ],
    )
    fig1.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Month: %{x}<br>"
            "MRR: <b>$%{y:,.0f}</b>"
            "<extra></extra>"
        )
    )
    fig1.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=5, b=0),
        xaxis_title='', yaxis_title='MRR (USD)',
        legend=dict(
            orientation='h', y=1.15, title=None
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='rgba(128,128,128,0.12)')
    )
    st.plotly_chart(fig1, use_container_width=True)

with r2:
    st.markdown(
        "<p class='chart-title'>"
        "Revenue Concentration by Plan</p>",
        unsafe_allow_html=True
    )
    rev_plan = (
        filtered.groupby('plan')['mrr']
        .sum().reset_index()
    )
    fig2 = px.pie(
        rev_plan, values='mrr', names='plan',
        color_discrete_sequence=[
            '#4285F4', '#34A853', '#FBBC04', '#EA4335'
        ],
        hole=0.62
    )
    fig2.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=5, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h', y=-0.15, title=None
        )
    )
    fig2.update_traces(
        textinfo='percent', textposition='outside'
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---- ENGAGEMENT ----
st.markdown(
    "<p class='section-label'>User Engagement</p>",
    unsafe_allow_html=True
)

e1, e2, e3 = st.columns(3)

CHART_H = 260

with e1:
    st.markdown(
        "<p class='chart-title'>Sessions by Plan</p>",
        unsafe_allow_html=True
    )
    sess = (
        filtered.groupby('plan')['sessions']
        .mean().reset_index()
        .sort_values('sessions', ascending=True)
    )
    fig3 = px.bar(
        sess, x='sessions', y='plan',
        orientation='h',
        color='sessions',
        color_continuous_scale=['#1a4fa0', '#4285F4'],
        text=sess['sessions'].round(1)
    )
    fig3.update_traces(
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg Sessions: <b>%{x:.1f}</b> / month<br>"
            "<i>Higher = more engaged users</i>"
            "<extra></extra>"
        )
    )
    fig3.update_layout(
        height=CHART_H,
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
    st.markdown(
        "<p class='chart-title'>"
        "Feature Adoption by Plan</p>",
        unsafe_allow_html=True
    )
    feat = (
        filtered.groupby('plan')['features_used']
        .mean().reset_index()
        .sort_values('features_used', ascending=True)
    )
    fig4 = px.bar(
        feat, x='features_used', y='plan',
        orientation='h',
        color='features_used',
        color_continuous_scale=['#1a6e3c', '#34A853'],
        text=feat['features_used'].round(1)
    )
    fig4.update_traces(
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg Features Used: <b>%{x:.1f}</b> / 10<br>"
            "<i>Higher = deeper product adoption</i>"
            "<extra></extra>"
        )
    )
    fig4.update_layout(
        height=CHART_H,
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
    st.markdown(
        "<p class='chart-title'>"
        "Support Load per User</p>",
        unsafe_allow_html=True
    )
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
        tix, x='per_user', y='plan',
        orientation='h',
        color='per_user',
        color_continuous_scale=[
            '#34A853', '#FBBC04', '#EA4335'
        ],
        text='per_user'
    )
    fig5.update_traces(
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Tickets per User: <b>%{x:.2f}</b><br>"
            "<i>Lower = healthier, less frustrated users</i>"
            "<extra></extra>"
        )
    )
    fig5.update_layout(
        height=CHART_H,
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
    st.markdown(
        "<p class='chart-title'>NPS Score by Plan</p>",
        unsafe_allow_html=True
    )
    nps = (
        filtered.groupby('plan')['nps_score']
        .mean().reset_index()
    )
    nps.columns = ['plan', 'avg_nps']
    nps = nps.sort_values('avg_nps', ascending=True)

    fig6 = px.bar(
        nps, x='avg_nps', y='plan',
        orientation='h',
        color='avg_nps',
        color_continuous_scale=[
            '#EA4335', '#FBBC04', '#34A853'
        ],
        text=nps['avg_nps'].round(2),
        range_color=[0, 10],
        hover_data={'avg_nps': ':.2f'},
        custom_data=['avg_nps']
    )
    fig6.update_traces(
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg NPS Score: <b>%{x:.2f}</b> / 10<br>"
            "<i>Score above 7 = strong satisfaction</i>"
            "<extra></extra>"
        )
    )
    fig6.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=5, b=0),
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Avg NPS Score',
                   range=[0, 10], showgrid=False),
        yaxis=dict(title='')
    )
    st.plotly_chart(fig6, use_container_width=True)

with s2:
    st.markdown(
        "<p class='chart-title'>"
        "Monthly Active Users</p>",
        unsafe_allow_html=True
    )
    mau = (
        filtered.groupby('month')['user_id']
        .nunique().reset_index()
    )
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=mau['month'], y=mau['user_id'],
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(234,67,53,0.07)',
        line=dict(color='#EA4335', width=2.5),
        marker=dict(size=6)
    ))
    fig7.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=5, b=0),
        xaxis_title='', yaxis_title='Active Users',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.12)'
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
c1.metric("High Risk Users", len(high),
          "Needs immediate action",
          delta_color="inverse")
c2.metric("Medium Risk Users", len(med),
          "Monitor this week",
          delta_color="off")
c3.metric("Low Risk Users", len(low),
          "Healthy & stable")

ch1, ch2 = st.columns([1, 2])

with ch1:
    st.markdown(
        "<p class='chart-title'>"
        "Risk Distribution</p>",
        unsafe_allow_html=True
    )
    fig8 = px.pie(
        filtered_churn,
        names='risk_segment',
        color='risk_segment',
        color_discrete_map={
            'High Risk': '#EA4335',
            'Medium Risk': '#FBBC04',
            'Low Risk': '#34A853'
        },
        hole=0.6
    )
    fig8.update_traces(
        textinfo='percent',
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Users: <b>%{value}</b><br>"
            "Share: <b>%{percent}</b><br>"
            "<extra></extra>"
        )
    )
    fig8.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=5, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h', y=-0.2, title=None
        )
    )
    st.plotly_chart(fig8, use_container_width=True)

with ch2:
    st.markdown(
        "<p class='chart-title'>"
        "High Risk Accounts — Intervention Required</p>",
        unsafe_allow_html=True
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
        display['churn_probability'].round(3)
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
    "<p class='section-label'>User Analytics</p>",
    unsafe_allow_html=True
)

render_user_analytics_tab()