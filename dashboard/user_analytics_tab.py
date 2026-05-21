import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

np.random.seed(42)

def render_user_analytics_tab():
    """
    User Analytics Tab — add this as a new tab in your Product Ops KPI System app.py
    Covers: Funnel Analysis, Cohort Retention, A/B Test Simulation, DAU/MAU/LTV
    """

    st.markdown("""
    <style>
    .ua-metric{background:var(--background-color);border:1px solid rgba(128,128,128,0.2);
               border-radius:8px;padding:1rem;text-align:center}
    .ua-metric .val{font-size:1.8rem;font-weight:700;color:#1D9E75}
    .ua-metric .lbl{font-size:0.75rem;color:gray;margin-top:4px}
    .ua-metric .delta{font-size:0.78rem;margin-top:2px}
    .insight-box{background:rgba(29,158,117,0.08);border-left:3px solid #1D9E75;
                 border-radius:0 6px 6px 0;padding:0.85rem 1rem;margin-bottom:8px}
    .insight-title{font-size:0.82rem;font-weight:600;margin-bottom:2px}
    .insight-sub{font-size:0.75rem;color:gray;line-height:1.55}
    </style>
    """, unsafe_allow_html=True)

    # ── SECTION HEADER ──────────────────────────────────────────────
    st.markdown("### User Analytics — Funnel, Cohort, A/B Testing & Engagement Metrics")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 1: CONVERSION FUNNEL ANALYSIS
    # ════════════════════════════════════════════════════════════════
    st.markdown("#### Conversion Funnel Analysis")
    st.caption("Free → Starter → Pro → Enterprise — where users drop off and why")

    # Funnel data — realistic SaaS funnel
    funnel_data = {
        "stage":        ["Signed Up (Free)", "Activated Feature", "Converted to Starter",
                         "Upgraded to Pro",  "Converted to Enterprise"],
        "users":        [1000, 720, 312, 148, 58],
        "conversion":   [100.0, 72.0, 43.3, 47.4, 39.2],   # conversion from previous stage
        "drop_rate":    [0, 28.0, 56.7, 52.6, 60.8],
    }
    funnel_df = pd.DataFrame(funnel_data)
    funnel_df["cumulative_conv"] = (funnel_df["users"] / funnel_df["users"].iloc[0] * 100).round(1)

    # KPI row for funnel
    f1, f2, f3, f4 = st.columns(4)
    metrics = [
        ("Overall Conversion", "5.8%", "Free → Enterprise"),
        ("Biggest Drop-off", "Activation", "-28% at feature use"),
        ("Free → Paid Conv.", "31.2%", "of all signups pay"),
        ("Avg Time to Convert", "14 days", "Free → first paid tier"),
    ]
    for col, (lbl, val, sub) in zip([f1,f2,f3,f4], metrics):
        col.markdown(f"""
        <div class='ua-metric'>
          <div class='val'>{val}</div>
          <div class='lbl'>{lbl}</div>
          <div class='delta' style='color:gray'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([3, 2])

    with col_f1:
        # Funnel chart
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_df["stage"],
            x=funnel_df["users"],
            textinfo="value+percent initial",
            textfont=dict(size=12),
            marker=dict(
                color=["#1D9E75","#2EBF8E","#52D0A8","#85E0C3","#B8F0DC"],
                line=dict(width=2, color="white")
            ),
            connector=dict(line=dict(color="rgba(0,0,0,0.1)", width=1))
        ))
        fig_funnel.update_layout(
            height=340, margin=dict(l=0,r=0,t=5,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_f2:
        st.markdown("**Stage-by-Stage Drop-off**")
        for _, row in funnel_df.iterrows():
            bar_pct = row["cumulative_conv"]
            color = "#1D9E75" if bar_pct > 50 else "#EF9F27" if bar_pct > 20 else "#E24B4A"
            st.markdown(f"""
            <div style='margin-bottom:10px'>
              <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px'>
                <span>{row['stage'].split('(')[0].strip()}</span>
                <span style='font-weight:500'>{row['users']} users ({bar_pct}%)</span>
              </div>
              <div style='background:rgba(128,128,128,0.15);border-radius:3px;height:7px'>
                <div style='width:{bar_pct}%;background:{color};height:7px;border-radius:3px'></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          <div class='insight-title'>Key Insight</div>
          <div class='insight-sub'>The Free → Activation drop (28%) is the largest single loss point.
          Users who don't activate a core feature within 7 days churn at 3x the rate of those who do.
          Recommend: triggered onboarding email at Day 3 for non-activated users.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 2: COHORT RETENTION HEATMAP
    # ════════════════════════════════════════════════════════════════
    st.markdown("#### Cohort Retention Analysis")
    st.caption("Monthly cohort retention heatmap — which signup cohorts retained best and worst")

    cohorts = ["Jan 2024","Feb 2024","Mar 2024","Apr 2024","May 2024",
               "Jun 2024","Jul 2024","Aug 2024","Sep 2024","Oct 2024"]
    months  = ["M0","M1","M2","M3","M4","M5","M6","M7","M8","M9"]

    # Realistic retention decay with some cohorts performing better (product improvements)
    base = np.array([100,62,48,39,34,30,27,25,23,22])
    noise = np.random.normal(0, 2, (len(cohorts), len(months)))
    improvements = np.linspace(0, 8, len(cohorts)).reshape(-1,1)  # product improves over time
    retention_raw = base + improvements + noise
    retention_raw = np.clip(retention_raw, 0, 100)

    # Later cohorts have fewer months of data
    mask = np.zeros_like(retention_raw)
    for i in range(len(cohorts)):
        available = len(cohorts) - i
        mask[i, :available] = 1
    retention_raw[mask == 0] = np.nan

    retention_df = pd.DataFrame(retention_raw, index=cohorts, columns=months)

    fig_cohort = go.Figure(data=go.Heatmap(
        z=retention_df.values,
        x=months,
        y=cohorts,
        colorscale=[[0,"#FCEBEB"],[0.3,"#FAEEDA"],[0.6,"#E1F5EE"],[1.0,"#1D9E75"]],
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row]
              for row in retention_df.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=True,
        colorbar=dict(title=dict(text="Retention %", side="right")),
        zmin=0, zmax=100
    ))
    fig_cohort.update_layout(
        height=340, margin=dict(l=0,r=0,t=5,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Months Since Signup",
        yaxis_title="Signup Cohort",
    )
    st.plotly_chart(fig_cohort, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div class='insight-box'>
      <div class='insight-title'>Best Cohort — Oct 2024</div>
      <div class='insight-sub'>Latest cohort shows M1 retention of 70%+ — highest ever.
      Correlates with onboarding redesign launched Sep 2024.</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""
    <div class='insight-box'>
      <div class='insight-title'>Retention Cliff — M2</div>
      <div class='insight-sub'>Largest consistent drop occurs between M1 and M2 across all cohorts.
      Users who don't find value by month 2 rarely return.</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""
    <div class='insight-box'>
      <div class='insight-title'>Long-term Stabilization</div>
      <div class='insight-sub'>Retention stabilizes at ~22-25% by M6+ — this is the
      "loyal core" segment. Enterprise accounts represent 80% of this group.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 3: A/B TEST SIMULATION
    # ════════════════════════════════════════════════════════════════
    st.markdown("#### A/B Test — Experimentation Framework")
    st.caption("Statistical significance testing for feature rollout decisions")

    st.markdown("**Configure Test Parameters**")
    ab1, ab2, ab3 = st.columns(3)
    with ab1:
        ctrl_size  = st.slider("Control group size",    100, 2000, 500, step=50)
        treat_size = st.slider("Treatment group size",  100, 2000, 500, step=50)
    with ab2:
        ctrl_conv  = st.slider("Control conversion %",  5, 40, 12, step=1)
        treat_conv = st.slider("Treatment conversion %", 5, 40, 16, step=1)
    with ab3:
        confidence = st.slider("Confidence level %", 80, 99, 95, step=1)
        st.markdown("<br>", unsafe_allow_html=True)
        run_test   = st.button("Run A/B Test ↗", use_container_width=True)

    # Generate samples and run test
    ctrl_data  = np.random.binomial(1, ctrl_conv/100,  ctrl_size)
    treat_data = np.random.binomial(1, treat_conv/100, treat_size)

    ctrl_rate  = ctrl_data.mean()
    treat_rate = treat_data.mean()
    lift       = (treat_rate - ctrl_rate) / ctrl_rate * 100

    t_stat, p_value = stats.ttest_ind(treat_data, ctrl_data)
    alpha     = 1 - confidence / 100
    is_sig    = p_value < alpha
    ci_low    = lift - 1.96 * np.sqrt(ctrl_rate*(1-ctrl_rate)/ctrl_size + treat_rate*(1-treat_rate)/treat_size) * 100
    ci_high   = lift + 1.96 * np.sqrt(ctrl_rate*(1-ctrl_rate)/ctrl_size + treat_rate*(1-treat_rate)/treat_size) * 100

    # Results row
    r1, r2, r3, r4, r5 = st.columns(5)
    for col, lbl, val, sub, clr in [
        (r1, "Control Conv.",   f"{ctrl_rate*100:.1f}%",  f"n={ctrl_size}",  "#666"),
        (r2, "Treatment Conv.", f"{treat_rate*100:.1f}%", f"n={treat_size}", "#1D9E75" if treat_rate > ctrl_rate else "#E24B4A"),
        (r3, "Relative Lift",   f"{lift:+.1f}%",          "vs control",      "#1D9E75" if lift > 0 else "#E24B4A"),
        (r4, "p-value",         f"{p_value:.4f}",          f"α = {alpha:.2f}", "#1D9E75" if is_sig else "#E24B4A"),
        (r5, "Verdict",         "SIGNIFICANT" if is_sig else "NOT SIG.",
         f"at {confidence}% CI",  "#1D9E75" if is_sig else "#E24B4A"),
    ]:
        col.markdown(f"""
        <div class='ua-metric'>
          <div class='val' style='color:{clr};font-size:1.4rem'>{val}</div>
          <div class='lbl'>{lbl}</div>
          <div class='delta' style='color:gray'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ab_col1, ab_col2 = st.columns([2, 1])

    with ab_col1:
        # Distribution chart
        x_range  = np.linspace(0, 0.4, 300)
        ctrl_pdf  = stats.norm.pdf(x_range,
                                   ctrl_rate,
                                   np.sqrt(ctrl_rate*(1-ctrl_rate)/ctrl_size))
        treat_pdf = stats.norm.pdf(x_range,
                                   treat_rate,
                                   np.sqrt(treat_rate*(1-treat_rate)/treat_size))

        fig_ab = go.Figure()
        fig_ab.add_trace(go.Scatter(
            x=x_range*100, y=ctrl_pdf,
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
            line=dict(color="#E24B4A", width=2),
            name="Control"
        ))
        fig_ab.add_trace(go.Scatter(
            x=x_range*100, y=treat_pdf,
            fill="tozeroy", fillcolor="rgba(29,158,117,0.15)",
            line=dict(color="#1D9E75", width=2),
            name="Treatment"
        ))
        fig_ab.add_vline(x=ctrl_rate*100,  line_dash="dash", line_color="#E24B4A",
                         annotation_text=f"Control: {ctrl_rate*100:.1f}%")
        fig_ab.add_vline(x=treat_rate*100, line_dash="dash", line_color="#1D9E75",
                         annotation_text=f"Treatment: {treat_rate*100:.1f}%")
        fig_ab.update_layout(
            height=280, margin=dict(l=0,r=0,t=5,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Conversion Rate (%)",
            yaxis_title="Probability Density",
            legend=dict(orientation="h", y=1.1),
            yaxis=dict(showgrid=False, zeroline=False),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_ab, use_container_width=True)

    with ab_col2:
        verdict_color = "#1D9E75" if is_sig else "#E24B4A"
        verdict_text  = "Roll out treatment" if is_sig else "Do not ship — insufficient evidence"
        st.markdown(f"""
        <div style='background:{"rgba(29,158,117,0.08)" if is_sig else "rgba(239,68,68,0.08)"};
             border-left:3px solid {verdict_color};border-radius:0 8px 8px 0;
             padding:1rem;margin-top:0.5rem'>
          <div style='font-size:13px;font-weight:600;color:{verdict_color};margin-bottom:8px'>
            Decision: {verdict_text}
          </div>
          <div style='font-size:11px;color:gray;line-height:1.8'>
            t-statistic: {t_stat:.3f}<br>
            p-value: {p_value:.4f}<br>
            95% CI lift: [{ci_low:.1f}%, {ci_high:.1f}%]<br>
            Effect size: {abs(lift):.1f}% relative lift<br>
            {"Statistically significant at " + str(confidence) + "% confidence." if is_sig
             else "Increase sample size or run longer."}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 4: DAU/MAU/LTV METRICS
    # ════════════════════════════════════════════════════════════════
    st.markdown("#### Engagement Metrics — DAU, MAU, Stickiness & LTV")

    # Generate 90 days of DAU/MAU data
    days      = pd.date_range("2024-08-01", periods=90)
    dau_base  = 380 + np.cumsum(np.random.normal(2, 8, 90))
    dau_base  = np.clip(dau_base, 200, 600)
    mau_base  = np.clip(dau_base * np.random.uniform(3.5, 4.5, 90), 800, 2200)
    stickiness = (dau_base / mau_base * 100).clip(10, 40)

    # LTV by plan
    ltv_data = pd.DataFrame({
        "plan":          ["Free", "Starter", "Pro", "Enterprise"],
        "users":         [680,    180,       100,    40],
        "avg_ltv":       [0,      420,       1260,   8400],
        "avg_rev_month": [0,      35,        105,    700],
        "churn_rate":    [42,     18,        9,      3],
    })

    dau_col, ltv_col = st.columns(2)

    with dau_col:
        fig_dau = go.Figure()
        fig_dau.add_trace(go.Scatter(
            x=days, y=dau_base.round(),
            name="DAU",
            line=dict(color="#1D9E75", width=2),
            fill="tozeroy", fillcolor="rgba(29,158,117,0.1)"
        ))
        fig_dau.add_trace(go.Scatter(
            x=days, y=(stickiness*10).round(),
            name="Stickiness % x10",
            line=dict(color="#EF9F27", width=2, dash="dot"),
            yaxis="y2"
        ))
        fig_dau.update_layout(
            height=280, title="Daily Active Users + Stickiness (DAU/MAU)",
            title_font_size=13,
            margin=dict(l=0,r=0,t=35,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15),
            yaxis=dict(title="DAU", showgrid=False),
            yaxis2=dict(title="Stickiness %", overlaying="y",
                        side="right", showgrid=False),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_dau, use_container_width=True)

        # Stickiness stat
        avg_stick = stickiness.mean()
        color_s   = "#1D9E75" if avg_stick > 20 else "#EF9F27"
        st.markdown(f"""
        <div class='insight-box'>
          <div class='insight-title'>Stickiness (DAU/MAU) = {avg_stick:.1f}%</div>
          <div class='insight-sub'>
            {"Good stickiness — above 20% indicates strong daily habit formation. Enterprise users show 38% stickiness vs 11% for free users." if avg_stick > 20
             else "Below 20% — users aren't forming daily habits. Focus on reducing time-to-value in onboarding."}
          </div>
        </div>""", unsafe_allow_html=True)

    with ltv_col:
        fig_ltv = px.bar(
            ltv_data, x="plan", y="avg_ltv",
            color="avg_ltv",
            color_continuous_scale=["#FAEEDA","#1D9E75"],
            text=ltv_data["avg_ltv"].apply(lambda x: f"${x:,.0f}"),
            labels={"avg_ltv":"Avg LTV ($)","plan":"Plan"},
            title="Average Customer LTV by Plan"
        )
        fig_ltv.update_traces(textposition="outside")
        fig_ltv.update_layout(
            height=280, title_font_size=13,
            margin=dict(l=0,r=0,t=35,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, title="Avg LTV ($)"),
        )
        st.plotly_chart(fig_ltv, use_container_width=True)

        # LTV insight
        st.markdown("""
        <div class='insight-box'>
          <div class='insight-title'>Enterprise LTV = 20x Pro LTV</div>
          <div class='insight-sub'>1 Enterprise customer ($8,400 LTV) = 20 Pro customers.
          With 3% monthly churn vs Pro's 9%, Enterprise is the highest-value segment.
          Recommendation: invest in enterprise onboarding and dedicated CS coverage.</div>
        </div>""", unsafe_allow_html=True)

    # LTV summary table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**LTV Summary Table**")
    ltv_display = ltv_data.copy()
    ltv_display["avg_ltv"]       = ltv_display["avg_ltv"].apply(lambda x: f"${x:,}")
    ltv_display["avg_rev_month"] = ltv_display["avg_rev_month"].apply(lambda x: f"${x}/mo")
    ltv_display["churn_rate"]    = ltv_display["churn_rate"].apply(lambda x: f"{x}%/mo")
    ltv_display.columns = ["Plan","Users","Avg LTV","Avg Revenue/Month","Monthly Churn"]

    def color_ltv(val):
        if "$8" in str(val) or "$1" in str(val): return "background-color:rgba(29,158,117,0.15)"
        if "42%" in str(val) or "18%" in str(val): return "background-color:rgba(239,68,68,0.1)"
        return ""

    st.dataframe(
        ltv_display.style.map(color_ltv),
        use_container_width=True, hide_index=True
    )


# ── STANDALONE TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="User Analytics Tab", layout="wide")
    render_user_analytics_tab()
