import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy — Factory Optimizer",
    page_icon="🍬",
    layout="wide"
)

# ── Load data ──────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    df        = pd.read_csv(os.path.join(base, "outputs", "processed_data.csv"))
    sim       = pd.read_csv(os.path.join(base, "outputs", "simulation_results.csv"))
    recs      = pd.read_csv(os.path.join(base, "outputs", "recommendations.csv"))
    all_recs  = pd.read_csv(os.path.join(base, "outputs", "all_recommendations.csv"))
    fact_prof = pd.read_csv(os.path.join(base, "outputs", "factory_profile.csv"))
    prod_prof = pd.read_csv(os.path.join(base, "outputs", "product_profile.csv"))
    return df, sim, recs, all_recs, fact_prof, prod_prof

df, sim, recs, all_recs, fact_prof, prod_prof = load_data()

ALL_FACTORIES = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    'Sugar Shack',
    'Secret Factory',
    'The Other Factory',
]

FACTORY_MAP = {
    'Wonka Bar - Nutty Crunch Surprise'   : "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows'           : "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious'      : "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate'          : "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel'   : "Wicked Choccy's",
    'Laffy Taffy'                         : 'Sugar Shack',
    'SweeTARTS'                           : 'Sugar Shack',
    'Nerds'                               : 'Sugar Shack',
    'Fun Dip'                             : 'Sugar Shack',
    'Fizzy Lifting Drinks'                : 'Sugar Shack',
    'Everlasting Gobstopper'              : 'Secret Factory',
    'Lickable Wallpaper'                  : 'Secret Factory',
    'Wonka Gum'                           : 'Secret Factory',
    'Kazookles'                           : 'The Other Factory',
    'Hair Toffee'                         : 'The Other Factory',
}

# ── Sidebar ────────────────────────────────────────────


st.sidebar.title("🍬 Nassau Candy")
st.sidebar.markdown("**Factory Optimization System**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview",
     "🏭 Factory Simulator",
     "🔀 What-If Analysis",
     "🏆 Recommendations",
     "⚠️ Risk & Impact"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Optimization Priority**")
speed_weight = st.sidebar.slider(
    "Speed vs Profit", 0, 100, 60,
    help="60 = balanced, 100 = pure speed, 0 = pure profit"
)
profit_weight = 100 - speed_weight
st.sidebar.caption(f"Speed: {speed_weight}%  |  Profit: {profit_weight}%")

# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Executive Overview")
    st.markdown("Performance snapshot across all factories and products.")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Orders",      f"{len(df):,}")
    col2.metric("Total Products",    df['Product Name'].nunique())
    col3.metric("Avg Lead Time",     f"{df['Lead Time'].mean():.0f} days")
    col4.metric("Avg Profit Margin", f"{df['Profit Margin'].mean():.1f}%")
    col5.metric("Factories",         len(ALL_FACTORIES))

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Avg Lead Time by Factory")
        fig = px.bar(
            fact_prof.sort_values('Avg_Lead_Time'),
            x='Avg_Lead_Time', y='Factory',
            orientation='h',
            color='Avg_Lead_Time',
            color_continuous_scale='RdYlGn_r',
            labels={'Avg_Lead_Time':'Avg Lead Time (days)'}
        )
        fig.update_layout(showlegend=False, height=300,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Avg Profit Margin by Factory")
        fig = px.bar(
            fact_prof.sort_values('Avg_Profit_Margin'),
            x='Avg_Profit_Margin', y='Factory',
            orientation='h',
            color='Avg_Profit_Margin',
            color_continuous_scale='RdYlGn',
            labels={'Avg_Profit_Margin':'Profit Margin (%)'}
        )
        fig.update_layout(showlegend=False, height=300,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Lead Time by Product")
        prod_lt = df.groupby('Product Name')['Lead Time'].mean()\
                    .reset_index().sort_values('Lead Time')
        fig = px.bar(
            prod_lt, x='Lead Time', y='Product Name',
            orientation='h',
            color='Lead Time',
            color_continuous_scale='RdYlGn_r',
            labels={'Lead Time':'Avg Lead Time (days)'}
        )
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with col_d:
        st.subheader("Sales vs Profit by Division")
        fig = px.scatter(
            df, x='Sales', y='Gross Profit',
            color='Division', opacity=0.4, size_max=8,
            labels={'Sales':'Sales ($)','Gross Profit':'Gross Profit ($)'}
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Slow Routes — Region + Ship Mode")
    slow = df.groupby(['Region','Ship Mode'])['Lead Time']\
             .mean().reset_index()\
             .sort_values('Lead Time', ascending=False)
    fig = px.bar(
        slow, x='Ship Mode', y='Lead Time',
        color='Region', barmode='group',
        labels={'Lead Time':'Avg Lead Time (days)'}
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, width='stretch')

# ══════════════════════════════════════════════════════
# PAGE 2 — FACTORY SIMULATOR
# ══════════════════════════════════════════════════════
elif page == "🏭 Factory Simulator":
    st.title("🏭 Factory Optimization Simulator")
    st.markdown("Select a product to see predicted performance across all factories.")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_product = st.selectbox(
            "Select Product",
            sorted(df['Product Name'].unique())
        )
    with col2:
        selected_region = st.selectbox(
            "Destination Region",
            sorted(df['Region'].unique())
        )
    with col3:
        selected_ship = st.selectbox(
            "Ship Mode",
            sorted(df['Ship Mode'].unique())
        )

    current_factory = FACTORY_MAP.get(selected_product, 'Unknown')
    st.info(f"**Current Factory:** {current_factory}")

    # Get simulation results for this product
    prod_sim = sim[sim['Product'] == selected_product].copy()

    # Filter all factories
    all_factory_data = []
    for factory in ALL_FACTORIES:
        row = prod_sim[prod_sim['Alt Factory'] == factory]
        if len(row) > 0:
            lt  = row['Alt LT'].values[0]
            pm  = row['Alt Margin %'].values[0]
        else:
            lt  = fact_prof[fact_prof['Factory']==factory]['Avg_Lead_Time'].values[0]
            pm  = fact_prof[fact_prof['Factory']==factory]['Avg_Profit_Margin'].values[0]
        all_factory_data.append({
            'Factory'        : factory,
            'Lead Time'      : lt,
            'Profit Margin %': pm,
            'Is Current'     : factory == current_factory,
        })

    fdf = pd.DataFrame(all_factory_data).sort_values('Lead Time')
    best_factory = fdf.iloc[0]['Factory']

    # Cards
    st.markdown("### Performance across all factories")
    cols = st.columns(len(ALL_FACTORIES))
    for i, (_, row) in enumerate(fdf.iterrows()):
        with cols[i]:
            is_best    = row['Factory'] == best_factory
            is_current = row['Is Current']
            label = "⚡ Best" if is_best else ("📍 Current" if is_current else "")
            st.metric(
                label    = row['Factory'],
                value    = f"{row['Lead Time']:.0f} days",
                delta    = f"Margin: {row['Profit Margin %']:.1f}%",
                delta_color = "normal"
            )
            if label:
                st.caption(label)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Lead Time Comparison")
        colors = ['#2ecc71' if f == best_factory
                  else '#e74c3c' if f == current_factory
                  else '#95a5a6'
                  for f in fdf['Factory']]
        fig = go.Figure(go.Bar(
            x=fdf['Lead Time'], y=fdf['Factory'],
            orientation='h',
            marker_color=colors,
            text=fdf['Lead Time'].round(0).astype(int),
            textposition='outside'
        ))
        fig.update_layout(height=300, showlegend=False,
                          xaxis_title='Avg Lead Time (days)')
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Profit Margin Comparison")
        fig = go.Figure(go.Bar(
            x=fdf['Profit Margin %'], y=fdf['Factory'],
            orientation='h',
            marker_color='#3498db',
            text=fdf['Profit Margin %'].round(1),
            textposition='outside'
        ))
        fig.update_layout(height=300, showlegend=False,
                          xaxis_title='Profit Margin (%)')
        st.plotly_chart(fig, width='stretch')

# ══════════════════════════════════════════════════════
# PAGE 3 — WHAT-IF ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "🔀 What-If Analysis":
    st.title("🔀 What-If Scenario Analysis")
    st.markdown("Compare current vs recommended factory assignments.")

    col1, col2 = st.columns(2)
    with col1:
        wi_product = st.selectbox(
            "Select Product",
            sorted(df['Product Name'].unique())
        )
    with col2:
        wi_alt = st.selectbox(
            "Compare with Factory",
            ALL_FACTORIES
        )

    current_factory = FACTORY_MAP.get(wi_product, 'Unknown')
    cur_lt  = df[df['Product Name']==wi_product]['Lead Time'].mean()
    cur_pm  = df[df['Product Name']==wi_product]['Profit Margin'].mean()
    alt_lt  = fact_prof[fact_prof['Factory']==wi_alt]['Avg_Lead_Time'].values[0]
    alt_pm  = fact_prof[fact_prof['Factory']==wi_alt]['Avg_Profit_Margin'].values[0]

    lt_change = cur_lt - alt_lt
    pm_change = alt_pm - cur_pm

    st.markdown("---")
    st.subheader("Side-by-side comparison")

    col_a, col_b, col_c = st.columns([2, 1, 2])

    with col_a:
        st.markdown(f"### 📍 Current: {current_factory}")
        st.metric("Avg Lead Time",    f"{cur_lt:.0f} days")
        st.metric("Profit Margin",    f"{cur_pm:.1f}%")
        st.metric("Orders Processed", f"{len(df[df['Product Name']==wi_product]):,}")

    with col_b:
        st.markdown("###  ")
        st.markdown("###  ")
        if lt_change > 0:
            st.success(f"▼ {lt_change:.0f} days faster")
        else:
            st.error(f"▲ {abs(lt_change):.0f} days slower")
        if pm_change > 0:
            st.success(f"▲ +{pm_change:.1f}% margin")
        else:
            st.warning(f"▼ {pm_change:.1f}% margin")

    with col_c:
        st.markdown(f"### 🏭 Alternative: {wi_alt}")
        st.metric("Avg Lead Time",  f"{alt_lt:.0f} days",
                  delta=f"{-lt_change:.0f} days",
                  delta_color="inverse")
        st.metric("Profit Margin",  f"{alt_pm:.1f}%",
                  delta=f"{pm_change:+.1f}%")
        orders = len(df[df['Factory']==wi_alt])
        st.metric("Factory Capacity", f"{orders:,} orders")

    st.markdown("---")
    col_d, col_e = st.columns(2)

    with col_d:
        st.subheader("Lead Time: Current vs Alternative")
        fig = go.Figure()
        fig.add_bar(name='Current',     x=['Lead Time'],
                    y=[cur_lt],  marker_color='#e74c3c')
        fig.add_bar(name='Alternative', x=['Lead Time'],
                    y=[alt_lt],  marker_color='#2ecc71')
        fig.update_layout(barmode='group', height=300,
                          yaxis_title='Days')
        st.plotly_chart(fig, width='stretch')

    with col_e:
        st.subheader("Profit Margin: Current vs Alternative")
        fig = go.Figure()
        fig.add_bar(name='Current',     x=['Profit Margin'],
                    y=[cur_pm],  marker_color='#e74c3c')
        fig.add_bar(name='Alternative', x=['Profit Margin'],
                    y=[alt_pm],  marker_color='#2ecc71')
        fig.update_layout(barmode='group', height=300,
                          yaxis_title='Margin (%)')
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Confidence & Risk Assessment")
    factory_orders = len(df[df['Factory']==wi_alt])
    confidence = min(95, round(50 + factory_orders/200, 1))
    if pm_change < -10:
        risk = 'High ⚠️'
        risk_color = 'error'
    elif pm_change < 0:
        risk = 'Medium ⚡'
        risk_color = 'warning'
    else:
        risk = 'Low ✅'
        risk_color = 'success'

    col_f, col_g, col_h = st.columns(3)

    # ══════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════
elif page == "🏆 Recommendations":
    st.title("🏆 Factory Reassignment Recommendations")
    st.markdown("Ranked suggestions based on lead time reduction and profit impact.")

    # Recompute with slider weights
    recs_copy = all_recs.copy()
    sw = speed_weight / 100
    pw = profit_weight / 100
    recs_copy['Weighted Score'] = (
        sw * recs_copy['LT Reduction %'] +
        pw * recs_copy['Profit Impact %']
    ).round(2)

    top = recs_copy.sort_values('Weighted Score', ascending=False)\
                   .groupby('Product').first().reset_index()\
                   .sort_values('Weighted Score', ascending=False)

    # Summary KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products to Reassign",     len(top))
    col2.metric("Avg Lead Time Reduction",  f"{top['LT Reduction %'].mean():.1f}%")
    col3.metric("Low Risk Recommendations", len(top[top['Risk']=='Low']))
    col4.metric("Avg Confidence",           f"{top['Confidence'].mean():.0f}%")

    st.markdown("---")
    st.subheader("Ranked Recommendations")

    display_cols = ['Product','Current Factory','Alt Factory',
                    'LT Reduction %','Profit Impact %',
                    'Risk','Confidence','Weighted Score']
    styled = top[display_cols].reset_index(drop=True)
    styled.index += 1

    def color_risk(val):
        if val == 'High':   return 'background-color: #ffcccc'
        if val == 'Medium': return 'background-color: #fff3cc'
        if val == 'Low':    return 'background-color: #ccffcc'
        return ''

    st.dataframe(
        styled.style.map(color_risk, subset=['Risk']),
        width='stretch'
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Lead Time Reduction by Product")
        fig = px.bar(
            top.sort_values('LT Reduction %'),
            x='LT Reduction %', y='Product',
            orientation='h',
            color='Risk',
            color_discrete_map={
                'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'
            }
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Profit Impact by Product")
        fig = px.bar(
            top.sort_values('Profit Impact %'),
            x='Profit Impact %', y='Product',
            orientation='h',
            color='Profit Impact %',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("All Recommendations Detail")
    st.dataframe(
        top[['Product','Current Factory','Alt Factory',
             'Current LT','Alt LT','LT Reduction',
             'LT Reduction %','Current Margin %',
             'Alt Margin %','Profit Impact %',
             'Risk','Confidence']].reset_index(drop=True),
        width='stretch'
    )

# ══════════════════════════════════════════════════════
# PAGE 5 — RISK & IMPACT
# ══════════════════════════════════════════════════════
elif page == "⚠️ Risk & Impact":
    st.title("⚠️ Risk & Impact Panel")
    st.markdown("Profit impact alerts and high-risk reassignment warnings.")

    st.subheader("Profit Impact Alerts")

    high_risk = all_recs[all_recs['Risk']=='High'].copy()
    med_risk  = all_recs[all_recs['Risk']=='Medium'].copy()
    low_risk  = all_recs[all_recs['Risk']=='Low'].copy()

    if len(high_risk) > 0:
        for _, row in high_risk.iterrows():
            st.error(
                f"⚠️ **{row['Product']}** → {row['Alt Factory']}: "
                f"Margin drops by {abs(row['Profit Impact %']):.1f}% — HIGH RISK"
            )
    if len(med_risk) > 0:
        for _, row in med_risk.iterrows():
            st.warning(
                f"⚡ **{row['Product']}** → {row['Alt Factory']}: "
                f"Margin changes {row['Profit Impact %']:+.1f}% — MEDIUM RISK"
            )
    if len(low_risk) > 0:
        for _, row in low_risk.head(5).iterrows():
            st.success(
                f"✅ **{row['Product']}** → {row['Alt Factory']}: "
                f"Safe reassignment, margin {row['Profit Impact %']:+.1f}%"
            )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Factory Risk Matrix")
        risk_data = fact_prof.copy()
        risk_data['Risk Score'] = (
            risk_data['Avg_Lead_Time'] /
            risk_data['Avg_Lead_Time'].max() * 50 +
            (100 - risk_data['Avg_Profit_Margin']) / 100 * 50
        ).round(1)
        fig = px.scatter(
            risk_data,
            x='Avg_Lead_Time',
            y='Avg_Profit_Margin',
            size='Total_Orders',
            color='Risk Score',
            text='Factory',
            color_continuous_scale='RdYlGn_r',
            labels={
                'Avg_Lead_Time'     : 'Avg Lead Time (days)',
                'Avg_Profit_Margin' : 'Avg Profit Margin (%)',
            }
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Lead Time vs Margin Tradeoff")
        prod_summary = df.groupby('Product Name').agg(
            Lead_Time     = ('Lead Time',     'mean'),
            Profit_Margin = ('Profit Margin', 'mean'),
            Orders        = ('Row ID',        'count')
        ).reset_index()
        fig = px.scatter(
            prod_summary,
            x='Lead_Time',
            y='Profit_Margin',
            size='Orders',
            color='Profit_Margin',
            text='Product Name',
            color_continuous_scale='RdYlGn',
            labels={
                'Lead_Time'     : 'Avg Lead Time (days)',
                'Profit_Margin' : 'Profit Margin (%)'
            }
        )
        fig.update_traces(textposition='top center', textfont_size=8)
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Full Simulation Results")
    filter_risk = st.multiselect(
        "Filter by Risk",
        ['Low','Medium','High'],
        default=['High','Medium']
    )
    filtered = all_recs[all_recs['Risk'].isin(filter_risk)]\
               [['Product','Current Factory','Alt Factory',
                 'LT Reduction %','Profit Impact %',
                 'Risk','Confidence']]\
               .sort_values('LT Reduction %', ascending=False)
    st.dataframe(filtered, width='stretch')

# ── Footer ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Nassau Candy Distributor | Factory Optimization System v1.0")