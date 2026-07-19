"""
Nassau Candy Distributor — Factory-to-Customer Shipping Route Efficiency Dashboard
====================================================================================
Run locally with:
    pip install streamlit pandas plotly
    streamlit run app.py

Expects 'nassau_cleaned.csv' (produced by process.py) in the same directory.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy | Shipping Route Efficiency",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#1B2A4A"
BLUE = "#3B6FD6"
GOLD = "#D6A93B"
RED = "#C0455F"
GREEN = "#4F9D69"
GREY = "#8A93A6"
PALETTE = [BLUE, GOLD, GREEN, RED, "#7D5BA6"]

st.markdown(f"""
<style>
    .main {{ background-color: #F7F8FA; }}
    h1, h2, h3 {{ color: {NAVY}; }}
    div[data-testid="stMetric"] {{
        background-color: white; border-radius: 10px; padding: 14px 16px;
        border: 1px solid #E7EAF0; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}
    div[data-testid="stMetricLabel"] {{ color: {GREY}; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    section[data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid #E7EAF0; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Arial", color=NAVY),
        paper_bgcolor="white", plot_bgcolor="white",
        colorway=PALETTE,
        title_font=dict(size=16, color=NAVY),
        xaxis=dict(gridcolor="#EEF1F5"), yaxis=dict(gridcolor="#EEF1F5"),
        margin=dict(t=50, l=10, r=10, b=10),
    )
)

# ---------------------------------------------------------------------------
# DATA LOAD
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("nassau_cleaned.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df

df = load_data()

# ---------------------------------------------------------------------------
# SIDEBAR — USER CAPABILITIES (filters)
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🍬 Nassau Candy")
st.sidebar.caption("Factory-to-Customer Shipping Route Efficiency")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

min_d, max_d = df["Order Date"].min().date(), df["Order Date"].max().date()
date_range = st.sidebar.date_input("Order date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

regions = st.sidebar.multiselect("Region", sorted(df["Region"].unique()), default=list(sorted(df["Region"].unique())))
states_avail = sorted(df[df["Region"].isin(regions)]["State/Province"].unique()) if regions else sorted(df["State/Province"].unique())
states = st.sidebar.multiselect("State / Province", states_avail, default=[])

ship_modes = st.sidebar.multiselect("Ship Mode", sorted(df["Ship Mode"].unique()), default=list(sorted(df["Ship Mode"].unique())))

lead_max = float(np.ceil(df["Modeled_Lead_Time"].max()))
lead_threshold = st.sidebar.slider("Lead-time delay threshold (days)", 0.0, lead_max, round(df["Modeled_Lead_Time"].quantile(0.75), 1), 0.1,
                                    help="Shipments above this many days are treated as delayed for the KPIs below.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Data quality note:** the raw 'Ship Date' field in the source system is corrupted "
    "(shows dates 2.5–4.5 years after Order Date for 100% of records). Lead-time figures in "
    "this dashboard use a documented **modeled lead time** based on shipping distance and ship "
    "mode. See the Research Paper, Section 4.2, for full detail."
)

# ---------------------------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------------------------
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_d, max_d

f = df[
    (df["Order Date"].dt.date >= start_d) & (df["Order Date"].dt.date <= end_d) &
    (df["Region"].isin(regions) if regions else True) &
    (df["Ship Mode"].isin(ship_modes) if ship_modes else True)
]
if states:
    f = f[f["State/Province"].isin(states)]
f = f.assign(Delayed_Now=f["Modeled_Lead_Time"] > lead_threshold)

if f.empty:
    st.warning("No shipments match the current filters. Adjust filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("Factory-to-Customer Shipping Route Efficiency")
st.caption("Route Efficiency Overview · Geographic Shipping Map · Ship Mode Comparison · Route Drill-Down")

# ---------------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Shipments", f"{len(f):,}")
c2.metric("Avg. Lead Time", f"{f['Modeled_Lead_Time'].mean():.2f} days")
c3.metric("Delay Frequency", f"{f['Delayed_Now'].mean()*100:.1f}%")
c4.metric("Total Sales", f"${f['Sales'].sum():,.0f}")
c5.metric("Total Gross Profit", f"${f['Gross Profit'].sum():,.0f}")

st.markdown("")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Route Efficiency Overview", "🗺️ Geographic Shipping Map", "🚚 Ship Mode Comparison", "🔎 Route Drill-Down"])

# ---------------------------------------------------------------------------
# TAB 1 — ROUTE EFFICIENCY OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Average Lead Time by Route")
        route_agg = f.groupby(["Factory", "State/Province"]).agg(
            Shipments=("Row ID", "count"),
            Avg_Lead_Time=("Modeled_Lead_Time", "mean"),
            Delay_Rate=("Delayed_Now", "mean"),
            Sales=("Sales", "sum"),
        ).reset_index()
        route_agg["Route"] = route_agg["Factory"] + " → " + route_agg["State/Province"]
        route_agg = route_agg[route_agg["Shipments"] >= 5]

        min_ship = st.slider("Minimum shipments per route to include", 1, 50, 15, key="minship1")
        rr = route_agg[route_agg["Shipments"] >= min_ship].sort_values("Avg_Lead_Time")

        n_show = min(15, len(rr))
        combo = pd.concat([rr.head(n_show // 2 + n_show % 2), rr.tail(n_show // 2)]).drop_duplicates()
        combo["Efficiency"] = np.where(combo["Avg_Lead_Time"] <= combo["Avg_Lead_Time"].median(), "Fast", "Slow")
        fig = px.bar(combo.sort_values("Avg_Lead_Time"), x="Avg_Lead_Time", y="Route", orientation="h",
                     color="Efficiency", color_discrete_map={"Fast": GREEN, "Slow": RED},
                     hover_data={"Shipments": True, "Delay_Rate": ":.1%"})
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=520, showlegend=True, yaxis_title="", xaxis_title="Avg Lead Time (days)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Route Performance Leaderboard")
        board = route_agg.sort_values("Avg_Lead_Time")[["Route", "Shipments", "Avg_Lead_Time", "Delay_Rate", "Sales"]]
        board["Avg_Lead_Time"] = board["Avg_Lead_Time"].round(2)
        board["Delay_Rate"] = (board["Delay_Rate"] * 100).round(1).astype(str) + "%"
        board["Sales"] = board["Sales"].map(lambda v: f"${v:,.0f}")
        st.dataframe(board.rename(columns={"Avg_Lead_Time": "Avg Days", "Delay_Rate": "Delay %"}),
                     use_container_width=True, height=520, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sales & Profit by Division")
        div = f.groupby("Division").agg(Sales=("Sales", "sum"), Profit=("Gross Profit", "sum")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=div["Division"], y=div["Sales"], name="Sales", marker_color=BLUE)
        fig.add_bar(x=div["Division"], y=div["Profit"], name="Gross Profit", marker_color=GOLD)
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], barmode="group", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Monthly Shipment Volume")
        monthly = f.set_index("Order Date").resample("MS").size().reset_index(name="Shipments")
        fig = px.area(monthly, x="Order Date", y="Shipments")
        fig.update_traces(line_color=BLUE, fillcolor="rgba(59,111,214,0.12)")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — GEOGRAPHIC SHIPPING MAP
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("US Heatmap of Shipping Efficiency (by State)")
    us_states = {
        "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
        "Connecticut":"CT","Delaware":"DE","District of Columbia":"DC","Florida":"FL","Georgia":"GA",
        "Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY",
        "Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
        "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH",
        "New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND",
        "Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI",
        "South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT",
        "Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
    }
    metric_choice = st.radio("Map metric", ["Average Lead Time (days)", "Delay Frequency (%)", "Shipment Volume", "Total Sales ($)"], horizontal=True)
    us_only = f[f["State/Province"].isin(us_states.keys())].copy()
    us_only["Code"] = us_only["State/Province"].map(us_states)
    geo = us_only.groupby(["State/Province", "Code"]).agg(
        Avg_Lead_Time=("Modeled_Lead_Time", "mean"), Delay_Rate=("Delayed_Now", "mean"),
        Shipments=("Row ID", "count"), Sales=("Sales", "sum")
    ).reset_index()

    metric_map = {
        "Average Lead Time (days)": ("Avg_Lead_Time", "Reds"),
        "Delay Frequency (%)": ("Delay_Rate", "Reds"),
        "Shipment Volume": ("Shipments", "Blues"),
        "Total Sales ($)": ("Sales", "Greens"),
    }
    col, scale = metric_map[metric_choice]
    fig = px.choropleth(geo, locations="Code", locationmode="USA-states", color=col,
                         scope="usa", color_continuous_scale=scale, hover_name="State/Province",
                         hover_data={"Avg_Lead_Time":":.2f", "Delay_Rate":":.1%", "Shipments":True, "Sales":":$,.0f", "Code":False})
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regional Bottleneck Visualization")
    reg = f.groupby("Region").agg(Avg_Lead_Time=("Modeled_Lead_Time","mean"), Delay_Rate=("Delayed_Now","mean"),
                                   Shipments=("Row ID","count")).reset_index()
    fig2 = px.scatter(reg, x="Avg_Lead_Time", y="Delay_Rate", size="Shipments", color="Region",
                       text="Region", size_max=60, color_discrete_sequence=PALETTE)
    fig2.update_traces(textposition="top center")
    fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=420,
                        xaxis_title="Avg Lead Time (days)", yaxis_title="Delay Frequency",
                        yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Bubble size = shipment volume. Top-right bubbles are high-volume, high-delay bottlenecks warranting priority attention.")

# ---------------------------------------------------------------------------
# TAB 3 — SHIP MODE COMPARISON
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Lead Time Comparison by Shipping Method")
    order = ["Same Day", "First Class", "Second Class", "Standard Class"]
    order = [o for o in order if o in f["Ship Mode"].unique()]
    fig = px.box(f, x="Ship Mode", y="Modeled_Lead_Time", color="Ship Mode",
                 category_orders={"Ship Mode": order}, color_discrete_sequence=PALETTE, points=False)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420, showlegend=False, yaxis_title="Modeled Lead Time (days)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Delay Frequency by Ship Mode")
        sm = f.groupby("Ship Mode").agg(Delay_Rate=("Delayed_Now", "mean"), Shipments=("Row ID", "count")).reindex(order).reset_index()
        fig = px.bar(sm, x="Ship Mode", y="Delay_Rate", color="Ship Mode", color_discrete_sequence=PALETTE, text_auto=".1%")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360, showlegend=False, yaxis_tickformat=".0%", yaxis_title="Delay Frequency")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Cost-Time Tradeoff")
        sm2 = f.groupby("Ship Mode").agg(Avg_Lead_Time=("Modeled_Lead_Time","mean"),
                                          Avg_Sales_Per_Shipment=("Sales","mean"), Shipments=("Row ID","count")).reindex(order).reset_index()
        fig = px.scatter(sm2, x="Avg_Lead_Time", y="Avg_Sales_Per_Shipment", size="Shipments",
                          color="Ship Mode", text="Ship Mode", size_max=55, color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="top center")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360, xaxis_title="Avg Lead Time (days)", yaxis_title="Avg Sales / Shipment ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ship Mode Summary Table")
    summary = f.groupby("Ship Mode").agg(
        Shipments=("Row ID","count"), Avg_Lead_Time=("Modeled_Lead_Time","mean"),
        Delay_Rate=("Delayed_Now","mean"), Total_Sales=("Sales","sum"), Total_Profit=("Gross Profit","sum")
    ).reindex(order).reset_index()
    summary["Avg_Lead_Time"] = summary["Avg_Lead_Time"].round(2)
    summary["Delay_Rate"] = (summary["Delay_Rate"]*100).round(1).astype(str)+"%"
    summary["Total_Sales"] = summary["Total_Sales"].map(lambda v: f"${v:,.0f}")
    summary["Total_Profit"] = summary["Total_Profit"].map(lambda v: f"${v:,.0f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4 — ROUTE DRILL-DOWN
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("State-Level Performance Insight")
    c1, c2 = st.columns([1, 2])
    with c1:
        state_pick = st.selectbox("Select a state / province", sorted(f["State/Province"].unique()))
    sdf = f[f["State/Province"] == state_pick]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shipments", f"{len(sdf):,}")
    m2.metric("Avg. Lead Time", f"{sdf['Modeled_Lead_Time'].mean():.2f} days")
    m3.metric("Delay Frequency", f"{sdf['Delayed_Now'].mean()*100:.1f}%")
    m4.metric("Total Sales", f"${sdf['Sales'].sum():,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Shipments by Factory**")
        fac = sdf.groupby("Factory").size().reset_index(name="Shipments")
        fig = px.pie(fac, names="Factory", values="Shipments", color_discrete_sequence=PALETTE, hole=0.45)
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Shipments by Ship Mode**")
        sm = sdf.groupby("Ship Mode").size().reset_index(name="Shipments")
        fig = px.pie(sm, names="Ship Mode", values="Shipments", color_discrete_sequence=PALETTE, hole=0.45)
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Order-Level Shipment Timeline")
    tl = sdf.sort_values("Order Date")[["Order Date", "Order ID", "Product Name", "Ship Mode", "Factory", "Modeled_Lead_Time", "Sales", "Delayed_Now"]].copy()
    tl["Modeled_Lead_Time"] = tl["Modeled_Lead_Time"].round(2)
    tl["Sales"] = tl["Sales"].map(lambda v: f"${v:,.2f}")
    tl = tl.rename(columns={"Modeled_Lead_Time": "Lead Time (days)", "Delayed_Now": "Delayed?"})
    st.dataframe(tl, use_container_width=True, height=380, hide_index=True)

    fig = px.scatter(sdf.sort_values("Order Date"), x="Order Date", y="Modeled_Lead_Time", color="Ship Mode",
                      color_discrete_sequence=PALETTE, opacity=0.7)
    fig.add_hline(y=lead_threshold, line_dash="dash", line_color=RED, annotation_text="Delay threshold")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360, yaxis_title="Lead Time (days)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Nassau Candy Distributor — Factory-to-Customer Shipping Route Efficiency Dashboard · Built with Streamlit · Lead-time figures are modeled (see sidebar note)")
