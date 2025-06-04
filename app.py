import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
# ─── 0) Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Colchester Crime & Wellbeing Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",   # <-- was "expanded"
)

# ─── 1) Load & harmonise data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    crime = pd.read_csv("df.csv", parse_dates=["Month"])
    crime["Quarter_str"] = crime["Month"].dt.to_period("Q").astype(str)
    crime["Quarter_dt"]  = crime["Month"].dt.to_period("Q").dt.to_timestamp()

    combined = pd.read_csv("combined_df.csv")
    combined["Quarter_str"] = combined["Quarter"].astype(str)
    combined["Quarter_dt"]  = pd.to_datetime(dict(
        year=combined["Year"].astype(int),
        month=(combined["QtrNum"].astype(int)-1)*3+1,
        day=1
    ))
    return crime, combined

crime_df, combined_df = load_data()

# ─── 2) Sidebar filters ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Filters")

    types = crime_df["Crime type"].unique().tolist()
    sel_types = st.multiselect(
        "Crime types",
        options=types,
        default=[]            # <-- default to nothing selected
    )
    if not sel_types:
        sel_types = types   # fallback to all if user clears everything

    quarters = combined_df["Quarter_str"].tolist()
    sel_qtrs = st.multiselect(
        "Quarters",
        options=quarters,
        default=[]           # <-- default to nothing selected
    )
    if not sel_qtrs:
        sel_qtrs = quarters # fallback to all if user clears everything

# ─── 3) Build dash_df ─────────────────────────────────────────────────────────
f_crime = crime_df[crime_df["Crime type"].isin(sel_types)]
counts = (
    f_crime
    .groupby("Quarter_str")
    .size()
    .reset_index(name="Crime Count")
)

dash_df = (
    combined_df
    .merge(counts, on="Quarter_str", how="left")
    .fillna({"Crime Count": 0})
    .assign(**{"Crime Rate per 1ₖ": lambda df: df["Crimes per 1000"]})
    .query("Quarter_str in @sel_qtrs")
    .sort_values("Quarter_dt")
)

# ─── 4) Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Overview",
    "📉 Trends",
    "🗺️ Map",
    "🔮 Forecast"
])

#tab 1
# inside tabs[0]: your existing Overview code…

with tabs[0]:
    st.title("Colchester Crime, Wellbeing & Housing Dashboard")
    st.markdown(
        "Overall crime has averaged "
        f"**{dash_df['Crime Rate per 1ₖ'].mean():.1f} per 1ₖ** over the last "
        f"{dash_df.shape[0]} quarters, peaking in mid-2022. "
        "Anxiety and low-life satisfaction have crept upward alongside "
        "minor dips in crime since 2021."
    )
    st.markdown("---")

    # compute your KPIs
    avg_crime = dash_df["Crime Rate per 1ₖ"].mean()
    avg_anx   = dash_df["High Anxiety %"].mean()
    avg_hap   = dash_df["Low Happiness %"].mean()
    avg_life  = dash_df["Low Life Satisfaction %"].mean()
    avg_price = dash_df["House Price Change (%)"].mean()    # ← new

    # now render them in five columns instead of four
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Crimes per 1ₖ", f"{avg_crime:.1f}")
    c2.metric("Avg High Anxiety %", f"{avg_anx:.1f}%")
    c3.metric("Avg Low Happiness %", f"{avg_hap:.1f}%")
    c4.metric("Avg Low Life Satisfaction %", f"{avg_life:.1f}%")
    c5.metric("Avg House Price Change %", f"{avg_price:.1f}%")  # ← new


    # 4) Row: Line chart + Donut chart
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### Crime Rate Trend")
        fig_line = px.line(
            dash_df,
            x="Quarter_str",
            y="Crime Rate per 1ₖ",
            markers=True,
            labels={"Crime Rate per 1ₖ":"Crimes per 1ₖ","Quarter_str":"Quarter"},
            template="plotly_white",
        )
        fig_line.update_layout(margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.markdown("#### Top 5 Crime Types")
        top5 = crime_df["Crime type"].value_counts().nlargest(5)
        pie_df = pd.DataFrame({
            "Crime type": top5.index,
            "Count":      top5.values
        })
        fig_donut = px.pie(
            pie_df,
            names="Crime type",
            values="Count",
            hole=0.4,
            color="Crime type",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont_size=14,
            marker=dict(line=dict(color="white", width=2)),
            pull=[0.1] + [0] * (len(pie_df) - 1)
        )
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=30, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)



# --- Tab 2: Trends (Two‐by‐Two Layout + Dark→Less‐Light Blue & Red) ---
with tabs[1]:
    st.subheader("Crime, Wellbeing & Housing Trends")

    # Row 1: Time Series
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Crime & Wellbeing Over Time")
        fig1 = px.line(
            dash_df,
            x="Quarter_str",
            y=["Crime Rate per 1ₖ", "High Anxiety %", "Low Life Satisfaction %"],
            markers=True,
            labels={"value":"Value","variable":"Indicator","Quarter_str":"Quarter"},
            template="plotly_white",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        st.markdown("#### Crime Rate vs Lagged House Price Change")
        dfc = combined_df.rename(columns={
            "Crimes per 1000":              "Crime Rate per 1ₖ",
            "Lagged House Price Change (%)": "Lagged Price Change"
        })
        ts2 = (
            dfc[["Quarter_str","Quarter_dt","Crime Rate per 1ₖ","Lagged Price Change"]]
              .query("Quarter_str in @sel_qtrs")
              .sort_values("Quarter_dt")
        )
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(
            go.Scatter(x=ts2["Quarter_str"],
                       y=ts2["Crime Rate per 1ₖ"],
                       name="Crime Rate per 1ₖ",
                       mode="lines+markers"),
            secondary_y=False,
        )
        fig2.add_trace(
            go.Scatter(x=ts2["Quarter_str"],
                       y=ts2["Lagged Price Change"],
                       name="Lagged House Price Change (%)",
                       mode="lines+markers"),
            secondary_y=True,
        )
        fig2.update_layout(template="plotly_white",
                           margin=dict(t=40, r=20, l=20, b=20))
        fig2.update_xaxes(title_text="Quarter")
        fig2.update_yaxes(title_text="Crimes per 1ₖ", secondary_y=False)
        fig2.update_yaxes(title_text="Lagged Price Change (%)", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Row 2: Bar Charts with darker discrete palettes
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Frequency of Crime Types (Top 10)")
        top10 = crime_df["Crime type"].value_counts().nlargest(10)
        top10_df = pd.DataFrame({
            "Crime type": top10.index,
            "Count":      top10.values
        })
        # reverse Blues, then drop the last two (lightest)
        blues = px.colors.sequential.Blues[::-1][:-2]
        fig3 = px.bar(
            top10_df,
            x="Count",
            y="Crime type",
            orientation="h",
            color="Crime type",
            color_discrete_sequence=blues,
            labels={"Count":"Number of Crimes","Crime type":""},
            template="plotly_white",
        )
        fig3.update_layout(
            yaxis={"categoryorder":"total ascending"},
            showlegend=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("#### Crime Outcome Categories")
        oc = crime_df["Last outcome category"].value_counts()
        outcome_df = pd.DataFrame({"Outcome": oc.index, "Count": oc.values})
        # reverse Reds, then drop the last two (lightest)
        reds = px.colors.sequential.Reds[::-1][:-2]
        fig4 = px.bar(
            outcome_df,
            x="Count",
            y="Outcome",
            orientation="h",
            color="Outcome",
            color_discrete_sequence=reds,
            labels={"Count":"Number of Crimes","Outcome":""},
            template="plotly_white",
        )
        fig4.update_layout(
            yaxis={"categoryorder":"total ascending"},
            showlegend=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig4, use_container_width=True)



# --- Tab 3: Map & Location Hotspots ---
with tabs[2]:
    st.subheader("Incident Map")
    # 3.1 All incidents (filtered by crime type)
    if {"Latitude", "Longitude"}.issubset(f_crime.columns):
        df_map = f_crime.dropna(subset=["Latitude", "Longitude"])
        fig_inc = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            hover_name="Crime type",
            zoom=11,
            height=400,
            mapbox_style="open-street-map",
            title="All Reported Incidents"
        )
        fig_inc.update_layout(margin=dict(r=0, t=30, l=0, b=0))
        st.plotly_chart(fig_inc, use_container_width=True)
    else:
        st.info("No geographic data available for incidents.")


    # 3.3 Map of Top 10 Hotspots (mean lat/lon)
    st.subheader("Map of Top 10 Crime Hotspots")
    loc_geo = (
        crime_df
        .dropna(subset=["Latitude","Longitude"])
        .groupby("Location")
        .agg(
            Count=("Location","size"),
            Latitude=("Latitude","mean"),
            Longitude=("Longitude","mean")
        )
        .reset_index()
    )
    top10_geo = loc_geo.nlargest(10, "Count")
    fig_loc_map = px.scatter_mapbox(
        top10_geo,
        lat="Latitude",
        lon="Longitude",
        size="Count",
        hover_name="Location",
        hover_data={"Count":True},
        zoom=12,
        height=400,
        mapbox_style="open-street-map",
        title="Average Position of Top 10 Crime Locations"
    )
    fig_loc_map.update_layout(margin=dict(r=0, t=30, l=0, b=0))
    st.plotly_chart(fig_loc_map, use_container_width=True)

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# --- Tab 3: Forecast ---
# ─── Forecast tab (tabs[3]) ───────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🔮 4-Quarter Crime Rate Forecast")
    st.markdown("### Crime Rate Forecast in Colchester")

    # build the time series exactly as in your analysis
    crime_ts = combined_df[["Quarter_str", "Crimes per 1000"]].copy()
    crime_ts["Quarter_dt"] = pd.to_datetime(
        crime_ts["Quarter_str"]
        .str.replace("Q1 ", "01-")
        .str.replace("Q2 ", "04-")
        .str.replace("Q3 ", "07-")
        .str.replace("Q4 ", "10-"),
        format="%m-%Y",
    )
    crime_ts = crime_ts.set_index("Quarter_dt").sort_index()["Crimes per 1000"]

    # fit ARIMA(1,1,1)
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(crime_ts, order=(1, 1, 1))
    model_fit = model.fit()

    # forecast next 4 quarters
    forecast_steps = 4
    forecast = model_fit.forecast(steps=forecast_steps)

    # build future-dates index
    last = crime_ts.index[-1]
    future_idx = pd.date_range(
        start=last + pd.offsets.QuarterBegin(), periods=forecast_steps, freq="Q"
    )

    # now plot with matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(crime_ts.index, crime_ts.values, "-o", label="Observed")
    ax.plot(future_idx, forecast.values, "-s", color="crimson", label="Forecast")
    ax.set_title("Crime Rate Forecast in Colchester")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Crimes per 1000 Residents")
    ax.legend()
    ax.grid(True)

    # hand it off to Streamlit
    st.pyplot(fig)
