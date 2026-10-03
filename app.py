
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="Financial Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Analytics Dashboard")
st.subheader("Reliance Industries & TCS")
st.write(
    "Interactive analysis of market prices, returns, risk and portfolio allocation."
)

# -----------------------------
# DOWNLOAD MARKET DATA
# -----------------------------

@st.cache_data
def load_data():

    reliance = yf.download(
        "RELIANCE.NS",
        start="2021-01-01",
        end="2026-10-01",
        auto_adjust=False
    )

    tcs = yf.download(
        "TCS.NS",
        start="2021-01-01",
        end="2026-10-01",
        auto_adjust=False
    )

    reliance.columns = reliance.columns.get_level_values(0)
    tcs.columns = tcs.columns.get_level_values(0)

    reliance["Return"] = reliance["Adj Close"].pct_change()
    tcs["Return"] = tcs["Adj Close"].pct_change()

    return reliance, tcs


reliance, tcs = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("⚙️ Dashboard Controls")

reliance_weight = st.sidebar.slider(
    "Reliance Weight (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=5
)

tcs_weight = 100 - reliance_weight

ma_short = st.sidebar.slider(
    "Short Moving Average",
    min_value=5,
    max_value=50,
    value=20,
    step=5
)

ma_long = st.sidebar.slider(
    "Long Moving Average",
    min_value=20,
    max_value=200,
    value=50,
    step=10
)

# -----------------------------
# RETURNS
# -----------------------------

returns = pd.DataFrame({
    "Reliance": reliance["Return"],
    "TCS": tcs["Return"]
}).dropna()

mean_daily_return = returns.mean()
daily_volatility = returns.std()

annual_return = mean_daily_return * 252
annual_volatility = daily_volatility * np.sqrt(252)

cov_matrix = returns.cov() * 252

# -----------------------------
# PORTFOLIO CALCULATION
# -----------------------------

w1 = reliance_weight / 100
w2 = tcs_weight / 100

weights = np.array([w1, w2])

portfolio_return = np.sum(
    weights * annual_return
)

portfolio_variance = np.dot(
    weights.T,
    np.dot(cov_matrix, weights)
)

portfolio_risk = np.sqrt(portfolio_variance)

portfolio_sharpe = (
    portfolio_return / portfolio_risk
)

# -----------------------------
# DASHBOARD METRICS
# -----------------------------

st.markdown("### 📌 Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Reliance Weight",
    f"{reliance_weight}%"
)

col2.metric(
    "TCS Weight",
    f"{tcs_weight}%"
)

col3.metric(
    "Expected Annual Return",
    f"{portfolio_return * 100:.2f}%"
)

col4.metric(
    "Portfolio Risk",
    f"{portfolio_risk * 100:.2f}%"
)

st.divider()

# -----------------------------
# PRICE CHART
# -----------------------------

st.subheader("📈 Market Price Analysis")

company = st.selectbox(
    "Select Company",
    ["Reliance Industries", "TCS"]
)

if company == "Reliance Industries":

    data = reliance.copy()
    ticker_name = "Reliance"

else:

    data = tcs.copy()
    ticker_name = "TCS"

data["MA Short"] = data["Close"].rolling(
    ma_short
).mean()

data["MA Long"] = data["Close"].rolling(
    ma_long
).mean()

chart_data = data[
    ["Close", "MA Short", "MA Long"]
].dropna()

fig = px.line(
    chart_data,
    x=chart_data.index,
    y=["Close", "MA Short", "MA Long"],
    title=f"{ticker_name} Price & Moving Averages"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# PORTFOLIO SECTION
# -----------------------------

st.divider()

st.subheader("💼 Portfolio Analysis")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Expected Return",
    f"{portfolio_return * 100:.2f}%"
)

col2.metric(
    "Portfolio Risk",
    f"{portfolio_risk * 100:.2f}%"
)

col3.metric(
    "Sharpe Ratio",
    f"{portfolio_sharpe:.3f}"
)

# -----------------------------
# PORTFOLIO ALLOCATION
# -----------------------------

allocation = pd.DataFrame({
    "Company": ["Reliance", "TCS"],
    "Weight": [reliance_weight, tcs_weight]
})

fig_pie = px.pie(
    allocation,
    names="Company",
    values="Weight",
    title="Portfolio Allocation"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

# -----------------------------
# RISK-RETURN PORTFOLIO GRID
# -----------------------------

portfolio_results = []

for reliance_w in np.arange(0, 1.01, 0.05):

    tcs_w = 1 - reliance_w

    w = np.array([
        reliance_w,
        tcs_w
    ])

    p_return = np.sum(
        w * annual_return
    )

    p_variance = np.dot(
        w.T,
        np.dot(cov_matrix, w)
    )

    p_risk = np.sqrt(p_variance)

    portfolio_results.append({
        "Reliance Weight": reliance_w * 100,
        "TCS Weight": tcs_w * 100,
        "Return": p_return * 100,
        "Risk": p_risk * 100
    })

portfolio_df = pd.DataFrame(
    portfolio_results
)

fig_risk = px.scatter(
    portfolio_df,
    x="Risk",
    y="Return",
    size="Reliance Weight",
    color="Reliance Weight",
    hover_data=["TCS Weight"],
    title="Markowitz Risk–Return Analysis"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

st.caption(
    "Portfolio calculations use historical daily returns and annualization based on 252 trading days."
)
