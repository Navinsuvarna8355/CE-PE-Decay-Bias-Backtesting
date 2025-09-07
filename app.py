import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Helper Functions ---

def generate_trading_days(num_days, seed=42):
    """Simulate recent trading days with random walk index prices."""
    np.random.seed(seed)
    start_price = 22000  # Start BankNifty-like
    drift = np.random.normal(0, 50, num_days)
    index_prices = start_price + np.cumsum(drift)
    date_range = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq='B')
    return pd.DataFrame({'date': date_range, 'index_price': index_prices})

def simulate_option_decay(
    day_index,
    index_price,
    bias, 
    steps=13,     # e.g. 10am-3:30pm, every 30 mins
    ce_premium_0=100,
    pe_premium_0=100,
    ce_theta=0.23,
    pe_theta=0.23,
    volatility=3.0,
    seed=None
):
    """Creates time series for ATM CE and PE premium decay for one day."""
    if seed is not None:
        np.random.seed(seed + day_index)
    time_grid = np.linspace(0, 1, steps)
    # Bias logic: Decay rates shift depending on bias
    ce_decay = ce_theta
    pe_decay = pe_theta
    if bias == "PE Decay (Bullish)":
        pe_decay *= 1.25   # accelerate PE decay
        ce_decay *= 0.80   # slow CE decay
    elif bias == "CE Decay (Bearish)":
        ce_decay *= 1.25
        pe_decay *= 0.80
    # Exponential decay with noise
    ce_series = (
        ce_premium_0 * np.exp(-ce_decay * time_grid)
        + np.random.normal(0, volatility, steps)
    ).clip(min=2)
    pe_series = (
        pe_premium_0 * np.exp(-pe_decay * time_grid)
        + np.random.normal(0, volatility, steps)
    ).clip(min=2)
    # Compose DataFrame
    df = pd.DataFrame({
        'time': np.linspace(9.5, 15.5, steps), # 9:30 to 15:30
        'ce': ce_series,
        'pe': pe_series
    })
    return df

def backtest_strategy(
    all_days_df,
    side,            # "Sell CE", "Sell PE", or "Sell Both"
    bias,
    stop_loss,
    profit_target,
    entry_hour=9.5,
    exit_hour=15.25
):
    """
    Simulates the decay bias strategy over all trading days.
    Returns detailed trade log and metrics.
    """
    trade_log = []
    equity_curve = []
    pnl_cum = 0
    drawdown = 0

    for idx, row in all_days_df.iterrows():
        # Each day: generate fresh decay curves with bias
        ce0 = np.random.randint(80, 140)
        pe0 = np.random.randint(80, 140)
        day_decay = simulate_option_decay(
            idx, row['index_price'], bias, 
            ce_premium_0=ce0, pe_premium_0=pe0, 
            seed=1000+idx
        )
        entry_ce = day_decay.iloc[0]['ce']
        entry_pe = day_decay.iloc[0]['pe']

        exit_ce = entry_ce
        exit_pe = entry_pe
        trade_exit_time = exit_hour
        exit_reason = "EOD"

        # Position: Each leg is 1 lot. P&L for shorts = Entry - Exit.
        pnl = 0
        max_mtm_loss = 0
        # Track P&L at each time for stop out
        for i, step in day_decay.iterrows():
            ce_mtm = entry_ce - step['ce']
            pe_mtm = entry_pe - step['pe']

            # Determine open position and P&L
            leg_pnl = []
            if side == "Sell CE":
                leg_pnl.append(ce_mtm)
            elif side == "Sell PE":
                leg_pnl.append(pe_mtm)
            elif side == "Sell Both":
                leg_pnl.append(ce_mtm + pe_mtm)
            mtm = sum(leg_pnl)
            max_mtm_loss = min(max_mtm_loss, mtm)

            # Stop loss or profit?
            if stop_loss and mtm <= -abs(stop_loss):
                trade_exit_time = step['time']
                exit_ce = step['ce']
                exit_pe = step['pe']
                exit_reason = "Stop Loss"
                pnl = mtm
                break
            if profit_target and mtm >= abs(profit_target):
                trade_exit_time = step['time']
                exit_ce = step['ce']
                exit_pe = step['pe']
                exit_reason = "Target"
                pnl = mtm
                break
        else:
            # Not exited: Mark EOD
            if side == "Sell CE":
                pnl = entry_ce - day_decay.iloc[-1]['ce']
                exit_ce = day_decay.iloc[-1]['ce']
                exit_pe = entry_pe
            elif side == "Sell PE":
                pnl = entry_pe - day_decay.iloc[-1]['pe']
                exit_pe = day_decay.iloc[-1]['pe']
                exit_ce = entry_ce
            elif side == "Sell Both":
                pnl = (entry_ce - day_decay.iloc[-1]['ce']) + \
                      (entry_pe - day_decay.iloc[-1]['pe'])
                exit_ce = day_decay.iloc[-1]['ce']
                exit_pe = day_decay.iloc[-1]['pe']

        pnl_cum += pnl
        equity_curve.append(pnl_cum)

        trade_log.append({
            'date': row['date'],
            'bias': bias,
            'side': side,
            'open_idx': row['index_price'],
            'entry_CE': round(entry_ce,2) if side in ["Sell CE", "Sell Both"] else "-",
            'entry_PE': round(entry_pe,2) if side in ["Sell PE", "Sell Both"] else "-",
            'exit_CE': round(exit_ce,2) if side in ["Sell CE", "Sell Both"] else "-",
            'exit_PE': round(exit_pe,2) if side in ["Sell PE", "Sell Both"] else "-",
            'exit_time': f"{trade_exit_time:.2f}",
            'exit_reason': exit_reason,
            'PnL': round(pnl,2),
            'max_mtm_loss': round(max_mtm_loss,2)
        })
    # Build DataFrames
    trade_log_df = pd.DataFrame(trade_log)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    equity = pd.Series(equity_curve, index=all_days_df['date'])
    # Compute Drawdown
    rolling_max = equity.cummax()
    dds = equity - rolling_max
    max_dd = dds.min()
    return trade_log_df, equity, max_dd

def compute_performance_metrics(trade_log_df, equity, max_dd):
    wins = trade_log_df[trade_log_df['PnL'] > 0].shape[0]
    losses = trade_log_df[trade_log_df['PnL'] <= 0].shape[0]
    profit_factor = trade_log_df[trade_log_df['PnL'] > 0]['PnL'].sum() / abs(max(1, trade_log_df[trade_log_df['PnL'] < 0]['PnL'].sum()))
    total_pnl = trade_log_df['PnL'].sum()
    avg_pnl = trade_log_df['PnL'].mean()
    winrate = wins / (wins + losses) * 100 if wins + losses > 0 else 0
    volatility = trade_log_df['PnL'].std()
    avg_decay = 0
    # Decay: For each sell, how much did premium fall from entry to exit?
    ce_mask = trade_log_df['side'].isin(['Sell CE', 'Sell Both'])
    pe_mask = trade_log_df['side'].isin(['Sell PE', 'Sell Both'])
    if ce_mask.any():
        avg_decay_CE = (trade_log_df.loc[ce_mask, 'entry_CE'].astype(float) - trade_log_df.loc[ce_mask, 'exit_CE'].astype(float)).mean()
    else:
        avg_decay_CE = 0
    if pe_mask.any():
        avg_decay_PE = (trade_log_df.loc[pe_mask, 'entry_PE'].astype(float) - trade_log_df.loc[pe_mask, 'exit_PE'].astype(float)).mean()
    else:
        avg_decay_PE = 0
    metrics = {
        'Total Net P&L': round(total_pnl,2),
        'Win Rate (%)': round(winrate,2),
        'Average Daily P&L': round(avg_pnl,2),
        'Maximum Drawdown': round(max_dd,2),
        'Profit Factor': round(profit_factor,2),
        'Volatility of Returns': round(volatility,2),
        'Avg CE Decay': round(avg_decay_CE,2),
        'Avg PE Decay': round(avg_decay_PE,2),
        'Total Trades': len(trade_log_df)
    }
    return metrics

def plot_equity_curve(equity, title="Equity Curve"):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, mode='lines', line=dict(color='limegreen'), name='Equity')
    )
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Equity',
        showlegend=False,
        template='plotly_white'
    )
    return fig

def plot_sample_decay(day_decay, side, day_label):
    fig = go.Figure()
    # Option decay lineplots
    if side in ["Sell CE", "Sell Both"]:
        fig.add_trace(go.Scatter(x=day_decay['time'], y=day_decay['ce'],
                                 mode='lines+markers', name='CE Premium', line=dict(color="dodgerblue")))
    if side in ["Sell PE", "Sell Both"]:
        fig.add_trace(go.Scatter(x=day_decay['time'], y=day_decay['pe'],
                                 mode='lines+markers', name='PE Premium', line=dict(color="orange")))
    fig.update_layout(
        title=f"Intraday Option Premium Decay ({day_label})",
        xaxis_title='Time (Hours, 24h)',
        yaxis_title='Premium',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        template='plotly_white'
    )
    return fig

# --- Streamlit App Code ---

st.set_page_config(page_title="CE/PE Decay Bias Strategy Backtest", layout="wide")

# --- Sidebar Controls ---
st.sidebar.title("Strategy Parameters")
num_days = st.sidebar.slider("Number of Trading Days", 20, 80, 40, step=5)
strategy_side = st.sidebar.selectbox("Sell Leg", ["Sell CE", "Sell PE", "Sell Both"], index=2)
bias_setting = st.sidebar.selectbox(
    "Decay Bias (sets the expected premium decay pattern for mock data)", 
    ["Neutral", "PE Decay (Bullish)", "CE Decay (Bearish)"]
)
stop_loss = st.sidebar.number_input("Per Trade Stop Loss (0=disabled)", min_value=0, value=40, step=5)
profit_target = st.sidebar.number_input("Per Trade Profit Target (0=disabled)", min_value=0, value=40, step=5)

with st.sidebar.expander("Help: Strategy Info"):
    st.markdown("""
**Decay bias strategies** use observed patterns in **option premium decay rates** for Call (CE) and Put (PE) options.  
- In **bullish or sideways markets**, PEs tend to decay faster.
- In **bearish/trending down markets**, CEs decay faster.
- On expiry days, decay can be rapid for both.  
Change the "Decay Bias" option to simulate different environments.
    """)

# --- Main Panel ---

st.title("CE/PE Decay Bias Option Selling Strategy • Backtest Simulator")

st.markdown("""
This app simulates a **CE/PE Decay Bias intraday option selling strategy**.  
It uses **randomized, realistic option premium decay paths** to backtest the strategy over recent past trading days.  
Adjust the sidebar to set the number of days, which legs to sell (CE, PE, or both),  
and simulate various premium decay environments.
""")

# --- Simulate Data ---
days_df = generate_trading_days(num_days)
trade_log_df, equity_curve, max_dd = backtest_strategy(
    days_df, strategy_side, bias_setting, stop_loss, profit_target
)
metrics = compute_performance_metrics(trade_log_df, equity_curve, max_dd)
mtitle = f"Backtest Results: {strategy_side} | Bias = {bias_setting} | Days = {num_days}"

# --- Layout: Two Columns ---
col1, col2 = st.columns(2)
with col1:
    st.header("Key Metrics")
    metrics_table = pd.DataFrame.from_dict(metrics, orient='index', columns=["Value"])
    st.table(metrics_table)
    st.header("Equity Curve")
    equity_fig = plot_equity_curve(equity_curve)
    st.plotly_chart(equity_fig, use_container_width=True)

with col2:
    st.header(f"Sample Option Premium Decay ({strategy_side})")
    sample_idx = np.random.randint(0, num_days)
    sample_price = days_df.iloc[sample_idx]['index_price']
    day_decay = simulate_option_decay(
        sample_idx, sample_price, bias_setting, seed=1000+sample_idx
    )
    day_label = days_df.iloc[sample_idx]['date'].strftime('%Y-%m-%d')
    decay_fig = plot_sample_decay(day_decay, strategy_side, day_label)
    st.plotly_chart(decay_fig, use_container_width=True)

with st.expander("Show Detailed Trade Log and Returns Distribution"):
    st.subheader("Trade-by-Trade Log")
    st.dataframe(
        trade_log_df.style.background_gradient(cmap='RdYlGn', subset='PnL'),
        use_container_width=True
    )
    st.subheader("Distribution of Daily Returns")
    histfig = go.Figure()
    histfig.add_trace(go.Histogram(
        x=trade_log_df['PnL'],
        nbinsx=20,
        marker_color='slateblue',
        opacity=0.7
    ))
    histfig.update_layout(
        title="Histogram: Per-Trade P&L",
        xaxis_title='P&L',
        yaxis_title='Frequency',
        template='plotly_white'
    )
    st.plotly_chart(histfig, use_container_width=True)

with st.expander("About this App / References"):
    st.markdown("""
**How It Works:**  
- **Simulated Data:** Premiums and decay behaviors mimic expiry-day option writing environments.
- **Risk Management:** Trades autoclosed at day-end or when stop/target is hit.
- **Metrics & Visualization:** Interactive charts and tables help assess strategy viability.

**Tech Stack:**  
- Streamlit for UI ([docs](https://docs.streamlit.io/get-started/tutorials/create-an-app))
- Numpy/Pandas for vectorized sim & backtest engine
- Plotly for charts

**To Deploy Remotely:**  
- Upload this file to a public GitHub repository.
- Add a requirements.txt:
    ```
    streamlit
    numpy
    pandas
    plotly
    ```
- Deploy using [Streamlit Community Cloud](https://share.streamlit.io/).

**References & Community:**
- Streamlit backtesting/templates: [VectorBT-Streamlit](https://github.com/marketcalls/VectorBT-Streamlit), [zhong-us/backtesting](https://github.com/zhong-us/backtesting)
- Options theory/decay insights: [TradingView CE/PE](https://in.tradingview.com/chart/NIFTY/zodMJsjT-PE-Writing-vs-CE-Writing-Core-Difference-Explained/)
    """)

st.caption("© 2025 | CE/PE Decay Bias Backtesting Demo | Built with Streamlit, Numpy, Pandas, Plotly")

