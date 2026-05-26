import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #f0f2f6; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

section[data-testid="stSidebar"] {
    background-color: #1e2a3a !important;
    overflow: hidden !important;
    height: 100vh !important;
}
section[data-testid="stSidebar"] .block-container {
    overflow: hidden !important;
    padding: 1.2rem !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #90caf9 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2d3f55 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #2d3f55 !important;
    color: #e0e0e0 !important;
}

.main-header {
    background: linear-gradient(135deg, #1565c0, #0288d1, #00838f);
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0;
}
.main-header p {
    color: #e3f2fd;
    font-size: 1rem;
    margin-top: 0.5rem;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #dde3ed;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: bold;
    color: #1565c0;
}
.metric-label {
    font-size: 0.82rem;
    color: #555;
    margin-top: 0.2rem;
}
.metric-delta {
    font-size: 0.78rem;
    color: #2e7d32;
    margin-top: 0.2rem;
    font-weight: 500;
}

.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1565c0;
    padding: 0.4rem 0;
    border-bottom: 2px solid #bbdefb;
    margin-bottom: 1rem;
}

.winner-badge {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
    border: 1px solid #43a047;
    color: #1b5e20;
    font-weight: bold;
    font-size: 0.95rem;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.insight-card {
    background: white;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border-left: 4px solid #1565c0;
    margin-bottom: 0.8rem;
    color: #222;
    font-size: 0.88rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    line-height: 1.5;
}

.prediction-result {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    border: 2px solid #1565c0;
    margin-top: 1.2rem;
    box-shadow: 0 4px 15px rgba(21,101,192,0.15);
}
.prediction-number {
    font-size: 3.2rem;
    font-weight: 800;
    color: #1565c0;
}
.prediction-sublabel {
    font-size: 0.95rem;
    color: #1976d2;
    margin-top: 0.3rem;
    font-weight: 500;
}
.prediction-range {
    font-size: 0.85rem;
    color: #444;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('app/models/lgb_model.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    with open('app/models/prophet_model.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    return lgb_model, prophet_model

@st.cache_data
def load_data():
    with open('app/models/test_data.pkl', 'rb') as f:
        return pickle.load(f)

lgb_model, prophet_model = load_models()
data = load_data()
test        = data['test']
y_test      = data['y_test']
lgb_preds   = data['lgb_predictions']
prop_preds  = data['prophet_predictions']
train       = data['train']
feature_cols = data['feature_cols']
full_df = pd.concat([train, test]).sort_values('date').reset_index(drop=True)

# ── Chart helper ─────────────────────────────────────────────────
def base_layout(title='', height=350, xtitle='', ytitle=''):
    return dict(
        template=None,
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        height=height,
        margin=dict(l=60, r=30, t=50, b=60),
        title=dict(text=title, font=dict(color='#1565c0', size=14),
                   x=0, xanchor='left'),
        font=dict(color='#111111', size=11),
        xaxis=dict(
            title=dict(text=xtitle, font=dict(color='#111111', size=12)),
            tickfont=dict(color='#111111', size=10),
            gridcolor='#e8e8e8', linecolor='#aaaaaa',
            linewidth=1, showgrid=True, zeroline=False,
            ticks='outside', tickcolor='#555555'
        ),
        yaxis=dict(
            title=dict(text=ytitle, font=dict(color='#111111', size=12)),
            tickfont=dict(color='#111111', size=10),
            gridcolor='#e8e8e8', linecolor='#aaaaaa',
            linewidth=1, showgrid=True, zeroline=False,
            ticks='outside', tickcolor='#555555'
        ),
        legend=dict(
            bgcolor='white', bordercolor='#cccccc',
            borderwidth=1, font=dict(color='#111111', size=10)
        ),
        modebar_remove=['resetaxes', 'autoscale', 'lasso2d',
                        'select2d', 'zoom2d', 'pan2d']
    )

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Retail Demand Forecasting</h1>
    <p>Walmart CA1 Store &mdash; FOODS Category &nbsp;|&nbsp;
       M5 Competition Dataset &nbsp;|&nbsp; 2011&ndash;2016</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dashboard Controls")
    st.markdown("---")
    st.markdown("#### Select Model")
    selected_model = st.selectbox(
        "Forecasting Model",
        ["LightGBM", "Prophet"],
        help="LightGBM uses 19 engineered features. Prophet uses date and sales only."
    )
    st.markdown("---")
    st.markdown("#### Dataset Info")
    st.markdown("""
**Training:** Feb 2011 — Dec 2015  
**Test:** Jan 2016 — Apr 2016  
**Product:** FOODS\\_3\\_090  
**Store:** CA1, California  
**Total Rows:** 2.7 million  
    """)
    st.markdown("---")
    st.markdown("#### Model Leaderboard")
    st.markdown("""
| Model | MAE |
|-------|-----|
| LightGBM | 15.66 |
| Prophet | 22.43 |
| SARIMA | 56.48 |
    """)
    st.markdown("---")
    st.markdown("#### Built With")
    st.markdown("""
Python · Pandas · NumPy  
LightGBM · Prophet  
Plotly · Streamlit  
    """)

# ── Active predictions ───────────────────────────────────────────
predictions = lgb_preds if selected_model == "LightGBM" else prop_preds

mae_val  = mean_absolute_error(y_test, predictions)
rmse_val = np.sqrt(mean_squared_error(y_test, predictions))
mask     = y_test != 0
mape_val = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
lgb_mae  = mean_absolute_error(y_test, lgb_preds)
prop_mae = mean_absolute_error(y_test, prop_preds)

# ── Section 1 — KPI Cards ────────────────────────────────────────
st.markdown('<p class="section-header">Model Performance Metrics</p>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{round(mae_val,2)}</div>
        <div class="metric-label">MAE (units / day)</div>
        <div class="metric-delta">Mean Absolute Error</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{round(rmse_val,2)}</div>
        <div class="metric-label">RMSE (units / day)</div>
        <div class="metric-delta">Root Mean Squared Error</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{round(mape_val,2)}%</div>
        <div class="metric-label">MAPE</div>
        <div class="metric-delta">Mean Absolute Percentage Error</div>
    </div>""", unsafe_allow_html=True)
with c4:
    winner = "LightGBM" if lgb_mae <= prop_mae else "Prophet"
    st.markdown(f"""<div class="winner-badge">
        {winner}<br>
        <span style="font-size:0.8rem;font-weight:normal;">
        Best Performing Model</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 2 — Actual vs Predicted ──────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">Actual vs Predicted Sales — 2016 Test Period</p>',
            unsafe_allow_html=True)

upper = predictions + mae_val
lower = np.maximum(predictions - mae_val, 0)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=test['date'], y=list(upper) + list(lower)[::-1],
    fill='toself', fillcolor='rgba(229,57,53,0.08)',
    line=dict(color='rgba(0,0,0,0)'), showlegend=False,
    hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=test['date'], y=y_test,
    mode='lines', name='Actual Sales',
    line=dict(color='#1565c0', width=2)
))
fig.add_trace(go.Scatter(
    x=test['date'], y=predictions,
    mode='lines', name=f'{selected_model} Predictions',
    line=dict(color='#e53935', width=2, dash='dash')
))
layout = base_layout('', 420, 'Date', 'Units Sold')
layout['hovermode'] = 'x unified'
layout['legend'] = dict(
    bgcolor='white', bordercolor='#cccccc',
    borderwidth=1, font=dict(color='#111111', size=11),
    orientation='h', yanchor='bottom', y=1.02,
    xanchor='right', x=1
)
fig.update_layout(**layout)
st.plotly_chart(fig, use_container_width=True)

# ── Section 3 — Model Comparison ─────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">Model Comparison</p>',
            unsafe_allow_html=True)

model_names = ['LightGBM', 'Prophet', 'SARIMA']
mae_vals    = [15.66, 22.43, 56.48]
rmse_vals   = [19.80, 27.62, 59.61]
mape_vals   = [35.56, 38.56, 124.11]
bar_colors  = ['#1565c0', '#0288d1', '#e53935']

c1, c2 = st.columns(2)
with c1:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name='MAE', x=model_names, y=mae_vals,
        marker_color=bar_colors,
        text=[str(v) for v in mae_vals],
        textposition='outside',
        textfont=dict(color='#111111', size=11),
        cliponaxis=False
    ))
    fig2.add_trace(go.Bar(
        name='RMSE', x=model_names, y=rmse_vals,
        marker_color=['rgba(21,101,192,0.45)',
                      'rgba(2,136,209,0.45)',
                      'rgba(229,57,53,0.45)'],
        text=[str(v) for v in rmse_vals],
        textposition='outside',
        textfont=dict(color='#111111', size=11),
        cliponaxis=False
    ))
    layout2 = base_layout('MAE and RMSE by Model', 370,
                           'Model', 'Error (units)')
    layout2['barmode'] = 'group'
    layout2['yaxis']['range'] = [0, 80]
    layout2['bargap'] = 0.25
    fig2.update_layout(**layout2)
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    fig3 = go.Figure(go.Bar(
        x=model_names, y=mape_vals,
        marker_color=bar_colors,
        text=[f'{v}%' for v in mape_vals],
        textposition='outside',
        textfont=dict(color='#111111', size=11),
        cliponaxis=False
    ))
    layout3 = base_layout('MAPE by Model', 370, 'Model', 'MAPE (%)')
    layout3['yaxis']['range'] = [0, 145]
    fig3.update_layout(**layout3)
    st.plotly_chart(fig3, use_container_width=True)

# ── Section 4 — Residual Analysis ────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">Residual Analysis</p>',
            unsafe_allow_html=True)

residuals = y_test.values - predictions
c1, c2 = st.columns(2)

with c1:
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=test['date'], y=residuals,
        mode='lines', name='Residuals',
        line=dict(color='#7b1fa2', width=1.5)
    ))
    fig4.add_hline(y=0, line_dash='dash',
                   line_color='#333333', line_width=1.2)
    layout4 = base_layout(
        f'{selected_model} — Residuals Over Time', 310, 'Date', 'Residual (units)')
    fig4.update_layout(**layout4)
    st.plotly_chart(fig4, use_container_width=True)

with c2:
    fig5 = go.Figure(go.Histogram(
        x=residuals, nbinsx=20,
        marker_color='#7b1fa2', opacity=0.85,
        name='Frequency'
    ))
    fig5.add_vline(x=0, line_dash='dash',
                   line_color='#333333', line_width=1.2)
    layout5 = base_layout('Error Distribution', 310,
                           'Residual Value', 'Frequency')
    fig5.update_layout(**layout5)
    st.plotly_chart(fig5, use_container_width=True)

# ── Section 5 — EDA Insights ─────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">EDA Insights</p>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
day_order   = ['Monday','Tuesday','Wednesday',
               'Thursday','Friday','Saturday','Sunday']
month_names_list = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']

with c1:
    day_sales = full_df.groupby('weekday')['sales'].mean().reset_index()
    day_sales['weekday'] = pd.Categorical(
        day_sales['weekday'], categories=day_order, ordered=True)
    day_sales = day_sales.sort_values('weekday')
    fig6 = go.Figure(go.Bar(
        x=day_sales['weekday'],
        y=day_sales['sales'].round(1),
        marker_color='#1565c0', opacity=0.85,
        text=day_sales['sales'].round(1),
        textposition='outside',
        textfont=dict(color='#111111', size=10),
        cliponaxis=False
    ))
    layout6 = base_layout('Avg Sales by Day of Week', 310,
                           'Day', 'Avg Units Sold')
    layout6['yaxis']['range'] = [0, day_sales['sales'].max() * 1.25]
    fig6.update_layout(**layout6)
    st.plotly_chart(fig6, use_container_width=True)

with c2:
    month_sales = full_df.groupby('month')['sales'].mean().reset_index()
    month_sales['month_name'] = month_sales['month'].apply(
        lambda x: month_names_list[x-1])
    fig7 = go.Figure(go.Bar(
        x=month_sales['month_name'],
        y=month_sales['sales'].round(1),
        marker_color='#0288d1', opacity=0.85,
        text=month_sales['sales'].round(1),
        textposition='outside',
        textfont=dict(color='#111111', size=10),
        cliponaxis=False
    ))
    layout7 = base_layout('Avg Sales by Month', 310,
                           'Month', 'Avg Units Sold')
    layout7['yaxis']['range'] = [0, month_sales['sales'].max() * 1.25]
    fig7.update_layout(**layout7)
    st.plotly_chart(fig7, use_container_width=True)

with c3:
    snap_sales = full_df.groupby('snap_CA')['sales'].mean().reset_index()
    snap_sales['label'] = snap_sales['snap_CA'].map(
        {0: 'Non-SNAP Day', 1: 'SNAP Day'})
    fig8 = go.Figure(go.Bar(
        x=snap_sales['label'],
        y=snap_sales['sales'].round(1),
        marker_color=['#1565c0', '#43a047'],
        opacity=0.85,
        text=snap_sales['sales'].round(1),
        textposition='outside',
        textfont=dict(color='#111111', size=10),
        cliponaxis=False
    ))
    layout8 = base_layout('SNAP vs Non-SNAP Day Sales', 310,
                           '', 'Avg Units Sold')
    layout8['yaxis']['range'] = [0, snap_sales['sales'].max() * 1.25]
    fig8.update_layout(**layout8)
    st.plotly_chart(fig8, use_container_width=True)

# ── Section 6 — Feature Importance + 28 Day Forecast ─────────────
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.markdown('<p class="section-header">Feature Importance</p>',
                unsafe_allow_html=True)
    if selected_model == "LightGBM":
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': lgb_model.feature_importance()
        }).sort_values('importance', ascending=True)
        fig9 = go.Figure(go.Bar(
            x=imp_df['importance'],
            y=imp_df['feature'],
            orientation='h',
            marker=dict(color=imp_df['importance'], colorscale='Blues'),
            text=imp_df['importance'],
            textposition='outside',
            textfont=dict(color='#111111', size=9),
            cliponaxis=False
        ))
        layout9 = base_layout('', 430, 'Importance Score', '')
        layout9['showlegend'] = False
        layout9['xaxis']['range'] = [0, imp_df['importance'].max() * 1.2]
        layout9['margin'] = dict(l=130, r=40, t=20, b=50)
        fig9.update_layout(**layout9)
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.info("Feature importance is only available for LightGBM. Showing Prophet components instead.")
        fig_comp = prophet_model.plot_components(
            prophet_model.predict(
                prophet_model.make_future_dataframe(periods=115)))
        st.pyplot(fig_comp)

with c2:
    st.markdown('<p class="section-header">28 Day Future Forecast</p>',
                unsafe_allow_html=True)

    if selected_model == "LightGBM":
        last_date    = test['date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=28)
        fdf = pd.DataFrame({'date': future_dates})
        fdf['day_of_week']  = fdf['date'].dt.dayofweek
        fdf['day_of_month'] = fdf['date'].dt.day
        fdf['week_of_year'] = fdf['date'].dt.isocalendar().week.astype(int)
        fdf['month']        = fdf['date'].dt.month
        fdf['year']         = fdf['date'].dt.year
        fdf['is_weekend']   = (fdf['day_of_week'] >= 5).astype(int)
        fdf['lag_7']        = list(y_test.values[-7:]) + [0]*21
        fdf['lag_28']       = list(y_test.values[-28:])
        fdf['rolling_mean_7']  = fdf['lag_7'].rolling(7,  min_periods=1).mean()
        fdf['rolling_mean_28'] = fdf['lag_28'].rolling(28, min_periods=1).mean()
        fdf['rolling_std_7']   = fdf['lag_7'].rolling(7,  min_periods=1).std().fillna(0)
        for col in ['price_change','price_change_pct','is_event','is_snap',
                    'is_superbowl','is_thanksgiving','is_christmas','is_easter']:
            fdf[col] = 0
        future_preds = lgb_model.predict(fdf[feature_cols])
    else:
        future_all   = prophet_model.make_future_dataframe(periods=28 + 115)
        future_fc    = prophet_model.predict(future_all)
        future_preds = future_fc['yhat'].tail(28).values
        future_dates = pd.date_range(
            start=test['date'].max() + pd.Timedelta(days=1), periods=28)

    uf = future_preds + mae_val
    lf = np.maximum(future_preds - mae_val, 0)

    fig10 = go.Figure()
    fig10.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(uf) + list(lf)[::-1],
        fill='toself', fillcolor='rgba(229,57,53,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='skip'
    ))
    fig10.add_trace(go.Scatter(
        x=test['date'].tail(30), y=y_test.tail(30),
        mode='lines', name='Recent Actual Sales',
        line=dict(color='#1565c0', width=2)
    ))
    fig10.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        mode='lines+markers', name='28 Day Forecast',
        line=dict(color='#e53935', width=2, dash='dash'),
        marker=dict(size=5, color='#e53935')
    ))
    layout10 = base_layout('', 430, 'Date', 'Units Sold')
    layout10['hovermode'] = 'x unified'
    layout10['legend'] = dict(
        bgcolor='white', bordercolor='#cccccc', borderwidth=1,
        font=dict(color='#111111', size=10),
        orientation='h', yanchor='bottom', y=1.02,
        xanchor='right', x=1
    )
    fig10.update_layout(**layout10)
    st.plotly_chart(fig10, use_container_width=True)

# ── Section 7 — Interactive Prediction Panel ──────────────────────
st.markdown("---")
st.markdown('<p class="section-header">Interactive Demand Prediction</p>',
            unsafe_allow_html=True)
st.markdown("Configure the inputs and click **Run Prediction** to get an instant sales forecast.")

c1, c2, c3 = st.columns(3)
day_labels   = ['Monday','Tuesday','Wednesday',
                'Thursday','Friday','Saturday','Sunday']
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

with c1:
    st.markdown("**Time Features**")
    day_of_week = st.selectbox(
        "Day of Week",
        options=list(range(7)),
        format_func=lambda x: day_labels[x]
    )
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: month_labels[x-1]
    )
    year = st.selectbox("Year", options=[2014, 2015, 2016])

with c2:
    st.markdown("**Sales History**")
    lag_7_input  = st.slider("Sales 7 Days Ago",
                              0, 300, 50, step=5)
    lag_28_input = st.slider("Sales 28 Days Ago",
                              0, 300, 45, step=5)
    sell_price   = st.slider("Sell Price ($)",
                              0.5, 15.0, 1.99, step=0.25)

with c3:
    st.markdown("**Event Flags**")
    is_snap  = st.radio("SNAP Day?", [0, 1],
                         format_func=lambda x: "Yes" if x else "No",
                         horizontal=True)
    is_event = st.radio("Holiday / Event Day?", [0, 1],
                         format_func=lambda x: "Yes" if x else "No",
                         horizontal=True)
    is_weekend = 1 if day_of_week >= 5 else 0
    st.markdown(f"**Weekend:** {'Yes' if is_weekend else 'No'} *(auto-detected)*")

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1.5, 1, 1.5])
with btn_col:
    run = st.button("Run Prediction", use_container_width=True, type="primary")

if run:
    input_df = pd.DataFrame([{
        'day_of_week':     day_of_week,
        'day_of_month':    15,
        'week_of_year':    25,
        'month':           month,
        'year':            year,
        'is_weekend':      is_weekend,
        'lag_7':           lag_7_input,
        'lag_28':          lag_28_input,
        'rolling_mean_7':  (lag_7_input + lag_28_input) / 2,
        'rolling_mean_28': lag_28_input,
        'rolling_std_7':   abs(lag_7_input - lag_28_input) / 2,
        'price_change':     0,
        'price_change_pct': 0,
        'is_event':         is_event,
        'is_snap':          is_snap,
        'is_superbowl':     0,
        'is_thanksgiving':  0,
        'is_christmas':     0,
        'is_easter':        0
    }])

    pred_val = max(0, round(lgb_model.predict(input_df[feature_cols])[0]))
    lo = max(0, pred_val - round(mae_val))
    hi = pred_val + round(mae_val)

    st.markdown(f"""
    <div class="prediction-result">
        <div class="prediction-number">{pred_val} units</div>
        <div class="prediction-sublabel">Predicted Daily Sales — LightGBM</div>
        <div class="prediction-range">Confidence Range: {lo} &ndash; {hi} units</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"**Day:** {day_labels[day_of_week]}  |  "
        f"**Month:** {month_labels[month-1]} {year}  |  "
        f"**Weekend:** {'Yes' if is_weekend else 'No'}  |  "
        f"**SNAP:** {'Yes' if is_snap else 'No'}  |  "
        f"**Event:** {'Yes' if is_event else 'No'}  |  "
        f"**Sales 7d ago:** {lag_7_input}  |  "
        f"**Sales 28d ago:** {lag_28_input}  |  "
        f"**Price:** ${sell_price}"
    )

# ── Section 8 — Key Insights ──────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">Key Insights</p>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="insight-card"><b>Weekly Seasonality</b> — Sunday and Saturday
    are the highest sales days, with weekends showing 28–37% above average demand.</div>
    <div class="insight-card"><b>Summer Peak</b> — Sales build steadily from
    June through August, driven by back-to-school shopping and summer gatherings.</div>
    <div class="insight-card"><b>SNAP Effect</b> — Food stamp payout days show
    a clear and reliable uplift in food sales, making snap_CA a strong model feature.</div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="insight-card"><b>LightGBM Wins</b> — MAE of 15.66 vs Prophet's
    22.43 and SARIMA's 56.48. Feature engineering gave LightGBM a decisive advantage.</div>
    <div class="insight-card"><b>Top Features</b> — Rolling mean (7-day), lag_7
    and rolling std are the strongest predictors, confirming that recent history
    drives future demand.</div>
    <div class="insight-card"><b>Price Elasticity</b> — Strong inverse
    relationship between price and demand. Products priced $0–$5 account for
    the majority of sales volume.</div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""**Dataset**  
M5 Forecasting Competition  
Walmart Sales Data 2011–2016  
42,000+ time series""")
with c2:
    st.markdown("""**Models**  
LightGBM — MAE: 15.66  
Prophet — MAE: 22.43  
SARIMA — MAE: 56.48""")
with c3:
    st.markdown("""**Built By**  
Soumya Patil  
Data Analytics Portfolio Project  
2026""")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#9e9e9e;font-size:0.8rem;'>"
    "Built with Python &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly "
    "&nbsp;·&nbsp; LightGBM &nbsp;·&nbsp; Prophet &nbsp;·&nbsp; Statsmodels"
    "</p>",
    unsafe_allow_html=True
)