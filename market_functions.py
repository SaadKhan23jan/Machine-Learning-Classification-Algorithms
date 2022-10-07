from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
import pandas as pd
from dash import dash_table
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def sarimax_pred(df, crypto, p, i, q, sarimax_model, days, sp, si, sq, seasonal_factor,
                 start_p_order, start_i_order, start_q_order, max_p_order, max_i_order, max_q_order,
                 auto_arima_seasonal_factor, auto_arima_seasonal, auto_arima_stationary,
                 auto_arima_information_criterion, auto_arima_method
                 ):
    if sarimax_model == 'MA':
        order = (p, 0, 0)
    elif sarimax_model == 'AR':
        order = (0, 0, q)
    elif sarimax_model == 'ARMA':
        order = (p, 0, q)
    else:
        order = (p, i, q)

    if sarimax_model == 'SARIMAX':
        model = SARIMAX(df['Close'], trend='c', order=order, seasonal_order=(sp, si, sq, seasonal_factor))
        results = model.fit(df['Close'])
        preds = SARIMAXResults.predict(results, start=len(df), end=len(df) + days)
        # actual = df['Close'].values
        results_summary = results.summary()

    elif sarimax_model == 'Auto ARIMA':
        model = auto_arima(df['Close'], start_p=start_p_order, d=start_i_order, start_q=start_q_order,
                           max_p=max_p_order, max_d=max_i_order, max_q=max_q_order,
                           m=auto_arima_seasonal_factor,
                           seasonal=auto_arima_seasonal, stationary=auto_arima_stationary,
                           information_criterion=auto_arima_information_criterion, method=auto_arima_method)
        results = model.fit(df['Close'])
        preds = model.predict(n_periods=days, start=len(df), end=len(df) + days)
        # actual = df['Close'].values
        results_summary = results.summary()
    else:
        model = ARIMA(df['Close'], order=order)
        results = model.fit()
        preds = SARIMAXResults.predict(results, start=len(df), end=len(df) + days)
        # actual = df['Close'].values
        results_summary = results.summary()

    results_as_html = results_summary.tables[1].as_html()
    results = pd.read_html(results_as_html, header=0, index_col=0)[0]
    data = results.to_dict(orient='records')
    columns = [{"name": i, "id": i, } for i in results.columns]

    pred_fig = px.line(y=preds)
    pred_fig.update_layout(title='Predictions')
    pred_fig.update_xaxes(title='Number of Forecast Days')
    pred_fig.update_yaxes(title=f'{crypto} Price')

    return dash_table.DataTable(data=data, columns=columns), pred_fig


def crypto_plots(df_live, crypto):

    fig1 = go.Figure(data=[go.Candlestick(x=df_live.index,
                                          open=df_live.Open,
                                          high=df_live.High,
                                          low=df_live.Low,
                                          close=df_live.Close)])
    fig1 = go.Figure(data=[go.Candlestick(x=df_live.index,
                                          open=df_live.Open,
                                          high=df_live.High,
                                          low=df_live.Low,
                                          close=df_live.Close)])
    fig1.update_layout(title=f'Candle Chart of {crypto}', xaxis_title='Time', yaxis_title=f'{crypto}')

    fig2 = px.line(data_frame=df_live, x=df_live.index, y=[df_live['Open'], df_live['High'],
                                                           df_live['Low'], df_live['Close']])
    fig2.update_layout(title=f'History of Price {crypto}', xaxis_title='Time', yaxis_title=f'Price History of {crypto}')

    fig3 = px.line(data_frame=df_live, x=df_live.index, y=df_live['Volume'], markers='o')
    fig3.update_layout(title=f'History of Volume {crypto}', xaxis_title='Time', yaxis_title=f'Volume of {crypto}')

    return fig1, fig2, fig3


def decomposition(df, time_frame):
    df_decomposed = df['Close'].asfreq(freq='h')
    df_decomposed = df_decomposed.fillna(method='ffill')
    res = seasonal_decompose(df_decomposed)
    fig_seasonality_decompose = make_subplots(rows=4, cols=1,
                                              subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"])
    fig_seasonality_decompose.add_trace(go.Scatter(x=res.observed.index, y=res.observed.values, ), row=1, col=1, )
    fig_seasonality_decompose.add_trace(go.Scatter(x=res.trend.index, y=res.trend.values), row=2, col=1)
    fig_seasonality_decompose.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal.values, ), row=3, col=1)
    fig_seasonality_decompose.add_trace(go.Scatter(x=res.resid.index, y=res.resid.values, ), row=4, col=1)
    fig_seasonality_decompose.update_layout(height=900)

    return fig_seasonality_decompose
