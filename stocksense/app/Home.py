import streamlit as st
import yfinance as yf
from pandas import DataFrame
import numpy as np
from stocksense.src.model.autoregressor import AutoRegressor
from stocksense.src.backend import moving_average
import streamlit.components.v1 as components
import plotly.graph_objs as go


currency = '$'


def home_page():
    st.set_page_config(
        page_title='Stock Sense',
        page_icon='📈',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title('📈Stock Sense📈')

    left_column, right_column = st.columns([2, 3], gap='medium')

    with left_column:
        with st.form('stock_form'):
            stock_name = st.selectbox('Select stock for analysis: ',
                                      ('Apple', 'Microsoft',
                                       'Tesla', 'Alphabet'))

            period = st.selectbox('Select period for analysis: ',
                                  ('3 month', '6 month', '1 year',
                                   '2 years', '5 years', '10 years',
                                   'All time'))
            predict_period = st.selectbox('Select period for forecasting: ',
                                          ('1 day', '3 days', '1 week',
                                           '2 weeks', '1 month'))

            submitted = st.form_submit_button("Submit")
            if submitted:
                df = __load_ticker(stock_name, period)
                __stock_stats(df, stock_name, period)
                __predict_block(df, stock_name, predict_period)

    with right_column:
        if submitted:
            __right_column(df, stock_name)

    __sidebar()


def __load_ticker(stock_name: str, period: str) -> DataFrame:
    st.write(f'Stock {stock_name} selected.')
    stock_name2ticket = {
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Tesla': 'TSLA',
        'Alphabet': 'GOOG'
    }
    periods = {
        '3 month': '3mo',
        '6 month': '6mo',
        '1 year': '1y',
        '2 years': '2y',
        '5 years': '5y',
        '10 years': '10y',
        'All time': 'max'
    }

    progress_text = 'Downloading stock data...'
    load_bar = st.progress(0, text=progress_text)

    load_bar.progress(25, text=progress_text)
    df = yf.download(tickers=[stock_name2ticket[stock_name]],
                     period=periods[period])
    load_bar.progress(50, text=progress_text)

    load_bar.progress(75, text=progress_text)
    load_bar.progress(100, text=progress_text)
    load_bar.empty()

    return df


def __right_column(df: DataFrame, stock_name: str):
    tab_chart, tab_df = st.tabs(['📈 Chart', '🗃 Data'])

    tab_chart.subheader(f'Close prices of {stock_name}')
    df['Date'] = df.index
    graphic_data = df[['Date', 'Close']]
    graphic_data['Moving Average'] = moving_average(df['Close'].to_numpy())
    close_prices = go.Line(x=graphic_data['Date'],
                           y=graphic_data['Close'],
                           name='Close Prices')
    ma = go.Line(x=graphic_data['Date'],
                 y=graphic_data['Moving Average'],
                 name='Moving Average')

    layout = go.Layout(xaxis=dict(title='Date'),
                       yaxis=dict(title='Price'),
                       height=500)

    fig = go.Figure(data=[close_prices, ma], layout=layout)

    tab_chart.plotly_chart(fig,
                           use_container_width=True)

    tab_df.subheader("Dataframe")
    tab_df.write(f'Last 20 close prices of {stock_name}')
    tab_df.dataframe(df[['Date', 'Close']].tail(20), use_container_width=True,
                     hide_index=True)

    st.subheader('Candlestick:')
    candlestick = go.Candlestick(x=df['Date'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'])

    layout = go.Layout(xaxis=dict(title='Date'),
                       yaxis=dict(title='Price'),
                       height=600)

    fig = go.Figure(data=[candlestick], layout=layout)

    st.plotly_chart(fig,
                    use_container_width=True)


def __sidebar():
    st.sidebar.markdown('> Make smarter investment decisions.')
    st.sidebar.markdown('''**Stock Sense** provides clear, concise stock
                        analysis to help you navigate the market.''')
    st.sidebar.write('Mean accuracy of models: 97,88 %')
    st.sidebar.markdown('author: [zibestr](https://github.com/zibestr)')


def __stock_stats(df: DataFrame, stock_name: str, period: str):
    st.header(f'Stock Stats for {stock_name}')
    st.write(f'Period: {period}')
    st.markdown(
        f'- Mean price: {df["Close"].mean():.2f} {currency}' +
        '\n' +
        f'- Median price: {df["Close"].median():.2f} {currency}' +
        '\n' +
        '- First Quantile price: ' +
        f'{df["Close"].quantile(0.25):.2f} {currency}' +
        '\n' +
        '- Third Quantile price: ' +
        f'{df["Close"].quantile(0.75):.2f} {currency}' +
        '\n' +
        f'- Max price: {df["Close"].max():.2f} {currency}' +
        '\n' +
        f'- Min price: {df["Close"].min():.2f} {currency}'
    )


def __predict_block(df: DataFrame, stock_name: str, str_period: str):
    periods = {
        '1 day': 1,
        '3 days': 3,
        '1 week': 7,
        '2 weeks': 14,
        '1 month': 30,
        None: 1
    }
    period = periods[str_period]
    progress_text = 'Downloading regressor algorithm...'
    load_bar = st.progress(0, text=progress_text)

    load_bar.progress(50, text=progress_text)
    regressor = AutoRegressor(stock_name)
    load_bar.progress(100, text=progress_text)
    load_bar.empty()

    st.markdown('### Forecasting prices')

    predicted = np.round(
            regressor.predict(
                df['Close'].iloc[-30:].to_numpy().reshape(1, -1),
                period
            ),
            2
    )
    last_price = df['Close'].iloc[-1]
    delta = round((predicted[0] - last_price) / last_price * 100, 2)

    price_col, predicted_col = st.columns(2)
    with price_col:
        st.metric('#### Last Price', f'{df["Close"].iloc[-1]:0.2f} {currency}')
    with predicted_col:
        st.metric('#### Next price', f'{predicted[0]} {currency}',
                  f'{(predicted[0] - last_price):0.2f} ({delta}%)')

    if period > 1:
        predicted_df = DataFrame({'Day': [f'Day {i}'
                                          for i in range(1, period + 1)],
                                  'Price': predicted})
        st.markdown('#### Future prices')
        tab_chart, tab_df = st.tabs(['📈 Chart', '🗃 Data'])

        tab_chart.subheader(f'Future prices of {stock_name}')
        line_chart = go.Line(x=predicted_df['Day'],
                          y=predicted_df['Price'])

        layout = go.Layout(xaxis=dict(title='Day'),
                           yaxis=dict(title='Price'),
                           height=500)

        fig = go.Figure(data=[line_chart], layout=layout)

        tab_chart.plotly_chart(fig,
                               use_container_width=True)

        tab_df.subheader("Dataframe")
        tab_df.dataframe(predicted_df,
                         use_container_width=True,
                         hide_index=True)

    st.markdown('#### Regression algorithm:')
    components.html(regressor.html_repr, height=300, scrolling=True)
