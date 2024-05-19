import streamlit as st
import yfinance as yf
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.model.autoregressor import AutoRegressor
from src.backend import moving_average
import streamlit.components.v1 as components

mpl.rc('axes', facecolor='#0E1117', edgecolor='#0E1117')
mpl.rc('savefig', facecolor='#0E1117', edgecolor='#0E1117')
mpl.rc('xtick', color='#0E1117')
mpl.rc('ytick', color='white')
mpl.rc('font', size=6, weight='bold', style='normal')
mpl.rc('boxplot.capprops', linewidth=1.5, color='white')
mpl.rc('boxplot.boxprops', linewidth=1.5, color='white')
mpl.rc('boxplot.whiskerprops', linewidth=0.5, color='white')
mpl.rc('boxplot.medianprops', linewidth=1.5, color='#FF4B4B')


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
    df = df[['Close']]

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
    tab_chart.line_chart(graphic_data, x='Date', y=['Close', 'Moving Average'],
                         use_container_width=True,
                         height=700)

    tab_df.subheader("Dataframe")
    tab_df.write(f'Last 20 close prices of {stock_name}')
    tab_df.dataframe(df.tail(20), use_container_width=True, hide_index=True)

    st.subheader('Boxplot:')

    fig = plt.figure(figsize=(3, 3))
    ax = fig.subplots(1, 1)
    ax.boxplot(df['Close'])
    st.pyplot(fig,
              use_container_width=True)


def __sidebar():
    st.sidebar.markdown('> Make smarter investment decisions.')
    st.sidebar.markdown('''**Stock Sense** provides clear, concise stock
                        analysis to help you navigate the market.''')
    st.sidebar.write('Mean accuracy of models: 97,84 %')
    st.sidebar.markdown('author: [zibestr](https://github.com/zibestr)')


def __stock_stats(df: DataFrame, stock_name: str, period: str):
    st.header(f'Stock Stats for {stock_name}')
    st.write(f'Period: {period}')
    st.markdown(
        f'- Mean price: {df["Close"].mean():.3f}' +
        '\n' +
        f'- Median price: {df["Close"].median():.3f}' +
        '\n' +
        f'- First Quantile price: {df["Close"].quantile(0.25):.3f}' +
        '\n' +
        f'- Third Quantile price: {df["Close"].quantile(0.75):.3f}' +
        '\n' +
        f'- Max price: {df["Close"].max():.3f}' +
        '\n' +
        f'- Min price: {df["Close"].min():.3f}'
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
    st.write('Regression algorithm:')
    components.html(regressor.html_repr, height=300, scrolling=True)

    predicted = np.round(
            regressor.predict(
                df['Close'].iloc[-30:].to_numpy().reshape(1, -1),
                period
            ),
            2
    )
    last_price = df['Close'].iloc[-1]
    delta = round((predicted[0] - last_price) / last_price * 100, 2)
    st.metric('#### Next price', predicted[0], f'{delta}%')

    if period > 1:
        predicted_df = DataFrame({'Day': [f'Day {i}'
                                          for i in range(1, period + 1)],
                                  'Price': predicted})
        st.markdown('#### Future prices')
        tab_chart, tab_df = st.tabs(['📈 Chart', '🗃 Data'])

        tab_chart.subheader(f'Future prices of {stock_name}')
        tab_chart.line_chart(predicted_df, x='Day', y='Price',
                             height=500)

        tab_df.subheader("Dataframe")
        tab_df.dataframe(predicted_df,
                         use_container_width=True,
                         hide_index=True)
