import streamlit as st
import yfinance as yf
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
from stocksense.src.model import get_apple_model
import streamlit.components.v1 as components

mpl.rc('axes', facecolor='#0E1117', edgecolor='#0E1117')
mpl.rc('savefig', facecolor='#0E1117', edgecolor='#0E1117')
mpl.rc('xtick', color='#0E1117')
mpl.rc('font', size=13, weight='bold', style='normal')
mpl.rc('boxplot.capprops', linewidth=3)
mpl.rc('boxplot.medianprops', linewidth=3, color='#FF4B4B')


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
                                  ('1 month', '3 month',
                                   '6 month', '1 year',
                                   '2 years', '5 years',
                                   '10 years', 'All time'))

            submitted = st.form_submit_button("Submit")
            if submitted:
                df = __load_ticker(stock_name, period)
                __stock_stats(df, stock_name, period)

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
        '1 month': '1mo',
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
    tab_chart.line_chart(df)

    tab_df.subheader("Dataframe")
    tab_df.write(f'Last 20 close prices of {stock_name}')
    tab_df.dataframe(df.tail(20))

    st.subheader('Boxplot:')

    fig = plt.figure(figsize=(16, 12))
    ax = fig.subplots(1, 1)
    ax.boxplot(df['Close'])
    st.pyplot(fig)


def __sidebar():
    st.sidebar.markdown('> Make smarter investment decisions.')
    st.sidebar.markdown('''**Stock Sense** provides clear, concise stock
                        analysis to help you navigate the market.''')
    st.sidebar.write('Mean accuracy of models: 97 %')
    st.sidebar.markdown('author: [zibestr](https://github.com/zibestr)')


def __stock_stats(df: DataFrame, stock_name: str, period: str):
    st.header(f'Stock Stats for {stock_name}')
    st.write(f'Period: {period}')
    st.markdown(
        f'- Mean price: {df['Close'].mean():.3f}' +
        '\n' +
        f'- Median price: {df['Close'].median():.3f}' +
        '\n' +
        f'- First Quantile price: {df['Close'].quantile(0.25):.3f}' +
        '\n' +
        f'- Third Quantile price: {df['Close'].quantile(0.75):.3f}' +
        '\n' +
        f'- Max price: {df['Close'].max():.3f}' +
        '\n' +
        f'- Min price: {df['Close'].min():.3f}'
    )
    model, html = __get_model(stock_name)

    predicted = model.predict(df['Close'].iloc[-30:].to_numpy()
                              .reshape(1, -1))[-1]
    st.markdown(f'#### Predicted next Close price: {predicted:.3f}')
    components.html(html, height=300)


def __get_model(stock_name: str):
    match stock_name:
        case 'Apple':
            return get_apple_model()
