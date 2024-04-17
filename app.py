import streamlit as st
import pandas as pd
import ccxt
import requests
import plotly.express as px
import plotly.graph_objects as go
from helpers import *

st.set_page_config(
    theme="dark"
)

exchange = ccxt.binanceus()

ohlc = ['Open','High','Low','Close']
_o,_h,_l,_c = [ohlc[h] for h in range(len(ohlc))]


def get_top_n_crypto_tickers(per_page=100, page=1):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": False
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        tickers = [crypto["symbol"].upper() for crypto in data]
        return tickers
    else:
        st.error("Failed to retrieve data.")
        return None


def relative(df,_o,_h,_l,_c, bm_df, bm_col, dgt, start, end,rebase=True):
    '''
    df: df
    bm_df, bm_col: df benchmark dataframe & column name
    dgt: rounding decimal
    start/end: string or offset
    rebase: boolean rebase to beginning or continuous series
    '''
    # Slice df dataframe from start to end period: either offset or datetime
    df = df.loc[start:end, :] 
    
    # inner join of benchmark & currency: only common values are preserved
    df = df.join(bm_df[[bm_col]],how='inner') 

    # rename benchmark name as bm and currency as ccy
    df.rename(columns={bm_col:'bm'},inplace=True)
    
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot proceed further.")
    
    if rebase == True:
        df['bm'] = df['bm'].div(df['bm'].iloc[0])

    # Divide absolute price by fxcy adjustment factor and rebase to first value
    df['r' + str(_o)] = round(df[_o].div(df['bm']),dgt)
    df['r' + str(_h)] = round(df[_h].div(df['bm']),dgt)
    df['r'+ str(_l)] = round(df[_l].div(df['bm']),dgt)
    df['r'+ str(_c)] = round(df[_c].div(df['bm']),dgt)
    df = df.drop(['bm'],axis=1)
    
    return (df)


def get_relative_performance(ticker, start, end, ohlcv_data_bm=None, bm='BTC/USDT', limit=365*2):

    ohlcv_data = exchange.fetch_ohlcv(ticker, timeframe='1d', limit=limit) 

    df = pd.DataFrame(ohlcv_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    bm_df = pd.DataFrame()
    dgt = 5

    if not ohlcv_data_bm:
        ohlcv_data_bm = exchange.fetch_ohlcv(bm, timeframe='1d', limit=limit)

    bm_df[['Timestamp', bm]] =  pd.DataFrame(ohlcv_data_bm, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])[['Timestamp', 'Close']].iloc[:-1, :]
    bm_df['Timestamp'] = pd.to_datetime(bm_df['Timestamp'], unit='ms')
    bm_df.set_index('Timestamp', inplace=True)

    df = relative(df,_o,_h,_l,_c, bm_df, bm, dgt, start, end, rebase=True)

    return df


@st.cache_data()
def load_data(start_date, end_date, market_cap):
    dict_page = {
        'Top 100': 1,
        'Top 101-200': 2,
        'Top 201-300': 3
    }
    # Fetching top tickers
    top_100_tickers = get_top_n_crypto_tickers(per_page=100, page=dict_page[market_cap])
    symbols_with_usdt = [symbol + '/USDT' for symbol in top_100_tickers]

    # Processing data for each ticker
    list_ticker = symbols_with_usdt
    list_relative_benchmark = []

    for ticker in list_ticker:
        try:
            df = get_relative_performance(ticker, start=start_date, end=end_date, ohlcv_data_bm=ohlcv_data_bm, bm='BTC/USDT', limit=365)
            relative_benchmark = (df['rClose'].iloc[-1] - df['rClose'].iloc[0])*100/df['rClose'].iloc[0]
            list_relative_benchmark.append(relative_benchmark)
        except ccxt.BadSymbol as e:
            list_relative_benchmark.append(0.0)
        except ValueError as e:
            list_relative_benchmark.append(0.0)
    benchmark_df = pd.DataFrame({'ticker':list_ticker, 'score':list_relative_benchmark})
    benchmark_df.sort_values('score', ascending=False, inplace=True)
    filtered_df = benchmark_df[benchmark_df['score'] != 0]
    return filtered_df


st.title('Cryptocurrency Relative Performance Analyzer')

# Sidebar
st.sidebar.title('Settings')

with st.sidebar.form("Date Range Selector"):
    start_date = st.date_input("Start Date", pd.to_datetime('2024-03-13'))
    end_date = st.date_input("End Date", pd.to_datetime('2024-04-15'))
    market_cap = st.selectbox("Select market cap", ["Top 100", "Top 101-200", "Top 201-300"], index=0)

    submit_button = st.form_submit_button("Submit")



# Search form
with st.sidebar.form("Search Form", clear_on_submit=True):
    search_query = st.text_input("Search by Name")

    submitted = st.form_submit_button("Search")

# # Fetching top tickers
# top_100_tickers = get_top_n_crypto_tickers(per_page=100, page=1)
# symbols_with_usdt = [symbol + '/USDT' for symbol in top_100_tickers]

# # Fetching benchmark data
ohlcv_data_bm = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=365*2)

# # Processing data for each ticker
# list_ticker = symbols_with_usdt
# list_relative_benchmark = []

filtered_df = load_data(start_date, end_date, market_cap)

# Displaying top performers
st.subheader('Top Performers')
fig = px.bar(filtered_df, x='ticker', y='score',
             color='score',  # if you want to color bars by score
             color_continuous_scale=px.colors.diverging.RdYlBu,  # choose color scale
             title=f'Scores by Ticker {market_cap}')
fig.update_xaxes(tickangle=90)
st.plotly_chart(fig)



# Plotting relative performance
st.subheader('Relative Performance Plot')
if submitted:
    selected_ticker = st.selectbox('Select a ticker to plot relative performance', list(filtered_df['ticker']), index=None)
    selected_ticker = search_query
else:   
    selected_ticker = st.selectbox('Select a ticker to plot relative performance', list(filtered_df['ticker']))


df_selected = get_relative_performance(selected_ticker, start=start_date, end=end_date, ohlcv_data_bm=ohlcv_data_bm, bm='BTC/USDT', limit=365*2)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['Close'], mode='lines', name='Close', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['rClose'], mode='lines', name='rClose', line=dict(color='red')))
fig.update_layout(title=f'Close and rClose of {selected_ticker}',
                xaxis_title='Date',
                yaxis_title='Value')
st.plotly_chart(fig)



difference = end_date - start_date
df_selected = df_selected.iloc[-difference.days:] # Last 30 candles in data
support_coefs_c, resist_coefs_c = fit_trendlines_single(df_selected['Close'])

support_line_c = support_coefs_c[0] * np.arange(len(df_selected)) + support_coefs_c[1]
resist_line_c = resist_coefs_c[0] * np.arange(len(df_selected)) + resist_coefs_c[1]
fig = go.Figure(data=[go.Candlestick(x=df_selected.index,
                open=df_selected['Open'],
                high=df_selected['High'],
                low=df_selected['Low'],
                close=df_selected['Close'])])

# Add support and resistance lines
fig.add_trace(go.Scatter(x=df_selected.index, y=support_line_c, mode='lines', name='Support', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_selected.index, y=resist_line_c, mode='lines', name='Resistance', line=dict(color='red')))

fig.update_layout(title=f'{selected_ticker} Chart with Support and Resistance Lines',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)


st.plotly_chart(fig)