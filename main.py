#%% md
# 
# # Moving Average Convergence Divergence (MACD) analysis
#%%
from typing import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.core.interchange.dataframe_protocol import DataFrame

WINDOW_LEN = 1000
#%%
# Read stock data
csvFile = pd.read_csv('pkn_orlen.csv', sep=';', usecols=['Data', 'Otwarcie'])
stock_data = csvFile.rename(columns={'Data': 'Date', 'Otwarcie': 'Price'})
stock_data = stock_data.tail(1026)

# Make indexes start at 0
smallest_index = stock_data.index.min()
stock_data.index = stock_data.index - smallest_index

# Fix dates for plotting
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

stock_data
#%%
def calc_ema(data: DataFrame, n : int) -> DataFrame:
    alpha = 2 / (n + 1)
    prices =  data.iloc[:, 1].to_list()
    ema_values = [prices[0]]
    for i in range(1, len(prices)):
        new_ema = round((alpha * prices[i]) + (1 - alpha) * ema_values[-1], 2)
        ema_values.append(new_ema)
    ema_df = pd.DataFrame({
        'Date' : data['Date'].tail(len(ema_values)),
        f'EMA_{n}': ema_values}
    )
    return ema_df
#%%
def calc_macd(data: DataFrame, n1 = 12, n2=26) -> DataFrame:
    ema_1 = calc_ema(data, n1).iloc[:, 1]
    ema_2 = calc_ema(data, n2).iloc[:, 1]
    macd_values = ema_1 - ema_2
    macd_df = pd.DataFrame({
        'Date' : data['Date'].tail(len(macd_values)),
        f'MACD_{n1}_{n2}': macd_values}
    )
    return macd_df
#%%
def calc_signal(macd: DataFrame, n = 9):
    return calc_ema(macd, n)

#%%
# calculate ema
EMA_12 = calc_ema(stock_data, 12)[-WINDOW_LEN:]
EMA_26 = calc_ema(stock_data, 26)[-WINDOW_LEN:]
EMAS = pd.merge(EMA_12, EMA_26, on='Date')
EMAS
#%%
# calculate macd and signal
MACD = calc_macd(stock_data)[-WINDOW_LEN:]
SIGNAL = calc_signal(MACD)[-WINDOW_LEN:]
SIGNAL = SIGNAL.rename(columns={SIGNAL.columns[1]: 'Signal'})
MS = pd.merge(MACD, SIGNAL, on='Date')
MS

#%%
def calc_cross_points(macd: DataFrame, signal: DataFrame, prices: DataFrame) -> DataFrame:
    cp = pd.DataFrame(columns=['Date', 'Cross Point', 'Price'])

    for i in range(1, macd.shape[0]):
        date = macd.iloc[i]['Date']
        price = prices[prices['Date'] == date].iloc[0]['Price']
        value = None

        if macd.iloc[i, 1] >= signal.iloc[i, 1] and macd.iloc[i - 1, 1] < signal.iloc[i - 1, 1]:
            value = 'Buy'
        elif macd.iloc[i, 1] <= signal.iloc[i, 1] and macd.iloc[i - 1, 1] > signal.iloc[i - 1, 1]:
            value = 'Sell'
        if value is not None:
            cp.loc[len(cp)] = [date, value, price]
    return cp

#%%
# calculate cross points
cross_points = calc_cross_points(MACD, SIGNAL, stock_data)
cross_points
#%%
# MACD and Signal Lines with Cross Points
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(MS['Date'], MS.iloc[:, 1], label='MACD')
ax.plot(MS['Date'], MS['Signal'], label='Signal')

buy_points = cross_points[cross_points['Cross Point'] == 'Buy']
ax.scatter(buy_points['Date'], MS[MS['Date'].isin(buy_points['Date'])].iloc[:, 1],
           marker='^', color='green', s=40, label='Buy', zorder=2)

sell_points = cross_points[cross_points['Cross Point'] == 'Sell']
ax.scatter(sell_points['Date'], MS[MS['Date'].isin(sell_points['Date'])].iloc[:, 1],
           marker='v', color='red', s=40, label='Sell', zorder=2)


ax.set_xlabel('Date')
ax.set_ylabel('MACD / Signal Value')
ax.set_title('MACD and Signal Lines with Cross Points')
ax.legend()
ax.grid(True)
plt.tight_layout()

locator = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(locator)
# Style dates
formatter = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(formatter)
# Rotate dates
plt.xticks(rotation=-45, ha='left')
first_date = stock_data['Date'].iloc[0]
last_date = stock_data['Date'].iloc[-1]
ax.set_xlim(first_date, last_date)

first_date_num = mdates.date2num(first_date)
last_date_num = mdates.date2num(last_date)
current_ticks = ax.get_xticks()
new_ticks = sorted(list(set(list(current_ticks) + [first_date_num, last_date_num])))
ax.set_xticks(new_ticks)

plt.savefig('plots/macd_signal_cross_points.png',dpi=200, bbox_inches='tight')

plt.show()
#%%
# Stock Price with Buy and Sell Signals
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(stock_data['Date'], stock_data['Price'], label='Stock Price', color='blue')

buy_points_price = cross_points[cross_points['Cross Point'] == 'Buy']
ax.scatter(buy_points_price['Date'], buy_points_price['Price'],
           marker='^', color='green', s=40, label='Buy Signal', zorder=2)

sell_points_price = cross_points[cross_points['Cross Point'] == 'Sell']
ax.scatter(sell_points_price['Date'], sell_points_price['Price'],
           marker='v', color='red', s=40, label='Sell Signal', zorder=2)

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Stock Price with Buy and Sell Signals')
ax.legend()
ax.grid(True)

locator = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(locator)
formatter = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=-45, ha='left')
plt.tight_layout()

first_date = stock_data['Date'].iloc[0]
last_date = stock_data['Date'].iloc[-1]
ax.set_xlim(first_date, last_date)

first_date_num = mdates.date2num(first_date)
last_date_num = mdates.date2num(last_date)
current_ticks = ax.get_xticks()
new_ticks = sorted(list(set(list(current_ticks) + [first_date_num, last_date_num])))
ax.set_xticks(new_ticks)

plt.savefig('plots/stock_price_with_signals.png', dpi=200, bbox_inches='tight')

plt.show()
#%%
# Stock price
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(stock_data['Date'], stock_data['Price'], label='Stock Price', color='blue')

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Stock price of Orlen within a chosen interval')
ax.legend()
ax.grid(True)

locator = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(locator)
formatter = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(formatter)

plt.xticks(rotation=-45, ha='left')
plt.tight_layout()

first_date = stock_data['Date'].iloc[0]
last_date = stock_data['Date'].iloc[-1]
ax.set_xlim(first_date, last_date)

first_date_num = mdates.date2num(first_date)
last_date_num = mdates.date2num(last_date)
current_ticks = ax.get_xticks()
new_ticks = sorted(list(set(list(current_ticks) + [first_date_num, last_date_num])))
ax.set_xticks(new_ticks)

plt.savefig('plots/stock_price.png', dpi=200, bbox_inches='tight')

plt.show()
#%%
# MACD Plot 1
start_date_plot1 = pd.to_datetime('2022-07-01')
end_date_plot1 = pd.to_datetime('2023-03-01')

ms_plot1 = MS[(MS['Date'] >= start_date_plot1) & (MS['Date'] <= end_date_plot1)].copy()
stock_data_plot1 = stock_data[(stock_data['Date'] >= start_date_plot1) & (stock_data['Date'] <= end_date_plot1)].copy()
cross_points_plot1 = cross_points[(cross_points['Date'] >= start_date_plot1) & (cross_points['Date'] <= end_date_plot1)].copy()

fig1, ax1 = plt.subplots(figsize=(15, 7))
ax1.plot(ms_plot1['Date'], ms_plot1.iloc[:, 1], label='MACD')
ax1.plot(ms_plot1['Date'], ms_plot1['Signal'], label='Signal')

buy_points_plot1 = cross_points_plot1[cross_points_plot1['Cross Point'] == 'Buy']
ax1.scatter(buy_points_plot1['Date'], ms_plot1[ms_plot1['Date'].isin(buy_points_plot1['Date'])].iloc[:, 1],
            marker='^', color='green', s=40, label='Buy', zorder=2)

sell_points_plot1 = cross_points_plot1[cross_points_plot1['Cross Point'] == 'Sell']
ax1.scatter(sell_points_plot1['Date'], ms_plot1[ms_plot1['Date'].isin(sell_points_plot1['Date'])].iloc[:, 1],
            marker='v', color='red', s=40, label='Sell', zorder=2)

ax1.set_xlabel('Date')
ax1.set_ylabel('MACD / Signal Value')
ax1.set_title('MACD and Signal Lines with Cross Points (2022.07.01 - 2023.03.01)')
ax1.legend()
ax1.grid(True)
plt.tight_layout()

locator1 = mdates.MonthLocator(interval=2)
ax1.xaxis.set_major_locator(locator1)
formatter1 = mdates.DateFormatter('%Y-%m-%d')
ax1.xaxis.set_major_formatter(formatter1)
plt.xticks(rotation=-45, ha='left')

plt.savefig('plots/macd_signal_transaction1.png', dpi=200, bbox_inches='tight')
plt.show()
#%%
# MACD Plot 2
start_date_plot2 = pd.to_datetime('2023-09-01')
end_date_plot2 = pd.to_datetime('2024-05-01')

ms_plot2 = MS[(MS['Date'] >= start_date_plot2) & (MS['Date'] <= end_date_plot2)].copy()
stock_data_plot2 = stock_data[(stock_data['Date'] >= start_date_plot2) & (stock_data['Date'] <= end_date_plot2)].copy()
cross_points_plot2 = cross_points[(cross_points['Date'] >= start_date_plot2) & (cross_points['Date'] <= end_date_plot2)].copy()

fig2, ax2 = plt.subplots(figsize=(15, 7))
ax2.plot(ms_plot2['Date'], ms_plot2.iloc[:, 1], label='MACD')
ax2.plot(ms_plot2['Date'], ms_plot2['Signal'], label='Signal')

buy_points_plot2 = cross_points_plot2[cross_points_plot2['Cross Point'] == 'Buy']
ax2.scatter(buy_points_plot2['Date'], ms_plot2[ms_plot2['Date'].isin(buy_points_plot2['Date'])].iloc[:, 1],
            marker='^', color='green', s=40, label='Buy', zorder=2)

sell_points_plot2 = cross_points_plot2[cross_points_plot2['Cross Point'] == 'Sell']
ax2.scatter(sell_points_plot2['Date'], ms_plot2[ms_plot2['Date'].isin(sell_points_plot2['Date'])].iloc[:, 1],
            marker='v', color='red', s=40, label='Sell', zorder=2)

ax2.set_xlabel('Date')
ax2.set_ylabel('MACD / Signal Value')
ax2.set_title('MACD and Signal Lines with Cross Points (2022.09.01 - 2023.05.01)')
ax2.legend()
ax2.grid(True)
plt.tight_layout()

locator2 = mdates.MonthLocator(interval=2)
ax2.xaxis.set_major_locator(locator2)
formatter2 = mdates.DateFormatter('%Y-%m-%d')
ax2.xaxis.set_major_formatter(formatter2)
plt.xticks(rotation=-45, ha='left')

plt.savefig('plots/macd_signal_transaction2.png', dpi=200, bbox_inches='tight')
plt.show()
#%% md
# ## Corresponding Plots for Stock Prices with Transactions
#%%
# Stock Price Plot 1
fig3, ax3 = plt.subplots(figsize=(15, 7))
ax3.plot(stock_data_plot1['Date'], stock_data_plot1['Price'], label='Stock Price', color='blue')
ax3.scatter(buy_points_plot1['Date'], stock_data_plot1[stock_data_plot1['Date'].isin(buy_points_plot1['Date'])]['Price'],
            marker='^', color='green', s=40, label='Buy Signal', zorder=2)
ax3.scatter(sell_points_plot1['Date'], stock_data_plot1[stock_data_plot1['Date'].isin(sell_points_plot1['Date'])]['Price'],
            marker='v', color='red', s=40, label='Sell Signal', zorder=2)

# Add vertical line for buy point
for date in buy_points_plot1['Date']:
    if not date == pd.Timestamp('2022-10-26 00:00:00'):
        continue
    ax3.axvline(date, color='green', linestyle='--', label='Buy' if date == buy_points_plot1['Date'].iloc[0] else "")

# Add vertical lines for sell point
for date in sell_points_plot1['Date']:
    if not date == pd.Timestamp('2022-11-29 00:00:00'):
        continue
    ax3.axvline(date, color='red', linestyle='--', label='Sell' if date == sell_points_plot1['Date'].iloc[0] else "")

# Add legend for vertical lines
handles, labels = ax3.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax3.legend(by_label.values(), by_label.keys())

ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.set_title('Stock Price with Buy and Sell Signals (2022.07.01 - 2023.03.01)')
ax3.legend()
ax3.grid(True)
ax3.xaxis.set_major_locator(locator1)
ax3.xaxis.set_major_formatter(formatter1)
plt.xticks(rotation=-45, ha='left')
plt.tight_layout()
plt.savefig('plots/stock_price_transaction1.png', dpi=200, bbox_inches='tight')
plt.show()
#%%
# Stock Price Plot 2
fig4, ax4 = plt.subplots(figsize=(15, 7))
ax4.plot(stock_data_plot2['Date'], stock_data_plot2['Price'], label='Stock Price', color='blue')
ax4.scatter(buy_points_plot2['Date'], stock_data_plot2[stock_data_plot2['Date'].isin(buy_points_plot2['Date'])]['Price'],
            marker='^', color='green', s=40, label='Buy Signal', zorder=2)
ax4.scatter(sell_points_plot2['Date'], stock_data_plot2[stock_data_plot2['Date'].isin(sell_points_plot2['Date'])]['Price'],
            marker='v', color='red', s=40, label='Sell Signal', zorder=2)

# Add vertical lines for buy points
for date in buy_points_plot2['Date']:
    if not date == pd.Timestamp('2023-09-25 00:00:00'):
        continue
    ax4.axvline(date, color='green', linestyle='--', linewidth=0.8, label='Buy' if date == buy_points_plot2['Date'].iloc[0] else "")

# Add vertical lines for sell points
for date in sell_points_plot2['Date']:
    if not date == pd.Timestamp('2023-09-28 00:00:00'):
        continue
    ax4.axvline(date, color='red', linestyle='--', linewidth=0.8, label='Sell' if date == sell_points_plot2['Date'].iloc[0] else "")

handles4, labels4 = ax4.get_legend_handles_labels()
by_label4 = dict(zip(labels4, handles4))
ax4.legend(by_label4.values(), by_label4.keys())

ax4.set_xlabel('Date')
ax4.set_ylabel('Price')
ax4.set_title('Stock Price with Buy and Sell Signals (2022.09.01 - 2023.05.01)')
ax4.legend()
ax4.grid(True)
ax4.xaxis.set_major_locator(locator2)
ax4.xaxis.set_major_formatter(formatter2)
plt.xticks(rotation=-45, ha='left')
plt.tight_layout()
plt.savefig('plots/stock_price_transaction2.png', dpi=200, bbox_inches='tight')
plt.show()
#%%
# Portfolio simulation
capital = 1000
shares = 0
portfolio_value = [capital]
trade_count = 0
profit_trades = 0
loss_trades = 0
last_buy_price = 0

for index, row in cross_points.iterrows():
    date = row['Date']
    signal = row['Cross Point']
    price = row['Price']

    if signal == 'Buy':
        if capital > 0:
            shares_to_buy = capital // price
            if shares_to_buy > 0:
                shares += shares_to_buy
                capital -= shares_to_buy * price
                last_buy_price = price
                trade_count += 1
                portfolio_value.append(capital + shares * price)

    elif signal == 'Sell':
        if shares > 0:
            capital += shares * price
            shares = 0
            trade_count += 1
            portfolio_value.append(capital)
            if price > last_buy_price:
                profit_trades += 1
            else:
                loss_trades += 1

# Sell remaining shares
if shares > 0:
    last_price = stock_data.iloc[-1]['Price']
    capital += shares * last_price
    portfolio_value.append(capital)

final_portfolio_value = capital

plt.figure(figsize=(15, 7))
plt.plot(portfolio_value, label='Portfolio Value', color='blue')
plt.xlabel('Transaction Number')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Transactions')
plt.legend()
plt.grid(True)
plt.savefig('plots/portfolio_value.png', dpi=200, bbox_inches='tight')
plt.show()

# Print results
print(f"Final Portfolio Value: {round(final_portfolio_value, 2)}")
print(f"Profitable Trades: {profit_trades}")
print(f"Lossing Trades: {loss_trades}")
print(f"Total Trades: {trade_count}")