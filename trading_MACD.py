# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 08:08:52 2022

@author: user
"""
#%%
import yahoo_fin.stock_info as si
import datetime
import tkinter as tk
from tkinter import filedialog
import mplfinance as mpf
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots as ms
import pandas as pd
import numpy as np
import webbrowser
import ta
from ta.utils import dropna

#%%

tcr='RUAL'

#%%

## get data from Excel !!! >>>

# initializing tcl/tk interpreter >>
window = tk.Tk()
# open filedialog window on top of other windows >>
window.wm_attributes('-topmost', 1)
# this will close empty tk-window after filedialog >>
window.withdraw()
#%%

# get file name & read data >>
fileName = filedialog.askopenfilename(title="Select file",
                    filetypes=(("Excel files", "*.xlsx"),
                               ("All files", "*.*")),
                    parent=window)
df = pd.read_excel(fileName)
print('\n', df)
# set column as the dataframe index >>
df = df.set_index(df.iloc[:, 0])
# delete the column >>
df.drop('Unnamed: 0', axis=1, inplace=True)
# delete the index name which was 'Unnamed: 0' >>
df.index.name = None
print('\n', df)

#%%

print('\n', df.isnull().sum())

#drop NaN values if they are >
df = dropna(df)

#%%
mpf.plot(df, type='candle', volume=True,
         main_panel=0, volume_panel=1, figsize=(12,7),
         show_nontrading=True, title=tcr)

#%%
##
### Moving Average Convergence Divergence (MACD) >>>
##
ind='MACD'
df

# Initialize the short and long windows
long = 26
short_m = round(long*12/26)
sign_m = round(long*9/26)

df['MACD']=ta.trend.macd_diff(df.close, window_slow = long, window_fast = short_m, window_sign = sign_m,
                         fillna=False)

#%%
# plot >>>
fig = plt.figure(figsize=(11,12))
ax1 = fig.add_subplot(211, ylabel='Price in $')
df.close.plot(ax=ax1, color='g', lw=2., title=tcr)
ax2 = fig.add_subplot(212,  ylabel='MACD')
df[['MACD']].plot(ax=ax2, lw=2., title=tcr)

plt.legend(['Close', ind+str(long)],
           fontsize='large')
plt.tight_layout()
#%%
# initialize `signal` column
df['signal'] = 0.0

#%%
df['signal'][long:] = np.where((df['MACD'][long:] > 0), 1.0, 0.0)
print(df['signal'])
'''
df['signal_2_m'][long:] = np.where((df['MACD'][long:] < 0), 1.0, 0.0)
print(df['signal_2_m'])
'''
#%%
df['positions'] = df['signal'].diff()
'''
df['signal_sell_m'] = df['signal_2_m'].diff()
print(df['signal_buy_m'],df['signal_sell_m'])

#%%
for index, item in enumerate(df['signal_buy_m']):
    if item < 1:
        df['signal_buy_m'][index] = 0
print(df['signal_buy_m'])
#%%

for index, item in enumerate(df['signal_sell_m']):
    if item > -1:
        df['signal_sell_m'][index] = 0
print(df['signal_sell_m'])

#%%
# generate trading orders
df['positions'] = df['signal_buy_m'] 

#%%
df['signal'] = 0.0
df['signal']=(df.positions).cumsum()
'''
#%%
# print `signals`
print(df['signal'],'\n\n',
      pd.crosstab(df.signal, df.positions))

#%%
s = df['signal'].index.min()
e = df['signal'].index.max()
print('\nTime period to plot >>>\n', s,'\n', e)

sp = df['signal'].loc[s:e,]
dfsp = df.loc[s:e,]

#%%

# Initialize the plot figure
fig = plt.figure(figsize=(11,15))

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(311,  ylabel='Price in $')

# Plot the closing price
df.close.plot(ax=ax1, color='g', lw=2., label="Close")
ax2 = fig.add_subplot(312,  ylabel='MACD')
df[['MACD']].plot(ax=ax2, lw=2., title=tcr)

# Plot the short and long moving averages

# Plot the buy signals
ax3=ax2.twinx()
#ax4 = fig.add_subplot(311, ylabel = 'CCI')
ax3.plot(df.loc[df.positions == 1.0].index, 
         df.MACD[df.positions == 1.0],
         '^', markersize=10, color='m')
         
# Plot the sell signals
ax3.plot(df.loc[df.positions == -1.0].index, 
         df.MACD[df.positions == -1.0],
         'v', markersize=10, color='k')

#ax1.grid()

plt.legend(['Close', ind+str(long)],
           loc='best', fontsize=15)

#%%
fig2=plt.figure(figsize=(11,7))

ax5 = fig2.add_subplot(111,  ylabel='Price in $')
df.close.plot(ax=ax5, color='g', lw=2.)

# Plot the buy signals
ax6=ax5.twinx()
ax6.plot(df.loc[df.positions == 1.0].index, 
         df.MACD[df.positions == 1.0],
         '^', markersize=10, color='m')
         
# Plot the sell signals
ax6.plot(df.loc[df.positions == -1.0].index, 
         df.MACD[df.positions == -1.0],
         'v', markersize=10, color='k')

plt.legend(['Close', ind+str(long)],
           loc='best', fontsize=15)
#%%
#%%
##
### Backtesting The Trading Strategy
##
# (calculate performance)
# ...
#
## Simple Backtester
#
'''
print('\n', dfsp.columns, '\n\n', df.index)
'''
# Set the initial capital
initial_capital= float(100000.0)

# Create a DataFrame `positions`
positions = pd.DataFrame(index=df.index).fillna(0.0)

# Would you buy *spd* shares a day?
spd = 300
positions[tcr] = spd*df.signal
'''
print('\n',pd.crosstab(df.signal, df.positions))
print('\n',df.groupby(by=['signal']).sum())
'''
  
# Initialize the portfolio with value owned, column *tcr*
# (store the market value of an open position) 
portfolio = positions.multiply(df.close, axis=0)

# Store the difference in shares owned
# (= +- *spd* in days when position was changed)
pos_diff = positions.diff()
'''
print('\n',pd.crosstab(positions[tcr], pos_diff[tcr]))
'''

# `holdings` in portfolio
# .sum(axis=1) --- сумма элементов в строках после умножения
portfolio['holdings'] = (positions.multiply(df.close,
                                            axis=0)).sum(axis=1)

# `cash` in portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(df.close,
                                                         axis=0)).sum(axis=1).cumsum()   

# `total` portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# `returns` of portfolio
portfolio['R'] = portfolio['total'].pct_change()
# cumulative returns >>
portfolio["Rc"] = (portfolio.R + 1).cumprod()

# Print the first lines of `portfolio`
'''
print(portfolio)
'''
#%%
# visualize the portfolio value & returns >>>

# Create a figure
fig = plt.figure(figsize=(11,7))

ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2., label=tcr)
plt.legend(loc='upper center', fontsize=15)

ax1.plot(portfolio.loc[df.positions == 1.0].index, 
         portfolio.total[df.positions == 1.0],
         '^', markersize=10, color='m', alpha=0.6)
ax1.plot(portfolio.loc[df.positions == -1.0].index, 
         portfolio.total[df.positions == -1.0],
         'v', markersize=10, color='k', alpha=0.6)
show="Rc"
show="close"
ax2=fig.add_subplot(111, ylabel=show, frame_on=False)
ax2.yaxis.tick_right()
ax2.xaxis.tick_top()
ax2.yaxis.set_label_position('right')
#ax2.plot(portfolio[show],color='g',alpha=0.8,label=show)
ax2.plot(df[show], color='g', alpha=0.8, label=show)
plt.legend(loc='upper left', fontsize=15)

#%%
# range(start, stop, step)
for long in range(5, 101, 1):
    for short_m in range (1, 55, 1):
        for sign_m in range (1, 31, 1):
        
            print('\n',long, short_m, sign_m)
         
#%%

i=0
for long in range(5, 101, 1):
    for short_m in range (1, 55, 1):
        for sign_m in range (1, 31, 1):
            print('\n',long, short_m, sign_m)
            i+=1

## create DataFrame >>
dft = pd.DataFrame(index=range(i))
dft['long']=0
dft['total']=0
dft['short_m'] = round(dft['long']*12/26)
dft['sign_m'] = round(dft['long']*9/26)
#%%
i=0
for long in range(5, 101, 1):
    for short_m in range (1, 55, 1):
        for sign_m in range (1, 31, 1):
            dft['long'].loc[i]=long
            dft['short_m'].loc[i]=short_m
            dft['sign_m'].loc[i]=sign_m
            dft.iloc[i,1]=long
            dft.iloc[i,1]=short_m
            dft.iloc[i,1]=sign_m
        
            df['MACD']=ta.trend.macd_diff(df.close, window_slow = long, window_fast = short_m, window_sign = sign_m,
                                     fillna=False)
        
            # initialize `signal` column
            df['signal'] = 0.0
        
            df['signal'][long:] = np.where((df['MACD'][long:] > 0), 1.0, 0.0)
            print(df['signal'])
            
            df['positions'] = df['signal'].diff()
            
            # Create a DataFrame `positions`
            positions = pd.DataFrame(index=df.index).fillna(0.0)
        
            # Would you buy *spd* shares a day?
            spd = 300
            positions[tcr] = spd*df.signal
            
            # Initialize the portfolio with value owned, column *tcr*
            # (store the market value of an open position) 
            portfolio = positions.multiply(df.close, axis=0)
        
            # Store the difference in shares owned
            # (= +- *spd* in days when position was changed)
            pos_diff = positions.diff()
            
            # `holdings` in portfolio
            # .sum(axis=1) --- сумма элементов в строках после умножения
            portfolio['holdings'] = (positions.multiply(df.close,
                                                        axis=0)).sum(axis=1)
        
            # `cash` in portfolio
            portfolio['cash'] = initial_capital - (pos_diff.multiply(df.close,
                                                                     axis=0)).sum(axis=1).cumsum()   
        
            # `total` portfolio
            portfolio['total'] = portfolio['cash'] + portfolio['holdings']
            dft['total'].loc[i]=portfolio['total'].iloc[-1]
            i+=1
            
            print('>>> Window ',long, short_m, sign_m,
                  '  ->', portfolio['total'].iloc[-1])

#%%    
fig = plt.figure(figsize=(9,6))
dft['total'].plot()
#%%

#
## find max profit >>>
#
dft['total'].max()

ind=dft.index[ dft['total']==dft['total'].max() ][0]
long=dft.long[ind]


print('>>> Windows', long,
      '  ->',dft.total[ind])