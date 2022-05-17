# -*- coding: utf-8 -*-
"""
Created on Apr 6 2022
@author: A.V.Aistov

Some ideas for
<<Финансовые активы: рынки, инструменты, сделки>>,
 автор курса Россохин Владимир Валерьевич
 (e-mail 10 января 2022 г. 10:10).
...соотношение выигрышных и проигрышных сделок...
"""
#%%
import yahoo_fin.stock_info as si
import datetime
import tkinter as tk
from tkinter import filedialog
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import webbrowser
import ta
from ta.utils import dropna
import plotly.graph_objects as go
#%%
'''
Выбирается график: финансовый актив (акция, фьючерс и т.п.);
 таймфрейм (минутный, часовой, дневной и проч. графики).
'''
#
## get data from "Yahoo! Finance" >>>
#
tcr='SNOW'
tcr='AAPL'

start='2019-12-01'
'''
start='2019-12-01'
end = '2020-09-01'
df=si.get_data(tcr, start_date=start, end_date=end, interval='1d')

help(si.get_data)

# Only 7 days worth of 1m (minute) granularity data
#  are allowed to be fetched per request !!! >>>

now = datetime.datetime.now()
now
d = datetime.timedelta(days = 6)
ago = now - d
ago
year = '{:02d}'.format(ago.year)
month = '{:02d}'.format(ago.month)
day = '{:02d}'.format(ago.day)
hour = '{:02d}'.format(ago.hour)
minute = '{:02d}'.format(ago.minute)
day_month_year = '{}-{}-{}'.format(year, month, day)
print('day_month_year: ' + day_month_year)
start = '{}-{}-{}'.format(year, month, day)
df=si.get_data(tcr, start_date=start, interval='1m')
'''
df=si.get_data(tcr, start_date=start, interval='1d')
print('\n', df.columns, '\n\n', df.index)
print('\n', df)
#%%
'''

#
## get data from Excel !!! >>>
#

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
'''
#%%

print('\n', df.isnull().sum())
'''
# drop NaN values if they are >>
df = dropna(df)
'''
#%%
mpf.plot(df, type='candle', volume=True,
         main_panel=0, volume_panel=1, figsize=(12,7),
         show_nontrading=True, title=tcr)
#%%
'''
Выбирается индикатор технического анализа
 (скользящие средние и их комбинации, MACD,
  Momentum, RSI, Stochastic и проч.)
 или индикатор пишется самостоятельно.
'''

# get ideas from "TA_library.py"...

# Technical Analysis Library
url="https://technical-analysis-library-in-python.readthedocs.io/en/latest/"
webbrowser.open(url)
# To use this library you should have
# a financial time series dataset including
# Timestamp, Open, High, Low, Close and Volume columns.
## You should clean or fill NaN values in your dataset
## before add technical analysis features.
#%%

##
### Simple Moving Averages (SMA) >>>
##
ind='SMA'

# Initialize the short and long windows
short = 10
long = 20
'''
short = 6
long = 9
'''
df['fast']=ta.trend.sma_indicator(df.close, window=short)
df['slow']=ta.trend.sma_indicator(df.close, window=long)
#%%

# initialize `signal` column
df['signal'] = 0.0

# create signals
df['signal'] = np.where(df['fast'] > df['slow'], 1.0, 0.0)   

# generate trading orders
df['positions'] = df['signal'].diff()

# print `signals`
print(df['signal'],'\n\n',
      pd.crosstab(df.signal, df.positions))
#%%

# Initialize the plot figure
fig = plt.figure(figsize=(11,7))

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')

# Plot the closing price
df.close.plot(ax=ax1, color='g', lw=2., label="Close")

# Plot the short and long moving averages
df[['fast', 'slow']].plot(ax=ax1, lw=2., title=tcr)

# Plot the buy signals
ax1.plot(df.loc[df.positions == 1.0].index, 
         df.fast[df.positions == 1.0],
         '^', markersize=10, color='m')
         
# Plot the sell signals
ax1.plot(df.loc[df.positions == -1.0].index, 
         df.fast[df.positions == -1.0],
         'v', markersize=10, color='k')

plt.legend(['Close', ind+str(short), ind+str(long)],
           loc='best', fontsize=15)
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
print('\n', df.columns, '\n\n', df.index)
'''
# Set the initial capital
initial_capital= float(100000.0)

# Create a DataFrame `positions`
positions = pd.DataFrame(index=df.index)

# Would you buy *spd* shares a day?
spd = 100
positions[tcr] = spd*df.signal
'''
print('\n',pd.crosstab(df.signal, df.positions))
print('\n',df.groupby(by=['signal']).sum())
'''
  
# Initialize the portfolio with value owned, column *tcr*
# (store the market value of an open position) 
portfolio = positions.multiply(df.adjclose, axis=0)

# Store the difference in shares owned
# (= +- *spd* in days when position was changed)
pos_diff = positions.diff()
'''
print('\n',pd.crosstab(positions[tcr], pos_diff[tcr]))
'''

# `holdings` in portfolio
# .sum(axis=1) --- сумма элементов в строках после умножения
portfolio['holdings'] = (positions.multiply(df.adjclose,
                                            axis=0)).sum(axis=1)

# `cash` in portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(df.adjclose,
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
# visualize the portfolio value  >>>

# Create a figure
fig = plt.figure(figsize=(11,7))

ax1 = fig.add_subplot(111, ylabel='Cash & holdings')

# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2., label=tcr)
plt.legend(loc='upper center', fontsize=15)

ax1.plot(portfolio.loc[df.positions == 1.0].index, 
         portfolio.total[df.positions == 1.0],
         '^', markersize=10, color='m', alpha=0.6)
ax1.plot(portfolio.loc[df.positions == -1.0].index, 
         portfolio.total[df.positions == -1.0],
         'v', markersize=10, color='k', alpha=0.6)
'''
#show="Rc"
show="adjclose"
ax2=fig.add_subplot(111, ylabel=show, frame_on=False)
ax2.yaxis.tick_right()
ax2.xaxis.tick_top()
ax2.yaxis.set_label_position('right')
#ax2.plot(portfolio[show],color='g',alpha=0.8,label=show)
ax2.plot(df[show], color='g', alpha=0.8, label=show)
plt.legend(loc='upper left', fontsize=15)
'''
#%%
#
## calculate количество выигрышных и проигрышных сделок >>
#
for day in df.index:
  print(day, end=" ")
#%%
win=0
lose=0
for day in df.index:
  if pos_diff[tcr].loc[day]>0:
    tot1=portfolio.total.loc[day]
    print('\nBuy', day, "->", tot1)
  elif pos_diff[tcr].loc[day]<0:
    tot2=portfolio.total.loc[day]
    if tot2>tot1:
      win+=1
    elif tot2<tot1:
      lose+=1
    print('\nSell', day, "->", tot2, 'win =', win, 'lose =',lose)
#%%
'''
Индикатор тестируется на исторических данных.
 При этом происходит перебор всех параметров
 выбранных индикаторов с определенным шагом.
 То есть не просто периоды, к примеру,
 двух скользящих средних 10/20 проверить.
 А перебрать период первой, к примеру, от 5 до 50 с шагом 1,
 перебирая период второй скользящей средней от 10 до 100,
 к примеру, с шагом 2. Шаг хорошо бы тоже установить.
'''

## calculate index range >>
'''
i=0
for short in range(5, 10, 1):
  print('\n',short)
  for long in range(short+3, 10, 2):
    print(long, end=' ')
    i+=1
'''
i=0
for short in range(5, 51, 1):
  print('\n',short)
  for long in range(short+3, 101, 2):
    print(long, end=' ')
    i+=1

## create DataFrame >>
dft = pd.DataFrame(index=range(i))
dft['short']=0
dft['long']=0
dft['total']=0
dft['win']=0
dft['lose']=0
#%%
i=0
'''
for short in range(5, 10, 1):
  for long in range(short+3, 10, 2):
'''
for short in range(5, 51, 1):
  for long in range(short+3, 101, 2):
#    dft['short'].loc[i]=short
#    dft['long'].loc[i]=long
    dft.iloc[i,0]=short
    dft.iloc[i,1]=long

    df['fast']=ta.trend.sma_indicator(df.close, window=short)
    df['slow']=ta.trend.sma_indicator(df.close, window=long)
    # initialize `signal` column
    df['signal'] = 0.0
    # create signals
    df['signal'] = np.where(df['fast'] > df['slow'], 1.0, 0.0)   
    # generate trading orders
    # Create a DataFrame `positions`
    positions = pd.DataFrame(index=df.index)
    positions[tcr] = spd*df.signal
    # Initialize the portfolio with value owned, column *tcr*
    # (store the market value of an open position) 
    portfolio = positions.multiply(df.adjclose, axis=0)
    # Store the difference in shares owned
    # (= +- *spd* in days when position was changed)
    pos_diff = positions.diff()
    # `holdings` in portfolio
    # .sum(axis=1) --- сумма элементов в строках после умножения
    portfolio['holdings'] = (positions.multiply(df.adjclose,
                                            axis=0)).sum(axis=1)
    # `cash` in portfolio
    portfolio['cash'] = initial_capital - (pos_diff.multiply(df.adjclose,
                                                         axis=0)).sum(axis=1).cumsum()   
    # `total` portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
#    dft['total'].loc[i]=portfolio['total'].iloc[-1]
#    dft['total'].loc[i]=portfolio.iloc[-1,-1]
    dft.iloc[i,2]=portfolio.iloc[-1,-1]

    win=0
    lose=0
    tot1=0
    tot2=0
    for day in df.index:
#      print('\n',day, pos_diff.loc[day,tcr], end=" ")
      if pos_diff.loc[day,tcr]>0:
        tot1=portfolio.loc[day,'total']
 #       print('\nBuy',day, "->", tot1)
      elif pos_diff.loc[day,tcr]<0:
        tot2=portfolio.loc[day,'total']
#        print('\nSell',day, "->", tot2)
        if tot2>tot1:
          win+=1
        elif tot2<tot1:
          lose+=1
  #      print(day, "->", tot2, 'win =', win, 'lose =',lose)
#    dft['win'].loc[i]=win
#    dft['lose'].loc[i]=lose
    dft.iloc[i,3]=win
    dft.iloc[i,4]=lose

    i+=1

    print('>>> Windows ',short,':',long,
          ' ->', portfolio.iloc[-1,-1],
          ' win =', win, 'lose =',lose)
#%%

## profit >>
dft['profit'] = dft['total'] / initial_capital

## win to lose ratio >>
dft['win_to_lose'] = dft['win'] / dft['lose']

#%%
fig = plt.figure(figsize=(9,6))
dft['total'].plot()
#%%
fig = plt.figure(figsize=(9,6))
dft['win_to_lose'].plot()
#%%
#
## run if it was not before  >>
# (for saving data to file)

# initializing tcl/tk interpreter >>
window = tk.Tk()
# open filedialog window on top of other windows >>
window.wm_attributes('-topmost', 1)
# this will close empty tk-window after filedialog >>
window.withdraw()

#%%
## idea from "make_finance_panel.py" >>
def save_xls(dfall, case):
    if case=="xlsx":
      print('\nSaving data in\n', SfileName)
      dfall.to_excel(SfileName, index=False)
    elif case=="xls":
      print('\nSaving data in\n', SfileName)
      dfall.to_excel(SfileName.replace(".xls", ".xlsx"),
                     index=False)
    else:
      print('\nSaving data in\n', SfileName + ".xlsx")
      dfall.to_excel(SfileName + ".xlsx", index=False)
#%%
#
## Saving data to file >>
#
SfileName = filedialog.asksaveasfilename(title="Select file",
                    filetypes=(("Excel files",
                                "*.xlsx *.xls"),
                               ("All files", "*.*")),
                    parent=window)

## nice work with file extension >>
save_xls(dft, SfileName.split('.')[-1])
'''
## simple run instead of "save_xls" >>
print('\nSaving data in\n', SfileName + ".xlsx")
dft.to_excel(SfileName + ".xlsx")
'''
#%%
#
## 3D plots >>
#

fig = go.Figure( data=[go.Mesh3d(x=dft['short'],
                   y=dft['long'],
                   z=dft['profit'],
                   opacity=0.7, color='royalblue'
                  )] )

fig.update_layout( scene = dict(
                    xaxis_title='Short window',
                    yaxis_title='Long window',
                    zaxis_title='Profit'),
  title=tcr )


# save figure on disk >>
fig.write_html(SfileName.split('.')[:-1][0] + "_my.html")

# open file in webbrowser >>
webbrowser.open(SfileName.split('.')[:-1][0] + "_my.html")

#%%
# Could you make plots more informative and colorful?
url="https://plotly.com/python/3d-axes/"
webbrowser.open(url)

#%%
'''
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot(dft['short'], dft['long'],
                       dft['profit'])
'''
#%%
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(dft['short'], dft['long'],
                       dft['profit'])
#%%

#
## find max profit >>>
#
dft['total'].max()

ind=dft.index[ dft['total']==dft['total'].max() ][0]
short=dft.short[ind]
long=dft.long[ind]

print('>>> Windows', short,' :', long,
      '  ->',dft.total[ind],
          ' win =', dft.win[ind], 'lose =',dft.lose[ind])
#%%
## trading with max profit >>>

    df['fast']=ta.trend.sma_indicator(df.close, window=short)
    df['slow']=ta.trend.sma_indicator(df.close, window=long)
    # initialize `signal` column
    df['signal'] = 0.0
    # create signals
    df['signal'] = np.where(df['fast'] > df['slow'], 1.0, 0.0)   
    # generate trading orders
    df['positions'] = df['signal'].diff()    
    # Create a DataFrame `positions`
    positions = pd.DataFrame(index=df.index)
    positions[tcr] = spd*df.signal
    # Initialize the portfolio with value owned, column *tcr*
    # (store the market value of an open position) 
    portfolio = positions.multiply(df.adjclose, axis=0)
    # Store the difference in shares owned
    # (= +- *spd* in days when position was changed)
    pos_diff = positions.diff()
    # `holdings` in portfolio
    # .sum(axis=1) --- сумма элементов в строках после умножения
    portfolio['holdings'] = (positions.multiply(df.adjclose,
                                            axis=0)).sum(axis=1)
    # `cash` in portfolio
    portfolio['cash'] = initial_capital - (pos_diff.multiply(df.adjclose,
                                                         axis=0)).sum(axis=1).cumsum()   
    # `total` portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    
    print('>>> Windows ',short,' :',long,
          ' ->', portfolio.iloc[-1,-1],
          ' win =', dft.win[ind], 'lose =',dft.lose[ind])
#%%
fig = plt.figure(figsize=(9,6))
portfolio['total'].plot()
plt.plot(portfolio.loc[df.positions == 1.0].index, 
         portfolio.total[df.positions == 1.0],
         '^', markersize=7, color='m', alpha=0.6)
plt.plot(portfolio.loc[df.positions == -1.0].index, 
         portfolio.total[df.positions == -1.0],
         'v', markersize=7, color='k', alpha=0.6)

print('\n',pd.crosstab(df.signal, df.positions),
  '\n\n>>> Windows ',short,':',long,
          ' ->', dft.total[ind],
          ' win =', dft.win[ind], 'lose =',dft.lose[ind])
#%%

#
## benchmarking
#

tcrspy="SPY"
dfspy=si.get_data(tcrspy, interval='1d', start_date=start)
print('\n', dfspy.columns, '\n\n', dfspy.index)

#%%

dfspy['R'] = dfspy.adjclose.pct_change()
# cumulative returns >>
dfspy["Rc"] = (dfspy.R + 1).cumprod()

ind = portfolio[portfolio['holdings']>0].index.min()
invest = portfolio['holdings'].loc[ind]
print('\nFirst purchase was ', invest,' in', ind)

dfspy["tot_spy"] = initial_capital-invest + invest*dfspy["Rc"]
dfspy.iloc[0,-1] = initial_capital

print('\n', dfspy)
#%%
print('\n>>> SMA windows ',short,' :',long)

# Cash & trading vs SPY investment >>>

fig = plt.figure(figsize=(9,6))

ax1 = fig.add_subplot(111, ylabel='Cash & holdings')

portfolio['total'].plot(ax=ax1, lw=2., alpha=0.3, label=tcr)

ax1.plot(portfolio.loc[df.positions == 1.0].index, 
         portfolio.total[df.positions == 1.0],
         '^', markersize=10, color='m', alpha=0.7)
ax1.plot(portfolio.loc[df.positions == -1.0].index, 
         portfolio.total[df.positions == -1.0],
         'v', markersize=10, color='k', alpha=0.7)

ax1.plot(dfspy["tot_spy"], color='orange', alpha=0.8,
         label=tcrspy)
plt.legend(loc='best', fontsize=15)
