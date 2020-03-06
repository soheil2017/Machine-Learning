from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

df = pd.read_pickle('all_banks')
print(df.head())
print("="*40)

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)

# Bank of America
BAC = data.DataReader("BAC", 'yahoo', start, end)
print(BAC)
print("="*40)

# CitiGroup
C = data.DataReader("C", 'yahoo', start, end)
print(C)
print("="*40)
# Goldman Sachs
GS = data.DataReader("GS", 'yahoo', start, end)
print(GS)
print("="*40)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'yahoo', start, end)
print(JPM)
print("="*40)

# Morgan Stanley
MS = data.DataReader("MS", 'yahoo', start, end)
print(MS)
print("="*40)

# Wells Fargo
WFC = data.DataReader("WFC", 'yahoo', start, end)
print(WFC)
print("="*40)

#** Create a list of the ticker symbols (as strings) in alphabetical order.
# Call this list: tickers**
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
#** Use pd.concat to concatenate the bank dataframes together to a single data frame called bank_stocks.
# Set the keys argument equal to the tickers list. Also pay attention to what axis you concatenate on.**
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)
print(bank_stocks.head())
print("="*40)

#** Set the column name levels (this is filled out for you):**
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
print(bank_stocks.head())
print("="*40)

#** What is the max Close price for each bank's stock throughout the time period?**
#print(bank_stocks.xs(key='Close',axis=1,level='Stock Info').max())
for tick in tickers:
    print(tick, bank_stocks[tick]["Close"].max())
print("="*40)

#** Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock.
returns = pd.DataFrame()

#** We can use pandas pct_change() method on the Close column to create a column representing this return value.
# Create a for loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.**
for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()
print(returns)
print("="*40)

#** Create a pairplot using seaborn of the returns dataframe.
sns.pairplot(returns[1:])
plt.show()

#** Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns. You should notice that 4 of the banks share the same day for the worst drop,
# did anything significant happen that day?**
print(returns.min())
print("="*40)

# Best Single Day Gain
# citigroup stock split in May 2011, but also JPM day after inauguration.
print(returns.max())
print("="*40)

#** Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire time period? Which would you classify as the riskiest for the year 2015?**
print(returns.loc['2015-01-01':'2015-12-31'].std())
# Very similar risk profiles, but Morgan Stanley or BofA
print("="*40)

#** Create a distplot using seaborn of the 2015 returns for Morgan Stanley **
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)
plt.show()

#** Create a distplot using seaborn of the 2008 returns for CitiGroup **
sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)
plt.show()

#** Create a line plot showing Close price for each bank for the entire index of time.
for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()
plt.show()

#** Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008**
plt.figure(figsize=(12,6))
BAC['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()
plt.show()

#** Create a heatmap of the correlation between the stocks Close Price.**
sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
plt.show()

#** Optional: Use seaborn's clustermap to cluster the correlations together:**
sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
plt.show()

