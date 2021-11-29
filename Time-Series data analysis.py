# ARIMA: Autoregressive Integrated Moving Averages
# SARIMA: Seasonal Autoregressive Integrated Moving Averages


"""
Steps:
    1. Visualize the Time Series Data
    2. Make the time series data stationary (if not)
    3. Plot the correlation and autocorrleation charts
    4. Construct the ARIMA model or Seasonal ARIMA based on the data
    5. Use the model to make predictions
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/asus/Desktop/time_series.csv')
df = df.dropna()
df.columns=["Month","Sales"]
df['Month']=pd.to_datetime(df['Month'])
df.set_index('Month',inplace=True)
#print (df.describe())


#step2: visualize
#print (df.plot())

from statsmodels.tsa.stattools import adfuller

test_result=adfuller(df['Sales'])


"""

adfuller gives five different values as below:  
'ADF Test Statistic','p-value','#Lags Used','Number of Observations Used'

This test is hypothesis testing where null hypothesis is "data is not stationary" 
and alternate hypothesis states that it is stationary

We will reject null hypothesis based on the p-value (Ref below function)


"""


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


#adfuller_test(df['Sales'])

#based on above result, data is not stationary and hence making the data stationary using Differencing


df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12) # 12 for 12 months

adfuller_test(df['Seasonal First Difference'].dropna())

df = df.dropna()

# from above u can see that data is stationary now
df['Seasonal First Difference'].plot()

"""
Now to predict today's data we need to find out how many previous day's data we need to take
which can be done using autoregressive model. We use auto-correlation and partial auto-correlation

We have to give 3 values to our ARIMA model
p: AR model lags  point from where value next to this point suddenly becomes zero in partial autocorrelation.
d: differencing  : No of times we performed shifted (in our case we did it only once the shift 12 operation)
q: Moving avergae lags  : see the point till which value is decreasing exponentially in Autocorrelation graph

AR model is best done with PACF

"""
import statsmodels.api 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = statsmodels.graphics.tsaplots.plot_acf(df['Seasonal First Difference'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = statsmodels.graphics.tsaplots.plot_pacf(df['Seasonal First Difference'],lags=40,ax=ax2)

model=statsmodels.api.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
#here order has values of p,d,q and seasonal order has first three values as p,d,q and last value is shift we made
results=model.fit()
print (df)

df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))

print (df)

from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]

print (future_dates)
# MAPE mean absolute percent error


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

