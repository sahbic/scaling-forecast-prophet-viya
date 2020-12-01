from fbprophet import Prophet
import pandas as pd

# init DataFrame
df = pd.DataFrame({'ds': DS, 'y': Y}) 
df.ds = pd.to_timedelta(df.ds, unit='D') + pd.Timestamp('1960-1-1')

# Prophet Fit/Predict
m = Prophet()
m.fit(df.iloc[:(int(NFOR) - int(HORIZON))])
future = m.make_future_dataframe(periods=int(HORIZON))
forecast = m.predict(future)

# Output
PRED = np.array(forecast['yhat'])