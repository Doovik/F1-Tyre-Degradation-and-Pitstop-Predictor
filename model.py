import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

data = pd.read_parquet('f12025Data.parquet')

data['DriverNumber'] = data['DriverNumber'].astype(int)
data = pd.get_dummies(data, columns=['EventName', 'Driver'])

colsToDrop = [
    'Time', 
    'LapTime', 
    'LapStartTime', 
    'CarAheadLapStartTime', 
    'LapTimeSeconds', 
    'IsAccurate', 
    'TrackStatus', 
    'DriverNumber',
    'Compound'
]
features = [col for col in data.columns if col not in colsToDrop]

target = 'LapTimeSeconds'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train) 
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae:.3f} seconds")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} seconds")
print(f"R-squared (R2): {r2:.3f}")

xgb.plot_importance(model, max_num_features=10)
plt.show()