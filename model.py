import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

# Read Data File
data = pd.read_parquet('f12025Data.parquet')

# One Hot Encode Categorical Variables for Context
data = pd.get_dummies(data, columns=['EventName', 'Driver'])

# Columns to drop for modeling
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
# Columns to keep for modeling
features = [col for col in data.columns if col not in colsToDrop]

# Target variable
target = 'LapTimeSeconds'

# Prepare data for modeling
X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train) 
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae:.3f} seconds")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} seconds")
print(f"R-squared (R2): {r2:.3f}")

# Plot feature importance
xgb.plot_importance(model, max_num_features=10)
plt.show()