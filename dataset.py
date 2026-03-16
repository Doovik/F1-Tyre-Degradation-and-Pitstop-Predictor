import fastf1
import pandas as pd
import os

# 1. Setup cache on your PC
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")
allModelData = []
# 2. Load the race session
for round in range(1, 21):
    raceSession = fastf1.get_session(2025, round, 'R')
    raceSession.load()

    # 3. Extract the laps data
    lapsData = raceSession.laps

    # 4. Select the required columns
    columnsToKeep = [
        'Driver', 'DriverNumber', 'Position',
        'Time','LapTime', 'LapNumber', 'Compound', 
        'TyreLife', 'Stint', 'TrackStatus', 'IsAccurate'
    ]
    modelData = lapsData[columnsToKeep].copy()
    # Add additonal column for Race name
    modelData['EventName'] = raceSession.event['EventName']
    # 5. Clean the data
    # Keep only accurate laps and remove empty values
    modelData = modelData[modelData['IsAccurate'] == True]
    modelData = modelData.dropna(subset=['LapTime'])

    # Keep only racing laps (TrackStatus '1' means clear track)
    modelData = modelData[modelData['TrackStatus'] == '1']

    # Convert LapTime format to total seconds for the model
    modelData['LapTimeSeconds'] = modelData['LapTime'].dt.total_seconds()

    # 6. Create the target variable for pit stop prediction
    modelData['PitStopTarget'] = (modelData['Stint'] > modelData['Stint'].shift(1)).astype(int)

    # Map tyre compounds to numerical values
    compoundMapping = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 4, 'WET': 5}
    modelData['CompoundNumeric'] = modelData['Compound'].map(compoundMapping)

    allModelData.append(modelData)

finalSeasonData = pd.concat(allModelData, ignore_index=True)
finalSeasonData = pd.get_dummies(finalSeasonData, columns=['Compound'], prefix='Tyre')
    # Check the results
finalSeasonData.to_parquet('f12025Data.parquet')    
print(finalSeasonData.head())

#modelData['Time'] = modelData['Time']
#modelData[`AirTemperature`] = raceSession.weather_data['AirTemperature']
#modelData[`TrackTemperature`] = raceSession.weather_data['TrackTemperature']
#modelData[`Humidity`] = raceSession.weather_data['Humidity']
#modelData['AirPressure'] = raceSession.weather_data['Pressure']
#modelData['Rainfall'] = raceSession.weather_data['Rainfall']

#To Do
# 1. Add weather data to the model dataset
# 2. Line up time of weather data with lap times (weather data is recorded every 10 minutes, so we need to find the closest weather data point for each lap time) use merge_asof()
