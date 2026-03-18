import fastf1
import pandas as pd
import os

# Initialize cache and data collection
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")
allModelData = []

for round in range(1, 25):
    # Load session and lap data
    raceSession = fastf1.get_session(2025, round, 'R')
    raceSession.load()
    lapsData = raceSession.laps

    # Extract required lap columns
    columnsToKeep = [
        'Driver', 'DriverNumber', 'Position',
        'Time', 'LapTime', 'LapNumber', 'LapStartTime', 'Compound', 
        'TyreLife', 'Stint', 'TrackStatus', 'IsAccurate'
    ]
    modelData = lapsData[columnsToKeep].copy()
    modelData['EventName'] = raceSession.event['EventName']

    # Filter invalid data and engineer features
    modelData = modelData[modelData['IsAccurate'] == True]
    modelData = modelData.dropna(subset=['LapTime'])
    modelData = modelData[modelData['TrackStatus'] == '1']
    modelData['LapTimeSeconds'] = modelData['LapTime'].dt.total_seconds()
    modelData['PitStopTarget'] = (modelData['Stint'] > modelData['Stint'].shift(1)).astype(int)

    # Calculate gap to car ahead for dirty air impact
    # Sort by LapNumber and LapStartTime to determine the actual on-track order
    modelData = modelData.sort_values(by=['LapNumber', 'LapStartTime'])
    
    # Get the time the car crossed the line
    modelData['CarAheadLapStartTime'] = modelData.groupby('LapNumber')['LapStartTime'].shift(1)
    
    # Calculate gap in seconds
    modelData['GapToCarAhead'] = (modelData['LapStartTime'] - modelData['CarAheadLapStartTime']).dt.total_seconds()
    
    # Fill Large number for Clean Air
    modelData['GapToCarAhead'] = modelData['GapToCarAhead'].fillna(999.0)
    
    # Mark as in dirty air if gap is less than 2 seconds
    modelData['InDirtyAir'] = (modelData['GapToCarAhead'] < 2.0).astype(int)

    # Convert tyre compounds to numeric values
    compoundMapping = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 4, 'WET': 5}
    modelData['CompoundNumeric'] = modelData['Compound'].map(compoundMapping)

    # Clean and structure weather data
    weatherData = raceSession.weather_data.copy()
    requiredWeatherColumns = ['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'Rainfall']
    weatherData = weatherData[requiredWeatherColumns]

    # Sort and align weather with lap times
    modelData = modelData.sort_values(by='Time')
    weatherData = weatherData.sort_values(by='Time')
    modelData = pd.merge_asof(modelData, weatherData, on='Time', direction='nearest')

    allModelData.append(modelData)

# Combine all races into a final dataset and encode variants
finalSeasonData = pd.concat(allModelData, ignore_index=True)

# Ensure all tyre columns exist even if a specific tyre wasn't used in the loaded races
expected_tyre_cols = ['Tyre_SOFT', 'Tyre_MEDIUM', 'Tyre_HARD', 'Tyre_INTERMEDIATE', 'Tyre_WET']
for tyre_col in expected_tyre_cols:
    if tyre_col not in finalSeasonData.columns:
        finalSeasonData[tyre_col] = False

# Save to file
finalSeasonData.to_parquet('f12025Data.parquet')
print(finalSeasonData.head())
