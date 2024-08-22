// ML-Model-of-FarmGuide
import pandas as pd
 import numpy as np
 from sklearn.model_selection import train_test_split
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import accuracy_score
 from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 import matplotlib.pyplot as plt
 from sklearn.utils import resample
 import seaborn as sns
 # Sample climate data
 climate_data = pd.DataFrame({
    'date': pd.date_range(start='1/1/2020', periods=100),
    'location': ['Location1'] * 100,
    'max_temp': np.random.randint(20, 40, size=100),
    'min_temp': np.random.randint(10, 20, size=100),
    'humidity': np.random.randint(50, 100, size=100)
 })
 climate_data.to_csv('climate_data.csv', index=False)
 # Sample weather forecast
 weather_forecast = pd.DataFrame({
    'date': pd.date_range(start='1/1/2020', periods=100),
    'location': ['Location1'] * 100,
    'forecast_temp': np.random.randint(20, 40, size=100),
    'forecast_humidity': np.random.randint(50, 100, size=100)
 })
 weather_forecast.to_csv('weather_forecast.csv', index=False)
 # Sample soil conditions
 soil_conditions = pd.DataFrame({
    'location': ['Location1'] * 100,
    'soil_ph': np.random.uniform(5.5, 7.5, size=100),
    'rainfall': np.random.randint(100, 300, size=100),
    'crop_type': np.random.choice(['Wheat', 'Rice', 'Corn'], size=100)
 })
 soil_conditions.to_csv('soil_conditions.csv', index=False)
# Save the sample data to a CSV file
 sample_data = """crop_type,temperature,humidity,yield
 Wheat,20,30,3.5
 Rice,25,70,4.2
 Maize,22,60,3.8
 Pulses,18,40,2.9
 Cotton,28,50,4.0
 """
 with open('sample_crop_data.csv', 'w') as file:
 file.write(sample_data)
 # Load the data into a DataFrame
 data = pd.read_csv('sample_crop_data.csv')
 # Check the data
 print(data.head())
 # Load your data into the 'data' variable
 data = pd.read_csv('sample_crop_data.csv')
 # Define the desired crops
 desired_crops = ['Wheat', 'Rice', 'Maize', 'Pulses', 'Cotton']
 # Filter the data
 data_filtered = data[data['crop_type'].isin(desired_crops)]
 # Check the filtered data
 print(data_filtered.head())
 # If the column name is 'crop_type', update the code
 desired_crops = ['Wheat', 'Rice', 'Maize', 'Pulses', 'Cotton']
 data_filtered = data[data['crop_type'].isin(desired_crops)]
 # Separate each class
 wheat = data_filtered[data_filtered['crop_type'] == 'Wheat']
 rice = data_filtered[data_filtered['crop_type'] == 'Rice']
 maize = data_filtered[data_filtered['crop_type'] == 'Maize']
 pulses = data_filtered[data_filtered['crop_type'] == 'Pulses']
 cotton = data_filtered[data_filtered['crop_type'] == 'Cotton']
 # Check the number of samples for each crop
 print(data_filtered['crop_type'].value_counts())
 # Ensure each class has samples before resampling
 if len(rice) == 0 or len(maize) == 0 or len(pulses) == 0 or len(cotton) == 0:
 print("One or more classes have zero samples. Please check your data.")
 else:
 # Resample the minority classes
 rice_upsampled = resample(rice, replace=True, n_samples=len(wheat), random_state=42)
 maize_upsampled = resample(maize, replace=True, n_samples=len(wheat), random_state=42
 pulses_upsampled = resample(pulses, replace=True, n_samples=len(wheat), random_state=
 cotton_upsampled = resample(cotton, replace=True, n_samples=len(wheat), random_state=
 # Combine the resampled datasets
 data_balanced = pd.concat([wheat, rice_upsampled, maize_upsampled, pulses_upsampled, 
# Check the new distribution
 print(data_balanced['crop_type'].value_counts())
 # Load data
 climate_data = pd.read_csv('climate_data.csv')
 weather_forecast = pd.read_csv('weather_forecast.csv')
 soil_conditions = pd.read_csv('soil_conditions.csv')
 crop_data = pd.read_csv('sample_crop_data.csv')
 # Merge datasets
 data = pd.merge(climate_data, weather_forecast, on=['date', 'location'])
 data = pd.merge(data, soil_conditions, on='location')
 # Preprocess data
 data.fillna(method='ffill', inplace=True)  # Fill missing values
 # Example features
 data['avg_temp'] = (data['max_temp'] + data['min_temp']) / 2
 data['temp_diff'] = data['max_temp'] - data['min_temp']
 data['humidity_level'] = data['humidity'] / 100
 data['crop_recommendation'] = data['crop_type']
 # Example features
 data['avg_temp'] = (data['max_temp'] + data['min_temp']) / 2
 data['temp_diff'] = data['max_temp'] - data['min_temp']
 data['humidity_level'] = data['humidity'] / 100
 # Target variable
 data['crop_recommendation'] = data['crop_type']
 X = data[['avg_temp', 'temp_diff', 'humidity_level', 'soil_ph', 'rainfall']]
 y = data['crop_recommendation']
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 accuracy = accuracy_score(y_test, y_pred)
 print(f'Accuracy: {accuracy * 100:.2f}%')
 new_data = pd.DataFrame({
 'avg_temp': [20],
 'temp_diff': [10],
 'humidity_level': [0.70],
 'soil_ph': [6.5],
 'rainfall': [300]
})
 new_data = pd.DataFrame({
 'avg_temp': [25],
 'temp_diff': [12],
 'humidity_level': [0.65],
 'soil_ph': [6.8],
 'rainfall': [250]
 })
new_data = pd.DataFrame({
    'avg_temp': [30],
    'temp_diff': [15],
    'humidity_level': [0.60],
    'soil_ph': [7.0],
    'rainfall': [200]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}')
 Recommended Crop: Wheat
 #Predicting Crop for Low Soil pH
 new_data = pd.DataFrame({
    'avg_temp': [28],
    'temp_diff': [12],
    'humidity_level': [0.70],
    'soil_ph': [5.5],
    'rainfall': [150]
 })
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 // Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
 # Predicting Crop for High Humidity
 new_data = pd.DataFrame({
    'avg_temp': [26],
    'temp_diff': [10],
    'humidity_level': [0.85],
    'soil_ph': [7.0],
    'rainfall': [300]
 })
 crop_prediction = model.predict(new_data)
 print(f'Recommended Crop: {crop_prediction[0]}'
