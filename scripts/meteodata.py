import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url = "https://archive-api.open-meteo.com/v1/era5"

params = {
	"latitude": 46.0572,
	"longitude": 14.5119,
	"hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
    "start_date": "2022-09-01",
    "end_date": "2025-05-19",
}
responses = openmeteo.weather_api(url, params=params)

response = responses[0]

print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()

hourly_data = {"timestamp": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit="s", utc=True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
	freq = pd.Timedelta(seconds=hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature"] = hourly_temperature_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["wind_speed"] = hourly_wind_speed_10m

df = pd.DataFrame(data=hourly_data)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%MZ")

df.to_csv("./data/weather_data.csv", index=False)