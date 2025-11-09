# streamlit_weather_city_fixed.py
import streamlit as st
import requests
import pandas as pd

# Function to get lat/lon from city name using Nominatim (OpenStreetMap)
def geocode_city(city_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city_name,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "weather-app-example (yfazul@gmail.com)"  # required by Nominatim
    }
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    result = r.json()
    if result:
        return float(result[0]["lat"]), float(result[0]["lon"])
    else:
        return None, None

# Function to fetch weather data from Open-Meteo
def get_open_meteo(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,apparent_temperature,windspeed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Singapore"
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# Streamlit UI
st.title("City-based Weather Forecast 🌤️")
st.markdown("Enter a city name to get hourly and daily weather forecasts.")

city = st.text_input("City Name", value="Singapore")

if st.button("Get Weather Forecast"):
    if city.strip() == "":
        st.warning("Please enter a city name.")
    else:
        lat, lon = geocode_city(city)
        if lat is None:
            st.error("City not found. Please check the spelling or try another city.")
        else:
            st.success(f"Coordinates for {city}: Latitude {lat}, Longitude {lon}")
            try:
                data = get_open_meteo(lat, lon)

                # Hourly forecast
                st.subheader("Hourly Forecast (Next 24 hours)")
                hourly_df = pd.DataFrame(data["hourly"])
                hourly_df['time'] = pd.to_datetime(hourly_df['time'])
                st.dataframe(hourly_df.head(24))

                # Daily forecast
                st.subheader("Daily Forecast")
                daily_df = pd.DataFrame(data["daily"])
                daily_df['time'] = pd.to_datetime(daily_df['time'])
                st.dataframe(daily_df)

                # Charts
                st.subheader("Hourly Temperature Chart")
                st.line_chart(hourly_df.set_index('time')['temperature_2m'])

                st.subheader("Daily Max/Min Temperature Chart")
                st.line_chart(daily_df.set_index('time')[['temperature_2m_max', 'temperature_2m_min']])

            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching weather data: {e}")
