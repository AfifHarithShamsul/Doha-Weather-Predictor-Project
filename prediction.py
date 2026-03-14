
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("doha_weather.csv", skiprows=3)


df = df.rename(columns={
    "time": "date",
    "temperature_2m_max (°C)": "temperature_2m_max",
    "temperature_2m_mean (°C)": "temperature_2m_min"
})


df = df[["date", "temperature_2m_max", "temperature_2m_min"]].dropna()
df["next_day_temp"] = df["temperature_2m_max"].shift(-1)
df = df.dropna()


X = df[["temperature_2m_max", "temperature_2m_min"]]
y = df["next_day_temp"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model accuracy score: {model.score(X_test, y_test):.2f}")


predictions = model.predict(X_test)
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:50], label="Actual Max Temp", color='blue')
plt.plot(predictions[:50], label="Predicted Max Temp", color='orange', linestyle='--')
plt.title("Doha Temperature Prediction (Daily)")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()



