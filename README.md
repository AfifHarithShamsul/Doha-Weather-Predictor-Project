Doha-Weather-Predictor-Project
A machine learning model for predicting next-day temperatures using previous weather data.

Project Overview
This project utilises a machine-learning approach to predict the next day's maximum temperature in Doha, Qatar. By leveraging historical climate data from 2021 to 2026, the model identifies patterns in local temperature changes to provide accurate short-term forecasts.

Key Results
- **Model Performance:** The model achieved an Accuracy Score of 0.92**.
- **Accuracy:** This indicates that the model accounts for 92% of the variance in Doha's daily temperature shifts.
- **Visualisation:** A comparison plot was generated to show the high degree of alignment between the "Actual" observed data and the "Predicted" values.

Skills Demonstrated
- **Data Science:** Data cleaning, feature engineering, and handling CSV metadata.
- **Machine Learning:** Implementing Supervised Learning (Linear Regression) via `scikit-learn`.
- **Data Visualisation:** Creating professional analytical plots using `Matplotlib`.
- **Statistical Reasoning:** Evaluating model reliability through training/testing splits.

Tech Stack
- **Language:** Python 3
- **Libraries:** Pandas, Scikit-learn, Matplotlib

How to Run
1. Ensure `doha_weather.csv` is in the same directory as the script.
2. Install dependencies: `pip install pandas scikit-learn matplotlib`
3. Execute the script: `python predictor.py`
