# Ready-made model

A VayuVaani model already trained on about 2 years of real Bengaluru data
(OpenAQ CPCB/KSPCB stations, Open-Meteo weather and NASA FIRMS fires), so a new
computer can start forecasting in about a minute instead of downloading and
training for 20 minutes or more.

The app uses it when you choose **Use real Bengaluru data** on a computer that
has no model yet. You still need a free OpenAQ key for the live readings.
To train your own instead, use **Rebuild with real data** in Settings.

- `forecaster.joblib`: the trained LightGBM quantile model
- `evaluation.json`, `test_predictions.csv.gz`: its test results (Accuracy page)
- `stations.csv`: the monitoring stations it was trained on
