# VayuVaani

Ward-level air quality forecasting and multilingual health alerts for Bengaluru.
Societal Based Mini Project (24CSE57), New Horizon College of Engineering.

It downloads pollution, weather, fire and event data, cleans it, and forecasts
PM2.5 for every monitoring station for the next 48 hours, with a confidence range
and the chance of "Poor or worse" air. It explains what is driving each forecast,
suggests the cleanest hours to be outside, and previews SMS and voice alerts in
English, Kannada and Hindi.

## Easiest way: the app (Windows)

Double-click `VayuVaani.bat`. The first time, it sets everything up (3 to 10 minutes, needs internet) and adds a VayuVaani shortcut to your desktop. After that, the shortcut opens VayuVaani in its own window. Closing the window stops the app.

In the app, choose **Try it with demo data** or **Use real Bengaluru data** (paste your free OpenAQ key in Settings first). Everything else is buttons.

**Use real Bengaluru data** starts from the ready-made model in `prebuilt/` (trained on about 2 years of real data), so a new computer is forecasting in about a minute. To train your own model on fresh data, use **Rebuild with real data** in Settings (20 minutes or more). On slower computers, set **Animations** to **Light** in Settings; the app also picks this by itself if the screen can't keep up.

Pages: Today's air, Stations, Plan outdoors, Alerts (SMS and voice in English, Kannada and Hindi), Accuracy and About. Settings also has a dark mode and an optional second language (Kannada or Hindi) for the menu labels.

## On every device

| Device | How |
|---|---|
| **Windows** | Double-click `VayuVaani.bat`. |
| **macOS** | Double-click `VayuVaani.command`. The first time, macOS may say it's from an unidentified developer: right-click it, choose **Open**, then **Open** again. Needs Python 3.10–3.13 ([python.org](https://www.python.org/downloads/macos/) or `brew install python@3.12`); the launcher installs OpenMP for the model with Homebrew if needed. |
| **Linux** | Run `./vayuvaani.sh` (or `bash vayuvaani.sh`). Needs Python 3.10–3.13 with venv (`sudo apt install python3 python3-venv` on Ubuntu/Debian). The first run adds VayuVaani to your app menu. |
| **Android and iPhone/iPad, at home** | On the computer running VayuVaani, open **Settings**, tick **Open on your phone**, and reopen the app. On the phone (same Wi-Fi), open the address Settings shows, such as `http://192.168.1.5:8765`. Phones can look at everything; keys, updates and training stay on the computer. Windows may ask once whether to allow VayuVaani on private networks: choose **Allow**. |
| **Any phone, tablet or computer, anywhere** | The **web version**: the same app as a website that updates itself every hour (GitHub Actions + GitHub Pages) and installs to the home screen (**Add to Home Screen** on iPhone, **Install app** on Android). Setup below. |

### Web version setup (one time)

1. On GitHub, open this repository's **Settings > Secrets and variables > Actions** and add a secret `OPENAQ_API_KEY` with your OpenAQ key (and optionally `FIRMS_MAP_KEY`).
2. In **Settings > Pages**, set **Source** to **GitHub Actions**.
3. In the **Actions** tab, open **Web version** and click **Run workflow**.

The site appears at `https://<username>.github.io/VayuVaani/` and refreshes every hour. GitHub pauses scheduled runs after 60 days without any commits; click **Run workflow** to restart them. In the web version, voice alerts use the phone's own voices.

## Setup for the command line

Needs Python 3.10 or newer.

```bash
cd vayuvaani
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev,fast]"
```

This installs the `vayu` command. (`python -m vayuvaani` works too.)

## Try it without API keys

```bash
vayu demo
pytest
```

`vayu demo` creates **synthetic** data (made-up numbers), trains the model,
prints an evaluation and makes a sample forecast. It takes a few minutes.
Use it to check everything works. **Never put demo results in your
report:** they aren't real measurements.

## Use real data

1. Get the free keys and put them in a file called `.env` (copy `.env.example`):
   - `OPENAQ_API_KEY`: https://explore.openaq.org/register (required)
   - `FIRMS_MAP_KEY`: https://firms.modaps.eosdis.nasa.gov/api/map_key/ (fire data)
   - `DATA_GOV_IN_API_KEY`: https://data.gov.in (CPCB live readings, used later)
2. Download data. Two or more years gives the model every season:
   ```bash
   vayu fetch --start 2023-01-01 --end 2026-10-31
   ```
   This takes a while (one request per station per 1,000 hours). Use
   `--no-fires` if you don't have a FIRMS key yet.
3. Clean and merge, then train:
   ```bash
   vayu build
   vayu train
   ```
4. Forecast:
   ```bash
   vayu forecast                          # live: next 48 hours from now
   vayu forecast --replay 2026-10-20T08:00   # pretend it's a past hour, compare with what happened
   ```

Outputs:
- `models/forecaster.joblib`: the trained model
- `reports/evaluation.json`: accuracy numbers for your report
- `reports/test_predictions.csv.gz`: every test forecast against the actual value
- `reports/latest_forecast.csv`: the latest 48-hour forecast per station

## How it works

| Stage | File | What it does |
|---|---|---|
| Collect | `sources/openaq.py` | Finds PM2.5 monitors in the city box and downloads hourly PM2.5/PM10. OpenAQ carries the CPCB/KSPCB stations. |
| | `sources/cpcb.py` | data.gov.in live readings, and a loader for files downloaded by hand from the CPCB CCR portal |
| | `sources/weather.py` | Open-Meteo hourly weather: past (archive) and next 3 days (forecast) |
| | `sources/firms.py` | NASA FIRMS fire detections within 150 km, as a stand-in for biomass and garbage burning |
| | `sources/events.py` | Deepavali, New Year, festivals, holidays, dry season from `data/events/bengaluru_events.csv` |
| Clean | `pipeline.py` | Removes negative, impossible, stuck and spike readings, fills gaps of up to 3 hours, joins everything into one hourly table |
| Features | `features.py` | For an origin hour t and horizon h: recent PM2.5 (lags, averages, trend), weather now and at t+h, average wind and total rain until t+h, time of day, weekday, season, events, recent fires |
| Model | `model.py` | One gradient-boosting model per quantile (10, 25, 50, 75, 90%), on log PM2.5. The 10–90% range is calibrated on held-out days so it covers about 80% of outcomes. |
| Evaluate | `train.py` | Time-based split (last 60 days held out). Compares with persistence ("stays the same") and "same hour, last day" baselines; reports MAE, RMSE, range coverage, category accuracy and alert precision/recall. |
| Forecast | `forecast.py` | Live and replay forecasts |

### Notes for the report and viva

- **AQI here is the PM2.5 sub-index.** The official CPCB AQI is the highest
  sub-index across several pollutants, averaged over 24 hours. VayuVaani forecasts
  PM2.5, the pollutant that usually decides Bengaluru's AQI, and labels it as such.
- **No future leakage.** Training origins stop 48 hours before the test period,
  calibration uses days the models never saw, and fire features only use past
  days. A test checks that live forecasts build exactly the same features as training.
- **Weather in training vs live.** Training uses observed weather for the target
  hour. Live forecasts use Open-Meteo's forecast, which has its own error, so live
  accuracy will be a little worse than the test numbers.
- **Check the festival dates** in `data/events/bengaluru_events.csv` against the
  Karnataka holiday list and add future years.
- **Backend.** LightGBM is used if installed (it is in requirements). Otherwise the
  code falls back to scikit-learn's HistGradientBoosting.

## Still to do

- Ward-level estimates between stations (kriging with road density and land use)
- Real SMS and missed-call alerts through an SMS gateway (the app previews them in mock mode)
- Citizen reports

## Tests

```bash
pytest
```

The tests cover AQI breakpoints, data cleaning, feature timing (no peeking at the
future), the quantile model, an end-to-end run on synthetic data, and the live
forecast path with the APIs replaced by fake data.
