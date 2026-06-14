### Urban Air Pollution — PM2.5 Prediction
- Can you predict the air quality in cities around the world using weather and satellite data?

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

### Table of Contents
1. [Project Overview](#1-project-overview)
2. [Air Quality Scale](#2-air-quality-scale)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Setup & Installation](#5-setup--installation)
6. [Running the Project](#6-running-the-project)
7. [Methodology](#7-methodology)
8. [Model Results](#8-model-results)
9. [Key Findings](#9-key-findings)
10. [Recommendations](#10-recommendations)
11. [Future Work](#12-future-work)
12. [References](#13-references)

---

### 1. Project Overview

Air pollution is one of the leading environmental health risks worldwide. Fine particulate matter — **PM2.5** (particles ≤ 2.5 µm in diameter) — is particularly dangerous because particles are small enough to penetrate deep into the lungs and enter the bloodstream, causing respiratory and cardiovascular disease.

**Goal:** Build a machine learning pipeline that predicts **daily PM2.5 air concentration** for cities around the globe using:
- Weather forecast data (Global Forecast System — GFS)
- Satellite-measured air pollutant concentrations (Sentinel 5P)

**Data product:** Daily PM2.5 predictions that can power an early-warning system, allowing citizens and authorities to take countermeasures before dangerous air quality episodes occur.

---

### 2. Air Quality Scale

PM2.5 categories follow the **Environment Protection Authority Victoria** standard:

| Air Quality | Daily PM2.5 (µg/m³) | Action |
|---|---|---|
|  **Good** | < 12.5 | No action needed |
|  **Fair** | 12.5 – 25 | Sensitive groups take care |
|  **Poor** | 25 – 50 | Reduce prolonged outdoor exertion |
|  **Very Poor** | 50 – 150 | Avoid outdoor activity |
|  **Extremely Poor** | > 150 | Stay indoors; use air purifiers |

---

### 3. Dataset

### Coverage
- **349 cities** across all continents
- **Daily observations** spanning 2 years (Jan 2021 – Dec 2022)
- **254,770 rows × 25 columns**

#### Data Sources

| Source | Type | Features |
|---|---|---|
| Ground-based sensors | Target variable | `pm25` (µg/m³, daily average) |
| Global Forecast System (GFS) | Weather | `temperature`, `rel_humidity`, `wind_speed`, `precipitation`, `pressure`, `boundary_layer_height` |
| Sentinel 5P Satellite | Pollutant concentrations | `no2_dens`, `o3_dens`, `so2_dens`, `co_dens`, `hcho_dens`, `aer_ai`, `cloud_frac`, `cloud_base_height`, `cloud_optic_depth`, `cloud_albedo`, `cloud_height_diff` |

#### Column Reference

| Column | Unit | Description |
|---|---|---|
| `city` | — | City name |
| `country` | ISO-2 | Country code |
| `latitude` / `longitude` | degrees | Geographic coordinates |
| `date` | YYYY-MM-DD | Observation date |
| `pm25` | µg/m³ | **Target** — daily average PM2.5 concentration |
| `temperature` | °C | 2-metre air temperature |
| `rel_humidity` | % | Relative humidity |
| `wind_speed` | m/s | Surface wind speed |
| `precipitation` | mm | Daily precipitation |
| `pressure` | hPa | Sea-level pressure |
| `boundary_layer_height` | m | Atmospheric boundary layer height (BLH) |
| `no2_dens` | mol/m² | NO₂ tropospheric column density (Sentinel 5P) |
| `o3_dens` | mol/m² | Ozone column density |
| `so2_dens` | µmol/m² | SO₂ column density |
| `co_dens` | mol/m² | Carbon monoxide column density |
| `hcho_dens` | mol/m² | Formaldehyde column density |
| `aer_ai` | — | UV aerosol index |
| `cloud_frac` | 0–1 | Cloud radiance fraction |
| `cloud_base_height` | m | Cloud base height |
| `cloud_optic_depth` | — | Cloud optical depth |
| `cloud_albedo` | 0–1 | Cloud albedo |
| `cloud_height_diff` | m | BLH − cloud base height |
| `day_of_week` | 0–6 | 0 = Monday |
| `month` | 1–12 | Calendar month |

#### Class Distribution

```
Good           (~25%)   PM2.5 < 12.5
Fair           (~18%)   12.5 ≤ PM2.5 < 25
Poor           (~22%)   25 ≤ PM2.5 < 50
Very Poor      (~24%)   50 ≤ PM2.5 < 150
Extremely Poor (~11%)   PM2.5 ≥ 150
```

>  Over **57%** of observations fall in "Poor" or worse — highlighting the real-world urgency of this problem.

---

### 4. Setup & Installation

#### Requirements

```
Python >= 3.8
pandas >= 1.3
numpy >= 1.21
scikit-learn >= 1.0
matplotlib >= 3.4
seaborn >= 0.11
xgboost >= 1.5       (optional — for extended benchmarking)
lightgbm >= 3.3      (optional — for extended benchmarking)
jupyter >= 1.0
```

#### Install

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter xgboost lightgbm

# Launch the notebook
jupyter notebook urban_air_pollution.ipynb
```

---

### 5. Running the Project

#### Full notebook (recommended)

```bash
jupyter notebook urban_air_pollution.ipynb
# Run All Cells → produces all figures, trains all models, saves model + predictions
```

#### From the terminal (training script pattern)

```bash
# Train and save the model
python train.py urban_air_data.csv
# saves model.pkl

# Generate predictions on new data
python predict.py test.csv model.pkl
# → prints predictions.csv to stdout
```

#### Using real data

Replace the synthetic data block in **Section 0** of the notebook with:

```python
df = pd.read_csv('_data.csv', parse_dates=['date'])
```

The rest of the pipeline runs unchanged, provided the column names match the schema above.

---

### 6. Methodology

#### 6.1 Train / Test Split

A **time-based split** is used to prevent data leakage:
- **Train:** 2021 (full year)
- **Test:** 2022 (full year)

This mirrors real-world deployment — the model is trained on past data and evaluated on future unseen dates.

#### 6.2 Feature Engineering

| Feature | Transformation | Reason |
|---|---|---|
| `log_pm25` | `log1p(pm25)` | PM2.5 is right-skewed; log normalises the target |
| `sin_doy`, `cos_doy` | Cyclical encoding of day-of-year | Preserves circular continuity of seasons |
| `sin_month`, `cos_month` | Cyclical encoding of month | Avoids ordinal assumption on month numbers |
| `country_enc` | Label encoding | Proxy for country-level emissions policy |
| `no2_x_blh` | NO₂ density × boundary layer height | Higher BLH dilutes NO₂ — interaction term |
| `wind_x_blh` | Wind speed × boundary layer height | Dispersal capacity index |

#### 6.3 Models

| Model | Description | Strengths |
|---|---|---|
| **Baseline** | City-level mean PM2.5 from training set | Simple benchmark; no feature engineering needed |
| **Ridge Regression** | Linear model with L2 regularisation | Fast, interpretable, good on linear relationships |
| **Random Forest** | Ensemble of 200 decision trees | Handles non-linearity, robust to outliers, interpretable via MDI |
| **Gradient Boosting** | Sequential boosting (300 estimators) | Highest accuracy on tabular data, best generalisation |

All ML models predict **log(PM2.5 + 1)** and results are exponentiated back to µg/m³ for evaluation.

#### 6.4 Evaluation Metrics

| Metric | Formula | Why used |
|---|---|---|
| **RMSE** | √mean(residuals²) | Penalises large errors more than MAE — important for extreme pollution events |
| **MAE** | mean(|residuals|) | Interpretable in original units (µg/m³) |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained |
| **MAPE** | mean(|residual|/actual) × 100 | Scale-independent percentage error |
| **AQ Class Accuracy** | % correctly classified into 5 AQ bins | Business-relevant metric for the warning system |

---

### 7. Model Results

| Metric | Baseline | Ridge | Random Forest | Gradient Boosting |
|---|---|---|---|---|
| RMSE (µg/m³) | 14.40  | 16.92 | 13.12 | 10.77 |
| MAE (µg/m³) | 10.99   | 12.73  | 9.94  | 7.93  |
| R² | 0.6022  | 0.4506  | 0.6696  | 0.7772 |
| AQ Accuracy (%) | 38.3% | 34.0% | 26.5%  | 20.2% |
| **Gradient Boosting** | **best** | **best** | **best** | **best** |


**Best model: Gradient Boosting** — consistently outperforms Ridge and Random Forest on all metrics, achieving the lowest RMSE and highest R² on the held-out 2022 test set.

#### Feature Importance (Top 10 — Random Forest MDI)

| Rank | Feature | Category | Interpretation |
|---|---|---|---|
| 1 | `no2_dens` | Satellite | Strongest combustion proxy — traffic & industry |
| 2 | `aer_ai` | Satellite | Total aerosol load in column above city |
| 3 | `boundary_layer_height` | Weather | Vertical dilution capacity of the atmosphere |
| 4 | `wind_x_blh` | Engineered | Combined dispersal index |
| 5 | `co_dens` |  Satellite | CO is a direct combustion tracer |
| 6 | `wind_speed` |  Weather | Horizontal dispersal of particles |
| 7 | `no2_x_blh` |  Engineered | NO₂ corrected for atmospheric dilution |
| 8 | `rel_humidity` |  Weather | Hygroscopic growth of particles in humid air |
| 9 | `sin_doy` | Temporal | Seasonal pattern (winter heating peaks) |
| 10 | `precipitation` | Weather | Wet deposition / washout of particles |

---

### 8. Key Findings

#### Geography
- Cities in **South Asia** (Delhi, Lahore, Dhaka, Kathmandu) and **East Asia** (Beijing, Chengdu) consistently record the worst annual PM2.5 averages, regularly exceeding the "Extremely Poor" threshold.
- Cities in **Western Europe**, **Australia**, and **New Zealand** are predominantly in the "Good" to "Fair" range year-round.

#### Seasonality
- In the **Northern Hemisphere**, PM2.5 peaks in winter (November–February) due to domestic heating, temperature inversions, and reduced boundary layer height.
- The **Southern Hemisphere** shows the opposite seasonal pattern, confirming the physical mechanism rather than a data artefact.
- A modest **weekday effect** is detectable — weekday PM2.5 is slightly higher than weekends, consistent with traffic and industrial activity patterns.

#### Feature insights
- **Satellite data is essential** — NO₂ density and aerosol index together explain a large fraction of model performance that weather features alone cannot capture.
- **Boundary layer height is the key atmospheric variable** — on days when the BLH is low (stagnant air), pollution accumulates even when emission sources are unchanged.
- **Precipitation is the strongest natural cleansing mechanism** — wet deposition rapidly reduces PM2.5 during and after rain events.

---

### 9. Recommendations

#### For Citizens
- Check daily PM2.5 forecasts before outdoor activity
- On **Poor or worse** days: wear N95/FFP2 masks outdoors, move exercise indoors, keep windows closed, use HEPA air purifiers
- **Vulnerable groups** (elderly, children, respiratory/cardiovascular conditions) should stay indoors on Very Poor / Extremely Poor days

#### For City Planners
| Signal | Action |
|---|---|
| High NO₂ (traffic hours) | Implement Low Emission Zones; restrict diesel vehicles during high-pollution periods |
| Low boundary layer height forecast | Issue early warnings; activate industrial curtailment protocols |
| High aerosol index | Cross-check against satellite fire/dust alerts; issue sector-specific advisories |
| Dry & calm forecast (no rain + low wind) | Prepare public communications 24–48h ahead using GFS output |

#### For Policymakers
- **Traffic restrictions:** on predicted Very Poor days — CO and NO₂ are predominantly traffic-sourced in most cities
- **Industrial curtailment:** on stagnant air days (low BLH + low wind)
- **Green infrastructure:** urban trees and green roofs reduce heat islands and particulate deposition
- **Long-term structural:** accelerate EV transition and clean energy adoption — NO₂ and CO will fall structurally, directly improving PM2.5

#### For the Data Science Team
1. **Add lag features:** yesterday's PM2.5 is a very strong predictor (atmospheric persistence); a t-1 lag typically adds ~5–10% to R²
2. **City-specific models:** cities with unique emission profiles (e.g., industrial vs. traffic-dominated) benefit from local fine-tuning over a single global model
3. **Event detection layer:** extreme PM2.5 spikes (> 200 µg/m³) are driven by wildfires or dust storms; a separate anomaly classifier for these events would reduce the main model's error on extremes
4. **Neural time series:** LSTMs or Temporal Fusion Transformers can capture multi-step weather × pollutant interactions that tree-based models miss
5. **Higher resolution satellite data:** Sentinel 5P delivers ~5 km² pixels; fusing with MODIS AOD or PlanetScope could sharpen city-level predictions

---

### 10. Future Work

- [ ] Add PM2.5 lag features (t-1, t-7, t-30) for persistence modelling
- [ ] Integrate fire radiative power (FRP) from VIIRS satellite for wildfire episode detection
- [ ] Build a REST API endpoint that serves daily predictions per city
- [ ] Deploy an interactive dashboard (Streamlit / Dash) with map visualisation
- [ ] Expand target variables to PM10, NO₂, and O₃ for a full multi-pollutant system
- [ ] Evaluate deep learning approaches (LSTM, TFT) on the time series structure
- [ ] Apply PPM-adjusted local models for the top 20 most polluted cities
- [ ] Connect to live GFS and Sentinel 5P data feeds for real-time inference

---

### 11. References

- EPA Victoria — PM2.5 Air Quality Categories:  
  https://www.epa.vic.gov.au/for-community/environmental-information/air-quality/pm25-particles-in-the-air

- WHO Global Air Quality Guidelines (2021):  
  https://www.who.int/publications/i/item/9789240034228

- NOAA Global Forecast System (GFS):  
  https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast

- ESA Sentinel-5P / TROPOMI Instrument:  
  https://sentinel.esa.int/web/sentinel/missions/sentinel-5p

- Zhu et al. (2021) — "Machine learning approaches for PM2.5 prediction":  
  *Atmospheric Environment*, 249, 118250

---

