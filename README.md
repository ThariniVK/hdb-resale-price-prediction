# HDB Resale Price Prediction

A machine learning project that predicts Singapore HDB resale flat prices using historical resale transaction data.

The project covers the complete machine learning workflow, including data exploration, preprocessing, model training, evaluation, error analysis, model serialization, and deployment through a FastAPI API running inside Docker.

## Project Structure

```text
HDB-ML-Project/
│
├── app/
│   └── main.py
│
├── data/
│   └── ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv
│
├── models/
│   └── hdb_price_model.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   └── predict.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```
## Dataset

The dataset contains historical Singapore HDB resale flat transactions from January 2017 onwards.

The original dataset contains:

- 239,467 transactions
- 11 columns
- 26 towns
- 7 flat types
- 21 flat models

The target variable is `resale_price`.

### Data Quality

Initial data inspection showed:

- 318 duplicate rows
- No missing values
- Both numerical and categorical features

The duplicate rows were removed, leaving **239,149 transactions** for modelling.

## Exploratory Data Analysis

Several relationships between property characteristics and resale prices were investigated.

### Resale Price Distribution

The resale price statistics were:

| Statistic | Price |
|---|---:|
| Mean | $534,386 |
| Median | $502,000 |
| Minimum | $140,000 |
| Maximum | $1,728,000 |

### Floor Area vs Resale Price

Larger flats generally tend to have higher resale prices, although the relationship is not perfectly linear. Other factors such as location, flat type, floor level, and lease characteristics also influence the price.

### Resale Price by Town

Median resale prices varied considerably across towns.

The highest median prices were observed in:

- Bukit Timah
- Bishan
- Queenstown
- Bukit Merah

This suggests that **location is an important factor in HDB resale pricing**.

### Resale Price by Flat Type

Median prices also increased substantially across larger flat types:

| Flat Type | Median Resale Price |
|---|---:|
| 1 ROOM | $212,500 |
| 2 ROOM | $313,000 |
| 3 ROOM | $360,000 |
| 4 ROOM | $510,000 |
| 5 ROOM | $612,000 |
| EXECUTIVE | $728,000 |
| MULTI-GENERATION | $846,500 |

### Resale Price by Storey

Higher-floor properties generally showed higher median resale prices.

Median prices increased from approximately **$455,000 for floors 1–3** to **$1.23 million for floors 49–51**.

This indicates that floor level can provide useful predictive information.

## Data Preprocessing & Feature Engineering

The raw dataset contained several categorical and text-based fields that needed to be transformed before model training.

### Duplicate Removal

Duplicate transactions were removed from the dataset:

```python
df_clean = df.drop_duplicates().copy()
```

### Date Features

The original `month` column was converted into a datetime format and separated into two numerical features:

- `sale_year`
- `sale_month`

The original `month` column was then removed.

### Storey Feature

The `storey_range` column contained ranges such as `01 TO 03`, `04 TO 06`, and `10 TO 12`.

To represent the approximate floor numerically, the midpoint of each range was calculated.

For example:

- `01 TO 03` → `2`
- `04 TO 06` → `5`
- `10 TO 12` → `11`

The original `storey_range` column was then removed.

### Remaining Lease

The `remaining_lease` column was originally stored as text, such as `61 years 04 months`.

This was converted into a numerical `remaining_lease_years` feature.

For example:

- `61 years 04 months` → `61.33 years`
- `60 years 07 months` → `60.58 years`

The original `remaining_lease` column was then removed.

### Feature Selection

The `block` and `street_name` columns were removed from the final modelling dataset.

The final features used for prediction were:

**Categorical features:**

- `town`
- `flat_type`
- `flat_model`

**Numerical features:**

- `floor_area_sqm`
- `lease_commence_date`
- `sale_year`
- `sale_month`
- `storey_mid`
- `remaining_lease_years`

The target variable was `resale_price`.

The final modelling dataset contained **239,149 transactions and 9 features**.

## Model Training

The dataset was split into training and testing sets using an 80/20 split.

- Training set: 191,319 transactions
- Test set: 47,830 transactions

A `ColumnTransformer` was used to preprocess the categorical and numerical features.

Categorical features were encoded using One-Hot Encoding with `handle_unknown="ignore"` so that unseen categories can be handled during prediction.

Two regression models were trained and compared:

- Ridge Regression
- XGBoost Regression

## Model Results

Two models were evaluated on the held-out test set using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge Regression | $52,913 | $69,547 | 0.8685 |
| **XGBoost** | **$30,101** | **$41,450** | **0.9533** |

### XGBoost Performance

XGBoost achieved an R² score of **0.9533**, meaning the model explained approximately **95.3% of the variation in resale prices on the test set**.

The model achieved:

- **MAE:** $30,101
- **RMSE:** $41,450
- **R²:** 0.9533

XGBoost substantially outperformed the Ridge Regression baseline across all three evaluation metrics and was therefore selected as the final model.

### Why XGBoost Was Selected

Ridge Regression provided a useful baseline, but its performance was lower than the XGBoost model.

XGBoost was better able to capture the non-linear relationships between property characteristics and resale prices, resulting in lower prediction errors and a higher R² score.


## Feature Importance

Feature importance was examined using the trained XGBoost model to understand which feature groups contributed most to the predictions.

| Feature Group | Importance |
|---|---:|
| Town | 52.1% |
| Flat Type | 18.3% |
| Flat Model | 15.6% |
| Sale Year | 5.6% |
| Floor Area | 4.4% |
| Storey | 2.3% |
| Lease Commence Year | 1.3% |
| Remaining Lease | 0.4% |
| Sale Month | 0.1% |

The results indicate that **town was the strongest overall feature group**, followed by flat type and flat model.

This is consistent with the exploratory analysis, which showed substantial differences in resale prices between towns.

## Error Analysis

Prediction errors were analysed to understand where the model performed less accurately.

The model generally produced good predictions, but larger errors were observed for some high-value properties.

Some of the largest prediction errors occurred for:

- Premium Apartment Loft units
- Executive flats
- Larger properties
- High-value properties in areas such as Queenstown and Bukit Merah

The largest errors in the test set were approximately **$300,000–$376,000**.

This suggests that some high-value transactions contain pricing patterns that are difficult for the model to capture using the available features.

## Model Serialization

The final XGBoost pipeline, including the preprocessing steps and trained model, was saved using `joblib`.

The saved model is stored in:

```text
models/hdb_price_model.pkl
```

## API

A REST API was built using FastAPI to serve predictions from the trained machine learning model.

## Docker

The FastAPI application was containerized using Docker to provide a consistent environment for running the application.

The Docker image includes:

Python 3.12
Required Python dependencies
FastAPI application
Prediction logic
Trained machine learning model

## Tech Stack

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Matplotlib** — Data visualization
- **Scikit-learn** — Data preprocessing, pipelines, and evaluation
- **XGBoost** — Machine learning regression model
- **Joblib** — Model serialization
- **FastAPI** — REST API development
- **Uvicorn** — API server
- **Docker** — Application containerization
- **Jupyter Notebook** — Data exploration and model development
- **Git & GitHub** — Version control and project management

## Project Workflow

The project follows an end-to-end machine learning workflow:

```text
Raw HDB Resale Data
        ↓
Data Exploration
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Feature Preprocessing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Error Analysis
        ↓
Model Serialization
        ↓
FastAPI API
        ↓
Docker Container
```


## Future Improvements

Potential improvements to the project include:

-Incorporating more detailed location-based
 features.
-Performing hyperparameter tuning for XGBoost.
-Using cross-validation for more robust model evaluation.
-Comparing XGBoost with additional models such as Random Forest and LightGBM.
-Adding automated model retraining when new HDB transaction data becomes available.
-Deploying the FastAPI application to a cloud platform.
-Developing a web interface where users can enter property details and receive a predicted resale price.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting Singapore HDB resale prices.

The final XGBoost model achieved an R² of 0.9533 and a Mean Absolute Error of approximately $30,101 on the held-out test set.

The trained model was then serialized and integrated into a FastAPI application, which was containerized using Docker to make the prediction service portable and reproducible.