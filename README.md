# Trip Duration Prediction using NYC Taxi Dataset

This project uses Apache Spark (PySpark) to analyze the NYC Yellow Taxi Trip Dataset and build scalable machine learning pipelines for:

- Trip Duration Prediction
- Anomaly Detection

The dataset contains millions of taxi trip records, making distributed processing necessary.

---

## Dataset

Dataset: NYC Yellow Taxi Trip Data

Source: https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data

The dataset includes features such as:

- VendorID
- passenger_count
- trip_distance
- pickup_longitude
- pickup_latitude
- dropoff_longitude
- dropoff_latitude
- RateCodeID
- payment_type
- fare_amount
- tip_amount
- total_amount

Target variable:

trip_duration = (dropoff_time - pickup_time) / 60

---

## Technologies Used

- Python
- PySpark
- Apache Spark MLlib
- Jupyter Notebook

Libraries:
- pyspark
- numpy
- pandas
- matplotlib

---

## Data Preprocessing

The dataset contains noisy and unrealistic values, so several preprocessing steps were applied:

- Removed rows containing missing values
- Converted timestamps to Unix format
- Created the trip_duration variable
- Removed trips outside NYC geographic bounds
- Filtered invalid passenger counts (>6)
- Removed trips longer than 720 minutes
- Removed invalid charge values
- Removed extreme outliers using boxplots and histograms

Final cleaned dataset size: ~12.4 million records.

---

## Feature Engineering

Feature preparation used Spark ML pipeline components.

StringIndexer  
Converted categorical variables to indexed form:
- store_and_fwd_flag
- RateCodeID

OneHotEncoder  
Applied to:
- VendorID
- RateCodeID
- store_and_fwd_flag
- payment_type

VectorAssembler  
Combined all features into a single feature vector.

StandardScaler  
withMean = False  
withStd = True

---

## Machine Learning Models

Three regression models were trained to predict trip duration.

Linear Regression  
Hyperparameter:
regParam ∈ {0.01, 0.1, 1.0}

RMSE: 6.45 minutes

Random Forest Regressor  
Hyperparameters:
minInstancesPerNode ∈ {1,5}  
minInfoGain ∈ {0.0,0.1}

RMSE: 6.28 minutes

Gradient Boosted Trees (Best Model)  
Hyperparameters:
minInstancesPerNode ∈ {1,5}  
stepSize ∈ {0.01,0.1}

RMSE: 5.89 minutes

---

## Evaluation Metric

Models were evaluated using Root Mean Squared Error (RMSE).

RMSE = sqrt((1/n) * Σ(y − ŷ)^2)

Mean trip duration ≈ 12.3 minutes  
Best model error ≈ 5.89 minutes

---

## Anomaly Detection

An unsupervised anomaly detection system was implemented using KMeans clustering.

k = 60

Process:

1. Train clustering model on January 2015 data
2. Apply model to January 2016 data
3. Compute distance to cluster centroid
4. Flag trips with high distance as anomalies

Distance formula:

d(x, μ) = sqrt(Σ(x − μ)^2)

Trips with distance greater than 42.8 were classified as anomalies.

---

## Results

Model Performance:

Linear Regression — RMSE: 6.45  
Random Forest — RMSE: 6.28  
Gradient Boosted Trees — RMSE: 5.89

Best model: Gradient Boosted Trees.

---

## Project Pipeline

Trip Duration Prediction Pipeline:

Data Ingestion  
→ Data Cleaning  
→ Feature Engineering  
→ Model Training  
→ Model Evaluation

Anomaly Detection Pipeline:

Data Ingestion  
→ Data Cleaning  
→ Feature Engineering  
→ KMeans Clustering  
→ Distance Calculation  
→ Anomaly Detection

---

## Authors

Annanya Tayal  
B.Tech Computer Science and Engineering  
Manipal Institute of Technology

Brendan Badhe
B.Tech Computer Science and Engineering  
Manipal Institute of Technology

Under the guidance of Dr. Anup Bhat B

---

## References

- NYC Taxi and Limousine Commission
- Apache Spark MLlib Documentation
- Kaggle NYC Taxi Dataset
