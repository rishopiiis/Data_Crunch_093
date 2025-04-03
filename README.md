# Data_Crunch_093
CSE-40 - DATA CRUNCH
This project tackles the challenge of predicting weather patterns in Harveston—a fictional agricultural region facing increasing climate instability. Using historical temperature, rainfall, radiation, and wind data, we developed a machine learning system to forecast future conditions. Our solution combines feature engineering, ensemble modeling (LightGBM, XGBoost, and neural networks), and domain-specific post-processing to generate actionable insights for farmers. The goal is to optimize crop planning, improve resource management, and mitigate risks from erratic weather. This repository contains the complete pipeline from data preprocessing to model deployment, demonstrating a practical application of ML for climate resilience in agriculture.
01. Initial Setup
    - Imported libraries: Pandas, NumPy, XGBoost, LightGBM, TensorFlow
    - Set random seeds (42) for reproducibility
    - Suppressed warnings for cleaner output

02. Data Loading
    - Loaded training/test CSV files
    - Verified dataset shapes (Train: (rows, cols), Test: (rows, cols))

03. Data Preprocessing
    - Converted temperature (Kelvin → Celsius)
    - Created datetime features from Year/Month/Day
    - Added temporal features:
      * Cyclical encoding (month_sin/month_cos)
      * Seasonal markers
      * Weekend flags

04. Feature Engineering
    - Generated kingdom-specific statistics
    - Created lag features (1-30 day lags)
    - Added rolling window features (3-30 day)
    - Built interaction terms (Radiation×Temp, Rain×Wind)

05. Model Training
    - LightGBM: Multi-output regressor
    - XGBoost: Parallel training
    - Neural Network: Wind direction specialist

06. Post-Processing
    - Ensemble predictions (60% LGBM + 40% XGB)
    - Applied physical constraints:
      * No negative rainfall/radiation
      * Normalized wind directions (0-360°)
    - Temporal smoothing (7-day rolling mean)

07. Output
    - Formatted final predictions
    - Saved as submission.csv
