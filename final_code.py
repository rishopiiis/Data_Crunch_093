# 1. INITIAL SETUP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# 2. DATA LOADING & EXPLORATION

print("Loading datasets...")
train = pd.read_csv("/kaggle/input/data-crunch-round-1/train.csv")
test = pd.read_csv("/kaggle/input/data-crunch-round-1/test.csv")
sample_sub = pd.read_csv("/kaggle/input/data-crunch-round-1/sample_submission.csv")

print(f"\nData Shapes:")
print(f"Train: {train.shape}, Test: {test.shape}")

# Visualize data distributions
print("\nVisualizing feature distributions...")
plt.figure(figsize=(15, 10))
features_to_plot = ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed']
for i, col in enumerate(features_to_plot):
    plt.subplot(2, 2, i+1)
    sns.histplot(train[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# 3. DATA PREPROCESSING

print("\nPreprocessing data...")
def preprocess_data(df, is_train=True):
    # Convert temperature from Kelvin to Celsius if needed
    if is_train and 'Avg_Temperature' in df.columns:
        df['Avg_Temperature'] = df['Avg_Temperature'].apply(lambda x: x - 273.15 if x > 100 else x)
    if is_train and 'Avg_Feels_Like_Temperature' in df.columns:
        df['Avg_Feels_Like_Temperature'] = df['Avg_Feels_Like_Temperature'].apply(lambda x: x - 273.15 if x > 100 else x)
    
    # Create datetime feature
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                               df['Month'].astype(str).str.zfill(2) + '-' + 
                               df['Day'].astype(str).str.zfill(2))
    
    # Enhanced temporal features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['season'] = df['date'].dt.month % 12 // 3 + 1  # 1:Winter, 2:Spring, etc.
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month/12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month/12)
    df['is_weekend'] = df['date'].dt.dayofweek // 5
    
    return df

train = preprocess_data(train, is_train=True)
test = preprocess_data(test, is_train=False)

# Visualize temporal patterns
print("\nVisualizing temporal trends...")
plt.figure(figsize=(15, 8))
train.groupby('date')['Avg_Temperature'].mean().plot(color='royalblue')
plt.title('Historical Temperature Trends')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()


# 4. FEATURE ENGINEERING
print("\nEngineering features...")
def create_features(df, is_train=True):
    # Calculate kingdom statistics from training data only
    if is_train:
        global kingdom_stats
        kingdom_stats = df.groupby('kingdom').agg({
            'Avg_Temperature': ['mean', 'std'],
            'Radiation': ['mean', 'std'],
            'Rain_Amount': ['mean', 'std'],
            'Wind_Speed': ['mean', 'std']
        }).reset_index()
        
        kingdom_stats.columns = ['kingdom', 'temp_mean', 'temp_std', 'rad_mean', 'rad_std', 
                               'rain_mean', 'rain_std', 'wind_mean', 'wind_std']
    
    # Merge kingdom stats
    df = df.merge(kingdom_stats, on='kingdom', how='left')
    
    # Create lag features for training data
    if is_train:
        for col in ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed']:
            for lag in [1, 3, 7, 14, 30]:
                df[f'{col}_lag{lag}'] = df.groupby('kingdom')[col].shift(lag)
    
    # Create rolling features
    for col in ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed']:
        if col in df.columns:
            for window in [3, 7, 14, 30]:
                df[f'{col}_rolling{window}_mean'] = df.groupby('kingdom')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean())
                df[f'{col}_rolling{window}_std'] = df.groupby('kingdom')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std())
    
    # Interaction features
    if 'Radiation' in df.columns and 'Avg_Temperature' in df.columns:
        df['rad_temp_interaction'] = df['Radiation'] * df['Avg_Temperature']
    if 'Rain_Amount' in df.columns and 'Wind_Speed' in df.columns:
        df['rain_wind_interaction'] = df['Rain_Amount'] * df['Wind_Speed']
    
    return df

train = create_features(train, is_train=True)
test = create_features(test, is_train=False)

# Handle missing values
train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)


# 5. MODEL TRAINING
print("\nPreparing for model training...")
# Define features and targets
common_features = list(set(train.columns).intersection(set(test.columns)))
features = [f for f in common_features if f not in ['ID', 'date', 'Year', 'Month', 'Day', 'kingdom'] + 
            ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']]

targets = ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']

print(f"Using {len(features)} features for modeling")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train[features])
X_test_scaled = scaler.transform(test[features])
y_train = train[targets]


# 6. MODEL IMPLEMENTATION
print("\nTraining models...")
"""
MODEL ARCHITECTURE:
1. LightGBM (60% weight):
   - 500 trees, depth 8
   - Learning rate 0.05
   - MAE optimization
   - Handles temporal patterns well

2. XGBoost (40% weight):
   - 300 trees, depth 6  
   - More conservative learning
   - Additional regularization
   - Complements LightGBM

3. Neural Network (Wind Direction):
   - 3 hidden layers (128-64-32)
   - Dropout for regularization
   - Sin/Cos output for circular data
   - Specialized for angular values
"""

# LightGBM model
lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'metric': 'mae'
}

lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_params))
lgb_model.fit(X_train_scaled, y_train.drop('Wind_Direction', axis=1))

# XGBoost model
xgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

xgb_model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
xgb_model.fit(X_train_scaled, y_train.drop('Wind_Direction', axis=1))

# Neural Network for Wind Direction
print("\nTraining Wind Direction model...")
y_wind = y_train['Wind_Direction']
y_wind_sin = np.sin(y_wind * np.pi / 180)
y_wind_cos = np.cos(y_wind * np.pi / 180)

wind_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2)
])
wind_model.compile(optimizer='adam', loss='mse')
history = wind_model.fit(X_train_scaled, np.column_stack([y_wind_sin, y_wind_cos]), 
                        epochs=30, batch_size=64, verbose=0)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.title('Wind Direction Model Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# 7. PREDICTION & VISUALIZATION
print("\nGenerating predictions...")
# Get predictions from both models
lgb_preds = lgb_model.predict(X_test_scaled)
xgb_preds = xgb_model.predict(X_test_scaled)

# Weighted ensemble predictions
ensemble_preds = 0.6 * lgb_preds + 0.4 * xgb_preds

# Predict wind direction
wind_preds = wind_model.predict(X_test_scaled)
wind_directions = np.arctan2(wind_preds[:, 0], wind_preds[:, 1]) * 180 / np.pi % 360

# Create submission
submission = test[['ID']].copy()
submission['Avg_Temperature'] = ensemble_preds[:, 0]
submission['Radiation'] = ensemble_preds[:, 1].clip(0)
submission['Rain_Amount'] = ensemble_preds[:, 2].clip(0)
submission['Wind_Speed'] = ensemble_preds[:, 3].clip(0)
submission['Wind_Direction'] = wind_directions

# Temporal smoothing
for col in ['Avg_Temperature', 'Wind_Speed']:
    submission[col] = submission[col].rolling(7, min_periods=1, center=True).mean()

# Visualize predictions
plt.figure(figsize=(15, 8))
plt.plot(train['date'], train['Avg_Temperature'], label='Historical', color='royalblue')
plt.plot(test['date'], submission['Avg_Temperature'], label='Forecast', color='darkorange')
plt.title('Temperature Forecast vs History')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Feature importance
lgb_importance = lgb_model.estimators_[0].feature_importances_
top_features = 15
sorted_idx = np.argsort(lgb_importance)[-top_features:]

plt.figure(figsize=(10, 6))
sns.barplot(x=lgb_importance[sorted_idx], y=np.array(features)[sorted_idx])
plt.title(f'Top {top_features} Important Features (LightGBM)')
plt.tight_layout()
plt.show()


# 8. FINAL OUTPUT
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved successfully!")
print("Sample predictions:")
print(submission.head().to_markdown())
