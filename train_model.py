import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import os

print("Loading data...")
training_data = pd.read_csv("train.csv")
training_data = training_data.drop_duplicates().reset_index(drop=True)

print("Engineering features...")
# Preprocessing logic matching the notebook (minus rank features)
training_data['temp_humidity_ratio'] = training_data['Temperature_C'] / (training_data['Humidity'] + 1e-5)
training_data['rainfall_temp_ratio'] = training_data['Rainfall_mm'] / (training_data['Temperature_C'] + 1e-5)
training_data['heat_index_proxy'] = training_data['Temperature_C'] * training_data['Sunlight_Hours']
training_data['wind_chill_proxy'] = training_data['Temperature_C'] * training_data['Wind_Speed_kmh']

# Soil interactions
training_data['soil_health_idx'] = training_data['Organic_Carbon'] * training_data['Soil_Moisture'] / (training_data['Soil_pH'] + 1e-5)
training_data['moisture_conductivity'] = training_data['Soil_Moisture'] * training_data['Electrical_Conductivity']
training_data['air_soil_moisture_mismatch'] = training_data['Humidity'] / (training_data['Soil_Moisture'] + 1e-5)
training_data['ph_conductivity_ratio'] = training_data['Soil_pH'] / (training_data['Electrical_Conductivity'] + 1e-5)

# Area and Volume interactions
training_data['rain_volume'] = training_data['Rainfall_mm'] * training_data['Field_Area_hectare']
training_data['previous_irrigation_volume'] = training_data['Previous_Irrigation_mm'] * training_data['Field_Area_hectare']
training_data['irrigation_diff'] = training_data['Rainfall_mm'] - training_data['Previous_Irrigation_mm']
training_data['total_water_supply'] = training_data['Rainfall_mm'] + training_data['Previous_Irrigation_mm']

# Log transforms
training_data['log_Rainfall_mm'] = np.log1p(training_data['Rainfall_mm'])
training_data['log_Previous_Irrigation_mm'] = np.log1p(training_data['Previous_Irrigation_mm'])

training_data['water_balance']= training_data['Rainfall_mm'] - training_data['Soil_Moisture']
training_data['Soil_health']= training_data['Soil_pH'] * training_data['Electrical_Conductivity']
training_data['Soil_moisture_sqrt']= np.sqrt(training_data['Soil_Moisture'])

# Drop ID and split x, y
x = training_data.drop(columns=['id', 'Irrigation_Need'])
mapping = {'Low': 0, "Medium": 1, 'High': 2}
y = training_data['Irrigation_Need'].map(mapping)

# Identify numerical and categorical columns before dropping
numeric_x = x.select_dtypes(exclude=['object'])
category_x = x.select_dtypes(include=['object'])

# Dropping columns specified in the notebook, modified for removed rank features
drop_col_2 = [
    'Organic_Carbon', 'Electrical_Conductivity', 'heat_index_proxy', 'Soil_pH', 
    'Sunlight_Hours', 'Field_Area_hectare', 'soil_health_idx', 'rain_volume', 
    'moisture_conductivity', 'ph_conductivity_ratio', 'previous_irrigation_volume', 
    'Soil_health', 'temp_humidity_ratio', 'total_water_supply'
]
numeric_x = numeric_x.drop(columns=[col for col in drop_col_2 if col in numeric_x.columns])

# Scaling
print("Scaling and Encoding...")
scaler = StandardScaler()
numeric_x_scaled = scaler.fit_transform(numeric_x)

# Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
category_x_encoded = encoder.fit_transform(category_x)

final_x = np.concatenate((numeric_x_scaled, category_x_encoded), axis=1)

# Sample weights
classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weights = dict(zip(classes, weights))
class_weights[2] *= 1.30 # According to notebook
sample_weights = np.array([class_weights[y_i] for y_i in y])

print("Training Models... (This might take a few minutes)")
os.makedirs('artifacts', exist_ok=True)

# 1. XGBoost
xgb_params = {
    'max_depth': 5,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.03,
    'n_estimators': 2000,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.5,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'random_state': 42
}
print("Training XGBoost...")
xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(final_x, y, sample_weight=sample_weights, verbose=0)
joblib.dump(xgb_model, 'artifacts/xgb_model.joblib')

# 2. LightGBM
print("Training LightGBM...")
lgb_model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=63,
    min_child_sample=50,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgb_model.fit(final_x, y, sample_weight=sample_weights)
joblib.dump(lgb_model, 'artifacts/lgb_model.joblib')

# 3. CatBoost
print("Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.04,
    depth=6,
    loss_function='MultiClass',
    l2_leaf_reg=5,               
    random_strength=1.5,         
    bagging_temperature=0.5,    
    border_count=128,
    early_stopping_rounds=100,
    random_seed=42,
    verbose=0
)
cat_model.fit(final_x, y)
joblib.dump(cat_model, 'artifacts/cat_model.joblib')

print("Saving Preprocessors...")
joblib.dump(scaler, 'artifacts/scaler.joblib')
joblib.dump(encoder, 'artifacts/encoder.joblib')

# Save expected column names to ensure consistency in inference
joblib.dump(list(numeric_x.columns), 'artifacts/numeric_cols.joblib')
joblib.dump(list(category_x.columns), 'artifacts/category_cols.joblib')

print("Training Complete! All artifacts saved.")
