# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

training_data= pd.read_csv("/kaggle/input/competitions/playground-series-s6e4/train.csv") 
testing_data= pd.read_csv("/kaggle/input/competitions/playground-series-s6e4/test.csv")

training_data.head()

training_data.shape

testing_data.shape

training_data.isnull().sum()

training_data.info()

training_data= training_data.drop_duplicates().reset_index(drop=True)

corr= training_data.corr(numeric_only=True)

fig= px.imshow(corr,text_auto='.2f',aspect="auto")
fig.show()

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
    
# Log transforms for highly skewed data limits outliers effect
training_data['log_Rainfall_mm'] = np.log1p(training_data['Rainfall_mm'])
training_data['log_Previous_Irrigation_mm'] = np.log1p(training_data['Previous_Irrigation_mm'])

#ranking features
training_data['Soil_Moisture_rank']= training_data['Soil_Moisture'].rank(pct=True)
training_data['EC_rank']= training_data['Electrical_Conductivity'].rank(pct=True)
training_data['Rainfall_rank']= training_data['Rainfall_mm'].rank(pct=True)
training_data['Irrigation_rank']= training_data['Previous_Irrigation_mm'].rank(pct=True)

training_data['water_balance']= training_data['Rainfall_mm']- training_data['Soil_Moisture']
training_data['Soil_health']= training_data['Soil_pH']*training_data['Electrical_Conductivity']
training_data['Soil_moisture_sqrt']= np.sqrt(training_data['Soil_Moisture'])

#training_data['moisture_to_rain_ratio'] = training_data['Soil_Moisture'] / (training_data['Rainfall_mm'] + 1e-5)
#training_data['irrigation_efficiency'] = training_data['Soil_Moisture'] / (training_data['Previous_Irrigation_mm'] + 1e-5)
#training_data['temp_x_rain'] = training_data['Temperature_C'] * training_data['Rainfall_mm']
#training_data['temp_humidity_diff'] = training_data['Temperature_C'] - training_data['Humidity']

#non linear features
#training_data['temp_squared'] = training_data['Temperature_C'] ** 2
#training_data['temp_cubed'] = training_data['Temperature_C'] ** 3
#training_data['humidity_power'] = training_data['Humidity'] ** 1.5
#training_data['soil_moisture_log_sq'] = np.log1p(training_data['Soil_Moisture']) ** 2
#training_data['rain_sqrt_temp'] = np.sqrt(training_data['Rainfall_mm']) * training_data['Temperature_C']
#training_data['ec_power'] = training_data['Electrical_Conductivity'] ** 1.3
#training_data['ph_squared'] = training_data['Soil_pH'] ** 2
#training_data['temp_moisture_nl'] = (training_data['Temperature_C'] ** 2) * np.sqrt(training_data['Soil_Moisture'])
#training_data['log_rain_temp'] = np.log1p(training_data['Rainfall_mm']) * training_data['Temperature_C']

testing_data['temp_humidity_ratio'] = testing_data['Temperature_C'] / (testing_data['Humidity'] + 1e-5)
testing_data['rainfall_temp_ratio'] = testing_data['Rainfall_mm'] / (testing_data['Temperature_C'] + 1e-5)
testing_data['heat_index_proxy'] = testing_data['Temperature_C'] * testing_data['Sunlight_Hours']
testing_data['wind_chill_proxy'] = testing_data['Temperature_C'] * testing_data['Wind_Speed_kmh']
    
# Soil interactions
testing_data['soil_health_idx'] = testing_data['Organic_Carbon'] * testing_data['Soil_Moisture'] / (testing_data['Soil_pH'] + 1e-5)
testing_data['moisture_conductivity'] = testing_data['Soil_Moisture'] * testing_data['Electrical_Conductivity']
testing_data['air_soil_moisture_mismatch'] = testing_data['Humidity'] / (testing_data['Soil_Moisture'] + 1e-5)
testing_data['ph_conductivity_ratio'] = testing_data['Soil_pH'] / (testing_data['Electrical_Conductivity'] + 1e-5)
    
# Area and Volume interactions
testing_data['rain_volume'] = testing_data['Rainfall_mm'] * testing_data['Field_Area_hectare']
testing_data['previous_irrigation_volume'] = testing_data['Previous_Irrigation_mm'] * testing_data['Field_Area_hectare']
testing_data['irrigation_diff'] = testing_data['Rainfall_mm'] - testing_data['Previous_Irrigation_mm']
testing_data['total_water_supply'] = testing_data['Rainfall_mm'] + testing_data['Previous_Irrigation_mm']
    
# Log transforms for highly skewed data limits outliers effect
testing_data['log_Rainfall_mm'] = np.log1p(testing_data['Rainfall_mm'])
testing_data['log_Previous_Irrigation_mm'] = np.log1p(testing_data['Previous_Irrigation_mm'])

#ranking features
testing_data['Soil_Moisture_rank']= testing_data['Soil_Moisture'].rank(pct=True)
testing_data['EC_rank']= testing_data['Electrical_Conductivity'].rank(pct=True)
testing_data['Rainfall_rank']= testing_data['Rainfall_mm'].rank(pct=True)
testing_data['Irrigation_rank']= testing_data['Previous_Irrigation_mm'].rank(pct=True)

testing_data['water_balance']= testing_data['Rainfall_mm']- testing_data['Soil_Moisture']
testing_data['Soil_health']= testing_data['Soil_pH']*testing_data['Electrical_Conductivity']
testing_data['Soil_moisture_sqrt']= np.sqrt(testing_data['Soil_Moisture'])

#testing_data['moisture_to_rain_ratio'] = testing_data['Soil_Moisture'] / (testing_data['Rainfall_mm'] + 1e-5)
#testing_data['irrigation_efficiency'] = testing_data['Soil_Moisture'] / (testing_data['Previous_Irrigation_mm'] + 1e-5)
#testing_data['temp_x_rain'] = testing_data['Temperature_C'] * testing_data['Rainfall_mm']
#testing_data['temp_humidity_diff'] = testing_data['Temperature_C'] - testing_data['Humidity']

#non linear features
#testing_data['temp_squared'] = testing_data['Temperature_C'] ** 2
#testing_data['temp_cubed'] = testing_data['Temperature_C'] ** 3
#testing_data['humidity_power'] = testing_data['Humidity'] ** 1.5
#testing_data['soil_moisture_log_sq'] = np.log1p(testing_data['Soil_Moisture']) ** 2
#testing_data['rain_sqrt_temp'] = np.sqrt(testing_data['Rainfall_mm']) * testing_data['Temperature_C']
#testing_data['ec_power'] = testing_data['Electrical_Conductivity'] ** 1.3
#testing_data['ph_squared'] = testing_data['Soil_pH'] ** 2
#testing_data['temp_moisture_nl'] = (testing_data['Temperature_C'] ** 2) * np.sqrt(testing_data['Soil_Moisture'])
#testing_data['log_rain_temp'] = np.log1p(testing_data['Rainfall_mm']) * testing_data['Temperature_C']

#training_data['Crop_Soil_Combo'] = training_data['Crop_Type'].astype(str) + "_" + training_data['Soil_Type'].astype(str)
#training_data['Region_Season_Combo'] = training_data['Region'].astype(str) + "_" + training_data['Season'].astype(str)

#testing_data['Crop_Soil_Combo'] = testing_data['Crop_Type'].astype(str) + "_" + testing_data['Soil_Type'].astype(str)
#testing_data['Region_Season_Combo'] = testing_data['Region'].astype(str) + "_" + testing_data['Season'].astype(str)

training_data.head()

corr_2= training_data.corr(numeric_only=True)

fig_2= px.imshow(corr_2,text_auto='.2f',aspect="auto")
fig_2.show()

x= training_data.drop(columns=['Irrigation_Need'])
y= training_data['Irrigation_Need']

print(x[0:5])
print(y[0:5])

mapping= {'Low':0,"Medium":1,'High':2}
y= y.map(mapping)

print(np.unique(y))

arr,count= np.unique(y,return_counts=True)

print(arr)
print(count)

drop_col= ['id']

x= x.drop(columns=drop_col)

x.info()

numeric_x= x.select_dtypes(include=['float64'])
category_x= x.select_dtypes(include=['object'])

drop_col_2=['Organic_Carbon','EC_rank','Electrical_Conductivity','heat_index_proxy','Soil_pH','Sunlight_Hours','Field_Area_hectare','soil_health_idx','rain_volume','moisture_conductivity','ph_conductivity_ratio','previous_irrigation_volume','Soil_health','temp_humidity_ratio','total_water_supply']

numeric_x= numeric_x.drop(columns=drop_col_2)

numeric_x.info()

#category_x= category_x.drop(columns=['Crop_Soil_Combo','Region_Season_Combo'])

#category_x_2= x[['Crop_Soil_Combo','Region_Season_Combo']]

category_x.info()

#category_x_2.info()

scale= StandardScaler()
numeric_x= scale.fit_transform(numeric_x)

numeric_x[0:3]

from sklearn.preprocessing import OneHotEncoder

encode= OneHotEncoder(sparse_output=False)

#from category_encoders import TargetEncoder

#te= TargetEncoder(cols=categorical_cols,smoothing=0.3)

#category_x_2= te.fit_transform(category_x_2,y)

#category_x_2= np.asarray(category_x_2)

category_x = encode.fit_transform(category_x)

category_x[0:3]

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report

print(numeric_x[0])
print(category_x[0])

print(np.shape(numeric_x))
print(np.shape(category_x))

params = {
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
    'eval_metric': 'mlogloss'
}

#xgb2= XGBClassifier(**params)

#xgb2.fit(numeric_x,y)

#numeric_x_df= pd.DataFrame(numeric_x)

#feat_imp = pd.DataFrame({
 #   'feature': numeric_x_df.columns,
  #  'importance': xgb2.feature_importances_
#}).sort_values(by='importance', ascending=False)

#feat_imp

final_x= np.concatenate((numeric_x,category_x),axis=1)

print(np.shape(final_x))

#x_train_f,x_test_f,y_train_f,y_test_f= train_test_split(final_x,y,test_size=0.20,stratify=y,random_state=42)

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)

class_weights = dict(zip(classes, weights))
class_weights[2] *= 1.22
class_weights[2] *= 1.24
class_weights[2] *= 1.26
class_weights[2] *= 1.28
class_weights[2] *= 1.30
sample_weights = np.array([class_weights[y_i] for y_i in y])

#rf_final= RandomForestClassifier(n_estimators=100,random_state=42)

#rf_final.fit(x_train_f,y_train_f,sample_weight=sample_weights)

#y_pred_f= rf_final.predict(x_test_f)

#accuracy_f= accuracy_score(y_pred_f,y_test_f)
#precision_f= precision_score(y_pred_f,y_test_f,average='weighted')
#recall_f= recall_score(y_pred_f,y_test_f,average='weighted')

#print(accuracy_f*100)
#print(precision_f*100)
#print(recall_f*100)

params = {
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
    'eval_metric': 'mlogloss'
}

#xgb= XGBClassifier(**params)

#xgb.fit(final_x,y,sample_weight=sample_weights)

seeds = [42, 2024, 99]

models = []

for seed in seeds:
    xgb = XGBClassifier(
        **params,
        random_state=seed
    )
    
    xgb.fit(
        final_x, y,
        sample_weight=sample_weights,
        verbose=0
    )
    
    models.append(xgb)

from lightgbm import LGBMClassifier

lgb= LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=63,
    min_child_sample=50,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb.fit(final_x,y,sample_weight=sample_weights)

#categorical_cols=[17,18,19,20,21,22,23,24]

from catboost import CatBoostClassifier

cat = CatBoostClassifier(
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

cat.fit(final_x,y)

#y_pred_xgb= xgb.predict(x_test_f)

#print(accuracy_score(y_pred_xgb,y_test_f)*100)
#print(precision_score(y_pred_xgb,y_test_f,average='weighted')*100)
#print(recall_score(y_pred_xgb,y_test_f,average='weighted')*100)

print(np.shape(y))

processed_data= pd.DataFrame(final_x)

processed_data.head()

#corr_data= processed_data.corr(numeric_only=True)

#figure= px.imshow(corr_data,text_auto='.2f',aspect='auto')
#figure.show()

testing_data.head()

testing_x= testing_data.drop(columns=drop_col)

testing_x= testing_x.drop(columns=drop_col_2)

test_x_numeric= testing_x.select_dtypes(include=['float64'])
test_x_category= testing_x.select_dtypes(include=['object'])

#test_x_category= test_x_category.drop(columns=['Crop_Soil_Combo','Region_Season_Combo'])

#test_x_category_2= testing_x[['Crop_Soil_Combo','Region_Season_Combo']]

test_x_numeric= scale.transform(test_x_numeric)

test_x_category= encode.transform(test_x_category)

#est_x_category_2= te.transform(test_x_category_2)

#test_x_category_2= np.asarray(test_x_category_2)

print(np.shape(test_x_numeric))
print(np.shape(test_x_category))
#print(np.shape(test_x_category_2))

testing_x_combined= np.concatenate((test_x_numeric,test_x_category),axis=1)

xgb_probs = []

for xgb in models:
    p = xgb.predict_proba(testing_x_combined)
    xgb_probs.append(p)

lgb_probs = lgb.predict_proba(testing_x_combined)

cat_probs = cat.predict_proba(testing_x_combined)

weights = [
    (0.88, 0.10, 0.02),
    (0.89, 0.09, 0.02),
    (0.90, 0.08, 0.02),
    (0.91, 0.07, 0.02),

    (0.88, 0.09, 0.03),
    (0.89, 0.08, 0.03),
    (0.90, 0.07, 0.03),

    (0.87, 0.10, 0.03),
]

bias_grid = [
    (1.0, 1.0, 1.005),
    (1.0, 1.0, 1.008),
    (0.995, 1.0, 1.01),
    (1.0, 0.998, 1.01),
    (0.998, 1.0, 1.012)
]

seed_weights = [
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2),
    (0.6, 0.25, 0.15)
]

p1= xgb_probs[0]
p2= xgb_probs[1]
p3= xgb_probs[2]

#avg_xgb_probs = np.mean(xgb_probs, axis=0)
def temp_scale(probs, T):
    probs = probs ** (1/T)
    return probs / probs.sum(axis=1, keepdims=True)

temps = [0.97, 1.0, 1.03]

avg_lgb_probs= np.mean(lgb_probs,axis=0)
avg_cat_probs= np.mean(cat_probs,axis=0)
for sw in seed_weights:
    avg_xgb_probs=sw[0]*p1 + sw[1]*p2 + sw[2]*p3
    for (w1,w2,w3) in weights:
        base_probs= w1*avg_xgb_probs +w2*avg_lgb_probs +w3*avg_cat_probs
        base_probs /= base_probs.sum(axis=1, keepdims=True)
        for T in temps:
            scaled_probs = temp_scale(base_probs, T)
            for (b0, b1, b2) in bias_grid:
                temp = scaled_probs.copy()
            
                temp[:, 0] *= b0
                temp[:, 1] *= b1
                temp[:, 2] *= b2
                y_pred_test = np.argmax(temp, axis=1)

#y_pred_test= xgb.predict(testing_x_combined)

predictions=[]
for p in y_pred_test:
   if p==0:
      predictions.append('Low')
   elif p==1:
      predictions.append('Medium')
   else:
      predictions.append('High')

for i in range(10):
    print(predictions[i])

data= {
    'id': np.asarray(testing_data['id']),
    'irrigation_need': np.asarray(predictions)
}
final_predicted_data= pd.DataFrame(data)

final_predicted_data.head()

final_predicted_data.to_csv('submission.csv',index=False)