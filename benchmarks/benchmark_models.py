import os
import sys
import time
import json
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def preprocess_dataframe(df, scaler, encoder, numeric_cols, category_cols):
    """
    Reproduces exact feature engineering pipeline from train_model.py and app.py
    """
    data = df.copy()
    data['temp_humidity_ratio'] = data['Temperature_C'] / (data['Humidity'] + 1e-5)
    data['rainfall_temp_ratio'] = data['Rainfall_mm'] / (data['Temperature_C'] + 1e-5)
    data['heat_index_proxy'] = data['Temperature_C'] * data['Sunlight_Hours']
    data['wind_chill_proxy'] = data['Temperature_C'] * data['Wind_Speed_kmh']

    data['soil_health_idx'] = data['Organic_Carbon'] * data['Soil_Moisture'] / (data['Soil_pH'] + 1e-5)
    data['moisture_conductivity'] = data['Soil_Moisture'] * data['Electrical_Conductivity']
    data['air_soil_moisture_mismatch'] = data['Humidity'] / (data['Soil_Moisture'] + 1e-5)
    data['ph_conductivity_ratio'] = data['Soil_pH'] / (data['Electrical_Conductivity'] + 1e-5)

    data['rain_volume'] = data['Rainfall_mm'] * data['Field_Area_hectare']
    data['previous_irrigation_volume'] = data['Previous_Irrigation_mm'] * data['Field_Area_hectare']
    data['irrigation_diff'] = data['Rainfall_mm'] - data['Previous_Irrigation_mm']
    data['total_water_supply'] = data['Rainfall_mm'] + data['Previous_Irrigation_mm']

    data['log_Rainfall_mm'] = np.log1p(data['Rainfall_mm'])
    data['log_Previous_Irrigation_mm'] = np.log1p(data['Previous_Irrigation_mm'])

    data['water_balance'] = data['Rainfall_mm'] - data['Soil_Moisture']
    data['Soil_health'] = data['Soil_pH'] * data['Electrical_Conductivity']
    data['Soil_moisture_sqrt'] = np.sqrt(data['Soil_Moisture'])

    num_data = data[numeric_cols]
    cat_data = data[category_cols]

    num_scaled = scaler.transform(num_data)
    cat_encoded = encoder.transform(cat_data)
    final_x = np.concatenate((num_scaled, cat_encoded), axis=1)
    return final_x

def evaluate_predictions(y_true, y_pred, y_prob=None, class_names=None):
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted'))
    macro_prec = float(precision_score(y_true, y_pred, average='macro'))
    weighted_prec = float(precision_score(y_true, y_pred, average='weighted'))
    macro_rec = float(recall_score(y_true, y_pred, average='macro'))
    weighted_rec = float(recall_score(y_true, y_pred, average='weighted'))
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    # Class-wise metrics
    class_metrics = {}
    cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    for c in class_names:
        class_metrics[c] = {
            'precision': float(cr[c]['precision']),
            'recall': float(cr[c]['recall']),
            'f1_score': float(cr[c]['f1-score']),
            'support': int(cr[c]['support'])
        }
        
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_prec,
        'weighted_precision': weighted_prec,
        'macro_recall': macro_rec,
        'weighted_recall': weighted_rec,
        'confusion_matrix': cm,
        'class_metrics': class_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark Model Performance")
    parser.add_argument("--sample-size", type=int, default=50000, 
                        help="Sample size for evaluation (default: 50,000 for fast reproducible benchmark; set 0 for all 630,000)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("BENCHMARK 1: MODEL PERFORMANCE (XGBoost vs LightGBM vs CatBoost vs Ensemble)")
    print("=" * 70)

    # 1. Load artifacts
    print("\n[1/4] Loading model artifacts from 'artifacts/'...")
    t0 = time.time()
    xgb_model = joblib.load('artifacts/xgb_model.joblib')
    lgb_model = joblib.load('artifacts/lgb_model.joblib')
    cat_model = joblib.load('artifacts/cat_model.joblib')
    scaler = joblib.load('artifacts/scaler.joblib')
    encoder = joblib.load('artifacts/encoder.joblib')
    numeric_cols = joblib.load('artifacts/numeric_cols.joblib')
    category_cols = joblib.load('artifacts/category_cols.joblib')
    print(f"Loaded artifacts in {time.time() - t0:.2f} seconds.")

    # 2. Load dataset
    print("\n[2/4] Loading train.csv...")
    t0 = time.time()
    df_raw = pd.read_csv('train.csv')
    print(f"Loaded {len(df_raw):,} records in {time.time() - t0:.2f} seconds.")

    mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    class_names = ['Low', 'Medium', 'High']

    if args.sample_size > 0 and args.sample_size < len(df_raw):
        print(f"Sampling {args.sample_size:,} records with stratified sampling (seed={args.random_seed})...")
        df_eval = df_raw.groupby('Irrigation_Need', group_keys=False).apply(
            lambda x: x.sample(int(np.rint(args.sample_size * len(x) / len(df_raw))), random_state=args.random_seed)
        ).reset_index(drop=True)
        eval_scope = f"Stratified Sample ({len(df_eval):,} records)"
    else:
        df_eval = df_raw
        eval_scope = f"Full Dataset ({len(df_eval):,} records)"

    print(f"Evaluation dataset shape: {df_eval.shape}")
    print("Target distribution in eval set:")
    print(df_eval['Irrigation_Need'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%").to_dict())

    # 3. Preprocessing
    print("\n[3/4] Transforming features...")
    t0 = time.time()
    x_eval = preprocess_dataframe(df_eval, scaler, encoder, numeric_cols, category_cols)
    y_true = df_eval['Irrigation_Need'].map(mapping).values
    print(f"Feature matrix shape: {x_eval.shape} (transformed in {time.time() - t0:.2f}s)")

    # 4. Model Inference & Evaluation
    print("\n[4/4] Running model predictions & scoring...")
    
    # XGBoost
    print(" - Evaluating XGBoost...")
    t_start = time.time()
    p_xgb = xgb_model.predict_proba(x_eval)
    t_xgb = time.time() - t_start
    y_xgb = np.argmax(p_xgb, axis=1)
    res_xgb = evaluate_predictions(y_true, y_xgb, p_xgb, class_names)
    res_xgb['inference_time_sec'] = t_xgb

    # LightGBM
    print(" - Evaluating LightGBM...")
    t_start = time.time()
    p_lgb = lgb_model.predict_proba(x_eval)
    t_lgb = time.time() - t_start
    y_lgb = np.argmax(p_lgb, axis=1)
    res_lgb = evaluate_predictions(y_true, y_lgb, p_lgb, class_names)
    res_lgb['inference_time_sec'] = t_lgb

    # CatBoost
    print(" - Evaluating CatBoost...")
    t_start = time.time()
    p_cat = cat_model.predict_proba(x_eval)
    t_cat = time.time() - t_start
    y_cat = np.argmax(p_cat, axis=1)
    res_cat = evaluate_predictions(y_true, y_cat, p_cat, class_names)
    res_cat['inference_time_sec'] = t_cat

    # Soft Voting Ensemble (Weights: 0.88 XGB, 0.10 LGB, 0.02 CAT as defined in app.py and notebook)
    print(" - Evaluating Soft-Voting Ensemble (0.88 XGB + 0.10 LGB + 0.02 CAT)...")
    t_start = time.time()
    p_ens = (0.88 * p_xgb) + (0.10 * p_lgb) + (0.02 * p_cat)
    y_ens = np.argmax(p_ens, axis=1)
    t_ens = t_xgb + t_lgb + t_cat + (time.time() - t_start)
    res_ens = evaluate_predictions(y_true, y_ens, p_ens, class_names)
    res_ens['inference_time_sec'] = t_ens

    # Compile Summary
    summary_data = {
        'evaluation_scope': eval_scope,
        'evaluation_samples': len(df_eval),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'models': {
            'XGBoost': res_xgb,
            'LightGBM': res_lgb,
            'CatBoost': res_cat,
            'Ensemble': res_ens
        }
    }

    # Save to JSON
    json_path = os.path.join(results_dir, "model_performance.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved detailed metrics to: {json_path}")

    # Display comparison table
    print("\n" + "=" * 90)
    print(f"{'MODEL PERFORMANCE COMPARISON TABLE':^90}")
    print(f"{'Evaluation Scope: ' + eval_scope:^90}")
    print("=" * 90)
    header = f"{'Model':<12} | {'Accuracy':<10} | {'Bal. Acc.':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'High Recall':<12} | {'Latency (s)':<10}"
    print(header)
    print("-" * 90)
    for name, res in [('XGBoost', res_xgb), ('LightGBM', res_lgb), ('CatBoost', res_cat), ('Ensemble', res_ens)]:
        high_rec = res['class_metrics']['High']['recall']
        row = (
            f"{name:<12} | "
            f"{res['accuracy']*100:8.2f}% | "
            f"{res['balanced_accuracy']*100:8.2f}% | "
            f"{res['macro_f1']*100:8.2f}% | "
            f"{res['weighted_f1']*100:10.2f}% | "
            f"{high_rec*100:10.2f}% | "
            f"{res['inference_time_sec']:8.3f}s"
        )
        print(row)
    print("=" * 90)

    # Class-wise breakdown for ensemble
    print("\n" + "-" * 70)
    print(f"{'ENSEMBLE CLASS-WISE METRICS BREAKDOWN':^70}")
    print("-" * 70)
    print(f"{'Class':<10} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<10}")
    print("-" * 70)
    for c in class_names:
        cm = res_ens['class_metrics'][c]
        print(f"{c:<10} | {cm['precision']*100:10.2f}% | {cm['recall']*100:10.2f}% | {cm['f1_score']*100:10.2f}% | {cm['support']:<10,}")
    print("-" * 70)
    print("\nEnsemble Confusion Matrix (rows: True, cols: Pred [Low, Med, High]):")
    for row in res_ens['confusion_matrix']:
        print("  ", row)

if __name__ == "__main__":
    main()
