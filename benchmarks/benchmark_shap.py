import os
import sys
import time
import json
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def main():
    parser = argparse.ArgumentParser(description="Benchmark SHAP Explainability Generation")
    parser.add_argument("--trials", type=int, default=15, help="Number of benchmark trials for explanation time")
    parser.add_argument("--sample-eval", type=int, default=200, help="Number of samples to compute global SHAP feature importance")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("BENCHMARK 4: SHAP EXPLAINABILITY (Latency & Feature Attribution)")
    print("=" * 80)

    # 1. Load XGBoost model & preprocessors
    print("\n[1/4] Loading XGBoost model and preprocessors...")
    xgb_model = joblib.load('artifacts/xgb_model.joblib')
    scaler = joblib.load('artifacts/scaler.joblib')
    encoder = joblib.load('artifacts/encoder.joblib')
    numeric_cols = joblib.load('artifacts/numeric_cols.joblib')
    category_cols = joblib.load('artifacts/category_cols.joblib')

    feature_names = numeric_cols + list(encoder.get_feature_names_out(category_cols))
    print(f"Total features: {len(feature_names)}")

    # 2. Benchmark TreeExplainer initialization
    print("\n[2/4] Initializing SHAP TreeExplainer...")
    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(xgb_model)
    t_init = (time.perf_counter() - t0) * 1000.0
    print(f"TreeExplainer initialization time: {t_init:.2f} ms")

    # 3. Benchmark explanation computation latency
    print("\n[3/4] Measuring explanation latency across sample sizes...")
    df_raw = pd.read_csv('train.csv', nrows=args.sample_eval + 50)
    
    # Preprocess
    df = df_raw.copy()
    df['temp_humidity_ratio'] = df['Temperature_C'] / (df['Humidity'] + 1e-5)
    df['rainfall_temp_ratio'] = df['Rainfall_mm'] / (df['Temperature_C'] + 1e-5)
    df['heat_index_proxy'] = df['Temperature_C'] * df['Sunlight_Hours']
    df['wind_chill_proxy'] = df['Temperature_C'] * df['Wind_Speed_kmh']

    df['soil_health_idx'] = df['Organic_Carbon'] * df['Soil_Moisture'] / (df['Soil_pH'] + 1e-5)
    df['moisture_conductivity'] = df['Soil_Moisture'] * df['Electrical_Conductivity']
    df['air_soil_moisture_mismatch'] = df['Humidity'] / (df['Soil_Moisture'] + 1e-5)
    df['ph_conductivity_ratio'] = df['Soil_pH'] / (df['Electrical_Conductivity'] + 1e-5)

    df['rain_volume'] = df['Rainfall_mm'] * df['Field_Area_hectare']
    df['previous_irrigation_volume'] = df['Previous_Irrigation_mm'] * df['Field_Area_hectare']
    df['irrigation_diff'] = df['Rainfall_mm'] - df['Previous_Irrigation_mm']
    df['total_water_supply'] = df['Rainfall_mm'] + df['Previous_Irrigation_mm']

    df['log_Rainfall_mm'] = np.log1p(df['Rainfall_mm'])
    df['log_Previous_Irrigation_mm'] = np.log1p(df['Previous_Irrigation_mm'])

    df['water_balance'] = df['Rainfall_mm'] - df['Soil_Moisture']
    df['Soil_health'] = df['Soil_pH'] * df['Electrical_Conductivity']
    df['Soil_moisture_sqrt'] = np.sqrt(df['Soil_Moisture'])

    num_scaled = scaler.transform(df[numeric_cols])
    cat_encoded = encoder.transform(df[category_cols])
    X_all = np.concatenate((num_scaled, cat_encoded), axis=1)

    # Warmup
    _ = explainer.shap_values(X_all[:1])

    shap_latency_benchmarks = {}
    sample_sizes = [1, 5, 10, 50, 100]

    for ss in sample_sizes:
        latencies = []
        X_sub = X_all[:ss]
        n_reps = args.trials if ss <= 10 else max(5, args.trials // 3)
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ = explainer.shap_values(X_sub)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        
        latencies = np.array(latencies)
        mean_ms = float(np.mean(latencies))
        p50_ms = float(np.percentile(latencies, 50))
        p95_ms = float(np.percentile(latencies, 95))
        per_sample_ms = mean_ms / ss

        shap_latency_benchmarks[ss] = {
            'mean_ms': mean_ms,
            'p50_ms': p50_ms,
            'p95_ms': p95_ms,
            'per_sample_ms': per_sample_ms,
            'trials': n_reps
        }

        print(f" - Batch {ss:3d} samples: Mean={mean_ms:7.2f} ms | p50={p50_ms:7.2f} ms | p95={p95_ms:7.2f} ms | Per Sample={per_sample_ms:6.2f} ms")

    # 4. Feature Importance Extraction (Global Attribution)
    print(f"\n[4/4] Computing Global Feature Attribution across {args.sample_eval} samples...")
    t0 = time.perf_counter()
    X_eval = X_all[:args.sample_eval]
    sv = explainer.shap_values(X_eval) # shape: (n_samples, n_features, n_classes) or list of (n_samples, n_features)
    print(f"SHAP values computed in {(time.perf_counter() - t0):.2f}s.")

    # Determine shape
    if isinstance(sv, list):
        # 3 classes: list of 3 arrays of shape (N, F)
        # Global mean absolute SHAP across all classes and samples
        mean_abs_per_class = [np.mean(np.abs(sv[c]), axis=0) for c in range(len(sv))]
        mean_abs_global = np.mean(mean_abs_per_class, axis=0)
    elif len(sv.shape) == 3:
        # (N, F, C)
        mean_abs_per_class = [np.mean(np.abs(sv[:, :, c]), axis=0) for c in range(sv.shape[2])]
        mean_abs_global = np.mean(np.abs(sv), axis=(0, 2))
    else:
        mean_abs_global = np.mean(np.abs(sv), axis=0)
        mean_abs_per_class = [mean_abs_global]

    # Rank features
    feature_ranking = []
    class_names = ['Low', 'Medium', 'High']
    for idx, name in enumerate(feature_names):
        entry = {
            'feature': name,
            'mean_abs_shap_global': float(mean_abs_global[idx]),
        }
        for c_idx, c_name in enumerate(class_names):
            if c_idx < len(mean_abs_per_class):
                entry[f'mean_abs_shap_{c_name}'] = float(mean_abs_per_class[c_idx][idx])
        feature_ranking.append(entry)

    feature_ranking.sort(key=lambda x: x['mean_abs_shap_global'], reverse=True)

    print("\n" + "=" * 80)
    print(f"{'TOP 12 MOST INFLUENTIAL ENVIRONMENTAL & SOIL FEATURES':^80}")
    print("=" * 80)
    print(f"{'Rank':<5} | {'Feature':<28} | {'Global |SHAP|':<14} | {'High Class |SHAP|':<18} | {'Low Class |SHAP|':<16}")
    print("-" * 80)
    for r, item in enumerate(feature_ranking[:12], start=1):
        print(
            f"{r:<5} | "
            f"{item['feature']:<28} | "
            f"{item['mean_abs_shap_global']:14.4f} | "
            f"{item.get('mean_abs_shap_High', 0.0):18.4f} | "
            f"{item.get('mean_abs_shap_Low', 0.0):16.4f}"
        )
    print("=" * 80)

    # Compile JSON report
    report = {
        'tree_explainer_init_ms': t_init,
        'explanation_latency': shap_latency_benchmarks,
        'top_features_ranked': feature_ranking[:20]
    }

    json_path = os.path.join(results_dir, "shap_explainability.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved SHAP explainability metrics to: {json_path}")

if __name__ == "__main__":
    main()
