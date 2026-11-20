import os
import sys
import time
import json
import tracemalloc
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def run_timing_trials(fn, args_tuple, n_warmup=10, n_trials=50):
    # Warmup
    for _ in range(n_warmup):
        fn(*args_tuple)
    
    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn(*args_tuple)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # Convert to ms
        
    latencies = np.array(latencies)
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'trials': n_trials
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark Prediction Efficiency & Latency")
    parser.add_argument("--trials", type=int, default=50, help="Number of benchmark trials per batch size")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("BENCHMARK 3: PREDICTION EFFICIENCY (Latency, Throughput, and Memory Usage)")
    print("=" * 80)

    # 1. Model Storage & Memory Footprint
    print("\n[1/3] Measuring Model Storage & RAM Allocation...")
    artifacts = {
        'xgb_model': 'artifacts/xgb_model.joblib',
        'lgb_model': 'artifacts/lgb_model.joblib',
        'cat_model': 'artifacts/cat_model.joblib',
        'scaler': 'artifacts/scaler.joblib',
        'encoder': 'artifacts/encoder.joblib',
        'numeric_cols': 'artifacts/numeric_cols.joblib',
        'category_cols': 'artifacts/category_cols.joblib'
    }

    disk_sizes = {k: get_file_size_mb(v) for k, v in artifacts.items()}
    total_disk_mb = sum(disk_sizes.values())

    tracemalloc.start()
    t_start = time.perf_counter()
    xgb_model = joblib.load('artifacts/xgb_model.joblib')
    lgb_model = joblib.load('artifacts/lgb_model.joblib')
    cat_model = joblib.load('artifacts/cat_model.joblib')
    scaler = joblib.load('artifacts/scaler.joblib')
    encoder = joblib.load('artifacts/encoder.joblib')
    numeric_cols = joblib.load('artifacts/numeric_cols.joblib')
    category_cols = joblib.load('artifacts/category_cols.joblib')
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Total Model Storage on Disk: {total_disk_mb:.2f} MB")
    print(f" - XGBoost model:  {disk_sizes['xgb_model']:.2f} MB")
    print(f" - LightGBM model: {disk_sizes['lgb_model']:.2f} MB")
    print(f" - CatBoost model: {disk_sizes['cat_model']:.2f} MB")
    print(f" - Preprocessors:  {disk_sizes['scaler'] + disk_sizes['encoder']:.3f} MB")
    print(f"Peak RAM allocated during pipeline load: {peak_mem / (1024 * 1024):.2f} MB")

    # 2. Batch Latency & Throughput Benchmark
    print("\n[2/3] Benchmarking Inference Latency Across Batch Sizes...")
    batch_sizes = [1, 10, 100, 1000, 10000]
    
    # Pre-generate synthetic batch data matching the 46 features
    n_features = 46
    feature_batches = {
        bs: np.random.randn(bs, n_features).astype(np.float32)
        for bs in batch_sizes
    }

    # Define prediction functions
    def predict_xgb(X):
        return xgb_model.predict_proba(X)

    def predict_lgb(X):
        return lgb_model.predict_proba(X)

    def predict_cat(X):
        return cat_model.predict_proba(X)

    def predict_ensemble(X):
        px = xgb_model.predict_proba(X)
        pl = lgb_model.predict_proba(X)
        pc = cat_model.predict_proba(X)
        p = 0.88 * px + 0.10 * pl + 0.02 * pc
        return np.argmax(p, axis=1)

    efficiency_results = {
        'memory_disk_mb': disk_sizes,
        'total_disk_mb': total_disk_mb,
        'peak_ram_load_mb': peak_mem / (1024 * 1024),
        'batch_benchmarks': {}
    }

    print(f"\nRunning {args.trials} trials per batch size...")
    print("-" * 95)
    print(f"{'Batch':<8} | {'Component':<12} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'Mean (ms)':<10} | {'Per Sample (ms)':<15} | {'Throughput (samp/s)':<20}")
    print("-" * 95)

    for bs in batch_sizes:
        efficiency_results['batch_benchmarks'][bs] = {}
        X_batch = feature_batches[bs]
        n_trials = args.trials if bs <= 1000 else max(15, args.trials // 3)

        models_to_test = [
            ('XGBoost', predict_xgb),
            ('LightGBM', predict_lgb),
            ('CatBoost', predict_cat),
            ('Ensemble', predict_ensemble)
        ]

        for comp_name, comp_fn in models_to_test:
            stats = run_timing_trials(comp_fn, (X_batch,), n_warmup=5, n_trials=n_trials)
            per_sample_ms = stats['mean_ms'] / bs
            throughput = (1000.0 / per_sample_ms) if per_sample_ms > 0 else 0
            
            stats['per_sample_ms'] = per_sample_ms
            stats['throughput_samples_per_sec'] = throughput
            efficiency_results['batch_benchmarks'][bs][comp_name] = stats

            print(
                f"{bs:<8} | "
                f"{comp_name:<12} | "
                f"{stats['p50_ms']:8.2f}ms | "
                f"{stats['p95_ms']:8.2f}ms | "
                f"{stats['mean_ms']:8.2f}ms | "
                f"{per_sample_ms:13.4f}ms | "
                f"{throughput:18.1f} s/sec"
            )
        print("-" * 95)

    # 3. Single-Query End-to-End Latency (UI / API simulation)
    print("\n[3/3] Single-Query End-to-End Latency Benchmark (Preprocessing + Ensemble)...")
    sample_input = {
        'Soil_Type': 'Clay', 'Soil_pH': 6.5, 'Soil_Moisture': 45.0, 
        'Organic_Carbon': 1.5, 'Electrical_Conductivity': 0.8, 
        'Temperature_C': 25.0, 'Humidity': 60.0, 'Rainfall_mm': 10.0, 
        'Sunlight_Hours': 8.0, 'Wind_Speed_kmh': 12.0, 
        'Crop_Type': 'Wheat', 'Crop_Growth_Stage': 'Vegetative', 
        'Season': 'Spring', 'Irrigation_Type': 'Drip', 'Water_Source': 'Well', 
        'Field_Area_hectare': 2.0, 'Mulching_Used': 'Yes', 
        'Previous_Irrigation_mm': 5.0, 'Region': 'North'
    }

    def end_to_end_single(input_dict):
        df = pd.DataFrame([input_dict])
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
        features = np.concatenate((num_scaled, cat_encoded), axis=1)

        px = xgb_model.predict_proba(features)
        pl = lgb_model.predict_proba(features)
        pc = cat_model.predict_proba(features)
        p = 0.88 * px + 0.10 * pl + 0.02 * pc
        classes = ['Low', 'Medium', 'High']
        idx = np.argmax(p, axis=1)[0]
        return classes[idx], p[0][idx]

    e2e_stats = run_timing_trials(end_to_end_single, (sample_input,), n_warmup=10, n_trials=args.trials)
    efficiency_results['end_to_end_single_query'] = e2e_stats

    print(f"End-to-End Single Query Latency: {e2e_stats['mean_ms']:.2f} ms (p50: {e2e_stats['p50_ms']:.2f} ms, p95: {e2e_stats['p95_ms']:.2f} ms)")

    # Save to JSON
    json_path = os.path.join(results_dir, "prediction_efficiency.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(efficiency_results, f, indent=2)
    print(f"\nSaved prediction efficiency metrics to: {json_path}")

if __name__ == "__main__":
    main()
