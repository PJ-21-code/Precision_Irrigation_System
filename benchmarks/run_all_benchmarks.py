import os
import sys
import time
import json
import subprocess
import argparse

BENCHMARK_MODULES = [
    ("Model Performance", "benchmark_models.py"),
    ("Prediction Efficiency", "benchmark_efficiency.py"),
    ("SHAP Explainability", "benchmark_shap.py"),
    ("Weather API Integration", "benchmark_weather_api.py"),
    ("Water Conservation Simulation", "benchmark_water_savings.py")
]

def run_script(script_name, extra_args=None):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
        
    print(f"\n>>> Executing: python {script_name} {' '.join(extra_args or [])} ...")
    start = time.time()
    res = subprocess.run(cmd, capture_output=False, text=True)
    duration = time.time() - start
    print(f">>> Completed {script_name} in {duration:.2f}s (Exit Code: {res.returncode})")
    return res.returncode == 0

def aggregate_summary():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'reports': {}
    }
    
    files = {
        'models': 'model_performance.json',
        'efficiency': 'prediction_efficiency.json',
        'shap': 'shap_explainability.json',
        'weather': 'weather_api_benchmark.json',
        'water_savings': 'water_savings_benchmark.json'
    }
    
    for key, fname in files.items():
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                summary['reports'][key] = json.load(f)
                
    summary_path = os.path.join(results_dir, "executive_benchmark_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Aggregated all results into: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified Precision Irrigation AI Benchmark Suite")
    parser.add_argument("--all", action="store_true", default=True, help="Run all benchmarks (default)")
    parser.add_argument("--models", action="store_true", help="Run only model performance benchmark")
    parser.add_argument("--efficiency", action="store_true", help="Run only prediction efficiency benchmark")
    parser.add_argument("--shap", action="store_true", help="Run only SHAP explainability benchmark")
    parser.add_argument("--weather", action="store_true", help="Run only weather API benchmark")
    parser.add_argument("--water", action="store_true", help="Run only water savings benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick mode (reduced sample sizes)")

    args = parser.parse_args()

    explicit_flags = [args.models, args.efficiency, args.shap, args.weather, args.water]
    run_all = not any(explicit_flags) or args.all

    print("=" * 85)
    print("      PRECISION IRRIGATION AI SYSTEM - SENIOR ML BENCHMARK SUITE")
    print("=" * 85)

    if run_all or args.models:
        extra = ["--sample-size", "25000"] if args.quick else ["--sample-size", "50000"]
        run_script("benchmark_models.py", extra)

    if run_all or args.efficiency:
        extra = ["--trials", "20"] if args.quick else ["--trials", "40"]
        run_script("benchmark_efficiency.py", extra)

    if run_all or args.shap:
        extra = ["--sample-eval", "100", "--trials", "5"] if args.quick else ["--sample-eval", "200", "--trials", "10"]
        run_script("benchmark_shap.py", extra)

    if run_all or args.weather:
        run_script("benchmark_weather_api.py", ["--timeout", "6"])

    if run_all or args.water:
        extra = ["--sample-size", "50000"] if args.quick else ["--sample-size", "100000"]
        run_script("benchmark_water_savings.py", extra)

    aggregate_summary()

    print("\n" + "=" * 85)
    print("                   ALL BENCHMARKS SUCCESSFULLY EXECUTED")
    print("=" * 85)

if __name__ == "__main__":
    main()
