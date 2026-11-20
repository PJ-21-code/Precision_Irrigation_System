import os
import sys
import time
import json
import warnings
import argparse
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

LOCATIONS_TO_TEST = [
    "New Delhi",
    "Des Moines",
    "Fresno",
    "Kyiv",
    "Nairobi",
    "Sao Paulo",
    "Punjab",
    "Cairo"
]

def benchmark_single_location(location, timeout=10):
    t0 = time.perf_counter()
    geo_success = False
    geo_status = None
    geo_time_ms = 0.0
    lat, lon, resolved_name = None, None, None

    # Step 1: Geocoding
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&format=json"
    t_geo_start = time.perf_counter()
    try:
        r_geo = requests.get(geo_url, timeout=timeout)
        geo_status = r_geo.status_code
        geo_time_ms = (time.perf_counter() - t_geo_start) * 1000.0
        if r_geo.status_code == 200:
            data = r_geo.json()
            if data.get('results'):
                lat = data['results'][0]['latitude']
                lon = data['results'][0]['longitude']
                resolved_name = f"{data['results'][0]['name']}, {data['results'][0].get('country', '')}"
                geo_success = True
    except Exception as e:
        geo_time_ms = (time.perf_counter() - t_geo_start) * 1000.0
        geo_status = str(e)

    # Step 2: Forecast
    weather_success = False
    weather_status = None
    weather_time_ms = 0.0
    forecast_days = 0
    precip_sum_3d = 0.0

    if geo_success and lat is not None and lon is not None:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max&timezone=auto&forecast_days=3"
        t_w_start = time.perf_counter()
        try:
            r_w = requests.get(weather_url, timeout=timeout)
            weather_status = r_w.status_code
            weather_time_ms = (time.perf_counter() - t_w_start) * 1000.0
            if r_w.status_code == 200:
                w_data = r_w.json()
                if 'daily' in w_data and 'precipitation_sum' in w_data['daily']:
                    precip_sum_3d = sum(w_data['daily']['precipitation_sum'])
                    forecast_days = len(w_data['daily']['time'])
                    weather_success = True
        except Exception as e:
            weather_time_ms = (time.perf_counter() - t_w_start) * 1000.0
            weather_status = str(e)

    total_time_ms = (time.perf_counter() - t0) * 1000.0
    overall_success = geo_success and weather_success

    return {
        'location_query': location,
        'resolved_name': resolved_name,
        'geo_success': geo_success,
        'geo_status': geo_status,
        'geo_time_ms': geo_time_ms,
        'weather_success': weather_success,
        'weather_status': weather_status,
        'weather_time_ms': weather_time_ms,
        'total_time_ms': total_time_ms,
        'overall_success': overall_success,
        'forecast_days_retrieved': forecast_days,
        'forecast_3d_rain_mm': precip_sum_3d
    }

def evaluate_irrigation_adjustment_curve():
    """
    Evaluates the application's weather-adjustment logic:
    base = {'High': 25, 'Medium': 15, 'Low': 5}[pred]
    adjusted = max(0, base - rain)
    if rain > 10: adjusted = 0
    """
    rainfall_levels = list(range(0, 31)) # 0 to 30 mm
    curves = {'High': [], 'Medium': [], 'Low': []}
    
    base_volumes = {'High': 25, 'Medium': 15, 'Low': 5}
    
    for r in rainfall_levels:
        for cat, base in base_volumes.items():
            if r > 10:
                adj = 0.0
                decision = "Skip (Rain > 10mm)"
            else:
                adj = float(max(0, base - r))
                decision = f"Adjusted to {adj:.1f} mm" if adj > 0 else "Skip (0 mm)"
            
            savings_pct = ((base - adj) / base) * 100.0
            curves[cat].append({
                'rainfall_mm': r,
                'base_irrigation_mm': base,
                'adjusted_irrigation_mm': adj,
                'water_savings_pct': savings_pct,
                'decision': decision
            })
            
    return curves

def main():
    parser = argparse.ArgumentParser(description="Benchmark Open-Meteo Weather API Integration")
    parser.add_argument("--timeout", type=int, default=8, help="HTTP request timeout in seconds")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 85)
    print("BENCHMARK 5: WEATHER API INTEGRATION & FORECAST-DRIVEN IRRIGATION RULES")
    print("=" * 85)

    # 1. API Benchmark across locations
    print("\n[1/2] Probing Open-Meteo Geocoding and Forecast API across 8 global agricultural hubs...")
    print("-" * 85)
    print(f"{'Location':<14} | {'Status':<10} | {'Geo Time':<12} | {'Weather Time':<14} | {'Total Latency':<15} | {'3-Day Rain':<10}")
    print("-" * 85)

    location_results = []
    latencies = []
    success_count = 0

    for loc in LOCATIONS_TO_TEST:
        res = benchmark_single_location(loc, timeout=args.timeout)
        location_results.append(res)
        status_str = "SUCCESS" if res['overall_success'] else "FAILED/OFFLINE"
        if res['overall_success']:
            success_count += 1
            latencies.append(res['total_time_ms'])
            print(
                f"{loc:<14} | "
                f"{status_str:<10} | "
                f"{res['geo_time_ms']:10.1f}ms | "
                f"{res['weather_time_ms']:12.1f}ms | "
                f"{res['total_time_ms']:13.1f}ms | "
                f"{res['forecast_3d_rain_mm']:8.1f}mm"
            )
        else:
            print(
                f"{loc:<14} | "
                f"{status_str:<10} | "
                f"{res['geo_time_ms']:10.1f}ms | "
                f"{res['weather_time_ms']:12.1f}ms | "
                f"{res['total_time_ms']:13.1f}ms | "
                f"N/A"
            )

    print("-" * 85)
    success_rate = (success_count / len(LOCATIONS_TO_TEST)) * 100.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    print(f"API Success Rate: {success_count}/{len(LOCATIONS_TO_TEST)} ({success_rate:.1f}%)")
    if latencies:
        print(f"API Latency Statistics: Mean={mean_lat:.1f} ms | Median(p50)={p50_lat:.1f} ms | p95={p95_lat:.1f} ms")
    else:
        print("Note: Network access was restricted/offline; API returned local failure or timeout as expected in isolated sandboxes.")

    # 2. Forecast-Driven Irrigation Adjustment Analysis
    print("\n[2/2] Evaluating Forecast-Driven Rainfall Offset Logic...")
    curves = evaluate_irrigation_adjustment_curve()

    print("\nSimulated Irrigation Volume Adjustment Curve (Sample Rainfall Levels):")
    print("-" * 75)
    print(f"{'Rain (mm)':<10} | {'High Need (25mm)':<20} | {'Med Need (15mm)':<20} | {'Low Need (5mm)':<18}")
    print("-" * 75)
    for r in [0, 2, 5, 8, 10, 11, 15, 25]:
        h_row = [c for c in curves['High'] if c['rainfall_mm'] == r][0]
        m_row = [c for c in curves['Medium'] if c['rainfall_mm'] == r][0]
        l_row = [c for c in curves['Low'] if c['rainfall_mm'] == r][0]
        print(
            f"{r:<10} | "
            f"{h_row['adjusted_irrigation_mm']:5.1f} mm (-{h_row['water_savings_pct']:4.0f}%)      | "
            f"{m_row['adjusted_irrigation_mm']:5.1f} mm (-{m_row['water_savings_pct']:4.0f}%)      | "
            f"{l_row['adjusted_irrigation_mm']:5.1f} mm (-{l_row['water_savings_pct']:4.0f}%)"
        )
    print("-" * 75)
    print("Key Finding: For rainfall > 10 mm, rule-based cutoff guarantees 100% water offset (skip irrigation).")

    # Save to JSON
    report = {
        'api_benchmarks': {
            'tested_locations': LOCATIONS_TO_TEST,
            'success_rate_pct': success_rate,
            'successful_calls': success_count,
            'total_calls': len(LOCATIONS_TO_TEST),
            'mean_latency_ms': mean_lat,
            'median_latency_ms': p50_lat,
            'p95_latency_ms': p95_lat,
            'individual_location_results': location_results
        },
        'rainfall_adjustment_curves': curves
    }

    json_path = os.path.join(results_dir, "weather_api_benchmark.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved Weather API benchmark results to: {json_path}")

if __name__ == "__main__":
    main()
