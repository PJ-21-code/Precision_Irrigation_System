# Precision Irrigation AI System 🌾

## Overview
The **Precision Irrigation AI System** is an advanced, data-driven web application designed to optimize water usage in agriculture. By leveraging state-of-the-art machine learning models and real-time weather forecasting, the system provides accurate, actionable recommendations on when and how much to irrigate crops.

### Here is the deployed streamlit app of Precision Irrigation System
https://precisionirrigationsystem-smrf36wqojgf34wlsayvy3.streamlit.app/
- It may take five minutes to wake up due to streamlit cloud free tier
## Technologies Used
- **Frontend & UI:** Streamlit
- **Machine Learning Models:** XGBoost, LightGBM, CatBoost
- **Explainable AI:** SHAP (SHapley Additive exPlanations)
- **Data Processing:** Pandas, NumPy, Scikit-Learn
- **APIs:** Open-Meteo API (for real-time weather forecasting and geocoding)
- **Model Serialization:** Joblib

## System Architecture

```mermaid
flowchart TD
    subgraph UI ["1. Presentation Layer (Streamlit)"]
        A["User Input / Farm Parameters<br/>(Soil, Weather, Crop & Field Data)"]
        B["Location Query<br/>(City / Region)"]
        UI_Dash["Interactive Dashboard<br/>(Recommendations, Simulation, Forecast, SHAP)"]
    end

    subgraph External ["2. External Services"]
        C["Open-Meteo Geocoding API"]
        D["Open-Meteo Weather Forecast API<br/>(3-Day Forecast: Rain, Temp, Wind)"]
    end

    subgraph DataPrep ["3. Data Preprocessing & Feature Engineering"]
        E["Data Aggregation & Cleaning<br/>(Pandas & NumPy)"]
        F["Feature Engineering<br/>(Ratios, Health Indexes, Interaction Terms)"]
        G["Transformation Pipeline<br/>(StandardScaler & OneHotEncoder)"]
    end

    subgraph Models ["4. Machine Learning Ensemble"]
        direction TB
        M1["XGBoost Model (Weight: 88%)"]
        M2["LightGBM Model (Weight: 10%)"]
        M3["CatBoost Model (Weight: 2%)"]
        Voter["Weighted Soft-Voting Classifier<br/>(Probability Aggregation)"]
    end

    subgraph XAI ["5. Explainability Layer"]
        SHAP_Engine["SHAP TreeExplainer"]
        SHAP_Plot["SHAP Waterfall & Feature Impact Plots"]
    end

    subgraph Advisory ["6. Decision & Advisory Engine"]
        Decision["Irrigation Need Classification<br/>(Low / Medium / High + Confidence %)"]
        Plan["Weather-Adjusted Irrigation Plan<br/>(Volume Calculation & Rainfall Offsets)"]
    end

    %% Data Flow Connections
    B --> C --> D --> E
    A --> E
    E --> F --> G
    G --> M1 & M2 & M3
    M1 & M2 & M3 --> Voter
    Voter --> Decision
    Decision --> Plan
    G --> SHAP_Engine --> SHAP_Plot
    Decision --> UI_Dash
    Plan --> UI_Dash
    SHAP_Plot --> UI_Dash
```

## What is Accomplished
- **Ensemble Prediction:** Combines the predictive power of multiple robust gradient boosting algorithms (XGBoost, LightGBM, CatBoost) to achieve an estimated 89.5% accuracy in irrigation recommendations.
- **Real-Time Forecast Integration:** Connects seamlessly with the Open-Meteo API to fetch a 3-day weather forecast based on a user-provided location. The system intelligently adjusts irrigation plans by factoring in predicted rainfall.
- **Interactive Simulation:** Allows users to adjust environmental parameters (Temperature, Humidity, Soil Moisture, Rainfall) via an intuitive slider interface to see how real-time changes impact irrigation requirements.
- **AI Explainability:** Demystifies the "black box" of machine learning by generating SHAP waterfall plots, showing precisely which environmental factors drove the model's decision.
- **Water Conservation:** Aims to reduce water waste, estimating a 34% saving in water usage compared to traditional irrigation methods.

