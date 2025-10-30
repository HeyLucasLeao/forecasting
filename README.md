# Time Series Forecasting Analysis

A comprehensive Python project for time series analysis and forecasting, featuring advanced statistical modeling, entropy-based measures, and modern forecasting techniques.

### Data Processing & Analysis
- **Data Preprocessing**: Gap filling, leading zero removal, obsolete series detection
- **Exploratory Data Analysis**: Trend significance testing, stationarity checks, seasonal decomposition
- **Feature Engineering**: Fourier seasonality features, holiday effects, lag features

### Advanced Metrics
- **Forecastability Measures**: ForeCA, Theoretical Limit, Stability Index, ADI CV, Hurst Exponent
- **Entropy Analysis**: Permutation Entropy, Permutation Auto-Mutual Information (PAMI)
- **Complexity Measures**: Novel ordinal pattern-based time series complexity metrics

### Forecasting Models
- **Statistical Models**: AutoARIMA, AutoETS, AutoTheta, AutoCES, SCUM
- **Machine Learning**: MLForecast with lag transforms and feature engineering

### Visualization
- **Interactive Plots**: Plotly-based visualizations for all analyses
- **Diagnostic Plots**: ACF/PACF, residual analysis, stationarity tests
- **Custom PAMI Plots**: Specialized visualizations for entropy measures with confidence bands

## 📦 Installation

### Prerequisites
- Python >= 3.10
- UV package manager (recommended) or pip

### Setup
```bash
# Clone the repository
git clone https://github.com/HeyLucasLeao/forecasting.git
cd forecasting

# Install dependencies with UV
uv sync

# Or with pip
pip install -r requirements.txt
```

## 📊 Key Components

### 1. Data Utilities (`utils.py`)
- `remove_leading_zeros()`: Clean time series data
- `is_obsolete()`: Detect inactive series
- `forecastability()`: Calculate multiple forecastability metrics
- `generate_lag()`: Create lag features

### 2. PAMI Implementation
Novel Permutation Auto-Mutual Information implementation for measuring temporal dependencies in time series using ordinal patterns.

### 3. Notebooks
- `1.air_passangers.ipynb`: Classic airline passengers analysis
- `2.sythentic_data.ipynb`: Comprehensive synthetic data analysis with PAMI


## Advanced Feature

### Permutation Auto-Mutual Information (PAMI)
A robust measure of temporal dependence that:
- Uses ordinal patterns instead of raw values
- Resistant to noise and outliers
- Captures non-linear dependencies
- Provides minimum criterion for optimal lag selection

```python
# Calculate PAMI for lag τ
pami_value = permutation_auto_mutual_information(
    X=time_series, 
    tau=1,           # lag
    m=3,            # embedding dimension
    delay=1         # embedding delay
)
```


