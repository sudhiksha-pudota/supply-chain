# Walmart Sales Forecasting System

https://supply-chain-rappczqchvo9v9elnfdgopg.streamlit.app/

## 📌Overview

This project presents a comprehensive **Sales Forecasting System** designed for a retail giant like Walmart. It moves beyond traditional one-size-fits-all forecasting by developing and comparing department-specific models. The system provides an interactive dashboard to visualize forecasts, analyze model performance, and quantify the financial impact of prediction accuracy on inventory costs.

##  Core Features

### Department-Specific Forecasting
- Analyzes and selects the optimal model (Naive, XGBoost, Ensemble) for each unique store-department combination based on its historical sales patterns
- Calculates volatility coefficients and holiday lift percentages for granular insights

### Interactive Analytics Dashboard
A multi-page Streamlit application for deep-dive analysis:

| Page | Description |
|------|-------------|
| **Performance Dashboard** | Executive-level view of sales trends and model performance with interactive filters for date ranges, stores, departments, and holiday periods |
| **Department Analyzer** | Deep dive into individual department metrics including volatility, holiday lift, and feature importance visualization |
| **Forecast Predictor** | Generate real-time, on-demand sales predictions by adjusting key parameters like temperature, fuel price, CPI, and holiday status |
| **Outlier Detection** | Statistically identifies departments with unusual sales patterns (using Z-scores and IQR methods) that require special attention |
| **Inventory Impact** | Translates forecast errors into tangible financial costs—stockout and holding costs |
| **Business Value** | Demonstrates ROI potential through improved forecasting accuracy with dollar-figure savings |
| **Performance Reports** | Detailed model comparisons and data export capabilities |

### Business Impact Quantification
- Converts technical accuracy metrics (MAPE) into actual dollar figures
- Calculates stockout costs (lost sales at 2× profit margin)
- Calculates holding costs (excess inventory carrying charges at 25%)
- Provides clear ROI justification for forecasting improvements

### Model Transparency
- Visualizes feature importance to show which factors drive predictions
- Top features include lagged sales, holiday indicators, fuel prices, and unemployment rates
- Builds trust in machine learning outputs through explainability

### Smart Analytics
- Automated outlier detection using statistical methods
- Volatility-based model recommendations (low to very high CV)
- Holiday impact measurement across different departments
- Real-time prediction generation with adjustable parameters

##  Technology Stack

| Component | Technology |
|-----------|------------|
| **Web Application Framework** | [Streamlit](https://streamlit.io/) |
| **Core Programming Language** | [Python](https://www.python.org/) (3.12) |
| **Data Manipulation** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Data Visualization** | [Plotly](https://plotly.com/python/) (Express & Graph Objects) |
| **Machine Learning** | [XGBoost](https://xgboost.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/) |
| **Model Serialization** | `pickle` |
| **Date/Time Handling** | `datetime` |
| **Deployment** | Streamlit Community Cloud |

##  Live Demo & Project Links

- **Live Application**: [Walmart Sales Forecasting App](https://supply-chain-rappczqchvo9v9elnfdgopg.streamlit.app/)
- **Source Code Repository**: [sudhiksha-pudota/supply-chain](https://github.com/sudhiksha-pudota/supply-chain)

## 💻 Local Installation & Setup

### Prerequisites
- [Python](https://www.python.org/downloads/) (version 3.8 or higher)
- `pip` package manager

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sudhiksha-pudota/supply-chain.git
   cd supply-chain
2.**Create virtual environment**

       ```bash
       python -m venv venv
    
       # On Windows
       .\venv\Scripts\activate
       
       # On macOS/Linux
       source venv/bin/activate
   
      ```bash
      pip install -r requirements.txt

##Software Requirements
Operating System: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)

Python Version: 3.12.12 (virtual environment recommended)

Browser: Chrome 90+, Firefox 88+, Edge 90+, or Safari 14+ (JavaScript enabled)
