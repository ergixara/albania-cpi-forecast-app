# 📊 Albania CPI Forecasting App

A Streamlit web application for time-series forecasting of Albania's Consumer Price Index (CPI) using SARIMA, Prophet, and XGBoost models.

Based on: *"Forecasting Consumer Price Index with ARIMA, Prophet and XGboost: A Comparative Analysis"* by Basha and Gjika


## 📋 Data Format

Upload an Excel file with the following structure:
- **Rows 0-2**: Headers/empty
- **Row 3**: Date columns (starting from column 1, format: YYYY-MM)
- **Rows 4+**: Category codes and CPI values
- Use category code `000000` for Total CPI

## 🤖 Models Implemented

1. **SARIMA** - Seasonal ARIMA benchmark
2. **Boosted SARIMA** - SARIMA + XGBoost on residuals
3. **Prophet** - Facebook's forecasting tool
4. **Prophet Boost** - Prophet + XGBoost 


## 🎯 Features

- Interactive data visualization
- 4 forecasting models with comparison
- 36-month future predictions
- Performance metrics (RMSE, MAE, MAPE, MASE)
- Downloadable results and forecasts

## 📝 License

MIT License
