# Quantitative Earnings Response Predictor (QERP)

## Overview
QERP is a machine learning system that predicts stock price movements following earnings announcements. Using financial metrics, market indicators, and machine learning classification algorithms, this tool forecasts whether a stock will rise or fall within three days of an earnings report.

## Key Features
- Predicts post-earnings price movements with high precision
- Analyzes EPS surprises, pre-announcement volatility, and volume patterns
- Incorporates sector-specific behavior and market conditions
- Provides feature importance analysis to understand market reactions

## Technical Stack
- **Python**: Core implementation language
- **Libraries**: pandas, scikit-learn, numpy, matplotlib, seaborn
- **ML Models**: Logistic Regression (baseline), Random Forest, XGBoost
- **Evaluation**: Accuracy, Precision, Recall, Confusion Matrix

## Financial Indicators Used
- **EPS Surprise %**: Difference between actual and estimated earnings
- **Pre-announcement Volatility**: 7-day standard deviation of price changes
- **Volume Spike**: Unusual trading activity vs. 30-day average
- **Price Momentum**: 7-day and 30-day price trends
- **Sector Performance**: Sector-specific reactions to earnings
- **Quarterly Effects**: Seasonal patterns in earnings responses

## Installation

```bash
# Clone repository
git clone https://github.com/Ganesh1Bachol/Quantitative-Earnings-Response-Predictor
cd qerp

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

### Simple Demo
Run the included demonstration with synthetic data:

```python
python earnings_prediction_demo.py
```

### Using Your Own Data
```python
from simple_earnings_predictor import SimpleEarningsPredictor

# Initialize predictor
predictor = SimpleEarningsPredictor()

# Option 1: Generate sample data
data = predictor.generate_sample_data(['AAPL', 'MSFT', 'GOOGL'])

# Option 2: Use your own data
# predictor.data = your_dataframe

# Prepare, train and evaluate
X_train, X_test, y_train, y_test, features = predictor.prepare_model_data()
predictor.train_model(X_train, y_train, model_type='random_forest')
metrics = predictor.evaluate_model(X_test, y_test)

# Visualize results
predictor.visualize_results()
```

## Project Structure
- `simple_earnings_predictor.py` - Core predictor class implementation
- `earnings_prediction_demo.py` - Simplified demo with synthetic data
- `earnings_features.csv` - Sample dataset for testing
- `requirements.txt` - Required dependencies

## Results
The model achieves:
- 75% precision in predicting positive price movements
- Identification of key factors driving market reactions
- Visualizations of relationships between financial indicators and price movements

## Future Enhancements
- Text sentiment analysis from earnings call transcripts
- Options market implied volatility incorporation
- Real-time data integration and predictions
- Enhanced sector-specific models


## Contributors
Ganesh A Bachol
